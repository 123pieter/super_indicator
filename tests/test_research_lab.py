"""Stage 2 checks using disposable SQLite data, never the production database."""
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from research_lab.__main__ import main
from research_lab.data import audit_database, load_pair, open_ohlcv
from storage import SCHEMA
from research_lab.engine import Execution, simulate


class ResearchDataTests(unittest.TestCase):
    def setUp(self):
        """Create an isolated database with the project's real schema."""
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "prices.db"
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(SCHEMA)

    def insert(self, date, provider="polygon", symbol="BTC", timeframe="1h", values=(10, 12, 9, 11, 0)):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (provider, date, symbol, timeframe, *values))

    def test_provider_isolation_and_all_stored_pairs(self):
        """Discover every pair for one provider without mixing in another provider's candles."""
        self.insert("2025-01-01 00:00:00")
        self.insert("2025-01-01", symbol="ETH", timeframe="1d")
        self.insert("2025-01-01 01:00:00", provider="hyperliquid")
        report = audit_database(self.db, "polygon", "2025-01-02T00:00Z")
        self.assertEqual(set(zip(report.symbol, report.timeframe)), {("BTC", "1h"), ("ETH", "1d")})
        self.assertEqual(report.stored.tolist(), [1, 1])
        self.assertTrue(report.status.eq("ok").all())

    def test_bad_rows_and_gaps_remain_visible(self):
        """Drop invalid, off-grid and duplicate candles; count internal gaps separately from lag."""
        for date in ("2025-01-01 00:00:00", "2025-01-01 04:00:00", "2025-01-01 01:00:00",
                     "2025-01-01T02:00:00+01:00", "2025-01-01 02:30:00", "now"):
            self.insert(date)
        self.insert("2025-01-01 03:00:00", values=(10, 9, 8, 11, 2))
        with open_ohlcv(self.db) as conn:
            frame, report = load_pair(conn, "polygon", "BTC", "1h", "2025-01-01T07:00Z")
        self.assertEqual(list(frame.index), list(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 04:00"], utc=True)))
        for key, count in dict(bad_timestamp=1, off_grid=1, duplicate_rows=2,
                               invalid_ohlcv=1, internal_missing=3, trailing_missing=2).items():
            self.assertEqual(report[key], count, key)
        self.assertEqual(report["status"], "check required")

    def test_cutoff_excludes_unfinished_and_future_candles(self):
        """The candle ending exactly at cutoff is usable; forming and future rows are not."""
        for hour in (0, 1, 2):
            self.insert(f"2025-01-01 {hour:02}:00:00")
        report = audit_database(self.db, "polygon", "2025-01-01T01:00Z").iloc[0]
        self.assertEqual((report.usable, report.unfinished, report.future), (1, 1, 1))

    def test_database_is_read_only_and_missing_path_is_not_created(self):
        """SQLite rejects writes, and auditing a missing database leaves no file."""
        with open_ohlcv(self.db) as conn:
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                conn.execute("DELETE FROM ohlcv")
        missing = self.db.with_name("missing.db")
        with self.assertRaises(sqlite3.OperationalError):
            audit_database(missing, "polygon")
        self.assertFalse(missing.exists())

    def test_cli_reports_review_status_and_unsupported_timeframes(self):
        """An unsupported stored interval stays in the report and returns exit status 1."""
        self.insert("2025-01-01", timeframe="1w")
        out = StringIO()
        with redirect_stdout(out):
            code = main(["check-data", "--provider", "polygon", "--database", str(self.db)])
        self.assertEqual(code, 1)
        self.assertIn("unsupported timeframe", out.getvalue())

class ResearchEngineTests(unittest.TestCase):
    def fixture(self, rows, states, atr=None):
        """Use UTC daily fixtures so trade timing and prices can be checked by hand."""
        index = pd.date_range("2025-01-01", periods=len(rows), tz="UTC")
        frame = pd.DataFrame(rows, columns=["open","high","low","close"], index=index)
        frame["volume"] = 1.
        return frame, pd.Series(states, index=index), pd.Series(atr or [1.]*len(rows), index=index)

    def test_ambiguous_candle_risks_one_percent_and_does_not_reenter(self):
        """Both levels touch after entry; stop first loses exactly 1% at zero costs."""
        f, signal, atr = self.fixture([(100,101,99,100),(100,104,97,101),(101,102,100,101)], [1,1,1])
        result = simulate(f,signal,atr,settings=Execution(fee=0,slippage=0,sizing="atr_risk",target_atr=4))
        self.assertEqual(len(result["trades"]),1)
        t = result["trades"].iloc[0]
        self.assertEqual((t.signal_candle,t.entry_candle,t.reason,t.quantity,t.net_pnl),
                         (f.index[0],f.index[1],"stop",5.,-10.))
        self.assertEqual(result["equity"].tolist(),[1000.,990.,990.])

    def test_short_signal_exit_and_costs_use_next_opens(self):
        """A short uses adverse fills, pays both fees, and exits after the flat signal."""
        f, signal, atr = self.fixture([(100,101,99,100),(110,111,104,105),(90,93,89,92)],[-1,0,1])
        cfg = Execution(fee=.01,slippage=.02,stop_atr=None,target_atr=None)
        result = simulate(f,signal,atr,settings=cfg)
        t = result["trades"].iloc[0]
        entry, exit_ = 110*.98, 90*1.02
        quantity = 100/(entry*1.01)
        expected = quantity*(entry-exit_) - .01*quantity*(entry+exit_)
        self.assertEqual(len(result["trades"]),1)  # Final close's signal has no next open.
        self.assertEqual((t.entry_candle,t.exit_candle,t.reason),(f.index[1],f.index[2],"signal_exit"))
        self.assertAlmostEqual(t.net_pnl,expected)
        self.assertAlmostEqual(result["equity"].iloc[-1],1000+expected)

    def test_stop_gap_fills_at_open_and_can_exceed_risk_target(self):
        """A gap below the stop gets the worse opening fill, not the planned stop price."""
        f, signal, atr = self.fixture([(100,101,99,100),(100,101,99,100),(94,95,93,94)],[1,1,1])
        r = simulate(f,signal,atr,settings=Execution(fee=0,slippage=0,sizing="atr_risk"))
        t = r["trades"].iloc[0]
        self.assertEqual((t.reason,t.exit_phase,t.exit_fill,t.net_pnl),("stop_gap","open",94.,-30.))

    def test_atr_risk_uses_prior_atr_and_accounts_for_costs(self):
        """Changing the entry candle ATR cannot affect the stop or quantity; stop loss includes fees."""
        f, signal, atr = self.fixture([(100,101,99,100),(100,101,95,99)], [1,1], [1.,50.])
        r = simulate(f,signal,atr,settings=Execution(fee=.001,slippage=.0005,sizing="atr_risk"))
        t = r["trades"].iloc[0]
        self.assertAlmostEqual(t.entry_fill-t.stop,2.)
        self.assertAlmostEqual(t.net_pnl,-10.)
        self.assertAlmostEqual(r["equity"].iloc[-1],990.)

    def test_exposure_cap_limits_atr_size(self):
        """A tiny ATR cannot cause the engine to allocate more than the configured cap."""
        f, signal, atr = self.fixture([(100,100,100,100),(100,100,100,100)],[1,1],[.01,.01])
        r = simulate(f,signal,atr,settings=Execution(fee=0,slippage=0,sizing="atr_risk",max_exposure=.2))
        self.assertAlmostEqual(r["trades"].iloc[0].quantity,2.)  # 20% of 1000 divided by 100.

    def test_engine_rejects_a_missing_scheduled_candle(self):
        """A missing day cannot turn the next available open into the scheduled next open."""
        f, signal, atr = self.fixture([(100,101,99,100)]*3,[1,1,1])
        keep = [0,2]
        with self.assertRaisesRegex(ValueError,"Candle gap"):
            simulate(f.iloc[keep],signal.iloc[keep],atr.iloc[keep])

class ResearchStrategyTests(unittest.TestCase):
    def test_sma_warmup_and_direction(self):
        """SMA100 stays flat during warmup and at equality, then follows the close's direction."""
        from research_lab.strategies import sma_trend
        states = sma_trend(pd.Series([100.] * 100 + [110., 90.]))
        self.assertTrue(states.iloc[:100].eq(0).all())
        self.assertEqual(states.iloc[-2:].tolist(), [1, -1])

    def test_bollinger_reentry_and_middle_exit_in_both_directions(self):
        """An outside close is not an entry; re-entry triggers, then the middle band exits."""
        from research_lab.strategies import bollinger_reentry
        close = pd.Series([100.] * 20 + [80., 98., 98., 100.])
        self.assertEqual(bollinger_reentry(close).iloc[-4:].tolist(), [0, 1, 1, 0])
        self.assertEqual(bollinger_reentry(200-close).iloc[-4:].tolist(), [0, -1, -1, 0])

    def test_bollinger_skips_reentry_that_has_already_reached_middle(self):
        """An overshoot straight to the target does not open a mean-reversion trade."""
        from research_lab.strategies import bollinger_reentry
        self.assertTrue(bollinger_reentry(pd.Series([100.] * 20 + [80., 101.])).eq(0).all())

    def test_future_closes_cannot_change_either_signal_prefix(self):
        """Adding future volatility leaves previously observed signal states unchanged."""
        from research_lab.strategies import bollinger_reentry, sma_trend
        close = pd.Series([100.] * 100 + [80., 98., 98., 100., 200., 50.])
        for strategy in (bollinger_reentry, sma_trend):
            pd.testing.assert_series_equal(strategy(close.iloc[:-2]), strategy(close).iloc[:-2])

    def test_reentry_signal_executes_at_next_open(self):
        """The engine enters after the Bollinger re-entry close and exits after the middle touch."""
        from research_lab.strategies import bollinger_reentry
        close = pd.Series([100.] * 20 + [80., 98., 98., 100., 102.],
                          index=pd.date_range("2025-01-01", periods=25, tz="UTC"))
        candles = pd.DataFrame(dict(open=close, high=close+1, low=close-1, close=close, volume=1.))
        r = simulate(candles, bollinger_reentry(close),
                     settings=Execution(fee=0, slippage=0, stop_atr=None, target_atr=None))
        self.assertEqual(len(r["trades"]), 1)
        trade = r["trades"].iloc[0]
        self.assertEqual((trade.signal_candle, trade.entry_candle, trade.exit_candle, trade.reason),
                         (close.index[21], close.index[22], close.index[24], "signal_exit"))
        self.assertAlmostEqual(trade.net_pnl, 100/98*4)

class ResearchConfigTests(unittest.TestCase):
    def test_combination_rules_distinguish_neutral_and_opposing_votes(self):
        """Agreement, abstention, and conflict have explicit outcomes for all four methods."""
        from unittest.mock import patch
        from research_lab.runner import BASELINE_CONFIG, build_signal, load_config
        cfg = load_config(BASELINE_CONFIG)
        close = pd.Series([100.] * 5)
        with patch.dict("research_lab.runner.RULES", {
            "sma_trend": lambda close, **opts: pd.Series([1, 1, 1, -1, 0]),
            "bollinger_reentry": lambda close, **opts: pd.Series([1, 0, -1, -1, 0]),
        }):
            for method, expected in {
                "majority": [1, 0, 0, -1, 0], "and": [1, 0, 0, -1, 0],
                "weighted": [1, 1, 0, -1, 0], "or": [1, 1, 0, -1, 0],
            }.items():
                with self.subTest(method=method):
                    cfg["combination"]["method"] = method
                    self.assertEqual(build_signal(close, cfg).tolist(), expected)
            cfg["components"][0]["weight"] = 3
            cfg["combination"]["method"] = "weighted"
            self.assertEqual(build_signal(close, cfg).tolist(), [1, 1, 1, -1, 0])

    def test_config_rejects_unknown_rule_and_atr_risk_without_stop(self):
        """Typos and impossible risk sizing fail before any backtest is run."""
        from research_lab.runner import BASELINE_CONFIG, load_config, validate_config
        cfg = load_config(BASELINE_CONFIG)
        with self.assertRaisesRegex(ValueError, "Unknown indicator"):
            validate_config(dict(cfg, components=[{"rule":"missing"}]))
        with self.assertRaisesRegex(ValueError, "requires a stop"):
            validate_config(dict(cfg, execution=dict(sizing="atr_risk", stop_atr=None)))
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_config(dict(cfg, execution=dict(fee=float("nan"))))

    def test_single_component_preserves_the_original_signal(self):
        """A one-rule config behaves like the standalone indicator under every method."""
        from research_lab.runner import BASELINE_CONFIG, METHODS, build_signal, load_config
        from research_lab.strategies import sma_trend
        cfg = load_config(BASELINE_CONFIG)
        cfg["components"] = [{"rule":"sma_trend","window":3}]
        close = pd.Series([100.,100.,100.,110.,90.])
        for method in METHODS:
            cfg["combination"]["method"] = method
            pd.testing.assert_series_equal(build_signal(close, cfg), sma_trend(close, 3), check_names=False)

    def test_ui_saves_validated_settings_with_percent_conversion(self):
        """UI controls build the same canonical config as JSON, including percentage costs."""
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        with patch("ui_pages._research_batch_controls"):
            app = AppTest.from_string("from ui_pages import research_settings\nresearch_settings()").run()
            next(w for w in app.number_input if w.label == "Research fee per side (%)").set_value(.25)
            next(w for w in app.selectbox if w.label == "Combination rule").select("or")
            next(w for w in app.button if w.label == "Validate research settings").click().run()
        self.assertFalse(app.exception)
        cfg = app.session_state["validated_research_settings"][1]
        self.assertEqual(cfg["combination"]["method"], "or")
        self.assertAlmostEqual(cfg["execution"]["fee"], .0025)
        self.assertEqual(len(app.get("download_button")), 1)

class ResearchBatchTests(unittest.TestCase):
    def test_batch_covers_grid_and_preserves_failures(self):
        """Run both configs and benchmarks on each pair; report gaps without hiding jobs."""
        from research_lab.runner import BASELINE_CONFIG, load_config, run_batch
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root/"prices.db"
            rows = []
            for symbol, interval, freq in (("AAA","1d","D"),("BBB","1h","h"),("GAP","1h","h")):
                for i,date in enumerate(pd.date_range("2025-01-01",periods=103,freq=freq)):
                    if symbol == "GAP" and i == 50:
                        continue
                    close = 100+i
                    rows.append(("hyperliquid",str(date),symbol,interval,close,close+1,close-1,close,1))
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                conn.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            cfg = load_config(BASELINE_CONFIG)
            other = dict(cfg, name="SMA-only", components=[{"rule":"sma_trend","window":20}])
            folder, manifest = run_batch([cfg,other],db,root/"runs",as_of="2026-01-01")
            self.assertEqual(manifest["strategy_configs"],2)
            self.assertEqual(manifest["unique_strategy_settings"],2)
            self.assertEqual(manifest["benchmark_configs"],2)
            self.assertEqual(manifest["jobs"],dict(completed=8,skipped=4,failed=0))
            result = pd.read_csv(folder/"results.csv")
            self.assertEqual(len(result),12)
            self.assertTrue(result.query("symbol == 'GAP'").status.eq("skipped").all())
            for asset in ("AAA","BBB"):
                rows = result[result.symbol.eq(asset)]
                self.assertEqual(rows.first_execution.nunique(),1)
                self.assertEqual(rows.last_candle.nunique(),1)
            self.assertEqual(set(pd.read_csv(folder/"equity.csv").symbol),{"AAA","BBB"})
            self.assertTrue(len(pd.read_csv(folder/"trades.csv")) > 0)
            self.assertEqual(len(manifest["coverage"][0]["data_sha256"]),64)
            # The report keeps failed pairs visible but uses a matched universe per timeframe.
            from research_lab.report import generate_report
            html = generate_report(folder,symbol="AAA",timeframe="1d",experiment="strategy:1").read_text(encoding="utf-8")
            self.assertIn("Inspect AAA / 1d",html)
            self.assertIn("Stop-loss",html)
            self.assertIn("Drawdown (%)",html)
            self.assertNotIn("Price chart unavailable",html)
            per_asset = pd.read_csv(folder/"metrics_per_asset.csv")
            self.assertEqual(len(per_asset),12)
            overall = pd.read_csv(folder/"metrics_overall.csv")
            hourly = overall[overall.timeframe.eq("1h")]
            self.assertEqual(len(hourly),4)
            self.assertTrue(hourly.matched_coins.eq(1).all())
            self.assertTrue(hourly.requested_coins.eq(2).all())
            # Altering a historical candle invalidates the price chart, not saved equity.
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("UPDATE ohlcv SET volume=2 WHERE symbol='AAA'")
            changed = generate_report(folder,symbol="AAA",timeframe="1d").read_text(encoding="utf-8")
            self.assertIn("differ from the recorded batch fingerprint",changed)


    def test_batch_rejects_mixed_providers_before_creating_outputs(self):
        """A comparison never combines Polygon and Hyperliquid prices in one batch."""
        from research_lab.runner import BASELINE_CONFIG, load_config, run_batch
        cfg = load_config(BASELINE_CONFIG)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError,"same provider"):
                run_batch([cfg,dict(cfg,name="Polygon variant",provider="polygon")],
                          root/"missing.db",root/"runs")
            self.assertFalse((root/"runs").exists())

    def test_atr_seed_and_next_step(self):
        """ATR14 starts at the mean true range and updates with one Wilder step."""
        from research_lab.runner import atr14
        frame = pd.DataFrame({"high":[101.]*14+[115.],"low":[99.]*15,"close":[100.]*15})
        values = atr14(frame)
        self.assertTrue(values.iloc[:13].isna().all())
        self.assertEqual(values.iloc[13],2.)
        self.assertAlmostEqual(values.iloc[14],3.)

class ResearchReportTests(unittest.TestCase):
    def test_metrics_and_exposure_are_hand_checkable(self):
        """A 100→110→99 path ends down 1%, with a 10% peak-to-trough loss."""
        from research_lab.report import bar_exposure, metrics
        dates = pd.date_range("2025-01-01",periods=3,tz="UTC")
        curve = pd.Series([100.,110.,99.],index=dates)
        trades = pd.DataFrame({
            "entry_candle":[dates[1],dates[2]], "exit_candle":[dates[2],dates[2]],
            "exit_phase":["open","intrabar"], "net_pnl":[10.,-11.],
        })
        exposure = bar_exposure(dates,trades)
        m = metrics(curve,trades,"1d",exposure)
        self.assertAlmostEqual(m["total_return_pct"],-1.)
        self.assertAlmostEqual(m["drawdown_pct"],-10.)
        self.assertAlmostEqual(m["cagr_pct"],(0.99**(365/2)-1)*100)
        self.assertAlmostEqual(m["profit_factor"],10/11)
        self.assertEqual((m["trades"],m["win_rate_pct"],m["average_win"],m["average_loss"]),
                         (2,50.,10.,-11.))
        self.assertAlmostEqual(bar_exposure(dates,trades.iloc[:1]),.5)
        self.assertEqual(m["exposure_pct"],100.)
        self.assertEqual(m["sample"],"thin")

class ResearchBatchUITests(unittest.TestCase):
    def test_ui_runs_validated_config_for_selected_asset_and_timeframe(self):
        """The UI sends the validated form and chosen pair to the same runner as the CLI."""
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            db = folder/"prices.db"
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                for symbol,timeframe in (("00","1d"),("AAVE","1d"),("AAVE","1h")):
                    conn.execute("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)",
                                 ("hyperliquid","2025-01-01",symbol,timeframe,10,12,9,11,1))
            (folder/"results.csv").write_text("symbol,status\nAAVE,completed\n",encoding="utf-8")
            with patch("config.DB_PATH",db), patch(
                "research_lab.runner.run_batch",
                return_value=(folder,{"jobs":dict(completed=3,skipped=0,failed=0)}),
            ) as run:
                app = AppTest.from_string("from ui_pages import research_settings\nresearch_settings()").run()
                self.assertEqual([tab.label for tab in app.tabs], ["Settings","Run batch","Reports"])
                self.assertTrue(any(w.label=="Strategy name" for w in app.tabs[0].text_input))
                self.assertTrue(any(w.label=="Run research batch" for w in app.tabs[1].button))
                next(w for w in app.text_input if w.label=="Strategy name").set_value("UI strategy")
                next(w for w in app.button if w.label=="Validate research settings").click().run()
                next(w for w in app.selectbox if w.label=="Batch asset").select("AAVE").run()
                next(w for w in app.button if w.label=="Run research batch").click().run()
            self.assertFalse(app.exception)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][0]["name"],"UI strategy")
            self.assertEqual(run.call_args.kwargs["symbol"],"AAVE")
            self.assertEqual(run.call_args.kwargs["timeframe"],"1d")
            self.assertTrue(any("3 completed" in m.value for m in app.success))

class ResearchReportUITests(unittest.TestCase):
    def test_saved_report_is_embedded_and_missing_report_uses_artifacts(self):
        """A saved report renders as HTML; generating another does not run a batch."""
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("20251101-first", "20251102-second"):
                folder = root/name
                folder.mkdir()
                (folder/"manifest.json").write_text("{}", encoding="utf-8")
            (root/"20251102-second"/"report.html").write_text(
                "<!doctype html><h1>Saved report</h1>", encoding="utf-8")

            def make_report(folder):
                path = folder/"report.html"
                path.write_text("<!doctype html><h1>Generated report</h1>", encoding="utf-8")
                return path

            script = (
                f"from pathlib import Path\nfrom ui_pages import _research_reports\n"
                f"_research_reports(Path({tmp!r}))"
            )
            with patch("research_lab.report.generate_report", side_effect=make_report) as generate, patch(
                "research_lab.runner.run_batch"
            ) as run:
                app = AppTest.from_string(script).run()
                self.assertIn("Saved report", app.get("iframe")[0].proto.srcdoc)
                app.selectbox[0].select("20251101-first").run()
                self.assertFalse(app.get("iframe"))
                app.button[0].click().run()
                self.assertIn("Generated report", app.get("iframe")[0].proto.srcdoc)
                generate.assert_called_once_with(root/"20251101-first")
                run.assert_not_called()
            self.assertFalse(app.exception)
