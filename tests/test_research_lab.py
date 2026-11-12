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

    def test_batch_reports_internal_gaps_separately_from_trailing_lag(self):
        """A skipped pair reports the internal gap that blocked every experiment."""
        from csv import DictReader
        from research_lab.runner import run_batch

        self.insert("2025-01-01", timeframe="1d")
        self.insert("2025-01-03", timeframe="1d")
        config = dict(version=1, name="Test", provider="polygon",
                      components=[dict(rule="sma_trend")])
        saved = run_batch([config], self.db, self.db.parent / "runs", timeframe="1d",
                          as_of="2025-01-10T00:00Z", both_views=True)
        for folder, manifest in saved:
            self.assertEqual(manifest["jobs"], dict(completed=0, skipped=4, failed=0))
            with (folder / "results.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual({row["message"] for row in DictReader(stream)},
                                 {"Data audit: 1 missing candles within stored history"})

    def test_latest_continuous_history_restarts_warmup_in_both_views(self):
        """A gap removes old candles from warmup and moves the matched window for clean assets."""
        from csv import DictReader
        from research_lab.runner import run_batch

        dates = pd.date_range("2025-01-01", periods=130, freq="D")
        for i, date in enumerate(dates):
            self.insert(str(date.date()), symbol="ETH", timeframe="1d")
            if i not in (2, 7):
                self.insert(str(date.date()), timeframe="1d")
        config = dict(version=1, name="Test", provider="polygon",
                      components=[dict(rule="sma_trend", window=2)])
        full, matched = run_batch([config], self.db, self.db.parent / "runs", timeframe="1d",
                                  as_of="2025-06-01", both_views=True, latest_continuous=True)
        for (folder, manifest), execution_index in ((full, 22), (matched, 108)):
            btc = next(row for row in manifest["coverage"] if row["symbol"] == "BTC")
            self.assertEqual(btc["internal_missing"], 2)
            self.assertEqual(btc["discarded_before_gap"], 6)
            self.assertEqual(btc["backtest_first_utc"], dates[8].tz_localize("UTC"))
            with (folder / "results.csv").open(newline="", encoding="utf-8") as stream:
                strategies = [r for r in DictReader(stream) if r["experiment"].startswith("strategy:")]
            btc_starts = {r["first_execution"] for r in strategies if r["symbol"] == "BTC"}
            self.assertEqual(btc_starts, {str(dates[execution_index].tz_localize("UTC"))})
            if folder == matched[0]:
                self.assertEqual({r["first_execution"] for r in strategies}, btc_starts)

    def test_latest_continuous_keeps_other_audit_checks(self):
        """Invalid OHLCV still blocks a series even when a later continuous segment exists."""
        from research_lab.data import backtest_history

        self.insert("2025-01-01", timeframe="1d")
        self.insert("2025-01-02", timeframe="1d", values=(10, 8, 9, 11, 0))
        self.insert("2025-01-03", timeframe="1d")
        with open_ohlcv(self.db) as conn:
            candles, audit = load_pair(conn, "polygon", "BTC", "1d", "2025-01-04")
        self.assertEqual(audit["internal_missing"], 1)
        history, trimmed = backtest_history(candles, audit, True)
        self.assertFalse(trimmed)
        self.assertIs(history, candles)

    def test_confirmed_gap_candles_are_marked_without_filling_unknown_gaps(self):
        """Only a fully confirmed gap is carried forward and the real rows remain untouched."""
        from research_lab.data import fill_empty_candles

        for date in ("2025-01-01", "2025-01-03", "2025-01-05"):
            self.insert(date, timeframe="1d")
        with open_ohlcv(self.db) as conn:
            candles, _ = load_pair(conn, "polygon", "BTC", "1d", "2025-01-06")
        self.assertIs(fill_empty_candles(candles, "1d", ["2025-01-02"]), candles)
        filled = fill_empty_candles(candles, "1d", ["2025-01-02", "2025-01-04"])
        self.assertEqual(filled.synthetic.tolist(), [False, True, False, True, False])
        self.assertEqual(filled.loc[filled.synthetic].iloc[0].tolist(), [11., 11., 11., 11., 0., True])
        self.assertNotIn("synthetic", candles)

    def test_synthetic_candles_defer_execution_and_preserve_signal_time(self):
        """A pending entry and exit wait for the next real open and keep their signal snapshot."""
        index = pd.date_range("2025-01-01", periods=5, tz="UTC")
        candles = pd.DataFrame(dict(open=[100,100,102,102,105], high=[100,100,102,102,105],
                                    low=[100,100,102,102,105], close=[100,100,102,102,105],
                                    volume=[1,0,1,0,1], synthetic=[False,True,False,True,False]), index=index)
        signal = pd.Series([1,1,0,0,0], index=index)
        allowed = pd.Series([True,False,True,False,True], index=index)
        settings = Execution(fee=0, slippage=0, stop_atr=None, target_atr=None)
        trade = simulate(candles, signal, settings=settings, entry_allowed=allowed)["trades"].iloc[0]
        self.assertEqual((trade.signal_candle,trade.entry_candle,trade.exit_candle),
                         (index[0],index[2],index[4]))
        self.assertEqual((trade.entry_fill,trade.exit_fill), (102,105))

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

def sample_strategy():
    """Fixed in-memory fixture, independent of the editable project workbook."""
    return dict(version=1, name="Example", provider="hyperliquid",
                components=[dict(rule="sma_trend"), dict(rule="bollinger_reentry")],
                combination=dict(method="weighted", threshold=.5))


class ResearchConfigTests(unittest.TestCase):
    def test_combination_rules_distinguish_neutral_and_opposing_votes(self):
        """Agreement, abstention, and conflict have explicit outcomes for all four methods."""
        from unittest.mock import patch
        from research_lab.runner import build_signal
        cfg = sample_strategy()
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
        from research_lab.runner import validate_config
        cfg = sample_strategy()
        with self.assertRaisesRegex(ValueError, "Unknown indicator"):
            validate_config(dict(cfg, components=[{"rule":"missing"}]))
        with self.assertRaisesRegex(ValueError, "requires a stop"):
            validate_config(dict(cfg, execution=dict(sizing="atr_risk", stop_atr=None)))
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_config(dict(cfg, execution=dict(fee=float("nan"))))

    def test_single_component_preserves_the_original_signal(self):
        """A one-rule config behaves like the standalone indicator under every method."""
        from research_lab.runner import METHODS, build_signal
        from research_lab.strategies import sma_trend
        cfg = sample_strategy()
        cfg["components"] = [{"rule":"sma_trend","window":3}]
        close = pd.Series([100.,100.,100.,110.,90.])
        for method in METHODS:
            cfg["combination"]["method"] = method
            pd.testing.assert_series_equal(build_signal(close, cfg), sma_trend(close, 3), check_names=False)

    def test_macd_periods_validate_and_use_distinct_cached_signals(self):
        """Different periods reach the indicator and cannot share a cached state."""
        from unittest.mock import patch
        from research_lab.runner import build_signal, component_warmup, validate_config
        from research_lab.strategies import macd_direction

        base = sample_strategy()
        rule = dict(rule="macd_direction", fast=8, slow=21, signal=5)
        cfg = dict(base, components=[rule])
        other = dict(base, components=[dict(rule, slow=26)])
        close = pd.Series([100. + i*i/100 for i in range(90)])
        cache = {}
        with patch.dict("research_lab.runner.RULES", {"macd_direction": macd_direction}):
            actual = build_signal(close, cfg, cache=cache)
            other_signal = build_signal(close, other, cache=cache)
        expected = macd_direction(close, fast=8, slow=21, signal=5)
        pd.testing.assert_series_equal(actual, expected, check_names=False)
        self.assertEqual(len(cache), 2)
        self.assertFalse(actual.equals(other_signal))
        self.assertEqual(component_warmup(validate_config(cfg)["components"][0]), 25)
        for invalid in (dict(rule, fast=21), dict(rule, signal=0), dict(rule, slow=21.5),
                        dict(rule, fast=True), dict(rule, window=20)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(dict(base, components=[invalid]))

    def test_macd_warmup_and_causal_histogram_sign(self):
        """A positive or negative histogram starts only after 34 closes; later closes cannot alter it."""
        from research_lab.strategies import macd_direction

        close = pd.Series([100. + i*i/100 for i in range(50)])
        positive = macd_direction(close)
        negative = macd_direction(300-close)
        self.assertTrue(positive.iloc[:33].eq(0).all())
        self.assertTrue(negative.iloc[:33].eq(0).all())
        self.assertEqual((positive.iloc[33], negative.iloc[33]), (1, -1))
        pd.testing.assert_series_equal(positive.iloc[:40], macd_direction(close.iloc[:40]))
        self.assertTrue(macd_direction(pd.Series([100.] * 50)).eq(0).all())

class ResearchBatchTests(unittest.TestCase):
    def test_batch_covers_grid_and_preserves_failures(self):
        """Run both configs and benchmarks on each pair; report gaps without hiding jobs."""
        from research_lab.runner import run_batch
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root/"prices.db"
            rows = []
            for symbol, interval, freq in (("AAA","1d","D"),("BBB","1h","h"),("GAP","1h","h")):
                for i,date in enumerate(pd.date_range("2025-01-01",periods=103,freq=freq)):
                    if symbol == "GAP" and i == 50:
                        continue
                    close = 300-i if symbol == "BBB" else 100+i
                    rows.append(("hyperliquid",str(date),symbol,interval,close,close+1,close-1,close,1))
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                conn.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            cfg = dict(sample_strategy(), strategy_number=37)
            other = dict(cfg, name="MACD-only", components=[dict(rule="macd_direction", fast=12, slow=101, signal=2)])
            folder, manifest = run_batch([cfg,other],db,root/"runs",as_of="2026-01-01")
            self.assertEqual(manifest["strategy_configs"],2)
            self.assertEqual(manifest["warmup_candles"], 102)
            self.assertEqual(manifest["unique_strategy_settings"],2)
            self.assertEqual(manifest["benchmark_configs"],2)
            self.assertEqual(manifest["directional_strategy_configs"],4)
            self.assertEqual(manifest["planned_strategy_jobs"],12)
            self.assertEqual(manifest["jobs"],dict(completed=12,skipped=6,failed=0))
            result = pd.read_csv(folder/"results.csv")
            self.assertEqual(len(result),18)
            first_strategy = result[result.experiment.str.startswith("strategy:1:")]
            self.assertTrue(first_strategy.strategy_number.eq(37).all())
            self.assertTrue(first_strategy.strategy_name.eq("Example").all())
            self.assertTrue(result.loc[result.experiment.str.startswith("benchmark:"),"strategy_number"].isna().all())
            self.assertTrue(result.query("symbol == 'GAP'").status.eq("skipped").all())
            for asset in ("AAA","BBB"):
                rows = result[result.symbol.eq(asset)]
                self.assertEqual(rows.first_execution.nunique(),1)
                self.assertEqual(rows.last_candle.nunique(),1)
            self.assertEqual(set(pd.read_csv(folder/"equity.csv").symbol),{"AAA","BBB"})
            trades = pd.read_csv(folder/"trades.csv")
            long = trades[trades.experiment.str.endswith(":long")]
            short = trades[trades.experiment.str.endswith(":short")]
            self.assertTrue(len(long) and long.side.eq(1).all())
            self.assertTrue(len(short) and short.side.eq(-1).all())
            self.assertEqual(set(long.symbol), {"AAA"})
            self.assertEqual(set(short.symbol), {"BBB"})
            curves = pd.read_csv(folder/"equity.csv")
            starts = curves.groupby(["symbol","timeframe","experiment"]).equity.first()
            self.assertTrue(starts.eq(1000.).all())
            self.assertEqual(len(manifest["coverage"][0]["data_sha256"]),64)
            # The report keeps failed pairs visible but uses a matched universe per timeframe.
            from research_lab.report import generate_report
            html = generate_report(folder,symbol="AAA",timeframe="1d",experiment="strategy:1:long").read_text(encoding="utf-8")
            self.assertIn("Inspect AAA / 1d",html)
            self.assertIn("Stop-loss",html)
            self.assertIn("Drawdown (%)",html)
            self.assertNotIn("Price chart unavailable",html)
            per_asset = pd.read_csv(folder/"metrics_per_asset.csv")
            self.assertEqual(len(per_asset),18)
            overall = pd.read_csv(folder/"metrics_overall.csv")
            hourly = overall[overall.timeframe.eq("1h")]
            self.assertEqual(len(hourly),6)
            self.assertTrue(hourly.loc[hourly.experiment.str.startswith("strategy:1:"),"strategy_name"].eq("Example").all())
            self.assertTrue(hourly.matched_coins.eq(1).all())
            self.assertTrue(hourly.requested_coins.eq(2).all())
            # Intraday VWAP remains visible as a daily skip and participates in hourly comparisons.
            vwap = dict(sample_strategy(), name="VWAP", components=[dict(rule="vwap_band_reentry")])
            _, (matched_folder, _) = run_batch([cfg, other, vwap], db, root/"runs",
                                              as_of="2026-01-01", both_views=True)
            generate_report(matched_folder)
            combined = pd.read_csv(matched_folder/"metrics_overall.csv")
            self.assertEqual(combined.groupby("timeframe").size().to_dict(), {"1d": 6, "1h": 8})
            self.assertEqual(set(combined.loc[combined.experiment.str.startswith("strategy:3:"), "timeframe"]), {"1h"})
            daily_details = pd.read_csv(matched_folder/"metrics_per_asset.csv")
            daily_details = daily_details[daily_details.timeframe.eq("1d") & daily_details.experiment.str.startswith("strategy:3:")]
            self.assertEqual(daily_details.status.tolist(), ["skipped", "skipped"])
            # Altering a historical candle invalidates the price chart, not saved equity.
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("UPDATE ohlcv SET volume=2 WHERE symbol='AAA'")
            changed = generate_report(folder,symbol="AAA",timeframe="1d").read_text(encoding="utf-8")
            self.assertIn("differ from the recorded batch fingerprint",changed)


    def test_batch_rejects_mixed_providers_before_creating_outputs(self):
        """A comparison never combines Polygon and Hyperliquid prices in one batch."""
        from research_lab.runner import run_batch
        cfg = sample_strategy()
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

class ResearchWorkbookTests(unittest.TestCase):
    def workbook(self, slow=21):
        """In-memory input fixture; the project workbook remains editable by the user."""
        from io import BytesIO
        from openpyxl import Workbook
        book = Workbook()
        indicators = book.active
        indicators.title = "individual_indicators"
        for name, rows in {
            "individual_indicators": [
                ["indicator_number", "indicator_name", "strategy_function_name"],
                [1, "Standard MACD", "macd_direction"], [2, "Faster MACD", "macd_direction"],
            ],
            "individual_indicator_settings": [
                ["indicator_number", "setting", "value"], [2, "fast", 8],
                [2, "slow", slow], [2, "signal", 5],
            ],
            "strategies": [
                ["strategy_number", "strategy_name", "combination_rule", "weighted_score_threshold", "fee", "Entry filters"],
                [10, "Blend", "weighted", .5, .002, '=IF(TRUE,"None","")'],
                [20, "Solo", "majority", None, None, "Display only"],
            ],
            "strategy_components": [
                ["strategy_number", "indicator_number", "weight"],
                [10, 1, 1], [10, 2, 3], [20, 2, 1],
            ],
        }.items():
            sheet = book[name] if name in book.sheetnames else book.create_sheet(name)
            for row in rows:
                sheet.append(row)
        stream = BytesIO()
        book.save(stream)
        stream.seek(0)
        return stream

    def test_excel_and_direct_configs_have_identical_signals_trades_and_equity(self):
        """Two parameter variants retain separate votes, while shared signals are cached."""
        from research_lab.runner import atr14, build_signal, validate_config
        from research_lab.workbook import load_workbook_configs

        blend, solo = load_workbook_configs(self.workbook(), "hyperliquid")
        direct_config = validate_config(dict(
            version=1, name="Equivalent direct config", provider="hyperliquid",
            components=[dict(rule="macd_direction",weight=1),
                        dict(rule="macd_direction",fast=8,slow=21,signal=5,weight=3)],
            combination=dict(method="weighted",threshold=.5), execution=dict(fee=.002),
        ))
        self.assertEqual(blend["strategy_number"],10)
        self.assertEqual([c["indicator_number"] for c in blend["components"]], [1,2])
        self.assertEqual(blend["execution"], direct_config["execution"])
        index = pd.date_range("2025-01-01", periods=140, tz="UTC")
        close = pd.Series([100+10*((i%40)/20-1)**2 for i in range(140)],index=index)
        candles = pd.DataFrame(dict(open=close, high=close+1, low=close-1, close=close, volume=1.))
        cache = {}
        excel_signal = build_signal(close,blend,cache=cache)
        self.assertEqual(len(cache),2)
        build_signal(close,solo,cache=cache)
        self.assertEqual(len(cache),2)
        direct_signal = build_signal(close,direct_config)
        pd.testing.assert_series_equal(excel_signal,direct_signal)
        atr = atr14(candles)
        a = simulate(candles,excel_signal,atr,settings=Execution(**blend["execution"]))
        b = simulate(candles,direct_signal,atr,settings=Execution(**direct_config["execution"]))
        self.assertGreater(len(a["trades"]),0)
        pd.testing.assert_frame_equal(a["trades"],b["trades"])
        pd.testing.assert_series_equal(a["equity"],b["equity"])

    def test_workbook_errors_identify_sheet_and_rows(self):
        """Report the location of invalid MACD periods and reject formulas as inputs."""
        from research_lab.workbook import load_workbook_configs
        with self.assertRaisesRegex(ValueError, "individual_indicator_settings, row 3.*MACD fast"):
            load_workbook_configs(self.workbook(slow=8), "polygon")
        with self.assertRaisesRegex(ValueError, "individual_indicator_settings, row 3.*formulas"):
            load_workbook_configs(self.workbook(slow="=21"), "polygon")

    def test_excel_mode_previews_and_runs_selected_strategies(self):
        """Selecting strategies from Excel forwards only those configs to the batch runner."""
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from research_lab.workbook import load_workbook_configs
        configs = load_workbook_configs(self.workbook(), "polygon")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root/"prices.db"
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                conn.execute("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)",
                             ("polygon","2025-01-01","AAVE","1d",10,12,9,11,1))
            (root/"results.csv").write_text(
                "symbol,status,strategy_number,trades,total_return_pct\n"
                "00,completed,1,10,12.5\nNA,completed,2,2,-3.5\nAAVE,skipped,3,,\n",
                encoding="utf-8",
            )
            with patch("config.DB_PATH",db), patch(
                "research_lab.workbook.load_workbook_configs",return_value=configs
            ), patch("ui_pages._research_reports"), patch(
                "research_lab.runner.run_batch",return_value=[(root,{"window_view":"matched-window","jobs":dict(completed=3,skipped=0,failed=0)})]
            ) as run:
                app = AppTest.from_string("from ui_pages import research_settings\nresearch_settings()").run()
                self.assertEqual([tab.label for tab in app.tabs], ["Settings", "Run batch", "Reports"])
                self.assertFalse(app.radio)
                self.assertTrue(any("2 strategies validated" in item.value for item in app.success))
                next(w for w in app.multiselect if w.label=="Batch strategies").set_value(["Solo"]).run()
                next(w for w in app.button if w.label=="Run research batch").click().run()
                self.assertFalse(app.exception)
                self.assertEqual([c["name"] for c in run.call_args.args[0]], ["Solo"])
                # Streamlit receives numeric columns even when some runs were skipped.
                results = next(table.value for table in app.dataframe if "total_return_pct" in table.value)
                self.assertEqual(results["symbol"].tolist(), ["00", "NA", "AAVE"])
                self.assertTrue(pd.api.types.is_integer_dtype(results["trades"]))
                self.assertTrue(pd.api.types.is_float_dtype(results["total_return_pct"]))
                self.assertEqual(results.sort_values("trades")["symbol"].tolist(), ["NA", "00", "AAVE"])

class DonchianResearchTests(unittest.TestCase):
    def test_uses_prior_highs_and_lows_and_exits_before_reversing(self):
        """A wick can block a breakout; exits and short re-entries are separate states."""
        from research_lab.runner import build_signal, component_warmup, validate_config
        from research_lab.strategies import donchian_breakout

        close = pd.Series([100.,100.,100.,105.,106.,95.,93.,100.])
        candles = pd.DataFrame(dict(close=close, high=close+1, low=close-1))
        cfg = validate_config(dict(sample_strategy(), components=[
            dict(rule="donchian_breakout",entry_window=3,exit_window=2)]))
        self.assertEqual(component_warmup(cfg["components"][0]),4)
        signal = build_signal(candles,cfg)
        self.assertEqual(signal.tolist(),[0,0,0,1,1,0,-1,0])
        candles.loc[2,"high"] = 107.
        self.assertEqual(build_signal(candles,cfg).iloc[3],0)
        prefix = donchian_breakout(candles.iloc[:6],3,2)
        pd.testing.assert_series_equal(prefix,donchian_breakout(candles,3,2).iloc[:6])
        with self.assertRaisesRegex(ValueError,"requires candle high, low, and close"):
            build_signal(close,cfg)

class AdditionalIndicatorTests(unittest.TestCase):
    def test_roc_sign_and_exact_lookback(self):
        """ROC stays flat before the lookback and distinguishes positive, negative and zero."""
        from research_lab.strategies import roc_direction
        self.assertEqual(roc_direction(pd.Series([10.,20.,10.,15.,12.]), 2).tolist(),
                         [0,0,0,-1,1])

    def test_rsi_reentry_holds_until_midline_and_is_symmetric(self):
        """Two down moves seed RSI at zero; a small rebound enters and the midline exits."""
        from research_lab.strategies import rsi_reentry
        close = pd.Series([100.,99.,98.,98.25,98.25,99.25])
        self.assertEqual(rsi_reentry(close).tolist(),[0,0,0,1,1,0])
        self.assertEqual(rsi_reentry(200-close).tolist(),[0,0,0,-1,-1,0])
        self.assertTrue(rsi_reentry(pd.Series([100.,99.,98.,101.])).eq(0).all())
        pd.testing.assert_series_equal(rsi_reentry(close.iloc[:4]),rsi_reentry(close).iloc[:4])

    def test_adx_warmup_dmi_direction_and_flat_range(self):
        """After 28 candles, only a strong up/down move votes; flat markets stay neutral."""
        from research_lab.runner import build_signal
        cfg = dict(sample_strategy(), components=[dict(rule="adx_dmi")])
        for direction in (1,-1,0):
            close = pd.Series([100. + direction*i for i in range(32)])
            frame = pd.DataFrame(dict(close=close,high=close+1,low=close-1))
            signal = build_signal(frame,cfg)
            self.assertTrue(signal.iloc[:27].eq(0).all())
            self.assertTrue(signal.iloc[27:].eq(direction).all())
            pd.testing.assert_series_equal(signal.iloc[:29],build_signal(frame.iloc[:29],cfg))
        cfg["components"][0]["adx_threshold"] = 100
        rising = pd.Series(range(100,132),dtype=float)
        self.assertTrue(build_signal(pd.DataFrame(dict(close=rising,high=rising+1,low=rising-1)),cfg).eq(0).all())

    def test_indicator_parameter_limits(self):
        """Reject nonsensical RSI thresholds and ADX strength outside its scale."""
        from research_lab.runner import validate_config
        for rule in (dict(rule="rsi_reentry",lower=50),dict(rule="rsi_reentry",upper=101),
                     dict(rule="adx_dmi",adx_threshold=101),dict(rule="roc_direction",window=1)):
            with self.subTest(rule=rule),self.assertRaises(ValueError):
                validate_config(dict(sample_strategy(),components=[rule]))

class EntryFilterAndSplitTests(unittest.TestCase):
    def test_filter_rejects_fresh_entry_without_reentry_or_early_exit(self):
        """A recovering filter cannot revive a stale signal or close an open position."""
        index = pd.date_range("2025-01-01",periods=7,tz="UTC")
        candles = pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=1.),index=index)
        states = pd.Series([1,1,0,1,1,0,0],index=index)
        gate = pd.Series([False,True,True,True,False,False,False],index=index)
        execution = Execution(stop_atr=None,target_atr=None)
        trades = simulate(candles,states,settings=execution,entry_allowed=gate)["trades"]
        self.assertEqual(len(trades),1)
        self.assertEqual(trades.iloc[0].entry_candle,index[4])
        self.assertEqual(trades.iloc[0].exit_candle,index[6])

    def test_filter_uses_direction_and_disallows_unwarmed_values(self):
        """EMA alignment is directional; no entry before its window has formed."""
        from research_lab.filters import entry_mask, validate_filters
        candles = pd.DataFrame({"close":[100.,101.,102.,99.]})
        filters = validate_filters([dict(filter_rule="ema_alignment",period=2)])
        cache = {}
        self.assertEqual(entry_mask(candles,filters,"long",cache).tolist(),[False,True,True,False])
        self.assertEqual(entry_mask(candles,filters,"short",cache).tolist(),[False,False,False,True])
        self.assertEqual(len(cache),1)

    def test_fixed_split_keeps_capital_and_execution_periods_separate(self):
        """Development stops before the boundary; a stale pre-split state cannot enter."""
        from research_lab.runner import run_split
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root/"prices.db"
            dates = pd.date_range("2025-01-01",periods=125)
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                conn.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", [
                    ("hyperliquid",str(day),"BTC","1d",100+i,101+i,99+i,100+i,1)
                    for i,day in enumerate(dates)
                ])
            cfg = dict(sample_strategy(),components=[dict(rule="sma_trend",window=20)])
            boundary = dates[110]
            (dev, dev_manifest), (evaluation, eval_manifest) = run_split(
                [cfg],db,root/"runs",evaluation_start=boundary,as_of=dates[-1]+pd.Timedelta(days=1))
            self.assertEqual(dev_manifest["split_id"],eval_manifest["split_id"])
            self.assertEqual(dev_manifest["experiments"],eval_manifest["experiments"])
            self.assertEqual(dev_manifest["jobs"]["completed"],4)
            self.assertEqual(eval_manifest["jobs"]["completed"],4)
            old = pd.read_csv(dev/"results.csv")
            new = pd.read_csv(evaluation/"results.csv")
            self.assertTrue(pd.to_datetime(old.last_candle,utc=True).lt(boundary.tz_localize("UTC")).all())
            self.assertTrue(pd.to_datetime(new.first_execution,utc=True).eq(boundary.tz_localize("UTC")).all())
            self.assertTrue(new.loc[new.experiment.str.startswith("strategy:"),"trades"].eq(0).all())
            self.assertTrue(new.loc[new.experiment.str.startswith("strategy:"),"final_equity"].eq(1000.).all())

class ResearchWindowViewTests(unittest.TestCase):
    def test_own_history_is_stable_and_matched_runs_share_actual_dates(self):
        """Longer lookbacks cannot shorten full results; matched trades are rerun on common dates."""
        from research_lab.runner import run_batch
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root/"prices.db"
            base = pd.Timestamp("2025-01-01")
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute(SCHEMA)
                for symbol,offset,count in (("A",0,160),("B",10,145),("NEW",100,35)):
                    conn.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", [
                        ("hyperliquid",str(base+pd.Timedelta(days=offset+i)),symbol,"1d",
                         100+i,101+i,99+i,100+i,1) for i in range(count)
                    ])
            fast = dict(sample_strategy(),name="Fast",components=[dict(rule="sma_trend",window=20)])
            slow = dict(fast,name="Slow",entry_filters=[dict(filter_rule="ema_alignment",period=120)])
            def run(configs, **options):
                return run_batch(configs,db,root/"runs",both_views=True,as_of="2026-01-01",**options)
            (solo,_),_ = run([fast],symbol="A")
            (full,fm),(matched,mm) = run([fast,slow])
            self.assertEqual((fm["window_view"],mm["window_view"]),("full-history","matched-window"))
            read = lambda folder: pd.read_csv(folder/"results.csv")
            keys = ["first_execution","last_candle","execution_candles","trades","final_equity"]
            expected = read(solo).query("experiment == 'strategy:1:long'")[keys].reset_index(drop=True)
            actual = read(full).query("symbol == 'A' and experiment == 'strategy:1:long'")[keys].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected,actual,check_dtype=False)
            self.assertEqual(actual.execution_candles.iloc[0],140)
            self.assertEqual(read(full).query("symbol == 'NEW' and experiment == 'strategy:1:long'").status.iloc[0],"completed")
            shared = read(matched)
            self.assertTrue(shared.query("symbol == 'NEW'").status.eq("skipped").all())
            done = shared[shared.status.eq("completed")]
            self.assertEqual(set(done.symbol),{"A","B"})
            self.assertEqual(set(pd.to_datetime(done.first_execution).dt.tz_localize(None)),{base+pd.Timedelta(days=130)})
            self.assertEqual(set(pd.to_datetime(done.last_candle).dt.tz_localize(None)),{base+pd.Timedelta(days=154)})
            self.assertTrue(done.execution_candles.eq(25).all())
            self.assertTrue(done[done.experiment.str.startswith("strategy:")].trades.eq(0).all())
            self.assertTrue(done[done.experiment.str.startswith("strategy:")].final_equity.eq(1000.).all())

class FollowupStrategyTests(unittest.TestCase):
    def test_relative_volume_uses_previous_bars_and_is_direction_neutral(self):
        """Zero denominators are blocked; the signal candle is not in its own average."""
        from research_lab.filters import entry_mask, validate_filters, filter_warmup
        candles = pd.DataFrame(dict(close=[100.]*6, volume=[0.,0.,1.,1.,2.,1.]))
        filters = validate_filters([dict(filter_rule="relative_volume",period=2,threshold=1.)])
        cache = {}
        long = entry_mask(candles,filters,"long",cache)
        self.assertEqual(long.tolist(),[False,False,False,True,True,False])
        pd.testing.assert_series_equal(long,entry_mask(candles,filters,"short",cache))
        self.assertEqual(filter_warmup(filters[0]),3)
        pd.testing.assert_series_equal(long.iloc[:5],entry_mask(candles.iloc[:5],filters,"long",{}))

    def test_rsi_trend_exit_ignores_moving_average_filter_and_equality(self):
        """A fall below the averages does not exit while the comparison still qualifies."""
        from unittest.mock import patch
        from research_lab.strategies import rsi_adx_long, rsi_midline_long
        close = pd.Series([100.,102.,101.,100.,103.,104.])
        candles = pd.DataFrame(dict(close=close))
        oscillator = pd.Series([10.,60.,60.,50.,40.,60.])
        reference = pd.Series([50.,50.,50.,50.,50.,65.])
        with patch("research_lab.strategies.rsi_value",return_value=oscillator), patch(
                "research_lab.strategies.dmi_values",return_value=(reference,None,None)):
            self.assertEqual(rsi_adx_long(candles,2,2).tolist(),[0,1,1,1,0,0])
            self.assertEqual(rsi_midline_long(close,2,2).tolist(),[0,1,1,1,0,1])

    def test_ema_vote_and_direction_selection(self):
        """EMA waits for warmup, and a long-only config produces no short experiment."""
        from research_lab.runner import _experiments, validate_config
        from research_lab.strategies import ema_trend
        self.assertEqual(ema_trend(pd.Series([10.,10.,12.,9.]),2).tolist(),[0,0,1,-1])
        cfg = validate_config(dict(sample_strategy(),trade_direction="long"))
        strategies = [e["id"] for e in _experiments([cfg]) if e["kind"]=="strategy"]
        self.assertEqual(strategies,["strategy:1:long"])
        with self.assertRaisesRegex(ValueError,"Trade direction"):
            validate_config(dict(sample_strategy(),trade_direction="invalid"))

class IndicatorCatalogTests(unittest.TestCase):
    def test_added_rules_never_revise_earlier_signals(self):
        """Truncating future candles preserves every earlier vote and filter result."""
        import numpy as np
        from research_lab.filters import entry_mask
        from research_lab.runner import build_signal

        index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
        t = np.arange(len(index))
        close = 100 + .035*t + 8*np.sin(t/5) + np.cos(t*1.7)
        candles = pd.DataFrame(dict(high=close+2, low=close-1, close=close, volume=1+t%17), index=index)
        for rule in ("connors_reentry", "supertrend_direction", "vwap_band_reentry", "keltner_breakout",
                     "chandelier_breakout", "mfi_reentry", "cmf_direction", "ichimoku_direction",
                     "hma_direction", "obv_direction", "stoch_rsi_reentry", "williams_reentry"):
            with self.subTest(rule=rule):
                config = dict(version=1, name=rule, provider="hyperliquid", components=[dict(rule=rule)])
                actual = build_signal(candles, config)
                self.assertTrue(actual.isin([-1, 0, 1]).all())
                pd.testing.assert_series_equal(build_signal(candles.iloc[:180], config), actual.iloc[:180])
        for rule, period, threshold in (("bollinger_compression", 20, 20), ("low_choppiness", 14, 38.2)):
            with self.subTest(rule=rule):
                config = [dict(filter_rule=rule, period=period, threshold=threshold)]
                actual = entry_mask(candles, config, "long", {})
                pd.testing.assert_series_equal(entry_mask(candles.iloc[:180], config, "short", {}), actual.iloc[:180])

    def test_volume_indicators_handle_neutral_and_zero_volume(self):
        """Closing at the top/bottom of each bar sets CMF direction; zero volume stays flat."""
        from research_lab.strategies import cmf_direction, obv_direction

        candles = pd.DataFrame(dict(high=[2, 3, 4, 5], low=[1, 2, 3, 4], close=[2, 3, 3, 4], volume=[1]*4))
        self.assertEqual(cmf_direction(candles, period=2).tolist(), [0, 1, 0, -1])
        self.assertEqual(obv_direction(candles, period=2).tolist(), [0, 1, 0, 1])
        candles["volume"] = 0
        self.assertFalse(cmf_direction(candles, period=2).any())

    def test_vwap_resets_and_ignores_partial_first_day(self):
        """VWAP uses complete UTC sessions and clears signals before the midnight execution."""
        from research_lab.strategies import vwap_band_reentry

        index = pd.date_range("2025-01-01 20:00", periods=36, freq="h", tz="UTC")
        close = pd.Series([100] * 36, index=index, dtype=float)
        close.iloc[[1, 20, 26]] = 50
        close.iloc[[2, 21, 27]] = 94
        candles = pd.DataFrame(dict(high=close, low=close, close=close, volume=1.))
        actual = vwap_band_reentry(candles, deviations=1.)
        self.assertFalse(actual.iloc[:4].any())
        self.assertEqual(actual.iloc[21], 1)
        self.assertEqual(actual.iloc[27], 0)  # 23:00 UTC: flatten at midnight.
        self.assertEqual(actual.iloc[28], 0)  # Midnight begins a new session.

