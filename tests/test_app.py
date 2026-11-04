"""Streamlit interaction tests with synthetic data and a mocked fetch service."""
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import analyze
import config
import fetch_data
from tests.helpers import bars
from ui_pages import candles

ROOT = Path(__file__).resolve().parents[1]


class AppInteractionTests(unittest.TestCase):
    def setUp(self):
        """Keep the UI tests deterministic and offline by replacing market data reads."""
        candles.clear()
        self.addCleanup(candles.clear)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        database = patch.object(config, "DB_PATH", str(Path(folder.name)/"test.db"))
        database.start()
        self.addCleanup(database.stop)
        symbols = patch.object(analyze, "get_available_symbols", side_effect=lambda timeframe="1d", **options: ["TEST"] if timeframe=="1d" else [])
        data = patch.object(analyze, "load_data", side_effect=lambda symbol, timeframe="1d", **options: bars())
        symbols.start()
        data.start()
        self.addCleanup(symbols.stop)
        self.addCleanup(data.stop)

    def page(self, name, provider="polygon"):
        """Render a single view with the same settings used by the application entrypoint."""
        script = f'import streamlit as st\nst.session_state.provider={provider!r}\nst.session_state.timeframe="1d"\nst.session_state.threshold=.7\nfrom ui_pages import {name}\n{name}()'
        return AppTest.from_string(script, default_timeout=30).run()

    def test_scanner_analysis_filters_and_timeframe_switch(self):
        """Analysis persists, filters show a useful empty state, and hourly selection hides daily results."""
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=="Run analysis").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["analyses"][("polygon","1d")]["results"]), 1)
        next(w for w in app.text_input if w.label=="Search symbol").set_value("MISSING").run()
        self.assertTrue(any("No coins match" in i.value for i in app.info))
        next(s for s in app.selectbox if s.label=="Timeframe").select("1h").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Run analysis" in i.value for i in app.info))

    def test_coin_detail_renders_interactive_chart_and_votes(self):
        """A stored coin shows a Plotly financial chart, nine vote explanations, benchmarks, and a marked equity curve."""
        app = self.page("coin_detail")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(len(app.dataframe[0].value), 9)
        self.assertEqual(app.dataframe[1].value["Strategy"].tolist(), ["Super Indicator", "Buy and hold", "SMA50/200"])
        self.assertTrue(any(m.label=="Conditions met" for m in app.metric))
        self.assertEqual(len(app.date_input),2)

    def test_fetch_progress_and_summary_are_retained(self):
        """Mocked fetch events update the data page and leave a visible success summary."""
        def fake_fetch(limit, timeframe, on_progress, **options):
            for event in [dict(event="started",total=1,completed=0),dict(event="success",symbol="TEST",bars=300,total=1,completed=1),dict(event="finished",total=1,completed=1,succeeded=1,failed=0)]:
                on_progress(event)
            return 0
        app = self.page("data_management")
        self.assertFalse(app.exception)
        with patch.object(fetch_data,"main",side_effect=fake_fetch):
            next(b for b in app.button if b.label=="Fetch / Update data").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["last_fetch"]["summary"]["succeeded"],1)
        self.assertTrue(any("1 succeeded" in s.value for s in app.success))

    def test_fetch_setup_error_is_visible(self):
        """A credential or provider setup failure is shown to the user instead of crashing the page."""
        app = self.page("data_management")
        with patch.object(fetch_data,"main",side_effect=RuntimeError("Missing credentials")):
            next(b for b in app.button if b.label=="Fetch / Update data").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Missing credentials" in e.value for e in app.error))

    def test_provider_settings_persist_across_ui_sessions(self):
        """A saved exclusion policy is restored when a new Streamlit session is created."""
        app = self.page("data_management")
        next(c for c in app.checkbox if c.label=="Exclude coins above a previous missing-bar limit").check()
        next(n for n in app.number_input if n.label=="Maximum previous missing bars").set_value(95)
        next(c for c in app.checkbox if c.label=="Exclude coins whose previous attempt returned no bars").check()
        next(b for b in app.button if b.label=="Save exclusion settings").click().run()
        self.assertFalse(app.exception)
        restarted = self.page("data_management")
        self.assertFalse(restarted.exception)
        self.assertEqual(next(n for n in restarted.number_input if n.label=="Maximum previous missing bars").value,95)
        self.assertTrue(next(c for c in restarted.checkbox if c.label=="Exclude coins whose previous attempt returned no bars").value)

    def test_force_refresh_control_is_forwarded_and_current_summary_is_visible(self):
        """The UI passes Force refresh and explains a run that made no candle requests."""
        def fake_fetch(limit, timeframe, on_progress, **options):
            self.assertTrue(options["force_refresh"])
            on_progress(dict(event="current",symbol="TEST",reason="Already stored",total=1,completed=1))
            on_progress(dict(event="finished",total=1,completed=1,succeeded=0,failed=0,current=1,skipped=0,fetched=0,discovery_cached=True))
            return 0
        app=self.page("data_management")
        next(c for c in app.checkbox if c.label=="Force refresh").check()
        with patch.object(fetch_data,"main",side_effect=fake_fetch):
            next(b for b in app.button if b.label=="Fetch / Update data").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("1 already current" in e.value for e in app.success))

    def test_provider_switch_keeps_analysis_snapshots_separate(self):
        """Switching suppliers hides the other provider's results and preserves both snapshots."""
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
        next(b for b in app.button if b.label=="Run analysis").click().run()
        next(s for s in app.selectbox if s.label=="Data provider").select("hyperliquid").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Run analysis" in i.value for i in app.info))
        next(b for b in app.button if b.label=="Run analysis").click().run()
        self.assertFalse(app.exception)
        self.assertIn(("polygon","1d"),app.session_state["analyses"])
        self.assertIn(("hyperliquid","1d"),app.session_state["analyses"])
        self.assertEqual(app.session_state["analyses"][("hyperliquid","1d")]["results"][0]["provider"],"hyperliquid")

    def test_provider_policies_are_independent_in_the_ui(self):
        """Saving Polygon exclusions cannot enable those rules in the Hyperliquid settings form."""
        app=self.page("data_management")
        next(c for c in app.checkbox if c.label=="Exclude coins whose previous attempt returned no bars").check()
        next(b for b in app.button if b.label=="Save exclusion settings").click().run()
        self.assertFalse(app.exception)
        hyper=self.page("data_management",provider="hyperliquid")
        self.assertFalse(hyper.exception)
        self.assertFalse(next(c for c in hyper.checkbox if c.label=="Exclude coins whose previous attempt returned no bars").value)
        self.assertTrue(any("5,000" in i.value for i in hyper.info))
        polygon=self.page("data_management")
        self.assertTrue(next(c for c in polygon.checkbox if c.label=="Exclude coins whose previous attempt returned no bars").value)

    def test_internal_gaps_warn_without_hiding_coin(self):
        """Coin detail retains live price and votes while disabling research across missing candles."""
        with patch.object(analyze, "load_data", return_value=bars().drop(bars().index[220])):
            app = self.page("coin_detail")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 9)
        self.assertTrue(any("missing candles" in warning.value for warning in app.warning))

    def test_old_analysis_snapshot_requests_new_run(self):
        """Results from the old ten-vote schema cannot crash the updated Scanner."""
        script = ('import streamlit as st\n'
                  'st.session_state.provider="polygon"\n'
                  'st.session_state.timeframe="1d"\n'
                  'st.session_state.threshold=.7\n'
                  'st.session_state.analyses={("polygon","1d"): {"threshold":.7,"results":[]}}\n'
                  'from ui_pages import scanner\nscanner()')
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("engine changed" in info.value for info in app.info))

    def test_indicator_page_runs_long_short_comparisons(self):
        """Selected candidates yield independent long/short results and baseline trade inspections."""
        app = self.page("indicator_backtest")
        self.assertFalse(app.exception)
        next(m for m in app.multiselect if m.label=="Indicator candidates").set_value(["rsi"])
        next(b for b in app.button if b.label=="Run indicator backtest").click().run()
        self.assertFalse(app.exception)
        result = app.session_state["indicator_runs"][("polygon","1d",("TEST",))]["result"]
        self.assertIn("rsi:long",result["coins"]["TEST"]["results"])
        self.assertIn("rsi:short",result["coins"]["TEST"]["results"])
        self.assertEqual(len(result["per_coin"]),5)

    def test_indicator_page_empty_selection_is_clear(self):
        """Submitting no candidates reports an actionable error rather than crashing the page."""
        app = self.page("indicator_backtest")
        next(m for m in app.multiselect if m.label=="Indicator candidates").set_value([])
        next(b for b in app.button if b.label=="Run indicator backtest").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Select one or more" in e.value for e in app.error))

    def test_inspecting_experiment_updates_equity_and_return_together(self):
        """Choosing Bollinger long binds its chart and displayed final return to the same result."""
        app = self.page("indicator_backtest")
        next(m for m in app.multiselect if m.label=="Indicator candidates").set_value(["bb","ema"])
        next(b for b in app.button if b.label=="Run indicator backtest").click().run()
        next(s for s in app.selectbox if s.label=="Inspect experiment").select("bb:long").run()
        self.assertFalse(app.exception)
        r = app.session_state["indicator_runs"][("polygon","1d",("TEST",))]["result"]["coins"]["TEST"]["results"]["bb:long"]
        self.assertEqual(next(m for m in app.metric if m.label=="Selected experiment: final equity").value, f"{r['equity'].iloc[-1]:.5f}")
        self.assertEqual(next(m for m in app.metric if m.label=="Selected experiment: total return").value, f"{r['stats']['total_return_pct']:.2f}%")
        self.assertEqual(len(app.get("plotly_chart")),1)

    def test_all_coins_runs_full_grid_and_aggregate(self):
        """Selecting all coins produces every coin/candidate combination and one aggregate per direction."""
        with patch.object(analyze,"get_available_symbols",return_value=["AAA","BBB"]):
            app = self.page("indicator_backtest")
            next(c for c in app.checkbox if c.label=="Use all backtest coins").check().run()
            next(m for m in app.multiselect if m.label=="Indicator candidates").set_value(["rsi","ema"])
            next(b for b in app.button if b.label=="Run indicator backtest").click().run()
        self.assertFalse(app.exception)
        r = app.session_state["indicator_runs"][("polygon","1d",("AAA","BBB"))]["result"]
        self.assertEqual(len(r["per_coin"]),14)
        self.assertEqual(len(r["aggregate"]),7)
        self.assertEqual(set(r["per_coin"].Coin),{"AAA","BBB"})
        self.assertTrue(r["aggregate"].valid_coins.eq(2).all())
