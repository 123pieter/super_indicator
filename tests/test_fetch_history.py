"""Persistent fetch-policy tests use temporary databases and mocked API responses."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import fetch_history
import fetch_data
from providers.provider_polygon import PolygonProvider
from tests.helpers import bars


class FetchHistoryTests(unittest.TestCase):
    def setUp(self):
        """Isolate credentials, storage, policy, API calls, and waits from real application data."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.db = Path(self.folder)/"test.db"
        for name,value in {"DB_PATH":self.db,"DATA_FOLDER":self.folder,"SAVE_TO":"both","POLYGON_API_KEY":"dummy","START_DATE":"2019-01-01","ACTIVE_PROVIDER":"polygon"}.items():
            self.stack.enter_context(patch.object(fetch_data.config,name,value))
        self.provider = Mock()
        self.provider.get_available_symbols.return_value = {"AAA":"X:AAAUSD","BBB":"X:BBBUSD"}
        self.provider.fetch_ohlcv.return_value = bars(12)
        self.stack.enter_context(patch.object(fetch_data,"get_provider",return_value=self.provider))
        self.stack.enter_context(patch.object(fetch_data.time,"sleep"))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def record(self, symbol="AAA", status="success", missing=95, provider="polygon", timeframe="1d"):
        fetch_history.record_attempt(self.db,provider,timeframe,symbol,"2025-01-01T00:00:00+00:00",status,0 if status=="no_data" else 10,"2024-01-01",missing)

    def test_reading_unset_state_creates_no_files(self):
        """Looking up policies and history is read-only until settings or an attempt are saved."""
        self.assertEqual(fetch_history.load_settings(self.db,"polygon"), {"max_missing_bars":None,"exclude_no_data":False})
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d"),{})
        self.assertFalse(self.db.exists())

    def test_settings_persist_separately_for_each_provider(self):
        """Saving Polygon's exclusions cannot change another provider's independently saved policy."""
        fetch_history.save_settings(self.db,"polygon",30,True)
        fetch_history.save_settings(self.db,"other",100,False)
        self.assertEqual(fetch_history.load_settings(self.db,"polygon"),{"max_missing_bars":30,"exclude_no_data":True})
        self.assertEqual(fetch_history.load_settings(self.db,"other"),{"max_missing_bars":100,"exclude_no_data":False})

    def test_missing_threshold_is_strict_and_can_be_disabled(self):
        """Only counts greater than the selected limit are excluded; None disables that rule."""
        previous={"status":"success","missing_bars":30}
        self.assertIsNone(fetch_history.exclusion_reason(previous,{"max_missing_bars":30,"exclude_no_data":False}))
        self.assertIsNotNone(fetch_history.exclusion_reason(previous,{"max_missing_bars":29,"exclude_no_data":False}))
        self.assertIsNone(fetch_history.exclusion_reason(previous,{"max_missing_bars":None,"exclude_no_data":False}))
        self.assertIsNone(fetch_history.exclusion_reason(None,{"max_missing_bars":0,"exclude_no_data":True}))

    def test_invalid_thresholds_are_rejected(self):
        """Fractional, negative, and boolean thresholds cannot become ambiguous persisted limits."""
        for value in [-1, 1.5, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                fetch_history.save_settings(self.db,"polygon",value)

    def test_status_is_scoped_by_provider_and_timeframe(self):
        """A bad daily result from one provider does not affect hourly data or another provider."""
        self.record()
        self.record(provider="other",missing=5)
        self.record(timeframe="1h",missing=2)
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]["missing_bars"],95)
        self.assertEqual(fetch_history.load_statuses(self.db,"other","1d")["AAA"]["missing_bars"],5)
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1h")["AAA"]["missing_bars"],2)

    def test_success_records_returned_bars_and_known_staleness(self):
        """After a successful fetch, returned row count, latest candle, and staleness persist."""
        self.assertEqual(fetch_data.main(limit=1),0)
        previous=fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]
        self.assertEqual(previous["status"],"success")
        self.assertEqual(previous["bars_returned"],12)
        self.assertEqual(previous["latest_candle"],bars(12).index[-1].isoformat())
        self.assertGreater(previous["missing_bars"],0)

    def test_saved_exclusions_apply_before_fetch_limit(self):
        """Excluded AAA is skipped and the one-coin test limit is filled by eligible BBB."""
        self.record()
        before=fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]
        fetch_history.save_settings(self.db,"polygon",30,False)
        events=[]
        self.assertEqual(fetch_data.main(limit=1,on_progress=events.append),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[0],"BBB")
        self.assertEqual(events[-1]["skipped"],1)
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d")["AAA"],before)

    def test_no_data_from_one_run_is_excluded_on_the_next(self):
        """A real empty response persists across runs and triggers the saved no-data rule."""
        self.provider.get_available_symbols.return_value={"AAA":"X:AAAUSD"}
        fetch_history.save_settings(self.db,"polygon",None,True)
        self.provider.fetch_ohlcv.return_value=None
        self.assertEqual(fetch_data.main(limit=1),1)
        previous=fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]
        self.assertEqual(previous["status"],"no_data")
        self.assertIsNone(previous["missing_bars"])
        self.provider.fetch_ohlcv.reset_mock()
        events=[]
        self.assertEqual(fetch_data.main(limit=1,on_progress=events.append),0)
        self.provider.fetch_ohlcv.assert_not_called()
        self.assertEqual(events[-1]["skipped"],1)
        self.assertEqual(events[-1]["total"],1)

    def test_retry_override_updates_previously_excluded_coin(self):
        """A one-run override retries an excluded coin, updates its state, and preserves settings."""
        self.record(status="no_data",missing=None)
        fetch_history.save_settings(self.db,"polygon",None,True)
        self.assertEqual(fetch_data.main(limit=1,ignore_exclusions=True),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[0],"AAA")
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]["status"],"success")
        self.assertTrue(fetch_history.load_settings(self.db,"polygon")["exclude_no_data"])

    def test_api_error_is_not_mistaken_for_empty_data(self):
        """A 429/network exception is recorded as an error and remains eligible under the no-data rule."""
        self.provider.get_available_symbols.return_value={"AAA":"X:AAAUSD"}
        fetch_history.save_settings(self.db,"polygon",None,True)
        self.provider.fetch_ohlcv.side_effect=RuntimeError("429 Too Many Requests")
        self.assertEqual(fetch_data.main(limit=1),1)
        previous=fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]
        self.assertEqual(previous["status"],"error")
        self.assertIn("429",previous["error"])
        self.provider.fetch_ohlcv.side_effect=None
        self.assertEqual(fetch_data.main(limit=1),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)

    def test_only_unfinished_bars_are_not_classed_as_no_data(self):
        """Receiving an unfinished candle is distinguishable from a successful empty response."""
        frame=bars(1)
        frame.index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)],name="date")
        self.provider.fetch_ohlcv.return_value=frame
        self.assertEqual(fetch_data.main(limit=1),1)
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]["status"],"no_completed_data")

    def test_polygon_propagates_request_errors(self):
        """The provider cannot turn an HTTP exception into the None reserved for empty results."""
        provider=PolygonProvider.__new__(PolygonProvider)
        provider.client=Mock()
        provider.client.list_aggs.side_effect=RuntimeError("429")
        with self.assertRaisesRegex(RuntimeError,"429"):
            provider.fetch_ohlcv("AAA","X:AAAUSD","2025-01-01","2025-02-01")

    def test_unknown_missing_count_does_not_trigger_missing_limit(self):
        """A coin without any known completed candles has an unknown count, not an invented large gap."""
        previous={"status":"no_data","missing_bars":None}
        self.assertIsNone(fetch_history.exclusion_reason(previous,{"max_missing_bars":0,"exclude_no_data":False}))

    def test_empty_response_retains_the_last_known_missing_bar_count(self):
        """No-data attempts still save the stale count of existing completed candles for later filtering."""
        frame=bars(3)
        latest=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)-pd.Timedelta(days=96)
        frame.index=pd.date_range(end=latest,periods=3,freq="D",name="date")
        fetch_data.save_data("AAA",frame)
        self.provider.fetch_ohlcv.return_value=None
        self.assertEqual(fetch_data.main(limit=1),1)
        previous=fetch_history.load_statuses(self.db,"polygon","1d")["AAA"]
        self.assertEqual(previous["status"],"no_data")
        self.assertEqual(previous["bars_returned"],0)
        self.assertEqual(previous["latest_candle"],latest.isoformat())
        self.assertEqual(previous["missing_bars"],95)
