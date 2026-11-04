"""Offline tests proving API requests are avoided only when stored candles are current."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import fetch_data
import fetch_history
from tests.helpers import bars


class FetchFreshnessTests(unittest.TestCase):
    def setUp(self):
        """Use fixed UTC time, disposable storage, and a provider that never contacts the network."""
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.folder=self.stack.enter_context(tempfile.TemporaryDirectory())
        self.db=Path(self.folder)/"test.db"
        for name,value in {"DB_PATH":self.db,"DATA_FOLDER":self.folder,"SAVE_TO":"both","ACTIVE_PROVIDER":"polygon","POLYGON_API_KEY":"dummy","START_DATE":"2020-01-01"}.items():
            self.stack.enter_context(patch.object(fetch_data.config,name,value))
        self.clock=self.stack.enter_context(patch.object(fetch_data,"utc_now",return_value=pd.Timestamp("2025-01-02 12:30Z")))
        self.provider=Mock()
        self.provider.get_available_symbols.return_value={"AAA":"X:AAAUSD","BBB":"X:BBBUSD"}
        self.provider.fetch_ohlcv.return_value=self.daily()
        self.factory=self.stack.enter_context(patch.object(fetch_data,"get_provider",return_value=self.provider))
        self.sleep=self.stack.enter_context(patch.object(fetch_data.time,"sleep"))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def daily(self):
        """Three daily bars ending on the latest complete day at the fixed clock."""
        frame=bars(3)
        frame.index=pd.date_range("2024-12-30",periods=3,freq="D",name="date")
        return frame

    def cache(self, now="2025-01-02 00:00Z", provider="polygon"):
        fetch_history.save_symbols(self.db,provider,self.provider.get_available_symbols.return_value,now)

    def test_same_day_complete_rerun_makes_zero_provider_calls(self):
        """A full first fetch is followed by a same-day run with zero discovery or candle requests."""
        self.assertEqual(fetch_data.main(limit=0),0)
        previous=fetch_history.load_statuses(self.db,"polygon","1d")
        self.provider.reset_mock()
        self.factory.reset_mock()
        self.sleep.reset_mock()
        with patch.object(fetch_data.config,"POLYGON_API_KEY",""), patch.object(fetch_data.config,"load_api_key",side_effect=AssertionError("Credentials must not be needed")):
            events=[]
            self.assertEqual(fetch_data.main(limit=0,on_progress=events.append),0)
        self.factory.assert_not_called()
        self.provider.get_available_symbols.assert_not_called()
        self.provider.fetch_ohlcv.assert_not_called()
        self.sleep.assert_not_called()
        self.assertEqual(events[-1]["current"],2)
        self.assertEqual(events[-1]["fetched"],0)
        self.assertTrue(events[-1]["discovery_cached"])
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d"),previous)

    def test_current_local_data_is_enough_without_attempt_history(self):
        """Existing completed candles are authoritative even if no fetch status has been recorded."""
        self.cache()
        for symbol in ["AAA","BBB"]:
            fetch_data.save_data(symbol,self.daily())
        self.assertEqual(fetch_data.main(limit=0),0)
        self.factory.assert_not_called()
        self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d"),{})

    def test_new_utc_day_refreshes_discovery_and_outdated_coins(self):
        """At UTC midnight yesterday's stored last candle is outdated and discovery expires."""
        self.assertEqual(fetch_data.main(limit=0),0)
        self.provider.reset_mock()
        self.clock.return_value=pd.Timestamp("2025-01-03 00:00Z")
        self.assertEqual(fetch_data.main(limit=0),0)
        self.provider.get_available_symbols.assert_called_once()
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)

    def test_hourly_boundary_fetches_newly_completed_interval(self):
        """Hourly data stays current within its hour but needs an update at the next completion boundary."""
        self.cache()
        frame=bars(3)
        frame.index=pd.date_range("2025-01-02 09:00",periods=3,freq="h",name="date")
        for symbol in ["AAA","BBB"]:
            fetch_data.save_data(symbol,frame,"1h")
        self.assertEqual(fetch_data.main(limit=0,timeframe="1h"),0)
        self.factory.assert_not_called()
        self.clock.return_value=pd.Timestamp("2025-01-02 13:00Z")
        self.provider.fetch_ohlcv.return_value=frame
        self.assertEqual(fetch_data.main(limit=0,timeframe="1h"),0)
        self.provider.get_available_symbols.assert_not_called()
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)

    def test_limit_applies_only_to_outdated_coins(self):
        """Current AAA does not consume a one-coin fetch limit; missing BBB is requested."""
        self.cache()
        fetch_data.save_data("AAA",self.daily())
        events=[]
        self.assertEqual(fetch_data.main(limit=1,on_progress=events.append),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[0],"BBB")
        self.assertEqual(events[-1]["current"],1)
        self.assertEqual(events[-1]["fetched"],1)

    def test_force_refresh_updates_discovery_and_current_candles(self):
        """The explicit force option bypasses freshness for both symbol discovery and price requests."""
        self.assertEqual(fetch_data.main(limit=0),0)
        self.provider.reset_mock()
        events=[]
        self.assertEqual(fetch_data.main(limit=0,force_refresh=True,on_progress=events.append),0)
        self.provider.get_available_symbols.assert_called_once()
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)
        self.assertEqual(events[-1]["current"],0)
        self.assertFalse(events[-1]["discovery_cached"])

    def test_force_refresh_still_respects_saved_exclusions(self):
        """Force refetch and exclusion override are independent controls."""
        self.cache()
        fetch_history.record_attempt(self.db,"polygon","1d","AAA","2025-01-02T00:00Z","no_data",0)
        fetch_history.save_settings(self.db,"polygon",None,True)
        self.assertEqual(fetch_data.main(limit=0,force_refresh=True),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_count,1)
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[0],"BBB")

    def test_error_and_empty_attempts_remain_retryable_when_outdated(self):
        """Neither a previous failure nor empty response marks missing local candles as current."""
        self.cache()
        for symbol,status in [("AAA","error"),("BBB","no_data")]:
            fetch_history.record_attempt(self.db,"polygon","1d",symbol,"2025-01-02T00:00Z",status,0)
        self.assertEqual(fetch_data.main(limit=0),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)
        self.provider.get_available_symbols.assert_not_called()

    def test_current_counter_is_based_on_storage_not_success_metadata(self):
        """A saved success cannot hide candles that have since been removed from local storage."""
        self.cache()
        for symbol in ["AAA","BBB"]:
            fetch_history.record_attempt(self.db,"polygon","1d",symbol,"2025-01-02T00:00Z","success",3,"2025-01-01",0)
        self.assertEqual(fetch_data.main(limit=0),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_count,2)

    def test_current_destination_repairs_other_destination_locally(self):
        """A current CSV can populate a missing SQLite copy without contacting the provider."""
        self.cache()
        with patch.object(fetch_data.config,"SAVE_TO","csv"):
            for symbol in ["AAA","BBB"]:
                fetch_data.save_data(symbol,self.daily())
        self.assertEqual(fetch_data.main(limit=0),0)
        self.factory.assert_not_called()
        for symbol in ["AAA","BBB"]:
            frame=fetch_data.storage.read_data(symbol,"1d",self.folder,self.db,"sqlite")
            self.assertEqual(len(frame),3)

    def test_today_forming_candle_does_not_hide_missing_yesterday(self):
        """An unfinished daily row cannot substitute for the previous completed day."""
        self.cache()
        frame=self.daily().iloc[[-1]].copy()
        frame.index=pd.DatetimeIndex([pd.Timestamp("2025-01-02")],name="date")
        fetch_data.save_data("AAA",frame)
        self.assertEqual(fetch_data.main(limit=1),0)
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[0],"AAA")

    def test_discovery_failure_preserves_previous_cache(self):
        """Failed forced discovery must not replace a previously valid symbol snapshot."""
        self.cache()
        self.provider.get_available_symbols.side_effect=RuntimeError("429")
        with self.assertRaisesRegex(RuntimeError,"429"):
            fetch_data.main(limit=0,force_refresh=True)
        self.assertEqual(len(fetch_history.load_symbols(self.db,"polygon",self.clock.return_value)),2)

    def test_discovery_cache_is_scoped_to_provider_and_utc_day(self):
        """Daily discovery snapshots are independent of timezone offsets and other providers."""
        self.cache()
        self.assertIsNone(fetch_history.load_symbols(self.db,"other",self.clock.return_value))
        self.assertIsNotNone(fetch_history.load_symbols(self.db,"polygon","2025-01-02 23:59Z"))
        self.assertIsNone(fetch_history.load_symbols(self.db,"polygon","2025-01-03 00:00Z"))
        self.assertIsNone(fetch_history.load_symbols(self.db,"polygon","2025-01-01 23:59Z"))

    def test_empty_discovery_is_not_cached(self):
        """An empty universe does not create a fresh cache that suppresses later retries."""
        self.provider.get_available_symbols.return_value={}
        with self.assertRaisesRegex(RuntimeError,"no USD pairs"):
            fetch_data.main(limit=0)
        self.assertIsNone(fetch_history.load_symbols(self.db,"polygon",self.clock.return_value))
