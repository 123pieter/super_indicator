"""Hyperliquid conversion, provider isolation, and offline integration tests."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
import pandas as pd
from pandas.testing import assert_frame_equal
import analyze
import config
import fetch_data
import fetch_history
import storage
from providers import get_provider, validate_provider
from providers.provider_hyperliquid import HyperliquidProvider
from tests.helpers import bars


class HyperliquidProviderTests(unittest.TestCase):
    def setUp(self):
        """All responses are mocked and time is fixed, so provider tests never contact the API."""
        self.provider=HyperliquidProvider()
        self.now=patch.object(self.provider,"_now",return_value=pd.Timestamp("2025-01-10 12:00Z"))
        self.now.start()
        self.addCleanup(self.now.stop)
        self.post=patch.object(self.provider,"_post")
        self.request=self.post.start()
        self.addCleanup(self.post.stop)

    def row(self,date="2025-01-08",close="12"):
        return dict(t=int(pd.Timestamp(date,tz="UTC").timestamp()*1000),o="10",h="15",l="8",c=close,v="100.5")

    def test_native_discovery_excludes_delisted_markets(self):
        """Active perpetual names become their API tickers; delisted or unnamed rows are skipped."""
        self.request.return_value={"universe":[{"name":"BTC"},{"name":"OLD","isDelisted":True},{}]}
        self.assertEqual(self.provider.get_available_symbols(),{"BTC":"BTC"})
        self.request.assert_called_once_with({"type":"meta"})

    def test_optional_builder_names_keep_namespace(self):
        """HIP-3 market names keep their dex prefix so unrelated CL markets cannot collide."""
        self.provider.dexes=("xyz",)
        self.request.return_value={"universe":[{"name":"CL"},{"name":"xyz:GOLD"}]}
        self.assertEqual(self.provider.get_available_symbols(),{"xyz:CL":"xyz:CL","xyz:GOLD":"xyz:GOLD"})

    def test_candle_strings_are_converted_sorted_and_deduplicated(self):
        """API price strings and millisecond timestamps become numeric UTC OHLCV rows."""
        self.request.return_value=[self.row("2025-01-09"),self.row(),self.row(close="13")]
        result=self.provider.fetch_ohlcv("BTC","BTC","2025-01-08","2025-01-09")
        self.assertEqual(list(result.columns),["open","high","low","close","volume"])
        self.assertEqual(result.index.name,"date")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(len(result),2)
        self.assertEqual(result.close.iloc[0],13.)
        self.assertEqual(result.volume.iloc[0],100.5)
        req=self.request.call_args.args[0]
        self.assertEqual(req["type"],"candleSnapshot")
        self.assertEqual(req["req"]["coin"],"BTC")
        self.assertEqual(req["req"]["interval"],"1d")
        self.assertEqual(req["req"]["endTime"],int(pd.Timestamp("2025-01-10",tz="UTC").timestamp()*1000)-1)

    def test_old_intraday_start_is_clamped_to_available_window(self):
        """Minute requests cannot pretend to backfill beyond Hyperliquid's latest 5,000 candles."""
        self.request.return_value=[]
        self.provider.fetch_ohlcv("BTC","BTC","2020-01-01","2025-01-10","1m")
        actual=self.request.call_args.args[0]["req"]["startTime"]
        expected=pd.Timestamp("2025-01-10 12:00Z")-pd.Timedelta(minutes=4999)
        self.assertEqual(actual,int(expected.timestamp()*1000))

    def test_empty_response_is_distinct_from_http_failure(self):
        """An empty successful response returns None, while a 429 remains an exception."""
        self.request.return_value=[]
        self.assertIsNone(self.provider.fetch_ohlcv("BTC","BTC","2025-01-01","2025-01-09"))
        self.request.side_effect=HTTPError("https://example.invalid",429,"Too Many Requests",{},None)
        with self.assertRaises(HTTPError):
            self.provider.fetch_ohlcv("BTC","BTC","2025-01-01","2025-01-09")

    def test_unavailable_old_range_makes_no_request(self):
        """An entirely unavailable minute range is rejected locally instead of wasting an API call."""
        self.assertIsNone(self.provider.fetch_ohlcv("BTC","BTC","2020-01-01","2020-01-02","1m"))
        self.request.assert_not_called()

    def test_invalid_response_shape_is_an_error(self):
        """Unexpected response objects must not be classified as empty market history."""
        self.request.return_value={"error":"bad request"}
        with self.assertRaises(ValueError):
            self.provider.fetch_ohlcv("BTC","BTC","2025-01-01","2025-01-09")
        with self.assertRaises(ValueError):
            self.provider.get_available_symbols()

    def test_unknown_timeframe_is_rejected_before_request(self):
        """Only registered candle intervals can be sent to Hyperliquid."""
        with self.assertRaises(ValueError):
            self.provider.fetch_ohlcv("BTC","BTC","2025-01-01","2025-01-09","bad")
        self.request.assert_not_called()

    def test_factory_requires_no_key_for_hyperliquid(self):
        """Constructing the public provider requires neither a Polygon key nor a wallet."""
        self.assertIsInstance(get_provider("hyperliquid"),HyperliquidProvider)
        with self.assertRaises(ValueError):
            validate_provider("unknown")


class ProviderIsolationTests(unittest.TestCase):
    def setUp(self):
        """Redirect both providers' candle stores and shared metadata into disposable directories."""
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.folder=self.stack.enter_context(tempfile.TemporaryDirectory())
        self.db=Path(self.folder)/"data.db"
        for module in [config,analyze]:
            self.stack.enter_context(patch.object(module,"DATA_FOLDER",Path(self.folder)/"candles"))
            self.stack.enter_context(patch.object(module,"DB_PATH",self.db))
        self.stack.enter_context(patch.object(config,"SAVE_TO","both"))
        self.stack.enter_context(patch.object(fetch_data.time,"sleep"))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def test_identical_symbol_dates_do_not_mix_providers(self):
        """BTC candles from two suppliers coexist independently in CSV and SQLite."""
        polygon=bars(3)
        hyper=bars(3)
        hyper["close"]=999.
        fetch_data.save_data("BTC",polygon,provider="polygon")
        fetch_data.save_data("BTC",hyper,provider="hyperliquid")
        for backend in ["csv","sqlite"]:
            with patch.object(analyze,"LOAD_FROM",backend):
                assert_frame_equal(analyze.load_data("BTC",provider="polygon"),polygon,check_freq=False)
                assert_frame_equal(analyze.load_data("BTC",provider="hyperliquid"),hyper,check_freq=False)

    def test_builder_symbols_use_safe_windows_csv_names(self):
        """A colon in a Hyperliquid API symbol is escaped on disk and restored during discovery."""
        fetch_data.save_data("xyz:CL",bars(2),provider="hyperliquid")
        folder,_=config.storage_paths("hyperliquid")
        self.assertTrue((folder/"1d"/"xyz%3ACL.csv").exists())
        self.assertEqual(storage.list_symbols("1d",folder,self.db,"csv"),["xyz:CL"])

    def test_hyper_fetch_and_zero_call_rerun_need_no_polygon_key(self):
        """A public Hyperliquid fetch persists its own candles, then a current rerun makes zero API calls."""
        frame=bars(3)
        now=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        frame.index=pd.date_range(end=now-pd.Timedelta(days=1),periods=3,name="date")
        provider=Mock()
        provider.get_available_symbols.return_value={"BTC":"BTC"}
        provider.fetch_ohlcv.return_value=frame
        with patch.object(fetch_data,"get_provider",return_value=provider) as factory, patch.object(config,"load_api_key",side_effect=AssertionError("No credentials needed")):
            self.assertEqual(fetch_data.main(limit=1,provider="hyperliquid"),0)
            factory.assert_called_once_with("hyperliquid",None)
            self.assertEqual(len(fetch_data.read_existing("BTC",provider="hyperliquid")),3)
            self.assertTrue(fetch_data.read_existing("BTC",provider="polygon").empty)
            self.assertIn("BTC",fetch_history.load_statuses(self.db,"hyperliquid","1d"))
            self.assertEqual(fetch_history.load_statuses(self.db,"polygon","1d"),{})
            factory.reset_mock()
            self.assertEqual(fetch_data.main(limit=1,provider="hyperliquid"),0)
            factory.assert_not_called()

    def test_new_hyper_sqlite_store_has_empty_universe(self):
        """Choosing Hyperliquid before its first fetch gives an empty universe rather than an error."""
        with patch.object(analyze,"LOAD_FROM","sqlite"):
            self.assertEqual(analyze.get_available_symbols(provider="hyperliquid"),[])

    def test_both_providers_use_the_same_database_and_symmetric_csv_paths(self):
        """Provider routing yields one database and matching data/provider/timeframe CSV layout."""
        polygon_folder, polygon_db = config.storage_paths("polygon")
        hyper_folder, hyper_db = config.storage_paths("hyperliquid")
        self.assertEqual(polygon_db,hyper_db)
        self.assertEqual(polygon_folder,Path(config.DATA_FOLDER)/"polygon")
        self.assertEqual(hyper_folder,Path(config.DATA_FOLDER)/"hyperliquid")
