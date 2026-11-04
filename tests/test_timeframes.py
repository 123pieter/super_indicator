"""Regression coverage for timeframe isolation, current storage, and provider mapping."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from pandas.testing import assert_frame_equal
import analyze
import fetch_data
import storage
from timeframes import completed_bars
from providers.provider_polygon import PolygonProvider
from tests.helpers import bars


class TimeframeTests(unittest.TestCase):
    def setUp(self):
        """Isolate all files and databases so storage and intraday tests cannot touch real history."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.db = str(Path(self.folder)/"test.db")
        for module in [fetch_data.config, analyze]:
            self.stack.enter_context(patch.object(module, "DATA_FOLDER", self.folder))
            self.stack.enter_context(patch.object(module, "DB_PATH", self.db))

    def test_same_symbol_and_timestamp_are_separate_by_timeframe(self):
        """Daily and hourly candles at the same timestamp must never overwrite one another."""
        for backend in ["csv", "sqlite", "both"]:
            with self.subTest(backend=backend):
                daily, hourly = bars(3), bars(3)
                hourly["close"] = 999.0
                storage.save_data("BTC", daily, "1d", self.folder, self.db, backend)
                storage.save_data("BTC", hourly, "1h", self.folder, self.db, backend)
                for timeframe, expected in [("1d", daily), ("1h", hourly)]:
                    actual = storage.read_data("BTC", timeframe, self.folder, self.db, backend)
                    assert_frame_equal(actual, expected, check_freq=False)

    def test_symbol_listing_is_filtered_by_timeframe(self):
        """An hourly-only coin must not appear in the daily universe, and vice versa."""
        for backend in ["csv", "sqlite"]:
            storage.save_data("DAILY", bars(2), "1d", self.folder, self.db, backend)
            storage.save_data("HOURLY", bars(2), "1h", self.folder, self.db, backend)
            self.assertEqual(storage.list_symbols("1d", self.folder, self.db, backend), ["DAILY"])
            self.assertEqual(storage.list_symbols("1h", self.folder, self.db, backend), ["HOURLY"])

    def test_csv_partial_update_preserves_other_dates(self):
        """A small update replaces matching daily candles without deleting the rest of their history."""
        storage.save_data("BTC", bars(3), "1d", self.folder, self.db, "csv")
        update = bars(3).tail(1).copy()
        update["close"] = 555.0
        storage.save_data("BTC", update, "1d", self.folder, self.db, "csv")
        merged = storage.read_data("BTC", "1d", self.folder, self.db, "csv")
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.close.iloc[-1], 555.0)
        self.assertTrue(storage.read_data("BTC", "1h", self.folder, self.db, "csv").empty)

    def test_sqlite_update_does_not_change_another_timeframe(self):
        """Revising a daily candle must preserve the hourly candle at the same timestamp."""
        for timeframe in ["1d", "1h"]:
            storage.save_data("BTC", bars(3), timeframe, self.folder, self.db, "sqlite")
        update = bars(3).tail(1).copy()
        update["close"] = 777.0
        storage.save_data("BTC", update, "1d", self.folder, self.db, "sqlite")
        daily = storage.read_data("BTC", "1d", self.folder, self.db, "sqlite")
        hourly = storage.read_data("BTC", "1h", self.folder, self.db, "sqlite")
        self.assertEqual(daily.close.iloc[-1], 777.0)
        self.assertEqual(hourly.close.iloc[-1], 12.0)
        self.assertEqual(len(daily), 3)

    def test_hourly_completion_uses_candle_end(self):
        """At 10:30 UTC the 09:00 hourly candle is complete, but the 10:00 candle is not."""
        df = bars(3)
        df.index = pd.date_range("2025-01-01 09:00", periods=3, freq="h", name="date")
        result = completed_bars(df, "1h", now="2025-01-01 10:30Z")
        self.assertEqual(list(result.index), [df.index[0]])
        self.assertEqual(len(completed_bars(df, "1h", now="2025-01-01 11:00Z")), 2)

    def test_timezone_normalization_preserves_utc_instants(self):
        """Timezone-aware timestamps are converted to UTC rather than losing their offsets."""
        df = bars(2)
        df.index = pd.date_range("2025-01-01 10:00", periods=2, freq="h", tz="Europe/Berlin")
        self.assertEqual(storage.normalize(df).index[0], pd.Timestamp("2025-01-01 09:00"))

    def test_unknown_timeframe_is_rejected(self):
        """Invalid intervals fail before a storage path or provider request can be used."""
        with self.assertRaisesRegex(ValueError, "Unsupported timeframe"):
            storage.read_data("BTC", "../invalid", self.folder, self.db, "csv")

    def test_polygon_maps_timeframes_to_multiplier_and_timespan(self):
        """The provider sends correct minute/hour/day parameters to the SDK, without network calls."""
        provider = PolygonProvider.__new__(PolygonProvider)
        provider.client = Mock()
        provider.client.list_aggs.return_value = [SimpleNamespace(open=1, high=2, low=1, close=2, volume=10, timestamp=0)]
        for name, multiplier, timespan in [("1d", 1, "day"), ("4h", 4, "hour"), ("15m", 15, "minute")]:
            provider.fetch_ohlcv("BTC", "X:BTCUSD", "2020-01-01", "2020-02-01", timeframe=name)
            kwargs = provider.client.list_aggs.call_args.kwargs
            self.assertEqual((kwargs["multiplier"], kwargs["timespan"]), (multiplier, timespan))

    def test_intraday_fetch_resumes_from_its_own_history(self):
        """Hourly fetches use hourly history and leave daily rows untouched for the same symbol."""
        provider = Mock()
        provider.get_available_symbols.return_value = {"BTC": "X:BTCUSD"}
        hourly = bars(4)
        hourly.index = pd.date_range("2025-01-02 00:00", periods=4, freq="h", name="date")
        daily = bars(3)
        provider.fetch_ohlcv.return_value = hourly
        with patch.object(fetch_data.config, "SAVE_TO", "both"), patch.object(fetch_data.config, "POLYGON_API_KEY", "dummy"), patch.object(fetch_data.config, "START_DATE", "2019-01-01"), patch.object(fetch_data, "get_provider", return_value=provider), patch.object(fetch_data.time, "sleep"), redirect_stdout(StringIO()):
            fetch_data.save_data("BTC", daily, "1d")
            fetch_data.save_data("BTC", hourly.iloc[:2], "1h")
            self.assertEqual(fetch_data.main(limit=1, timeframe="1h"), 0)
            self.assertEqual(provider.fetch_ohlcv.call_args.kwargs["timeframe"], "1h")
            self.assertEqual(provider.fetch_ohlcv.call_args.args[2], "2025-01-01")
            self.assertEqual(len(fetch_data.read_existing("BTC", "1h")), 4)
            assert_frame_equal(fetch_data.read_existing("BTC", "1d"), daily, check_freq=False)

    def test_intraday_analysis_keeps_completed_candles_from_today(self):
        """Hourly analysis must not discard today's completed candles as daily analysis would."""
        now = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
        df = bars(300)
        df.index = pd.date_range(end=now, periods=300, freq="h", name="date")
        result = analyze.analyze_coin(df, "BTC", timeframe="1h")
        self.assertEqual(result["timeframe"], "1h")
        self.assertEqual(result["data_as_of"], (now-pd.Timedelta(hours=1)).isoformat())
