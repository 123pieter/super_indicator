"""Offline fetch tests with a mocked provider and disposable storage."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from pandas.testing import assert_frame_equal
import fetch_data
from tests.helpers import bars


class FetchingTests(unittest.TestCase):
    def setUp(self):
        """Replace API access, credentials, sleep calls, and storage with isolated test doubles."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        for name, value in {"DATA_FOLDER": folder, "DB_PATH": str(Path(folder)/"test.db"), "SAVE_TO": "both", "START_DATE": "2019-01-01", "POLYGON_API_KEY": "dummy"}.items():
            self.stack.enter_context(patch.object(fetch_data.config, name, value))
        self.provider = Mock()
        self.provider.get_available_symbols.return_value = {"BBB": "X:BBBUSD", "AAA": "X:AAAUSD"}
        self.provider.fetch_ohlcv.return_value = bars(12)
        self.factory = self.stack.enter_context(patch.object(fetch_data, "get_provider", return_value=self.provider))
        self.stack.enter_context(patch.object(fetch_data.time, "sleep"))
        self.output = StringIO()
        self.stack.enter_context(redirect_stdout(self.output))

    def test_limit_selects_first_sorted_pair(self):
        """A limit of one fetches only the first alphabetically sorted symbol."""
        self.assertEqual(fetch_data.main(limit=1), 0)
        self.provider.fetch_ohlcv.assert_called_once()
        self.assertEqual(self.provider.fetch_ohlcv.call_args.args[:2], ("AAA", "X:AAAUSD"))

    def test_zero_limit_fetches_all(self):
        """Zero disables the testing cap and processes every pair."""
        self.assertEqual(fetch_data.main(limit=0), 0)
        self.assertEqual(self.provider.fetch_ohlcv.call_count, 2)

    def test_incremental_overlap_and_merge(self):
        """Existing history requests a two-day overlap, preserves old dates, and replaces revised bars."""
        fetch_data.save_data("AAA", bars(10))
        new = bars(12).iloc[-3:].copy()
        new.iloc[0, new.columns.get_loc("close")] = 321.0
        self.provider.fetch_ohlcv.return_value = new
        self.assertEqual(fetch_data.main(limit=1), 0)
        args = self.provider.fetch_ohlcv.call_args.args
        self.assertEqual(args[2], "2020-01-08")
        self.assertEqual(args[3], (pd.Timestamp.now(tz="UTC").normalize()-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        loaded = fetch_data.read_existing("AAA")
        self.assertEqual(len(loaded), 12)
        self.assertEqual(loaded.loc[new.index[0], "close"], 321.0)

    def test_empty_response_preserves_saved_history(self):
        """No returned candles means a failure status; previous data remains intact."""
        original = bars(10)
        fetch_data.save_data("AAA", original)
        self.provider.fetch_ohlcv.return_value = None
        self.assertEqual(fetch_data.main(limit=1), 1)
        assert_frame_equal(fetch_data.read_existing("AAA"), original, check_freq=False)
        self.assertIn("ERROR AAA", self.output.getvalue())

    def test_failure_does_not_stop_later_pairs(self):
        """One provider exception is reported while the next symbol is still saved."""
        self.provider.fetch_ohlcv.side_effect = [RuntimeError("offline"), bars(12)]
        self.assertEqual(fetch_data.main(limit=0), 1)
        self.assertEqual(self.provider.fetch_ohlcv.call_count, 2)
        self.assertEqual(len(fetch_data.read_existing("BBB")), 12)

    def test_today_candle_is_not_saved(self):
        """A provider returning an unfinished UTC candle cannot put it into stored history."""
        df = bars(12)
        df.loc[pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)] = 999.0
        self.provider.fetch_ohlcv.return_value = df
        self.assertEqual(fetch_data.main(limit=1), 0)
        self.assertEqual(len(fetch_data.read_existing("AAA")), 12)

    def test_empty_universe_raises_error(self):
        """No available symbols must fail visibly rather than report a successful fetch."""
        self.provider.get_available_symbols.return_value = {}
        with self.assertRaisesRegex(RuntimeError, "no USD pairs"):
            fetch_data.main(limit=1)

    def test_invalid_storage_rejected_before_api_setup(self):
        """An invalid SAVE_TO setting must fail before constructing the provider."""
        with patch.object(fetch_data.config, "SAVE_TO", "invalid"):
            with self.assertRaisesRegex(ValueError, "SAVE_TO"):
                fetch_data.main(limit=1)
        self.factory.assert_not_called()

    def test_progress_reports_each_outcome_and_final_counts(self):
        """The UI receives per-coin success/error progress and an accurate completion summary."""
        events = []
        self.provider.fetch_ohlcv.side_effect = [bars(12), RuntimeError("network unavailable")]
        self.assertEqual(fetch_data.main(limit=0, on_progress=events.append), 1)
        self.assertEqual([e["event"] for e in events], ["started","fetching","success","fetching","error","finished"])
        self.assertEqual(events[-1]["succeeded"],1)
        self.assertEqual(events[-1]["failed"],1)
        self.assertEqual(events[-1]["completed"],2)
