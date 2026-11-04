"""Disposable local CSV/SQLite persistence tests; real user data is never used."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pandas.testing import assert_frame_equal
import analyze
import fetch_data
from tests.helpers import bars


class StorageTests(unittest.TestCase):
    def setUp(self):
        """Redirect all reads and writes into a temporary folder, removed after each test."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        for module in [fetch_data.config, analyze]:
            self.stack.enter_context(patch.object(module, "DATA_FOLDER", self.folder))
            self.stack.enter_context(patch.object(module, "DB_PATH", str(Path(self.folder)/"test.db")))

    def test_missing_history_is_empty(self):
        """Reading an unsaved symbol returns empty history without inventing candles."""
        with patch.object(fetch_data.config, "SAVE_TO", "both"):
            self.assertTrue(fetch_data.read_existing("MISSING").empty)

    def test_backends_round_trip_and_replace_overlap(self):
        """CSV and SQLite preserve prices, replace overlapping dates, and list saved symbols."""
        for backend in ["csv", "sqlite"]:
            with self.subTest(backend=backend), patch.object(fetch_data.config, "SAVE_TO", backend), patch.object(analyze, "LOAD_FROM", backend):
                df = bars(10)
                fetch_data.save_data("TEST", df)
                df.iloc[-1, df.columns.get_loc("close")] = 123.0
                fetch_data.save_data("TEST", df)
                assert_frame_equal(analyze.load_data("TEST"), df, check_freq=False)
                self.assertEqual(analyze.get_available_symbols(), ["TEST"])
                self.assertIsNone(analyze.load_data("MISSING"))

    def test_dual_storage_deduplicates(self):
        """Identical candles read from CSV plus SQLite produce one row per date."""
        with patch.object(fetch_data.config, "SAVE_TO", "both"):
            fetch_data.save_data("TEST", bars(10))
            loaded = fetch_data.read_existing("TEST")
        self.assertEqual(len(loaded), 10)
        self.assertTrue(loaded.index.is_unique)

    def test_csv_replacement_leaves_no_temporary_file(self):
        """Successful CSV persistence leaves a final file and no unfinished .tmp file."""
        with patch.object(fetch_data.config, "SAVE_TO", "csv"):
            fetch_data.save_data("TEST", bars(10))
        self.assertTrue((Path(self.folder)/"polygon"/"1d"/"TEST.csv").exists())
        self.assertFalse((Path(self.folder)/"polygon"/"1d"/"TEST.csv.tmp").exists())

    def test_database_schema_errors_are_visible(self):
        """Missing SQLite tables raise actionable errors instead of silently hiding failures."""
        with patch.object(analyze, "LOAD_FROM", "sqlite"):
            with self.assertRaisesRegex(RuntimeError, "Unable to list symbols"):
                analyze.get_available_symbols()
            with self.assertRaisesRegex(RuntimeError, "Unable to load TEST"):
                analyze.load_data("TEST")
