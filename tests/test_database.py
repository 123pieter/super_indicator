"""Current database schema tests use disposable SQLite files."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
import storage
from tests.helpers import bars


class DatabaseTests(unittest.TestCase):
    def test_first_save_creates_only_ohlcv_and_timeframe_primary_key(self):
        """A fresh database needs only ohlcv, keyed by provider, symbol, timeframe, and timestamp."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.db"
            storage.save_data("BTC", bars(2), "1d", folder, path, "sqlite")
            with closing(sqlite3.connect(path)) as conn:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                self.assertEqual(tables, [("ohlcv",)])
                columns = conn.execute("PRAGMA table_info(ohlcv)").fetchall()
                primary_key = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
                self.assertEqual(primary_key, ["provider", "symbol", "timeframe", "date"])

    def test_repeated_saves_preserve_values_without_extra_tables(self):
        """Saving the same key again updates its values without duplicate rows or supporting tables."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.db"
            df = bars(2)
            storage.save_data("BTC", df, "1d", folder, path, "sqlite")
            df["close"] = 99.0
            storage.save_data("BTC", df, "1d", folder, path, "sqlite")
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT DISTINCT close FROM ohlcv").fetchall(), [(99.0,)])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0], 1)

    def test_two_providers_share_one_table_without_overwriting(self):
        """Matching symbol/timeframe/timestamp keys retain separate prices for each supplier."""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"test.db"
            polygon=bars(2)
            hyper=bars(2)
            hyper["close"]=999.
            storage.save_data("BTC",polygon,"1d",folder,path,"sqlite",provider="polygon")
            storage.save_data("BTC",hyper,"1d",folder,path,"sqlite",provider="hyperliquid")
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("SELECT provider,COUNT(*) FROM ohlcv GROUP BY provider ORDER BY provider").fetchall(),[("hyperliquid",2),("polygon",2)])
                self.assertEqual(conn.execute("SELECT DISTINCT close FROM ohlcv WHERE provider='hyperliquid'").fetchall(),[(999.,)])
            self.assertEqual(storage.list_symbols("1d",folder,path,"sqlite",provider="polygon"),["BTC"])
            self.assertEqual(storage.list_symbols("1d",folder,path,"sqlite",provider="hyperliquid"),["BTC"])
