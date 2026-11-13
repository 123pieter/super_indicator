import sqlite3
from unittest import TestCase

import pandas as pd

from research_lab.funding import load_candle_funding


def event(hour, rate, interval):
    return pd.Timestamp(hour, tz="UTC").value // 1_000_000 + 31, rate, interval


class CandleFundingTests(TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.execute("""CREATE TABLE funding_rates (
            provider TEXT, symbol TEXT, time_ms INTEGER, funding_rate REAL, interval_hours REAL,
            PRIMARY KEY (provider, symbol, time_ms)
        )""")

    def insert(self, provider, records):
        self.conn.executemany("INSERT INTO funding_rates VALUES (?, 'BTC', ?, ?, ?)",
                              ((provider, *row) for row in records))

    def test_hyperliquid_preserves_both_signed_components(self):
        index = pd.date_range("2024-01-01", periods=2, freq="4h", name="date")
        hourly = pd.date_range("2024-01-01", periods=8, freq="h")
        rates = (0.001, -0.002, 0.003, 0., -0.001, 0.002, -0.003, 0.)
        self.insert("hyperliquid", (event(hour, rate, 1) for hour, rate in zip(hourly, rates)))
        result = load_candle_funding(self.conn, "hyperliquid", "BTC", index, "4h")
        expected = pd.DataFrame({"signed_rate": [0.002, -0.002],
                                 "positive_rate": [0.004, 0.002],
                                 "negative_rate": [-0.002, -0.004]}, index=index)
        pd.testing.assert_frame_equal(result, expected)
        self.conn.execute("DELETE FROM funding_rates WHERE time_ms=?", (event(hourly[2], 0, 1)[0],))
        with self.assertRaisesRegex(ValueError, "every UTC hour"):
            load_candle_funding(self.conn, "hyperliquid", "BTC", index, "4h")

    def test_binance_empty_candles_require_verified_surrounding_events(self):
        index = pd.date_range("2024-01-01 04:00", periods=3, freq="4h", tz="UTC")
        self.insert("binance", [event("2024-01-01 00:00", 0.003, 8),
                                event("2024-01-01 08:00", -0.002, 8),
                                event("2024-01-01 16:00", 0.001, 8)])
        result = load_candle_funding(self.conn, "binance", "BTC", index, "4h")
        self.assertEqual(result.signed_rate.tolist(), [0., -0.002, 0.])
        self.conn.execute("DELETE FROM funding_rates WHERE time_ms=?", (event("2024-01-01 16:00", 0, 8)[0],))
        with self.assertRaisesRegex(ValueError, "bracketing"):
            load_candle_funding(self.conn, "binance", "BTC", index, "4h")

    def test_binance_variable_schedule_rejects_unexplained_gap(self):
        index = pd.date_range("2024-01-01 04:00", periods=4, freq="4h", tz="UTC")
        self.insert("binance", [event("2024-01-01 00:00", 0.003, 8),
                                event("2024-01-01 08:00", 0.002, 8),
                                event("2024-01-01 12:00", 0.002, 4),
                                event("2024-01-01 16:00", 0.001, 4),
                                event("2024-01-01 20:00", 0.001, 4)])
        result = load_candle_funding(self.conn, "binance", "BTC", index, "4h")
        self.assertEqual(result.signed_rate.tolist(), [0., 0.002, 0.002, 0.001])
        self.conn.execute("DELETE FROM funding_rates WHERE time_ms=?", (event("2024-01-01 12:00", 0, 4)[0],))
        with self.assertRaisesRegex(ValueError, "irregular Binance settlement schedule"):
            load_candle_funding(self.conn, "binance", "BTC", index, "4h")
