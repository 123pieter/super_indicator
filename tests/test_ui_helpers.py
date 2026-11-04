"""Unit tests for display logic; no browser or live API is needed."""
import unittest
import pandas as pd
from ui_helpers import freshness, filter_scanner, format_price


class PresentationTests(unittest.TestCase):
    def test_freshness_allows_the_forming_candle(self):
        """Yesterday's daily candle is current until today's candle has completed."""
        self.assertEqual(freshness("2025-01-01", "1d", "2025-01-02 18:00Z"), ("Up to date", 0))
        self.assertEqual(freshness("2025-01-01", "1d", "2025-01-03 00:00Z")[1], 1)
        self.assertEqual(freshness("2025-01-01 09:00", "1h", "2025-01-01 11:00Z")[1], 1)

    def test_missing_candles_have_explicit_status(self):
        """An empty dataset cannot be presented as up to date."""
        self.assertEqual(freshness(None,"1d"), ("No completed data",None))

    def test_tiny_prices_keep_significant_digits(self):
        """Sub-cent tokens retain a visible nonzero price while large prices use separators."""
        self.assertNotEqual(format_price(.00000012), "$0")
        self.assertEqual(format_price(12345.678), "$12,345.68")

    def test_scanner_filters_and_ranking(self):
        """Fresh signals, evidence, regime, score, and search filters compose without mutating input."""
        rows = pd.DataFrame([
            dict(symbol="BTC", current_score=.9, fresh_cross=True, sufficient_history=True, regime_ok=True),
            dict(symbol="ETH", current_score=.8, fresh_cross=False, sufficient_history=True, regime_ok=True),
            dict(symbol="SOL", current_score=.9, fresh_cross=True, sufficient_history=False, regime_ok=False),
        ])
        result = filter_scanner(rows, minimum=.8, fresh=True, regime=True, sufficient=True, search="bt")
        self.assertEqual(result.symbol.tolist(), ["BTC"])
        self.assertEqual(len(rows),3)
        self.assertEqual(filter_scanner(rows, search="[").symbol.tolist(), [])
        self.assertEqual(filter_scanner(rows).symbol.tolist(), ["BTC","SOL","ETH"])
