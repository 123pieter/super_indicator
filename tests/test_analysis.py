"""Live-signal tests deliberately isolate current decisions from historical research."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import analyze
from tests.helpers import bars


class AnalysisTests(unittest.TestCase):
    def evaluate(self, scores, threshold=.7, regime=True):
        """Inject known scores to test crossing semantics independently of indicator maths."""
        df = bars(len(scores))
        calculated = analyze.compute_indicators(df)
        calculated["super_score"] = scores
        calculated["regime_ok"] = regime
        with patch.object(analyze, "compute_indicators", return_value=calculated):
            return analyze.analyze_coin(df, "TEST", threshold)

    def test_insufficient_bars_returns_none(self):
        """Skip insufficient live history before any indicator or research work."""
        with patch.object(analyze, "compute_indicators") as compute:
            self.assertIsNone(analyze.analyze_coin(bars(analyze.MIN_BARS-1), "TEST"))
            compute.assert_not_called()

    def test_inclusive_cross_and_regime(self):
        """Exactly reaching the chosen threshold signals only when ATR regime is eligible."""
        scores = np.zeros(300)
        scores[-1] = .8
        self.assertTrue(self.evaluate(scores, .8)["fresh_cross"])
        self.assertFalse(self.evaluate(scores, .8, False)["fresh_cross"])

    def test_no_repeated_or_warmup_cross(self):
        """Remaining above threshold and first valid scores after NaN are not new crosses."""
        scores = np.zeros(300)
        scores[-2:] = .8
        self.assertFalse(self.evaluate(scores)["fresh_cross"])
        scores[-2] = np.nan
        self.assertFalse(self.evaluate(scores)["fresh_cross"])

    def test_live_path_has_no_research_or_future_columns(self):
        """Current signals neither invoke a backtest nor return forward-outcome statistics."""
        with patch("research.backtest", side_effect=AssertionError("Live invoked research")):
            result = analyze.analyze_coin(bars(), "TEST")
        self.assertIn("votes_met", result)
        self.assertNotIn("n_trades", result)
        self.assertFalse(any(key.startswith("ret_") for key in result))

    def test_today_and_future_candles_are_excluded(self):
        """Incomplete UTC candles and future prices cannot affect the latest signal."""
        df = bars()
        today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        df.loc[today] = 99999.0
        df.loc[today+pd.Timedelta(days=1)] = 99999.0
        result = analyze.analyze_coin(df, "TEST")
        self.assertEqual(result["data_as_of"], bars().index[-1].isoformat())
        self.assertEqual(result["price"], bars().close.iloc[-1])
