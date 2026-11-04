"""Unit tests for numerical calculations and input preparation."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from analyze import compute_indicators, wilder, SIGNAL_COLS
from tests.helpers import bars


class IndicatorCalculationTests(unittest.TestCase):
    def test_rsi_rising_falling_and_flat_prices(self):
        """RSI must be 100 for gains only, 0 for losses only, and 50 when flat."""
        for direction, expected in [(1, 100), (-1, 0), (0, 50)]:
            with self.subTest(direction=direction):
                df = bars()
                close = 500 + direction * np.arange(len(df), dtype=float)
                df["open"] = df["close"] = close
                df["high"], df["low"] = close+2, close-2
                self.assertEqual(compute_indicators(df).rsi.iloc[-1], expected)

    def test_score_waits_for_200_bars(self):
        """The first 199 bars cannot produce a score; the 200th bar can."""
        result = compute_indicators(bars())
        self.assertTrue(result.super_score.iloc[:199].isna().all())
        self.assertTrue(result.super_score.iloc[199:].notna().all())

    def test_cci_uses_mean_absolute_deviation(self):
        """Compare CCI to an independently calculated 20-bar reference value."""
        df = bars()
        typical = ((df.high+df.low+df.close)/3).tail(20)
        expected = (typical.iloc[-1]-typical.mean()) / (.015*np.abs(typical-typical.mean()).mean())
        self.assertAlmostEqual(compute_indicators(df).cci.iloc[-1], expected)

    def test_input_is_not_modified(self):
        """Adding indicators must not modify the caller's original price frame."""
        original = bars()
        before = original.copy(deep=True)
        compute_indicators(original)
        assert_frame_equal(original, before)

    def test_unsorted_duplicate_dates_are_normalized(self):
        """Unsorted input becomes chronological; duplicate dates keep the last row."""
        df = bars()
        duplicate = df.iloc[[-1]].copy()
        duplicate["volume"] = 456.0
        result = compute_indicators(pd.concat([df.iloc[::-1], duplicate]))
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(result.index.is_unique)
        self.assertEqual(len(result), len(df))
        self.assertEqual(result.volume.iloc[-1], 456.0)

    def test_invalid_values_are_rejected(self):
        """Missing, infinite, and nonnumeric prices must fail rather than create signals."""
        for value in [np.nan, np.inf, -np.inf, "bad-price"]:
            with self.subTest(value=value):
                df = bars()
                df["close"] = df.close.astype(object)
                df.iloc[-1, df.columns.get_loc("close")] = value
                with self.assertRaises(ValueError):
                    compute_indicators(df)

    def test_scores_stay_between_zero_and_one(self):
        """A combined vote score must remain in [0, 1] after warm-up."""
        values = compute_indicators(bars()).super_score.dropna()
        self.assertTrue(values.between(0, 1).all())

    def test_wilder_sma_seed_and_recursion(self):
        """The first three observations seed at 2; the next follows Wilder's recursion."""
        actual = wilder(pd.Series([np.nan, 1., 2., 3., 8.]), 3)
        self.assertTrue(actual.iloc[:3].isna().all())
        self.assertEqual(actual.iloc[3], 2.)
        self.assertEqual(actual.iloc[4], 4.)

    def test_rsi_seed_known_gains_and_losses(self):
        """RSI14 starts from arithmetic means of the first 14 actual price changes."""
        df = bars(30)
        close = 100 + np.r_[0, np.cumsum([2., -1.]*14+[2.])]
        df["open"] = df["close"] = close
        df["high"], df["low"] = close+1, close-1
        ind = compute_indicators(df)
        self.assertTrue(np.isnan(ind.rsi.iloc[13]))
        self.assertAlmostEqual(ind.rsi.iloc[14], 100-100/(1+2))

    def test_atr_adx_known_trending_reference(self):
        """Constant true range seeds ATR at 4 and an uninterrupted uptrend gives ADX100."""
        ind = compute_indicators(bars())
        self.assertAlmostEqual(ind.atr.iloc[13], 4.)
        self.assertTrue(np.isnan(ind.atr.iloc[12]))
        self.assertAlmostEqual(ind.adx.iloc[-1], 100.)

    def test_flat_stochastic_and_nine_votes(self):
        """A flat price range leaves stochastic undefined; ATR never contributes a vote."""
        df = bars()
        df[["open", "high", "low", "close"]] = 100.
        ind = compute_indicators(df)
        self.assertTrue(ind.stoch_k.isna().all())
        self.assertEqual(len(SIGNAL_COLS), 9)
        self.assertNotIn("atr_signal", ind)
        np.testing.assert_allclose(ind.super_score.iloc[199:], ind[SIGNAL_COLS].iloc[199:].sum(axis=1)/9)

    def test_indicators_are_causal(self):
        """Appending later price data must not revise any earlier indicator or signal."""
        df = bars(350)
        before = compute_indicators(df.iloc[:300])
        after = compute_indicators(df).iloc[:300]
        assert_frame_equal(before, after)
