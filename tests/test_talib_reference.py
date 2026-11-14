"""Optional independent checks of the Research Lab indicator calculations."""
from unittest import TestCase

import numpy as np
import pandas as pd

from research_lab.strategies import _atr, dmi_values, rsi_value

try:
    import talib
except ModuleNotFoundError as exc:
    if exc.name != "talib":
        raise
    talib = None


class TalibReferenceTests(TestCase):
    def setUp(self):
        if talib is None:
            self.skipTest("Install requirements-indicator-tests.txt for TA-Lib comparisons")
        rng = np.random.default_rng(732)
        close = 100 + rng.normal(0, .5, 900).cumsum()
        self.bars = pd.DataFrame(dict(high=close+rng.uniform(0, 2, len(close)),
                                      low=close-rng.uniform(0, 2, len(close)), close=close))

    def test_rsi_matches_reference_on_nonflat_prices(self):
        """Check RSI values and warmup for both the fast strategy and standard periods."""
        for period in (2, 14):
            with self.subTest(period=period):
                close = self.bars.close
                np.testing.assert_allclose(rsi_value(close, period), talib.RSI(close, period), atol=1e-10)

    def test_atr_and_dmi_match_reference(self):
        """Check the first valid bar and compare the entire output."""
        h, low, close = (self.bars[c] for c in ("high", "low", "close"))
        period = 14
        atr = _atr(self.bars, period)
        ref_atr = talib.ATR(h, low, close, period)
        self.assertEqual(atr.first_valid_index(), period)
        self.assertEqual(ref_atr.first_valid_index(), period)
        adx, plus, minus = dmi_values(self.bars, period)
        ref_adx = talib.ADX(h, low, close, period)
        self.assertEqual(adx.first_valid_index(), 2*period-1)
        self.assertEqual(ref_adx.first_valid_index(), adx.first_valid_index())
        for ours, reference in ((atr, ref_atr), (adx, ref_adx),
                                (plus, talib.PLUS_DI(h, low, close, period)),
                                (minus, talib.MINUS_DI(h, low, close, period))):
            np.testing.assert_allclose(ours, reference, rtol=0, atol=0)

    def test_flat_price_conventions(self):
        """Document the neutral flat RSI convention and verify zero ATR and ADX."""
        flat = self.bars.assign(high=100., low=100., close=100.)
        close = flat.close
        self.assertEqual(rsi_value(close, 14).iloc[-1], 50.)
        self.assertEqual(talib.RSI(close, 14).iloc[-1], 0.)
        self.assertEqual(_atr(flat, 14).iloc[-1], talib.ATR(close, close, close, 14).iloc[-1])
        self.assertEqual(dmi_values(flat, 14)[0].iloc[-1], talib.ADX(close, close, close, 14).iloc[-1])

    def test_registered_aroon_and_kama_preserve_direction_and_warmup(self):
        """Compare both strategy votes against direct TA-Lib outputs."""
        from research_lab.runner import build_signal, validate_config

        c = self.bars.close
        aroon = talib.AROONOSC(self.bars.high, self.bars.low, 25)
        kama = talib.KAMA(c, 30)
        for rule, long, short in (
            ("aroon_oscillator", aroon.gt(50), aroon.lt(-50)),
            ("kama_trend", c.gt(kama), c.lt(kama)),
        ):
            config = validate_config(dict(version=1, name=rule, provider="hyperliquid",
                                          components=[dict(rule=rule)]))
            np.testing.assert_array_equal(build_signal(self.bars, config),
                                          long.astype(int) - short.astype(int))

    def test_natr_gates_both_sides_at_inclusive_percent_threshold(self):
        """NATR is an entry filter with percent units and blocks warmup bars."""
        from research_lab.filters import entry_mask, validate_filters

        reference = talib.NATR(self.bars.high, self.bars.low, self.bars.close, 14)
        threshold = reference.iloc[14]
        filters = validate_filters([dict(filter_rule="natr_max", threshold=float(threshold))])
        cache = {}
        for side in ("long", "short"):
            allowed = entry_mask(self.bars, filters, side, cache)
            np.testing.assert_array_equal(allowed, reference.le(threshold))
            self.assertTrue(allowed.iloc[14])
            self.assertFalse(allowed.iloc[:14].any())

    def test_tiny_prices_and_flat_start_preserve_rsi(self):
        """RSI is scale invariant, with the established neutral flat warmup."""
        close = pd.Series([10.] * 16 + [12., 11., 15., 9., 10.])
        np.testing.assert_allclose(rsi_value(close * 2**-50, 14), rsi_value(close, 14), atol=1e-10)
        self.assertEqual(rsi_value(close, 14).iloc[14], 50.)

    def test_short_hull_windows_and_flat_prices(self):
        """Period-one WMA components act as identity and a flat price casts no vote."""
        from research_lab.strategies import hma_direction

        close = pd.Series([0.3] * 30)
        for window in (2, 3, 55):
            self.assertTrue(hma_direction(close, window).eq(0).all())
        rising = pd.Series(range(1, 20))
        self.assertTrue(hma_direction(rising, 2).iloc[2:].eq(1).all())

    def test_new_talib_rules_use_registered_parameters_and_completed_prefix(self):
        """Each rule stays flat through warmup and gives the same votes on a truncated history."""
        from research_lab.runner import build_signal, strategy_warmup, validate_config

        candles = self.bars.assign(volume=np.arange(len(self.bars), dtype=float)+1)
        for component in (
            dict(rule="parabolic_sar", acceleration=.03),
            dict(rule="ultimate_oscillator", fast=5, medium=10, slow=21),
            dict(rule="chaikin_oscillator", fast=4, slow=12),
        ):
            cfg = validate_config(dict(version=1, name=component["rule"], provider="hyperliquid",
                                       components=[component], execution=dict(stop_atr=None, target_atr=None)))
            full = build_signal(candles, cfg)
            self.assertFalse(full.iloc[:strategy_warmup(cfg)-1].any())
            self.assertTrue(full.any())
            pd.testing.assert_series_equal(build_signal(candles.iloc[:425], cfg), full.iloc[:425])

    def test_new_price_and_candle_rules_follow_talib(self):
        """Check trend and pressure signs, plus the engulfing hold and reversal."""
        from unittest.mock import patch
        from research_lab.strategies import balance_of_power, engulfing_pattern, linear_regression_trend

        close = self.bars.close
        candles = self.bars.assign(open=close.shift().fillna(close))
        slope = talib.LINEARREG_SLOPE(close, 30)
        pressure = talib.SMA(talib.BOP(candles.open, candles.high, candles.low, close), 14)
        np.testing.assert_array_equal(linear_regression_trend(close), np.sign(slope.fillna(0)))
        np.testing.assert_array_equal(balance_of_power(candles), np.sign(pressure.fillna(0)))
        candles = candles.iloc[:8]
        pattern = pd.Series([0, 100, 0, -100, 0, 0, 0, 0], index=candles.index)
        with patch("talib.CDLENGULFING", return_value=pattern):
            self.assertEqual(engulfing_pattern(candles, 3).tolist(), [0, 1, 1, -1, -1, -1, 0, 0])
