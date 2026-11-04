"""Candidate rules, common windows and independent directional experiments."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import analyze
from indicator_backtests import CANDIDATES, candidate_condition, compare_candidates
from tests.helpers import bars


class CandidateTests(unittest.TestCase):
    def test_long_rules_match_existing_votes(self):
        """Bullish candidate rules use the actual votes without other indicator contributions."""
        d = analyze.compute_indicators(bars())
        for name in CANDIDATES:
            np.testing.assert_array_equal(candidate_condition(d,name,"long"),d[f"{name}_signal"].eq(1))

    def test_short_rules_are_not_bullish_inverses(self):
        """Flat RSI, extreme positive CCI and overbought stochastic are not bearish conditions."""
        d = analyze.compute_indicators(bars())
        d["rsi"],d["cci"],d["stoch_k"],d["stoch_d"] = 50.,150.,90.,95.
        for name in ("rsi","cci","stoch"):
            self.assertFalse(candidate_condition(d,name,"short").any())
        d["rsi"],d["cci"],d["stoch_k"],d["stoch_d"] = 49.,-50.,40.,50.
        for name in ("rsi","cci","stoch"):
            self.assertTrue(candidate_condition(d,name,"short").all())

    def test_undefined_inputs_do_not_short(self):
        """Undefined RSI or stochastic inputs never manufacture a bearish entry."""
        d = analyze.compute_indicators(bars())
        d["rsi"],d["stoch_k"] = np.nan,np.nan
        self.assertFalse(candidate_condition(d,"rsi","short").any())
        self.assertFalse(candidate_condition(d,"stoch","short").any())

    def test_common_window_and_directions(self):
        """Every candidate and baseline uses identical dates after the 200-bar warmup."""
        r = compare_candidates(bars(),["rsi","ema"],fee=0,slippage=0)
        self.assertEqual(len(r["summary"]),7)
        self.assertEqual(len({x["start"] for x in r["results"].values()}),1)
        self.assertEqual(len({x["end"] for x in r["results"].values()}),1)
        self.assertEqual(r["start"],bars().index[200])

    def test_date_filter_preserves_warmup_and_previous_signal(self):
        """A signal just before the selected start can enter at the first selected open."""
        raw = bars()
        d = analyze.compute_indicators(raw)
        d["rsi_signal"] = 0.
        d.loc[d.index[249]:,"rsi_signal"] = 1.
        with patch("indicator_backtests.compute_indicators",return_value=d):
            r = compare_candidates(raw,["rsi"],["long"],start=raw.index[250],end=raw.index[260],fee=0,slippage=0)
        self.assertEqual(r["results"]["rsi:long"]["trades"].iloc[0].entry_time,raw.index[250])
        self.assertEqual(r["end"],raw.index[260])

    def test_regime_and_condition_exit_options(self):
        """ATR filtering is optional; failed entry conditions close at the following open."""
        raw = bars()
        d = analyze.compute_indicators(raw)
        d["rsi_signal"] = 0.
        d.loc[d.index[205:208],"rsi_signal"] = 1.
        d["regime_ok"] = False
        with patch("indicator_backtests.compute_indicators",return_value=d):
            allowed = compare_candidates(raw,["rsi"],["long"],fee=0,slippage=0)
            blocked = compare_candidates(raw,["rsi"],["long"],use_regime=True,fee=0,slippage=0)
            fixed = compare_candidates(raw,["rsi"],["long"],exit_on_condition=False,max_hold=10,fee=0,slippage=0)
        self.assertEqual(allowed["results"]["rsi:long"]["trades"].iloc[0].exit_time,raw.index[209])
        self.assertTrue(blocked["results"]["rsi:long"]["trades"].empty)
        self.assertEqual(fixed["results"]["rsi:long"]["trades"].iloc[0].reason,"max_hold")

    def test_inputs_unchanged_and_invalid_selections_rejected(self):
        """Experiments preserve source candles; unknown, duplicate and empty selections fail clearly."""
        raw = bars()
        before = raw.copy(deep=True)
        compare_candidates(raw,["rsi"],["long"])
        assert_frame_equal(raw,before)
        for names,sides in [([], ["long"]),(["unknown"],["long"]),(["rsi","rsi"],["long"]),(["rsi"],[])]:
            with self.assertRaises(ValueError):
                compare_candidates(raw,names,sides)

    def test_gaps_and_empty_date_ranges_fail(self):
        """Missing evaluation bars and reversed dates cannot produce apparently valid metrics."""
        raw = bars()
        with self.assertRaisesRegex(ValueError,"missing candles"):
            compare_candidates(raw.drop(raw.index[220]),["rsi"])
        with self.assertRaisesRegex(ValueError,"No completed evaluation"):
            compare_candidates(raw,["rsi"],start=raw.index[290],end=raw.index[250])

    def test_short_exhaustion_marks_only_failed_experiment_unavailable(self):
        """A bankrupt short cannot hide other experiments or appear as a clipped return."""
        raw = bars()
        d = analyze.compute_indicators(raw)
        d["ema12"],d["ema26"] = 3.,2.
        d.loc[d.index[205]:,"ema12"] = 1.
        d.loc[d.index[210],["open","high","low","close"]] = [900,902,898,900]
        with patch("indicator_backtests.compute_indicators",return_value=d):
            r = compare_candidates(raw,["ema"],["short"],fee=0,slippage=0)
        failed = r["summary"].iloc[0]
        self.assertIn("capital exhausted",failed.Error)
        self.assertTrue(np.isnan(failed.total_return_pct))
        self.assertNotIn("ema:short",r["results"])
        self.assertIn("buy_hold:long",r["results"])

    def test_future_candles_do_not_change_experiments(self):
        """Incomplete future rows cannot alter historical trade statistics or evaluation dates."""
        raw = bars()
        before = compare_candidates(raw,["rsi"],["long"])
        raw.loc[pd.Timestamp.now(tz="UTC").tz_localize(None)+pd.Timedelta(days=1)] = 999.
        after = compare_candidates(raw,["rsi"],["long"])
        assert_frame_equal(before["summary"],after["summary"])
        self.assertEqual(before["end"],after["end"])
