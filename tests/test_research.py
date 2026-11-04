"""Deterministic execution tests with hand-calculated trades; no API or database writes."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from unittest.mock import patch
import config
from research import backtest, compare_strategies, evaluate_history


def market(n=8):
    """Flat, valid daily candles; a known crossing at index1 enters index2."""
    d = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1.,
                          super_score=0.8, atr=1., regime_ok=True, ready=True,
                          sma50=110., sma200=100.), index=pd.date_range("2020-01-01", periods=n))
    d.iloc[0, d.columns.get_loc("super_score")] = 0.
    return d


class ResearchTests(unittest.TestCase):
    def run_market(self, d=None, **kwargs):
        """Zero costs isolate execution; individual cost tests override these assumptions."""
        return backtest(market() if d is None else d, fee=kwargs.pop("fee", 0),
                        slippage=kwargs.pop("slippage", 0), **kwargs)

    def test_next_open_entry_and_end_liquidation(self):
        """The signal close is not an execution price; final positions close at the last close."""
        d = market()
        d.loc[d.index[2], ["open", "high", "low", "close"]] = [105, 106, 104, 105]
        r = self.run_market(d, stop_atr=100, tp_atr=None)
        trade = r["trades"].iloc[0]
        self.assertEqual(trade.entry_time, d.index[2])
        self.assertEqual(trade.entry_price, 105)
        self.assertAlmostEqual(trade.net_return, 100/105-1)
        self.assertEqual(trade.reason, "end_of_data")

    def test_stop_wins_when_both_levels_touched(self):
        """An ambiguous candle touching stop97 and target106 exits at97, conservatively."""
        d = market()
        d.loc[d.index[2], ["high", "low"]] = [107, 96]
        t = self.run_market(d)["trades"].iloc[0]
        self.assertEqual(t.reason, "stop")
        self.assertAlmostEqual(t.net_return, -.03)

    def test_stop_gap_executes_at_open(self):
        """An opening gap below the stop fills at the worse open rather than at the stop."""
        d = market()
        d.loc[d.index[3], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
        t = self.run_market(d)["trades"].iloc[0]
        self.assertEqual(t.reason, "stop_gap")
        self.assertEqual(t.exit_price, 90)

    def test_take_profit_and_gap_target(self):
        """Target fills at its level intrabar or at the better open on an opening gap."""
        for gap, expected in [(False,106), (True,110)]:
            d = market()
            if gap:
                d.loc[d.index[3], ["open", "high", "low", "close"]] = [110,111,109,110]
            else:
                d.loc[d.index[3], "high"] = 107
            self.assertEqual(self.run_market(d)["trades"].iloc[0].exit_price, expected)

    def test_score_exit_is_next_open_before_intrabar_stop(self):
        """A known prior-close exit executes at the open before that candle's later low."""
        d = market()
        d.loc[d.index[3], "super_score"] = .1
        d.loc[d.index[4], "low"] = 90
        t = self.run_market(d)["trades"].iloc[0]
        self.assertEqual(t.reason, "score_exit")
        self.assertEqual(t.exit_time, d.index[4])
        self.assertEqual(t.exit_price, 100)

    def test_max_hold_includes_entry_candle(self):
        """A two-bar hold enters at index2 and exits at the close of index3."""
        t = self.run_market(max_hold=2)["trades"].iloc[0]
        self.assertEqual(t.exit_time, market().index[3])
        self.assertEqual(t.bars, 2)

    def test_fees_and_slippage_charged_both_sides(self):
        """Flat prices still lose the exact multiplicative entry and exit costs."""
        r = self.run_market(fee=.01, slippage=.02)
        expected = (.98*.99)/(1.02*1.01)-1
        self.assertAlmostEqual(r["trades"].iloc[0].net_return, expected)
        self.assertAlmostEqual(r["stats"]["total_return_pct"], expected*100)

    def test_drawdown_includes_first_loss(self):
        """A first trade losing 10% must report10% drawdown from initial capital, not zero."""
        d = market()
        d.loc[d.index[3], ["open", "high", "low", "close"]] = [90,91,89,90]
        r = self.run_market(d)
        self.assertAlmostEqual(r["stats"]["max_drawdown_pct"], -10)
        self.assertEqual(r["equity"].iloc[0], 1.)

    def test_open_position_drawdown_is_marked_each_close(self):
        """A recovered open trade still records its intervening 20% close-to-close drawdown."""
        d = market()
        d.loc[d.index[3], ["open", "high", "low", "close"]] = [100,101,79,80]
        r = self.run_market(d, stop_atr=50, tp_atr=None)
        self.assertAlmostEqual(r["stats"]["total_return_pct"], 0)
        self.assertAlmostEqual(r["stats"]["max_drawdown_pct"], -20)

    def test_exit_candle_cross_can_reenter_next_open(self):
        """A crossing on an intrabar-stop exit candle is eligible at the following open."""
        d = market()
        d.loc[d.index[2], "super_score"] = 0
        d.loc[d.index[3], "low"] = 96
        r = self.run_market(d)
        self.assertEqual(len(r["trades"]), 2)
        self.assertEqual(r["trades"].iloc[1].entry_time, d.index[4])
        self.assertTrue((r["trades"].entry_time.iloc[1:].to_numpy() > r["trades"].exit_time.iloc[:-1].to_numpy()).all())

    def test_regime_blocks_entries_and_missing_prior_score_blocks_cross(self):
        """Neither volatility-ineligible crossings nor a NaN-to-high warmup open trades."""
        for regime, prior in [(False,0.), (True,np.nan)]:
            d = market()
            d["regime_ok"] = regime
            d.loc[d.index[0], "super_score"] = prior
            self.assertTrue(self.run_market(d)["trades"].empty)

    def test_no_trades_and_empty_history_are_safe(self):
        """No fabricated win rates or profit factors when no trade exists; empty input is safe."""
        d = market()
        d["super_score"] = 0
        r = self.run_market(d)
        self.assertEqual(r["stats"]["n_trades"], 0)
        self.assertTrue(np.isnan(r["stats"]["win_pct"]))
        self.assertTrue(self.run_market(d.iloc[:0])["trades"].empty)

    def test_parameter_validation(self):
        """Reject invalid costs, holding periods, thresholds, and stop/target distances."""
        for options in [dict(fee=-1),dict(slippage=1),dict(stop_atr=0),dict(tp_atr=0),
                        dict(max_hold=0),dict(max_hold=1.5),dict(threshold=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_market(**options)

    def test_gaps_disable_research_without_hiding_live_coin(self):
        """Missing candles produce an explicit research error instead of a misleading simulation."""
        d = market().drop(market().index[3])
        r = evaluate_history(d)
        self.assertIn("missing candles", r["research_error"])
        self.assertFalse(r["sufficient_history"])
        self.assertTrue(np.isnan(r["total_return_pct"]))

    def test_benchmarks_use_same_dates_and_costs(self):
        """Always-bullish SMA and buy-and-hold have identical equity under identical costs."""
        results = compare_strategies(market())
        self.assertEqual(len({r["start"] for r in results.values()}), 1)
        self.assertEqual(len({r["end"] for r in results.values()}), 1)
        assert_series_equal(results["sma"]["equity"], results["buy_hold"]["equity"])
        expected = (.9995*.999)/(1.0005*1.001)-1
        self.assertAlmostEqual(results["buy_hold"]["stats"]["total_return_pct"],expected*100)

    def test_provider_costs_are_independent(self):
        """The selected provider determines costs without modifying another provider's assumptions."""
        costs = dict(polygon=dict(fee=0.,slippage=0.),hyperliquid=dict(fee=.01,slippage=0.))
        with patch.object(config, "BACKTEST_COSTS", costs):
            a = compare_strategies(market(), "polygon")
            b = compare_strategies(market(), "hyperliquid")
        self.assertEqual(a["super"]["stats"]["total_return_pct"], 0)
        self.assertLess(b["super"]["stats"]["total_return_pct"], 0)

    def test_later_prices_cannot_change_entry_decisions_in_prefix(self):
        """Changing future candles cannot change an earlier entry time or execution price."""
        d = market()
        first = self.run_market(d)["trades"].iloc[0]
        d.loc[d.index[-1], ["open","high","low","close"]] = [200,201,199,200]
        second = self.run_market(d)["trades"].iloc[0]
        self.assertEqual(first.entry_time, second.entry_time)
        self.assertEqual(first.entry_price, second.entry_price)

    def test_compounded_capital_is_reused_without_overlap(self):
        """Two independent one-bar winners compound capital rather than adding percentages."""
        d = market()
        d["super_score"] = [0,.8,0,.8,.8,.8,.8,.8]
        for i in (2,4):
            d.loc[d.index[i], ["high", "close"]] = [111,110]
        r = self.run_market(d, max_hold=1, tp_atr=None)
        self.assertEqual(r["stats"]["n_trades"], 2)
        self.assertAlmostEqual(r["stats"]["total_return_pct"], 21.)

    def test_sma_uses_prior_candle_and_next_open(self):
        """SMA turns bullish at index2 and bearish at index4, trading opens3 and5."""
        d = market()
        d["sma50"] = [90,90,110,110,90,90,90,90]
        r = self.run_market(d, strategy="sma")
        t = r["trades"].iloc[0]
        self.assertEqual(t.entry_time, d.index[3])
        self.assertEqual(t.exit_time, d.index[5])
        self.assertEqual(t.reason, "sma_exit")

    def test_intraday_research_uses_selected_interval(self):
        """Hourly candles simulate normally when 1h is selected, not against daily spacing."""
        d = market()
        d.index = pd.date_range("2020-01-01", periods=len(d), freq="h")
        self.assertEqual(self.run_market(d, timeframe="1h")["stats"]["n_trades"], 1)
        with self.assertRaisesRegex(ValueError,"irregular"):
            self.run_market(d, timeframe="1d")

    def test_no_ready_history_has_no_benchmark_return(self):
        """Before warmup finishes no strategy or benchmark fabricates an evaluation span."""
        d = market()
        d["ready"] = False
        for r in compare_strategies(d).values():
            self.assertIsNone(r["start"])
            self.assertTrue(np.isnan(r["stats"]["total_return_pct"]))

    def test_short_profit_and_gap_stop(self):
        """A short gains on a falling market; an adverse stop gap executes at its open."""
        for price,expected in [(90,.10),(110,-.10)]:
            d = market()
            d.loc[d.index[3],["open","high","low","close"]] = [price,price+1,price-1,price]
            r = self.run_market(d,direction="short")
            self.assertAlmostEqual(r["trades"].iloc[0].net_return,expected)

    def test_short_stop_wins_ambiguous_touches(self):
        """A short touching stop103 and target94 in one candle exits at the losing stop."""
        d = market()
        d.loc[d.index[2],["high","low"]] = [104,93]
        t = self.run_market(d,direction="short")["trades"].iloc[0]
        self.assertEqual(t.reason,"stop")
        self.assertAlmostEqual(t.net_return,-.03)

    def test_short_target_and_costs(self):
        """Short target profit and two-sided fees/slippage match explicit collateral P&L."""
        d = market()
        d.loc[d.index[3],"low"] = 93
        self.assertAlmostEqual(self.run_market(d,direction="short")["trades"].iloc[0].net_return,.06)
        t = self.run_market(direction="short",fee=.01,slippage=.02)["trades"].iloc[0]
        units = 1/(98*1.01)
        self.assertAlmostEqual(t.net_return,units*(98-102)-.01*units*(98+102))

    def test_short_capital_exhaustion_is_not_clipped(self):
        """An adverse short gap beyond available collateral fails rather than capping losses."""
        d = market()
        d.loc[d.index[3],["open","high","low","close"]] = [250,251,249,250]
        with self.assertRaisesRegex(ValueError,"capital exhausted"):
            self.run_market(d,direction="short")

    def test_atr_exits_can_be_disabled(self):
        """Turning ATR exits off ignores an intrabar stop touch when testing the entry rule alone."""
        d = market()
        d.loc[d.index[3],"low"] = 90
        self.assertEqual(self.run_market(d,atr_exits=False)["trades"].iloc[0].reason,"end_of_data")
