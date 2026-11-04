"""Cross-coin statistics and full-grid regression tests with deterministic data."""
import unittest
import numpy as np
import pandas as pd
from indicator_backtests import aggregate_coins, compare_coin_universe
from tests.helpers import bars


class CoinAggregationTests(unittest.TestCase):
    def sample(self):
        """Two coins have unequal trade counts so equal-coin and pooled averages differ."""
        index = pd.date_range("2020-01-01",periods=3)
        results = {}
        rows = []
        for coin,curve,trades in [("AAA",[1,1.1,.88],[.1,-.2]),("BBB",[1,1,1.3],[.3])]:
            results[coin] = {"results": {"rsi:long":dict(label="RSI",direction="long",equity=pd.Series(curve,index=index),trades=pd.DataFrame({"net_return":trades}))}}
            rows.append(dict(Coin=coin,Experiment="RSI",Direction="long",Error="",n_trades=len(trades),total_return_pct=(curve[-1]-1)*100,max_drawdown_pct=-20 if coin=="AAA" else 0))
        return pd.DataFrame(rows),results

    def test_equal_coin_returns_and_pooled_trade_statistics(self):
        """Returns average across coins; win rate and PF pool trades rather than average ratios."""
        rows,results = self.sample()
        r = aggregate_coins(rows,results,2).iloc[0]
        self.assertAlmostEqual(r.mean_coin_return_pct,9.)
        self.assertAlmostEqual(r.basket_return_pct,9.)
        self.assertAlmostEqual(r.pooled_avg_trade_pct,100*.2/3)
        self.assertAlmostEqual(r.pooled_win_pct,200/3)
        self.assertAlmostEqual(r.pooled_profit_factor,2.)
        self.assertEqual(r.total_trades,3)
        self.assertEqual(r.profitable_coins,1)

    def test_basket_drawdown_uses_equity_not_average_drawdown(self):
        """Offsetting paths give zero basket drawdown even though mean coin drawdown is -10%."""
        rows,results = self.sample()
        r = aggregate_coins(rows,results,2).iloc[0]
        self.assertAlmostEqual(r.basket_drawdown_pct,0.)
        self.assertAlmostEqual(r.mean_coin_drawdown_pct,-10.)

    def test_failed_coin_is_counted_and_basket_unavailable(self):
        """Missing data cannot disappear from coverage or become a silent zero-loss portfolio allocation."""
        rows,results = self.sample()
        rows.loc[len(rows)] = dict(Coin="CCC",Experiment="RSI",Direction="long",Error="missing data",n_trades=0,total_return_pct=np.nan,max_drawdown_pct=np.nan)
        r = aggregate_coins(rows,results,3).iloc[0]
        self.assertEqual(r.valid_coins,2)
        self.assertEqual(r.failed_coins,1)
        self.assertTrue(np.isnan(r.basket_return_pct))
        self.assertAlmostEqual(r.mean_coin_return_pct,9.)

    def test_full_grid_and_common_dates(self):
        """Every selected coin receives every selected candidate/direction on the shared date intersection."""
        histories = {"AAA":bars(300),"BBB":bars(280)}
        events = []
        r = compare_coin_universe(histories,histories.get,["ema","rsi"],on_progress=lambda *args: events.append(args))
        self.assertEqual(len(r["per_coin"]),14)
        self.assertEqual(len(r["aggregate"]),7)
        self.assertEqual(r["per_coin"]["Evaluation start"].nunique(),1)
        self.assertEqual(r["per_coin"]["Evaluation end"].nunique(),1)
        self.assertEqual(r["end"],bars(280).index[-1])
        self.assertEqual(events[-1],(2,2,"BBB"))

    def test_coin_failure_does_not_stop_other_combinations(self):
        """A missing coin creates explicit failed rows while valid coins continue and count in aggregates."""
        r = compare_coin_universe(["AAA","BAD"],lambda coin:bars() if coin=="AAA" else None,["rsi"],["long"])
        self.assertEqual(len(r["per_coin"]),6)
        self.assertEqual(r["per_coin"].query('Coin == "BAD"').Error.ne("").sum(),3)
        self.assertTrue(r["aggregate"].failed_coins.eq(1).all())
        self.assertTrue(r["aggregate"].basket_return_pct.isna().all())

    def test_all_failed_and_empty_selection_are_safe(self):
        """An entirely unavailable universe reports coverage rather than fabricating returns or crashing."""
        r = compare_coin_universe(["BAD"],lambda coin:None,["rsi"],["long"])
        self.assertFalse(r["coins"])
        self.assertTrue(r["aggregate"].valid_coins.eq(0).all())
        with self.assertRaises(ValueError):
            compare_coin_universe([],lambda coin:bars(),["rsi"])
