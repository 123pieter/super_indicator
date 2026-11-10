"""Chart structure tests guard signal markers and aligned time axes."""
import unittest
from analyze import compute_indicators
from charts import coin_chart, equity_chart
from tests.helpers import bars


class ChartTests(unittest.TestCase):
    def test_chart_has_candles_volume_score_and_only_real_signals(self):
        """The financial chart contains three panels and markers only on flagged crossing dates."""
        frame = compute_indicators(bars())
        frame["cross"] = False
        frame.iloc[-2, frame.columns.get_loc("cross")] = True
        fig = coin_chart(frame.tail(20), .7)
        self.assertEqual([trace.type for trace in fig.data], ["candlestick","scatter","bar","scatter"])
        self.assertEqual(list(fig.data[1].x), [frame.index[-2]])
        self.assertEqual(fig.layout.shapes[0].y0, .7)
        self.assertEqual(fig.layout.xaxis.matches, "x3")

    def test_bollinger_overlay_uses_full_history_and_price_axis(self):
        """A zoomed date range keeps precomputed bands and plots all three on the price panel."""
        frame = compute_indicators(bars()).tail(5).assign(cross=False)
        fig = coin_chart(frame, .7, show_bollinger=True)
        overlays = {t.name: t for t in fig.data if t.name.startswith("Bollinger")}
        self.assertEqual(set(overlays), {"Bollinger Upper", "Bollinger Middle", "Bollinger Lower"})
        for label, multiplier in (("Upper", 2), ("Middle", 0), ("Lower", -2)):
            trace = overlays[f"Bollinger {label}"]
            self.assertEqual(trace.yaxis, "y")
            self.assertEqual(list(trace.x), list(frame.index))
            self.assertEqual(list(trace.y), list(frame.bb_mid + multiplier * frame.bb_std))

    def test_selected_equity_matches_loss_instead_of_another_experiment(self):
        """A -58.57% experiment plots its0.4143 equity, not another curve above2."""
        import pandas as pd
        index = pd.date_range("2020-01-01",periods=3)
        results = {
            "bb:long": dict(label="Bollinger bands",direction="long",equity=pd.Series([1.,.7,.41433980785834956],index=index)),
            "ema:long": dict(label="EMA trend",direction="long",equity=pd.Series([1.,2.,2.5],index=index)),
        }
        fig = equity_chart(results,"bb:long")
        self.assertEqual(len(fig.data),1)
        self.assertEqual(fig.data[0].name,"Bollinger bands / long")
        self.assertAlmostEqual(fig.data[0].y[-1],.41433980785834956)
        self.assertAlmostEqual(fig.data[0].customdata[-1],-58.566019214165046)
        compared = equity_chart(results,"bb:long",True)
        self.assertEqual(len(compared.data),2)
        self.assertTrue(all(trace.stackgroup is None for trace in compared.data))
