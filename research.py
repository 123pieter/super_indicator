"""Long-only, unlevered research. No fetching, selection, or live signal decisions.

Equity is marked at each candle close after estimated liquidation costs. Drawdown
therefore includes open trades but not intrabar extremes. Funding is not modelled.
"""
import numpy as np
import pandas as pd
import config
from timeframes import get_timeframe
from analyze import crosses

TRADE_COLUMNS = ["signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
                 "bars", "net_return", "reason"]


def _validate(d, threshold, fee, slippage, stop_atr, tp_atr, max_hold, timeframe):
    crosses(d, threshold)
    if not all(np.isfinite(x) and 0 <= x < 1 for x in (fee, slippage)):
        raise ValueError("Fee and slippage must be finite fractions in [0, 1)")
    if not np.isfinite(stop_atr) or stop_atr <= 0:
        raise ValueError("Stop ATR must be positive")
    if tp_atr is not None and (not np.isfinite(tp_atr) or tp_atr <= 0):
        raise ValueError("Take-profit ATR must be positive or None")
    if isinstance(max_hold, bool) or not isinstance(max_hold, int) or max_hold < 1:
        raise ValueError("Maximum hold must be a positive integer")
    if not isinstance(d.index, pd.DatetimeIndex) or not d.index.is_monotonic_increasing or not d.index.is_unique:
        raise ValueError("Research requires unique chronological candle timestamps")
    duration = get_timeframe(timeframe).duration
    if len(d) > 1 and not (d.index.to_series().diff().iloc[1:] == duration).all():
        raise ValueError("Research unavailable: missing candles or irregular intervals")
    values = d[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Research requires positive finite prices")
    if ((d.high < d[["open", "close", "low"]].max(axis=1)) |
            (d.low > d[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC candle range")


def _stats(trades, equity, exposure):
    """Keep full precision; initial capital is part of every drawdown calculation."""
    returns = trades.net_return
    wins, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    return dict(n_trades=len(trades),
                win_pct=float((returns > 0).mean()*100) if len(returns) else np.nan,
                avg_trade_pct=float(returns.mean()*100),
                profit_factor=float(wins/losses) if losses else (np.inf if wins else np.nan),
                total_return_pct=float((equity.iloc[-1]-1)*100) if len(equity) else np.nan,
                max_drawdown_pct=float((equity/equity.cummax()-1).min()*100) if len(equity) else np.nan,
                exposure_pct=exposure*100)


def backtest(d, threshold=config.BUY_THRESHOLD, *, fee=0.001, slippage=0.0005,
             stop_atr=config.STOP_ATR, tp_atr=config.TP_ATR, max_hold=config.MAX_HOLD,
             exit_threshold=None, timeframe=config.DEFAULT_TIMEFRAME, strategy="super", direction="long", atr_exits=True, evaluation_start=None):
    """Trade the same evaluation span for Super, buy_hold, and sma strategies.

    The first eligible signal is the close of the first ready candle; execution
    starts at the next open. SMA holds when the prior SMA50 exceeds SMA200.
    Stops/targets apply only to Super. At a candle open, gap stops, scheduled
    score exits, then gap targets precede intrabar tests. If both intrabar levels
    are touched, stop wins. Signals on an exit candle can enter next candle.
    Final positions are liquidated at the final close, including both costs.
    direction="short" mirrors stop/target prices and P&L at 1x collateralised
    exposure. Funding/borrow and venue liquidation rules are not modelled;
    simulated capital exhaustion raises an error rather than clipping losses.
    """
    if direction not in {"long", "short"}:
        raise ValueError("Direction must be long or short")
    side = 1 if direction == "long" else -1
    if strategy not in {"super", "buy_hold", "sma"}:
        raise ValueError("Unknown strategy")
    _validate(d, threshold, fee, slippage, stop_atr, tp_atr, max_hold, timeframe)
    exit_threshold = threshold-config.EXIT_SCORE_DROP if exit_threshold is None else exit_threshold
    if not np.isfinite(exit_threshold) or not -1 <= exit_threshold <= 1:
        raise ValueError("Exit threshold must be finite and between -1 and 1")
    ready = np.flatnonzero(d.ready.to_numpy(bool))
    first = int(ready[0])+1 if len(ready) else len(d)
    if evaluation_start is not None:
        first = max(first, int(d.index.searchsorted(pd.Timestamp(evaluation_start))))
    trades, marks = [], []
    cash, position, exposed = 1.0, None, 0
    signals = crosses(d, threshold).to_numpy(bool)
    entry_multiplier = (1+side*slippage)*(1+fee)

    def liquidation_value(position, raw_price):
        # 1x cash-backed exposure. Short proceeds are collateral, not reinvested.
        fill = raw_price*(1-side*slippage)
        wealth = (position["capital"] + side*position["units"]*(fill-position["fill"])
                  - fee*position["units"]*(position["fill"]+fill))
        if wealth <= 0:
            raise ValueError("Assumed capital exhausted: short gap risk requires a liquidation model")
        return wealth

    def close_position(i, price, reason):
        nonlocal position, cash
        cash = liquidation_value(position, price)
        trades.append(dict(signal_time=d.index[position["entry"]-1],
                           entry_time=d.index[position["entry"]], exit_time=d.index[i],
                           entry_price=position["price"], exit_price=price,
                           bars=i-position["entry"]+1,
                           net_return=cash/position["capital"]-1, reason=reason))
        position = None

    for i in range(first, len(d)):
        row, prev = d.iloc[i], d.iloc[i-1]
        exited_open = False
        if position is not None:
            if strategy == "super" and atr_exits and side*(row.open-position["stop"]) <= 0:
                close_position(i, row.open, "stop_gap")
                exited_open = True
            elif ((strategy == "super" and prev.super_score < exit_threshold) or
                  (strategy == "sma" and not (prev.sma50 > prev.sma200))):
                close_position(i, row.open, "score_exit" if strategy == "super" else "sma_exit")
                exited_open = True
            elif strategy == "super" and atr_exits and side*(row.open-position["target"]) >= 0:
                close_position(i, row.open, "target_gap")
                exited_open = True
        enter = (signals[i-1] and np.isfinite(prev.atr) and prev.atr > 0) if strategy == "super" else (
            strategy == "buy_hold" and i == first or strategy == "sma" and prev.sma50 > prev.sma200)
        if position is None and not exited_open and enter:
            position = dict(entry=i, price=float(row.open), capital=cash,
                            units=cash/(row.open*entry_multiplier), fill=row.open*(1+side*slippage),
                            stop=row.open-side*stop_atr*prev.atr,
                            target=row.open+side*tp_atr*prev.atr if tp_atr is not None else side*np.inf)
        if position is not None:
            exposed += 1
            if strategy == "super" and atr_exits and side*((row.low if side == 1 else row.high)-position["stop"]) <= 0:
                close_position(i, position["stop"], "stop")
            elif strategy == "super" and atr_exits and side*((row.high if side == 1 else row.low)-position["target"]) >= 0:
                close_position(i, position["target"], "take_profit")
            elif strategy == "super" and i-position["entry"]+1 >= max_hold:
                close_position(i, row.close, "max_hold")
            elif i == len(d)-1:
                close_position(i, row.close, "end_of_data")
        marks.append(cash if position is None else liquidation_value(position, row.close))
    # Include starting capital before any entry fees or trading losses.
    index = d.index[first-1:] if first < len(d) else d.index[:0]
    equity = pd.Series([1.0]+marks if len(index) else [], index=index, dtype=float, name="equity")
    tdf = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    stats = _stats(tdf, equity, exposed/len(marks) if marks else 0)
    return dict(trades=tdf, equity=equity, stats=stats,
                start=d.index[first] if first < len(d) else None,
                end=d.index[-1] if first < len(d) else None)


def compare_strategies(d, provider=config.ACTIVE_PROVIDER, threshold=config.BUY_THRESHOLD,
                       timeframe=config.DEFAULT_TIMEFRAME):
    """Identical dates, full-capital long exposure, and provider-specific assumed costs."""
    costs = config.BACKTEST_COSTS[provider]
    return {name: backtest(d, threshold, strategy=name, timeframe=timeframe, **costs)
            for name in ("super", "buy_hold", "sma")}


def evaluate_history(d, provider=config.ACTIVE_PROVIDER, threshold=config.BUY_THRESHOLD,
                     timeframe=config.DEFAULT_TIMEFRAME):
    """Descriptive in-sample results only; never select coins using these returns."""
    try:
        results = compare_strategies(d, provider, threshold, timeframe)
    except ValueError as exc:
        return dict(research_error=str(exc), sufficient_history=False,
                    n_trades=0, total_return_pct=np.nan, max_drawdown_pct=np.nan,
                    buy_hold_pct=np.nan, sma_return_pct=np.nan)
    stats = results["super"]["stats"]
    return dict(**stats, buy_hold_pct=results["buy_hold"]["stats"]["total_return_pct"],
                sma_return_pct=results["sma"]["stats"]["total_return_pct"],
                research_error="", sufficient_history=stats["n_trades"] >= config.MIN_RESEARCH_TRADES)
