"""Standalone bullish/bearish candidate experiments using the shared execution engine."""
import numpy as np
import pandas as pd
import config
from analyze import compute_indicators
from research import backtest
from timeframes import completed_bars, get_timeframe

# Explicit bearish rules avoid treating every failed bullish condition as a short.
CANDIDATES = {
    "sma": ("Moving averages", "Close > SMA50 > SMA200", "Close < SMA50 < SMA200"),
    "ema": ("EMA trend", "EMA12 > EMA26", "EMA12 < EMA26"),
    "macd": ("MACD", "MACD histogram > 0", "MACD histogram < 0"),
    "rsi": ("RSI", "RSI14 > 50", "RSI14 < 50"),
    "bb": ("Bollinger bands", "Middle < close < upper", "Lower < close < middle"),
    "stoch": ("Stochastic", "%K > %D and 20 < %K < 80", "%K < %D and 20 < %K < 80"),
    "adx": ("ADX trend", "ADX14 > 20 and +DI > -DI", "ADX14 > 20 and -DI > +DI"),
    "obv": ("Volume trend", "OBV > its SMA20", "OBV < its SMA20"),
    "cci": ("CCI", "0 < CCI20 < 100", "-100 < CCI20 < 0"),
}


def candidate_condition(d, candidate, direction):
    """Return the selected directional condition; undefined values never qualify."""
    if candidate not in CANDIDATES or direction not in {"long", "short"}:
        raise ValueError("Unknown indicator candidate or direction")
    if direction == "long":
        return d[f"{candidate}_signal"].eq(1)
    rules = {
        "sma": lambda: (d.close < d.sma50) & (d.sma50 < d.sma200),
        "ema": lambda: d.ema12 < d.ema26,
        "macd": lambda: d.macd_hist < 0,
        "rsi": lambda: d.rsi < 50,
        "bb": lambda: (d.close < d.bb_mid) & (d.close > d.bb_mid-2*d.bb_std),
        "stoch": lambda: (d.stoch_k < d.stoch_d) & (d.stoch_k > 20) & (d.stoch_k < 80),
        "adx": lambda: (d.adx > 20) & (d.minus_di > d.plus_di),
        "obv": lambda: d.obv < d.obv_sma,
        "cci": lambda: (d.cci > -100) & (d.cci < 0),
    }
    return rules[candidate]().fillna(False)


def compare_candidates(raw, candidates, directions=("long", "short"), *,
                       provider=config.ACTIVE_PROVIDER, timeframe=config.DEFAULT_TIMEFRAME,
                       start=None, end=None, fee=None, slippage=None,
                       max_hold=config.MAX_HOLD, exit_on_condition=True,
                       use_regime=False, atr_exits=False,
                       stop_atr=config.STOP_ATR, tp_atr=config.TP_ATR):
    """Independent experiments; never combine candidate trades or pick a winner.

    Calculate causal indicators with full warmup history before applying dates.
    Dates are UTC candle-start dates, inclusive. Every experiment uses the same
    200-bar warmup and evaluation span. At the first eligible close, the entry
    condition must switch from false to true; persistent conditions do not
    re-enter after a timed exit. If enabled, condition failure exits next open.
    Baselines are long buy-and-hold and directional SMA50/200 under identical
    costs and dates, without the candidate's optional stop/target/hold rules.
    """
    candidates, directions = list(candidates), list(directions)
    if not candidates or len(set(candidates)) != len(candidates) or any(c not in CANDIDATES for c in candidates):
        raise ValueError("Select one or more distinct indicator candidates")
    if not directions or len(set(directions)) != len(directions) or any(s not in {"long", "short"} for s in directions):
        raise ValueError("Select long, short, or both directions")
    costs = config.BACKTEST_COSTS[provider]
    costs = dict(fee=costs["fee"] if fee is None else fee,
                 slippage=costs["slippage"] if slippage is None else slippage)
    d = compute_indicators(completed_bars(raw, timeframe))
    if len(d) < 201:
        raise ValueError("At least 201 completed candles are needed for a common evaluation span")
    eligible = d.index[d.ready]
    first = eligible[0]+get_timeframe(timeframe).duration
    if start is not None:
        first = max(first, pd.Timestamp(start))
    last = d.index[-1] if end is None else min(d.index[-1], pd.Timestamp(end)+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1))
    eligible = d.index[(d.index >= first) & (d.index <= last)]
    if not len(eligible):
        raise ValueError("No completed evaluation candles in the selected date range after warmup")
    first_i = d.index.get_loc(eligible[0])
    # Two prior candles preserve a crossing at the close just before the first open.
    window = d.iloc[max(0, first_i-2):d.index.get_loc(eligible[-1])+1].copy()
    rows, results = [], {}
    common = dict(timeframe=timeframe, max_hold=max_hold,
                  stop_atr=stop_atr, tp_atr=tp_atr, evaluation_start=eligible[0], **costs)

    def record(key, label, direction, frame, **options):
        try:
            result = backtest(frame, direction=direction, **common, **options)
        except ValueError as exc:
            if "capital exhausted" not in str(exc):
                raise
            rows.append(dict(Experiment=label, Direction=direction, Error=str(exc),
                             n_trades=0, total_return_pct=np.nan, max_drawdown_pct=np.nan,
                             avg_trade_pct=np.nan, win_pct=np.nan, profit_factor=np.nan,
                             exposure_pct=np.nan))
            return
        result["label"], result["direction"] = label, direction
        results[key] = result
        rows.append(dict(Experiment=label, Direction=direction, Error="", **result["stats"]))

    for candidate in candidates:
        for direction in directions:
            frame = window.copy()
            condition = candidate_condition(frame, candidate, direction)
            frame["super_score"] = condition.astype(float).where(frame.ready)
            frame["regime_ok"] = frame.regime_ok if use_regime else True
            record(f"{candidate}:{direction}", CANDIDATES[candidate][0], direction, frame,
                   threshold=.5, exit_threshold=.5 if exit_on_condition else -1,
                   atr_exits=atr_exits)
    record("buy_hold:long", "Buy and hold (baseline)", "long", window, strategy="buy_hold")
    for direction in directions:
        # The existing SMA baseline tests SMA50 > SMA200; invert for the short side.
        frame = window.copy()
        if direction == "short":
            frame["sma50"], frame["sma200"] = window.sma200, window.sma50
        record(f"baseline_sma:{direction}", "SMA50/200 (baseline)", direction, frame, strategy="sma")
    return dict(summary=pd.DataFrame(rows), results=results,
                start=eligible[0], end=window.index[-1], costs=costs,
                provider=provider, timeframe=timeframe)


def aggregate_coins(per_coin, coin_results, requested_coins):
    """Equal-coin summaries and pooled trade statistics; never sum coin returns.

    Means use valid coins only, including zero-trade cash strategies. Coverage is
    explicit. Basket return/drawdown require every requested coin to be valid,
    sharing the same equity timestamps; failed experiments are never hidden by
    treating their losses as cash or dropping them from a portfolio silently.
    """
    rows = []
    for (label, direction), group in per_coin.groupby(["Experiment", "Direction"], sort=False):
        valid = group[group.Error.eq("")]
        experiments = [item for result in coin_results.values() for item in result["results"].values()
                       if item["label"] == label and item["direction"] == direction]
        returns = pd.concat([item["trades"].net_return for item in experiments], ignore_index=True) if experiments else pd.Series(dtype=float)
        wins, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
        basket_return = basket_dd = np.nan
        if len(valid) == requested_coins and len(experiments) == requested_coins:
            curves = [item["equity"] for item in experiments]
            if all(curves[0].index.equals(curve.index) for curve in curves):
                equity = pd.concat(curves, axis=1).mean(axis=1)
                basket_return = (equity.iloc[-1]-1)*100
                basket_dd = (equity/equity.cummax()-1).min()*100
        rows.append(dict(Experiment=label, Direction=direction,
                         requested_coins=requested_coins, valid_coins=len(valid),
                         failed_coins=requested_coins-len(valid),
                         coins_with_trades=int(valid.n_trades.gt(0).sum()),
                         profitable_coins=int(valid.total_return_pct.gt(0).sum()),
                         mean_coin_return_pct=valid.total_return_pct.mean(),
                         median_coin_return_pct=valid.total_return_pct.median(),
                         worst_coin_return_pct=valid.total_return_pct.min(),
                         best_coin_return_pct=valid.total_return_pct.max(),
                         mean_coin_drawdown_pct=valid.max_drawdown_pct.mean(),
                         basket_return_pct=basket_return, basket_drawdown_pct=basket_dd,
                         total_trades=len(returns),
                         pooled_win_pct=(returns > 0).mean()*100 if len(returns) else np.nan,
                         pooled_avg_trade_pct=returns.mean()*100,
                         pooled_profit_factor=wins/losses if losses else (np.inf if wins else np.nan)))
    return pd.DataFrame(rows)


def compare_coin_universe(coins, loader, candidates, directions=("long", "short"),
                          on_progress=None, **settings):
    """Run the full selected coin x candidate x direction grid, offline.

    A common post-warmup date intersection makes cross-coin comparisons fair.
    Bad/short/missing histories remain explicit failed rows; other coins continue.
    The callback receives (completed, total, coin) after each attempted simulation.
    """
    coins, candidates, directions = list(coins), list(candidates), list(directions)
    if not coins or len(set(coins)) != len(coins):
        raise ValueError("Select one or more distinct backtest coins")
    if not candidates or len(set(candidates)) != len(candidates) or any(c not in CANDIDATES for c in candidates):
        raise ValueError("Select one or more distinct indicator candidates")
    if not directions or len(set(directions)) != len(directions) or any(d not in {"long","short"} for d in directions):
        raise ValueError("Select long, short, or both directions")
    timeframe = settings.get("timeframe", config.DEFAULT_TIMEFRAME)
    histories, errors = {}, {}
    for coin in coins:
        try:
            raw = loader(coin)
            if raw is None:
                raise ValueError("No stored candles")
            raw = completed_bars(raw, timeframe).sort_index(kind="stable")
            raw = raw.loc[~raw.index.duplicated(keep="last")]
            if len(raw) < 201:
                raise ValueError("At least 201 completed candles are required")
            histories[coin] = raw
        except Exception as exc:
            errors[coin] = str(exc)
    start = max((raw.index[200] for raw in histories.values()), default=None)
    end = min((raw.index[-1] for raw in histories.values()), default=None)
    if start is not None and settings.get("start") is not None:
        start = max(start,pd.Timestamp(settings["start"]))
    if end is not None and settings.get("end") is not None:
        end = min(end,pd.Timestamp(settings["end"])+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1))
    if start is not None and start > end:
        raise ValueError("Selected coins have no common evaluation dates after warmup")
    frames, results, coverage = [], {}, []
    planned = [(CANDIDATES[name][0],side) for name in candidates for side in directions]
    planned += [("Buy and hold (baseline)","long")]+[("SMA50/200 (baseline)",side) for side in directions]
    for i, coin in enumerate(coins):
        if coin not in errors:
            try:
                options = dict(settings, start=start)
                result = compare_candidates(histories[coin].loc[:end],candidates,directions,**options)
                results[coin] = result
                frame = result["summary"].copy()
            except Exception as exc:
                errors[coin] = str(exc)
        if coin in errors:
            frame = pd.DataFrame([dict(Experiment=label,Direction=side,Error=errors[coin],
                                      n_trades=0,total_return_pct=np.nan,max_drawdown_pct=np.nan,
                                      avg_trade_pct=np.nan,win_pct=np.nan,profit_factor=np.nan,exposure_pct=np.nan)
                                  for label,side in planned])
        frame.insert(0,"Coin",coin)
        frame["Evaluation start"] = results[coin]["start"] if coin in results else pd.NaT
        frame["Evaluation end"] = results[coin]["end"] if coin in results else pd.NaT
        frames.append(frame)
        coverage.append(dict(Coin=coin,Error=errors.get(coin,""),
                             failed_experiments=int(frame.Error.ne("").sum())))
        if on_progress:
            on_progress(i+1,len(coins),coin)
    per_coin = pd.concat(frames, ignore_index=True)
    return dict(per_coin=per_coin, aggregate=aggregate_coins(per_coin,results,len(coins)),
                coins=results, coverage=pd.DataFrame(coverage), start=start, end=end)
