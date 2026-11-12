"""Stage 4 research signals. Inputs are completed candles in time order."""
import math

import pandas as pd


def _window(window):
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("Window must be an integer of at least 2 candles")


def sma_trend(close, window=100):
    """Long above SMA, short below it, flat at equality or during warmup."""
    _window(window)
    middle = close.rolling(window).mean()
    return (close.gt(middle).astype("int8") - close.lt(middle).astype("int8")).rename("sma_trend")


def bollinger_reentry(close, window=20, deviations=2.0):
    """Enter on an outer-band re-entry; hold until the close reaches the middle.

    Long: prior close below its own lower band, then a close strictly between
    the current lower and middle bands. Short mirrors this. A close at or
    beyond the middle exits; no reversal is emitted on the exit candle.
    The engine handles execution and any separately configured stop/target.
    """
    _window(window)
    if not math.isfinite(deviations) or deviations <= 0:
        raise ValueError("Band deviations must be positive and finite")
    middle = close.rolling(window).mean()
    # Sample standard deviation matches the existing app's chart convention.
    spread = deviations * close.rolling(window).std()
    lower, upper = middle - spread, middle + spread
    long_entry = close.lt(lower).shift(fill_value=False) & close.gt(lower) & close.lt(middle)
    short_entry = close.gt(upper).shift(fill_value=False) & close.lt(upper) & close.gt(middle)
    state, states = 0, []
    for buy, sell, long_exit, short_exit in zip(
            long_entry, short_entry, close.ge(middle), close.le(middle)):
        if state == 1:
            state = 0 if long_exit else 1
        elif state == -1:
            state = 0 if short_exit else -1
        elif buy:
            state = 1
        elif sell:
            state = -1
        states.append(state)
    return pd.Series(states, index=close.index, dtype="int8", name="bollinger_reentry")

def macd_direction(close, fast=12, slow=26, signal=9):
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd - signal_line
    return (
        histogram.gt(0).astype("int8")
        - histogram.lt(0).astype("int8")
    ).rename("macd_direction")



def donchian_breakout(candles, entry_window=55, exit_window=20):
    """Close-confirmed channel breakouts, with exits before any new entry."""
    _window(entry_window)
    _window(exit_window)
    close = candles["close"]
    # Shift the channels so the signal candle never sets its own threshold.
    high, low = candles["high"].shift(), candles["low"].shift()
    upper = close.gt(high.rolling(entry_window).max())
    lower = close.lt(low.rolling(entry_window).min())
    exit_long = close.lt(low.rolling(exit_window).min())
    exit_short = close.gt(high.rolling(exit_window).max())
    warmup = max(entry_window, exit_window)
    state, states = 0, [0] * min(warmup, len(close))
    for buy, sell, long_exit, short_exit in zip(
            upper.iloc[warmup:], lower.iloc[warmup:],
            exit_long.iloc[warmup:], exit_short.iloc[warmup:]):
        if state == 1:
            state = 0 if long_exit else 1
        elif state == -1:
            state = 0 if short_exit else -1
        elif buy:
            state = 1
        elif sell:
            state = -1
        states.append(state)
    return pd.Series(states, index=close.index, dtype="int8", name="donchian_breakout")

def _wilder(values, period):
    """Seed at the first full arithmetic mean, then apply Wilder's recursion."""
    seed = values.rolling(period).mean()
    start = seed.first_valid_index()
    if start is None:
        return seed
    seeded = values.copy()
    pos = values.index.get_loc(start)
    seeded.iloc[:pos] = float("nan")
    seeded.iloc[pos] = seed.iloc[pos]
    return seeded.ewm(alpha=1/period, adjust=False).mean()


def roc_direction(close, window=60):
    """Vote by the sign of the return over the preceding window candles."""
    _window(window)
    change = close - close.shift(window)
    return (change.gt(0).astype("int8") - change.lt(0).astype("int8")).rename("roc_direction")


def rsi_value(close, period):
    """Wilder RSI, with a neutral value for a flat seeded series."""
    _window(period)
    delta = close.diff()
    gains = _wilder(delta.clip(lower=0), period)
    losses = _wilder(-delta.clip(upper=0), period)
    total = gains + losses
    return (100*gains/total.where(total.ne(0))).mask(total.eq(0),50.)


def rsi_reentry(close, period=2, lower=10., upper=90.):
    """Enter on re-entry from an extreme; exit at RSI 50, without reversal."""
    rsi = rsi_value(close, period)
    previous = rsi.shift()
    buys = previous.lt(lower) & rsi.gt(lower) & rsi.lt(50)
    sells = previous.gt(upper) & rsi.lt(upper) & rsi.gt(50)
    state, states = 0, []
    for buy, sell, value in zip(buys, sells, rsi):
        if state == 1:
            state = 0 if value >= 50 else 1
        elif state == -1:
            state = 0 if value <= 50 else -1
        elif buy:
            state = 1
        elif sell:
            state = -1
        states.append(state)
    return pd.Series(states, index=close.index, dtype="int8", name="rsi_reentry")


def dmi_values(candles, period):
    """Return Wilder ADX and the two directional lines."""
    _window(period)
    high, low, close = candles["high"], candles["low"], candles["close"]
    up, down = high.diff(), -low.diff()
    plus = up.where(up.gt(down) & up.gt(0),0.).where(up.notna())
    minus = down.where(down.gt(up) & down.gt(0),0.).where(down.notna())
    previous = close.shift()
    ranges = pd.concat([high-low,(high-previous).abs(),(low-previous).abs()],axis=1).max(axis=1)
    # DM starts at the first price change; align the true range seed to it.
    atr = _wilder(ranges.where(previous.notna()), period)
    plus_di = (100*_wilder(plus,period)/atr.where(atr.ne(0))).mask(atr.eq(0),0.)
    minus_di = (100*_wilder(minus,period)/atr.where(atr.ne(0))).mask(atr.eq(0),0.)
    total = plus_di + minus_di
    dx = (100*(plus_di-minus_di).abs()/total.where(total.ne(0))).mask(total.eq(0),0.)
    return _wilder(dx,period), plus_di, minus_di


def adx_dmi(candles, period=14, adx_threshold=20.):
    """Follow the stronger directional line only above the ADX threshold."""
    adx, plus_di, minus_di = dmi_values(candles, period)
    strong = adx.gt(adx_threshold)
    return ((strong & plus_di.gt(minus_di)).astype("int8")
            - (strong & minus_di.gt(plus_di)).astype("int8")).rename("adx_dmi")

def ema_value(close, window):
    """EMA with no signal value before the full window is observed."""
    _window(window)
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def ema_trend(close, window=200):
    """Long above EMA, short below EMA, flat during warmup or at equality."""
    average = ema_value(close,window)
    return (close.gt(average).astype("int8") - close.lt(average).astype("int8")).rename("ema_trend")


def _rsi_trend_states(close, oscillator, reference, sma_window, ema_window):
    """Moving averages gate entry only; the oscillator comparison controls exit."""
    _window(sma_window)
    aligned = close.gt(close.rolling(sma_window).mean()) & close.gt(ema_value(close,ema_window))
    enter = aligned & oscillator.gt(reference)
    exit_ = oscillator.lt(reference)
    state, states = 0, []
    for buy, sell in zip(enter,exit_):
        if state:
            state = 0 if sell else 1
        elif buy:
            state = 1
        states.append(state)
    return pd.Series(states,index=close.index,dtype="int8")


def rsi_adx_long(candles, sma_window=50, ema_window=7, rsi_period=2, adx_period=2):
    """Reported long-only rule: above both averages and RSI > ADX; exit on RSI < ADX."""
    close = candles.close
    adx, _, _ = dmi_values(candles,adx_period)
    return _rsi_trend_states(close,rsi_value(close,rsi_period),adx,sma_window,ema_window).rename("rsi_adx_long")


def rsi_midline_long(close, sma_window=50, ema_window=7, rsi_period=2):
    """Control: identical trend alignment, replacing ADX with an RSI reference of 50."""
    return _rsi_trend_states(close,rsi_value(close,rsi_period),50.,sma_window,ema_window).rename("rsi_midline_long")

def _direction(value):
    """Convert signed values to votes and keep unavailable values flat."""
    return value.gt(0).astype("int8") - value.lt(0).astype("int8")


def _atr(candles, period):
    return _wilder(_true_range(candles), period)


def _true_range(candles):
    previous = candles.close.shift()
    return pd.concat([candles.high - candles.low, (candles.high - previous).abs(),
                      (candles.low - previous).abs()], axis=1).max(axis=1)


def bollinger_width_percentile(candles, period):
    """Rank prior bandwidth against the preceding five windows, before a breakout."""
    close = candles.close
    middle = close.rolling(period).mean()
    width = 4 * close.rolling(period).std() / middle.where(middle.gt(0))
    return 100 * width.shift().rolling(5 * period).rank(pct=True)


def choppiness_index(candles, period):
    span = candles.high.rolling(period).max() - candles.low.rolling(period).min()
    ratio = _true_range(candles).rolling(period).sum() / span.where(span.gt(0))
    return 100 * ratio.clip(lower=1).map(math.log10) / math.log10(period)


def _reentry(oscillator, lower, upper, middle=50):
    """Trade a return from an extreme and exit on reaching the midpoint."""
    previous = oscillator.shift()
    return _hold_states(previous.le(lower) & oscillator.gt(lower) & oscillator.lt(middle),
                        previous.ge(upper) & oscillator.lt(upper) & oscillator.gt(middle),
                        oscillator.ge(middle) | oscillator.isna(),
                        oscillator.le(middle) | oscillator.isna())


def _hold_states(buy, sell, exit_long, exit_short):
    state, states = 0, []
    for long_entry, short_entry, long_exit, short_exit in zip(buy, sell, exit_long, exit_short):
        if state == 1:
            state = 0 if long_exit else 1
        elif state == -1:
            state = 0 if short_exit else -1
        elif long_entry:
            state = 1
        elif short_entry:
            state = -1
        states.append(state)
    return pd.Series(states, index=buy.index, dtype="int8")


def connors_reentry(close, rsi_period=3, streak_period=2, rank_period=100, lower=10., upper=90.):
    """Average price RSI, streak RSI and the return's strict prior-window percentile."""
    direction = _direction(close.diff())
    groups = direction.ne(direction.shift()).cumsum()
    streak = (direction.groupby(groups).cumcount() + 1) * direction
    returns = close.pct_change(fill_method=None)
    rank = returns.rolling(rank_period + 1).apply(
        lambda window: 100 * (window[:-1] < window[-1]).mean(), raw=True)
    score = (rsi_value(close, rsi_period) + rsi_value(streak, streak_period) + rank) / 3
    return _reentry(score, lower, upper)


def supertrend_direction(candles, period=10, multiplier=3.):
    """Direction from recursive ATR bands; reversals are confirmed at the close."""
    middle = (candles.high + candles.low) / 2
    offset = multiplier * _atr(candles, period)
    upper, lower, previous, state = math.nan, math.nan, math.nan, 0
    states = []
    for close, raw_upper, raw_lower in zip(candles.close, middle + offset, middle - offset):
        if pd.notna(raw_upper):
            if state == 0:
                upper, lower, state = raw_upper, raw_lower, -1
            else:
                if raw_upper < upper or previous > upper:
                    upper = raw_upper
                if raw_lower > lower or previous < lower:
                    lower = raw_lower
                if state == -1 and close > upper:
                    state = 1
                elif state == 1 and close < lower:
                    state = -1
        states.append(state)
        previous = close
    return pd.Series(states, index=candles.index, dtype="int8")


def vwap_band_reentry(candles, deviations=2.):
    """Use volume-weighted HLC3 and variance, anchored at UTC midnight."""
    index = pd.to_datetime(candles.index, utc=True)
    session = index.normalize()
    session_start = pd.Series(index == session, index=candles.index)
    elapsed = pd.Series(index, index=candles.index).diff()
    session_end = (pd.Series(index, index=candles.index) + elapsed).dt.normalize().ne(session) & elapsed.notna()
    anchored = session_start.groupby(session).cummax()
    typical = (candles.high + candles.low + candles.close) / 3
    volume = candles.volume.groupby(session).cumsum()
    volume = volume.where(volume.gt(0) & anchored)
    vwap = (typical * candles.volume).groupby(session).cumsum() / volume
    second_moment = (typical.pow(2) * candles.volume).groupby(session).cumsum() / volume
    spread = (second_moment - vwap.pow(2)).clip(lower=0).pow(.5)
    close = candles.close
    lower, upper = vwap - deviations * spread, vwap + deviations * spread
    # Flatten at the next open (UTC midnight), and never trade across a band reset.
    reset = session_start | session_end
    buy = ~reset & close.lt(lower).shift(fill_value=False) & close.gt(lower) & close.lt(vwap)
    sell = ~reset & close.gt(upper).shift(fill_value=False) & close.lt(upper) & close.gt(vwap)
    return _hold_states(buy, sell, reset | close.ge(vwap), reset | close.le(vwap))


def keltner_breakout(candles, period=20, multiplier=1.5):
    """Enter beyond the EMA/ATR channel and exit at the center EMA."""
    middle = ema_value(candles.close, period)
    offset = multiplier * _atr(candles, period)
    return _hold_states(candles.close.gt(middle + offset), candles.close.lt(middle - offset),
                        candles.close.le(middle), candles.close.ge(middle))


def chandelier_breakout(candles, entry_window=40, period=22, multiplier=3.):
    """Donchian entries with a ratcheting Chandelier exit confirmed on close."""
    close = candles.close
    distance = multiplier * _atr(candles, period)
    long_level = candles.high.rolling(period).max() - distance
    short_level = candles.low.rolling(period).min() + distance
    buy = close.gt(candles.high.shift().rolling(entry_window).max())
    sell = close.lt(candles.low.shift().rolling(entry_window).min())
    state, stop, states = 0, math.nan, []
    for price, enter_long, enter_short, lower, upper in zip(close, buy, sell, long_level, short_level):
        if state == 1:
            state = 0 if price <= stop else 1
            stop = max(stop, lower)
        elif state == -1:
            state = 0 if price >= stop else -1
            stop = min(stop, upper)
        elif enter_long and pd.notna(lower) and price > lower:
            state, stop = 1, lower
        elif enter_short and pd.notna(upper) and price < upper:
            state, stop = -1, upper
        states.append(state)
    return pd.Series(states, index=close.index, dtype="int8")


def mfi_reentry(candles, period=14, lower=20., upper=80.):
    """Separate positive and negative money flow using changes in typical price."""
    typical = (candles.high + candles.low + candles.close) / 3
    flow = typical * candles.volume
    change = typical.diff()
    positive = flow.where(change.gt(0), 0).where(change.notna()).rolling(period).sum()
    negative = flow.where(change.lt(0), 0).where(change.notna()).rolling(period).sum()
    total = positive + negative
    return _reentry(100 * positive / total.where(total.gt(0)), lower, upper)


def cmf_direction(candles, period=20):
    span = candles.high - candles.low
    location = ((2 * candles.close - candles.high - candles.low) / span.where(span.gt(0))).fillna(0)
    volume = candles.volume.rolling(period).sum()
    return _direction((location * candles.volume).rolling(period).sum() / volume.where(volume.gt(0)))


def ichimoku_direction(candles, conversion=9, base=26, span_b=52, displacement=26):
    """Compare today's close with today's cloud, calculated displacement bars ago."""
    def midpoint(period):
        return (candles.high.rolling(period).max() + candles.low.rolling(period).min()) / 2

    span_a = ((midpoint(conversion) + midpoint(base)) / 2).shift(displacement)
    span_b_line = midpoint(span_b).shift(displacement)
    above = candles.close.gt(span_a) & candles.close.gt(span_b_line)
    below = candles.close.lt(span_a) & candles.close.lt(span_b_line)
    return above.astype("int8") - below.astype("int8")


def hma_direction(close, window=55):
    """Use Hull MA slope for direction."""
    def weighted(values, period):
        weights = pd.Series(range(1, period + 1)).to_numpy(dtype=float)
        weights /= weights.sum()
        return values.rolling(period).apply(lambda x: x @ weights, raw=True)

    hull = weighted(2 * weighted(close, window // 2) - weighted(close, window), math.isqrt(window))
    return _direction(hull.diff())


def obv_direction(candles, period=20):
    """Compare OBV to its own trailing mean; the starting level cancels out."""
    obv = (_direction(candles.close.diff()) * candles.volume).cumsum()
    return _direction(obv - obv.rolling(period).mean())


def stoch_rsi_reentry(close, period=14, stoch_period=14, smooth=3, lower=20., upper=80.):
    rsi = rsi_value(close, period)
    bottom = rsi.rolling(stoch_period).min()
    span = rsi.rolling(stoch_period).max() - bottom
    score = (100 * (rsi - bottom) / span.where(span.gt(0))).rolling(smooth).mean()
    return _reentry(score, lower, upper)


def williams_reentry(candles, period=14, lower=-80., upper=-20.):
    high = candles.high.rolling(period).max()
    low = candles.low.rolling(period).min()
    span = high - low
    score = -100 * (high - candles.close) / span.where(span.gt(0))
    return _reentry(score, lower, upper, middle=-50)

