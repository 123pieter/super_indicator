"""Stage 4 research signals. Inputs are completed candles in time order."""
import math

import pandas as pd
import talib


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
    close = close.astype(float)
    if fast == 1 or signal == 1:
        # Period-one EMA is the input itself. MACD's native fast EMA requires 2+.
        line = ema_value(close, fast) - ema_value(close, slow)
        histogram = line - ema_value(line, signal)
    else:
        _, _, histogram = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
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
    upper = close.gt(talib.MAX(high.astype(float), entry_window))
    lower = close.lt(talib.MIN(low.astype(float), entry_window))
    exit_long = close.lt(talib.MIN(low.astype(float), exit_window))
    exit_short = close.gt(talib.MAX(high.astype(float), exit_window))
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

def roc_direction(close, window=60):
    """Vote by the sign of the return over the preceding window candles."""
    _window(window)
    change = close - close.shift(window)
    return (change.gt(0).astype("int8") - change.lt(0).astype("int8")).rename("roc_direction")


def rsi_value(close, period):
    """Wilder RSI, with a neutral value for a flat seeded series."""
    _window(period)
    close = close.astype(float)
    # Power-of-two rescaling keeps small token prices above TA-Lib's zero tolerance.
    first = close.first_valid_index()
    anchor = abs(close.loc[first]) if first is not None else 0.
    if 0 < anchor < 1.:
        _, exponent = math.frexp(anchor)
        close = close * math.ldexp(1., min(20 - exponent, 1023))
    value = talib.RSI(close, timeperiod=period)
    # TA-Lib returns zero before any price movement, including on flat streaks.
    moved = close.diff().abs().fillna(0).ne(0).cummax()
    return value.mask(value.notna() & ~moved, 50.)


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
    """Return TA-Lib ADX and the two directional lines."""
    _window(period)
    high, low, close = (candles[name].astype(float) for name in ("high", "low", "close"))
    return (talib.ADX(high, low, close, timeperiod=period),
            talib.PLUS_DI(high, low, close, timeperiod=period),
            talib.MINUS_DI(high, low, close, timeperiod=period))


def adx_dmi(candles, period=14, adx_threshold=20.):
    """Follow the stronger directional line only above the ADX threshold."""
    adx, plus_di, minus_di = dmi_values(candles, period)
    strong = adx.gt(adx_threshold)
    return ((strong & plus_di.gt(minus_di)).astype("int8")
            - (strong & minus_di.gt(plus_di)).astype("int8")).rename("adx_dmi")

def ema_value(close, window):
    """EMA with no signal value before the full window is observed."""
    close = close.astype(float)
    return close if window == 1 else talib.EMA(close, timeperiod=window)


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
    high, low, close = (candles[name].astype(float) for name in ("high", "low", "close"))
    return talib.ATR(high, low, close, timeperiod=period)


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
    span = talib.MAX(candles.high.astype(float), period) - talib.MIN(candles.low.astype(float), period)
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
    lower_count = returns.rolling(rank_period + 1).rank(method="min") - 1
    rank = 100 * (lower_count / rank_period)
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
    long_level = talib.MAX(candles.high.astype(float), period) - distance
    short_level = talib.MIN(candles.low.astype(float), period) + distance
    buy = close.gt(talib.MAX(candles.high.shift().astype(float), entry_window))
    sell = close.lt(talib.MIN(candles.low.shift().astype(float), entry_window))
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
        return (talib.MAX(candles.high.astype(float), period) + talib.MIN(candles.low.astype(float), period)) / 2

    span_a = ((midpoint(conversion) + midpoint(base)) / 2).shift(displacement)
    span_b_line = midpoint(span_b).shift(displacement)
    above = candles.close.gt(span_a) & candles.close.gt(span_b_line)
    below = candles.close.lt(span_a) & candles.close.lt(span_b_line)
    return above.astype("int8") - below.astype("int8")


def hma_direction(close, window=55):
    """Use Hull MA slope for direction."""
    close = close.astype(float)
    half, smoothing = window // 2, math.isqrt(window)
    fast = talib.WMA(close, half) if half > 1 else close
    raw = 2 * fast - talib.WMA(close, window)
    hull = talib.WMA(raw, smoothing) if smoothing > 1 else raw
    slope = hull.diff()
    span = close.rolling(window + smoothing)
    return _direction(slope.mask(span.max().eq(span.min()), 0.))


def obv_direction(candles, period=20):
    """Compare OBV to its own trailing mean; the starting level cancels out."""
    volume = candles.volume.astype(float).copy()
    if not volume.empty:
        volume.iloc[0] = 0.
    obv = talib.OBV(candles.close.astype(float), volume)
    return _direction(obv - obv.rolling(period).mean())


def stoch_rsi_reentry(close, period=14, stoch_period=14, smooth=3, lower=20., upper=80.):
    rsi = rsi_value(close, period)
    bottom = rsi.rolling(stoch_period).min()
    span = rsi.rolling(stoch_period).max() - bottom
    score = (100 * (rsi - bottom) / span.where(span.gt(0))).rolling(smooth).mean()
    return _reentry(score, lower, upper)


def williams_reentry(candles, period=14, lower=-80., upper=-20.):
    high = talib.MAX(candles.high.astype(float), period)
    low = talib.MIN(candles.low.astype(float), period)
    span = high - low
    score = -100 * (high - candles.close) / span.where(span.gt(0))
    return _reentry(score, lower, upper, middle=-50)



def bb_keltner_squeeze(candles, period=20, bb_deviations=2., kc_multiplier=1.5,
                       breakout_window=20, squeeze_lookback=5):
    """Trade a range breakout after prior compression; zero lookback removes that gate."""
    close = candles.close
    bb_middle = close.rolling(period).mean()
    bb_radius = bb_deviations * close.rolling(period).std(ddof=1)
    ema = ema_value(close, period)
    kc_radius = kc_multiplier * _atr(candles, period)
    compressed = (bb_middle - bb_radius).gt(ema - kc_radius) & (
        bb_middle + bb_radius).lt(ema + kc_radius)
    if squeeze_lookback:
        recent = compressed.shift(fill_value=False).rolling(squeeze_lookback, min_periods=1).max().eq(1)
    else:
        recent = ema.shift().notna()

    above = close.gt(talib.MAX(candles.high.shift().astype(float), breakout_window))
    below = close.lt(talib.MIN(candles.low.shift().astype(float), breakout_window))
    long_entry = recent & above & ~above.shift(fill_value=False)
    short_entry = recent & below & ~below.shift(fill_value=False)
    return _hold_states(long_entry, short_entry, close.le(ema), close.ge(ema)).rename("bb_keltner_squeeze")


def adx_switch(candles, entry_window=40, exit_window=20, adx_period=14, adx_threshold=20.,
               rsi_period=2, lower=10., upper=90.):
    """Use ADX to choose a fresh entry; retain the entering rule through its exit."""
    breakout = donchian_breakout(candles, entry_window, exit_window)
    reversal = rsi_reentry(candles.close, rsi_period, lower, upper)
    adx = dmi_values(candles, adx_period)[0]
    fresh_breakout = breakout.ne(breakout.shift(fill_value=0))
    fresh_reversal = reversal.ne(reversal.shift(fill_value=0))
    trend = adx.gt(adx_threshold)
    range_bound = adx.le(adx_threshold)
    # Both the Donchian entry and exit windows must be ready to compare branches.
    warmup = max(entry_window+1, exit_window+1, 2*adx_period, rsi_period+2)
    state, active, states = 0, None, []
    for i, (b, r, allow_breakout, allow_reversal) in enumerate(zip(
            breakout, reversal, trend & fresh_breakout, range_bound & fresh_reversal)):
        if active is not None:
            branch_state = b if active == "breakout" else r
            if branch_state != state:
                state, active = 0, None
        elif i >= warmup-1:
            if allow_breakout and b:
                state, active = b, "breakout"
            elif allow_reversal and r:
                state, active = r, "reversal"
        states.append(state)
    return pd.Series(states, index=candles.index, dtype="int8", name="adx_switch")


def trend_efficiency(close, period=20):
    """Absolute net move divided by the sum of absolute moves, on a 0–100 scale."""
    _window(period)
    travel = close.diff().abs().rolling(period).sum()
    displacement = close.diff(period).abs()
    return (100 * displacement / travel.where(travel.gt(0))).mask(travel.eq(0), 0.)


def realized_volatility_percentile(close, period=20):
    """Rank current log-return volatility against 5×period strictly earlier values."""
    _window(period)
    volatility = (close / close.shift()).map(math.log).rolling(period).std(ddof=1)

    previous_count = 5 * period
    window = volatility.rolling(previous_count + 1)
    minimum = window.rank(method="min")
    maximum = window.rank(method="max")
    lower_count = minimum - 1
    equal_count = maximum - minimum
    return 100 * (lower_count / previous_count + .5 * (equal_count / previous_count))


def atr_pullback(candles, fast_window=20, slow_window=100, atr_period=14, pullback_atr=1.):
    """Trade a recovery from an ATR-normalized pullback in an established EMA trend.

    Long entries require normalized close-to-fast-EMA distance to move from at
    most -pullback_atr into (-pullback_atr, 0). Both the signal close and fast
    EMA must be above the slow EMA. Exit when the close reaches the fast EMA
    or breaks the slow EMA. Short entries and exits mirror these rules.
    """
    close = candles.close
    fast = ema_value(close, fast_window)
    slow = ema_value(close, slow_window)
    atr = _atr(candles, atr_period)
    distance = (close - fast) / atr.where(atr.gt(0))
    prior = distance.shift()
    buy = prior.le(-pullback_atr) & distance.gt(-pullback_atr) & distance.lt(0) & close.gt(slow) & fast.gt(slow)
    sell = prior.ge(pullback_atr) & distance.lt(pullback_atr) & distance.gt(0) & close.lt(slow) & fast.lt(slow)
    return _hold_states(buy, sell, close.ge(fast) | close.le(slow),
                        close.le(fast) | close.ge(slow)).rename("atr_pullback")



def aroon_oscillator(candles, period=25, threshold=50.):
    """Vote beyond the symmetric Aroon Oscillator thresholds."""
    value = talib.AROONOSC(candles.high, candles.low, timeperiod=period)
    return (value.gt(threshold).astype("int8") - value.lt(-threshold).astype("int8")).rename("aroon_oscillator")


def kama_trend(close, period=30):
    """Vote by close relative to Kaufman's adaptive moving average."""
    average = talib.KAMA(close, timeperiod=period)
    return (close.gt(average).astype("int8") - close.lt(average).astype("int8")).rename("kama_trend")


def normalized_atr(candles, period):
    """TA-Lib normalized ATR in percent of close."""
    return talib.NATR(candles.high, candles.low, candles.close, timeperiod=period)


def parabolic_sar(candles, acceleration=.02, maximum=.2):
    """Vote by close relative to Parabolic SAR."""
    level = talib.SAR(candles.high.astype(float), candles.low.astype(float), acceleration=acceleration, maximum=maximum)
    return _direction(candles.close - level)


def ultimate_oscillator(candles, fast=7, medium=14, slow=28, lower=30., upper=70.):
    """Enter on re-entry from an extreme and exit at the midpoint."""
    value = talib.ULTOSC(candles.high.astype(float), candles.low.astype(float), candles.close.astype(float),
                        timeperiod1=fast, timeperiod2=medium, timeperiod3=slow)
    return _reentry(value, lower, upper)


def chaikin_oscillator(candles, fast=3, slow=10):
    """Vote by the sign of the Chaikin accumulation/distribution oscillator."""
    value = talib.ADOSC(candles.high.astype(float), candles.low.astype(float), candles.close.astype(float), candles.volume.astype(float),
                       fastperiod=fast, slowperiod=slow)
    return _direction(value)


def linear_regression_trend(close, period=30):
    """Vote by the slope of a line fitted to recent closes."""
    return _direction(talib.LINEARREG_SLOPE(close.astype(float), timeperiod=period))


def balance_of_power(candles, period=14):
    """Vote by the sign of simple-average Balance of Power."""
    open_, high, low, close = (candles[name].astype(float) for name in ("open", "high", "low", "close"))
    return _direction(talib.SMA(talib.BOP(open_, high, low, close), timeperiod=period))


def engulfing_pattern(candles, hold_bars=1):
    """Use the most recent engulfing pattern until its holding period expires."""
    open_, high, low, close = (candles[name].astype(float) for name in ("open", "high", "low", "close"))
    pattern = _direction(talib.CDLENGULFING(open_, high, low, close))
    if hold_bars == 1:
        return pattern
    return pattern.where(pattern.ne(0)).ffill(limit=hold_bars - 1).fillna(0).astype("int8")
