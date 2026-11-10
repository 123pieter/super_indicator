"""Stage 4 research signals. Inputs are completed candle closes, in time order."""
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



