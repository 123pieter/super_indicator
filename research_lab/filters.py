"""Entry filters are boolean gates, never directional votes or exit rules."""
import math

import pandas as pd

from .strategies import bollinger_width_percentile, choppiness_index, dmi_values, ema_value, rsi_value

DEFAULTS = {"ema_alignment": (200,None), "adx_strength": (14,20.), "rsi_confirmation": (14,50.),
            "relative_volume": (20,1.), "bollinger_compression": (20,20.),
            "low_choppiness": (14,38.2)}


def validate_filters(items):
    if not isinstance(items,list):
        raise ValueError("Entry filters must be a list")
    result = []
    for item in items:
        if not isinstance(item,dict) or item.keys()-{"filter_rule","period","threshold"}:
            raise ValueError("Invalid entry filter fields")
        rule = item.get("filter_rule")
        if rule not in tuple(DEFAULTS):
            raise ValueError(f"Unknown entry filter: {rule}")
        if any(f["filter_rule"]==rule for f in result):
            raise ValueError(f"Duplicate entry filter: {rule}")
        default_period, default_threshold = DEFAULTS[rule]
        period = item.get("period",default_period)
        if type(period) is not int or period < 2:
            raise ValueError("Filter period must be an integer of at least 2")
        threshold = item.get("threshold",default_threshold)
        if default_threshold is None:
            if threshold is not None:
                raise ValueError("EMA alignment does not take a threshold")
        elif rule == "relative_volume":
            if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("Relative volume threshold must be positive and finite")
        elif type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 100:
            raise ValueError("Filter threshold must be a finite number between 0 and 100")
        result.append(dict(filter_rule=rule,period=period,threshold=threshold))
    return result


def filter_warmup(f):
    rule, period = f["filter_rule"], f["period"]
    if rule == "bollinger_compression":
        return 6 * period
    if rule == "adx_strength":
        return 2 * period
    return period + (rule in ("rsi_confirmation", "relative_volume"))


def entry_mask(candles, filters, direction, cache):
    """Require every filter on the signal candle; cache indicator values per asset."""
    if not filters:
        return None
    close = candles.close
    allowed = pd.Series(True,index=candles.index)
    for f in filters:
        rule, period, threshold = f["filter_rule"], f["period"], f["threshold"]
        key = rule,period
        if key not in cache:
            if rule=="ema_alignment":
                cache[key] = ema_value(close,period)
            elif rule=="adx_strength":
                cache[key] = dmi_values(candles,period)[0]
            elif rule=="relative_volume":
                previous_mean = candles.volume.shift().rolling(period).mean()
                cache[key] = candles.volume / previous_mean.where(previous_mean.gt(0))
            elif rule == "bollinger_compression":
                cache[key] = bollinger_width_percentile(candles, period)
            elif rule == "low_choppiness":
                cache[key] = choppiness_index(candles, period)
            else:
                cache[key] = rsi_value(close,period)
        value = cache[key]
        if rule=="ema_alignment":
            allowed &= close.gt(value) if direction=="long" else close.lt(value)
        elif rule in ("adx_strength","relative_volume"):
            allowed &= value.gt(threshold)
        elif rule in ("bollinger_compression", "low_choppiness"):
            allowed &= value.le(threshold)
        else:
            allowed &= value.gt(threshold) if direction=="long" else value.lt(threshold)
    return allowed
