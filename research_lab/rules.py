"""One declaration per research rule: parameters, validation and warmup."""
from dataclasses import dataclass
from typing import Callable
from math import isqrt

from .strategies import (balance_of_power, chaikin_oscillator, engulfing_pattern,
                         linear_regression_trend, parabolic_sar, ultimate_oscillator)
from .strategies import (adx_dmi, bollinger_reentry, donchian_breakout, ema_trend, macd_direction,
                         roc_direction, rsi_adx_long, rsi_midline_long, rsi_reentry, sma_trend)

from .strategies import (aroon_oscillator, kama_trend, adx_switch, atr_pullback, bb_keltner_squeeze, chandelier_breakout, cmf_direction, connors_reentry, hma_direction,
                         ichimoku_direction, keltner_breakout, mfi_reentry, obv_direction,
                         stoch_rsi_reentry, supertrend_direction, vwap_band_reentry, williams_reentry)


@dataclass(frozen=True)
class Parameter:
    label: str
    default: int | float
    minimum: int | float
    exclusive: bool = False


@dataclass(frozen=True)
class Rule:
    label: str
    function: Callable
    parameters: dict
    warmup: Callable
    check: Callable | None = None
    needs_ohlc: bool = False
    needs_volume: bool = False
    intraday_only: bool = False


def check_macd(params):
    if params["fast"] >= params["slow"]:
        raise ValueError("MACD fast period must be smaller than slow period")


def check_rsi(params):
    if not 0 <= params["lower"] < 50 < params["upper"] <= 100:
        raise ValueError("RSI thresholds must satisfy 0 <= lower < 50 < upper <= 100")


def check_adx(params):
    if params["adx_threshold"] > 100:
        raise ValueError("ADX threshold must be at most 100")


RSI_TREND_PARAMS = {
    "sma_window": Parameter("SMA entry window", 50, 2),
    "ema_window": Parameter("EMA entry window", 7, 2),
    "rsi_period": Parameter("RSI period", 2, 2),
}


def rsi_trend_warmup(p):
    return max(p["sma_window"], p["ema_window"], p["rsi_period"]+1)


REGISTRY = {
    "sma_trend": Rule("SMA trend", sma_trend, {
        "window": Parameter("SMA window", 100, 2),
    }, lambda p: p["window"]),
    "bollinger_reentry": Rule("Bollinger mean reversion", bollinger_reentry, {
        "window": Parameter("Bollinger window", 20, 2),
        "deviations": Parameter("Bollinger standard deviations", 2., 0., True),
    }, lambda p: p["window"] + 1),
    "macd_direction": Rule("MACD histogram direction", macd_direction, {
        "fast": Parameter("MACD fast period", 12, 1),
        "slow": Parameter("MACD slow period", 26, 1),
        "signal": Parameter("MACD signal period", 9, 1),
    }, lambda p: p["slow"] + p["signal"] - 1, check_macd),
    "donchian_breakout": Rule("Donchian breakout", donchian_breakout, {
        "entry_window": Parameter("Entry channel candles", 55, 2),
        "exit_window": Parameter("Exit channel candles", 20, 2),
    }, lambda p: max(p["entry_window"], p["exit_window"]) + 1, needs_ohlc=True),
    "roc_direction": Rule("ROC direction", roc_direction, {
        "window": Parameter("ROC window", 60, 2),
    }, lambda p: p["window"] + 1),
    "rsi_reentry": Rule("RSI re-entry", rsi_reentry, {
        "period": Parameter("RSI period", 2, 2),
        "lower": Parameter("Lower RSI threshold", 10., 0.),
        "upper": Parameter("Upper RSI threshold", 90., 0.),
    }, lambda p: p["period"] + 2, check_rsi),
    "adx_dmi": Rule("ADX/DMI direction", adx_dmi, {
        "period": Parameter("ADX/DMI period", 14, 2),
        "adx_threshold": Parameter("ADX strength threshold", 20., 0.),
    }, lambda p: 2*p["period"], check_adx, needs_ohlc=True),
    "ema_trend": Rule("EMA trend", ema_trend, {
        "window": Parameter("EMA window", 200, 2),
    }, lambda p: p["window"]),
    "rsi_adx_long": Rule("SMA + EMA + RSI versus ADX", rsi_adx_long,
                        RSI_TREND_PARAMS | {"adx_period": Parameter("ADX period", 2, 2)},
                        lambda p: max(rsi_trend_warmup(p),2*p["adx_period"]), needs_ohlc=True),
    "rsi_midline_long": Rule("SMA + EMA + RSI versus 50", rsi_midline_long,
                            RSI_TREND_PARAMS, rsi_trend_warmup),
}

# New candidates use one starting configuration each, before any parameter search.
def _period(default, label="Candles", minimum=2):
    return Parameter(label, default, minimum)


def _multiplier(default):
    return Parameter("Band multiplier", default, 0., True)


def _oscillator_bounds(lower, upper, minimum=0.):
    return {"lower": Parameter("Lower threshold", lower, minimum),
            "upper": Parameter("Upper threshold", upper, minimum)}


def _check_williams(p):
    if not -100 <= p["lower"] < -50 < p["upper"] <= 0:
        raise ValueError("Williams %R thresholds must satisfy -100 <= lower < -50 < upper <= 0")



def _check_adx_switch(params):
    check_adx(params)
    check_rsi(params)



def _check_pullback(params):
    if params["fast_window"] >= params["slow_window"]:
        raise ValueError("Fast EMA window must be smaller than slow EMA window")


def _check_aroon(params):
    if params["threshold"] >= 100:
        raise ValueError("Aroon threshold must be below 100")


def _check_sar(params):
    if params["acceleration"] > params["maximum"]:
        raise ValueError("SAR acceleration must be at most its maximum")


def _check_ultimate(params):
    if not params["fast"] < params["medium"] < params["slow"]:
        raise ValueError("Ultimate Oscillator periods must be strictly increasing")
    check_rsi(params)


def _check_chaikin(params):
    if params["fast"] >= params["slow"]:
        raise ValueError("Chaikin fast period must be smaller than slow period")


REGISTRY.update({
    "linear_regression_trend": Rule("Linear Regression Slope", linear_regression_trend, {
        "period": _period(30),
    }, lambda p: p["period"]),
    "balance_of_power": Rule("Smoothed Balance of Power", balance_of_power, {
        "period": _period(14),
    }, lambda p: p["period"], needs_ohlc=True),
    "engulfing_pattern": Rule("Engulfing Candle", engulfing_pattern, {
        "hold_bars": _period(1, "Holding candles", minimum=1),
    }, lambda p: 3, needs_ohlc=True),
    "parabolic_sar": Rule("Parabolic SAR", parabolic_sar, {
        "acceleration": Parameter("Acceleration", .02, 0., True),
        "maximum": Parameter("Maximum acceleration", .2, 0., True),
    }, lambda p: 2, _check_sar, needs_ohlc=True),
    "ultimate_oscillator": Rule("Ultimate Oscillator", ultimate_oscillator, {
        "fast": _period(7), "medium": _period(14), "slow": _period(28),
        **_oscillator_bounds(30., 70.),
    }, lambda p: p["slow"] + 2, _check_ultimate, needs_ohlc=True),
    "chaikin_oscillator": Rule("Chaikin A/D Oscillator", chaikin_oscillator, {
        "fast": _period(3), "slow": _period(10),
    }, lambda p: p["slow"], _check_chaikin, needs_ohlc=True, needs_volume=True),
    "aroon_oscillator": Rule("Aroon Oscillator", aroon_oscillator, {
        "period": _period(25), "threshold": Parameter("Aroon threshold", 50., 0.),
    }, lambda p: p["period"] + 1, _check_aroon, needs_ohlc=True),
    "kama_trend": Rule("KAMA trend", kama_trend, {
        "period": _period(30),
    }, lambda p: p["period"] + 1),
    "connors_reentry": Rule("Connors RSI re-entry", connors_reentry, {
        "rsi_period": _period(3), "streak_period": _period(2), "rank_period": _period(100),
        **_oscillator_bounds(10., 90.),
    }, lambda p: max(p["rsi_period"], p["streak_period"], p["rank_period"] + 1) + 2, check_rsi),
    "supertrend_direction": Rule("SuperTrend", supertrend_direction, {
        "period": _period(10), "multiplier": _multiplier(3.),
    }, lambda p: p["period"] + 1, needs_ohlc=True),
    "vwap_band_reentry": Rule("UTC-session VWAP band re-entry", vwap_band_reentry, {
        "deviations": _multiplier(2.),
    }, lambda p: 3, needs_ohlc=True, needs_volume=True, intraday_only=True),
    "keltner_breakout": Rule("Keltner breakout", keltner_breakout, {
        "period": _period(20), "multiplier": _multiplier(1.5),
    }, lambda p: p["period"] + 1, needs_ohlc=True),
    "bb_keltner_squeeze": Rule("Bollinger/Keltner squeeze breakout", bb_keltner_squeeze, {
        "period": _period(20, "Band period"),
        "bb_deviations": _multiplier(2.),
        "kc_multiplier": _multiplier(1.5),
        "breakout_window": _period(20, "Breakout channel candles"),
        "squeeze_lookback": _period(5, "Prior squeeze lookback candles (0 disables)", minimum=0),
    }, lambda p: max(p["period"], p["breakout_window"]) + 1, needs_ohlc=True),
    "adx_switch": Rule("ADX Donchian/RSI regime switch", adx_switch, {
        "entry_window": _period(40, "Entry channel candles"),
        "exit_window": _period(20, "Exit channel candles"),
        "adx_period": _period(14, "ADX period"),
        "adx_threshold": Parameter("ADX strength threshold", 20., 0.),
        "rsi_period": _period(2, "RSI period"),
        **_oscillator_bounds(10., 90.),
    }, lambda p: max(p["entry_window"] + 1, p["exit_window"] + 1, 2*p["adx_period"],
                     p["rsi_period"] + 2), _check_adx_switch, needs_ohlc=True),
    "atr_pullback": Rule("ATR-normalized EMA pullback", atr_pullback, {
        "fast_window": _period(20, "Fast EMA candles"),
        "slow_window": _period(100, "Slow EMA candles"),
        "atr_period": _period(14, "ATR period"),
        "pullback_atr": Parameter("Pullback distance in ATR units", 1., 0., True),
    }, lambda p: max(p["slow_window"], p["fast_window"] + 1, p["atr_period"] + 2),
       _check_pullback, needs_ohlc=True),
    "chandelier_breakout": Rule("Donchian with Chandelier Exit", chandelier_breakout, {
        "entry_window": _period(40), "period": _period(22), "multiplier": _multiplier(3.),
    }, lambda p: max(p["entry_window"] + 1, p["period"] + 1), needs_ohlc=True),
    "mfi_reentry": Rule("Money Flow Index re-entry", mfi_reentry, {
        "period": _period(14), **_oscillator_bounds(20., 80.),
    }, lambda p: p["period"] + 2, check_rsi, needs_ohlc=True, needs_volume=True),
    "cmf_direction": Rule("Chaikin Money Flow", cmf_direction, {"period": _period(20)},
                          lambda p: p["period"], needs_ohlc=True, needs_volume=True),
    "ichimoku_direction": Rule("Ichimoku Cloud direction", ichimoku_direction, {
        "conversion": _period(9), "base": _period(26), "span_b": _period(52),
        "displacement": _period(26, minimum=0),
    }, lambda p: max(p["conversion"], p["base"], p["span_b"]) + p["displacement"], needs_ohlc=True),
    "hma_direction": Rule("Hull Moving Average slope", hma_direction, {"window": _period(55)},
                          lambda p: p["window"] + isqrt(p["window"])),
    "obv_direction": Rule("On-Balance Volume", obv_direction, {"period": _period(20)},
                          lambda p: p["period"], needs_ohlc=True, needs_volume=True),
    "stoch_rsi_reentry": Rule("Stochastic RSI re-entry", stoch_rsi_reentry, {
        "period": _period(14), "stoch_period": _period(14), "smooth": _period(3, minimum=1),
        **_oscillator_bounds(20., 80.),
    }, lambda p: p["period"] + p["stoch_period"] + p["smooth"], check_rsi),
    "williams_reentry": Rule("Williams %R re-entry", williams_reentry, {
        "period": _period(14), **_oscillator_bounds(-80., -20., -100.),
    }, lambda p: p["period"] + 1, _check_williams, needs_ohlc=True),
})

