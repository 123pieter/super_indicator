"""Provider-aware data loading and causal nine-vote live analysis."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import talib
from numpy.lib.stride_tricks import sliding_window_view

import storage
from research_lab.strategies import _atr, dmi_values, rsi_value
from timeframes import completed_bars, get_timeframe
from config import (DATA_FOLDER, DB_PATH, LOAD_FROM, DEFAULT_TIMEFRAME,
                    MIN_BARS, BUY_THRESHOLD, ACTIVE_PROVIDER, storage_paths)

# Nine equally weighted directional votes; volatility is a separate entry filter.
SIGNAL_COLS = ["sma_signal", "ema_signal", "macd_signal", "rsi_signal", "bb_signal",
               "stoch_signal", "adx_signal", "obv_signal", "cci_signal"]
ENGINE_VERSION = 3


# ============================================================
# Data loading (unchanged)
# ============================================================

def load_data(symbol, timeframe=DEFAULT_TIMEFRAME, provider=None):
    folder, database = storage_paths(provider or ACTIVE_PROVIDER, DATA_FOLDER, DB_PATH)
    get_timeframe(timeframe)
    try:
        df = storage.read_data(symbol, timeframe, folder, database, LOAD_FROM, strict=True, provider=provider or ACTIVE_PROVIDER)
        return None if df.empty else df
    except Exception as exc:
        raise RuntimeError(f"Unable to load {symbol} from {LOAD_FROM.title()}") from exc


def get_available_symbols(timeframe=DEFAULT_TIMEFRAME, provider=None):
    folder, database = storage_paths(provider or ACTIVE_PROVIDER, DATA_FOLDER, DB_PATH)
    get_timeframe(timeframe)
    if (provider or ACTIVE_PROVIDER) != "polygon" and LOAD_FROM == "sqlite" and not database.exists():
        return []
    try:
        return storage.list_symbols(timeframe, folder, database, LOAD_FROM, provider=provider or ACTIVE_PROVIDER)
    except Exception as exc:
        raise RuntimeError(f"Unable to list symbols from {LOAD_FROM.title()}") from exc


# ============================================================
# Super Indicator
# ============================================================

def _mean_abs_dev(x: pd.Series, n: int) -> pd.Series:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x.to_numpy(dtype=float), n)
        out[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=x.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="last")]
    required = ["open", "high", "low", "close", "volume"]
    df[required] = df[required].apply(pd.to_numeric, errors="raise")
    if df[required].isna().any().any() or not np.isfinite(df[required]).all().all():
        raise ValueError("OHLCV data contains missing or non-finite values")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
            (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC candle range")
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # 1. SMA
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["sma_signal"] = ((close > df["sma50"]) & (df["sma50"] > df["sma200"])).astype(float)

    # 2. EMA
    df["ema12"] = talib.EMA(close.astype(float), timeperiod=12)
    df["ema26"] = talib.EMA(close.astype(float), timeperiod=26)
    df["ema_signal"] = (df["ema12"] > df["ema26"]).astype(float)

    # 3. MACD
    df["macd"], df["macd_signal_line"], df["macd_hist"] = talib.MACD(
        close.astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd_signal"] = (df["macd_hist"] > 0).astype(float)   # same as macd > signal line

    # 4. RSI (Wilder)
    df["rsi"] = rsi_value(close, 14)
    df["rsi_signal"] = (df["rsi"] > 50).astype(float)

    # 5. Bollinger Bands
    df["bb_mid"] = close.rolling(20).mean()
    df["bb_std"] = close.rolling(20).std()
    df["bb_signal"] = ((close > df["bb_mid"]) & (close < df["bb_mid"] + 2 * df["bb_std"])).astype(float)

    # 6. Stochastic (flat range -> NaN, not a fake value)
    low14, high14 = talib.MIN(low.astype(float), 14), talib.MAX(high.astype(float), 14)
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_signal"] = ((df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] > 20) & (df["stoch_k"] < 80)).astype(float)

    # 7. ADX and directional lines
    df["adx"], plus_di, minus_di = dmi_values(df, 14)
    df["plus_di"], df["minus_di"] = plus_di, minus_di
    df["adx_signal"] = ((df["adx"] > 20) & (plus_di > minus_di)).astype(float)

    # 8. ATR -> regime filter, not a directional vote
    df["atr"] = _atr(df, 14)
    df["regime_ok"] = df["atr"] > df["atr"].rolling(20).mean() * 0.8

    # 9. OBV
    obv_volume = volume.astype(float).copy()
    if not obv_volume.empty:
        obv_volume.iloc[0] = 0.
    df["obv"] = talib.OBV(close.astype(float), obv_volume)
    df["obv_sma"] = df["obv"].rolling(20).mean()
    df["obv_signal"] = (df["obv"] > df["obv_sma"]).astype(float)

    # 10. CCI (vectorised)
    tp = (high + low + close) / 3
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * _mean_abs_dev(tp, 20).replace(0, np.nan))
    df["cci_signal"] = ((df["cci"] > 0) & (df["cci"] < 100)).astype(float)

    # Score. Scores start only after SMA200 is ready.
    df["ready"] = df["sma200"].notna()
    df["super_score"] = df[SIGNAL_COLS].mean(axis=1).where(df["ready"])
    return df


def signal_correlation(df_ind: pd.DataFrame) -> pd.DataFrame:
    """Pairs near 1.0 are the same information counted twice."""
    return df_ind.loc[df_ind["ready"], SIGNAL_COLS].corr().round(2)


def crosses(df, threshold=BUY_THRESHOLD):
    """A valid prior score and volatility regime are required; no warmup crossing."""
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    return ((df.super_score >= threshold) & (df.super_score.shift() < threshold)
            & df.regime_ok)


def live_result(df, symbol, threshold=BUY_THRESHOLD, timeframe=DEFAULT_TIMEFRAME):
    """Describe the latest signal from an already calculated frame; never run a backtest."""
    if len(df) < MIN_BARS or not np.isfinite(df.super_score.iloc[-1]):
        return None
    return dict(symbol=symbol, timeframe=timeframe, price=float(df.close.iloc[-1]),
                current_score=float(df.super_score.iloc[-1]),
                prev_score=float(df.super_score.iloc[-2]),
                votes_met=int(df[SIGNAL_COLS].iloc[-1].sum()),
                regime_ok=bool(df.regime_ok.iloc[-1]),
                fresh_cross=bool(crosses(df, threshold).iloc[-1]),
                data_as_of=df.index[-1].isoformat())


def analyze_coin(df, symbol, threshold=BUY_THRESHOLD, timeframe=DEFAULT_TIMEFRAME):
    """Live-only entrypoint: completed candles, causal indicators, current signal."""
    df = completed_bars(df, timeframe)
    if len(df) < MIN_BARS:
        return None
    return live_result(compute_indicators(df), symbol, threshold, timeframe)


def main():
    from research import evaluate_history
    results = []
    for symbol in get_available_symbols():
        raw = load_data(symbol)
        if raw is None:
            continue
        d = compute_indicators(completed_bars(raw, DEFAULT_TIMEFRAME))
        result = live_result(d, symbol)
        if result:
            result.update(evaluate_history(d, provider=ACTIVE_PROVIDER))
            results.append(result)
    print(f"Super Indicator - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(pd.DataFrame(results).to_string(index=False) if results else "No coins with sufficient data.")


if __name__ == "__main__":
    main()
