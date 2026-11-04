"""Provider-aware data loading and causal nine-vote live analysis."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import storage
from timeframes import completed_bars, get_timeframe
from config import (DATA_FOLDER, DB_PATH, LOAD_FROM, DEFAULT_TIMEFRAME,
                    MIN_BARS, BUY_THRESHOLD, ACTIVE_PROVIDER, storage_paths)

# Nine equally weighted directional votes; volatility is a separate entry filter.
SIGNAL_COLS = ["sma_signal", "ema_signal", "macd_signal", "rsi_signal", "bb_signal",
               "stoch_signal", "adx_signal", "obv_signal", "cci_signal"]
ENGINE_VERSION = 2


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

def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing, the standard for RSI / ATR / ADX."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Smoothing period must be a positive integer")
    # Seed with the first n contiguous observations, then recurse. A gap resets the seed.
    result = np.full(len(s), np.nan)
    seed, previous = [], np.nan
    for i, value in enumerate(s.to_numpy(dtype=float)):
        if not np.isfinite(value):
            seed, previous = [], np.nan
        elif np.isnan(previous):
            seed.append(value)
            if len(seed) == n:
                previous = float(np.mean(seed))
                result[i] = previous
        else:
            previous += (value - previous) / n
            result[i] = previous
    return pd.Series(result, index=s.index)


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
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    df["ema_signal"] = (df["ema12"] > df["ema26"]).astype(float)

    # 3. MACD
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal_line"]
    df["macd_signal"] = (df["macd_hist"] > 0).astype(float)   # same as macd > signal line

    # 4. RSI (Wilder)
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), 14)
    loss = wilder((-delta).clip(lower=0), 14)
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi"] = 50
    df["rsi_signal"] = (df["rsi"] > 50).astype(float)

    # 5. Bollinger Bands
    df["bb_mid"] = close.rolling(20).mean()
    df["bb_std"] = close.rolling(20).std()
    df["bb_signal"] = ((close > df["bb_mid"]) & (close < df["bb_mid"] + 2 * df["bb_std"])).astype(float)

    # 6. Stochastic (flat range -> NaN, not a fake value)
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_signal"] = ((df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] > 20) & (df["stoch_k"] < 80)).astype(float)

    # 7. ADX (Wilder throughout)
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    plus_di = 100 * wilder(plus_dm, 14) / atr
    minus_di = 100 * wilder(minus_dm, 14) / atr
    df["plus_di"], df["minus_di"] = plus_di, minus_di
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = wilder(dx, 14)
    df["adx_signal"] = ((df["adx"] > 20) & (plus_di > minus_di)).astype(float)

    # 8. ATR -> regime filter, not a directional vote
    df["atr"] = atr
    df["regime_ok"] = df["atr"] > df["atr"].rolling(20).mean() * 0.8

    # 9. OBV
    df["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
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
