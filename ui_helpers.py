"""Presentation calculations independent of Streamlit and provider access."""
import pandas as pd
from timeframes import missing_bars

INDICATORS = {
    "sma": ("Moving averages", "Close > SMA50 > SMA200"),
    "ema": ("EMA trend", "EMA12 > EMA26"),
    "macd": ("MACD", "MACD above its signal line"),
    "rsi": ("RSI", "RSI > 50"),
    "bb": ("Bollinger bands", "Close above middle band, below upper band"),
    "stoch": ("Stochastic", "%K above %D and between 20 and 80"),
    "adx": ("ADX trend", "ADX > 20 and +DI > -DI"),
    "obv": ("Volume trend", "OBV above its 20-bar average"),
    "cci": ("CCI", "CCI between 0 and 100"),
}


def freshness(latest, timeframe, now=None):
    """Report missing completed candles, allowing the currently forming candle."""
    missing = missing_bars(latest, timeframe, now)
    if missing is None:
        return "No completed data", None
    return ("Up to date" if missing == 0 else f"Stale: {missing} missing candles"), missing


def filter_scanner(frame, minimum=0., fresh=False, regime=False, sufficient=False, search=""):
    """Apply independent scanner filters without changing stored analysis results."""
    selected = frame[frame.current_score >= minimum]
    if fresh:
        selected = selected[selected.fresh_cross]
    if sufficient:
        selected = selected[selected.sufficient_history]
    if regime:
        selected = selected[selected.regime_ok]
    if search.strip():
        selected = selected[selected.symbol.str.contains(search.strip(), case=False, regex=False)]
    return selected.sort_values(["current_score", "symbol"], ascending=[False, True]).copy()


def format_price(value):
    """Retain useful precision for sub-cent tokens while keeping large prices readable."""
    if pd.isna(value):
        return "--"
    if value == 0:
        return "$0"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return "$" + f"{value:.8g}"


def contribution_table(row):
    """Explain each binary vote and its equal contribution to the total score."""
    return pd.DataFrame([{
        "Indicator": label, "Condition": condition,
        "Status": "Met" if row[f"{name}_signal"] else "Not met",
        "Contribution": float(row[f"{name}_signal"]) / len(INDICATORS),
    } for name, (label, condition) in INDICATORS.items()])
