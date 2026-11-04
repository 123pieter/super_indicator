"""Shared timeframe definitions. Candle timestamps always denote their UTC start."""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Timeframe:
    multiplier: int
    timespan: str
    duration: pd.Timedelta


# Add future fixed-duration intervals here; storage and fetching use these identifiers.
TIMEFRAMES = {
    "1m": Timeframe(1, "minute", pd.Timedelta(minutes=1)),
    "5m": Timeframe(5, "minute", pd.Timedelta(minutes=5)),
    "15m": Timeframe(15, "minute", pd.Timedelta(minutes=15)),
    "30m": Timeframe(30, "minute", pd.Timedelta(minutes=30)),
    "1h": Timeframe(1, "hour", pd.Timedelta(hours=1)),
    "4h": Timeframe(4, "hour", pd.Timedelta(hours=4)),
    "1d": Timeframe(1, "day", pd.Timedelta(days=1)),
}


def get_timeframe(name):
    """Reject unknown identifiers before creating paths or requesting provider data."""
    if name not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {name}. Choose from {', '.join(TIMEFRAMES)}")
    return TIMEFRAMES[name]


def completed_bars(df, timeframe="1d", now=None):
    """A candle is complete when its start plus the selected duration is <= now."""
    interval = get_timeframe(timeframe)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return df.loc[pd.to_datetime(df.index, utc=True) + interval.duration <= now].copy()


def missing_bars(latest, timeframe, now=None):
    """Count expected completed intervals after the latest candle, not internal gaps."""
    if latest is None or pd.isna(latest):
        return None
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    latest = pd.Timestamp(latest)
    latest = latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
    return max(0, int((now-latest) // get_timeframe(timeframe).duration)-1)
