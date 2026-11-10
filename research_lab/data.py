"""Read and audit stored OHLCV without changing the market database."""
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from storage import COLUMNS
from timeframes import TIMEFRAMES, missing_bars

# Require an absolute ISO date or datetime, not relative strings such as "now".
_ISO_TIMESTAMP = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})?)?"


@contextmanager
def open_ohlcv(database):
    """Open a read-only SQLite snapshot; missing paths never create a database."""
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("BEGIN")
        required = {"provider", "date", "symbol", "timeframe", *COLUMNS}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ohlcv)")}
        if not required <= columns:
            raise ValueError("Missing ohlcv table or columns: " + ", ".join(sorted(required - columns)))
        yield conn


def stored_pairs(conn, provider):
    """List stored symbol/timeframe pairs, including unsupported intervals for audit."""
    return conn.execute(
        "SELECT DISTINCT symbol, timeframe FROM ohlcv WHERE provider=? ORDER BY timeframe, symbol",
        (provider,),
    ).fetchall()


def load_pair(conn, provider, symbol, timeframe, as_of=None):
    """Return sorted, completed UTC candles and audit counts for one stored pair.

    Invalid rows and all copies of a duplicated timestamp are excluded. The
    usable frame may still contain gaps; callers must inspect its audit before
    simulating trades. No candles are filled or written back.
    """
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    if pd.isna(cutoff):
        raise ValueError("Audit cutoff must be a valid timestamp")
    raw = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM ohlcv "
        "WHERE provider=? AND symbol=? AND timeframe=?",
        conn, params=(provider, symbol, timeframe),
    )
    report = dict(symbol=symbol, timeframe=timeframe, stored=len(raw))
    if timeframe not in TIMEFRAMES:
        return pd.DataFrame(columns=COLUMNS), dict(report, usable=0, status="unsupported timeframe")
    duration = TIMEFRAMES[timeframe].duration
    date = raw.date.astype("string")
    timestamps = pd.to_datetime(date.where(date.str.fullmatch(_ISO_TIMESTAMP, na=False)),
                                utc=True, format="mixed", errors="coerce")
    prices = raw[COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_values = (
        np.isfinite(prices).all(axis=1)
        & prices[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & prices.volume.ge(0)
        & prices.high.ge(prices[["open", "low", "close"]].max(axis=1))
        & prices.low.le(prices[["open", "high", "close"]].min(axis=1))
    )
    aligned = timestamps.eq(timestamps.dt.floor(duration))
    duplicates = timestamps.notna() & timestamps.duplicated(keep=False)
    future = timestamps.gt(cutoff)
    complete = timestamps.le(cutoff - duration)
    usable = prices.loc[aligned & valid_values & ~duplicates & complete].copy()
    usable.index = pd.DatetimeIndex(timestamps.loc[usable.index], name="date")
    usable = usable.sort_index()
    first = usable.index[0] if len(usable) else None
    last = usable.index[-1] if len(usable) else None
    gaps = int((last-first)//duration + 1 - len(usable)) if len(usable) else 0
    report.update(
        usable=len(usable), first_utc=first, last_utc=last,
        bad_timestamp=int(timestamps.isna().sum()),
        off_grid=int((timestamps.notna() & ~aligned).sum()),
        duplicate_rows=int(duplicates.sum()), invalid_ohlcv=int((~valid_values).sum()),
        future=int(future.sum()), unfinished=int((timestamps.notna() & ~future & ~complete).sum()),
        internal_missing=gaps, trailing_missing=missing_bars(last, timeframe, cutoff),
    )
    issues = ("bad_timestamp", "off_grid", "duplicate_rows", "invalid_ohlcv", "future", "internal_missing")
    report["status"] = "check required" if not len(usable) or any(report[key] for key in issues) else "ok"
    return usable, report


def audit_database(database, provider, as_of=None):
    """Audit all stored pairs with one cutoff and one consistent SQLite snapshot."""
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    with open_ohlcv(database) as conn:
        return pd.DataFrame([
            load_pair(conn, provider, symbol, timeframe, cutoff)[1]
            for symbol, timeframe in stored_pairs(conn, provider)
        ])
