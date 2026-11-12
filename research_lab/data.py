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


def load_pair(conn, provider, symbol, timeframe, as_of=None, *, historical_cutoff=False):
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
    if historical_cutoff:
        # A deliberate development boundary is not a future-data anomaly.
        keep = timestamps.isna() | timestamps.lt(cutoff)
        raw, timestamps = raw.loc[keep], timestamps.loc[keep]
        report["stored"] = len(raw)
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
    report["status"] = "check required" if audit_problem(report) else "ok"
    return usable, report


def audit_problem(report, *, allow_internal_gaps=False):
    """Describe the blocking checks. Unfinished candles and trailing lag are informational."""
    if report.get("status") == "unsupported timeframe":
        return "unsupported timeframe"
    checks = {
        "internal_missing": "missing candles within stored history",
        "bad_timestamp": "unreadable timestamps",
        "off_grid": "timestamps off the timeframe grid",
        "duplicate_rows": "rows with duplicate timestamps",
        "invalid_ohlcv": "invalid OHLCV rows",
        "future": "candles dated after the audit cutoff",
    }
    if allow_internal_gaps:
        del checks["internal_missing"]
    details = [f"{report[key]} {label}" for key, label in checks.items() if report[key]]
    if not report["usable"]:
        details.insert(0, "no usable completed candles")
    return "; ".join(details)


def backtest_history(candles, audit, latest_continuous):
    """Use the newest uninterrupted suffix when internal gaps are the only issue."""
    if (not latest_continuous or not audit.get("internal_missing")
            or audit_problem(audit, allow_internal_gaps=True)):
        return candles, False
    duration = TIMEFRAMES[audit["timeframe"]].duration
    jumps = candles.index.to_series().diff().gt(duration)
    start = jumps[jumps].index[-1]
    return candles.loc[start:], True


def fill_empty_candles(candles, timeframe, confirmed_dates):
    """Fill only confirmed internal gaps; leave unresolved histories untouched."""
    if not confirmed_dates or candles.empty:
        return candles
    grid = pd.date_range(candles.index[0], candles.index[-1],
                         freq=TIMEFRAMES[timeframe].duration, name=candles.index.name)
    missing = grid.difference(candles.index)
    confirmed = pd.to_datetime(list(confirmed_dates), utc=True)
    if missing.empty or not missing.isin(confirmed).all():
        return candles
    frame = candles.reindex(grid)
    frame["synthetic"] = grid.isin(missing)
    carry = frame.close.ffill()
    for column in ("open", "high", "low", "close"):
        frame.loc[missing, column] = carry.loc[missing]
    frame.loc[missing, "volume"] = 0.
    return frame


def audit_database(database, provider, as_of=None):
    """Audit all stored pairs with one cutoff and one consistent SQLite snapshot."""
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    with open_ohlcv(database) as conn:
        return pd.DataFrame([
            load_pair(conn, provider, symbol, timeframe, cutoff)[1]
            for symbol, timeframe in stored_pairs(conn, provider)
        ])
