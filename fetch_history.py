"""Latest fetch observations and exclusion policies, persisted in the project database.

Only actual attempts update observations. Skipping preserves the previous attempt.
Missing bars are a saved trailing-staleness estimate, not an audit of internal gaps.
"""
from contextlib import closing
from pathlib import Path
import sqlite3
import json
import pandas as pd
from timeframes import get_timeframe

SETTINGS_SCHEMA = """CREATE TABLE IF NOT EXISTS provider_fetch_settings (
    provider TEXT PRIMARY KEY,
    max_missing_bars INTEGER CHECK(max_missing_bars >= 0),
    exclude_no_data INTEGER NOT NULL CHECK(exclude_no_data IN (0,1))
)"""
STATUS_SCHEMA = """CREATE TABLE IF NOT EXISTS fetch_status (
    provider TEXT NOT NULL, timeframe TEXT NOT NULL, symbol TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success','no_data','no_completed_data','error')),
    bars_returned INTEGER NOT NULL,
    latest_candle TEXT, missing_bars INTEGER, error TEXT,
    PRIMARY KEY(provider,timeframe,symbol)
)"""


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [name]).fetchone() is not None


def load_settings(database, provider, defaults=None):
    """Reading unset settings returns defaults without creating files or tables."""
    settings = {"max_missing_bars": None, "exclude_no_data": False, **(defaults or {})}
    if Path(database).exists():
        with closing(sqlite3.connect(database)) as conn:
            if _has_table(conn, "provider_fetch_settings"):
                row = conn.execute("SELECT max_missing_bars,exclude_no_data FROM provider_fetch_settings WHERE provider=?", [provider]).fetchone()
                if row:
                    settings = {"max_missing_bars": row[0], "exclude_no_data": bool(row[1])}
    return settings


def save_settings(database, provider, max_missing_bars=None, exclude_no_data=False):
    """Save a provider's policy; None disables the missing-bar exclusion."""
    if max_missing_bars is not None and (isinstance(max_missing_bars, bool) or not isinstance(max_missing_bars, int) or max_missing_bars < 0):
        raise ValueError("Missing-bar limit must be a nonnegative integer or None")
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute(SETTINGS_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO provider_fetch_settings VALUES (?,?,?)", [provider, max_missing_bars, int(bool(exclude_no_data))])


def load_statuses(database, provider, timeframe):
    """Return the latest attempt per coin, scoped to this provider and interval."""
    get_timeframe(timeframe)
    if not Path(database).exists():
        return {}
    with closing(sqlite3.connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        if not _has_table(conn, "fetch_status"):
            return {}
        rows = conn.execute("SELECT * FROM fetch_status WHERE provider=? AND timeframe=? ORDER BY symbol", [provider, timeframe]).fetchall()
        return {row["symbol"]: dict(row) for row in rows}


def record_attempt(database, provider, timeframe, symbol, attempted_at, status, bars_returned, latest_candle=None, missing_bars=None, error=None):
    """Replace one coin's latest observation without changing any OHLCV rows."""
    get_timeframe(timeframe)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute(STATUS_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO fetch_status VALUES (?,?,?,?,?,?,?,?,?)", [
            provider, timeframe, symbol, attempted_at, status, int(bars_returned),
            latest_candle, missing_bars, error,
        ])


def exclusion_reason(previous, settings):
    """Use the last actual attempt; equality to the limit is still eligible."""
    if not previous:
        return None
    if settings["exclude_no_data"] and previous["status"] == "no_data":
        return "Previous attempt returned no bars"
    limit = settings["max_missing_bars"]
    missing = previous["missing_bars"]
    if limit is not None and missing is not None and missing > limit:
        return f"Previous attempt had {missing} missing bars (limit {limit})"
    return None


UNIVERSE_SCHEMA = """CREATE TABLE IF NOT EXISTS provider_symbols (
    provider TEXT PRIMARY KEY, refreshed_at TEXT NOT NULL, symbols_json TEXT NOT NULL
)"""


def load_symbols(database, provider, now=None):
    """Reuse discovery only within its UTC calendar day, including across app restarts."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if not Path(database).exists():
        return None
    with closing(sqlite3.connect(database)) as conn:
        if not _has_table(conn, "provider_symbols"):
            return None
        row = conn.execute("SELECT refreshed_at,symbols_json FROM provider_symbols WHERE provider=?", [provider]).fetchone()
    if not row:
        return None
    refreshed = pd.Timestamp(row[0])
    if refreshed.normalize() != now.normalize() or refreshed > now:
        return None
    return json.loads(row[1])


def save_symbols(database, provider, symbols, now=None):
    """Replace a discovery snapshot only after a successful, nonempty provider response."""
    if not symbols:
        raise ValueError("Cannot cache an empty symbol universe")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute(UNIVERSE_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO provider_symbols VALUES (?,?,?)", [provider,now.isoformat(),json.dumps(symbols,sort_keys=True)])
