"""CSV and SQLite persistence for the current timeframe-aware data layout."""
from contextlib import closing
from pathlib import Path
import sqlite3
from urllib.parse import quote, unquote
import pandas as pd
from timeframes import get_timeframe
from providers import validate_provider

COLUMNS = ["open", "high", "low", "close", "volume"]
SCHEMA = """CREATE TABLE IF NOT EXISTS ohlcv (
    provider TEXT NOT NULL, date TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY(provider, symbol, timeframe, date)
)"""


def normalize(df):
    """Use UTC-naive start timestamps; the last incoming value wins on overlaps."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index(kind="stable")
    return df.loc[~df.index.duplicated(keep="last"), COLUMNS]


def read_data(symbol, timeframe, folder, database, backend, strict=False, provider="polygon"):
    provider = validate_provider(provider)
    get_timeframe(timeframe)
    if backend not in ("csv", "sqlite", "both"):
        raise ValueError("Storage backend must be csv, sqlite, or both")
    frames = []
    if backend in ("csv", "both"):
        path = Path(folder)/timeframe/f"{quote(symbol, safe='')}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, index_col="date", parse_dates=["date"]))
    if backend in ("sqlite", "both"):
        if Path(database).exists():
            with closing(sqlite3.connect(database)) as conn:
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ohlcv'").fetchone()
                if exists:
                    frames.append(pd.read_sql_query(
                        "SELECT date, open, high, low, close, volume FROM ohlcv WHERE provider=? AND symbol=? AND timeframe=?",
                        conn, params=[provider, symbol, timeframe], parse_dates=["date"]
                    ).set_index("date"))
                elif strict:
                    raise RuntimeError("No ohlcv table found")
        elif strict:
            raise RuntimeError("OHLCV database does not exist")
    frames = [frame for frame in frames if not frame.empty]
    return normalize(pd.concat(frames)) if frames else pd.DataFrame()


def list_symbols(timeframe, folder, database, backend, provider="polygon"):
    provider = validate_provider(provider)
    get_timeframe(timeframe)
    if backend == "csv":
        return sorted(unquote(path.stem) for path in (Path(folder)/timeframe).glob("*.csv"))
    if backend == "sqlite":
        if not Path(database).exists():
            raise RuntimeError("OHLCV database does not exist")
        with closing(sqlite3.connect(database)) as conn:
            return [row[0] for row in conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv WHERE provider=? AND timeframe=? ORDER BY symbol", [provider, timeframe]
            )]
    raise ValueError("LOAD_FROM must be csv or sqlite")


def save_data(symbol, df, timeframe, folder, database, backend, provider="polygon"):
    provider = validate_provider(provider)
    get_timeframe(timeframe)
    if backend not in ("csv", "sqlite", "both"):
        raise ValueError("Storage backend must be csv, sqlite, or both")
    df = normalize(df)
    if backend in ("sqlite", "both"):
        with closing(sqlite3.connect(database)) as conn, conn:
            conn.execute(SCHEMA)
            rows = [(provider, str(date), symbol, timeframe, *[float(row[c]) for c in COLUMNS]) for date, row in df.iterrows()]
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv (provider,date,symbol,timeframe,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
    if backend in ("csv", "both"):
        target = Path(folder)/timeframe/f"{quote(symbol, safe='')}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = read_data(symbol, timeframe, folder, database, "csv", provider=provider)
        if not previous.empty:
            df = normalize(pd.concat([previous, df]))
        temporary = target.with_suffix(".csv.tmp")
        df.to_csv(temporary, index_label="date")
        temporary.replace(target)
