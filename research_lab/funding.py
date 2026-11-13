"""Load verified perpetual settlement rates on an OHLCV candle grid."""
import math
import sqlite3

import numpy as np
import pandas as pd

from timeframes import get_timeframe


HOUR = pd.Timedelta(hours=1)
HOUR_MS = 3_600_000


def _milliseconds(timestamp):
    return timestamp.value // 1_000_000


def load_candle_funding(conn, provider, symbol, index, timeframe):
    """Sum signed, positive and negative rates per [candle start, candle end).

    Raw event milliseconds determine the candle receiving a rate. UTC hours are
    used only to verify the settlement schedule.
    """
    if provider not in ("hyperliquid", "binance"):
        raise ValueError(f"Funding data are unavailable for provider {provider}")
    duration = get_timeframe(timeframe).duration
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("Funding requires a nonempty DatetimeIndex")
    utc = pd.to_datetime(index, utc=True)
    if utc.hasnans or not utc.equals(utc.floor(duration)) or not (utc[1:] - utc[:-1] == duration).all():
        raise ValueError("Funding requires consecutive candles aligned to their UTC timeframe")

    start, end = utc[0], utc[-1] + duration
    first_hour, end_hour = start.floor(HOUR), end.ceil(HOUR)
    prefix = f"Funding coverage for {provider}/{symbol} ({start.isoformat()} to {end.isoformat()})"
    # Binance no-settlement bars need observations bracketing the requested window.
    padding = pd.Timedelta(days=1) + HOUR if provider == "binance" else pd.Timedelta(0)
    try:
        rows = conn.execute(
            """SELECT time_ms, funding_rate, interval_hours FROM funding_rates
               WHERE provider=? AND symbol=? AND time_ms>=? AND time_ms<? ORDER BY time_ms""",
            (provider, symbol, _milliseconds(first_hour - padding), _milliseconds(end_hour + padding)),
        ).fetchall()
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            raise
        raise ValueError(f"{prefix}: no funding rates have been imported") from error
    if not rows:
        raise ValueError(f"{prefix}: no settlement events stored")

    event_hours = [time_ms // HOUR_MS for time_ms, _, _ in rows]
    first_bucket = _milliseconds(first_hour) // HOUR_MS
    end_bucket = _milliseconds(end_hour) // HOUR_MS
    if provider == "hyperliquid":
        if (event_hours != list(range(first_bucket, end_bucket))
                or any(interval != 1 for _, _, interval in rows)):
            raise ValueError(f"{prefix}: expected one verified settlement in every UTC hour")
    else:
        left = next((i for i in range(len(rows) - 1, -1, -1) if event_hours[i] <= first_bucket), None)
        right = next((i for i, hour in enumerate(event_hours) if hour >= end_bucket), None)
        if left is None or right is None or left >= right:
            raise ValueError(f"{prefix}: missing Binance settlements bracketing the requested range")
        if ((left and event_hours[left - 1] == event_hours[left])
                or (right + 1 < len(rows) and event_hours[right + 1] == event_hours[right])):
            raise ValueError(f"{prefix}: duplicate Binance settlements at a coverage boundary")
        # The interval on an arriving settlement describes the elapsed funding period.
        # Reject unexplained gaps, duplicate hour buckets, and off-grid schedule changes.
        for i in range(left + 1, right + 1):
            interval = rows[i][2]
            if (interval is None or not math.isfinite(interval) or not 0 < interval <= 24
                    or event_hours[i] - event_hours[i - 1] != interval):
                when = pd.Timestamp(event_hours[i] * HOUR_MS, unit="ms", tz="UTC")
                raise ValueError(f"{prefix}: incomplete or irregular Binance settlement schedule at {when.isoformat()}")

    start_ms, end_ms = _milliseconds(start), _milliseconds(end)
    duration_ms = duration // pd.Timedelta(milliseconds=1)
    rates = np.zeros((len(index), 2))
    for time_ms, rate, _ in rows:
        if start_ms <= time_ms < end_ms:
            if not math.isfinite(rate):
                raise ValueError(f"{prefix}: non-finite funding rate")
            candle = (time_ms - start_ms) // duration_ms
            rates[candle, int(rate < 0)] += rate
    frame = pd.DataFrame(rates, index=index, columns=["positive_rate", "negative_rate"])
    return frame.assign(signed_rate=frame.positive_rate + frame.negative_rate)[
        ["signed_rate", "positive_rate", "negative_rate"]
    ]


def funding_coverage(conn, providers):
    """List stored funding endpoints; run-level checks still prove continuity."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='funding_rates'").fetchone():
        return pd.DataFrame()
    selected = tuple(providers)
    if not selected:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in selected)
    frame = pd.read_sql_query(
        f"""SELECT provider AS Provider, symbol AS Asset, count(*) AS Settlements,
                   min(time_ms) AS first_ms, max(time_ms) AS last_ms
            FROM funding_rates WHERE provider IN ({placeholders})
            GROUP BY provider, symbol ORDER BY provider, symbol""", conn, params=selected
    )
    frame["First settlement (UTC)"] = pd.to_datetime(frame.pop("first_ms"), unit="ms", utc=True)
    frame["Last settlement (UTC)"] = pd.to_datetime(frame.pop("last_ms"), unit="ms", utc=True)
    return frame
