"""Fetch bars for a selected timeframe with overlapping incremental updates and atomic persistence."""
import argparse
import time
import pandas as pd
import config
import storage
import fetch_history
from timeframes import TIMEFRAMES, get_timeframe, completed_bars, missing_bars
from providers import get_provider, validate_provider, PROVIDERS


def read_existing(symbol, timeframe=config.DEFAULT_TIMEFRAME, provider=None):
    folder, database = config.storage_paths(provider or config.ACTIVE_PROVIDER)
    return storage.read_data(symbol, timeframe, folder, database, config.SAVE_TO, provider=provider or config.ACTIVE_PROVIDER)


def save_data(symbol, df, timeframe=config.DEFAULT_TIMEFRAME, provider=None):
    folder, database = config.storage_paths(provider or config.ACTIVE_PROVIDER)
    storage.save_data(symbol, df, timeframe, folder, database, config.SAVE_TO, provider=provider or config.ACTIVE_PROVIDER)


def utc_now():
    """One clock boundary shared by discovery and freshness planning."""
    return pd.Timestamp.now(tz="UTC")


def current_local_data(symbol, timeframe, now, provider=None):
    """Return (current, locally_synced), checking actual candles rather than attempt metadata."""
    frame = read_existing(symbol, timeframe, provider)
    if frame.empty:
        return False, False
    frame = completed_bars(frame, timeframe, now)
    if frame.empty or missing_bars(frame.index.max(), timeframe, now) != 0:
        return False, False
    # If only one of the requested destinations is current, repair the other locally.
    if config.SAVE_TO == "both":
        folder, database = config.storage_paths(provider or config.ACTIVE_PROVIDER)
        for backend in ("csv", "sqlite"):
            saved = storage.read_data(symbol, timeframe, folder, database, backend, provider=provider or config.ACTIVE_PROVIDER)
            saved = completed_bars(saved, timeframe, now) if not saved.empty else saved
            if saved.empty or missing_bars(saved.index.max(), timeframe, now) != 0:
                save_data(symbol, frame, timeframe, provider)
                return True, True
    return True, False


def main(limit=config.FETCH_LIMIT, timeframe=config.DEFAULT_TIMEFRAME, on_progress=None, ignore_exclusions=False, force_refresh=False, provider=None):
    def report(**event):
        if on_progress is not None:
            on_progress(event)

    interval = get_timeframe(timeframe)
    if limit is not None and limit < 0:
        raise ValueError("Fetch limit must be nonnegative")
    if config.SAVE_TO not in ("csv", "sqlite", "both"):
        raise ValueError("SAVE_TO must be csv, sqlite, or both")
    provider_name = validate_provider(provider or config.ACTIVE_PROVIDER)
    policy = fetch_history.load_settings(config.DB_PATH, provider_name, config.PROVIDER_FETCH_DEFAULTS.get(provider_name))
    previous = fetch_history.load_statuses(config.DB_PATH, provider_name, timeframe)
    now = utc_now()
    provider = None
    def client():
        nonlocal provider
        if provider is None:
            api_key = (config.POLYGON_API_KEY or config.load_api_key()) if provider_name == "polygon" else None
            provider = get_provider(provider_name, api_key)
        return provider

    symbols = None if force_refresh else fetch_history.load_symbols(config.DB_PATH, provider_name, now)
    discovery_cached = symbols is not None
    if symbols is None:
        symbols = client().get_available_symbols()
        if not symbols:
            raise RuntimeError("Provider returned no USD pairs")
        fetch_history.save_symbols(config.DB_PATH, provider_name, symbols, now)
    universe = sorted(symbols.items())
    tickers, skipped, current = [], [], []
    local_synced = 0
    for symbol, ticker in universe:
        try:
            if not force_refresh:
                is_current, synced = current_local_data(symbol, timeframe, now, provider_name)
                if is_current:
                    current.append(symbol)
                    local_synced += synced
                    continue
        except Exception:
            # Let the normal per-coin path report and persist unreadable storage as an error.
            pass
        reason = None if ignore_exclusions else fetch_history.exclusion_reason(previous.get(symbol), policy)
        if reason:
            skipped.append((symbol, reason))
        else:
            tickers.append((symbol, ticker))
    # Testing limits apply to eligible coins, not coins that are excluded already.
    if limit:
        tickers = tickers[:limit]
    total = len(tickers) + len(skipped) + len(current)
    failures = empty_responses = 0
    report(event="started", total=total, completed=0, discovery_cached=discovery_cached)
    for i, symbol in enumerate(current, 1):
        print(f"CURRENT {symbol}: latest completed candle already stored", flush=True)
        report(event="current", symbol=symbol, reason="Latest completed candle already stored", completed=i, total=total)
    for i, (symbol, reason) in enumerate(skipped, len(current)+1):
        print(f"SKIPPED {symbol}: {reason}", flush=True)
        report(event="skipped", symbol=symbol, reason=reason, completed=i, total=total)
    today = now.normalize()
    end = (today - pd.Timedelta(days=1) if timeframe == "1d" else today).strftime("%Y-%m-%d")
    print(f"Fetching {len(tickers)} {timeframe} pairs through {end}; {len(skipped)} excluded", flush=True)
    for i, (symbol, ticker) in enumerate(tickers, len(current)+len(skipped)+1):
        report(event="fetching", symbol=symbol, completed=i-1, total=total)
        time.sleep(config.REQUEST_DELAY if provider_name == "polygon" else config.HYPERLIQUID_REQUEST_DELAY)
        known = pd.DataFrame()
        returned = 0
        status, error = "success", None
        try:
            existing = read_existing(symbol, timeframe, provider_name)
            known = completed_bars(existing, timeframe, utc_now()) if not existing.empty else existing
            start = config.START_DATE
            if not existing.empty:
                start = max(pd.Timestamp(start), existing.index.max() - 2 * interval.duration).strftime("%Y-%m-%d")
            print(f"[{i}/{total}] {symbol}: {start} to {end}", flush=True)
            new = client().fetch_ohlcv(symbol, ticker, start, end, timeframe=timeframe)
            if new is None or new.empty:
                status, error = "no_data", "No bars returned; stored data retained"
            else:
                returned = len(new)
                new = completed_bars(storage.normalize(new), timeframe, utc_now())
                if new.empty:
                    status, error = "no_completed_data", "Only unfinished candles returned; stored data retained"
                else:
                    merged = storage.normalize(pd.concat([known, new])) if not known.empty else new
                    save_data(symbol, merged, timeframe, provider_name)
                    known = merged
        except Exception as exc:
            status, error = "error", str(exc)
        observed_at = utc_now()
        latest = known.index.max() if not known.empty else None
        # Keep earlier known coverage when a read or request fails before it can be observed.
        latest_text = latest.isoformat() if latest is not None else previous.get(symbol, {}).get("latest_candle")
        missing = missing_bars(latest_text, timeframe, observed_at)
        fetch_history.record_attempt(config.DB_PATH, provider_name, timeframe, symbol,
                                     observed_at.isoformat(), status, returned, latest_text, missing, error)
        if status == "success":
            print(f"  Saved {len(known)} bars", flush=True)
            report(event="success", symbol=symbol, bars=len(known), returned=returned, missing_bars=missing, completed=i, total=total)
        else:
            failures += 1
            empty_responses += status == "no_data"
            print(f"  ERROR {symbol}: {error}", flush=True)
            report(event=status, symbol=symbol, error=error, returned=returned, missing_bars=missing, completed=i, total=total)
    succeeded = len(tickers)-failures
    print(f"Finished: {succeeded} succeeded, {failures} failed, {len(skipped)} excluded, {len(current)} already current", flush=True)
    report(event="finished", total=total, completed=total, succeeded=succeeded,
           failed=failures, skipped=len(skipped), current=len(current), no_data=empty_responses,
           fetched=len(tickers), local_synced=local_synced, discovery_cached=discovery_cached)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.FETCH_LIMIT, help="0 means all pairs")
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--ignore-exclusions", action="store_true", help="Retry previously excluded coins for this run")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh discovery and refetch current candles; exclusions still apply")
    parser.add_argument("--provider", choices=PROVIDERS, default=config.ACTIVE_PROVIDER)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be nonnegative")
    raise SystemExit(main(args.limit, args.timeframe, ignore_exclusions=args.ignore_exclusions, force_refresh=args.force_refresh, provider=args.provider))
