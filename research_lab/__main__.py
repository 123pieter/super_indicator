"""Audit stored candles, inspect signals, or run research simulations."""
import argparse
import sqlite3
import sys

from config import DB_PATH
from providers import PROVIDERS
from timeframes import TIMEFRAMES
from .data import audit_database, load_pair, open_ohlcv


def demo():
    """One synthetic trade with an ambiguous candle and hand-calculated P&L."""
    import pandas as pd
    from .engine import Execution, simulate

    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    candles = pd.DataFrame(
        [(100, 101, 99, 100, 1), (100, 104, 97, 101, 1), (101, 102, 100, 101, 1)],
        columns=["open", "high", "low", "close", "volume"], index=index)
    settings = Execution(fee=0, slippage=0, sizing="atr_risk", target_atr=4)
    result = simulate(candles, pd.Series([1, 1, 1], index=index),
                      pd.Series(1.0, index=index), settings=settings)
    print("Signal Jan 1 close; entry Jan 2 open at 100. Stop 98; target 104.")
    print("Both levels touched on Jan 2. Expected: stop first, 5 units, -10 P&L, 990 equity.")
    print(result["trades"].to_string(index=False))
    print(f"Final equity: {result['equity'].iloc[-1]:.2f}")
    return 0


def check_signals(args):
    """Inspect close-based states for one audited pair, without running trades."""
    from .strategies import bollinger_reentry, sma_trend

    cfg = None
    if args.config:
        from .runner import build_signal, load_config
        cfg = load_config(args.config)
        if args.provider is not None and args.provider != cfg["provider"]:
            raise ValueError("--provider does not match the config provider")
    provider = cfg["provider"] if cfg else args.provider
    with open_ohlcv(args.database) as conn:
        frame, report = load_pair(conn, provider, args.symbol, args.timeframe, args.as_of)
    if report["status"] != "ok":
        print(f"Signals unavailable: {report}", file=sys.stderr)
        return 1
    if cfg:
        view = frame[["close"]].assign(combined=build_signal(frame.close, cfg))
        print(f"Config: {cfg['name']} ({cfg['combination']['method']})")
    else:
        view = frame[["close"]].assign(sma100=sma_trend(frame.close), bb_reentry=bollinger_reentry(frame.close))
    print(f"{provider} / {args.symbol} / {args.timeframe}: {len(view)} usable candles")
    print("States at candle close: -1 short, 0 flat (including warmup), +1 long. Counts are candles, not trades.")
    for name in view.columns[1:]:
        counts = view[name].value_counts().reindex([-1, 0, 1], fill_value=0)
        print(f"{name}: {counts[-1]} short, {counts[0]} flat, {counts[1]} long")
    print(view.tail(10).to_string())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-data", "demo-engine", "check-signals", "check-config", "run-batch", "report"])
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--config", help="Path to a research strategy JSON file")
    parser.add_argument("--strategy", action="append", metavar="JSON",
                        help="Strategy config for run-batch; repeat to run several")
    parser.add_argument("--run", help="Completed batch folder for report")
    parser.add_argument("--experiment", help="Experiment ID to inspect in the report")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe", choices=TIMEFRAMES)
    parser.add_argument("--database", default=DB_PATH)
    parser.add_argument("--as-of", help="Optional UTC audit cutoff (ISO datetime)")
    args = parser.parse_args(argv)
    if args.command == "demo-engine":
        return demo()
    if args.command == "check-config" and not args.config:
        parser.error("check-config requires --config")
    if args.command in ("check-data", "check-signals") and args.provider is None and (
            args.command == "check-data" or not args.config):
        parser.error(f"{args.command} requires --provider (or --config for check-signals)")
    if args.command == "check-signals" and (not args.symbol or not args.timeframe):
        parser.error("check-signals requires --symbol and --timeframe")
    if args.command == "run-batch" and not (args.strategy or args.config):
        parser.error("run-batch requires --strategy JSON (repeatable) or --config")
    if args.command == "report" and not args.run:
        parser.error("report requires --run")
    try:
        if args.command == "report":
            from .report import generate_report
            print(generate_report(args.run, symbol=args.symbol, timeframe=args.timeframe,
                                  experiment=args.experiment))
            return 0
        if args.command == "run-batch":
            from .runner import BASELINE_CONFIG, load_config, run_batch
            paths = ([args.config] if args.config else []) + (args.strategy or [])
            configs = [load_config(path) for path in paths]
            if args.provider is not None and any(cfg["provider"] != args.provider for cfg in configs):
                raise ValueError("--provider does not match the config provider")
            folder, manifest = run_batch(
                configs, args.database, BASELINE_CONFIG.parents[1]/"research_runs",
                symbol=args.symbol, timeframe=args.timeframe, as_of=args.as_of,
                on_progress=lambda i,n,s,t: print(f"[{i}/{n}] {s} / {t}", flush=True),
            )
            print(f"Historical batch: {manifest['jobs']}")
            print(f"Outputs: {folder}")
            return int(bool(manifest["jobs"]["failed"] or manifest["jobs"]["skipped"]))
        if args.command == "check-config":
            import json
            from .runner import load_config
            print(json.dumps(load_config(args.config), indent=2))
            return 0
        if args.command == "check-signals":
            return check_signals(args)
        report = audit_database(args.database, args.provider, args.as_of)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Research check failed: {exc}", file=sys.stderr)
        return 2
    if report.empty:
        print(f"No stored OHLCV pairs for {args.provider}.", file=sys.stderr)
        return 2
    issues = report.status.ne("ok")
    print(f"{args.provider}: {len(report)} asset/timeframe pairs; {issues.sum()} require review.")
    print(report.fillna("--").to_string(index=False))
    print("Internal missing bars are gaps between usable candles; trailing missing bars follow the last usable candle.")
    print("Issue counts can overlap. Unfinished candles are excluded; trailing lag alone does not fail the audit.")
    return int(issues.any())


if __name__ == "__main__":
    raise SystemExit(main())
