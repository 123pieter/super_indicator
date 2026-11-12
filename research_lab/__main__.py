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


def check_signals(args, configs):
    """Inspect close-based states for one audited pair, without running trades."""
    from .runner import build_signal

    with open_ohlcv(args.database) as conn:
        frame, report = load_pair(conn, args.provider, args.symbol, args.timeframe, args.as_of)
    if report["status"] != "ok":
        print(f"Signals unavailable: {report}", file=sys.stderr)
        return 1
    print(f"{args.provider} / {args.symbol} / {args.timeframe}: {len(frame)} usable candles")
    print("States at candle close: -1 short, 0 flat (including warmup), +1 long. Counts are candles, not trades.")
    cache = {}
    for cfg in configs:
        signal = build_signal(frame, cfg, cache=cache)
        counts = signal.value_counts().reindex([-1, 0, 1], fill_value=0)
        print(f"{cfg['name']}: {counts[-1]} short, {counts[0]} flat, {counts[1]} long")
        print(frame[["close"]].assign(combined=signal).tail(10).to_string())
    return 0


def workbook_strategies(args):
    """Load the project workbook by default, with optional exact name filtering."""
    from .workbook import WORKBOOK_PATH, load_workbook_configs

    configs = load_workbook_configs(args.workbook or WORKBOOK_PATH, args.provider)
    if args.strategy_name:
        selected = set(args.strategy_name)
        missing = selected - {cfg["name"] for cfg in configs}
        if missing:
            raise ValueError("Unknown workbook strategy names: " + ", ".join(sorted(missing)))
        configs = [cfg for cfg in configs if cfg["name"] in selected]
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-data", "demo-engine", "check-signals", "check-workbook", "run-batch", "report"])
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--workbook", help="Strategy workbook path (defaults to the project workbook)")
    parser.add_argument("--strategy-name", action="append", help="Exact workbook strategy name; repeat to select several")
    parser.add_argument("--run", help="Completed batch folder for report")
    parser.add_argument("--experiment", help="Experiment ID to inspect in the report")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe", choices=TIMEFRAMES)
    parser.add_argument("--database", default=DB_PATH)
    parser.add_argument("--latest-continuous", action="store_true",
                        help="For a batch, use the latest uninterrupted history after internal gaps")
    parser.add_argument("--fill-confirmed-gaps", action="store_true",
                        help="For a batch, carry the close across confirmed empty intervals")
    parser.add_argument("--evaluation-start", help="Split daily batch at this UTC date")
    parser.add_argument("--as-of", help="Optional UTC audit cutoff (ISO datetime)")
    args = parser.parse_args(argv)
    if args.command == "demo-engine":
        return demo()
    if args.command != "report" and args.provider is None:
        parser.error(f"{args.command} requires --provider")
    if args.command == "check-signals" and (not args.symbol or not args.timeframe):
        parser.error("check-signals requires --symbol and --timeframe")
    if args.command == "report" and not args.run:
        parser.error("report requires --run")
    try:
        if args.command == "report":
            from .report import generate_report
            print(generate_report(args.run, symbol=args.symbol, timeframe=args.timeframe,
                                  experiment=args.experiment))
            return 0
        if args.command in ("check-workbook", "check-signals", "run-batch"):
            configs = workbook_strategies(args)
            if args.command == "check-workbook":
                for cfg in configs:
                    print(f"{cfg['name']}: {len(cfg['components'])} components, {cfg['combination']['method']}")
                print(f"{len(configs)} strategies validated for {args.provider}.")
                return 0
            if args.command == "check-signals":
                return check_signals(args, configs)
            from .runner import RUNS_DIR, run_batch, run_split
            options = dict(symbol=args.symbol, timeframe=args.timeframe, as_of=args.as_of,both_views=True,
                           latest_continuous=args.latest_continuous, fill_confirmed_gaps=args.fill_confirmed_gaps,
                           on_progress=lambda i,n,s,t: print(f"[{i}/{n}] {s} / {t}", flush=True))
            if args.evaluation_start:
                options["timeframe"] = args.timeframe or "1d"
                saved = run_split(configs, args.database, RUNS_DIR,
                                  evaluation_start=args.evaluation_start, **options)
            else:
                saved = run_batch(configs, args.database, RUNS_DIR, **options)
            for folder, manifest in saved:
                print(f"{manifest.get('phase', 'Historical batch')} / {manifest['window_view']}: {manifest['jobs']}")
                print(f"Outputs: {folder}")
            return int(any(m["jobs"]["failed"] or m["jobs"]["skipped"] for _,m in saved))
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
