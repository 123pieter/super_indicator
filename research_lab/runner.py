"""Validate portable research strategy configs and build their close-based signal."""
import csv
from dataclasses import asdict
import hashlib
import json
import math
from uuid import uuid4
from pathlib import Path

import pandas as pd

from providers import PROVIDERS
from .data import load_pair, open_ohlcv, stored_pairs
from .engine import Execution, TRADE_COLUMNS, simulate
from .strategies import bollinger_reentry, sma_trend

RULES = {"sma_trend": sma_trend, "bollinger_reentry": bollinger_reentry}
METHODS = ("majority", "weighted", "and", "or")
BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "research_configs" / "baseline.json"


def _object(value, allowed, label):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError(f"{label} must be an object with keys from: {', '.join(allowed)}")
    return value


def _number(value, label):
    if type(value) in (int, float):
        try:
            if math.isfinite(value):
                return float(value)
        except OverflowError:
            pass
    raise ValueError(f"{label} must be a finite number")


def validate_config(config):
    """Return a fully specified config; reject typos and invalid risk assumptions."""
    _object(config, ("version", "name", "provider", "components", "combination", "execution"), "Config")
    if type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Config version must be 1")
    name, provider = config.get("name"), config.get("provider")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Strategy name is required")
    if provider not in PROVIDERS:
        raise ValueError("Select one stored data provider")
    items = config.get("components")
    if not isinstance(items, list) or not items:
        raise ValueError("Select at least one indicator component")
    components = []
    for item in items:
        _object(item, ("rule", "window", "deviations", "weight"), "Component")
        rule = item.get("rule")
        if rule not in tuple(RULES):
            raise ValueError(f"Unknown indicator rule: {rule}")
        if any(c["rule"] == rule for c in components):
            raise ValueError("Each rule can appear only once in a strategy")
        window = item.get("window", 100 if rule == "sma_trend" else 20)
        if type(window) is not int or window < 2:
            raise ValueError("Indicator windows must be integer candle counts of at least 2")
        weight = _number(item.get("weight", 1.0), "Weight")
        if weight <= 0:
            raise ValueError("Weights must be positive")
        component = dict(rule=rule, window=window, weight=weight)
        if rule == "bollinger_reentry":
            deviations = _number(item.get("deviations", 2.0), "Band deviations")
            if deviations <= 0:
                raise ValueError("Band deviations must be positive")
            component["deviations"] = deviations
        elif "deviations" in item:
            raise ValueError("Band deviations apply only to Bollinger re-entry")
        components.append(component)

    combination = _object(config.get("combination", {}), ("method", "threshold"), "Combination")
    method = combination.get("method", "majority")
    if method not in METHODS:
        raise ValueError("Combination method must be one of: majority, weighted, and, or")
    threshold = _number(combination.get("threshold", .5), "Weighted threshold")
    if not 0 < threshold <= 1:
        raise ValueError("Weighted threshold must be in (0, 1]")
    if not math.isfinite(sum(c["weight"] for c in components)):
        raise ValueError("Sum of weights must be finite")

    defaults = asdict(Execution())
    execution = _object(config.get("execution", {}), tuple(defaults), "Execution")
    for key, value in execution.items():
        if key != "sizing" and not (key in ("stop_atr", "target_atr") and value is None):
            _number(value, key)
    if execution.get("sizing", defaults["sizing"]) not in ("fixed", "atr_risk"):
        raise ValueError("Sizing must be fixed or atr_risk")
    settings = Execution(**(defaults | execution))
    return dict(version=1, name=name.strip(), provider=provider, components=components,
                combination=dict(method=method, threshold=threshold), execution=asdict(settings))


def load_config(path):
    """Read JSON without modifying it."""
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def build_signal(close, config, *, cache=None):
    """Combine close states; an optional cache is local to one candle history."""
    cfg = validate_config(config)
    components = cfg["components"]
    cache = {} if cache is None else cache
    states = {}
    for component in components:
        rule = component["rule"]
        params = {key: component[key] for key in ("window", "deviations") if key in component}
        key = (rule, tuple(params.items()))
        if key not in cache:
            cache[key] = RULES[rule](close, **params)
        states[rule] = cache[key]
    votes = pd.DataFrame(states, index=close.index)
    method = cfg["combination"]["method"]
    if method == "weighted":
        weights = pd.Series({c["rule"]: c["weight"] for c in components})
        score = votes.mul(weights).sum(axis=1)
        threshold = cfg["combination"]["threshold"] * weights.sum()
        long, short = score.ge(threshold), score.le(-threshold)
    elif method == "majority":
        long = votes.eq(1).sum(axis=1).gt(len(components)/2)
        short = votes.eq(-1).sum(axis=1).gt(len(components)/2)
    elif method == "and":
        long, short = votes.eq(1).all(axis=1), votes.eq(-1).all(axis=1)
    else:
        long, short = votes.eq(1).any(axis=1), votes.eq(-1).any(axis=1)
    # OR with opposing votes, ties, and insufficient support all produce flat.
    return (long.astype("int8") - short.astype("int8")).rename("combined")

def atr14(candles):
    """True range with an arithmetic seed, followed by Wilder smoothing."""
    previous = candles.close.shift()
    ranges = pd.concat([candles.high-candles.low, (candles.high-previous).abs(),
                        (candles.low-previous).abs()], axis=1).max(axis=1)
    seed = ranges.iloc[:14].mean()
    ranges.iloc[:13] = float("nan")
    if len(ranges) >= 14:
        ranges.iloc[13] = seed
    return ranges.ewm(alpha=1/14, adjust=False).mean()


def _experiments(configs):
    """Pair each cost/capital assumption with fully invested comparison rules."""
    experiments, benchmarks = [], {}
    for i, cfg in enumerate(configs, 1):
        execution = Execution(**cfg["execution"])
        cost_key = (execution.initial_capital, execution.fee, execution.slippage)
        if cost_key not in benchmarks:
            group = len(benchmarks)+1
            benchmarks[cost_key] = group
        else:
            group = benchmarks[cost_key]
        experiments.append(dict(id=f"strategy:{i}", name=cfg["name"], kind="strategy",
                                group=group, execution=execution, config=cfg))
    for (capital, fee, slip), group in benchmarks.items():
        settings = Execution(initial_capital=capital, fee=fee, slippage=slip, fraction=1.,
                             stop_atr=None, target_atr=None)
        for rule, name in (("buy_hold", "Buy and hold"), ("sma100", "SMA100 long/cash")):
            experiments.append(dict(id=f"benchmark:{group}:{rule}", name=name, kind=rule,
                                    group=group, execution=settings, config=None))
    return experiments


def run_batch(configs, database, output_root, *, symbol=None, timeframe=None, as_of=None,
              on_progress=None):
    """Run one provider's grid, streaming outputs instead of retaining all equity in memory.

    Each pair shares the longest selected indicator/ATR/benchmark warmup.
    Pairs with audit problems are reported as skipped; no gaps are bridged.
    Results use all eligible history and are descriptive, not holdout tests.
    """
    configs = [validate_config(cfg) for cfg in configs]
    if not configs or len({cfg["name"] for cfg in configs}) != len(configs):
        raise ValueError("Supply at least one config and give each strategy a unique name")
    providers = {cfg["provider"] for cfg in configs}
    if len(providers) != 1:
        raise ValueError("All configs in a batch must select the same provider")
    provider = providers.pop()
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    if pd.isna(cutoff):
        raise ValueError("Batch cutoff must be a valid timestamp")
    experiments = _experiments(configs)
    warmup = max(100, 14, *(c["window"] + (c["rule"] == "bollinger_reentry")
                            for cfg in configs for c in cfg["components"]))
    settings_count = len({json.dumps({k:v for k,v in cfg.items() if k != "name"}, sort_keys=True)
                          for cfg in configs})
    coverage, counts = [], {"completed": 0, "skipped": 0, "failed": 0}
    columns = ["provider", "symbol", "timeframe", "experiment"]
    result_columns = columns + ["status", "message", "first_execution", "last_candle",
                                "trades", "final_equity", "total_return_pct"]
    with open_ohlcv(database) as conn:
        pairs = [(asset, interval) for asset, interval in stored_pairs(conn, provider)
                 if (symbol is None or asset == symbol) and (timeframe is None or interval == timeframe)]
        if not pairs:
            raise ValueError("No stored asset/timeframe pairs match the batch")
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        # Ordinary directories inherit the project's Windows ACL. Python's
        # mkdtemp deliberately creates a private directory, hiding Codex runs
        # from the account that opens the project in PyCharm.
        folder = root / (cutoff.strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex)
        folder.mkdir(exist_ok=False)
        with (folder/"results.csv").open("w", newline="", encoding="utf-8") as summary, (
                folder/"trades.csv").open("w", newline="", encoding="utf-8") as trades, (
                folder/"equity.csv").open("w", newline="", encoding="utf-8") as equity:
            result_writer = csv.DictWriter(summary, fieldnames=result_columns)
            trade_writer = csv.DictWriter(trades, fieldnames=columns+TRADE_COLUMNS)
            equity_writer = csv.DictWriter(equity, fieldnames=columns+["candle", "equity"])
            for writer in (result_writer, trade_writer, equity_writer):
                writer.writeheader()
            for i, (asset, interval) in enumerate(pairs, 1):
                candles, audit = load_pair(conn, provider, asset, interval, cutoff)
                audit["data_sha256"] = hashlib.sha256(
                    pd.util.hash_pandas_object(candles, index=True).to_numpy().tobytes()
                ).hexdigest()
                coverage.append(audit)
                problem = ""
                if audit["status"] != "ok":
                    problem = "Data audit requires review"
                elif len(candles) <= warmup:
                    problem = f"Need {warmup} warmup candles and at least one execution candle"
                if not problem:
                    atr = atr14(candles)
                    # Keep the last warmup close so the first eligible open can trade it.
                    window = candles.iloc[warmup-1:]
                    baseline = sma_trend(candles.close)
                    trend = baseline.clip(lower=0)
                    signal_cache = {("sma_trend", (("window", 100),)): baseline}
                for exp in experiments:
                    key = dict(provider=provider, symbol=asset, timeframe=interval, experiment=exp["id"])
                    if problem:
                        result_writer.writerow(dict(key, status="skipped", message=problem))
                        counts["skipped"] += 1
                        continue
                    try:
                        if exp["kind"] == "strategy":
                            signal = build_signal(candles.close, exp["config"], cache=signal_cache)
                        elif exp["kind"] == "buy_hold":
                            signal = pd.Series(1, index=candles.index, dtype="int8")
                        else:
                            signal = trend
                        result = simulate(window, signal.loc[window.index], atr.loc[window.index],
                                          timeframe=interval, settings=exp["execution"])
                    except ValueError as exc:
                        result_writer.writerow(dict(key, status="failed", message=str(exc)))
                        counts["failed"] += 1
                        continue
                    final = result["equity"].iloc[-1]
                    result_writer.writerow(dict(key, status="completed", first_execution=window.index[1],
                                                last_candle=window.index[-1], trades=len(result["trades"]),
                                                final_equity=final,
                                                total_return_pct=(final/exp["execution"].initial_capital-1)*100))
                    trade_writer.writerows(dict(key, **row) for row in result["trades"].to_dict("records"))
                    equity_writer.writerows(dict(key, candle=date, equity=value)
                                            for date, value in result["equity"].items())
                    counts["completed"] += 1
                if on_progress is not None:
                    on_progress(i, len(pairs), asset, interval)

    manifest = dict(
        version=1, provider=provider, as_of=str(cutoff), database=str(Path(database).resolve()),
        evaluation="historical descriptive; no holdout or parameter selection",
        independent_capital_per_pair=True, shorts="hypothetical; excludes funding, borrow and venue liquidation",
        warmup_candles=warmup, asset_timeframe_pairs=len(pairs),
        strategy_configs=len(configs), unique_strategy_settings=settings_count,
        benchmark_configs=len(experiments)-len(configs),
        planned_strategy_jobs=len(pairs)*len(configs), planned_jobs=len(pairs)*len(experiments),
        jobs=counts, experiments=[dict(exp, execution=asdict(exp["execution"])) for exp in experiments],
        coverage=coverage,
    )
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n", encoding="utf-8")
    return folder, manifest
