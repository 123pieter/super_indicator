"""Validate portable research strategy configs and build their close-based signal."""
from contextlib import ExitStack, nullcontext
import csv
from dataclasses import asdict
import hashlib
import json
import math
from uuid import uuid4
from pathlib import Path

import pandas as pd

from config import CONFIRMED_EMPTY_CANDLES
from providers import PROVIDERS
from timeframes import TIMEFRAMES
from .data import audit_problem, backtest_history, fill_empty_candles, load_pair, open_ohlcv, stored_pairs
from .engine import Execution, TRADE_COLUMNS, simulate
from .filters import entry_mask, filter_warmup, validate_filters
from .rules import REGISTRY
from .strategies import _atr, sma_trend

RULES = {name: rule.function for name, rule in REGISTRY.items()}
METHODS = ("majority", "weighted", "and", "or")
RUNS_DIR = Path(__file__).resolve().parents[1] / "research_runs"


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


def validate_component(item):
    """Normalize one indicator instance using its registered parameter schema."""
    if not isinstance(item, dict) or item.get("rule") not in tuple(REGISTRY):
        raise ValueError(f"Unknown indicator rule: {item.get('rule') if isinstance(item, dict) else item}")
    name = item["rule"]
    rule = REGISTRY[name]
    _object(item, ("rule", "weight", "indicator_number", "indicator_name", "category", *rule.parameters), "Component")
    weight = _number(item.get("weight", 1.), "Weight")
    if weight <= 0:
        raise ValueError("Weights must be positive")
    params = {}
    for key, spec in rule.parameters.items():
        value = item.get(key, spec.default)
        if type(spec.default) is int:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        else:
            value = _number(value, key)
        if value < spec.minimum or (spec.exclusive and value == spec.minimum):
            operator = "greater than" if spec.exclusive else "at least"
            raise ValueError(f"{key} must be {operator} {spec.minimum}")
        params[key] = value
    if rule.check:
        rule.check(params)
    result = dict(rule=name, weight=weight, **params)
    if "indicator_number" in item:
        number = item["indicator_number"]
        if type(number) is not int or number < 1:
            raise ValueError("Indicator number must be a positive integer")
        result["indicator_number"] = number
    for field in ("indicator_name", "category"):
        if field in item:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonempty text")
            result[field] = value.strip()
    return result


def component_key(component):
    """Only rule and normalized calculation parameters affect the signal cache."""
    rule = component["rule"]
    return rule, tuple((k, component[k]) for k in REGISTRY[rule].parameters)


def duplicate_signal_count(config):
    """Count extra votes using exactly the same rule and parameters."""
    keys = [component_key(c) for c in config["components"]]
    return len(keys) - len(set(keys))


def validate_config(config):
    """Return a fully specified config; reject typos and invalid risk assumptions."""
    _object(config, ("version", "name", "provider", "components", "combination", "execution", "entry_filters", "strategy_number", "trade_direction"), "Config")
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
    components = [validate_component(item) for item in items]
    ids = [c["indicator_number"] for c in components if "indicator_number" in c]
    if len(ids) != len(set(ids)):
        raise ValueError("The same indicator number cannot vote twice in one strategy")

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
    result = dict(version=1, name=name.strip(), provider=provider, components=components,
                  combination=dict(method=method, threshold=threshold), execution=asdict(settings))
    if "strategy_number" in config:
        number = config["strategy_number"]
        if type(number) is not int or number < 1:
            raise ValueError("Strategy number must be a positive integer")
        result["strategy_number"] = number
    direction = config.get("trade_direction","both")
    if direction not in ("both","long","short"):
        raise ValueError("Trade direction must be both, long or short")
    if direction != "both":
        result["trade_direction"] = direction
    filters = validate_filters(config.get("entry_filters", []))
    if filters:
        result["entry_filters"] = filters
    return result


def is_intraday_strategy(cfg):
    """Identify strategies with a component requiring multiple candles per UTC day."""
    return any(REGISTRY[c["rule"]].intraday_only for c in cfg["components"])


def build_signal(candles, config, *, cache=None):
    """Combine close states; close-only strategies also accept a Series.

    An optional cache is local to one candle history.
    """
    frame = candles if isinstance(candles, pd.DataFrame) else None
    close = frame["close"] if frame is not None else candles
    cfg = validate_config(config)
    components = cfg["components"]
    cache = {} if cache is None else cache
    states = {}
    for position, component in enumerate(components):
        key = component_key(component)
        rule, parameters = key
        if REGISTRY[rule].needs_ohlc and (frame is None or not {"high", "low"} <= set(frame.columns)):
            raise ValueError(f"{rule} requires candle high, low, and close")
        if REGISTRY[rule].needs_volume and (frame is None or "volume" not in frame):
            raise ValueError(f"{rule} requires candle volume")
        if key not in cache:
            source = frame if REGISTRY[rule].needs_ohlc else close
            cache[key] = RULES[rule](source, **dict(parameters))
        states[position] = cache[key]
    votes = pd.DataFrame(states, index=close.index)
    method = cfg["combination"]["method"]
    if method == "weighted":
        weights = pd.Series([c["weight"] for c in components])
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
    return _atr(candles, 14)


def strategy_identity(experiment):
    """Use the saved workbook identity; benchmarks have no workbook number."""
    cfg = experiment.get("config") or {}
    return dict(strategy_number=cfg.get("strategy_number"),
                strategy_name=cfg.get("name", experiment["name"]))


def strategy_directions(cfg):
    """Older workbooks default to independent long and short simulations."""
    direction = cfg.get("trade_direction","both")
    return ("long","short") if direction=="both" else (direction,)


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
        for direction in strategy_directions(cfg):
            experiments.append(dict(id=f"strategy:{i}:{direction}",
                                    name=f"{cfg['name']} ({direction} only)", kind="strategy",
                                    direction=direction, group=group,
                                    execution=execution, config=cfg))
    for (capital, fee, slip), group in benchmarks.items():
        settings = Execution(initial_capital=capital, fee=fee, slippage=slip, fraction=1.,
                             stop_atr=None, target_atr=None)
        for rule, name in (("buy_hold", "Buy and hold"), ("sma100", "SMA100 long/cash")):
            experiments.append(dict(id=f"benchmark:{group}:{rule}", name=name, kind=rule,
                                    direction="long", group=group, execution=settings, config=None))
    return experiments


def component_warmup(component):
    """Completed candles needed before a component can first produce a signal."""
    return REGISTRY[component["rule"]].warmup(component)


def shared_warmup(configs):
    """Return the longest warmup of the selected strategies and benchmarks."""
    return max(100, 14, *(component_warmup(c) for cfg in configs for c in cfg["components"]),
               *(filter_warmup(f) for cfg in configs for f in cfg.get("entry_filters", [])))


def strategy_warmup(cfg):
    """Minimum candle count for a strategy, including ATR only when used."""
    execution = cfg["execution"]
    atr = 14 if execution["stop_atr"] is not None or execution["target_atr"] is not None else 0
    return max(atr, *(component_warmup(c) for c in cfg["components"]),
               *(filter_warmup(f) for f in cfg.get("entry_filters", [])))


def _prepare_history(candles, audit, provider, latest_continuous, fill_confirmed_gaps):
    if (fill_confirmed_gaps and audit.get("internal_missing")
            and not audit_problem(audit, allow_internal_gaps=True)):
        key = provider, audit["symbol"], audit["timeframe"]
        candles = fill_empty_candles(candles, key[2], CONFIRMED_EMPTY_CANDLES.get(key))
    return backtest_history(candles, audit, latest_continuous)


def _matched_dates(conn, provider, pairs, cutoff, warmup, phase, split,
                   latest_continuous, fill_confirmed_gaps):
    """Find one common window per timeframe using eligible, audited histories."""
    dates = {}
    for asset, interval in pairs:
        candles, audit = load_pair(conn,provider,asset,interval,cutoff,
                                  historical_cutoff=phase=="development")
        candles, trimmed = _prepare_history(candles, audit, provider, latest_continuous, fill_confirmed_gaps)
        first = _first_candle(candles, warmup, phase, split, exact=True)
        if (audit["status"] != "ok" and not trimmed and "synthetic" not in candles) or first is None:
            continue
        start, end = candles.index[first], candles.index[-1]
        if interval in dates:
            old_start, old_end = dates[interval]
            start, end = max(start,old_start), min(end,old_end)
        dates[interval] = (start,end)
    return dates


def _first_candle(candles, warmup, phase, split, *, exact):
    if len(candles) <= warmup:
        return None
    if phase != "evaluation":
        return warmup
    boundary = candles.index.searchsorted(split)
    if exact and (boundary < warmup or boundary == len(candles) or candles.index[boundary] != split):
        return None
    first = max(warmup,boundary)
    return first if first < len(candles) else None


def _open_run(root, cutoff, phase, view, stack, columns, result_columns):
    suffix = "-".join(x for x in (phase, view) if x)
    folder = root/(cutoff.strftime("%Y%m%dT%H%M%SZ-")+uuid4().hex+(f"-{suffix}" if suffix else ""))
    folder.mkdir(exist_ok=False)  # Inherit the project ACL on Windows.
    writers = []
    for filename, fields in (("results.csv",result_columns),("trades.csv",columns+TRADE_COLUMNS),
                             ("equity.csv",columns+["candle","equity","synthetic"])):
        handle = stack.enter_context((folder/filename).open("w",newline="",encoding="utf-8"))
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writers.append(writer)
    return folder, writers, dict(completed=0,skipped=0,failed=0)


def run_batch(configs, database, output_root, *, symbol=None, timeframe=None, as_of=None,
              on_progress=None, _connection=None, _split=None, _phase=None, _split_id=None,
              both_views=False, latest_continuous=False, fill_confirmed_gaps=False):
    """Run one provider's grid, streaming outputs instead of retaining all equity in memory.

    The default direct call retains the legacy single view. With both_views,
    full history uses own warmups and matched windows align all eligible pairs.
    With latest_continuous, a pair with only internal gaps uses its final contiguous segment.
    Results use all eligible history and are descriptive, not holdout tests.
    """
    if latest_continuous and fill_confirmed_gaps:
        raise ValueError("Choose one candle gap policy")
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
    directional_count = sum(len(strategy_directions(cfg)) for cfg in configs)
    warmup = shared_warmup(configs)
    settings_count = len({json.dumps(dict(
        {k:v for k,v in cfg.items() if k not in ("name", "strategy_number", "components")},
        components=[{k:v for k,v in c.items() if k not in ("indicator_number", "indicator_name", "category")}
                    for c in cfg["components"]]), sort_keys=True) for cfg in configs})
    coverage = []
    columns = ["provider", "symbol", "timeframe", "strategy_number", "strategy_name", "experiment"]
    result_columns = columns + ["direction", "status", "message", "warmup_candles",
                                "first_execution", "last_candle", "execution_candles",
                                "synthetic_candles_used", "synthetic_candles_in_test_window",
                                "trades", "final_equity", "total_return_pct"]
    source = open_ohlcv(database) if _connection is None else nullcontext(_connection)
    views = ("full-history", "matched-window") if both_views else (None,)
    with source as conn, ExitStack() as stack:
        pairs = [(asset, interval) for asset, interval in stored_pairs(conn, provider)
                 if (symbol is None or asset == symbol) and (timeframe is None or interval == timeframe)]
        if not pairs:
            raise ValueError("No stored asset/timeframe pairs match the batch")
        common = (_matched_dates(conn, provider, pairs, cutoff, warmup, _phase, _split,
                                 latest_continuous, fill_confirmed_gaps) if both_views else {})
        root = Path(output_root)
        root.mkdir(parents=True,exist_ok=True)
        outputs = {view: _open_run(root,cutoff,_phase,view,stack,columns,result_columns)
                   for view in views}
        for i, (asset, interval) in enumerate(pairs, 1):
            candles, audit = load_pair(conn,provider,asset,interval,cutoff,
                                       historical_cutoff=_phase=="development")
            audit["data_sha256"] = hashlib.sha256(
                pd.util.hash_pandas_object(candles,index=True).to_numpy().tobytes()
            ).hexdigest()
            original_count = len(candles)
            candles, trimmed = _prepare_history(candles, audit, provider, latest_continuous, fill_confirmed_gaps)
            synthetic = candles.get("synthetic", pd.Series(False, index=candles.index))
            audit["synthetic_candles"] = int(synthetic.sum())
            if audit["synthetic_candles"]:
                audit["synthetic_dates_utc"] = [date.isoformat() for date in candles.index[synthetic]]
            if trimmed:
                audit.update(backtest_first_utc=candles.index[0],
                             backtest_usable=len(candles),
                             discarded_before_gap=original_count - len(candles))
            coverage.append(audit)
            valid = audit["status"] == "ok" or trimmed or bool(audit["synthetic_candles"])
            if valid:
                atr = atr14(candles)
                baseline = sma_trend(candles.close)
                trend = baseline.clip(lower=0)
                hold = pd.Series(1,index=candles.index,dtype="int8")
                signal_cache = {("sma_trend", (("window",100),)): baseline}
                filter_cache, combined_cache = {}, {}
            for exp in experiments:
                cfg = exp["config"]
                intraday_mismatch = (cfg is not None and is_intraday_strategy(cfg)
                                     and interval in TIMEFRAMES and TIMEFRAMES[interval].duration >= pd.Timedelta(days=1))
                own = strategy_warmup(cfg) if cfg else (1 if exp["kind"]=="buy_hold" else 100)
                key = dict(provider=provider,symbol=asset,timeframe=interval,experiment=exp["id"],
                           **strategy_identity(exp))
                for view, (_, writers, counts) in outputs.items():
                    result_writer, trade_writer, equity_writer = writers
                    required = own if view=="full-history" else warmup
                    summary = dict(key,direction=exp["direction"],warmup_candles=required)
                    first = _first_candle(candles,required,_phase,_split,exact=view!="full-history") if valid else None
                    end = len(candles)
                    problem = ""
                    if intraday_mismatch:
                        problem = "UTC-session VWAP requires intraday candles"
                    elif not valid:
                        problem = f"Data audit: {audit_problem(audit)}"
                        if fill_confirmed_gaps and audit.get("internal_missing"):
                            problem += " (only fully confirmed gaps can be filled)"
                    elif first is None:
                        problem = (f"Need {required} warmup candles and at least one execution candle"
                                   if len(candles)<=required else
                                   "Evaluation boundary needs an exact candle and enough preceding warmup")
                    elif view=="matched-window":
                        start_date, end_date = common[interval]
                        first = candles.index.searchsorted(start_date)
                        end = candles.index.searchsorted(end_date,side="right")
                        if first >= end:
                            problem = "No common execution dates for eligible assets in this timeframe"
                    if problem:
                        result_writer.writerow(dict(summary,status="skipped",message=problem))
                        counts["skipped"] += 1
                        continue
                    window = candles.iloc[first-1:end]
                    if synthetic.iloc[first:end].all():
                        result_writer.writerow(dict(summary,status="skipped",message="No real execution candles"))
                        counts["skipped"] += 1
                        continue
                    try:
                        gate, prior = None, 0
                        if cfg:
                            if cfg["name"] not in combined_cache:
                                combined_cache[cfg["name"]] = build_signal(candles,cfg,cache=signal_cache)
                            combined = combined_cache[cfg["name"]]
                            signal = (combined.clip(lower=0) if exp["direction"]=="long"
                                      else combined.clip(upper=0))
                            allowed = entry_mask(candles,cfg.get("entry_filters",[]),
                                                 exp["direction"],filter_cache)
                            gate = None if allowed is None else allowed.loc[window.index]
                            if first >= 2 and (_phase=="evaluation" or view is not None):
                                # A state already active before the opening boundary is not a new entry.
                                prior = signal.iloc[first-2]
                        else:
                            signal = hold if exp["kind"]=="buy_hold" else trend
                        result = simulate(window,signal.loc[window.index],atr.loc[window.index],
                                          timeframe=interval,settings=exp["execution"],
                                          entry_allowed=gate,prior_signal=prior)
                    except ValueError as exc:
                        result_writer.writerow(dict(summary,status="failed",message=str(exc)))
                        counts["failed"] += 1
                        continue
                    if trimmed:
                        summary["message"] = (f"Continuous history from {candles.index[0].isoformat()} "
                                              f"({original_count - len(candles)} earlier candles excluded)")
                    final = result["equity"].iloc[-1]
                    result_writer.writerow(dict(summary,status="completed",first_execution=window.index[1],
                                                last_candle=window.index[-1],execution_candles=len(window)-1,
                                                synthetic_candles_used=int(synthetic.iloc[:end].sum()),
                                                synthetic_candles_in_test_window=int(synthetic.iloc[first:end].sum()),
                                                trades=len(result["trades"]),final_equity=final,
                                                total_return_pct=(final/exp["execution"].initial_capital-1)*100))
                    trade_writer.writerows(dict(key,**row) for row in result["trades"].to_dict("records"))
                    equity_writer.writerows(dict(key,candle=date,equity=value,synthetic=bool(synthetic.at[date]))
                                            for date,value in result["equity"].items())
                    counts["completed"] += 1
            if on_progress is not None:
                on_progress(i,len(pairs),asset,interval)

    saved = []
    comparison_id = uuid4().hex if both_views else None
    for view, (folder, _, counts) in outputs.items():
        manifest = dict(
            version=1, provider=provider, as_of=str(cutoff), database=str(Path(database).resolve()),
            evaluation="historical descriptive; no holdout or parameter selection",
            independent_capital_per_pair=True, shorts="hypothetical; excludes funding, borrow and venue liquidation",
            warmup_candles=warmup, asset_timeframe_pairs=len(pairs),
            strategy_configs=len(configs), unique_strategy_settings=settings_count,
            directional_strategy_configs=directional_count,
            benchmark_configs=len(experiments)-directional_count,
            planned_strategy_jobs=len(pairs)*directional_count, planned_jobs=len(pairs)*len(experiments),
            jobs=counts, experiments=[dict(exp, execution=asdict(exp["execution"])) for exp in experiments],
            history_policy=("carry forward confirmed empty candles" if fill_confirmed_gaps else
                            "latest continuous history" if latest_continuous else "strict audit"),
            coverage=coverage, duplicate_signal_votes={cfg["name"]: duplicate_signal_count(cfg)
                                                       for cfg in configs if duplicate_signal_count(cfg)},
        )
        if _phase:
            manifest.update(phase=_phase, split_id=_split_id, evaluation_start=str(_split),
                            evaluation=("development; historical"
                                        if _phase=="development" else
                                        "historical evaluation; not an untouched forward test"))
        if view:
            manifest["window_view"] = view
            manifest["comparison_id"] = comparison_id
            manifest["window_policy"] = ("own strategy warmup; dates can differ" if view=="full-history"
                                         else "fresh simulations on one common asset window per timeframe")
            if view=="full-history":
                manifest["warmup_candles"] = None
            else:
                manifest["matched_dates"] = {interval: dict(first_execution=str(start),last_candle=str(end))
                                             for interval,(start,end) in common.items() if start<=end}
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n",encoding="utf-8")
        saved.append((folder,manifest))
    return saved if both_views else saved[0]


def run_split(configs, database, output_root, *, evaluation_start, timeframe="1d",
              symbol=None, as_of=None, on_progress=None, both_views=False,
              latest_continuous=False, fill_confirmed_gaps=False):
    """Run fixed daily periods with frozen configs and a single database snapshot."""
    split = pd.to_datetime(evaluation_start, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    if timeframe != "1d":
        raise ValueError("The fixed split currently requires daily (1d) candles")
    if pd.isna(split) or pd.isna(cutoff) or split != split.normalize() or split >= cutoff:
        raise ValueError("Evaluation start must be UTC midnight before the run cutoff")
    configs = list(configs)
    split_id = uuid4().hex
    results = []
    with open_ohlcv(database) as conn:
        for phase, end in (("development", split), ("evaluation", cutoff)):
            callback = None
            if on_progress is not None:
                offset = 0 if phase == "development" else 1
                callback = lambda i,n,s,t: on_progress(offset*n+i,2*n,s,f"{t} / {phase}")
            batch = run_batch(configs,database,output_root,symbol=symbol,timeframe=timeframe,
                                     as_of=end,on_progress=callback,_connection=conn,_split=split,
                                     _phase=phase,_split_id=split_id,both_views=both_views,
                                     latest_continuous=latest_continuous, fill_confirmed_gaps=fill_confirmed_gaps)
            results.extend(batch if both_views else [batch])
    return results
