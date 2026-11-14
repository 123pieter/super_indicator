"""Validate portable research strategy configs and build their close-based signal."""
from collections import Counter
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
from .strategies import _atr

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
    _object(config, ("version", "name", "provider", "components", "combination", "execution",
                     "entry_filters", "strategy_number", "trade_direction", "invert_signal"), "Config")
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
    invert_signal = config.get("invert_signal", False)
    if type(invert_signal) is not bool:
        raise ValueError("Invert signal must be true or false")
    if invert_signal:
        result["invert_signal"] = True
    filters = validate_filters(config.get("entry_filters", []))
    if filters:
        result["entry_filters"] = filters
    return result


def is_intraday_strategy(cfg):
    """Identify strategies with a component requiring multiple candles per UTC day."""
    return any(REGISTRY[c["rule"]].intraday_only for c in cfg["components"])


def build_signal(candles, config, *, cache=None):
    """Combine close states and apply the configured signal orientation.

    Close-only strategies accept a Series. A component cache is local to one
    candle history and can be shared by standard and inverted strategies.
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
    signal = (long.astype("int8") - short.astype("int8")).rename("combined")
    return -signal if cfg.get("invert_signal") else signal


def atr14(candles):
    """TA-Lib ATR used for sizing and protective orders."""
    return _atr(candles, 14)


def strategy_identity(experiment):
    """Use the saved workbook identity when available."""
    cfg = experiment.get("config") or {}
    return dict(strategy_number=cfg.get("strategy_number"),
                strategy_name=cfg.get("name", experiment["name"]))


def strategy_directions(cfg):
    """Return executed sides for the configured original signal sides."""
    direction = cfg.get("trade_direction", "both")
    if direction == "both":
        return "long", "short"
    if cfg.get("invert_signal"):
        direction = "short" if direction == "long" else "long"
    return (direction,)


def _experiments(configs):
    """Create one replay for each configured strategy direction."""
    experiments = []
    for i, cfg in enumerate(configs, 1):
        execution = Execution(**cfg["execution"])
        for direction in strategy_directions(cfg):
            experiments.append(dict(id=f"strategy:{i}:{direction}",
                                    name=f"{cfg['name']} ({direction} only)", kind="strategy",
                                    direction=direction, execution=execution, config=cfg))
    return experiments


def component_warmup(component):
    """Completed candles needed before a component can first produce a signal."""
    return REGISTRY[component["rule"]].warmup(component)


def strategy_warmup(cfg):
    """Minimum candle count for a strategy, including ATR only when used."""
    execution = cfg["execution"]
    atr = 15 if execution["stop_atr"] is not None or execution["target_atr"] is not None else 0
    return max(atr, *(component_warmup(c) for c in cfg["components"]),
               *(filter_warmup(f) for f in cfg.get("entry_filters", [])))


def _prepare_history(candles, audit, provider, latest_continuous, fill_confirmed_gaps):
    if (fill_confirmed_gaps and audit.get("internal_missing")
            and not audit_problem(audit, allow_internal_gaps=True)):
        key = provider, audit["symbol"], audit["timeframe"]
        candles = fill_empty_candles(candles, key[2], CONFIRMED_EMPTY_CANDLES.get(key))
    return backtest_history(candles, audit, latest_continuous)


def _first_candle(candles, warmup, phase, split, *, execution_start=None):
    if len(candles) <= warmup:
        return None
    earliest = warmup if execution_start is None else max(warmup, candles.index.searchsorted(execution_start))
    if len(candles) <= earliest:
        return None
    if phase != "evaluation":
        return earliest
    boundary = candles.index.searchsorted(split)
    first = max(earliest,boundary)
    return first if first < len(candles) else None


def _new_batch_folder(output_root):
    """Reserve a batch folder without overwriting an earlier run."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%d%H%M")
    folder = root / stamp
    counter = 2
    while True:
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            folder = root / f"{stamp}-{counter}"
            counter += 1


def _open_run(root, phase, stack, columns, result_columns, save_equity, signals_only):
    folder = root / phase if phase else root
    folder.mkdir(parents=True, exist_ok=True)
    writers = []
    files = [("results.csv", result_columns), ("trades.csv", columns + TRADE_COLUMNS)]
    if save_equity:
        files.append(("equity.csv", columns + ["candle", "equity", "synthetic"]))
    for filename, fields in (() if signals_only else files):
        handle = stack.enter_context((folder/filename).open("w",newline="",encoding="utf-8"))
        if filename == "equity.csv":
            writer = csv.writer(handle)
            writer.writerow(fields)
        else:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
        writers.append(writer)
    writers.extend([None] * (3 - len(writers)))
    signal_writer = None
    if phase != "development":
        handle = stack.enter_context((folder/"latest_signals.csv").open("w", newline="", encoding="utf-8"))
        signal_columns = [column for column in columns if column != "experiment"]
        signal_writer = csv.DictWriter(handle, fieldnames=signal_columns+[
            "signal_candle_utc", "execution_candle_utc", "event", "action", "position_side",
            "reference_entry_price", "stop_loss", "take_profit", "quantity", "reason",
        ])
        signal_writer.writeheader()
    return folder, writers, dict(completed=0,skipped=0,failed=0), signal_writer


def run_batch(configs, database, output_root, *, symbol=None, timeframe=None, as_of=None,
              on_progress=None, _connection=None, _split=None, _phase=None, _split_id=None, _batch_folder=None,
              latest_continuous=False, fill_confirmed_gaps=False, coverage_on=None,
              execution_start=None, include_funding=False, signal_delay_bars=0, save_equity=True,
              signals_only=False):
    """Run one provider's grid, streaming outputs instead of retaining all equity in memory.

    Each strategy starts after its own warmup on each asset.
    With latest_continuous, a pair with only internal gaps uses its final contiguous segment.
    Signal delay shifts strategy states, entry filters and signal-time ATR together.
    Without equity export, summarize each curve immediately and release it.
    Signals-only mode replays strategies and writes only next-open entries.
    """
    if type(signal_delay_bars) is not int or signal_delay_bars < 0:
        raise ValueError("Signal delay must be a nonnegative whole number of candles")
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
    if execution_start is not None:
        execution_start = pd.to_datetime(execution_start, utc=True)
        if pd.isna(execution_start) or execution_start >= cutoff:
            raise ValueError("Execution start must be before the batch cutoff")
    if include_funding and provider not in ("binance", "hyperliquid"):
        raise ValueError("Estimated historical funding requires Binance or Hyperliquid")
    # A chosen historical end clips later stored prices. Ordinary current runs
    # still flag genuinely future-dated rows.
    historical_cutoff = _phase=="development" or (
        as_of is not None and cutoff.normalize() < pd.Timestamp.now(tz="UTC").normalize())
    save_equity = save_equity and not signals_only
    experiments = _experiments(configs)
    directional_count = len(experiments)
    settings_count = len({json.dumps(dict(
        {k:v for k,v in cfg.items() if k not in ("name", "strategy_number", "components")},
        components=[{k:v for k,v in c.items() if k not in ("indicator_number", "indicator_name", "category")}
                    for c in cfg["components"]]), sort_keys=True) for cfg in configs})
    coverage = []
    columns = ["provider", "symbol", "timeframe", "strategy_number", "strategy_name", "experiment"]
    result_columns = columns + ["direction", "status", "message", "warmup_candles",
                                "first_execution", "last_candle", "execution_candles",
                                "synthetic_candles_used", "synthetic_candles_in_test_window", "non_tradable_candles",
                                "trades", "final_equity", "total_return_pct", "funding_paid"]
    source = open_ohlcv(database) if _connection is None else nullcontext(_connection)
    with source as conn, ExitStack() as stack:
        pairs = stored_pairs(conn, provider, coverage_on=coverage_on, symbol=symbol, timeframe=timeframe)
        if not pairs:
            raise ValueError("No stored asset/timeframe pairs match the batch")
        root = _new_batch_folder(output_root) if _batch_folder is None else _batch_folder
        folder, writers, counts, signal_writer = _open_run(root, _phase, stack, columns, result_columns, save_equity, signals_only)
        result_writer, trade_writer, equity_writer = writers

        replay_issues = Counter()
        replay_examples = {}

        def record_issue(status, message, summary):
            counts[status] += 1
            if result_writer is not None:
                result_writer.writerow(dict(summary, status=status, message=message))
            elif signals_only:
                issue = status, message
                replay_issues[issue] += 1
                replay_examples.setdefault(issue, f"{summary['symbol']} / {summary['timeframe']} / {summary['experiment']}")

        metric_rows = {}
        if not save_equity and not signals_only:
            from .report import bar_exposure, metrics
        latest_signal_candle = None
        latest_entries = []
        for i, (asset, interval) in enumerate(pairs, 1):
            candles, audit = load_pair(conn,provider,asset,interval,cutoff,
                                       historical_cutoff=historical_cutoff)
            original_count = len(candles)
            if original_count and (latest_signal_candle is None or candles.index[-1] > latest_signal_candle):
                latest_signal_candle = candles.index[-1]
                latest_entries.clear()
            if signals_only and (not original_count or candles.index[-1] < latest_signal_candle):
                if on_progress is not None:
                    on_progress(i, len(pairs), asset, interval)
                continue
            if not signals_only:
                audit["data_sha256"] = hashlib.sha256(
                    pd.util.hash_pandas_object(candles, index=True).to_numpy().tobytes()
                ).hexdigest()
            candles, trimmed = _prepare_history(candles, audit, provider, latest_continuous, fill_confirmed_gaps)
            synthetic = candles.get("synthetic", pd.Series(False, index=candles.index))
            audit["synthetic_candles"] = int(synthetic.sum())
            if audit["synthetic_candles"]:
                audit["synthetic_dates_utc"] = [date.isoformat() for date in candles.index[synthetic]]
            if save_equity:
                equity_dates = [str(date) for date in candles.index]
                equity_synthetic = synthetic.tolist()
            if trimmed:
                audit.update(backtest_first_utc=candles.index[0],
                             backtest_usable=len(candles),
                             discarded_before_gap=original_count - len(candles))
            coverage.append(audit)
            valid = audit["status"] == "ok" or trimmed or bool(audit["synthetic_candles"])
            if valid:
                atr = atr14(candles)
                strategy_atr = atr.shift(signal_delay_bars) if signal_delay_bars else atr
                signal_cache, filter_cache, combined_cache, funding_cache = {}, {}, {}, {}
            for exp in experiments:
                cfg = exp["config"]
                intraday_mismatch = (is_intraday_strategy(cfg)
                                     and interval in TIMEFRAMES and TIMEFRAMES[interval].duration >= pd.Timedelta(days=1))
                own = strategy_warmup(cfg) + signal_delay_bars
                key = dict(provider=provider,symbol=asset,timeframe=interval,experiment=exp["id"],
                           **strategy_identity(exp))
                summary = dict(key,direction=exp["direction"],warmup_candles=own)
                first = (_first_candle(candles, own, _phase, _split,
                                       execution_start=execution_start) if valid else None)
                end = len(candles)
                problem = ""
                if intraday_mismatch:
                    problem = "UTC-session VWAP requires intraday candles"
                elif not valid:
                    problem = f"Data audit: {audit_problem(audit)}"
                    if fill_confirmed_gaps and audit.get("internal_missing"):
                        problem += " (only fully confirmed gaps can be filled)"
                elif first is None and execution_start is not None and len(candles) and candles.index[-1] < execution_start:
                    problem = "Stored candles end before the requested execution start"
                elif first is None:
                    problem = (f"Need {own} warmup candles and at least one execution candle"
                               if len(candles)<=own else
                               "No execution candles at or after the evaluation boundary")
                if problem:
                    record_issue("skipped", problem, summary)
                    continue
                window = candles.iloc[first-1:end]
                tradable = ~synthetic.loc[window.index]
                if provider in ("binance", "hyperliquid"):
                    tradable &= window.volume.gt(0)
                if not tradable.iloc[1:].any():
                    record_issue("skipped", "No tradable execution candles", summary)
                    continue
                funding = None
                if include_funding:
                    from .funding import load_candle_funding
                    bounds = first, end
                    if bounds not in funding_cache:
                        try:
                            observed = load_candle_funding(conn, provider, asset, window.index[1:], interval)
                            funding_cache[bounds] = observed.reindex(window.index, fill_value=0.)
                        except ValueError as exc:
                            funding_cache[bounds] = str(exc)
                    funding = funding_cache[bounds]
                    if isinstance(funding, str):
                        record_issue("skipped", f"Funding unavailable: {funding}", summary)
                        continue
                try:
                    gate, prior = None, 0
                    if cfg["name"] not in combined_cache:
                        combined_cache[cfg["name"]] = build_signal(candles,cfg,cache=signal_cache)
                    combined = combined_cache[cfg["name"]]
                    side = exp["direction"]
                    signal = combined.clip(lower=0) if side == "long" else combined.clip(upper=0)
                    # Invert the order side after evaluating the original entry rules.
                    filter_side = ("short" if side == "long" else "long") if cfg.get("invert_signal") else side
                    allowed = entry_mask(candles, cfg.get("entry_filters", []), filter_side, filter_cache)
                    if signal_delay_bars:
                        signal = signal.shift(signal_delay_bars, fill_value=0)
                        if allowed is not None:
                            allowed = allowed.shift(signal_delay_bars, fill_value=False)
                    gate = None if allowed is None else allowed.loc[window.index]
                    if first >= 2:
                        # A state already active before the opening boundary is not a new entry.
                        prior = signal.iloc[first-2]
                    result = simulate(window,signal.loc[window.index],strategy_atr.loc[window.index],
                                      timeframe=interval,settings=exp["execution"],
                                      entry_allowed=gate,prior_signal=prior,funding=funding,tradable=tradable,
                                      preview_next_open=signal_writer is not None,
                                      record_history=not signals_only)
                except ValueError as exc:
                    record_issue("failed", str(exc), summary)
                    continue
                for order in result["next_open_orders"]:
                    order["signal_candle_utc"] -= signal_delay_bars * TIMEFRAMES[interval].duration
                    if order["event"] == "entry" and order["signal_candle_utc"] == latest_signal_candle:
                        latest_entries.append(dict({k: v for k, v in key.items() if k != "experiment"}, **order))
                if signals_only:
                    counts["completed"] += 1
                    continue
                if trimmed:
                    summary["message"] = (f"Continuous history from {candles.index[0].isoformat()} "
                                          f"({original_count - len(candles)} earlier candles excluded)")
                final = result["equity"].iloc[-1]
                result_writer.writerow(dict(summary,status="completed",first_execution=window.index[1],
                                            last_candle=window.index[-1],execution_candles=len(window)-1,
                                            synthetic_candles_used=int(synthetic.iloc[:end].sum()),
                                            synthetic_candles_in_test_window=int(synthetic.iloc[first:end].sum()),
                                            non_tradable_candles=int((~tradable.iloc[1:]).sum()),
                                            trades=len(result["trades"]),final_equity=final,
                                            total_return_pct=(final/exp["execution"].initial_capital-1)*100,
                                            funding_paid=result["trades"].funding_paid.sum() if include_funding else ""))
                trades = result["trades"]
                if signal_delay_bars and not trades.empty:
                    # Retain the original decision candle in the exported trade log.
                    trades["signal_candle"] -= signal_delay_bars * TIMEFRAMES[interval].duration
                trade_writer.writerows(dict(key,**row) for row in trades.to_dict("records"))
                if equity_writer is None:
                    curve = result["equity"]
                    metric_rows[interval, asset, exp["id"]] = metrics(
                        curve, trades, interval, bar_exposure(curve.index, trades)
                    )
                else:
                    identity = tuple(key[column] for column in columns)
                    equity_writer.writerows(
                        (*identity, date, value, flag)
                        for date, value, flag in zip(
                            equity_dates[first-1:end], result["equity"].tolist(),
                            equity_synthetic[first-1:end], strict=True)
                    )
                counts["completed"] += 1
            if on_progress is not None:
                on_progress(i,len(pairs),asset,interval)

        if signal_writer is not None:
            signal_writer.writerows(latest_entries)

    if signals_only:
        issues = [
            {"Status": status, "Reason": message, "Replays": count,
             "Example (asset / timeframe / experiment)": replay_examples[status, message]}
            for (status, message), count in replay_issues.most_common()
        ]
        return folder, {"jobs": counts, "latest_signal_candle_utc": latest_signal_candle,
                        "replay_issues": issues}

    manifest = dict(
        version=1, provider=provider, as_of=str(cutoff), database=str(Path(database).resolve()),
        evaluation="historical descriptive; no holdout or parameter selection",
        independent_capital_per_pair=True,
        shorts="hypothetical; excludes borrowing and venue liquidation",
        funding_model=("observed signed rates using candle-close oracle proxy" if include_funding else "off"),
        execution_start=str(execution_start) if execution_start is not None else None,
        signal_delay_bars=signal_delay_bars,
        window_view="full-history", asset_timeframe_pairs=len(pairs),
        strategy_configs=len(configs), unique_strategy_settings=settings_count,
        directional_strategy_configs=directional_count,
        planned_strategy_jobs=len(pairs)*directional_count, planned_jobs=len(pairs)*len(experiments),
        jobs=counts, experiments=[dict(exp, execution=asdict(exp["execution"])) for exp in experiments],
        history_policy=("carry forward confirmed empty candles" if fill_confirmed_gaps else
                        "latest continuous history" if latest_continuous else "strict audit"),
        coverage_on=str(coverage_on) if coverage_on is not None else None,
        coverage=coverage, duplicate_signal_votes={cfg["name"]: duplicate_signal_count(cfg)
                                                   for cfg in configs if duplicate_signal_count(cfg)},
    )
    if signal_writer is not None:
        manifest["latest_signal_candle_utc"] = latest_signal_candle
    if _phase:
        manifest.update(phase=_phase, split_id=_split_id, evaluation_start=str(_split),
                        evaluation=("development; historical"
                                    if _phase=="development" else
                                    "historical evaluation; not an untouched forward test"))
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n", encoding="utf-8")
    if not save_equity:
        from .report import export_metrics

        export_metrics(folder, metric_rows=metric_rows)
    return folder, manifest


def run_split(configs, database, output_root, *, evaluation_start, timeframe="1d",
              symbol=None, as_of=None, on_progress=None,
              latest_continuous=False, fill_confirmed_gaps=False, coverage_on=None,
              execution_start=None, include_funding=False, signal_delay_bars=0, save_equity=True,
              signals_only=False):
    """Run a fixed chronological split with frozen configs and one database snapshot."""
    split = pd.to_datetime(evaluation_start, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    if timeframe not in TIMEFRAMES:
        raise ValueError("A fixed split requires one supported timeframe")
    if pd.isna(split) or pd.isna(cutoff) or split != split.floor(TIMEFRAMES[timeframe].duration) or split >= cutoff:
        raise ValueError("Evaluation start must align with a UTC candle open before the run cutoff")
    if execution_start is not None:
        execution_start = pd.to_datetime(execution_start, utc=True)
        if pd.isna(execution_start) or execution_start >= split:
            raise ValueError("Execution start must precede the historical evaluation split")
    configs = list(configs)
    split_id = uuid4().hex
    results = []
    with open_ohlcv(database) as conn:
        batch_folder = _new_batch_folder(output_root)
        periods = (("evaluation", cutoff),) if signals_only else (("development", split), ("evaluation", cutoff))
        for phase, end in periods:
            callback = on_progress
            if on_progress is not None and not signals_only:
                offset = 0 if phase == "development" else 1
                callback = lambda i,n,s,t: on_progress(offset*n+i,2*n,s,f"{t} / {phase}")
            batch = run_batch(configs,database,output_root,symbol=symbol,timeframe=timeframe,
                                     as_of=end,on_progress=callback,_connection=conn,_split=split,
                                     _phase=phase,_split_id=split_id,_batch_folder=batch_folder,
                                     latest_continuous=latest_continuous, fill_confirmed_gaps=fill_confirmed_gaps,
                                     coverage_on=coverage_on, execution_start=execution_start,
                                     include_funding=include_funding, signal_delay_bars=signal_delay_bars,
                                     save_equity=save_equity, signals_only=signals_only)
            results.append(batch)
    return results
