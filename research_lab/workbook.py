"""Read explicit strategy combinations from an Excel workbook."""
from dataclasses import fields
from itertools import product
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from .engine import Execution
from .filters import validate_filters
from .rules import REGISTRY
from .runner import validate_component, validate_config

WORKBOOK_PATH = Path(__file__).resolve().parents[1]/"research_configs"/"strategies_config.xlsx"
EXECUTION_KEYS = tuple(field.name for field in fields(Execution))


def _rows(book, sheet, required, optional=(), *, display_only=()):
    if sheet not in book.sheetnames:
        raise ValueError(f"Missing worksheet: {sheet}")
    rows = book[sheet].iter_rows()
    headers = [cell.value for cell in next(rows, ())]
    names = [name for name in headers if name is not None]
    if len(names) != len(set(names)) or not set(required) <= set(names) or set(names) - set(required + optional + display_only):
        raise ValueError(f"{sheet}, row 1: expected {', '.join(required)}"
                         + (f"; optional: {', '.join(optional)}" if optional else ""))
    for row, cells in enumerate(rows, 2):
        if all(cell.value is None or name in display_only for name, cell in zip(headers, cells)):
            continue
        where = f"{sheet}, row {row}"
        data = {}
        for name, cell in zip(headers, cells):
            if name in display_only:
                continue
            if cell.data_type == "f":
                raise ValueError(f"{where}: use values rather than Excel formulas")
            if cell.value is not None:
                if name is None:
                    raise ValueError(f"{where}: value has no column header")
                data[name] = cell.value
        yield where, data


def _id(data, key, where):
    value = data.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{where}: {key} must be a positive integer")
    return value


def _text(data, key, where):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be nonempty text")
    return value.strip()


def load_workbook_configs(source, provider):
    """Resolve the workbook to normalized strategy dictionaries for the research runner."""
    try:
        book = load_workbook(source, read_only=True, data_only=False, keep_links=False)
    except (BadZipFile, InvalidFileException, ParseError, KeyError) as exc:
        raise ValueError(f"Unable to read .xlsx workbook: {exc}") from exc
    try:
        return _resolve(book, provider)
    finally:
        book.close()


def _resolve(book, provider):
    indicators, locations, parameter_rows = {}, {}, {}
    for where, row in _rows(book, "individual_indicators",
                            ("indicator_number", "indicator_name", "strategy_function_name"), ("category",)):
        number = _id(row, "indicator_number", where)
        if number in indicators:
            raise ValueError(f"{where}: duplicate indicator_number {number}")
        indicators[number] = dict(indicator_number=number, indicator_name=_text(row,"indicator_name",where),
                                  rule=_text(row,"strategy_function_name",where))
        if "category" in row:
            indicators[number]["category"] = _text(row,"category",where)
        locations[number] = where
        parameter_rows[number] = []
    seen, sweeps = set(), {}
    for where, row in _rows(book, "individual_indicator_settings",
                            ("indicator_number","setting","value"), ("indicator_name",)):
        number = _id(row,"indicator_number",where)
        if number not in indicators:
            raise ValueError(f"{where}: unknown indicator_number {number}")
        key = _text(row,"setting",where)
        if (number,key) in seen or key in ("rule","indicator_name","indicator_number","weight","category"):
            raise ValueError(f"{where}: duplicate or reserved setting {key}")
        if "value" not in row:
            raise ValueError(f"{where}: value is required")
        seen.add((number,key))
        value = row["value"]
        if isinstance(value, str) and "," in value:
            rule = REGISTRY.get(indicators[number]["rule"])
            parameter = rule.parameters.get(key) if rule else None
            if parameter is None:
                raise ValueError(f"{where}: unknown indicator parameter {key!r}")
            convert = int if type(parameter.default) is int else float
            try:
                values = tuple(dict.fromkeys(convert(v.strip()) for v in value.split(",")))
            except ValueError as exc:
                raise ValueError(f"{where}: enter comma-separated numeric values for {key}") from exc
            value = values[0]
            sweeps.setdefault(number, {})[key] = (values, where)
        indicators[number][key] = value
        parameter_rows[number].append(where)
    for number, indicator in indicators.items():
        try:
            indicators[number] = validate_component(indicator)
        except ValueError as exc:
            rows = ", ".join(parameter_rows[number])
            raise ValueError(f"{locations[number]}{'; '+rows if rows else ''}: {exc}") from exc

    strategies, strategy_rows, signal_modes = {}, {}, {}
    names = set()
    for where, row in _rows(book, "strategies",
                            ("strategy_number","strategy_name","combination_rule","weighted_score_threshold"),
                            (*EXECUTION_KEYS, "trade_direction", "signal_mode"), display_only=("Entry filters",)):
        number = _id(row,"strategy_number",where)
        name = _text(row,"strategy_name",where)
        if number in strategies or name in names:
            raise ValueError(f"{where}: duplicate strategy number or name")
        names.add(name)
        method = _text(row,"combination_rule",where)
        combination = dict(method=method)
        threshold = row.get("weighted_score_threshold")
        if threshold is None or threshold == "N/A":
            if method == "weighted":
                raise ValueError(f"{where}: weighted strategies need a numeric weighted_score_threshold")
        else:
            combination["threshold"] = threshold
        execution = {key: row[key] for key in EXECUTION_KEYS if key in row}
        for key in ("stop_atr","target_atr"):
            if execution.get(key) == "off":
                execution[key] = None
        strategies[number] = dict(version=1, name=name, strategy_number=number, provider=provider, components=[],
                                  combination=combination, execution=execution)
        if "trade_direction" in row:
            strategies[number]["trade_direction"] = _text(row,"trade_direction",where)
        mode = row.get("signal_mode", "normal")
        if not isinstance(mode, str) or mode.strip().lower() not in ("normal", "inverted", "both"):
            raise ValueError(f"{where}: signal_mode must be normal, inverted, or both")
        signal_modes[number] = mode.strip().lower()
        strategy_rows[number] = where

    members = set()
    for where, row in _rows(book, "strategy_components", ("strategy_number","indicator_number","weight")):
        strategy = _id(row,"strategy_number",where)
        indicator = _id(row,"indicator_number",where)
        if strategy not in strategies or indicator not in indicators:
            raise ValueError(f"{where}: unknown strategy or indicator number")
        if (strategy,indicator) in members:
            raise ValueError(f"{where}: indicator {indicator} already votes in strategy {strategy}")
        members.add((strategy,indicator))
        try:
            component = validate_component(dict(indicators[indicator], weight=row.get("weight",1.)))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        strategies[strategy]["components"].append(component)
    if "strategy_entry_filters" in book.sheetnames:
        for where, row in _rows(book, "strategy_entry_filters", ("strategy_number", "filter_rule"),
                                ("period", "threshold")):
            number = _id(row,"strategy_number",where)
            if number not in strategies:
                raise ValueError(f"{where}: unknown strategy number")
            filters = strategies[number].setdefault("entry_filters", [])
            try:
                strategies[number]["entry_filters"] = validate_filters(
                    filters + [{k:v for k,v in row.items() if k != "strategy_number"}])
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
    if not strategies:
        raise ValueError("strategies: add at least one strategy")
    configs = []
    resolved_names = set()
    for number, strategy in strategies.items():
        where = strategy_rows[number]
        for cfg in _variants(strategy, signal_modes[number], sweeps, where):
            if cfg["name"] in resolved_names:
                raise ValueError(f"{where}: duplicate generated strategy name {cfg['name']!r}")
            resolved_names.add(cfg["name"])
            configs.append(cfg)
    return configs


def _variants(strategy, mode, sweeps, where):
    """Expand workbook parameter lists and signal modes using the runner's validation."""
    axes = [
        (index, key, values, location)
        for index, component in enumerate(strategy["components"])
        for key, (values, location) in sweeps.get(component["indicator_number"], {}).items()
    ]
    for values in product(*(axis[2] for axis in axes)):
        components = [dict(component) for component in strategy["components"]]
        labels = []
        for (index, key, _, _), value in zip(axes, values):
            components[index][key] = value
            labels.append(f"{components[index]['indicator_number']}.{key}={value}")
        for invert in ((False, True) if mode == "both" else (mode == "inverted",)):
            tags = labels + (["Inverted" if invert else "Normal"] if mode != "normal" else [])
            name = f"{strategy['name']} [{', '.join(tags)}]" if tags else strategy["name"]
            try:
                yield validate_config(dict(strategy, name=name, components=components, invert_signal=invert))
            except ValueError as exc:
                sources = ", ".join(dict.fromkeys(axis[3] for axis in axes))
                context = f"{where}, {sources}" if sources else where
                raise ValueError(f"{context}: {name}: {exc}") from exc
