"""Read explicit strategy combinations from an Excel workbook."""
from dataclasses import fields
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from .engine import Execution
from .filters import validate_filters
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
    seen = set()
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
        indicators[number][key] = row["value"]
        parameter_rows[number].append(where)
    for number, indicator in indicators.items():
        try:
            indicators[number] = validate_component(indicator)
        except ValueError as exc:
            rows = ", ".join(parameter_rows[number])
            raise ValueError(f"{locations[number]}{'; '+rows if rows else ''}: {exc}") from exc

    strategies, strategy_rows = {}, {}
    names = set()
    for where, row in _rows(book, "strategies",
                            ("strategy_number","strategy_name","combination_rule","weighted_score_threshold"),
                            (*EXECUTION_KEYS, "trade_direction"), display_only=("Entry filters",)):
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
    for number, strategy in strategies.items():
        try:
            configs.append(validate_config(strategy))
        except ValueError as exc:
            raise ValueError(f"{strategy_rows[number]}: {exc}") from exc
    return configs
