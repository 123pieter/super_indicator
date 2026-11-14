"""CSV metrics for a completed research batch."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from timeframes import get_timeframe
from .runner import strategy_identity


def metrics(equity, trades, timeframe, exposure):
    """Use bar-close equity for risk metrics and net trade P&L for trade metrics."""
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]) / pd.Timedelta(days=365)
    periods = pd.Timedelta(days=365) / get_timeframe(timeframe).duration
    growth = equity.iloc[-1] / equity.iloc[0]
    annual_log = math.log(growth) / years if years else np.nan
    vol = returns.std(ddof=1)
    downside = np.sqrt(returns.clip(upper=0).pow(2).mean())
    pnl = trades.net_pnl
    win, loss = pnl[pnl > 0], pnl[pnl < 0]
    return dict(
        total_return_pct=(growth-1)*100,
        cagr_pct=math.expm1(annual_log)*100 if annual_log < 700 else np.nan,
        drawdown_pct=(equity/equity.cummax()-1).min()*100,
        sharpe=returns.mean()/vol*np.sqrt(periods) if vol > 0 else np.nan,
        sortino=returns.mean()/downside*np.sqrt(periods) if downside > 0 else np.nan,
        profit_factor=win.sum()/-loss.sum() if len(loss) else (np.inf if len(win) else np.nan),
        win_rate_pct=(pnl > 0).mean()*100, average_win=win.mean(), average_loss=loss.mean(),
        trades=len(pnl), exposure_pct=exposure*100, sample="thin" if len(pnl) < 30 else "30+ trades",
        funding_paid=trades.funding_paid.sum() if "funding_paid" in trades else np.nan,
    )


def bar_exposure(index, trades):
    """Fraction of evaluation candles with exposure; intrabar exits count that candle."""
    bars = index[1:]  # First mark is starting equity, before any executable candle.
    changes = np.zeros(len(bars)+1, dtype=int)
    if not trades.empty:
        starts = bars.searchsorted(trades["entry_candle"])
        exits = trades["exit_candle"]
        ends = np.where(trades["exit_phase"].eq("open"),
                        bars.searchsorted(exits, side="left"),
                        bars.searchsorted(exits, side="right"))
        exposed = ends > starts
        # Multiple trades can affect the same candle boundary.
        np.add.at(changes, starts[exposed], 1)
        np.add.at(changes, ends[exposed], -1)
    return (changes.cumsum()[:-1] > 0).mean() if len(bars) else np.nan


def _label(exp):
    if exp["kind"] != "strategy":
        return "Benchmark"
    return "Individual" if len(exp["config"]["components"]) == 1 else "Super indicator"


def _direction(exp):
    """Older saved reports did not store direction separately."""
    return exp.get("direction", "long + short" if exp["kind"] == "strategy" else "long")


def _read(folder, name, dates=()):
    # Keep identifiers such as 00 and NA intact; parse blanks only as missing.
    df = pd.read_csv(folder/name, dtype={"symbol": str}, keep_default_na=False, na_values=[""])
    for col in dates:
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise", format="mixed")
    return df


def _tables(results, metric_rows, experiments):
    rows = []
    for r in results.to_dict("records"):
        key = (r["timeframe"], r["symbol"], r["experiment"])
        exp = experiments[key[2]]
        row = dict(r, strategy=exp["name"], category=_label(exp), direction=_direction(exp))
        row.update(strategy_identity(exp))
        row.update(metric_rows.get(key, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def _overall(per_asset):
    """Summarize independent asset results with equal asset weights."""
    keys = ["provider", "timeframe", "experiment"]
    medians = ("cagr_pct", "sharpe", "sortino", "drawdown_pct", "profit_factor",
               "win_rate_pct", "average_win", "average_loss", "exposure_pct")
    rows = []
    starts = pd.to_datetime(per_asset["first_execution"], utc=True, format="mixed")
    ends = pd.to_datetime(per_asset["last_candle"], utc=True, format="mixed")
    for _, group in per_asset.groupby(keys, sort=False):
        completed = group.loc[group.status.eq("completed")]
        returns = completed["total_return_pct"]
        start = starts.loc[completed.index]
        end = ends.loc[completed.index]
        row = group.iloc[0][keys + ["strategy", "strategy_number", "strategy_name",
                                   "category", "direction"]].to_dict()
        row.update(
            requested_assets=len(group), completed_assets=len(completed),
            skipped_assets=int(group.status.eq("skipped").sum()),
            failed_assets=int(group.status.eq("failed").sum()),
            earliest_execution_utc=start.min(), latest_candle_utc=end.max(),
            median_execution_candles=completed["execution_candles"].median(),
            median_test_days=((end - start).dt.total_seconds() / 86400).median(),
            mean_asset_return_pct=returns.mean(), median_asset_return_pct=returns.median(),
            positive_assets_pct=returns.gt(0).mean() * 100,
            total_trades=int(completed["trades"].sum()),
        )
        for metric in medians:
            values = completed.get(metric, pd.Series(dtype=float))
            row[f"median_asset_{metric}"] = values.dropna().median() if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def export_metrics(run, *, metric_rows=None):
    """Write metrics using per-job metric rows or the saved equity and trade CSVs."""
    folder = Path(run)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    experiments = {x["id"]:x for x in manifest["experiments"]}
    results = _read(folder,"results.csv")
    if metric_rows is None:
        equity = _read(folder, "equity.csv", ("candle",))
        trades = _read(folder, "trades.csv", ("entry_candle", "exit_candle", "signal_candle"))
        keys = ["timeframe", "symbol", "experiment"]
        logs = dict(tuple(trades.groupby(keys, sort=False)))
        empty = trades.iloc[:0]
        metric_rows = {}
        for key, rows in equity.groupby(keys, sort=False):
            curve = rows.set_index("candle").equity.sort_index()
            log = logs.get(key, empty)
            metric_rows[key] = metrics(curve, log, key[0], bar_exposure(curve.index, log))
    per_asset = _tables(results, metric_rows, experiments)
    overall = _overall(per_asset)
    overall.sort_values("median_asset_return_pct", ascending=False, kind="stable", inplace=True)
    per_asset.drop(columns="experiment").to_csv(folder/"metrics_per_asset.csv", index=False)
    overall.drop(columns="experiment").to_csv(folder/"metrics_overall.csv", index=False)
