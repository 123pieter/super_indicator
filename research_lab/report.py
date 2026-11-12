"""Offline historical reporting for a completed research batch."""
from html import escape
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from timeframes import TIMEFRAMES, get_timeframe
from .data import fill_empty_candles, load_pair, open_ohlcv
from .runner import is_intraday_strategy, strategy_identity


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
    )


def bar_exposure(index, trades):
    """Fraction of evaluation candles with exposure; intrabar exits count that candle."""
    bars = index[1:]  # First mark is starting equity, before any executable candle.
    changes = np.zeros(len(bars)+1, dtype=int)
    for t in trades.itertuples(index=False):
        start = bars.searchsorted(t.entry_candle)
        end = bars.searchsorted(t.exit_candle, side="left" if t.exit_phase == "open" else "right")
        if end > start:
            changes[start] += 1
            changes[end] -= 1
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


def _tables(results, equity, trades, experiments, *, include_overall=True):
    keys = ["timeframe", "symbol", "experiment"]
    curves = {key: rows.set_index("candle").equity.sort_index()
              for key, rows in equity.groupby(keys, sort=False)}
    logs = dict(tuple(trades.groupby(keys, sort=False)))
    empty = trades.iloc[:0]
    rows = []
    for r in results.to_dict("records"):
        key = (r["timeframe"], r["symbol"], r["experiment"])
        exp = experiments[key[2]]
        row = dict(r, strategy=exp["name"], category=_label(exp), direction=_direction(exp),
                   benchmark_group=exp["group"])
        row.update(strategy_identity(exp))
        if key in curves:
            curve, log = curves[key], logs.get(key, empty)
            row.update(metrics(curve, log, key[0], bar_exposure(curve.index, log)))
        rows.append(row)

    overall = []
    groups = results.groupby("timeframe", sort=False) if include_overall else ()
    for interval, group in groups:
        symbols = list(group.symbol.unique())
        # Require a common asset universe across strategies applicable to this timeframe.
        applicable = experiments
        if interval in TIMEFRAMES and TIMEFRAMES[interval].duration >= pd.Timedelta(days=1):
            applicable = {exp: definition for exp, definition in experiments.items()
                          if not (definition["kind"] == "strategy"
                                  and is_intraday_strategy(definition["config"]))}
        matched = [s for s in symbols if all((interval, s, exp) in curves for exp in applicable)]
        if not matched or not applicable:
            continue
        all_curves = [curves[(interval,s,exp)] for s in matched for exp in applicable]
        start = max(c.index[0] for c in all_curves)
        end = min(c.index[-1] for c in all_curves)
        if start >= end:
            continue
        for exp, definition in applicable.items():
            series = [curves[(interval,s,exp)].loc[start:end] for s in matched]
            panel = pd.concat(series, axis=1, join="inner")
            if len(panel) < 2:
                continue
            curve = panel.div(panel.iloc[0]).mean(axis=1)
            by_coin = [logs.get((interval,s,exp),empty) for s in matched]
            closed = pd.concat([t.loc[(t.entry_candle > start) & (t.exit_candle <= end)]
                                for t in by_coin], ignore_index=True)
            exposure = np.mean([bar_exposure(curve.index,t) for t in by_coin])
            overall.append(dict(timeframe=interval, experiment=exp, strategy=definition["name"],
                                category=_label(definition), direction=_direction(definition),
                                benchmark_group=definition["group"],
                                matched_coins=len(matched), requested_coins=len(symbols), **strategy_identity(definition),
                                start=start, end=end, **metrics(curve,closed,interval,exposure)))
    return pd.DataFrame(rows), pd.DataFrame(overall), curves, logs


def _price(manifest, key, curve, trades):
    interval, symbol, _ = key
    audit = next(r for r in manifest["coverage"]
                 if r["symbol"] == symbol and r["timeframe"] == interval)
    expected = audit["data_sha256"]
    with open_ohlcv(manifest["database"]) as conn:
        candles, _ = load_pair(conn, manifest["provider"], symbol, interval, manifest["as_of"])
    actual = hashlib.sha256(pd.util.hash_pandas_object(candles, index=True).to_numpy().tobytes()).hexdigest()
    if actual != expected:
        raise ValueError("Stored candles differ from the recorded batch fingerprint")
    candles = fill_empty_candles(candles, interval, audit.get("synthetic_dates_utc"))
    frame = candles.loc[curve.index]
    fig = go.Figure(go.Candlestick(x=frame.index, open=frame.open, high=frame.high, low=frame.low,
                                   close=frame.close, name="Price"))
    if "synthetic" in frame:
        filled = frame.loc[frame.synthetic]
        fig.add_scatter(x=filled.index, y=filled.close, mode="markers", name="Synthetic candle",
                        marker=dict(symbol="x", color="#a16207", size=10))
    for side, phase, label, marker in [(1,"entry","Buy to enter","triangle-up"),
                                       (-1,"entry","Sell short","triangle-down"),
                                       (1,"exit","Sell to exit","triangle-down"),
                                       (-1,"exit","Buy to cover","triangle-up")]:
        part = trades[trades.side.eq(side)]
        fig.add_scatter(x=part[f"{phase}_candle"], y=part[f"{phase}_fill"], mode="markers",
                        name=label, marker=dict(symbol=marker, size=9),
                        text=part.reason if phase == "exit" else None)
    for col, label, color in [("stop","Stop-loss","#b91c1c"),("target","Take-profit","#15803d")]:
        x, y = [], []
        for trade in trades.dropna(subset=[col]).itertuples(index=False):
            level = getattr(trade,col)
            x.extend([trade.entry_candle,trade.exit_candle,None])
            y.extend([level,level,None])
        fig.add_scatter(x=x,y=y,mode="lines+markers",marker_size=3,line=dict(color=color,dash="dot"),name=label)
    fig.update_layout(xaxis_rangeslider_visible=False, height=470)
    return fig


def _table(frame, caption):
    return (f"<h2>{escape(caption)}</h2><input placeholder='Filter this table' aria-label='Filter {escape(caption)}'>"
            + frame.to_html(index=False, escape=True, na_rep="", float_format=lambda x:f"{x:.6g}"))


_TABLE_JS = """
document.querySelectorAll("input").forEach(input => {
  const table = input.nextElementSibling;
  input.addEventListener("input", () => {
    for (const row of table.tBodies[0].rows)
      row.hidden = !row.textContent.toLowerCase().includes(input.value.toLowerCase());
  });
  [...table.tHead.rows[0].cells].forEach((header, col) => {
    header.tabIndex = 0; let sign = -1;
    const sort = () => {
      const body = table.tBodies[0];
      [...body.rows].sort((a,b) => {
        const x = a.cells[col].textContent.trim(), y = b.cells[col].textContent.trim();
        if (!x || !y) return Number(!x) - Number(!y);
        if (x === y) return 0;
        const number = v => v === "inf" ? Infinity : v === "-inf" ? -Infinity : Number(v);
        const nx = number(x), ny = number(y);
        return sign * ((!isNaN(nx) && !isNaN(ny)) ? (nx === ny ? 0 : nx < ny ? -1 : 1)
                                                : x.localeCompare(y));
      }).forEach(row => body.append(row));
      sign *= -1;
    };
    header.addEventListener("click", sort);
    header.addEventListener("keydown", event => { if (event.key === "Enter") sort(); });
  });
});
"""


def generate_report(run, *, symbol=None, timeframe=None, experiment=None):
    """Create one HTML report and two metric CSVs alongside the unchanged batch artifacts."""
    folder = Path(run)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    experiments = {x["id"]:x for x in manifest["experiments"]}
    results = _read(folder,"results.csv")
    equity = _read(folder,"equity.csv",("candle",))
    trades = _read(folder,"trades.csv",("entry_candle","exit_candle","signal_candle"))
    full = manifest.get("window_view") == "full-history"
    per_asset, overall, curves, logs = _tables(results,equity,trades,experiments,include_overall=not full)
    chosen = next((k for k in curves if (symbol is None or symbol == k[1])
                   and (timeframe is None or timeframe == k[0])
                   and (experiment is None or experiment == k[2])), None)
    if chosen is None and any(x is not None for x in (symbol,timeframe,experiment)):
        raise ValueError("No completed experiment matches the requested detail")
    for frame in (per_asset, overall):
        if not full and "total_return_pct" in frame:
            frame.sort_values("total_return_pct",ascending=False,inplace=True)
    per_asset.to_csv(folder/"metrics_per_asset.csv",index=False)
    overall.to_csv(folder/"metrics_overall.csv",index=False)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Research batch report</title>",
        "<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse;font-size:.9rem}"
        "th,td{padding:.35rem .6rem;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}"
        "th{cursor:pointer;background:#eef2ff}input{margin:.5rem 0;padding:.4rem}</style>",
        "<h1>Research batch report</h1>",
        ("<p><strong>Full history.</strong> Each strategy starts when ready. Dates can differ; "
         "use the paired Matched window run for comparisons.</p>" if full else
         "<p><strong>Matched window.</strong> Each timeframe uses the same execution dates for every included asset and strategy.</p>"
         if manifest.get("window_view")=="matched-window" else ""),
        f"<p>{escape(manifest['provider'])} | {escape(manifest['evaluation'])} | "
        f"{manifest['strategy_configs']} configs; {manifest['unique_strategy_settings']} distinct settings; "
        f"{manifest['planned_jobs']} planned jobs. {escape(str(manifest['jobs']))}</p>",
        ("<p>Synthetic candles carry forward the previous close with zero volume. "
         "The per-asset table counts those used by each run, including warmup, and those in its "
         "test window. Orders execute only on real candles.</p>"
         if manifest.get("history_policy") == "carry forward confirmed empty candles" else ""),
        ("<p>Long-only and short-only results use independent starting capital. Opposite signals "
         "go to cash. Buy-and-hold and SMA100 long/cash are long-side references.</p>"
         if "directional_strategy_configs" in manifest else ""),
        "<p>Shorts are hypothetical; funding, borrow and liquidation are excluded. Rankings are descriptive. "
        "Sharpe/Sortino assume zero cash return and annualise using the candle interval. Exposure estimates "
        "bars with a position. Below 30 trades is flagged; 30+ does not establish reliability.</p>",
        "<p>Overall rows use the same completed assets and overlapping dates for every experiment within "
        "each timeframe. Intraday-only strategies are excluded from daily comparisons. Equity is allocated equally at the common start without rebalancing. Trade "
        "statistics pool only trades entered after the common start and closed by its end. No overall "
        "row is produced if there is no matched sample.</p>",
        "<p><a href='metrics_per_asset.csv'>Per-asset CSV</a> · <a href='metrics_overall.csv'>Overall CSV</a>"
        " · <a href='trades.csv'>All trades CSV</a> · <a href='manifest.json'>Run manifest</a></p>",
        _table(overall,"Overall by timeframe") if len(overall) else ("" if full else "<p>No matched overall sample.</p>"),
        _table(per_asset,"Per asset and strategy"),
        "<script>"+get_plotlyjs()+"</script>",
    ]
    def add(fig):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    heatmap_groups = () if full else per_asset.groupby("timeframe",sort=False)
    for interval, group in heatmap_groups:
        done = group[group.status.eq("completed")]
        if len(done):
            heat = done.pivot(index="symbol",columns="experiment",values="total_return_pct")
            parts.append(f"<h2>{escape(interval)} return heatmap (%)</h2>")
            add(go.Figure(go.Heatmap(x=[f"{escape(experiments[e]['name'])} ({escape(e)})" for e in heat.columns],
                                     y=[escape(s) for s in heat.index],
                                     z=heat.to_numpy(), colorscale="RdBu",zmid=0)))
    if chosen is not None:
        interval, asset, exp = chosen
        curve, log = curves[chosen], logs.get(chosen,trades.iloc[:0])
        parts.append(f"<h2>Inspect {escape(asset)} / {escape(interval)} / {escape(experiments[exp]['name'])}</h2>")
        try:
            add(_price(manifest,chosen,curve,log))
        except (ValueError, KeyError, sqlite3.Error, OSError) as exc:
            parts.append(f"<p>Price chart unavailable: {escape(str(exc))}.</p>")
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Equity","Drawdown (%)"))
        fig.add_scatter(x=curve.index,y=curve,mode="lines",name="Equity",row=1,col=1)
        fig.add_scatter(x=curve.index,y=(curve/curve.cummax()-1)*100,mode="lines",name="Drawdown",row=2,col=1)
        add(fig)
        parts.append(_table(log,"Inspected trade log"))
    path = folder/"report.html"
    path.write_text("\n".join(parts+["<script>"+_TABLE_JS+"</script></html>"]),encoding="utf-8")
    return path
