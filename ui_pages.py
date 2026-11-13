"""Streamlit views. Market calculations and persistence remain in their own modules."""
import pandas as pd
import streamlit as st
import analyze
import config
import research
import fetch_history
from providers import PROVIDERS
from charts import coin_chart, equity_chart
from timeframes import TIMEFRAMES, completed_bars
from ui_helpers import freshness, filter_scanner, format_price, contribution_table


@st.cache_data(show_spinner=False, ttl=60)
def candles(symbol, timeframe, provider=None):
    return analyze.load_data(symbol, timeframe, provider=provider)


def selected_provider():
    return st.session_state.get("provider", config.ACTIVE_PROVIDER)


def analysis_key():
    return (selected_provider(), st.session_state.timeframe)


def settings():
    return st.session_state.timeframe, st.session_state.threshold


def snapshot():
    timeframe, threshold = settings()
    result = st.session_state.get("analyses", {}).get(analysis_key())
    if result and result.get("engine_version") != analyze.ENGINE_VERSION:
        st.info("The analysis engine changed. Run analysis to refresh the results.")
        return None
    if result and result["threshold"] != threshold:
        st.info(f"Showing the last run at threshold {result['threshold']:.1f}. Run analysis to apply {threshold:.1f}.")
    if result and result.get("outdated"):
        st.warning("Stored data was updated. Run analysis again to refresh these results.")
    return result


def run_analysis():
    timeframe, threshold = settings()
    candles.clear()
    try:
        symbols = analyze.get_available_symbols(timeframe, provider=selected_provider())
    except Exception as exc:
        st.error(str(exc))
        return
    results, errors, skipped = [], [], []
    progress = st.progress(0, text="Reading completed candles...")
    for i, symbol in enumerate(symbols):
        try:
            df = candles(symbol, timeframe, provider=selected_provider())
            prepared = analyze.compute_indicators(completed_bars(df, timeframe)) if df is not None else None
            result = analyze.live_result(prepared, symbol, threshold, timeframe) if prepared is not None else None
            if result is not None:
                result.update(research.evaluate_history(prepared, selected_provider(), threshold, timeframe))
            if result is not None:
                result["provider"] = selected_provider()
                results.append(result)
            else:
                skipped.append(symbol)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
        progress.progress((i+1)/len(symbols), text=f"Analyzed {symbol} ({i+1}/{len(symbols)})")
    progress.empty()
    st.session_state.setdefault("analyses", {})[analysis_key()] = dict(engine_version=analyze.ENGINE_VERSION, results=results, errors=errors, skipped=skipped, threshold=threshold, run_at=pd.Timestamp.now(tz="UTC").isoformat())


def scanner():
    timeframe, threshold = settings()
    st.title("Market scanner")
    if selected_provider() == "all":
        st.info("Select a single data provider for this view. All providers is available for fetching on Data management.")
        return
    st.caption(f"{selected_provider()} | {timeframe} candles | {config.LOAD_FROM.upper()} storage | Nine directional votes | Separate ATR regime filter")
    if st.button("Run analysis", type="primary", icon=":material/query_stats:"):
        run_analysis()
    data = snapshot()
    if not data:
        st.info("Run analysis to rank the coins in storage. Fetch prices on the Data management page if needed.")
        return
    st.caption(f"Analysis run: {pd.Timestamp(data['run_at']).strftime('%Y-%m-%d %H:%M')} UTC")
    if data["errors"] or data["skipped"]:
        with st.expander("Analysis coverage"):
            if data["skipped"]:
                st.write(f"Below {config.MIN_BARS} completed bars: " + ", ".join(data["skipped"]))
            for error in data["errors"]:
                st.error(error)
    if not data["results"]:
        st.info(f"No coins with enough completed {timeframe} candles. Check storage coverage on Data management.")
        return
    frame = pd.DataFrame(data["results"])
    frame["freshness"] = frame.data_as_of.map(lambda date: freshness(date, timeframe)[0])
    a,b,c,d = st.columns(4)
    a.metric("Coins analyzed", len(frame))
    b.metric("Fresh signals", int(frame.fresh_cross.sum()))
    c.metric("Regime eligible", int(frame.regime_ok.sum()))
    d.metric("Stale datasets", int(frame.freshness.str.startswith("Stale").sum()))
    if frame.freshness.str.startswith("Stale").any():
        st.warning("Some prices are behind the latest completed candle. Refresh data before interpreting their latest signals as current.")
    st.caption("Historical returns are descriptive, in-sample, and after assumed fees/slippage; funding is excluded. No performance-based universe selection is applied.")
    st.subheader("Ranked coins")
    left, right = st.columns([2,1])
    search = left.text_input("Search symbol", placeholder="BTC, ETH...")
    minimum = right.slider("Minimum score", 0., 1., 0., .1)
    a,b,c = st.columns(3)
    fresh = a.checkbox("Fresh signals only")
    regime = b.checkbox("Regime eligible only", help="ATR exceeds 80% of its 20-bar average. This is an entry filter, not a directional vote.")
    enough = c.checkbox("Sufficient trade sample only", help=f"At least {config.MIN_RESEARCH_TRADES} non-overlapping trades. This does not establish an edge.")
    selected = filter_scanner(frame, minimum, fresh, regime, enough, search)
    st.caption(f"{len(selected)} of {len(frame)} coins | Select a row to open Coin detail. Scores are votes, not probabilities.")
    if selected.empty:
        st.info("No coins match these filters. Try lowering the score or clearing a filter.")
        return
    display = selected[["symbol","price","current_score","votes_met","regime_ok","fresh_cross","n_trades","total_return_pct","buy_hold_pct","sma_return_pct","max_drawdown_pct","sufficient_history","research_error","data_as_of","freshness"]].copy()
    display["price"] = display.price.map(format_price)
    display["data_as_of"] = pd.to_datetime(display.data_as_of, utc=True).dt.strftime("%Y-%m-%d %H:%M")
    display = display.reset_index(drop=True)
    event = st.dataframe(display, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"scanner_table_{selected_provider()}_{timeframe}_{data['run_at']}", column_config={
        "symbol": "Coin", "price": "Price (USD)",
        "current_score": st.column_config.ProgressColumn("Score", min_value=0., max_value=1., format="%.3f"),
        "votes_met": "Votes (of 9)", "regime_ok": "ATR eligible", "fresh_cross": "Fresh cross",
        "n_trades": "Trades", "total_return_pct": st.column_config.NumberColumn("Strategy return", format="%.2f%%"),
        "buy_hold_pct": st.column_config.NumberColumn("Buy-and-hold", format="%.2f%%"),
        "sma_return_pct": st.column_config.NumberColumn("SMA50/200 return", format="%.2f%%"),
        "max_drawdown_pct": st.column_config.NumberColumn("Close equity drawdown", format="%.2f%%"),
        "sufficient_history": "Enough trades", "research_error": "Research status",
        "data_as_of": "Candle start (UTC)", "freshness": "Data status",
    })
    if event.selection.rows:
        st.session_state.selected_coin = display.iloc[event.selection.rows[0]].symbol
        st.session_state[f"detail_coin_{selected_provider()}_{timeframe}"] = st.session_state.selected_coin
        st.switch_page(st.session_state.detail_page)
    st.download_button("Export filtered results", selected.to_csv(index=False).encode("utf-8"), file_name=f"scanner_{selected_provider()}_{timeframe}.csv", mime="text/csv")


def coin_detail():
    timeframe, threshold = settings()
    st.title("Coin detail")
    if selected_provider() == "all":
        st.info("Select a single data provider for this view. All providers is available for fetching on Data management.")
        return
    try:
        symbols = analyze.get_available_symbols(timeframe, provider=selected_provider())
    except Exception as exc:
        st.error(str(exc))
        return
    if not symbols:
        st.info(f"No stored coins for {timeframe}. Fetch data on Data management.")
        return
    chosen = st.session_state.get("selected_coin")
    symbol = st.selectbox("Coin", symbols, index=symbols.index(chosen) if chosen in symbols else 0, key=f"detail_coin_{selected_provider()}_{timeframe}")
    st.session_state.selected_coin = symbol
    try:
        raw = candles(symbol, timeframe, provider=selected_provider())
        df = completed_bars(raw, timeframe) if raw is not None else pd.DataFrame()
        if df.empty:
            st.info("No completed candles are available for this coin.")
            return
        df = analyze.compute_indicators(df)
    except Exception as exc:
        st.error(f"Unable to prepare {symbol}: {exc}")
        return
    last = df.iloc[-1]
    status, missing = freshness(df.index[-1], timeframe)
    a,b,c,d = st.columns(4)
    a.metric(f"{symbol} price", format_price(last.close))
    score = last.super_score
    b.metric("Super score", "Warming up" if pd.isna(score) else f"{score:.3f}")
    c.metric("Conditions met", "--" if pd.isna(score) else f"{round(score*9)} of 9")
    d.metric("Completed bars", len(df))
    st.caption(f"{timeframe} | Latest candle start: {df.index[-1]} UTC | {status}")
    if missing:
        st.warning("This dataset is stale. Update prices on Data management.")
    if pd.isna(score):
        st.info("The combined score needs 200 completed candles to finish warming up.")
    st.caption("ATR regime: " + ("Eligible" if last.regime_ok else "Not eligible") + " â€” ATR must exceed 80% of its 20-bar average.")
    df["cross"] = analyze.crosses(df, threshold)
    a,b = st.columns(2)
    start = a.date_input("From", value=df.index[max(0,len(df)-180)].date(), min_value=df.index[0].date(), max_value=df.index[-1].date(), key=f"start_{selected_provider()}_{symbol}_{timeframe}")
    end = b.date_input("Through", value=df.index[-1].date(), min_value=df.index[0].date(), max_value=df.index[-1].date(), key=f"end_{selected_provider()}_{symbol}_{timeframe}")
    if start > end:
        st.warning("Choose a start date on or before the end date.")
        return
    view = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end)+pd.Timedelta(days=1))]
    if view.empty:
        st.info("There are no candles in this date range.")
        return
    show_bollinger = st.checkbox(
        "Show Bollinger Bands", key="coin_detail_bollinger",
        help="20-candle moving average with upper and lower bands two standard deviations away.")
    st.plotly_chart(coin_chart(view, threshold, show_bollinger=show_bollinger),
                    width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.caption(f"Triangle markers show crosses of {threshold:.1f} with the ATR regime filter met. Zoom or hover to inspect; double-click to reset. The score is not a probability of profit.")
    st.subheader("Latest indicator votes")
    st.dataframe(contribution_table(last), hide_index=True, width="stretch", column_config={"Contribution":st.column_config.ProgressColumn("Score contribution", min_value=0., max_value=1/9, format="%.3f")})
    st.caption("Each of nine directional votes contributes 1/9. ATR is a separate entry filter. Votes overlap and are not independent evidence.")
    st.subheader("Historical strategy comparison")
    costs = config.BACKTEST_COSTS[selected_provider()]
    st.caption(f"Assumptions per side: fee {costs['fee']:.3%}, slippage {costs['slippage']:.3%}. Full-capital long positions, no leverage. Funding excluded; these are configurable assumptions, not exchange fee quotes.")
    st.caption(f"Super: next-open entry after a crossing; stop {config.STOP_ATR} ATR, target {config.TP_ATR} ATR, score exit below threshold minus {config.EXIT_SCORE_DROP}, maximum {config.MAX_HOLD} bars. Stop wins ambiguous intrabar touches.")
    try:
        comparisons = research.compare_strategies(df, selected_provider(), threshold, timeframe)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        if comparisons["super"]["start"] is None:
            st.info("Research requires a completed candle after indicator warmup.")
        else:
            labels = {"super": "Super Indicator", "buy_hold": "Buy and hold", "sma": "SMA50/200"}
            rows = [{"Strategy": labels[name], **result["stats"]} for name, result in comparisons.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Identical evaluation dates: {comparisons['super']['start']} through {comparisons['super']['end']} UTC. Research uses all stored completed candles, independent of the chart date range.")
            st.line_chart(pd.DataFrame({labels[name]: result["equity"] for name, result in comparisons.items()}), y_label="Equity (initial capital = 1)")
            if comparisons["super"]["stats"]["n_trades"] < config.MIN_RESEARCH_TRADES:
                st.info(f"Limited evidence: fewer than {config.MIN_RESEARCH_TRADES} Super Indicator trades. Historical results do not establish an edge.")
            with st.expander("Super Indicator trades"):
                st.dataframe(comparisons["super"]["trades"], hide_index=True, width="stretch")
    st.caption("Results are in-sample descriptions, not holdout evidence. Equity includes open positions at candle closes after estimated liquidation costs; drawdown excludes intrabar extremes. Positions close at the end of the dataset. Missing intervals disable research. SMA holds when SMA50 > SMA200, executing at the next open.")
    with st.expander("Crossing dates in the selected range"):
        st.dataframe(view.loc[view.cross, ["close","super_score"]].rename(columns={"close":"Price","super_score":"Score"}), width="stretch")


def inventory(providers):
    """Inspect all stored timeframes using the currently configured analysis backend."""
    rows, errors = [], []
    for provider in providers:
        for timeframe in TIMEFRAMES:
            try:
                for symbol in analyze.get_available_symbols(timeframe, provider=provider):
                    raw = candles(symbol, timeframe, provider=provider)
                    if raw is None:
                        continue
                    df = completed_bars(raw, timeframe)
                    rows.append({"Provider":provider,"Coin":symbol,"Timeframe":timeframe,"Stored bars":len(raw),"Completed bars":len(df),"First candle (UTC)":df.index.min() if len(df) else pd.NaT,"Latest candle (UTC)":df.index.max() if len(df) else pd.NaT,"Status":freshness(df.index.max() if len(df) else None,timeframe)[0]})
            except Exception as exc:
                errors.append(f"{provider} / {timeframe}: {exc}")
    return pd.DataFrame(rows), errors


def fetch_policy(provider):
    policy = fetch_history.load_settings(config.DB_PATH, provider,
                                        config.PROVIDER_FETCH_DEFAULTS.get(provider))
    with st.expander(f"Fetch exclusions: {provider}"):
        with st.form(f"provider_fetch_policy_{provider}"):
            use_limit = st.checkbox("Exclude coins above a previous missing-bar limit", value=policy["max_missing_bars"] is not None)
            max_missing = st.number_input("Maximum previous missing bars", min_value=0, value=policy["max_missing_bars"] if policy["max_missing_bars"] is not None else 30, step=1)
            exclude_empty = st.checkbox("Exclude coins whose previous attempt returned no bars", value=policy["exclude_no_data"])
            st.caption("Rules apply to the last actual attempt for this provider and timeframe. A missing-bar count measures trailing staleness, not gaps throughout the history. API errors are not empty responses. Settings persist across restarts.")
            if st.form_submit_button("Save exclusion settings"):
                fetch_history.save_settings(config.DB_PATH, provider,
                                            int(max_missing) if use_limit else None, exclude_empty)
                policy = fetch_history.load_settings(config.DB_PATH, provider)
                st.success("Provider exclusion settings saved.")
    return policy


def fetch_provider(provider, timeframe, limit, retry_excluded, force_refresh, progress, position, total, only_symbols=()):
    """Run one provider/timeframe combination independently, including setup failures."""
    from fetch_data import main

    outcomes, summary, error = [], None, None
    prefix = f"{provider} / {timeframe} ({position}/{total})"
    progress.progress((position-1)/total, text=f"{prefix}: Discovering available pairs...")

    def update(event):
        nonlocal summary
        fraction = event.get("completed", 0)/max(1, event.get("total", 1))
        progress.progress((position-1+fraction)/total,
                          text=f"{prefix}: {event['event'].capitalize()} {event.get('symbol', timeframe)}")
        if event["event"] in ("success", "error", "no_data", "no_completed_data", "skipped", "current"):
            outcomes.append({"Coin": event["symbol"], "Result": event["event"].capitalize(),
                             "Bars": event.get("bars"),
                             "Message": event.get("error", event.get("reason", "Saved"))})
        if event["event"] == "finished":
            summary = event

    try:
        scope = {"only_symbols": only_symbols} if only_symbols else {}
        main(limit=limit, timeframe=timeframe, on_progress=update,
             ignore_exclusions=retry_excluded, force_refresh=force_refresh, provider=provider, **scope)
    except Exception as exc:
        error = str(exc)
    key = (provider, timeframe)
    analysis = st.session_state.get("analyses", {}).get(key)
    if analysis is not None and (summary is None or summary.get("succeeded") or summary.get("local_synced")):
        analysis["outdated"] = True
    return dict(provider=provider, timeframe=timeframe, at=pd.Timestamp.now(tz="UTC").isoformat(),
                summary=summary, outcomes=outcomes, error=error)


def data_management():
    timeframe, _ = settings()
    selection = selected_provider()
    providers = PROVIDERS if selection == "all" else (selection,)
    st.title("Data management")
    st.caption(f"Inspecting {config.LOAD_FROM.upper()} | Fetch destination: {config.SAVE_TO} | Providers: {', '.join(providers)}")
    policies = {provider: fetch_policy(provider) for provider in providers}
    if "binance" in providers:
        st.caption("Binance uses USDT-margined perpetual trade candles. If public REST is unavailable, "
                   "the published archive is used for the configured BTC, ETH and SOL markets. "
                   "The latest completed day may still be awaiting publication.")
    if "hyperliquid" in providers:
        st.info("Hyperliquid supplies perpetual-market candles without an API key. Only the most recent 5,000 candles per coin/timeframe are available; shorter history can limit analysis.")
    with st.container(border=True):
        st.subheader("Update market data")
        fetch_timeframes = st.multiselect(
            "Timeframes to fetch", list(TIMEFRAMES), default=[timeframe],
            key=f"fetch_timeframes_{timeframe}",
            help="Defaults to the sidebar timeframe. Add more intervals here without changing the analysis timeframe.")
        jobs = [(provider, interval) for provider in providers for interval in fetch_timeframes]
        st.caption(f"{len(jobs)} provider/timeframe combinations. Existing candle history is preserved.")
        if not jobs:
            st.info("Select at least one timeframe to fetch.")
        limit = st.number_input("Pairs to fetch (0 = all)", min_value=0, value=config.FETCH_LIMIT or 0, step=1)
        requested = st.text_input("Fetch only these symbols (optional)", placeholder="BTC, ETH, SOL",
                                  help="Comma or space separated. Leave blank for the usual provider universe.")
        only_symbols = requested.replace(",", " ").split()
        st.caption("The limit applies per provider and timeframe and counts only outdated eligible coins. Current coins use local data; discovery is cached until the next UTC day. Provider access and rate limits apply.")
        force_refresh = st.checkbox("Force refresh", help="Refresh coin discovery and request candles even when current. Exclusions still apply unless Retry previously excluded coins is also selected.")
        retry_excluded = st.checkbox("Retry previously excluded coins this run", help="Temporarily bypass both rules without changing the saved settings.")
        if st.button("Fetch / Update data", type="primary", icon=":material/sync:", disabled=not jobs):
            progress = st.progress(0, text="Preparing fetch...")
            reports = st.session_state.setdefault("fetch_reports", {})
            try:
                for position, (provider, interval) in enumerate(jobs, 1):
                    reports[(provider, interval)] = fetch_provider(
                        provider, interval, int(limit), retry_excluded, force_refresh,
                        progress, position, len(jobs), only_symbols)
            finally:
                progress.empty()
                candles.clear()
    reports = st.session_state.get("fetch_reports", {})
    for provider, interval in jobs:
        last = reports.get((provider, interval))
        if not last:
            continue
        st.caption(f"{last['provider']} — Last fetch attempt: {last['at']} | {last['timeframe']}")
        if last.get("error"):
            st.error(f"{provider} / {interval}: Unable to finish fetching: {last['error']}")
        if last["summary"]:
            report = last["summary"]
            message = f"{provider} / {interval}: {report.get('current', 0)} already current, {report.get('skipped', 0)} excluded, {report.get('fetched', report['succeeded']+report['failed'])} fetched: {report['succeeded']} succeeded, {report['failed']} failed."
            st.caption("Coin discovery: " + ("reused today's saved list" if report.get("discovery_cached") else "refreshed from provider"))
            (st.warning if report["failed"] else st.success)(message)
        if last["outcomes"]:
            st.dataframe(pd.DataFrame(last["outcomes"]), hide_index=True, width="stretch")
    st.subheader("Previous fetch attempts")
    for provider, interval in jobs:
        policy = policies[provider]
        attempts = fetch_history.load_statuses(config.DB_PATH, provider, interval)
        if attempts:
            rows = []
            for symbol, previous in attempts.items():
                rows.append({"Provider": provider, "Timeframe": interval, "Coin": symbol, "Outcome": previous["status"], "Attempted (UTC)": previous["attempted_at"],
                             "Bars returned": previous["bars_returned"], "Latest candle (UTC)": previous["latest_candle"],
                             "Previous missing bars": previous["missing_bars"],
                             "Next run": fetch_history.exclusion_reason(previous, policy) or "Eligible",
                             "Error": previous["error"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption(f"No attempts recorded yet for {provider} / {interval}. Existing prices alone do not establish a previous fetch outcome.")
    st.subheader("Storage coverage")
    if st.button("Refresh coverage"):
        candles.clear()
    frame, errors = inventory(providers)
    for error in errors:
        st.error(error)
    if frame.empty:
        st.info("No stored candles in the selected storage backend. Fetch a small test universe to get started.")
        return
    a,b,c = st.columns(3)
    a.metric("Unique coins", frame.Coin.nunique())
    b.metric("Stored timeframes", frame.Timeframe.nunique())
    c.metric("Stored candles", int(frame["Stored bars"].sum()))
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Timestamps are UTC candle starts. Staleness counts expected completed intervals since the latest candle; it does not prove that earlier history has no gaps.")
    funding_providers = tuple(p for p in providers if p in ("hyperliquid", "binance"))
    if funding_providers:
        import sqlite3
        from research_lab.data import open_ohlcv
        from research_lab.funding import funding_coverage
        with st.expander("Stored funding coverage"):
            try:
                with open_ohlcv(config.DB_PATH) as conn:
                    rates = funding_coverage(conn, funding_providers)
            except (ValueError, OSError, sqlite3.Error):
                rates = pd.DataFrame()
            if rates.empty:
                st.info("No historical funding rates stored for this provider.")
            else:
                st.dataframe(rates, hide_index=True, width="stretch")
            st.caption("These are settlement timestamps. Coverage can have gaps between them. "
                       "Fetch / Update data updates candles. Research Lab checks funding continuity before each funded run.")


def indicator_backtest():
    """Compare independent indicator entry rules, split by long and short direction."""
    from indicator_backtests import CANDIDATES, compare_coin_universe
    timeframe, _ = settings()
    provider = selected_provider()
    st.title("Indicator backtest")
    if selected_provider() == "all":
        st.info("Select a single data provider for this view. All providers is available for fetching on Data management.")
        return
    st.caption(f"{provider} | {timeframe} | Stored completed candles only | Independent long and short experiments")
    try:
        symbols = analyze.get_available_symbols(timeframe, provider=provider)
    except Exception as exc:
        st.error(str(exc))
        return
    if not symbols:
        st.info("No stored coins for this provider and timeframe. Fetch data on Data management first.")
        return
    all_coins = st.checkbox("Use all backtest coins", value=False, key=f"candidate_all_{provider}_{timeframe}")
    selected_coins = st.multiselect("Backtest coins", symbols, default=symbols[:1],
                                   key=f"candidate_coins_{provider}_{timeframe}", disabled=all_coins)
    selected_coins = list(symbols) if all_coins else selected_coins
    if not selected_coins:
        st.info("Select at least one backtest coin.")
        return
    bounds = []
    for coin in selected_coins:
        try:
            raw = candles(coin, timeframe, provider=provider)
            if raw is not None:
                raw = completed_bars(raw, timeframe).sort_index(kind="stable")
                raw = raw.loc[~raw.index.duplicated(keep="last")]
                if len(raw) >= 201:
                    bounds.append((raw.index[200],raw.index[-1]))
        except Exception:
            pass  # The batch runner records each failed read in coverage and result rows.
    if bounds:
        lower, upper = min(b[0] for b in bounds).date(), max(b[1] for b in bounds).date()
        default_start, default_end = max(b[0] for b in bounds).date(), min(b[1] for b in bounds).date()
        if default_start > default_end:
            default_start, default_end = lower, upper
    else:
        lower = upper = default_start = default_end = pd.Timestamp.now(tz="UTC").date()
    prefix = f"candidate_batch_{provider}_{timeframe}"
    default_costs = config.BACKTEST_COSTS[provider]
    with st.form(prefix):
        candidates = st.multiselect("Indicator candidates", list(CANDIDATES), default=list(CANDIDATES),
                                    format_func=lambda name: CANDIDATES[name][0])
        directions = st.multiselect("Trade directions", ["long", "short"], default=["long", "short"])
        a, b = st.columns(2)
        start = a.date_input("Backtest from (UTC)", value=default_start,
                            min_value=lower, max_value=upper)
        end = b.date_input("Backtest through (UTC)", value=default_end,
                          min_value=lower, max_value=upper)
        a, b = st.columns(2)
        fee = a.number_input("Fee per side (%)", min_value=0., max_value=10., value=default_costs["fee"]*100, step=.01, format="%.3f")/100
        slippage = b.number_input("Slippage per side (%)", min_value=0., max_value=10., value=default_costs["slippage"]*100, step=.01, format="%.3f")/100
        max_hold = st.number_input("Maximum holding bars", min_value=1, value=config.MAX_HOLD, step=1)
        condition_exit = st.checkbox("Exit when the entry condition stops holding", value=True)
        regime = st.checkbox("Require the ATR regime filter", value=False)
        atr_exits = st.checkbox("Use ATR stop and take-profit exits", value=False)
        a, b = st.columns(2)
        stop_atr = a.number_input("Stop distance (ATR)", min_value=.1, value=float(config.STOP_ATR), step=.1)
        tp_atr = b.number_input("Take-profit distance (ATR)", min_value=.1, value=float(config.TP_ATR or 6.), step=.1)
        run = st.form_submit_button("Run indicator backtest", type="primary")
    st.caption("Entry: condition switches from false to true at a completed close; execute at the next open. A condition already true at the start does not create a new crossing. Each candidate and direction runs independently with one position at a time. The sidebar Super Indicator threshold does not apply here.")
    st.caption("Default exits isolate the indicator condition and maximum holding period. ATR filtering and ATR exits are optional and off by default. Stops/targets are ignored when ATR exits are off.")
    rule_rows = [{"Indicator": CANDIDATES[name][0], "Long condition": CANDIDATES[name][1],
                  "Short condition": CANDIDATES[name][2]} for name in candidates]
    with st.expander("Rules being tested"):
        st.dataframe(pd.DataFrame(rule_rows), hide_index=True, width="stretch")
    params = dict(candidates=candidates, directions=directions, provider=provider, timeframe=timeframe,
                  start=start, end=end, fee=fee, slippage=slippage, max_hold=int(max_hold),
                  exit_on_condition=condition_exit, use_regime=regime, atr_exits=atr_exits,
                  stop_atr=stop_atr, tp_atr=tp_atr)
    # Store per provider/timeframe/coin selection so a switch cannot relabel another run.
    runs = st.session_state.setdefault("indicator_runs", {})
    key = (provider, timeframe, tuple(selected_coins))
    if run:
        try:
            with st.spinner("Testing indicator candidates..."):
                progress = st.progress(0., text="Reading selected coins...")
                try:
                    result = compare_coin_universe(
                        selected_coins, lambda coin: candles(coin, timeframe, provider=provider),
                        on_progress=lambda done,total,coin: progress.progress(done/total,text=f"Tested {coin} ({done}/{total})"),
                        **params)
                finally:
                    progress.empty()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
        else:
            runs[key] = dict(batch_version=1, params=params, result=result)
    saved = runs.get(key)
    if not saved or saved.get("batch_version") != 1:
        st.info("Choose candidates and click Run indicator backtest.")
        return
    if saved["params"] != params:
        st.info("Showing the previous run. Run indicator backtest to apply the changed settings.")
    result = saved["result"]
    st.subheader("Statistics across coins per indicator")
    st.caption(f"{provider} / {timeframe} | {len(selected_coins)} selected coins | Shared evaluation: {result['start']} through {result['end']} UTC.")
    st.caption("Mean/median coin returns give each valid coin equal weight, including zero-trade results. Pooled trade win rate, average return and profit factor weight each trade equally. Long and short remain separate. Basket statistics allocate equal starting capital per coin without rebalancing, and require valid results for every selected coin; failed experiments are never silently removed from a basket.")
    st.dataframe(result["aggregate"], hide_index=True, width="stretch")
    with st.expander("Coin and experiment coverage"):
        st.dataframe(result["coverage"], hide_index=True, width="stretch")
    st.subheader("Independent experiment results per coin")
    summary = result["per_coin"]
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "n_trades": "Trades", "win_pct": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
        "avg_trade_pct": st.column_config.NumberColumn("Average trade", format="%.2f%%"),
        "total_return_pct": st.column_config.NumberColumn("Total return", format="%.2f%%"),
        "max_drawdown_pct": st.column_config.NumberColumn("Close equity drawdown", format="%.2f%%"),
        "profit_factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
        "exposure_pct": st.column_config.NumberColumn("Exposure", format="%.1f%%"),
    })
    if summary.Error.ne("").any():
        st.warning("Some coin/indicator combinations are unavailable. See the Error column and coverage; they are included in failed-coin counts.")
    if (summary.n_trades < config.MIN_RESEARCH_TRADES).any():
        st.info(f"Some experiments have fewer than {config.MIN_RESEARCH_TRADES} trades. Small samples are weak evidence.")
    st.download_button("Export aggregate indicator statistics", result["aggregate"].to_csv(index=False).encode("utf-8"),
                       file_name=f"indicator_aggregate_{provider}_{timeframe}.csv", mime="text/csv")
    st.download_button("Export indicator comparison", summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"indicator_comparison_{provider}_{timeframe}.csv", mime="text/csv")
    if not result["coins"]:
        st.info("No coin has a valid experiment to inspect. Check coverage and date settings.")
        return
    inspected_coin = st.selectbox("Inspect coin", list(result["coins"]), key=f"{prefix}_inspect_coin_{tuple(result['coins'])}")
    detail = result["coins"][inspected_coin]
    st.caption(f"Inspecting {inspected_coin}: {detail['start']} through {detail['end']} UTC. Baselines use their own hold/exit rules. Each experiment starts with equity 1.")
    chosen = st.selectbox("Inspect experiment", list(detail["results"]),
                          format_func=lambda name: f"{detail['results'][name]['label']} / {detail['results'][name]['direction']}",
                          key=f"{prefix}_inspect_{tuple(detail['results'])}")
    experiment = detail["results"][chosen]
    a, b = st.columns(2)
    a.metric("Selected experiment: final equity", f"{experiment['equity'].iloc[-1]:.5f}")
    b.metric("Selected experiment: total return", f"{experiment['stats']['total_return_pct']:.2f}%")
    compare_all = st.checkbox("Compare all equity curves", value=False, key=f"{prefix}_compare_equity")
    st.plotly_chart(equity_chart(detail["results"], chosen, compare_all), width="stretch")
    st.caption("The chart and trade list follow Inspect experiment. Equity 1 is starting capital; 0.4143 means a return of approximately -58.57%. Enable Compare all equity curves to overlay other experiments; hover shows each curve's full name and value.")
    trades = experiment["trades"]
    st.dataframe(trades, hide_index=True, width="stretch")
    st.download_button("Export experiment trades", trades.to_csv(index=False).encode("utf-8"),
                       file_name=f"indicator_trades_{provider}_{timeframe}_{inspected_coin.replace(':','_')}_{chosen.replace(':','_')}.csv", mime="text/csv")
    st.caption("These are in-sample experiments, not proof of predictive value. No candidate is selected automatically. Funding, borrow costs, liquidity limits and venue liquidation rules are excluded. Experiments that exhaust short capital are marked unavailable; losses are never clipped. Drawdown uses candle closes and excludes intrabar extremes. Positions close at the end of the test.")

def research_settings():
    """Load workbook strategies and run batches."""

    st.title("Research lab")
    provider = st.selectbox("Research data provider", PROVIDERS, key="research_workbook_provider")
    settings_tab, batch_tab = st.tabs(["Settings", "Run batch"])
    with settings_tab:
        configs = _research_workbook_form(provider)
    with batch_tab:
        if configs:
            _research_batch_controls(configs)
        else:
            st.info("Open a valid strategy workbook in Settings before starting a batch.")


def _research_workbook_form(provider):
    """Preview a project workbook or an uploaded copy without running it."""
    from research_lab.runner import duplicate_signal_count, is_intraday_strategy
    from research_lab.workbook import WORKBOOK_PATH, load_workbook_configs

    upload = st.file_uploader("Upload a strategy workbook (optional)", type=["xlsx"],
                              key="research_workbook")
    st.caption("Without an upload, the app reads the project strategy workbook. Save your Excel changes before previewing.")
    try:
        configs = load_workbook_configs(upload or WORKBOOK_PATH, provider)
    except (ValueError, OSError) as exc:
        st.warning(f"Workbook validation failed: {exc}")
        return []
    st.success(f"{len(configs)} strategies validated.")
    st.dataframe(pd.DataFrame([
        {"Strategy": cfg["name"], "Indicator": c["indicator_name"], "Number": c["indicator_number"],
         "Category": c.get("category", ""), "Signals": "Inverted" if cfg.get("invert_signal") else "Normal",
         "Timeframes": "Intraday" if is_intraday_strategy(cfg) else "All", "Direction": cfg.get("trade_direction", "both"), "Rule": c["rule"], "Parameters": ", ".join(f"{k}={v}" for k,v in c.items()
             if k not in ("rule","indicator_name","indicator_number","weight","category")),
         "Weight": c["weight"], "Combination": cfg["combination"]["method"],
         "Threshold": cfg["combination"]["threshold"] if cfg["combination"]["method"]=="weighted" else None}
        for cfg in configs for c in cfg["components"]
    ]), hide_index=True, width="stretch")
    if any(is_intraday_strategy(cfg) for cfg in configs):
        st.caption("UTC-session VWAP uses intraday OHLCV. Daily runs list that strategy as skipped.")
    for cfg in configs:
        if duplicate_signal_count(cfg):
            st.warning(f"{cfg['name']}: multiple votes use identical rule parameters.")
    FILTER_HELP = {
        "ema_alignment": "Close above EMA for longs and below it for shorts.",
        "adx_strength": "ADX strictly above the threshold for either direction.",
        "rsi_confirmation": "RSI above the threshold for longs and below it for shorts.",
        "relative_volume": "Signal volume divided by average volume of the preceding candles must exceed the threshold.",
        "bollinger_compression": "Previous candle's band width percentile at or below the threshold.",
        "low_choppiness": "Choppiness Index at or below the threshold.",
        "high_trend_efficiency": "Net close movement as a percent of total absolute close movement, at or above the threshold.",
        "low_trend_efficiency": "Net close movement as a percent of total absolute close movement, at or below the threshold.",
        "low_realized_volatility": "Current realized volatility percentile at or below the threshold, ranked against the previous five lookback windows.",
    }
    filters = [{"Strategy": cfg["name"], **item,
                "What it checks": FILTER_HELP[item["filter_rule"]]}
               for cfg in configs for item in cfg.get("entry_filters", [])]
    if filters:
        with st.expander("Entry-only filters"):
            st.caption("All listed filters must pass on the signal candle. They gate new entries and leave exits to the strategy and execution settings.")
            st.dataframe(pd.DataFrame(filters), hide_index=True, width="stretch")
    if any(c["rule"] == "atr_pullback" for cfg in configs for c in cfg["components"]):
        with st.expander("How ATR pullback works"):
            st.markdown(
                "The rule measures the close's distance from the fast EMA in ATR units. "
                "It enters after a pullback recovers through the configured distance while "
                "price and the fast EMA remain on the trend side of the slow EMA. It exits "
                "when price reaches the fast EMA or crosses the slow EMA against the trend."
            )
    with st.expander("Resolved execution settings"):
        st.dataframe(pd.DataFrame([{"Strategy": c["name"], **c["execution"]} for c in configs]),
                     hide_index=True, width="stretch")
    return configs


def _research_batch_controls(strategies):
    """Run the selected workbook configurations against stored candles."""
    import sqlite3
    from research_lab.data import open_ohlcv, stored_pairs
    from research_lab.report import export_metrics
    from research_lab.runner import RUNS_DIR, run_batch, run_split, shared_warmup, strategy_directions

    st.subheader("Run research batch")
    by_name = {c["name"]: c for c in strategies}
    names = st.multiselect("Batch strategies", list(by_name), default=list(by_name))
    if not names:
        st.info("Select at least one strategy.")
        return
    configs = [by_name[name] for name in names]
    provider = configs[0]["provider"]
    st.caption(f"{len(configs):,} workbook configurations for {provider}. Long and short runs use separate capital.")
    try:
        with open_ohlcv(config.DB_PATH) as conn:
            pairs = stored_pairs(conn, provider)
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(f"Unable to list stored research candles: {exc}")
        return
    intervals = sorted({interval for _, interval in pairs})
    if not intervals:
        st.info("No stored SQLite candles for this provider. Fetch data before running a batch.")
        return
    all_intervals = "All stored timeframes"
    default = intervals.index(config.DEFAULT_TIMEFRAME)+1 if config.DEFAULT_TIMEFRAME in intervals else 1
    chosen_interval = st.selectbox("Batch timeframe", [all_intervals, *intervals], index=default,
                                   key=f"research_batch_interval_{provider}")
    timeframe = None if chosen_interval == all_intervals else chosen_interval
    symbols = sorted({symbol for symbol, interval in pairs if timeframe is None or interval == timeframe})
    all_symbols = "All stored assets"
    symbol_choice = st.selectbox("Batch asset", [all_symbols, *symbols], index=1,
                                 key=f"research_batch_asset_{provider}_{chosen_interval}")
    symbol = None if symbol_choice == all_symbols else symbol_choice
    coverage_on = None
    if symbol is None:
        coverage_on = st.date_input(
            "Require stored coverage on (UTC, optional)", value=None, key=f"batch_coverage_{provider}",
            help="Include only assets and timeframes with a stored candle on this date. "
                 "For historical validation, use a date known before the evaluation period.")
        if coverage_on is not None:
            try:
                with open_ohlcv(config.DB_PATH) as conn:
                    pairs = stored_pairs(conn, provider, coverage_on=coverage_on)
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(f"Unable to check coverage date: {exc}")
                return
    count = sum((symbol is None or s == symbol) and (timeframe is None or t == timeframe)
                for s, t in pairs)
    cost_groups = {tuple(c["execution"][key] for key in ("initial_capital","fee","slippage"))
                   for c in configs}
    signal_delay_bars = st.number_input(
        "Extra signal delay (candles)", min_value=0, value=0, step=1,
        help="0 executes at the next open. 1 waits one additional candle for strategy entries and signal-driven exits.")
    warmup = shared_warmup(configs) + signal_delay_bars
    st.caption(f"Matched window: {warmup} completed candles of warmup, then common dates for the largest "
               "overlapping asset group in each timeframe. Full history: each strategy starts when its own inputs are ready. "
               "Both views are saved separately.")

    history_policy = st.selectbox(
        "History with missing candles",
        ["Strict audit", "Latest continuous history", "Carry forward confirmed empty candles"],
        index=2,
        help="Latest continuous history starts after the last gap in each asset and timeframe. "
             "Warmup restarts there. Carry forward uses the previous close and zero volume "
             "for confirmed empty intervals, with trade execution blocked on those candles. "
             "Other data audit checks still apply. "
             "For a date split, each period uses the candles available at its own cutoff.")
    include_funding = False
    execution_start = None
    as_of = None
    if provider in ("binance", "hyperliquid"):
        st.caption("Zero-volume perpetual candles can mark positions but cannot fill orders.")
        include_funding = st.checkbox(
            "Include historical funding (estimated)", key=f"batch_funding_{provider}",
            help="Use stored signed rates. Candle closes approximate the oracle notional. "
                 "Runs with missing funding coverage are skipped. For a stop or target hit "
                 "inside a candle, the model charges adverse funding for the full candle.")
        if include_funding:
            start_column, end_column = st.columns(2)
            execution_start = start_column.date_input(
                "Start executing on (UTC, optional)", value=None, key=f"funding_start_{provider}",
                help="Indicators still warm up on earlier price candles. Set a date covered by funding.")
            through = end_column.date_input(
                "Backtest through (UTC, optional)", value=None, key=f"funding_end_{provider}",
                help="Last included UTC date. Leave blank to use all completed stored price candles.")
            as_of = pd.Timestamp(through, tz="UTC") + pd.Timedelta(days=1) if through else None
    evaluation_start = st.date_input(
        "Evaluation start (UTC, optional)", value=None, disabled=timeframe is None,
        help="Select one timeframe to split at UTC midnight. Development and evaluation each start flat.")
    if timeframe is None:
        evaluation_start = None
    if evaluation_start:
        st.info("The date is the first evaluation open. The same strategies run in both periods; "
                "this is historical evaluation, not an untouched forward test.")
    runs = 4 if evaluation_start else 2
    directional_count = sum(len(strategy_directions(cfg)) for cfg in configs)
    st.caption(f"{directional_count} directional strategies × {count} asset/timeframe pairs × {runs} saved views "
               f"= {directional_count*count*runs} planned strategy simulations. "
               f"{2*len(cost_groups)*count*runs} benchmark simulations are also planned.")
    if not count:
        st.info("No asset/timeframe pairs meet the selected coverage date.")
    save_equity = st.checkbox("Save equity.csv", value=False,
                              help="Keep the per-candle equity history in each run folder.")
    if st.button("Run research batch", type="primary", disabled=not count):
        progress = st.progress(0., text="Preparing research batch…")
        try:
            options = dict(symbol=symbol, timeframe=timeframe, both_views=True, coverage_on=coverage_on,
                           execution_start=execution_start, include_funding=include_funding, as_of=as_of,
                           signal_delay_bars=signal_delay_bars,
                           latest_continuous=history_policy == "Latest continuous history",
                           fill_confirmed_gaps=history_policy == "Carry forward confirmed empty candles",
                           on_progress=lambda i,n,s,t: progress.progress(i/n,text=f"{i}/{n}: {s} / {t}"))
            if evaluation_start:
                saved = run_split(configs, config.DB_PATH, RUNS_DIR,
                                  evaluation_start=evaluation_start, **options)
            else:
                saved = run_batch(configs, config.DB_PATH, RUNS_DIR, **options)
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f"Research batch failed: {exc}")
        else:
            for i, (path, _) in enumerate(saved, 1):
                progress.progress(i / len(saved), text=f"Writing CSV exports {i} of {len(saved)}…")
                try:
                    export_metrics(path, save_equity=save_equity)
                except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
                    st.warning(f"Saved {path.name}. CSV export failed. {exc}")
            folder, manifest = saved[-1]
            st.session_state["last_research_batch"] = dict(
                folder=str(folder), jobs=manifest["jobs"], name=", ".join(c["name"] for c in configs),
                provider=provider, symbol=symbol_choice, timeframe=chosen_interval, signal_delay_bars=signal_delay_bars,
                views=[dict(folder=str(path),jobs=m["jobs"],
                            label=" / ".join(part for part in
                                (m.get("phase"),m["window_view"].replace("-"," ")) if part))
                       for path,m in saved],
            )
        finally:
            progress.empty()

    last = st.session_state.get("last_research_batch")
    if last:
        from pathlib import Path
        st.caption(f"Last batch: {last['name']} | {last['provider']} / {last['symbol']} / {last['timeframe']}"
                   f" | Extra signal delay: {last.get('signal_delay_bars', 0)} candles")
        views = last.get("views")
        selected = last
        if views:
            choice = st.selectbox("Batch result view", range(len(views)), index=len(views)-1,
                                  format_func=lambda i: views[i]["label"])
            selected = views[choice]
        jobs = selected["jobs"]
        status = st.warning if jobs["failed"] or jobs["skipped"] else st.success
        status(f"{jobs['completed']} completed, {jobs['skipped']} skipped, {jobs['failed']} failed.")
        st.code(selected["folder"], language=None)
        try:
            results = (Path(selected["folder"])/"results.csv").read_bytes()
            from io import BytesIO
            # Blank metrics on skipped runs must still have a numeric column type.
            numeric_types = {
                "strategy_number": "Int64", "warmup_candles": "Int64", "execution_candles": "Int64",
                "synthetic_candles_used": "Int64", "synthetic_candles_in_test_window": "Int64",
                "non_tradable_candles": "Int64",
                "trades": "Int64", "final_equity": "Float64", "total_return_pct": "Float64", "funding_paid": "Float64",
            }
            frame = pd.read_csv(BytesIO(results), dtype={"symbol": str, **numeric_types},
                                keep_default_na=False, na_values={key: [""] for key in numeric_types})
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button("Download batch results", results, file_name="results.csv", mime="text/csv")
            signals = Path(selected["folder"])/"latest_signals.csv"
            if signals.is_file():
                st.download_button("Download latest signals", signals.read_bytes(),
                                   file_name=signals.name, mime="text/csv")
        except OSError as exc:
            st.warning(f"Unable to read saved results: {exc}")
