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


def fetch_provider(provider, timeframe, limit, retry_excluded, force_refresh, progress, position, total):
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
        main(limit=limit, timeframe=timeframe, on_progress=update,
             ignore_exclusions=retry_excluded, force_refresh=force_refresh, provider=provider)
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
                        progress, position, len(jobs))
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
    """Keep strategy editing, batch execution, and saved reports in separate tabs."""
    from research_lab.runner import BASELINE_CONFIG

    st.title("Research lab")
    settings_tab, batch_tab, reports_tab = st.tabs(["Settings", "Run batch", "Reports"])
    with settings_tab:
        selection = _research_config_form()
    with batch_tab:
        if selection is None:
            st.info("Open a valid strategy config in Settings before starting a batch.")
        else:
            strategy, using_validated = selection
            _research_batch_controls(strategy, using_validated=using_validated)
    with reports_tab:
        _research_reports(BASELINE_CONFIG.parents[1]/"research_runs")


def _research_config_form():
    """Return the last validated strategy for the current source, or its loaded defaults."""
    import hashlib
    import json
    from research_lab.runner import BASELINE_CONFIG, METHODS, load_config, validate_config

    st.caption("Configure a research strategy, validate it, then go to Run batch. You can also download its JSON. The live score is unchanged.")
    upload = st.file_uploader("Open an existing research config (optional)", type=["json"])
    try:
        if upload is not None:
            defaults = validate_config(json.loads(upload.getvalue().decode("utf-8-sig")))
        else:
            defaults = load_config(BASELINE_CONFIG)
    except (ValueError, OSError) as exc:
        st.error(f"Unable to open config: {exc}")
        return
    source = hashlib.sha256(json.dumps(defaults, sort_keys=True).encode()).hexdigest()[:16]
    labels = {"sma_trend": "SMA trend", "bollinger_reentry": "Bollinger mean reversion"}
    components = {item["rule"]: item for item in defaults["components"]}
    risk = defaults["execution"]
    combination = defaults["combination"]
    with st.form(f"research_settings_{source}"):
        name = st.text_input("Strategy name", value=defaults["name"])
        provider = st.selectbox("Research data provider", PROVIDERS, index=PROVIDERS.index(defaults["provider"]))
        rules = st.multiselect("Indicator components", list(labels), default=list(components), format_func=labels.get)
        st.caption("Only selected components are included. Windows count candles on the timeframe being tested.")
        a, b = st.columns(2)
        sma_window = a.number_input("SMA window", min_value=2, value=components.get("sma_trend", {}).get("window", 100))
        sma_weight = a.number_input("SMA weight", min_value=0., value=float(components.get("sma_trend", {}).get("weight", 1.)))
        bb_window = b.number_input("Bollinger window", min_value=2, value=components.get("bollinger_reentry", {}).get("window", 20))
        bb_width = b.number_input("Bollinger standard deviations", min_value=0., value=float(components.get("bollinger_reentry", {}).get("deviations", 2.)))
        bb_weight = b.number_input("Bollinger weight", min_value=0., value=float(components.get("bollinger_reentry", {}).get("weight", 1.)))

        method = st.selectbox("Combination rule", METHODS, index=METHODS.index(combination["method"]),
                              format_func={"majority":"Strict majority", "weighted":"Weighted score",
                                           "and":"AND (unanimous)", "or":"OR (no conflict)"}.get)
        threshold = st.number_input("Weighted score threshold", min_value=0., max_value=1.,
                                    value=float(combination["threshold"]), step=.05)
        st.caption("Weights and threshold apply only to Weighted score. Flat votes count toward the denominator. Opposing votes offset one another in the weighted score; OR goes flat on any conflict.")

        st.subheader("Execution and risk")
        a, b = st.columns(2)
        capital = a.number_input("Starting capital", min_value=0., value=float(risk["initial_capital"]))
        sizing = b.selectbox("Position sizing", ("fixed", "atr_risk"), index=int(risk["sizing"] == "atr_risk"),
                             format_func={"fixed":"Fixed fraction", "atr_risk":"ATR risk"}.get)
        fraction = a.number_input("Fixed allocation (%)", 0., 100., float(risk["fraction"])*100)
        risk_pct = b.number_input("Risk at stop (%)", 0., 100., float(risk["risk_fraction"])*100)
        exposure = a.number_input("Maximum exposure (%)", 0., 100., float(risk["max_exposure"])*100)
        fee = a.number_input("Research fee per side (%)", 0., 100., float(risk["fee"])*100, step=.01, format="%.4f")
        slippage = b.number_input("Research slippage per side (%)", 0., 100., float(risk["slippage"])*100, step=.01, format="%.4f")
        use_stop = a.checkbox("Enable stop-loss", value=risk["stop_atr"] is not None)
        stop = a.number_input("Stop distance (ATR multiples)", min_value=0., value=float(risk["stop_atr"] or 2.))
        use_target = b.checkbox("Enable take-profit", value=risk["target_atr"] is not None)
        target = b.number_input("Target distance (ATR multiples)", min_value=0., value=float(risk["target_atr"] or 3.))
        submit = st.form_submit_button("Validate research settings", type="primary")

    key = "validated_research_settings"
    if submit:
        options = {
            "sma_trend": dict(rule="sma_trend", window=sma_window, weight=sma_weight),
            "bollinger_reentry": dict(rule="bollinger_reentry", window=bb_window, weight=bb_weight, deviations=bb_width),
        }
        try:
            validated = validate_config(dict(
                version=1, name=name, provider=provider, components=[options[rule] for rule in rules],
                combination=dict(method=method, threshold=threshold),
                execution=dict(initial_capital=capital, sizing=sizing, fraction=fraction/100,
                               risk_fraction=risk_pct/100, max_exposure=exposure/100,
                               fee=fee/100, slippage=slippage/100,
                               stop_atr=stop if use_stop else None, target_atr=target if use_target else None),
            ))
        except ValueError as exc:
            st.session_state.pop(key, None)
            st.error(str(exc))
        else:
            st.session_state[key] = (source, validated)
    saved = st.session_state.get(key)
    if saved and saved[0] == source:
        st.success("Settings validated. Download to save this version.")
        output = json.dumps(saved[1], indent=2) + "\n"
        st.download_button("Download research config", output, file_name="research_strategy.json",
                           mime="application/json")
        with st.expander("Validated JSON"):
            st.code(output, language="json")

    if saved and saved[0] == source:
        return saved[1], True
    return defaults, False


def _research_batch_controls(strategy, *, using_validated):
    """Run the same provider/asset/timeframe selection as the research CLI."""
    import sqlite3
    from research_lab.data import open_ohlcv, stored_pairs
    from research_lab.runner import BASELINE_CONFIG, run_batch

    st.subheader("Run research batch")
    provider = strategy["provider"]
    source = "last validated settings" if using_validated else "loaded config"
    st.caption(f"Using {source}: {strategy['name']} ({provider}). "
               "Validate research settings in the Settings tab to apply any form edits. Results are historical only.")
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
    count = sum((symbol is None or s == symbol) and (timeframe is None or t == timeframe)
                for s, t in pairs)
    st.caption(f"{count} stored asset/timeframe pairs selected. Buy-and-hold and SMA100 benchmarks are included.")

    if st.button("Run research batch", type="primary"):
        progress = st.progress(0., text="Preparing research batch…")
        try:
            folder, manifest = run_batch(
                [strategy], config.DB_PATH, BASELINE_CONFIG.parents[1]/"research_runs",
                symbol=symbol, timeframe=timeframe,
                on_progress=lambda i,n,s,t: progress.progress(i/n,text=f"{i}/{n}: {s} / {t}"),
            )
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(f"Research batch failed: {exc}")
        else:
            st.session_state["last_research_batch"] = dict(
                folder=str(folder), jobs=manifest["jobs"], name=strategy["name"],
                provider=provider, symbol=symbol_choice, timeframe=chosen_interval,
            )
        finally:
            progress.empty()

    last = st.session_state.get("last_research_batch")
    if last:
        from pathlib import Path
        jobs = last["jobs"]
        status = st.warning if jobs["failed"] or jobs["skipped"] else st.success
        status(f"Last batch: {last['name']} | {last['provider']} / {last['symbol']} / {last['timeframe']} — "
               f"{jobs['completed']} completed, {jobs['skipped']} skipped, {jobs['failed']} failed.")
        st.code(last["folder"], language=None)
        try:
            results = (Path(last["folder"])/"results.csv").read_bytes()
            from io import BytesIO
            st.dataframe(pd.read_csv(BytesIO(results), dtype={"symbol": str}, keep_default_na=False),
                         hide_index=True, width="stretch")
            st.download_button("Download batch results", results, file_name="results.csv", mime="text/csv")
        except OSError as exc:
            st.warning(f"Unable to read saved results: {exc}")

def _research_reports(root):
    """Show saved, locally generated HTML reports without running a strategy."""
    import sqlite3
    from research_lab.report import generate_report

    st.subheader("Saved research reports")
    try:
        root = root.resolve()
        runs = {
            folder.name: folder for folder in sorted(root.iterdir(), reverse=True)
            if folder.is_dir() and folder.resolve().parent == root
            and (folder/"manifest.json").is_file()
        } if root.is_dir() else {}
    except OSError as exc:
        st.error(f"Unable to list research runs: {exc}")
        return
    if not runs:
        st.info("Complete a research batch to view its report here.")
        return
    selected = st.selectbox("Saved research run", list(runs), key="saved_research_report")
    folder = runs[selected]
    path = folder/"report.html"
    if path.resolve().parent != folder.resolve():
        st.error("The report is outside its run folder.")
        return
    if not path.is_file():
        st.info("This run has no HTML report yet. Generating it uses the saved results.")
        if not st.button("Generate report", key=f"generate_research_report_{selected}"):
            return
        try:
            with st.spinner("Building report from saved results…"):
                generate_report(folder)
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            st.error(f"Unable to generate report: {exc}")
            return
    try:
        html = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        st.error(f"Unable to open report: {exc}")
        return
    # Local sibling file links work when opening the file directly, not in srcdoc.
    head, marker, tail = html.partition("<p><a href='metrics_per_asset.csv'>")
    if marker and "</p>" in tail:
        html = head + tail.split("</p>", 1)[1]
    st.caption("Interactive view of the saved report. The original HTML and CSV files remain in the run folder.")
    st.iframe(html, height=800)
