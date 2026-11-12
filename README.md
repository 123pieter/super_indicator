# Super Indicator

A Streamlit crypto research app combining nine equally weighted directional conditions with a separate ATR volatility filter.

## Setup and run

All project files, credentials, data, configuration, and Python packages live in this folder. No Codex workspace or runtime is required. A standard Python 3.13 installation supplies the base interpreter for the local virtual environment.

From this project folder in PowerShell:

```powershell
# Only needed when creating a new environment:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Verify and launch:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe check_environment.py
.\run_app.ps1
```

The launcher always uses this folder's `.venv` and working directory. In PyCharm, select `.venv/Scripts/python.exe` as the project interpreter. Restart Streamlit after changing its environment.

For fetching, put only the Polygon API key in the ignored project-local `secrets.txt`, or set `POLYGON_API_KEY`. Local analysis needs no API key. Never commit credentials.

## App screens

- **Scanner:** rank coins, filter by score/signals/history/symbol, inspect freshness, and export results. Select a row to open that coin. Analysis snapshots are retained per timeframe for the current session.
- **Coin detail:** interactive price candles, volume, score history, threshold-crossing markers, date range, indicator votes, and historical outcomes.
- **Data management:** fetch controls, live per-coin progress, errors, and coverage across timeframes. Fetching runs synchronously. Update analysis after fetching new prices.

## Configuration and storage

Choose Polygon or Hyperliquid in the sidebar. Edit `config.py` for default provider, storage, dates, minimum history, threshold, and request delay. The default fetch limit is two pairs for testing; zero or None selects all pairs.

- Daily (`1d`) is the default. Supported intervals: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`.
- CSV: `data/<provider>/<timeframe>/<symbol>.csv`, e.g. `data/polygon/1d/BTC.csv` and `data/hyperliquid/1d/BTC.csv`.
- One SQLite database: `crypto_data.db`. Candles are in `ohlcv`, keyed by `(provider, symbol, timeframe, date)`. Fetch observations, policies, and discovery snapshots also live in this database.
- `LOAD_FROM` chooses the analysis source; `SAVE_TO` chooses csv, sqlite, or both.
- Updates overlap recent history and preserve other symbols/timeframes. Candle timestamps are UTC starts; unfinished candles are excluded.
- Provider access depends on your Polygon subscription. Weekly/monthly intervals require calendar-aware completion rules before being added.

```powershell
.\.venv\Scripts\python.exe fetch_data.py --timeframe 1d --limit 2
```

## Calculations

Scores start after the 200-bar SMA warm-up. RSI14, ATR14 and ADX14 use Wilder smoothing seeded with an arithmetic average. A flat stochastic range remains undefined; CCI uses vectorised 20-bar mean absolute deviation. The score is the fraction of nine directional conditions met. For example, seven votes give 0.7778; at threshold 0.7 at least seven votes are required. These are correlated votes, not probabilities.

ATR is an entry filter: ATR14 must exceed 80% of its 20-bar average. A fresh signal requires a valid previous score below the selected threshold, a current score at or above it, and an eligible ATR regime. Chart markers and Scanner use the same rule. `analyze_coin()` computes current signals only and never runs historical backtests.

### Historical research

`research.py` simulates unlevered, full-capital long positions, with one position at a time. Entries execute at the next candle open. Exit rules: fixed ATR stop/target from the signal candle ATR, a score below threshold minus `EXIT_SCORE_DROP` (next-open exit), `MAX_HOLD` candles including the entry candle (close exit), or the final dataset close. Stop wins when both stop and target are touched intrabar. Opening gaps execute at the open; a scheduled score exit executes before later intrabar movements. An exit-candle crossing can re-enter at the following open.

Configure `BACKTEST_COSTS` independently per provider in `config.py`. Defaults are assumptions: 0.1% fee and 0.05% slippage per side. Entry cost is `open * (1+slippage) * (1+fee)`; exit proceeds are `exit * (1-slippage) * (1-fee)`. Funding, leverage, liquidation, market impact and liquidity constraints are not modelled. Hyperliquid results explicitly exclude funding.

Buy-and-hold and SMA50/200 benchmarks use the identical evaluation dates and cost assumptions. SMA holds when the previous completed candle has SMA50 > SMA200, executing changes at the next open. Evaluation starts at the open after SMA200 first becomes ready, so no benchmark uses unavailable information. All strategies liquidate at the final close.

Equity starts at 1 and is marked every candle close after estimated liquidation costs. Drawdown includes unrealised close-to-close losses inside trades and the first trade loss, but excludes intrabar extremes. Full precision is retained internally. No-trade win rate, average trade return and profit factor remain undefined. `MIN_RESEARCH_TRADES` is only a descriptive sample-size flag, not proof of an edge.

The old preferred-universe rule is retired. Historical results are descriptive **in-sample** results; they do not select or rank a universe by performance. No holdout optimisation or walk-forward selection is active. `analyze_v2.py` is retained as the submitted proposal and is not imported by the app.

Research requires uninterrupted timestamps for the selected timeframe; missing intervals produce a visible warning and unavailable return metrics while the coin's live signal remains visible. The freshness label checks only trailing missing candles. Coin detail research uses all completed stored candles, independently of the chart date filter.

## Automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Every test has a plain-language description. Tests use synthetic data, temporary storage, and mocked provider calls; they never use real credentials or market-data APIs.

Tests cover indicator formulas and edge cases, live crossings, exact backtest executions, costs, equity drawdowns, benchmark alignment, CSV/SQLite persistence, schema/primary keys, timeframe isolation, fetch outcomes, local credential configuration, chart structure, and offline Streamlit interactions.

## Code layout

- `app.py`, `ui_pages.py`: navigation and views.
- `ui_helpers.py`, `charts.py`: display logic and Plotly figures.
- `analyze.py`: causal indicator calculations and current signals.
- `research.py`: historical trades, equity, statistics, and basic benchmarks.
- `fetch_data.py`, `fetch_history.py`, `providers/`: incremental fetching, persisted outcomes/policies, and provider access.
- `storage.py`, `timeframes.py`, `config.py`: persistence, intervals, and settings.
- `check_environment.py`, `run_app.ps1`: environment verification and launch.
- `tests/`: automated tests and synthetic data helpers.

## Fetch exclusions and previous attempts

Open **Data management > Fetch exclusions: polygon**. Two independent settings are saved for the active provider:

- Enable a missing-bar limit to skip coins whose **previous attempt** recorded more than that count. Equality to the limit remains eligible. Disabling the rule removes the limit.
- Enable the no-data rule to skip coins whose previous attempt successfully returned zero bars. API/rate-limit/network failures are recorded as errors and do not trigger this rule.

Both exclusions start disabled. The provider's policy applies independently to each timeframe's observations. Save settings before fetching. Configuration defaults are in `PROVIDER_FETCH_DEFAULTS`; persisted UI settings override them for that provider.

**Previous fetch attempts** shows each coin's last actual attempt, result, returned row count, latest known completed candle, saved missing-bar count, error, and next-run eligibility. Counts measure trailing staleness, not historical internal gaps. With no known completed candle, the missing count is unknown; the no-data rule still works.

Observations are stored in `fetch_status`, keyed by `(provider,timeframe,symbol)`. Settings are in `provider_fetch_settings`, keyed by provider. This metadata always uses the project database, even if candles are saved only to CSV. Only the latest actual attempt is retained. Existing prices alone do not imply a previous fetch outcome; observations start with the next fetch after enabling this feature.

Excluded coins do not use the pair limit or make individual bar requests, and skipping does not overwrite their last observation. Provider symbol discovery is reused within its UTC calendar day; it runs when that cache expires or Force refresh is selected. The one-run **Retry previously excluded coins this run** option bypasses both rules and updates the outcomes without changing the saved policy. The equivalent CLI option is:

```powershell
.\.venv\Scripts\python.exe fetch_data.py --timeframe 1d --limit 2 --ignore-exclusions
```

Changing either rule also lets you revisit excluded coins. Empty responses and responses containing only unfinished candles remain distinct. Per-coin failures retain stored prices. The completion summary shows successes, failures (including empty responses), and exclusions.

`tests/test_fetch_history.py` covers persistence, provider/timeframe isolation, thresholds, subsequent-run exclusions, retries, error classification, and saved staleness. UI tests verify saved policies survive a new session. All provider calls in tests are mocked.

## Avoiding unnecessary API requests

A normal fetch checks actual completed candles in local storage. Already-current coins are skipped without a candle request, and skipped checks do not overwrite previous attempt records. A previous success alone is insufficient if the stored candles are now missing or outdated.

Coin discovery is cached per provider in the `provider_symbols` table until the next UTC day, independently of candle timeframe. The cache survives restarts and is shared across timeframes for that provider. Failed or empty discovery responses do not replace a valid snapshot. New listings normally become visible at the next daily discovery refresh.

When discovery is cached and every eligible coin is current or excluded, a normal rerun makes **zero provider API requests** and needs no API credentials. The first run after this feature is installed must populate the discovery cache once. At the next UTC day, discovery is refreshed and daily prices are checked against the newly completed day. Hourly and minute prices become outdated at their own completion boundaries.

Current and excluded coins do not consume the configured fetch limit; it applies to pending eligible coins. If the limit left other coins outdated, a later rerun can fetch those coins. Current CSV data can repair a missing/outdated SQLite destination locally, or vice versa, when SAVE_TO is both.

**Force refresh** refreshes discovery and requests candles again even if already current. It still respects exclusions; combine it with **Retry previously excluded coins this run** to bypass those rules too. Both overrides are temporary.

```powershell
.\.venv\Scripts\python.exe fetch_data.py --timeframe 1d --force-refresh
```

The completion summary distinguishes already current, excluded, fetched, successful, and failed coins, and states whether discovery came from the saved list. `tests/test_fetch_freshness.py` verifies zero-call reruns, UTC/intraday boundaries, missing local data, per-provider caching, force behavior, and local destination synchronization with mocked providers.

## Hyperliquid provider

Select **hyperliquid** under **Data provider**. The same scanner, charts, timeframes, exclusions, incremental updates, daily discovery cache, and zero-call freshness checks apply. Native perpetual markets are discovered from public `/info` metadata; `candleSnapshot` supplies their candles. These prices describe Hyperliquid perpetual markets, while Polygon supplies its own crypto market bars.

No Hyperliquid API key, wallet, signing library, or new Python dependency is needed. The JSON POST and metadata-discovery pattern was adapted from the earlier arbitrage scanner; all implementation lives here and does not import that project.

Hyperliquid exposes only its **latest 5,000 candles per interval**. Requests are clamped to that available window. An older START_DATE cannot retrieve candles beyond it. Newly listed markets or short intraday windows may not meet MIN_BARS. Incremental collection retains previously stored candles as they age out of the public API window. Hyperliquid uses HYPERLIQUID_REQUEST_DELAY between coins. Polygon uses REQUEST_DELAY between individual HTTP requests, including discovery and pagination. HTTP 429 retries wait at least 60 seconds (or longer if Retry-After requests it), with at most three attempts per page.

The default HYPERLIQUID_DEXES tuple contains only the native perpetual dex (`""`). Optional builder-deployed dexes such as `"xyz"` or `"km"` can be added there. Their raw names, e.g. `xyz:CL`, retain the namespace and use percent-escaped CSV filenames for Windows compatibility. Changing the dex list requires Force refresh to refresh discovery immediately.

Candles share one table and are isolated by its provider key. Every candle query and upsert filters/includes the provider. Fetch state, exclusions, and discovery are also keyed by provider. Analysis snapshots and chart caches are keyed by provider/timeframe, preventing a switch from displaying the other supplier's data. Local paths are resolved by config.storage_paths.

```powershell
.\.venv\Scripts\python.exe fetch_data.py --provider hyperliquid --timeframe 1d --limit 2
```

Provider conversion, limits, errors, credential-free fetching, cross-provider candle isolation, namespaced filenames, UI switching, and independent policy forms are covered by offline automated tests. The public API reference is https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint.

## Shared provider storage

Both suppliers now use `crypto_data.db`; the separate Hyperliquid candle database has been consolidated and removed. The `ohlcv` columns are `provider`, `date`, `symbol`, `timeframe`, `open`, `high`, `low`, `close`, and `volume`, with primary key `(provider, symbol, timeframe, date)`.

CSV layout is symmetric: `data/polygon/<timeframe>/<symbol>.csv` and `data/hyperliquid/<timeframe>/<symbol>.csv`. Polygon's existing timeframe folders were moved into its provider folder; Hyperliquid's existing CSV layout is unchanged. Namespaced symbols are percent-escaped in filenames and decoded when listed.

Provider separation is covered by tests that store identical coin/timeframe/timestamp keys with different prices in the shared table, then read each provider independently. No ongoing migration scripts or compatibility tables are required.


## Individual indicator backtests

Open **Indicator backtest** and select the provider/timeframe in the sidebar. Choose
one or more stored coins, any of the nine directional indicator candidates, and long, short,
or both directions. Long and short are independent experiments, not a combined
portfolio or an automatic reversal strategy. Expand **Rules being tested** for the
exact bullish and bearish conditions. A failed bullish condition is not automatically
a bearish condition: for example, CCI shorts require -100 < CCI < 0.

Entries require a false-to-true condition transition at a completed close and execute
at the next open. By default, condition failure exits at the next open, or the trade
closes after the chosen maximum holding bars. Disable the condition exit to test
entries with a fixed maximum holding horizon. Persistent conditions do not re-enter
until a new crossing. Positions are closed at the final selected candle close.

All experiments retain 200 bars of common warmup and use identical selected UTC dates,
fee/slippage assumptions, and one position per experiment. Indicators are computed
before date filtering; two prior candles preserve a crossing just before the first
evaluation open. The sidebar combined-score threshold does not affect these tests.
ATR regime filtering and ATR stop/target exits are independently optional and disabled
by default. Stop/target distance controls only apply when ATR exits are enabled.

The comparison includes long buy-and-hold and SMA50/200 baselines. SMA is bullish
when SMA50 > SMA200 and bearish when SMA50 < SMA200, trading at the next open.
Baselines keep their own holding/exit rules instead of the candidate's timed/ATR exits.
Inspect total net return, average trade, win rate, profit factor, drawdown, exposure,
trade counts, individual trade lists and equity curves. Comparison and trade CSVs
can be downloaded. Results are retained separately for each provider/timeframe/coin selection.
Changed submitted settings require another run; settings here do not alter live signals.

Short positions use 1x collateralised exposure: quantity is capital divided by entry
fill plus entry fees; equity is initial collateral plus signed price P&L minus entry
and estimated exit fees. Short-sale proceeds are not reinvested. Funding, borrow,
liquidity and exchange liquidation rules are excluded. If a short experiment exhausts
capital, its result is marked unavailable, rather than clipping losses or allowing
negative capital to compound. Missing evaluation candles disable comparisons.

These are descriptive in-sample tests. Testing many candidates can produce apparent
winners by chance; passing these tests does not establish predictive value. No
indicator or parameter setting is automatically adopted by the live Super Indicator.
`tests/test_indicator_backtests.py` covers independent rules, window alignment,
warmup, optional filters/exits and validation. Short-execution tests are included in
`tests/test_research.py`, and offline page tests in `tests/test_app.py`.


### Multi-coin comparisons

Select multiple **Backtest coins**, or enable **Use all backtest coins**, and choose
indicator candidates and directions. Run once to evaluate the complete coin x
candidate x direction grid. A single shared evaluation period is the intersection
of selected coins' available history after 200 candles of warmup and your requested
dates. Later-listed coins can therefore shorten the common test. Invalid histories
produce explicit failed rows without stopping the remaining coins.

**Independent experiment results per coin** contains every selected combination,
including failures and actual evaluation dates. **Statistics across coins per indicator**
keeps directions separate and reports requested/valid/failed coins, coins with trades,
profitable coins, mean/median/best/worst coin return and mean coin drawdown.
Zero-trade valid coins retain a zero return and participate in equal-coin means.
Means use valid coins only; read the coverage counts alongside them.

Total trades, pooled win rate, average trade return and pooled profit factor use all
valid trades. Pooled profit factor is the sum of positive trade percentage returns
divided by the absolute sum of negative trade percentage returns, not the average
of per-coin ratios or a dollar-profit ratio. More active coins carry more weight in
these pooled trade statistics, but not in equal-coin return summaries.

Basket return and drawdown use the mean of equally allocated normalised equity
curves, without rebalancing. Basket drawdown is calculated from that combined curve,
not the mean of coin drawdowns. Basket statistics are unavailable if any requested
coin lacks a valid experiment or aligned timestamps: failures are never silently
dropped or converted into zero-loss cash. These baskets are hypothetical independent
allocations, not an execution or portfolio-sizing recommendation.

Export both per-coin and aggregate tables as CSV. Choose **Inspect coin** and then
**Inspect experiment** to view the exact matching equity curve and individual trades.
Runs remain separate by provider, timeframe and selected coin list. All data reads
are local; this feature does not fetch market data or change the live indicator.

### Fetching several timeframes

On **Data management**, choose a provider (or **All providers**) in the sidebar,
then select the intervals under **Timeframes to fetch** and click **Fetch / Update data**.
The list defaults to the sidebar timeframe. Fetches run sequentially for each
provider and timeframe combination, with separate progress and results. The pair
limit applies to each combination, and exclusion settings remain provider-specific.
The sidebar timeframe still controls analysis.

## Research lab

Edit the project workbook in the research_configs folder. Its **User Manual**
explains how to define indicator instances and combine them into strategies.
In Streamlit, open **Research lab → Settings** to preview the project workbook
or upload another workbook, then choose what to run under **Run batch**.

From the project PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -B -m research_lab check-workbook --provider hyperliquid
.\.venv\Scripts\python.exe -B -m research_lab run-batch --provider hyperliquid --symbol AAVE --timeframe 1d
```

Use **--workbook** to select another workbook and repeat **--strategy-name**
to select specific strategies. Saved runs keep their manifests so their reports
can be opened independently of later workbook edits.

### Donchian breakout

Register an indicator instance in the workbook using the function key
`donchian_breakout`. Optional parameter rows are `entry_window` (default 55)
and `exit_window` (default 20), both in candles and at least 2. Use a new
indicator number and link it to a strategy in the strategy_components sheet.

A close above the highest high (below the lowest low) of the prior entry
channel signals long (short). A close beyond the opposite exit channel
signals flat, with no same-candle reversal. The signal candle does not
contribute to either channel. Execution is at the next candle open;
configured ATR stops and targets still apply.


Research batches run each selected strategy twice: long only and short only.
Each direction starts with its own capital, and an opposite signal is treated
as cash. Reports show separate directional trade logs and equity curves.
Buy-and-hold and SMA100 long/cash remain long-side benchmarks.

### Donchian filter comparison

Select **D0–D3** together under **Research lab → Run batch**. The variants
share Donchian 40/20, sizing, costs and exits. Filters in the optional
**strategy_entry_filters** sheet are ANDed at the signal close. A rejected
transition is not retried when filters later pass, and filters never exit a trade.

For daily candles, **Evaluation start (UTC)** (CLI: **--evaluation-start YYYY-MM-DD**)
creates separate development and evaluation runs from one SQLite read snapshot.
Both use the same frozen settings and start with independent capital. The
evaluation needs a candle exactly at the boundary and sufficient prior warmup.
A pre-boundary signal state is not treated as a fresh entry. The benchmarks
initialize from the boundary, using only information already available.
Open either phase in **Reports**. Historical evaluation is not an untouched forward test.

### Research batch windows

The app and batch command save two views. **Full history** uses each strategy's
own indicator, filter and risk warmup; use it to inspect an individual history.
**Matched window** reruns all included strategies and benchmarks on common
execution dates across eligible assets within each timeframe. Strategies start
flat and a state already active before the boundary is not a fresh entry.
An asset can have a valid full-history result while being skipped in the matched
view. Dates, warmup and execution candle counts are included in the output.

When a daily split is set, each period saves both views. Pick the table using
**Batch result view**, or select a saved folder under **Reports**.
