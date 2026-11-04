"""Project-local paths and defaults. Credentials are loaded only when fetching."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
DB_PATH = BASE_DIR / "crypto_data.db"
SECRETS_FILE = BASE_DIR / "secrets.txt"
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

ACTIVE_PROVIDER = "polygon"
SAVE_TO = "both"  # csv, sqlite, or both
LOAD_FROM = "csv"  # csv or sqlite
DEFAULT_TIMEFRAME = "1d"
START_DATE = "2024-06-01"
MIN_BARS = 250
BUY_THRESHOLD = 0.7
FETCH_LIMIT = 0  # None or 0 fetches all available pairs.
REQUEST_DELAY = 12


def load_api_key(filepath=None):
    """Use the environment variable or the project's ignored secrets.txt file."""
    if filepath is None and os.environ.get("POLYGON_API_KEY", "").strip():
        return os.environ["POLYGON_API_KEY"].strip()
    path = Path(filepath) if filepath is not None else SECRETS_FILE
    if not path.is_file():
        raise FileNotFoundError("Set POLYGON_API_KEY or put the API key in the project's secrets.txt")
    key = path.read_text(encoding="utf-8-sig").strip()
    if not key or "POLYGON_API_KEY" in key or "key,value" in key:
        raise ValueError("secrets.txt must contain only the API key")
    return key

# Persisted settings override these defaults separately for each provider.
PROVIDER_FETCH_DEFAULTS = {
    "polygon": {"max_missing_bars": None, "exclude_no_data": False},
    "hyperliquid": {"max_missing_bars": None, "exclude_no_data": False},
}

# Native perpetuals by default; add e.g. "xyz" or "km" for builder-deployed markets.
HYPERLIQUID_DEXES = ("",)
HYPERLIQUID_REQUEST_DELAY = 6


def storage_paths(provider, data_folder=None, database=None):
    """Provider-specific CSV folders and one shared provider-keyed candle database."""
    from providers import validate_provider
    provider = validate_provider(provider)
    folder = Path(DATA_FOLDER if data_folder is None else data_folder)
    db = Path(DB_PATH if database is None else database)
    return folder/provider, db

# Research assumptions, not exchange fee quotes. Edit independently per provider.
BACKTEST_COSTS = {
    "polygon": {"fee": 0.001, "slippage": 0.0005},
    "hyperliquid": {"fee": 0.001, "slippage": 0.0005},
}
STOP_ATR = 3.0
TP_ATR = 6.0  # None disables the take-profit.
MAX_HOLD = 20  # Completed candles, including the entry candle.
EXIT_SCORE_DROP = 0.2
MIN_RESEARCH_TRADES = 20  # Descriptive sample-size flag; never a universe selection rule.
