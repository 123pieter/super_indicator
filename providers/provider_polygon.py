from email.utils import parsedate_to_datetime
from threading import Lock
import time

import config
from polygon import RESTClient
import pandas as pd
from typing import Dict, Optional
from .base import DataProvider
from timeframes import get_timeframe


def _cooldown(header):
    """Use Retry-After (seconds or an HTTP date), with a one-minute minimum."""
    try:
        value = int(header)
    except (TypeError, ValueError):
        try:
            value = parsedate_to_datetime(header).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError, AttributeError):
            value = 0
    return max(60, value)


class _RequestPacer:
    """Share one request budget across Polygon clients in this app process."""

    def __init__(self):
        self.lock = Lock()
        self.next_request = 0.0

    def request(self, send, *args, **kwargs):
        # Do not let the SDK transport send unpaced retries or redirects.
        kwargs.update(retries=False, redirect=False)
        for _ in range(3):
            with self.lock:
                delay = self.next_request - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self.next_request = time.monotonic() + config.REQUEST_DELAY
                response = send(*args, **kwargs)
                if response.status != 429:
                    return response
                self.next_request = max(
                    self.next_request, time.monotonic() + _cooldown(response.headers.get("Retry-After"))
                )
        raise RuntimeError("Polygon HTTP 429: rate limited after three attempts; try again later")


_PACER = _RequestPacer()


class _PacedTransport:
    """Keep the SDK's parsing and pagination while pacing each HTTP request."""

    def __init__(self, pool):
        self.pool = pool

    def request(self, *args, **kwargs):
        return _PACER.request(self.pool.request, *args, **kwargs)


class PolygonProvider(DataProvider):

    def __init__(self, api_key: str):
        self.client = RESTClient(api_key, retries=0)
        self.client.client = _PacedTransport(self.client.client)

    @property
    def name(self) -> str:
        return "polygon"

    def get_available_symbols(self) -> Dict[str, str]:
        tickers = {}
        for t in self.client.list_tickers(market="crypto", active=True, limit=1000):
            if t.ticker.startswith("X:") and t.ticker.endswith("USD"):
                symbol = t.ticker[2:-3]
                tickers[symbol] = t.ticker
        return tickers

    def fetch_ohlcv(self, symbol: str, ticker: str, start_date: str, end_date: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        interval = get_timeframe(timeframe)
        aggs = list(self.client.list_aggs(
            ticker=ticker,
            multiplier=interval.multiplier,
            timespan=interval.timespan,
            from_=start_date,
            to=end_date,
            limit=50000,
            adjusted=True
        ))
        if not aggs:
            return None

        df = pd.DataFrame([{
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume,
            "date": pd.to_datetime(a.timestamp, unit="ms")
        } for a in aggs])

        return df.set_index("date").sort_index()
