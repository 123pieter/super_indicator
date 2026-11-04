from polygon import RESTClient
import pandas as pd
from typing import Dict, Optional
from .base import DataProvider
from timeframes import get_timeframe


class PolygonProvider(DataProvider):

    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)

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
