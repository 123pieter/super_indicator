from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional


class DataProvider(ABC):
    """Abstract base class for all data providers."""

    @abstractmethod
    def get_available_symbols(self) -> Dict[str, str]:
        """
        Returns a dictionary of {symbol: provider_ticker}
        Example: {"BTC": "X:BTCUSD", "ETH": "X:ETHUSD"}
        """
        pass

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, ticker: str, start_date: str, end_date: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for the requested timeframe (default 1d).
        Timestamps represent candle starts in UTC; lengths and returns use bars.
        Must return a DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex named 'date'.
        Return None for a successful empty response. Raise request errors so callers
        can distinguish API failures from unavailable market data.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'polygon', 'binance')"""
        pass