"""Repeatable synthetic prices; tests never depend on downloaded market data."""
import numpy as np
import pandas as pd


def bars(n=300):
    """Create daily rising prices on fixed historical dates, with valid OHLCV."""
    close = np.arange(n, dtype=float) + 10
    return pd.DataFrame({"open": close, "high": close+2, "low": close-2,
                         "close": close, "volume": 100.0},
                        index=pd.date_range("2020-01-01", periods=n, name="date"))
