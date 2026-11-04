"""Public perpetual candles; reuses the earlier scanner's JSON POST pattern."""
import json
from urllib.request import Request, urlopen
import pandas as pd
from .base import DataProvider
from timeframes import get_timeframe


class HyperliquidProvider(DataProvider):
    ENDPOINT = "https://api.hyperliquid.xyz/info"
    MAX_CANDLES = 5000

    def __init__(self, dexes=("",), timeout=20):
        self.dexes = tuple(dexes)
        self.timeout = timeout

    @property
    def name(self):
        return "hyperliquid"

    def _post(self, payload):
        request = Request(self.ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json", "User-Agent": "super-indicator/1.0"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_available_symbols(self):
        """Native perpetuals; optional HIP-3 dex names retain their prefix."""
        symbols = {}
        for dex in self.dexes:
            payload = {"type": "meta"}
            if dex:
                payload["dex"] = dex
            meta = self._post(payload)
            if not isinstance(meta, dict) or not isinstance(meta.get("universe"), list):
                raise ValueError("Invalid Hyperliquid metadata response")
            for market in meta["universe"]:
                name = market.get("name")
                if name and not market.get("isDelisted", False):
                    coin = f"{dex}:{name}" if dex and ":" not in name else name
                    symbols[coin] = coin
        return symbols

    def _now(self):
        return pd.Timestamp.now(tz="UTC")

    def fetch_ohlcv(self, symbol, ticker, start_date, end_date, timeframe="1d"):
        """Only the latest 5,000 candles are available; no unlimited backfill."""
        interval = get_timeframe(timeframe)
        now = self._now()
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        end = min(end, now)
        earliest = now.floor(interval.duration) - (self.MAX_CANDLES-1)*interval.duration
        start = max(start, earliest)
        if start > end:
            return None
        rows = self._post({"type": "candleSnapshot", "req": {
            "coin": ticker, "interval": timeframe, "startTime": int(start.timestamp()*1000),
            "endTime": int(end.timestamp()*1000),
        }})
        if not isinstance(rows, list):
            raise ValueError("Invalid Hyperliquid candle response")
        if not rows:
            return None
        frame = pd.DataFrame([{
            "date": pd.to_datetime(int(row["t"]), unit="ms", utc=True).tz_localize(None),
            "open": float(row["o"]), "high": float(row["h"]), "low": float(row["l"]),
            "close": float(row["c"]), "volume": float(row["v"]),
        } for row in rows]).set_index("date").sort_index(kind="stable")
        frame = frame[~frame.index.duplicated(keep="last")]
        return frame.loc[(frame.index >= start.tz_localize(None)) & (frame.index <= end.tz_localize(None))]
