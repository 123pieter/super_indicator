"""Public Binance USDⓈ-M USDT perpetual trade candles."""
import csv
import io
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from timeframes import get_timeframe
from .base import DataProvider


class BinanceProvider(DataProvider):
    REST = "https://fapi.binance.com"
    ARCHIVE = "https://data.binance.vision/data/futures/um"
    PAGE_SIZE = 499
    MAX_PAGES = 10_000
    MAX_ARCHIVE_REQUESTS = 4_096

    def __init__(self, archive_symbols=(), timeout=20):
        # Archive tickers are an explicit scope; archive files do not prove active listing status.
        self.archive_symbols = {ticker[:-4]: ticker for ticker in archive_symbols if self._is_usdt(ticker)}
        self.timeout = timeout
        self._archive_only = False
        self._last_rest_request = float("-inf")
        self._onboard = {}

    @property
    def name(self):
        return "binance"

    @staticmethod
    def _is_usdt(ticker):
        # Exclude dated futures (e.g. BTCUSDT_260925); some token names are Unicode.
        return isinstance(ticker, str) and ticker.endswith("USDT") and ticker[:-4].isalnum()

    def _now(self):
        return pd.Timestamp.now(tz="UTC")

    def _download(self, url):
        request = Request(url, headers={"User-Agent": "super-indicator/1.0"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _json_get(self, path, **params):
        wait = .1 - (time.monotonic() - self._last_rest_request)
        if wait > 0:
            time.sleep(wait)
        self._last_rest_request = time.monotonic()
        query = "?" + urlencode(params) if params else ""
        return json.loads(self._download(self.REST + path + query))

    @staticmethod
    def _can_fallback(error):
        if isinstance(error, HTTPError):
            # Rate limits and invalid requests must stay visible to the caller.
            return error.code in (403, 404, 451, 500, 502, 503, 504)
        return isinstance(error, (URLError, TimeoutError))

    def get_available_symbols(self):
        if self._archive_only:
            return dict(self.archive_symbols)
        try:
            info = self._json_get("/fapi/v1/exchangeInfo")
        except (URLError, TimeoutError) as error:
            if not self._can_fallback(error) or not self.archive_symbols:
                raise
            self._archive_only = True
            return dict(self.archive_symbols)
        if not isinstance(info, dict) or not isinstance(info.get("symbols"), list):
            raise ValueError("Invalid Binance USD-M exchangeInfo response")
        symbols = {}
        for market in info["symbols"]:
            ticker = market.get("symbol")
            if (market.get("contractType") == "PERPETUAL" and market.get("status") == "TRADING"
                    and market.get("quoteAsset") == market.get("marginAsset") == "USDT"
                    and self._is_usdt(ticker)):
                symbols[ticker[:-4]] = ticker
                if market.get("onboardDate"):
                    self._onboard[ticker] = pd.Timestamp(market["onboardDate"], unit="ms", tz="UTC")
        return symbols

    def _rest_rows(self, ticker, timeframe, start, end, duration):
        """Request non-overlapping windows, even if a window precedes listing."""
        step = self.PAGE_SIZE * duration
        if (end - start) // step >= self.MAX_PAGES:
            raise ValueError("Binance history exceeds the per-request page budget; use shorter date ranges")
        rows = []
        cursor = start
        while cursor <= end:
            next_cursor = cursor + step
            page = self._json_get(
                "/fapi/v1/klines", symbol=ticker, interval=timeframe, limit=self.PAGE_SIZE,
                startTime=cursor.value // 1_000_000,
                endTime=min(end.value // 1_000_000, next_cursor.value // 1_000_000 - 1),
            )
            if not isinstance(page, list):
                raise ValueError("Invalid Binance USD-M kline response")
            rows.extend(page)
            cursor = next_cursor
        return rows

    def _archive_file(self, ticker, timeframe, frequency, date):
        symbol = quote(ticker, safe="")
        url = f"{self.ARCHIVE}/{frequency}/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{date}.zip"
        try:
            data = self._download(url)
        except HTTPError as error:
            if error.code == 404:
                return []
            raise
        with ZipFile(io.BytesIO(data)) as archive:
            files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(files) != 1:
                raise ValueError(f"Invalid Binance kline archive at {url}")
            with archive.open(files[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                rows = [row for row in csv.reader(stream) if row]
        if rows and rows[0][0].strip().lower() in ("open_time", "open time", "opentime"):
            rows.pop(0)
        if not rows:
            raise ValueError(f"Empty Binance kline archive at {url}")
        return rows

    def _archive_rows(self, ticker, timeframe, start, end, now):
        # The archive publishes files after the relevant UTC period closes.
        last = min(end.normalize(), now.normalize() - pd.Timedelta(days=1))
        first = start.normalize()
        if first > last:
            return []
        rows = []
        requests = 0

        def download(frequency, date):
            nonlocal requests
            if requests >= self.MAX_ARCHIVE_REQUESTS:
                raise ValueError("Binance archive request budget exceeded; use shorter date ranges")
            requests += 1
            return self._archive_file(ticker, timeframe, frequency, date)

        for month in pd.date_range(first.replace(day=1), last, freq="MS"):
            next_month = month + pd.offsets.MonthBegin()
            if next_month <= now.normalize():
                monthly = download("monthly", month.strftime("%Y-%m"))
                if monthly:
                    rows.extend(monthly)
                    continue
            for day in pd.date_range(max(first, month), min(last, next_month - pd.Timedelta(days=1)), freq="D"):
                rows.extend(download("daily", day.strftime("%Y-%m-%d")))
        return rows

    def fetch_ohlcv(self, symbol, ticker, start_date, end_date, timeframe="1d"):
        if not self._is_usdt(ticker) or symbol != ticker[:-4]:
            raise ValueError("Binance requires a base symbol and its USD-M USDT perpetual ticker")
        duration = get_timeframe(timeframe).duration
        now = self._now()
        start = pd.to_datetime(start_date, utc=True).normalize()
        # The first listed candle may open before the exact onboarding timestamp.
        if ticker in self._onboard:
            start = max(start, self._onboard[ticker].floor(duration))
        end = min(pd.to_datetime(end_date, utc=True).normalize() + pd.Timedelta(days=1),
                  now.floor(duration)) - duration
        if start > end:
            return None
        if self._archive_only:
            rows = self._archive_rows(ticker, timeframe, start, end, now)
        else:
            try:
                rows = self._rest_rows(ticker, timeframe, start, end, duration)
            except (URLError, TimeoutError) as error:
                if not self._can_fallback(error):
                    raise
                rows = self._archive_rows(ticker, timeframe, start, end, now)
        if not rows:
            return None
        if any(not isinstance(row, (list, tuple)) or len(row) < 6 for row in rows):
            raise ValueError("Invalid Binance kline row")
        frame = pd.DataFrame([row[:6] for row in rows], columns=["date", "open", "high", "low", "close", "volume"])
        frame["date"] = pd.to_datetime(frame["date"].astype("int64"), unit="ms", utc=True)
        frame = frame.set_index("date").astype(float).sort_index()
        frame = frame.loc[start:end]
        if not frame.index.is_unique:
            # Exact duplicate archive rows are harmless; conflicting prices are not.
            frame = frame.reset_index().drop_duplicates().set_index("date")
            if not frame.index.is_unique:
                raise ValueError("Conflicting Binance klines for the same open timestamp")
        if frame.empty:
            return None
        if not frame.index.equals(frame.index.floor(duration)):
            raise ValueError("Binance returned klines outside the requested UTC timeframe grid")
        # Match the project's provider convention: UTC times in a timezone-naive index.
        return frame.tz_localize(None)
