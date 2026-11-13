from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError
from zipfile import ZipFile

import pandas as pd

from providers.provider_binance import BinanceProvider


def open_ms(date):
    return pd.Timestamp(date, tz="UTC").value // 1_000_000


def kline(date):
    return [open_ms(date), "10", "12", "9", "11", "2.5"]


class BinanceProviderTests(TestCase):
    def test_discovery_requires_trading_usdt_perpetual(self):
        provider = BinanceProvider()
        valid = dict(symbol="BTCUSDT", quoteAsset="USDT", marginAsset="USDT",
                     contractType="PERPETUAL", status="TRADING", onboardDate=open_ms("2024-01-02"))
        markets = [
            valid,
            dict(valid, symbol="OLDUSDT", status="SETTLING"),
            dict(valid, symbol="BTCUSDC", marginAsset="USDC", quoteAsset="USDC"),
            dict(valid, symbol="ETHUSDT", contractType="CURRENT_QUARTER"),
            dict(valid, symbol="1000PEPEUSDT"),
        ]
        with patch.object(provider, "_json_get", return_value={"symbols": markets}) as request:
            self.assertEqual(provider.get_available_symbols(), {"BTC": "BTCUSDT", "1000PEPE": "1000PEPEUSDT"})
        request.assert_called_once_with("/fapi/v1/exchangeInfo")

    def test_rest_paginates_nonoverlapping_ranges_and_drops_unfinished_candle(self):
        provider = BinanceProvider()
        provider.PAGE_SIZE = 2
        with patch.object(provider, "_now", return_value=pd.Timestamp("2024-01-04T12:00Z")), patch.object(
            provider, "_json_get", side_effect=[
                [kline("2024-01-01"), kline("2024-01-02")],
                [kline("2024-01-03"), kline("2024-01-04")],
            ],
        ) as request:
            candles = provider.fetch_ohlcv("BTC", "BTCUSDT", "2024-01-01", "2024-01-04")
        windows = [(call.kwargs["startTime"], call.kwargs["endTime"]) for call in request.call_args_list]
        self.assertEqual(windows, [(open_ms("2024-01-01"), open_ms("2024-01-03") - 1),
                                   (open_ms("2024-01-03"), open_ms("2024-01-03"))])
        self.assertEqual(candles.index.tolist(), pd.date_range("2024-01-01", periods=3).tolist())
        self.assertEqual(candles.index.name, "date")
        self.assertEqual(candles.iloc[0].tolist(), [10., 12., 9., 11., 2.5])

    def test_archive_fallback_stays_in_usdm_and_uses_monthly_and_daily(self):
        provider = BinanceProvider(archive_symbols=("BTCUSDT", "BTCUSDC"))
        blocked = HTTPError("https://fapi.binance.com", 404, "Unavailable", {}, None)
        with patch.object(provider, "_now", return_value=pd.Timestamp("2024-02-03T12:00Z")), patch.object(
            provider, "_json_get", side_effect=blocked,
        ) as rest, patch.object(
            provider, "_archive_file", side_effect=[[kline("2024-01-31")], [], [kline("2024-02-02")]],
        ) as archive:
            self.assertEqual(provider.get_available_symbols(), {"BTC": "BTCUSDT"})
            candles = provider.fetch_ohlcv("BTC", "BTCUSDT", "2024-01-31", "2024-02-03")
        rest.assert_called_once_with("/fapi/v1/exchangeInfo")
        self.assertEqual([c.args for c in archive.call_args_list], [
            ("BTCUSDT", "1d", "monthly", "2024-01"),
            ("BTCUSDT", "1d", "daily", "2024-02-01"),
            ("BTCUSDT", "1d", "daily", "2024-02-02"),
        ])
        self.assertEqual(candles.index.tolist(), list(pd.to_datetime(["2024-01-31", "2024-02-02"])))

    def test_archive_parses_official_csv_without_extracting_files(self):
        provider = BinanceProvider()
        body = BytesIO()
        with ZipFile(body, "w") as archive:
            archive.writestr("BTCUSDT-1d-2024-01.csv",
                             f"open_time,open,high,low,close,volume\n{open_ms('2024-01-01')},10,12,9,11,2.5\n")
        with patch.object(provider, "_download", return_value=body.getvalue()) as download:
            rows = provider._archive_file("BTCUSDT", "1d", "monthly", "2024-01")
        self.assertEqual(rows, [[str(open_ms("2024-01-01")), "10", "12", "9", "11", "2.5"]])
        download.assert_called_once_with(
            "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
        )

    def test_fallback_does_not_hide_missing_configuration_or_rate_limits(self):
        provider = BinanceProvider()
        forbidden = HTTPError("https://fapi.binance.com", 451, "Unavailable", {}, None)
        with patch.object(provider, "_json_get", side_effect=forbidden), self.assertRaises(HTTPError):
            provider.get_available_symbols()
        provider = BinanceProvider(archive_symbols=("BTCUSDT",))
        limited = HTTPError("https://fapi.binance.com", 429, "Limited", {}, None)
        with patch.object(provider, "_json_get", side_effect=limited), patch.object(
            provider, "_archive_rows",
        ) as archive, self.assertRaises(HTTPError):
            provider.fetch_ohlcv("BTC", "BTCUSDT", "2024-01-01", "2024-01-02")
        archive.assert_not_called()
