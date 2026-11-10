"""Offline Polygon HTTP pacing tests using the installed SDK and a fake clock."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from providers import provider_polygon as polygon


def response(status=200, headers=None, **body):
    return SimpleNamespace(status=status, headers=headers or {}, data=json.dumps(body).encode())


class PolygonPacingTests(unittest.TestCase):
    def setUp(self):
        """Simulate elapsed time without network calls or real sleeps."""
        self.now = 100.0
        self.sent = []
        self.pacer = polygon._RequestPacer()
        for patcher in (
            patch.object(polygon.time, "monotonic", side_effect=lambda: self.now),
            patch.object(polygon.time, "sleep", side_effect=self.advance),
            patch.object(polygon.config, "REQUEST_DELAY", 15),
            patch.object(polygon, "_PACER", self.pacer),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def advance(self, seconds):
        self.now += seconds

    def send(self, *args, **kwargs):
        self.sent.append((self.now, args, kwargs))
        return next(self.responses)

    def test_sdk_paginates_and_shares_pacing_between_clients(self):
        """Discovery and both candle pages share one budget, even across SDK instances."""
        bar = dict(o=1, h=2, l=1, c=2, v=3, t=1753516800000)
        self.responses = iter([
            response(results=[{"ticker":"X:1INCHUSD"}]),
            response(results=[bar], next_url="https://api.polygon.io/v2/aggs/next?cursor=abc"),
            response(results=[dict(bar, t=bar["t"]+3600000)]),
        ])
        with patch("urllib3.PoolManager", return_value=Mock(request=self.send)):
            self.assertEqual(polygon.PolygonProvider("dummy").get_available_symbols(), {"1INCH":"X:1INCHUSD"})
            candles = polygon.PolygonProvider("dummy").fetch_ohlcv("1INCH","X:1INCHUSD","2025-07-26","2025-07-27","1h")
        self.assertEqual(len(candles), 2)
        self.assertEqual([x[0] for x in self.sent], [100,115,130])
        self.assertIn("cursor=abc", self.sent[-1][1][1])
        self.assertTrue(all(x[2]["retries"] is False for x in self.sent))

    def test_rate_limit_retries_same_request_after_server_cooldown(self):
        """A rate-limited page is retried in place, respecting a longer Retry-After."""
        self.responses = iter([response(429, {"Retry-After":"90"}), response()])
        result = self.pacer.request(self.send, "GET", "/page?cursor=abc")
        self.assertEqual(result.status, 200)
        self.assertEqual([x[0] for x in self.sent], [100,190])
        self.assertEqual(self.sent[0][1], self.sent[1][1])

    def test_exhausted_retries_preserve_cooldown_for_next_coin(self):
        """Repeated 429s fail explicitly and the next coin still waits out the cooldown."""
        self.responses = iter([response(429)]*3 + [response()])
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            self.pacer.request(self.send, "GET", "/first")
        self.pacer.request(self.send, "GET", "/next")
        self.assertEqual([x[0] for x in self.sent], [100,160,220,280])
