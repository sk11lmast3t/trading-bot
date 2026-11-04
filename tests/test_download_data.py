"""scripts/download_data.py: pagination, resume, forming-candle drop, coverage, ranking."""
import importlib.util
import os
import sys
import types

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_data", os.path.join(os.path.dirname(__file__), "..", "scripts", "download_data.py"))
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)

STEP = 300_000


class FakeExchange:
    """Serves 5m candles for [t0, t0 + n*STEP). Honors since/limit like a real exchange."""
    def __init__(self, t0, n, now_extra=0, fail_first=0, gaps=()):
        self.t0, self.n, self.calls, self.fail_first = t0, n, 0, fail_first
        self._now = t0 + n * STEP + now_extra
        self.gaps = set(gaps)

    def milliseconds(self):
        return self._now

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise sys.modules["ccxt"].NetworkError("boom")
        first = max(since, self.t0)
        first += (-(first - self.t0)) % STEP
        out = []
        ts = first
        while len(out) < limit and ts < self.t0 + self.n * STEP:
            if ts not in self.gaps:
                out.append([ts, 1.0, 2.0, 0.5, 1.5, 10.0])
            ts += STEP
        return out

    def fetch_tickers(self):
        return {
            "BTC/USDT": {"quoteVolume": 900}, "ETH/USDT": {"quoteVolume": 500},
            "USDC/USDT": {"quoteVolume": 9999}, "BTCUP/USDT": {"quoteVolume": 800},
            "SOL/USDT:USDT": {"quoteVolume": 700}, "XRP/USDT": {"quoteVolume": 300},
            "PEPE/USDT": {"quoteVolume": 600}, "DOGE/BTC": {"quoteVolume": 400},
        }


@pytest.fixture(autouse=True)
def ccxt_errors(monkeypatch):
    mod = sys.modules.get("ccxt") or types.ModuleType("ccxt")
    for name in ("NetworkError", "RateLimitExceeded", "ExchangeNotAvailable"):
        if not hasattr(mod, name):
            monkeypatch.setattr(mod, name, type(name, (Exception,), {}), raising=False)
    monkeypatch.setitem(sys.modules, "ccxt", mod)
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)


def test_paginates_and_drops_forming_candle(tmp_path):
    t0 = pd.Timestamp("2026-01-01").value // 10**6
    ex = FakeExchange(t0, n=2500)
    df = dd.download_symbol(ex, "BTC/USDT", "5m", months=1, out_dir=str(tmp_path))
    assert ex.calls >= 3                                   # 2500 candles / 1000 per call
    assert len(df) == 2499                                  # last (forming) candle dropped
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert os.path.exists(dd.csv_path(str(tmp_path), "BTC/USDT", "5m"))


def test_resume_appends_without_duplicates(tmp_path):
    t0 = pd.Timestamp("2026-01-01").value // 10**6
    dd.download_symbol(FakeExchange(t0, n=1500), "BTC/USDT", "5m", months=1, out_dir=str(tmp_path))
    ex2 = FakeExchange(t0, n=3000)
    df = dd.download_symbol(ex2, "BTC/USDT", "5m", months=1, out_dir=str(tmp_path))
    assert len(df) == 2999
    assert not df.index.duplicated().any()
    exp, cov, miss = dd.coverage(df, "5m")
    assert miss == 0 and cov == pytest.approx(1.0)


def test_coverage_reports_gaps(tmp_path):
    t0 = pd.Timestamp("2026-01-01").value // 10**6
    gaps = [t0 + i * STEP for i in range(100, 150)]
    df = dd.download_symbol(FakeExchange(t0, n=2000, gaps=gaps), "ETH/USDT", "5m",
                            months=1, out_dir=str(tmp_path))
    exp, cov, miss = dd.coverage(df, "5m")
    assert miss == 50 and cov < 1.0


def test_retries_transient_errors(tmp_path):
    t0 = pd.Timestamp("2026-01-01").value // 10**6
    ex = FakeExchange(t0, n=500, fail_first=2)
    df = dd.download_symbol(ex, "BTC/USDT", "5m", months=1, out_dir=str(tmp_path))
    assert len(df) == 499


def test_rank_symbols_filters_stables_leveraged_futures_and_exclusions():
    out = dd.rank_symbols(FakeExchange(0, 1), top=10, exclude={"PEPE/USDT"})
    assert out == ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
