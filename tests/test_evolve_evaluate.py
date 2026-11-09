from types import SimpleNamespace

import pandas as pd
import pytest

from evolve.evaluate import (
    CandidateError, _metrics, check_truncated_overlap, validate_candidate_source,
)


def _candidate(body):
    return """class Trial(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
    def analyze(self, df):
""" + "\n".join("        " + line for line in body.splitlines()) + "\n"


def test_candidate_that_peeks_with_negative_shift_is_rejected():
    source = _candidate("return TradeSetup(Signal.LONG if df['close'].shift(-1).iloc[-1] > 0 else Signal.HOLD)")

    with pytest.raises(CandidateError, match="look-ahead"):
        validate_candidate_source(source)


def test_candidate_importing_os_is_rejected():
    source = "import os\n" + _candidate("return TradeSetup(Signal.HOLD)")

    with pytest.raises(CandidateError, match="import is not allowed"):
        validate_candidate_source(source)


def test_truncated_overlap_check_accepts_causal_completed_bar_signals():
    class Setup:
        signal = SimpleNamespace(value="LONG")
        entry_price = 10.0
        stop_loss = 9.0
        take_profit = 12.0
        confidence = 0.5

    class Strategy:
        def analyze(self, df):
            assert len(df) > 0
            return Setup()

    frame = pd.DataFrame({"close": range(40)}, index=pd.date_range("2025-01-01", periods=40, freq="5min"))

    assert check_truncated_overlap(Strategy, frame, lookback=8)


def test_metrics_reports_bootstrap_month_symbol_and_exit_counts():
    trades = [
        SimpleNamespace(pnl=1.0, notional=100.0, entry_time=pd.Timestamp("2025-01-01"),
                        symbol="BTC/USDT", exit_reason="TP"),
        SimpleNamespace(pnl=-0.5, notional=100.0, entry_time=pd.Timestamp("2025-02-01"),
                        symbol="ETH/USDT", exit_reason="SL"),
    ]

    result = _metrics(trades, reps=100, seed=2)

    assert result["trades"] == 2
    assert result["net_bps_per_trade"] == pytest.approx(25.0)
    assert result["months"] == 2
    assert result["profitable_months"] == 1
    assert result["symbols"] == 2
    assert result["exit_reasons"] == {"SL": 1, "TP": 1}
