from types import SimpleNamespace

import pandas as pd
import pytest

from evolve.evaluate import (
    CandidateError, _metrics, check_truncated_overlap, constraint_failures, load_candidate,
    validate_candidate_source,
)


def _candidate(body):
    return """class Trial(BaseStrategy):
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


@pytest.mark.parametrize("body", [
    "return TradeSetup(Signal.HOLD)\n        x = ().__class__.__bases__[0].__subclasses__()",
    "return pd.read_csv('data/x.csv')",
    "return np.load('data/x.npy')",
    "return getattr(pd, 'read_pickle')",
])
def test_candidate_refuses_introspection_and_io(body):
    with pytest.raises(CandidateError):
        validate_candidate_source("import pandas as pd\nimport numpy as np\n" + _candidate(body))


@pytest.mark.parametrize("name", [
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "type",
    "dir", "breakpoint", "input",
])
def test_candidate_refuses_blocked_builtin_calls(name):
    with pytest.raises(CandidateError):
        validate_candidate_source(_candidate(f"return {name}(df, 'x')"))


@pytest.mark.parametrize("literal", ["getattr", "read_csv", "load", "__class__", "__internal"])
def test_candidate_refuses_sensitive_string_constants(literal):
    with pytest.raises(CandidateError):
        validate_candidate_source(_candidate(f"return TradeSetup(Signal.HOLD)\n        value = '{literal}'"))


def test_load_candidate_uses_restricted_builtins(tmp_path, monkeypatch):
    import evolve.evaluate as evaluate

    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    (candidate_dir / "safe.py").write_text(_candidate("return TradeSetup(Signal.HOLD)"), encoding="utf-8")
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(candidate_dir))
    strategy, _ = load_candidate("safe", object())
    assert strategy.analyze(pd.DataFrame()).signal.value == "HOLD"


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


def _geometry_trade(stop_pct, target_pct):
    return SimpleNamespace(pnl=1.0, notional=100.0,
                           entry_time=pd.Timestamp("2025-01-01"), symbol="BTC/USDT",
                           exit_reason="TP", entry_price=100.0,
                           sl=100.0 * (1 - stop_pct / 100),
                           tp=100.0 * (1 + target_pct / 100))


def test_candidate_with_03_percent_stops_is_rejected():
    metrics = _metrics([_geometry_trade(0.3, 5.0)], reps=10)
    failures = constraint_failures({"base": metrics}, stop_pct=0.02)
    assert any("median stop distance 0.300%" in reason for reason in failures)


def test_candidate_with_2_percent_stops_and_5_percent_targets_passes_geometry():
    metrics = _metrics([_geometry_trade(2.0, 5.0)], reps=10)
    assert constraint_failures({"base": metrics}, stop_pct=0.02) == []
