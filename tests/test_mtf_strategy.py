"""MtfTrendPullbackStrategy: no look-ahead in the higher-timeframe bars, trend gating, report wiring."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

from config.settings import BotConfig
from core.backtester import Backtester
from core.indicators import Indicators
from core.strategies import MtfTrendPullbackStrategy, Signal, completed_bars, get_strategy

SPEC = importlib.util.spec_from_file_location(
    "gen_report_mtf", os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_baseline_report.py"))
gr = importlib.util.module_from_spec(SPEC)
sys.modules["gen_report_mtf"] = gr
SPEC.loader.exec_module(gr)


def ramp(n=1000, start="2026-01-01 00:00"):
    """close[k] = k so every bar's value identifies its position."""
    idx = pd.date_range(start, periods=n, freq="5min")
    c = np.arange(n, dtype=float) + 100
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=idx)


def test_completed_bars_drops_partial_bucket():
    df = ramp()
    ends_at_1030 = df.loc[:"2026-01-02 10:30"]               # last 5m bar opens 10:30 -> 10:00 bucket incomplete
    h1 = completed_bars(ends_at_1030, "1h", 60)
    assert h1.index[-1] == pd.Timestamp("2026-01-02 09:00")
    h30 = completed_bars(ends_at_1030, "30min", 30)
    assert h30.index[-1] == pd.Timestamp("2026-01-02 10:00")  # the 10:30 bucket has only its first bar -> dropped


def test_completed_bars_keeps_bucket_when_last_bar_closes_it():
    df = ramp()
    ends_at_1055 = df.loc[:"2026-01-02 10:55"]               # 10:55 bar is the last of the 10:00 hour
    h1 = completed_bars(ends_at_1055, "1h", 60)
    assert h1.index[-1] == pd.Timestamp("2026-01-02 10:00")
    # and its close is the close of the 10:55 bar, never a later one
    assert h1["close"].iloc[-1] == ends_at_1055["close"].iloc[-1]


def test_htf_bars_ignore_future_data():
    """Changing bars AFTER the evaluation time must not change the higher-timeframe view."""
    df = ramp()
    cut = df.loc[:"2026-01-02 10:35"]
    tampered = df.copy()
    tampered.loc["2026-01-02 10:40":, ["open", "high", "low", "close"]] *= 5
    a = completed_bars(cut, "1h", 60)
    b = completed_bars(tampered.loc[:"2026-01-02 10:35"], "1h", 60)
    pd.testing.assert_frame_equal(a, b)


def trending(n, up=True, seed=0):
    rng = np.random.default_rng(seed)
    drift = 0.0004 if up else -0.0004
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    r2 = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * (1 + np.abs(r2.normal(0, 0.0007, n))),
        "low": np.minimum(open_, close) * (1 - np.abs(r2.normal(0, 0.0007, n))),
        "volume": r2.uniform(100, 1000, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="5min"))


def signals(df):
    cfg = BotConfig()
    strat = get_strategy_by_name(cfg)
    ind = Indicators.calculate_all(df, cfg)
    lb = strat.lookback_bars
    out = {"long": 0, "short": 0}
    for i in range(lb, len(ind)):
        s = strat.analyze(ind.iloc[i - lb + 1:i + 1])
        if s.signal == Signal.LONG:
            out["long"] += 1
            assert s.stop_loss < s.entry_price < s.take_profit
        elif s.signal == Signal.SHORT:
            out["short"] += 1
            assert s.take_profit < s.entry_price < s.stop_loss
    return out


def get_strategy_by_name(cfg):
    cfg.strategy.active_strategy = "mtf_trend_pullback"
    s = get_strategy(cfg)
    assert isinstance(s, MtfTrendPullbackStrategy)
    return s


def test_trend_gates_direction():
    up = signals(trending(6000, up=True))
    down = signals(trending(6000, up=False, seed=4))
    assert up["long"] > 0 and up["long"] > 3 * up["short"]
    assert down["short"] > 0 and down["short"] > 3 * down["long"]


def test_holds_when_history_too_short():
    cfg = BotConfig()
    strat = get_strategy_by_name(cfg)
    ind = Indicators.calculate_all(trending(200), cfg)
    assert strat.analyze(ind).signal == Signal.HOLD


def test_backtester_hands_strategy_its_requested_lookback():
    seen = []

    class Spy(MtfTrendPullbackStrategy):
        def analyze(self, df):
            seen.append(len(df))
            return super().analyze(df)

    bt = Backtester(BotConfig())
    bt.strategy = Spy(bt.config)
    bt.run(trending(1500), "X/USDT")
    assert max(seen) == Spy.lookback_bars


def test_other_strategies_keep_300_bar_window():
    seen = []
    cfg = BotConfig()
    cfg.strategy.active_strategy = "grid_scalper"
    bt = Backtester(cfg)
    orig = bt.strategy.analyze
    bt.strategy.analyze = lambda w: (seen.append(len(w)), orig(w))[1]
    bt.run(trending(1000), "X/USDT")
    assert max(seen) == 300


def test_report_accepts_hold_label_end_to_end(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    trending(26000, seed=3).rename_axis("timestamp").to_csv(data / "AAA_USDT_5m.csv")
    trending(26000, seed=5).rename_axis("timestamp").to_csv(data / "BBB_USDT_5m.csv")
    monkeypatch.setattr(sys, "argv", [
        "gen", "--data", str(data), "--out", str(tmp_path / "r"),
        "--strategies", "mtf_trend_pullback@60", "mtf_trend_pullback@240",
        "--scenarios", "all_taker", "--jobs", "1", "--bootstrap", "200", "--date", "2026-01-01"])
    gr.main()
    text = (tmp_path / "r" / "baseline_2026-01-01.md").read_text(encoding="utf-8")
    assert "mtf_trend_pullback@60" in text and "mtf_trend_pullback@240" in text
    assert "**PASS**" in text
