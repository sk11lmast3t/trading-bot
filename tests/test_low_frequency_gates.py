"""Paper-only low-frequency gates: R:R after costs, volume ratio, trade/loss limits.

Every gate is off by default; these tests check both the default-off behavior
and the active behavior.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import main
from config.settings import BotConfig
from core.risk_manager import RiskManager
from core.strategies import Signal, TradeSetup
from core.strategy_engine import (
    STRATEGIES, StrategyEngine, StrategyInfo, estimate_reward_cost, volume_confirmation,
)


def cost_cfg(**extra):
    cfg = BotConfig()
    cfg.backtest.maker_fee = 0.001
    cfg.backtest.taker_fee = 0.001
    cfg.backtest.slippage = 0.0003
    cfg.backtest.funding_rate = 0.0
    cfg.execution.cost_safety_buffer_pct = 0.0002
    cfg.execution.minimum_expected_net_reward_pct = 0.0002
    for key, value in extra.items():
        setattr(cfg, key, value)
    return cfg


# Round trip = 2 * (0.001 + 0.0003) = 0.0026. Target 2%, stop 0.6%:
# rr = (0.02 - 0.0026) / (0.006 + 0.0026) = 2.023
def long_setup(stop=99.4, target=102.0):
    return TradeSetup(Signal.LONG, 100.0, stop, target, 0.8, "x")


def test_rr_gate_is_off_by_default():
    estimate = estimate_reward_cost(long_setup(), cost_cfg())
    assert estimate["allowed"] is True
    assert "rr_after_costs" not in estimate


def test_rr_gate_rejects_below_minimum_and_accepts_at_or_above():
    rejected = estimate_reward_cost(long_setup(), cost_cfg(min_rr_after_costs=2.5))
    assert rejected["allowed"] is False
    assert rejected["rr_rejected"] is True
    assert rejected["rr_after_costs"] == pytest.approx((0.02 - 0.0026) / (0.006 + 0.0026))
    assert "risk/reward" in rejected["reason"]
    accepted = estimate_reward_cost(long_setup(), cost_cfg(min_rr_after_costs=2.0))
    assert accepted["allowed"] is True


def test_rr_gate_works_for_shorts_with_the_same_round_trip_cost():
    short = TradeSetup(Signal.SHORT, 100.0, 100.6, 98.0, 0.8, "x")
    estimate = estimate_reward_cost(short, cost_cfg(min_rr_after_costs=2.5))
    assert estimate["allowed"] is False
    assert estimate["rr_after_costs"] == pytest.approx((0.02 - 0.0026) / (0.006 + 0.0026))


@pytest.mark.parametrize("stop", [0.0, float("nan"), 100.0, 101.0])
def test_rr_gate_fails_closed_on_missing_or_wrong_side_stop(stop):
    estimate = estimate_reward_cost(long_setup(stop=stop), cost_cfg(min_rr_after_costs=1.0))
    assert estimate["allowed"] is False
    assert estimate["rr_rejected"] is True


def test_rr_gate_does_not_hide_the_existing_cost_rejection():
    tiny = TradeSetup(Signal.LONG, 100.0, 99.9, 100.1, 0.8, "x")
    estimate = estimate_reward_cost(tiny, cost_cfg(min_rr_after_costs=1.0))
    assert estimate["allowed"] is False
    assert "rr_rejected" not in estimate


def frame_with_volumes(volumes):
    index = pd.date_range("2026-01-01", periods=len(volumes), freq="5min")
    close = np.linspace(100, 104, len(volumes))
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": volumes}, index=index)


def test_volume_gate_off_by_default_even_with_no_volume_data():
    assert volume_confirmation(None, 0.0) == (True, "")
    assert volume_confirmation(frame_with_volumes([0.0] * 5), 0.0) == (True, "")


def test_volume_gate_compares_last_closed_bar_with_prior_20_bar_average():
    flat = [100.0] * 20
    assert volume_confirmation(frame_with_volumes(flat + [160.0]), 1.5)[0] is True
    ok, reason = volume_confirmation(frame_with_volumes(flat + [140.0]), 1.5)
    assert ok is False and "1.40x" in reason
    # The last bar is excluded from its own average.
    assert volume_confirmation(frame_with_volumes([100.0] * 20 + [150.0]), 1.5)[0] is True


@pytest.mark.parametrize("volumes", [
    [100.0] * 20,                       # only 20 bars: needs 21
    [0.0] * 20 + [50.0],                # zero average
    [100.0] * 19 + [float("nan"), 200.0],  # missing prior volume
    [100.0] * 20 + [float("nan")],      # missing last volume
])
def test_volume_gate_fails_closed_on_short_or_bad_history(volumes):
    ok, reason = volume_confirmation(frame_with_volumes(volumes), 1.5)
    assert ok is False and reason.startswith("volume filter")


def test_engine_applies_volume_gate_only_when_enabled(monkeypatch):
    class Long:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.LONG, 100, 98, 104, 0.8, "long")

    monkeypatch.setitem(STRATEGIES, "fixture_long", StrategyInfo(
        "fixture_long", Long, (), 1, "long", supported_timeframes=("5m",)))
    now = pd.Timestamp("2026-01-01 03:20")

    def run(last_volume, **cfg_extra):
        volumes = [100.0] * 39 + [last_volume]

        class Exchange:
            def fetch_ohlcv(self, symbol, timeframe, limit):
                return frame_with_volumes(volumes)

        cfg = BotConfig()
        for key, value in cfg_extra.items():
            setattr(cfg, key, value)
        engine = StrategyEngine(["fixture_long"], ["5m"], cfg)
        setup, _, _, _ = engine.evaluate(Exchange(), "BTC/USDT", now=now)
        return setup, engine

    assert run(100.0)[0].signal == Signal.LONG                       # gate off
    assert run(100.0, min_volume_ratio=1.5)[0].signal == Signal.HOLD  # 1.0x < 1.5x
    setup, engine = run(100.0, min_volume_ratio=1.5)
    assert "volume filter" in engine.last_rejection_reason["BTC/USDT"]
    assert run(200.0, min_volume_ratio=1.5)[0].signal == Signal.LONG  # 2.0x >= 1.5x


def test_engine_reports_rr_rejection_reason(monkeypatch):
    class Long:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.LONG, 100, 98, 104, 0.8, "long")  # rr ~1.8

    monkeypatch.setitem(STRATEGIES, "fixture_long", StrategyInfo(
        "fixture_long", Long, (), 1, "long", supported_timeframes=("5m",)))

    class Exchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return frame_with_volumes([100.0] * 40)

    cfg = cost_cfg(min_rr_after_costs=2.5)
    engine = StrategyEngine(["fixture_long"], ["5m"], cfg)
    setup, _, _, _ = engine.evaluate(Exchange(), "BTC/USDT", now=pd.Timestamp("2026-01-01 03:20"))
    assert setup.signal == Signal.HOLD
    assert "risk/reward" in engine.last_rejection_reason["BTC/USDT"]


def build(*extra, mode=()):
    return main.build_config_from_args([*mode, "--exchange", "kucoin", *extra], environ={})


def test_cli_defaults_leave_config_unchanged():
    cfg, base = build(), BotConfig()
    assert cfg.min_rr_after_costs == 0.0
    assert cfg.min_volume_ratio == 0.0
    assert cfg.risk.daily_trade_limit == base.risk.daily_trade_limit
    assert cfg.scalping.cooldown_seconds == base.scalping.cooldown_seconds
    assert cfg.risk.max_daily_loss_pct == base.risk.max_daily_loss_pct
    assert cfg.risk.max_drawdown_pct == base.risk.max_drawdown_pct


def test_cli_flags_set_the_documented_fields_in_percentage_points():
    cfg = build("--min-rr-after-costs", "2.5", "--min-volume-ratio", "1.5",
                "--daily-trade-limit", "3", "--cooldown-seconds", "1800",
                "--max-daily-loss-pct", "5", "--max-drawdown-pct", "8")
    assert cfg.min_rr_after_costs == 2.5
    assert cfg.min_volume_ratio == 1.5
    assert cfg.risk.daily_trade_limit == 3
    assert cfg.scalping.cooldown_seconds == 1800
    assert cfg.risk.max_daily_loss_pct == pytest.approx(0.05)
    assert cfg.risk.max_drawdown_pct == pytest.approx(0.08)
    summary = main.effective_config_summary(cfg)
    assert summary["MIN RR AFTER COSTS"] == 2.5
    assert summary["MIN VOLUME RATIO"] == 1.5
    assert summary["DAILY TRADE LIMIT"] == 3
    assert summary["COOLDOWN SECONDS"] == 1800
    assert summary["MAX DAILY LOSS PCT"] == pytest.approx(5.0)
    assert summary["MAX DRAWDOWN PCT"] == pytest.approx(8.0)


@pytest.mark.parametrize("args", [
    ["--min-rr-after-costs", "-1"], ["--min-rr-after-costs", "nan"],
    ["--min-volume-ratio", "-0.5"], ["--min-volume-ratio", "inf"],
    ["--daily-trade-limit", "0"], ["--cooldown-seconds", "-1"],
    ["--max-daily-loss-pct", "0"], ["--max-daily-loss-pct", "101"],
    ["--max-drawdown-pct", "-3"], ["--max-drawdown-pct", "nan"],
])
def test_cli_rejects_invalid_values(args):
    with pytest.raises(SystemExit):
        build(*args)


def test_new_flags_are_rejected_for_live_mode():
    with pytest.raises(SystemExit, match="paper mode"):
        main.build_config_from_args(
            ["live", "--daily-trade-limit", "2"], environ={"CONFIRM_LIVE": "YES"})


T0 = datetime(2026, 1, 1, 12, 0, 0)


def risk_manager(*flags):
    cfg = build(*flags)
    cfg.scalping.initial_capital = 1000.0
    return RiskManager(cfg)


def test_daily_trade_limit_flag_blocks_entries_after_the_cap():
    rm = risk_manager("--daily-trade-limit", "2", "--cooldown-seconds", "0")
    for minute in (0, 1):
        assert rm.can_trade(T0.replace(minute=minute))[0] is True
        rm.record_trade(1.0, "BTC/USDT", T0.replace(minute=minute))
    allowed, reason = rm.can_trade(T0.replace(minute=2))
    assert allowed is False and "Daily trade limit reached (2/2)" in reason


def test_cooldown_flag_blocks_entries_after_the_last_closed_trade():
    rm = risk_manager("--cooldown-seconds", "3600")
    rm.record_trade(1.0, "BTC/USDT", T0)
    assert rm.can_trade(T0.replace(minute=30)) == (False, "Cooldown between trades")
    assert rm.can_trade(T0.replace(hour=13, minute=1))[0] is True


def test_daily_loss_flag_blocks_until_the_utc_date_changes():
    rm = risk_manager("--max-daily-loss-pct", "5", "--max-drawdown-pct", "20")
    late = datetime(2026, 1, 1, 23, 59, 0)
    # RiskManager starts on today's real date; sync it to the simulated day first.
    rm.can_trade(datetime(2026, 1, 1, 23, 58, 0))
    rm.record_trade(-60.0, "BTC/USDT", late)
    allowed, reason = rm.can_trade(datetime(2026, 1, 1, 23, 59, 30))
    assert allowed is False and "Max daily loss" in reason
    assert rm.can_trade(datetime(2026, 1, 2, 0, 0, 1))[0] is True
