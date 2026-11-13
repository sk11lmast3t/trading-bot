"""Portfolio heat, margin capacity, and drawdown units."""
import pytest

from config.settings import BotConfig
from core.risk_manager import RiskManager


def manager(**overrides):
    cfg = BotConfig()
    cfg.scalping.initial_capital = 100.0
    cfg.risk.risk_per_trade = overrides.pop("risk_per_trade", 0.01)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg, RiskManager(cfg)


def open_position(rm):
    rm.register_position("BTC/USDT", 100.0, "long", leverage=5.0, margin=20.0,
                         quantity=1.0, stop_price=95.0)
    rm.update_position_price("BTC/USDT", 100.0)


def test_portfolio_heat_includes_stop_costs_and_margin_is_separate():
    cfg, rm = manager()
    open_position(rm)

    stats = rm.get_stats()
    expected_heat = rm.stop_risk_for_position(rm.open_positions["BTC/USDT"])

    assert stats["portfolio_heat"] == pytest.approx(expected_heat)
    assert stats["portfolio_heat"] > 5.0  # stop distance plus closing costs
    assert stats["total_margin_used"] == pytest.approx(20.0)
    assert stats["available_account_equity"] == pytest.approx(80.0)
    # The configured aggregate margin cap is 50% even when adaptive leverage
    # is disabled; remaining capacity is not 100% of equity.
    assert stats["margin_allocation_cap"] == pytest.approx(50.0)
    assert stats["remaining_margin_capacity"] == pytest.approx(30.0)
    assert stats["available_margin"] == pytest.approx(30.0)


def test_closed_position_releases_heat_and_margin():
    cfg, rm = manager()
    open_position(rm)
    rm.release_position("BTC/USDT")

    stats = rm.get_stats()

    assert stats["portfolio_heat"] == pytest.approx(0.0)
    assert stats["total_margin_used"] == pytest.approx(0.0)
    assert stats["unpriced_stop_risk_positions"] == 0
    assert stats["available_account_equity"] == pytest.approx(100.0)
    assert stats["remaining_margin_capacity"] == pytest.approx(50.0)


def test_unpriced_open_position_is_reported_not_silently_ignored():
    cfg, rm = manager()
    rm.register_position("BTC/USDT", 100.0, "long", leverage=1.0, margin=100.0,
                         quantity=1.0, stop_price=None)
    rm.update_position_price("BTC/USDT", 100.0)

    stats = rm.get_stats()

    assert stats["unpriced_stop_risk_positions"] == 1
    assert stats["portfolio_heat"] == pytest.approx(0.0)


def test_stop_risk_budget_uses_equity_and_keeps_one_percent_ceiling():
    cfg, rm = manager(risk_per_trade=0.02)
    open_position(rm)
    rm.update_position_price("BTC/USDT", 110.0)  # unrealized +10

    stats = rm.get_stats()

    assert stats["portfolio_equity"] == pytest.approx(110.0)
    assert stats["stop_risk_budget"] == pytest.approx(1.10)  # 1% hard cap, not 2%
    assert stats["stop_risk_budget_pct"] == pytest.approx(1.0)


def test_max_drawdown_reports_percentage_points_and_fraction():
    cfg, rm = manager()
    rm.max_drawdown_pct = 0.03

    stats = rm.get_stats()

    assert stats["max_drawdown"] == pytest.approx(3.0)
    assert stats["max_drawdown_pct"] == pytest.approx(3.0)
    assert stats["max_drawdown_fraction"] == pytest.approx(0.03)
