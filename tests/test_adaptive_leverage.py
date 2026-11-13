"""Deterministic, network-free tests for the paper adaptive leverage engine."""
import pytest

import main
from config.settings import BotConfig
from core.adaptive_leverage import (
    build_paper_trade_plan,
    estimate_edge,
    portfolio_snapshot,
    select_leverage,
)
from core.exchange import PaperExchange
from core.risk_manager import RiskManager


def outcomes(wins=75, losses=25, win=0.03, loss=0.01):
    return ([{"net_return": -loss, "oos_validated": True} for _ in range(losses)] +
            [{"net_return": win, "oos_validated": True} for _ in range(wins)])


def test_insufficient_history_and_eleven_trades_stay_at_one_x():
    for count in (0, 11, 99):
        estimate = estimate_edge(outcomes(wins=count, losses=0))
        assert select_leverage(estimate, max_leverage=3.0) == (1.0, "insufficient_oos_evidence")


def test_strong_validated_edge_increases_leverage_but_confidence_is_not_an_input():
    estimate = estimate_edge(outcomes())
    selected, reason = select_leverage(estimate, max_leverage=3.0)
    assert selected == 3.0
    assert reason == "exceptional_validated_edge"


def test_high_strategy_confidence_alone_cannot_bypass_oos_sample_gate():
    cfg = BotConfig()
    cfg.adaptive_leverage.enabled = True
    rm = RiskManager(cfg)
    plan = build_paper_trade_plan(
        cfg, rm, outcomes(wins=8, losses=3), strategy="fixed-rule",
        entry=100.0, stop=98.0, side="long", signal_confidence=0.99,
    )
    assert plan["allowed"]
    assert plan["leverage"] == 1.0
    assert plan["signal_confidence"] == 0.99
    assert plan["historical_sample_size"] == 11


@pytest.mark.parametrize("returns,expected_reason", [
    (outcomes(wins=20, losses=80), "non_positive_conservative_expectancy"),
    (outcomes(wins=100, losses=0), "insufficient_oos_evidence"),
])
def test_negative_or_one_sided_evidence_falls_back_to_one_x(returns, expected_reason):
    selected, reason = select_leverage(estimate_edge(returns), max_leverage=3.0)
    assert selected == 1.0
    assert reason == expected_reason


def test_deterioration_loss_streak_drawdown_and_circuit_breaker_reduce_leverage():
    history = outcomes()
    history[-25:] = [{"net_return": -0.02, "oos_validated": True} for _ in range(25)]
    estimate = estimate_edge(history)
    assert estimate.deteriorating
    assert select_leverage(estimate, max_leverage=3.0)[0] < 3.0

    strong = estimate_edge(outcomes())
    assert select_leverage(strong, max_leverage=3.0, consecutive_losses=1)[0] < 3.0
    assert select_leverage(strong, max_leverage=3.0, drawdown=0.04)[0] == 1.0
    assert select_leverage(strong, max_leverage=3.0, circuit_breaker_active=True)[0] == 1.0


def test_portfolio_snapshot_includes_margin_exposure_and_unrealized_pnl():
    snapshot = portfolio_snapshot(10.0, {
        "BTC/USDT": {"entry_price": 100.0, "mark_price": 110.0,
                     "quantity": 0.1, "leverage": 2.0, "margin": 5.0,
                     "side": "long"},
    }, 0.50, 1.50)
    assert snapshot.equity == pytest.approx(11.0)
    assert snapshot.used_margin == pytest.approx(5.0)
    assert snapshot.available_margin == pytest.approx(0.5)
    assert snapshot.gross_exposure == pytest.approx(11.0)
    assert snapshot.remaining_exposure == pytest.approx(5.5)


def test_new_trade_plan_is_limited_by_existing_margin_and_gross_exposure():
    cfg = BotConfig()
    cfg.scalping.initial_capital = 10.0
    cfg.adaptive_leverage.enabled = True
    cfg.adaptive_leverage.max_gross_exposure_pct = 1.5
    cfg.risk.max_position_size_pct = 1.5
    rm = RiskManager(cfg)
    rm.register_position("BTC/USDT", 14.5, "long", leverage=3.0,
                         margin=4.9, quantity=0.145, strategy="other")
    rm.update_position_price("BTC/USDT", 100.0)
    plan = build_paper_trade_plan(cfg, rm, outcomes(), strategy="fixed-rule",
                                  entry=100.0, stop=98.0, side="long")
    assert plan["allowed"]
    assert plan["position_notional"] <= 0.5 + 1e-9
    assert plan["required_margin"] <= 0.1 + 1e-9


def test_floating_portfolio_loss_updates_drawdown_telemetry():
    cfg = BotConfig()
    cfg.scalping.initial_capital = 10.0
    rm = RiskManager(cfg)
    rm.register_position("BTC/USDT", 5.0, "long", leverage=1.0,
                         margin=5.0, quantity=0.05)
    rm.update_position_price("BTC/USDT", 100.0)
    rm.update_position_price("BTC/USDT", 94.0)
    assert rm.get_stats()["max_drawdown"] == pytest.approx(0.03 * 100.0)


def test_trade_plan_preserves_one_percent_stop_risk_and_costs():
    cfg = BotConfig()
    cfg.scalping.initial_capital = 10.0
    cfg.adaptive_leverage.enabled = True
    cfg.adaptive_leverage.min_oos_trades = 100
    cfg.adaptive_leverage.max_leverage = 3.0
    cfg.adaptive_leverage.max_gross_exposure_pct = 1.5
    cfg.risk.max_position_size_pct = 1.5
    cfg.backtest.taker_fee = 0.001
    cfg.backtest.slippage = 0.001
    cfg.backtest.funding_rate = 0.0008
    cfg.risk.max_hold_time_minutes = 480
    rm = RiskManager(cfg)
    plan = build_paper_trade_plan(cfg, rm, outcomes(), strategy="fixed-rule",
                                  entry=100.0, stop=98.0, side="long",
                                  signal_confidence=0.99)
    assert plan["allowed"]
    assert plan["leverage"] == 3.0
    assert plan["planned_stop_risk"] <= 0.1 + 1e-9
    assert plan["planned_stop_risk"] == pytest.approx(plan["risk_budget"])
    assert plan["position_notional"] <= 15.0
    assert plan["required_margin"] <= 5.0
    assert plan["signal_confidence"] == 0.99
    assert plan["win_probability"] == pytest.approx(0.75)


def test_invalid_leverage_values_and_live_adaptive_mode_are_rejected():
    with pytest.raises(SystemExit, match="--min-leverage and --max-leverage"):
        main.build_config_from_args(["--adaptive-leverage", "--max-leverage", "0.5"], environ={})
    with pytest.raises(SystemExit, match="only in paper mode"):
        main.build_config_from_args(["live", "--adaptive-leverage"],
                                    environ={"CONFIRM_LIVE": "YES"})


def test_adaptive_configuration_keeps_notional_cap_distinct_from_margin_and_gross():
    cfg = main.build_config_from_args([
        "--adaptive-leverage", "--capital", "10", "--max-leverage", "3",
        "--max-margin-allocation-pct", "0.50", "--max-gross-exposure-pct", "1.50",
        "--max-position-notional-pct", "1.20",
    ], environ={})
    summary = main.effective_config_summary(cfg)
    assert cfg.risk.max_position_size_pct == pytest.approx(1.20)
    assert summary["MAX POSITION NOTIONAL"] == pytest.approx(12.0)
    assert summary["MAX MARGIN ALLOCATION"] == pytest.approx(5.0)
    assert summary["MAX GROSS EXPOSURE"] == pytest.approx(15.0)
    assert summary["REAL ORDER EXECUTION"] == "DISABLED"


def test_paper_order_rejects_leverage_above_configured_limit_without_network():
    cfg = BotConfig()
    cfg.adaptive_leverage.enabled = True
    cfg.adaptive_leverage.max_leverage = 3.0
    cfg.adaptive_leverage.max_gross_exposure_pct = 1.5
    cfg.risk.max_position_size_pct = 1.5
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = cfg
    exchange.exec_cfg = cfg.execution
    exchange._balance = 10.0
    exchange._balance_source = None
    exchange.latency_ms = 0
    exchange.base_slippage_pct = 0.0
    exchange.partial_fill_chance = 0.0
    exchange.reject_chance = 0.0
    exchange._requested_leverage = {}
    exchange.positions = {}
    exchange.orders = {}
    exchange.trade_history = []
    exchange.order_id = 0
    exchange._market_metadata = {"BTC/USDT": {"limits": {}}}
    exchange._market_metadata_error = None
    exchange._simulate_latency = lambda: None
    exchange._simulate_slippage = lambda price, side: price
    exchange._simulate_outcome = lambda: None

    order = exchange.create_market_order("BTC/USDT", "buy", 0.01,
                                         {"price": 100.0, "leverage": 4.0})
    assert order["status"] == "rejected"
    assert "configured adaptive range" in order["reject_reason"]

