"""Offline checks for portfolio-level paper profit targets."""
import json
from datetime import datetime
from pathlib import Path

import pytest

import main
from config.settings import BotConfig
from core import exchange as exchange_module
from core.adaptive_leverage import build_paper_trade_plan
from core.exchange import PaperExchange
from core.risk_manager import RiskManager


def _paper_bot(tmp_path, monkeypatch, *, capital=10.0, positions=None, marks=None):
    cfg = BotConfig()
    cfg.mode = "paper"
    cfg.scalping.initial_capital = capital
    cfg.state_db_path = str(tmp_path / "paper.db")
    cfg.state_file_path = str(tmp_path / "paper.state.json")
    cfg.portfolio_profit_target_pct = 70.0
    cfg.portfolio_profit_target_enabled = True
    cfg.backtest.taker_fee = 0.0005
    cfg.backtest.slippage = 0.0003
    cfg.backtest.funding_rate = 0.0

    monkeypatch.setattr(main.state, "DB_PATH", Path(cfg.state_db_path))
    rm = RiskManager(cfg)
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = cfg
    exchange._balance = capital
    exchange._balance_source = rm
    exchange.positions = {}
    exchange.orders = {}
    exchange.trade_history = []
    exchange._simulate_latency = lambda: None
    exchange._simulate_slippage = lambda price, _side: price
    exchange.cancel_entry_orders = PaperExchange.cancel_entry_orders.__get__(exchange)
    exchange.cancel_all_orders = PaperExchange.cancel_all_orders.__get__(exchange)

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.risk_manager = rm
    bot.exchange = exchange
    bot.clock = datetime.now
    bot._clock_uses_local_timezone = False
    bot.initial_portfolio_equity = capital
    bot.portfolio_target_status = "ACTIVE"
    bot.trading_halted = False
    bot.position_targets = {}
    bot.position_entry_cycle = {}
    bot.active_decisions = {}
    bot.paper_trade_meta = {}
    bot.cycle_count = 1
    bot.entries_blocked = False

    for symbol, (entry, quantity, side) in (positions or {}).items():
        mark = (marks or {}).get(symbol, entry)
        exchange.positions[symbol] = {
            "symbol": symbol, "entryPrice": entry, "markPrice": mark,
            "contracts": quantity, "side": side, "leverage": 1.0,
            "margin": entry * quantity,
        }
        rm.register_position(symbol, entry * quantity, side, leverage=1.0,
                             margin=entry * quantity, quantity=quantity,
                             strategy="test")
        rm.update_position_price(symbol, entry)
        rm.update_position_price(symbol, mark)
        bot.paper_trade_meta[symbol] = {
            "entry_fee": entry * quantity * cfg.backtest.taker_fee,
            "entry_slippage": entry * quantity * cfg.backtest.slippage,
            "entry_time": datetime.now(), "strategy": "test",
            "selected_leverage": 1.0,
        }
    return bot


@pytest.mark.parametrize("capital,target", [(10.0, 17.0), (100.0, 170.0)])
def test_target_equity_is_based_on_original_starting_capital(tmp_path, monkeypatch, capital, target):
    bot = _paper_bot(tmp_path, monkeypatch, capital=capital)
    assert bot.portfolio_target_metrics()["target_portfolio_equity"] == pytest.approx(target)
    bot.risk_manager.current_balance += 3.0
    assert bot.portfolio_target_metrics()["target_portfolio_equity"] == pytest.approx(target)


def test_unrealized_target_crossing_liquidates_but_fees_can_leave_target_unmet(
    tmp_path, monkeypatch,
):
    bot = _paper_bot(tmp_path, monkeypatch,
                     positions={"BTC/USDT": (100.0, 1.0, "long")},
                     marks={"BTC/USDT": 107.0})
    assert bot.portfolio_target_metrics()["current_portfolio_equity"] == pytest.approx(17.0)

    assert bot.check_portfolio_profit_target() is True

    assert not bot.exchange.positions
    assert bot.risk_manager.current_balance == pytest.approx(16.8344)
    assert bot.portfolio_target_status == "TARGET_NOT_REACHED_AFTER_LIQUIDATION"
    assert bot.trading_halted is True
    assert bot.portfolio_target_metrics()["unrealized_pnl"] == pytest.approx(0.0)
    report_path = Path(bot._portfolio_target_report_path())
    report = json.loads(report_path.read_text())
    assert report["target_reached"] is False
    assert report["target_status"] == "TARGET_NOT_REACHED_AFTER_LIQUIDATION"


def test_target_closes_every_paper_position_and_cancels_pending_entries(
    tmp_path, monkeypatch,
):
    bot = _paper_bot(
        tmp_path, monkeypatch,
        positions={"BTC/USDT": (100.0, 0.5, "long"),
                   "ETH/USDT": (100.0, 0.5, "long")},
        marks={"BTC/USDT": 107.2, "ETH/USDT": 107.2},
    )
    bot.exchange.orders = {
        "entry": {"id": "entry", "symbol": "SOL/USDT", "status": "open"},
        "reduce": {"id": "reduce", "symbol": "BTC/USDT", "status": "open",
                   "reduceOnly": True},
    }

    assert bot.check_portfolio_profit_target() is True

    assert bot.exchange.positions == {}
    assert bot.risk_manager.open_positions == {}
    assert "entry" not in bot.exchange.orders
    assert bot.portfolio_target_status == "TARGET_REACHED"
    assert bot.risk_manager.current_balance >= 17.0


def test_entries_are_blocked_after_target_activation():
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = BotConfig()
    bot.config.mode = "paper"
    bot.trading_halted = True
    bot.process_symbol("BTC/USDT")


def test_target_persists_through_session_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first_cfg = main.build_config_from_args([
        "--fresh", "--paper-session-id", "target-audit", "--capital", "10",
        "--portfolio-profit-target-pct", "70",
    ], environ={})
    bot = _paper_bot(tmp_path, monkeypatch)
    bot.config = first_cfg
    bot.initial_portfolio_equity = 10.0
    bot.risk_manager.current_balance = 12.25
    bot.portfolio_target_status = "ACTIVE"
    bot.persist_state()
    main.state.set_database(bot.config.state_db_path)
    main.state.save_risk_state("session_seed", "ready")

    resumed_cfg = main.build_config_from_args([
        "--paper-session-id", "target-audit", "--capital", "10",
        "--portfolio-profit-target-pct", "90",
    ], environ={})
    resumed = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    resumed.config = resumed_cfg
    resumed.risk_manager = RiskManager(resumed_cfg)
    resumed.exchange = type("PaperPositionView", (), {"get_positions": lambda _self: {}})()
    resumed.loaded_positions = []
    resumed.position_targets = {}
    resumed.position_entry_cycle = {}
    resumed.entries_blocked = False
    resumed.initial_portfolio_equity = 10.0
    resumed.portfolio_target_status = "ACTIVE"
    resumed.trading_halted = False
    main.state.set_database(resumed_cfg.state_db_path)

    resumed.load_state()

    assert resumed.risk_manager.current_balance == pytest.approx(12.25)
    assert resumed.initial_portfolio_equity == pytest.approx(10.0)
    assert resumed.config.portfolio_profit_target_pct == pytest.approx(70.0)
    assert resumed.portfolio_target_status == "ACTIVE"


@pytest.mark.parametrize("args", [
    ["--adaptive-leverage", "--max-leverage", "3.01"],
    ["--adaptive-leverage", "--max-margin-allocation-pct", "0.51"],
    ["--adaptive-leverage", "--max-gross-exposure-pct", "1.51"],
    ["--max-open-positions", "2"],
])
def test_portfolio_safety_caps_reject_overrides(args):
    with pytest.raises(SystemExit):
        main.build_config_from_args(args, environ={})


def test_target_can_be_disabled_and_target_percentage_is_validated():
    cfg = main.build_config_from_args(["--disable-portfolio-profit-target"], environ={})
    assert cfg.portfolio_profit_target_enabled is False
    with pytest.raises(SystemExit, match="portfolio-profit-target-pct"):
        main.build_config_from_args(["--portfolio-profit-target-pct", "nan"], environ={})


def test_profit_objective_does_not_raise_risk_or_override_drawdown_controls():
    cfg = BotConfig()
    cfg.scalping.initial_capital = 10.0
    cfg.portfolio_profit_target_pct = 70.0
    cfg.adaptive_leverage.enabled = True
    cfg.risk.risk_per_trade = 0.02  # planner must retain its hard 1% ceiling
    rm = RiskManager(cfg)
    rm.current_balance = 9.6
    rm.peak_equity = 10.0
    rm.max_drawdown_pct = 0.04
    history = ([{"net_return": -0.01, "oos_validated": True}] * 25 +
               [{"net_return": 0.03, "oos_validated": True}] * 75)

    plan = build_paper_trade_plan(cfg, rm, history, strategy="test",
                                  entry=100.0, stop=98.0, side="long")

    assert plan["allowed"]
    assert plan["leverage"] == 1.0
    assert plan["risk_budget"] <= 0.096


def test_paper_configuration_selects_simulator_not_live_connector(monkeypatch, tmp_path):
    class PublicDataOnly:
        id = "kucoin"

        def __init__(self, _options):
            pass

        def load_markets(self):
            return {"BTC/USDT": {"limits": {}}}

        def fetch_ohlcv(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(exchange_module.ccxt, "kucoin", PublicDataOnly, raising=False)
    monkeypatch.setattr(main.ExchangeConnector, "connect",
                        lambda _self: pytest.fail("live connector must not be used in paper mode"))
    monkeypatch.setattr(main.state, "DB_PATH", tmp_path / "paper.db")
    cfg = main.build_config_from_args(["--exchange", "kucoin", "--capital", "10"], environ={})
    bot = main.UnifiedCryptoBot(cfg)
    assert isinstance(bot.exchange, PaperExchange)
    assert bot.exchange.positions == {}

