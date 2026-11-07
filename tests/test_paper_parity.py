from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

import main
from config.settings import BotConfig
from core import exchange as exchange_module
from core.exchange import PaperExchange
from core.risk_manager import RiskManager
from core.strategies import MtfTrendPullbackStrategy, Signal, TradeSetup


def test_demo_always_selects_paper_mode():
    cfg = main.build_config_from_args(["live", "--demo"], environ={})
    assert cfg.mode == "paper"


def test_live_requires_explicit_confirmation():
    with pytest.raises(SystemExit, match="explicit 'live' argument and CONFIRM_LIVE=YES"):
        main.build_config_from_args(["live"], environ={})


def test_live_requires_live_argument_even_with_confirmation(monkeypatch):
    cfg = BotConfig()
    cfg.exchange.api_key = "test-key"
    cfg.exchange.api_secret = "test-secret"
    monkeypatch.setattr(main, "BotConfig", lambda: cfg)

    assert main.build_config_from_args([], environ={"CONFIRM_LIVE": "YES"}).mode == "paper"
    assert main.build_config_from_args(["live"], environ={"CONFIRM_LIVE": "YES"}).mode == "live"


def test_parity_profile_applies_requested_settings():
    cfg = main.build_config_from_args([
        "--parity", "--symbols", "BTC/USDT,SOL/USDT", "--exchange", "kraken",
        "--hold-minutes", "45", "--capital", "250",
    ], environ={})

    assert cfg.mode == "paper"
    assert cfg.parity is True
    assert cfg.exchange.name == "kraken"
    assert cfg.scalping.symbols == ["BTC/USDT", "SOL/USDT"]
    assert cfg.scalping.scan_all_coins is False
    assert cfg.risk.max_hold_time_minutes == 45
    assert cfg.risk.trailing_stop_enabled is False
    assert cfg.risk.breakeven_activation_pct == 0
    assert cfg.risk.trailing_stop_activation_pct == 0
    assert cfg.execution.paper_partial_fill_chance == 0
    assert cfg.execution.paper_reject_chance == 0
    assert cfg.scalping.initial_capital == 250


def test_parity_rejects_live_mode():
    with pytest.raises(SystemExit, match="--parity is available only in paper mode"):
        main.build_config_from_args(["live", "--parity"], environ={"CONFIRM_LIVE": "YES"})


def test_paper_exchange_sticks_to_one_candle_source_and_skips_on_error(monkeypatch, caplog):
    calls = []

    class FakeExchange:
        id = "kraken"

        def fetch_ohlcv(self, symbol, timeframe, limit):
            calls.append((symbol, timeframe, limit))
            raise RuntimeError("offline")

    fake_exchange = FakeExchange()
    monkeypatch.setattr(exchange_module.ccxt, "kraken", lambda config: fake_exchange)
    cfg = BotConfig()
    cfg.exchange.name = "kraken"
    exchange = PaperExchange(cfg)

    assert exchange._public_exchanges == [fake_exchange]
    assert exchange.fetch_ohlcv("BTC/USDT", "5m", 722) is None
    assert calls == [("BTC/USDT", "5m", 722)]
    assert "skipping bar" in caplog.text


def test_process_symbol_fetches_mtf_history_and_executes_synthetic_signal(monkeypatch, tmp_path):
    cfg = BotConfig()
    cfg.strategy.active_strategy = "mtf_trend_pullback"
    cfg.scalping.scan_all_coins = False
    cfg.execution.entry_mode = "market"

    idx = pd.date_range("2026-01-01", periods=724, freq="5min")
    close = np.linspace(100.0, 110.0, len(idx))
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=idx)

    class FakeExchange:
        def __init__(self):
            self.requested_limit = None
            self.positions = {}
            self.entries = []

        def fetch_ohlcv(self, symbol, timeframe, limit):
            self.requested_limit = limit
            return bars.iloc[-limit:]

        def get_position(self, symbol):
            return self.positions.get(symbol)

        def create_market_order(self, symbol, side, amount, params=None):
            price = params["price"]
            self.positions[symbol] = {
                "symbol": symbol, "side": "long", "contracts": amount,
                "entryPrice": price,
            }
            self.entries.append(symbol)
            return {"status": "closed", "filled": amount, "price": price}

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.strategy = MtfTrendPullbackStrategy(cfg)
    exchange = FakeExchange()
    bot.exchange = exchange
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {}
    bot.risk_manager.current_balance = cfg.scalping.initial_capital
    bot.risk_manager.can_trade.return_value = (True, "OK")
    bot.risk_manager.calculate_position_size.return_value = 0.1
    bot.risk_manager.check_portfolio_heat.return_value = (True, "OK")
    bot.cycle_count = 1
    bot.position_entry_cycle = {}
    bot.position_targets = {}
    bot.entries_blocked = False
    bot.last_rejection_log = {}
    bot.decision_log_path = str(tmp_path / "decisions.csv")
    seen_rows = []

    def synthetic_signal(window):
        seen_rows.append(len(window))
        return TradeSetup(Signal.LONG, 110.0, 108.0, 114.0, 0.7, "mtf_trend_pullback")

    monkeypatch.setattr(bot.strategy, "analyze", synthetic_signal)
    monkeypatch.setattr(main.state, "save_position_at_entry", lambda *args, **kwargs: None)
    bot.process_symbol("BTC/USDT")

    assert exchange.requested_limit >= 722
    assert seen_rows and max(seen_rows) >= 720
    assert exchange.entries == ["BTC/USDT"]
    meta = bot.paper_trade_meta["BTC/USDT"]
    assert meta["entry_fee"] == pytest.approx(110.0 * 0.1 * cfg.backtest.taker_fee)
    assert meta["entry_slippage"] == pytest.approx(110.0 * 0.1 * cfg.backtest.slippage)
    rows = pd.read_csv(bot.decision_log_path)
    assert rows.loc[0, "status"] == "entered"
    assert rows.loc[0, "fill_price"] == pytest.approx(110.0)


def test_process_symbol_enters_at_most_once_for_same_completed_bar(monkeypatch, tmp_path):
    cfg = BotConfig()
    cfg.scalping.scan_all_coins = False
    cfg.execution.entry_mode = "market"
    idx = pd.date_range("2026-01-01", periods=302, freq="5min")
    close = np.linspace(100.0, 105.0, len(idx))
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=idx)

    class FakeExchange:
        def __init__(self):
            self.positions = {}
            self.entries = []

        def fetch_ohlcv(self, symbol, timeframe, limit):
            return bars.iloc[-limit:]

        def get_position(self, symbol):
            return self.positions.get(symbol)

        def create_market_order(self, symbol, side, amount, params=None):
            price = params["price"]
            self.positions[symbol] = {
                "symbol": symbol, "side": "long", "contracts": amount,
                "entryPrice": price,
            }
            self.entries.append(symbol)
            return {"status": "closed", "filled": amount, "price": price}

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.strategy = MagicMock()
    bot.strategy.lookback_bars = 300
    bot.strategy.analyze.return_value = TradeSetup(Signal.LONG, 105.0, 103.0, 109.0, 0.7, "test")
    exchange = FakeExchange()
    bot.exchange = exchange
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {}
    bot.risk_manager.current_balance = cfg.scalping.initial_capital
    bot.risk_manager.can_trade.return_value = (True, "OK")
    bot.risk_manager.calculate_position_size.return_value = 0.1
    bot.risk_manager.check_portfolio_heat.return_value = (True, "OK")
    bot.cycle_count = 1
    bot.position_entry_cycle = {}
    bot.position_targets = {}
    bot.entries_blocked = False
    bot.last_rejection_log = {}
    bot.last_processed_bar = {}
    bot.decision_log_path = str(tmp_path / "decisions.csv")
    monkeypatch.setattr(main.state, "save_position_at_entry", lambda *args, **kwargs: None)

    bot.process_symbol("BTC/USDT")
    bot.process_symbol("BTC/USDT")

    assert bot.strategy.analyze.call_count == 1
    assert exchange.entries == ["BTC/USDT"]
    rows = pd.read_csv(bot.decision_log_path)
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "entered"


def test_hold_signal_is_logged_as_a_skip(monkeypatch, tmp_path):
    cfg = BotConfig()
    idx = pd.date_range("2026-01-01", periods=302, freq="5min")
    close = np.linspace(100.0, 105.0, len(idx))
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=idx)

    class FakeExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return bars.iloc[-limit:]

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.strategy = MagicMock()
    bot.strategy.lookback_bars = 300
    bot.strategy.analyze.return_value = TradeSetup(Signal.HOLD)
    bot.exchange = FakeExchange()
    bot.risk_manager = MagicMock()
    bot.risk_manager.current_balance = cfg.scalping.initial_capital
    bot.entries_blocked = False
    bot.last_processed_bar = {}
    bot.decision_log_path = str(tmp_path / "decisions.csv")

    bot.process_symbol("BTC/USDT")

    rows = pd.read_csv(bot.decision_log_path)
    assert list(rows.columns) == [
        "ts", "symbol", "strategy", "signal", "intended_entry", "sl", "tp",
        "status", "fill_price", "fee", "slippage", "exit_price", "exit_reason",
        "hold_minutes", "net_pnl", "balance_after",
    ]
    assert len(rows) == 1
    assert rows.loc[0, "signal"] == "HOLD"
    assert rows.loc[0, "status"] == "skipped:hold"


def test_paper_close_applies_net_costs_once_to_risk_manager_balance(monkeypatch, tmp_path):
    cfg = BotConfig()
    cfg.scalping.scan_all_coins = False
    risk_manager = RiskManager(cfg)
    position = {
        "symbol": "BTC/USDT", "side": "long", "contracts": 2.0,
        "entryPrice": 100.0,
    }
    exchange = PaperExchange.__new__(PaperExchange)
    exchange._balance = cfg.scalping.initial_capital
    exchange._balance_source = risk_manager
    exchange.latency_ms = 0
    exchange.base_slippage_pct = 0.0
    exchange.positions = {"BTC/USDT": dict(position)}

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.exchange = exchange
    bot.risk_manager = risk_manager
    bot.paper_trade_meta = {
        "BTC/USDT": {
            "entry_fee": 100.0 * 2.0 * cfg.backtest.maker_fee,
            "entry_slippage": 0.0,
        }
    }
    bot.position_entry_cycle = {}
    bot.position_targets = {}
    bot.decision_log_path = str(tmp_path / "decisions.csv")
    bot.active_decisions = {
        "BTC/USDT": {
            "ts": "2026-01-01T00:00:00",
            "symbol": "BTC/USDT",
            "strategy": "test",
            "entry_time": datetime.now() - timedelta(minutes=5),
        }
    }
    bot.persist_state = lambda: None
    monkeypatch.setattr(main.state, "record_trade", lambda *args, **kwargs: None)
    bot._append_decision({
        "ts": "2026-01-01T00:00:00", "symbol": "BTC/USDT", "strategy": "test",
        "signal": "LONG", "intended_entry": 100.0, "sl": 90.0, "tp": 120.0,
        "status": "entered", "fill_price": 100.0,
    })

    bot._close_position("BTC/USDT", position, 110.0, "TAKE PROFIT")

    gross = (110.0 - 100.0) * 2.0
    exit_fee = 110.0 * 2.0 * cfg.backtest.taker_fee
    exit_slippage = 110.0 * 2.0 * cfg.backtest.slippage
    net = gross - 100.0 * 2.0 * cfg.backtest.maker_fee - exit_fee - exit_slippage
    assert risk_manager.trade_log[-1]["pnl"] == pytest.approx(net)
    assert risk_manager.current_balance == pytest.approx(cfg.scalping.initial_capital + net)
    assert exchange.balance == pytest.approx(risk_manager.current_balance)
    rows = pd.read_csv(bot.decision_log_path)
    assert len(rows) == 1
    assert rows.loc[0, "exit_price"] == pytest.approx(110.0)
    assert rows.loc[0, "exit_reason"] == "TAKE PROFIT"
    assert rows.loc[0, "net_pnl"] == pytest.approx(net)
    assert rows.loc[0, "fee"] == pytest.approx(100.0 * 2.0 * cfg.backtest.maker_fee + exit_fee)
    assert rows.loc[0, "slippage"] == pytest.approx(exit_slippage)
    assert rows.loc[0, "balance_after"] == pytest.approx(risk_manager.current_balance)
