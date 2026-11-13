from datetime import datetime, timedelta
from pathlib import Path
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


def _offline_ccxt_exchange():
    fake = MagicMock()
    fake.id = "kucoin"
    fake.load_markets.return_value = {"BTC/USDT": {"limits": {}}}
    fake.fetch_ohlcv.side_effect = RuntimeError("offline test fixture")
    return fake


def test_ten_dollar_paper_config_propagates_account_limits(monkeypatch):
    cfg = main.build_config_from_args([
        "--demo", "--exchange", "kucoin", "--symbols", "BTC/USDT", "ETH/USDT",
        "SOL/USDT", "--capital", "10", "--max-position-notional-pct", "0.50",
        "--max-open-positions", "1", "--fresh", "--max-cycles", "60",
    ], environ={})
    rm = RiskManager(cfg)
    monkeypatch.setattr(exchange_module.ccxt, "kucoin", lambda _cfg: _offline_ccxt_exchange())
    exchange = PaperExchange(cfg)
    exchange.bind_balance_source(rm)

    assert cfg.scalping.initial_capital == 10.0
    assert rm.current_balance == 10.0
    assert exchange.balance == 10.0
    assert cfg.risk.circuit_breaker_enabled is True
    assert cfg.risk.max_daily_loss_pct > 0
    rm.record_trade(0.5, "BTC/USDT")
    assert exchange.balance == 10.5
    assert rm.get_stats()["balance"] == pytest.approx(10.5)
    assert cfg.scalping.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert cfg.scalping.scan_all_coins is False
    assert cfg.risk.max_open_positions == cfg.scalping.max_positions == 1
    assert cfg.scalping.leverage == 1
    assert cfg.risk.risk_per_trade == pytest.approx(0.01)
    assert cfg.max_cycles == 60
    assert cfg.risk.max_position_size_pct == pytest.approx(0.50)


def test_configuration_repr_never_displays_exchange_credentials():
    cfg = BotConfig()
    cfg.exchange.api_key = "sentinel-paper-api-key"
    cfg.exchange.api_secret = "sentinel-paper-api-secret"

    rendered = repr(cfg)

    assert "sentinel-paper-api-key" not in rendered
    assert "sentinel-paper-api-secret" not in rendered


def test_explicit_symbols_scan_only_when_scan_flag_is_given():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    fixed = main.build_config_from_args(["--symbols", *symbols], environ={})
    scanning = main.build_config_from_args(
        ["--symbols", *symbols, "--scan-all-coins"], environ={})
    default = main.build_config_from_args([], environ={})

    assert fixed.scalping.symbols == symbols
    assert fixed.scalping.scan_all_coins is False
    assert scanning.scalping.symbols == symbols
    assert scanning.scalping.scan_all_coins is True
    assert default.scalping.scan_all_coins is True
    assert default.scalping.initial_capital == 1000.0


@pytest.mark.parametrize("args,match", [
    (["--capital", "0"], "--capital must be finite"),
    (["--capital", "nan"], "--capital must be finite"),
    (["--max-position-notional-pct", "1.1"], "max-position-notional-pct"),
    (["--max-open-positions", "0"], "max-open-positions"),
    (["--max-cycles", "0"], "max-cycles"),
])
def test_account_limit_flags_validate_inputs(args, match):
    with pytest.raises(SystemExit, match=match):
        main.build_config_from_args(args, environ={})


def test_position_allocation_cap_preserves_one_percent_risk_budget():
    cfg = main.build_config_from_args([
        "--capital", "10", "--max-position-notional-pct", "0.50",
    ], environ={})
    rm = RiskManager(cfg)
    amount = rm.calculate_position_size(10.0, 100.0, 98.0)

    # Sizing reserves the round-trip stop costs, so notional stays at or below
    # the 50% allocation cap while the *total* stop risk (price + costs) is the
    # full 1% budget rather than exceeding it.
    notional = amount * 100.0
    assert notional <= 5.0
    assert notional >= 4.0
    stop_risk = rm.stop_risk_for_position({
        "entry_price": 100.0, "stop_price": 98.0,
        "quantity": amount, "side": "long",
    })
    assert stop_risk == pytest.approx(0.10)
    assert amount * 2.0 <= 0.10 + 1e-12
    assert cfg.scalping.leverage == 1


@pytest.mark.parametrize("price", [None, 0, -1, float("nan"), float("inf")])
def test_paper_market_order_rejects_invalid_execution_prices(price):
    cfg = BotConfig()
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = cfg
    exchange._balance = 10.0
    exchange._balance_source = None
    exchange.latency_ms = 0
    exchange.base_slippage_pct = 0.0
    exchange.partial_fill_chance = 0.0
    exchange.reject_chance = 0.0
    exchange.order_id = 0
    exchange.trade_history = []
    exchange.orders = {}
    exchange.positions = {}
    exchange._market_metadata = {}
    exchange._market_metadata_error = None

    result = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.01,
        None if price is None else {"price": price})

    assert result["status"] == "rejected"
    assert result["reject_reason"]
    assert exchange.positions == {}


def test_paper_market_order_enforces_five_dollar_cap_and_market_minimum(monkeypatch):
    cfg = main.build_config_from_args([
        "--capital", "10", "--max-position-notional-pct", "0.50",
    ], environ={})
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = cfg
    exchange._balance = 10.0
    exchange._balance_source = None
    exchange.latency_ms = 0
    exchange.base_slippage_pct = 0.0
    exchange.partial_fill_chance = 0.0
    exchange.reject_chance = 0.0
    exchange.order_id = 0
    exchange.trade_history = []
    exchange.orders = {}
    exchange.positions = {}
    exchange._market_metadata = {}
    exchange._market_metadata_error = None

    exact_cap = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.05, {"price": 100.0})
    assert exact_cap["status"] == "closed"
    exchange.positions.clear()
    exchange.trade_history.clear()
    exchange.orders.clear()
    exchange.order_id = 0

    monkeypatch.setattr(exchange, "_simulate_slippage", lambda _price, _side: 101.0)
    slipped_over_cap = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.05, {"price": 100.0})
    assert slipped_over_cap["status"] == "rejected"
    assert "Slippage-adjusted" in slipped_over_cap["reject_reason"]
    exchange.positions.clear()
    exchange.trade_history.clear()
    exchange.orders.clear()
    exchange.order_id = 0
    monkeypatch.setattr(exchange, "_simulate_slippage", lambda price, _side: price)

    over_cap = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.06, {"price": 100.0})
    assert over_cap["status"] == "rejected"
    assert "exceeds configured maximum" in over_cap["reject_reason"]

    exchange._market_metadata = {"BTC/USDT": {"limits": {"cost": {"min": 6.0}}}}
    below_minimum = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.05, {"price": 100.0})
    assert below_minimum["status"] == "rejected"
    assert "below market minimum" in below_minimum["reject_reason"]

    cfg.scalping.leverage = 2
    wrong_leverage = PaperExchange.create_market_order(
        exchange, "BTC/USDT", "buy", 0.01, {"price": 100.0})
    assert wrong_leverage["status"] == "rejected"
    assert "exactly 1x leverage" in wrong_leverage["reject_reason"]


def test_fresh_session_uses_isolated_database_and_preserves_old_database(
    monkeypatch, tmp_path,
):
    old_db = tmp_path / "paper.db"
    old_db.write_bytes(b"preserve existing account")
    monkeypatch.setattr(main.state, "DB_PATH", old_db)
    cfg = main.build_config_from_args(["--fresh", "--capital", "10"], environ={})

    assert cfg.state_db_path != str(old_db)
    assert "paper_sessions" in cfg.state_db_path
    assert cfg.scalping.initial_capital == 10.0
    assert old_db.read_bytes() == b"preserve existing account"


def test_fresh_bot_starts_ten_dollars_without_reading_or_overwriting_old_database(
    monkeypatch, tmp_path,
):
    old_db = tmp_path / "paper.db"
    old_db.write_bytes(b"old account must remain untouched")
    monkeypatch.setattr(main.state, "DB_PATH", old_db)
    monkeypatch.setattr(exchange_module.ccxt, "kucoin", lambda _cfg: _offline_ccxt_exchange())
    load_state = MagicMock()
    monkeypatch.setattr(main.UnifiedCryptoBot, "load_state", load_state)
    cfg = main.build_config_from_args([
        "--fresh", "--capital", "10", "--exchange", "kucoin",
        "--symbols", "BTC/USDT",
    ], environ={})

    bot = main.UnifiedCryptoBot(cfg)

    assert bot.risk_manager.current_balance == 10.0
    assert bot.exchange.balance == 10.0
    assert main.state.DB_PATH == Path(cfg.state_db_path)
    load_state.assert_not_called()
    assert old_db.read_bytes() == b"old account must remain untouched"


def test_effective_configuration_summary_matches_ten_dollar_experiment():
    cfg = main.build_config_from_args([
        "--demo", "--exchange", "kucoin", "--symbols", "BTC/USDT", "ETH/USDT",
        "SOL/USDT", "--capital", "10", "--max-position-notional-pct", "0.50",
        "--max-open-positions", "1", "--fresh", "--max-cycles", "60",
    ], environ={})
    summary = main.effective_config_summary(cfg)

    assert summary["MODE"] == "paper"
    assert summary["EXCHANGE"] == "kucoin"
    assert summary["CAPITAL"] == 10.0
    assert summary["SYMBOLS"] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert summary["AUTO SCAN"] is False
    assert summary["MAX OPEN POSITIONS"] == 1
    assert summary["MAX POSITION NOTIONAL"] == 5.0
    assert summary["LEVERAGE"] == 1.0
    assert summary["REAL ORDER EXECUTION"] == "DISABLED"


def test_portfolio_exposure_respects_fifty_percent_aggregate_cap():
    cfg = main.build_config_from_args([
        "--capital", "10", "--max-position-notional-pct", "0.50",
    ], environ={})
    rm = RiskManager(cfg)
    rm.symbol_exposure["BTC/USDT"] = 4.0

    allowed, reason = rm.check_portfolio_heat("ETH/USDT", 2.0)

    assert allowed is False
    assert "Portfolio heat limit" in reason


def test_max_open_position_limit_blocks_second_entry(monkeypatch, tmp_path):
    cfg = main.build_config_from_args([
        "--symbols", "BTC/USDT", "--max-open-positions", "1",
    ], environ={})
    cfg.scalping.scan_all_coins = False
    index = pd.date_range("2026-01-01", periods=302, freq="5min")
    close = np.linspace(100.0, 105.0, len(index))
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)

    class FakePaperExchange:
        def __init__(self):
            self.create_market_order = MagicMock()

        def fetch_ohlcv(self, _symbol, _timeframe, limit):
            return bars.iloc[-limit:]

        def get_position(self, _symbol):
            return None

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.strategy = MagicMock()
    bot.strategy.lookback_bars = 300
    bot.strategy.analyze.return_value = TradeSetup(
        Signal.LONG, 105.0, 103.0, 109.0, 0.7, "test")
    bot.exchange = FakePaperExchange()
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {"ETH/USDT": {"notional": 5.0}}
    bot.risk_manager.current_balance = 10.0
    bot.risk_manager.can_trade.return_value = (True, "OK")
    bot.last_processed_bar = {}
    bot.last_rejection_log = {}
    bot.entries_blocked = False
    bot.decision_log_path = str(tmp_path / "decisions.csv")

    bot.process_symbol("BTC/USDT")

    bot.risk_manager.calculate_position_size.assert_not_called()
    bot.exchange.create_market_order.assert_not_called()
    assert pd.read_csv(bot.decision_log_path).loc[0, "status"] == "skipped:max_positions"


def test_paper_startup_never_connects_live_connector_or_submits_ccxt_order(
    monkeypatch,
):
    cfg = main.build_config_from_args(
        ["--demo", "--exchange", "kucoin", "--symbols", "BTC/USDT"], environ={})
    cfg.exchange.api_key = "paper-test-key"
    cfg.exchange.api_secret = "paper-test-secret"
    fake_public = _offline_ccxt_exchange()
    public_factory = MagicMock(return_value=fake_public)
    monkeypatch.setattr(exchange_module.ccxt, "kucoin", public_factory)
    connect = MagicMock(side_effect=AssertionError("live connector called"))
    monkeypatch.setattr(main.ExchangeConnector, "connect", connect)
    monkeypatch.setattr(main.UnifiedCryptoBot, "load_state", lambda _self: None)

    bot = main.UnifiedCryptoBot(cfg)
    bot.exchange.reject_chance = 0.0
    bot.exchange.partial_fill_chance = 0.0
    fill = bot.exchange.create_market_order(
        "BTC/USDT", "buy", 0.001, {"price": 100.0})

    assert cfg.mode == "paper"
    assert fill["status"] == "closed", fill.get("reject_reason")
    connect.assert_not_called()
    public_factory.assert_called_once_with({"enableRateLimit": True})
    fake_public.create_order.assert_not_called()


def test_paper_restart_restores_positions_from_selected_paper_database(monkeypatch, tmp_path):
    cfg = BotConfig()
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.exchange = PaperExchange.__new__(PaperExchange)
    bot.exchange.positions = {}
    bot.exchange.get_positions = MagicMock(return_value={})
    bot.risk_manager = RiskManager(cfg)
    bot.loaded_positions = []
    bot.entries_blocked = False
    bot.position_targets = {}
    bot.position_entry_cycle = {}
    bot.clock = lambda: datetime(2026, 1, 1, 0, 5)
    bot._clock_uses_local_timezone = False
    bot._state_path = lambda: str(tmp_path / "paper-session.state.json")
    bot.exchange.fetch_ohlcv = MagicMock(return_value=pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0],
         "volume": [1.0]},
        index=[pd.Timestamp("2026-01-01 00:05:00")],
    ))
    monkeypatch.setattr(main.state, "load_open_positions", lambda: [{
        "symbol": "BTC/USDT", "side": "long", "size": 0.05,
        "entry_price": 100.0, "tp": 105.0, "sl": 98.0,
        "strategy": "test", "entry_time": "2026-01-01T00:00:00", "status": "open",
    }])

    bot.load_state()

    assert bot.exchange.positions["BTC/USDT"]["contracts"] == pytest.approx(0.05)
    assert bot.risk_manager.open_positions["BTC/USDT"]["notional"] == pytest.approx(5.0)
    assert bot.risk_manager.symbol_exposure["BTC/USDT"] == pytest.approx(5.0)
    assert bot.entries_blocked is False


def test_max_cycles_stops_saves_state_and_prints_final_telemetry(monkeypatch, tmp_path):
    cfg = BotConfig()
    cfg.scalping.symbols = ["BTC/USDT"]
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.max_cycles = 1
    bot.cycle_count = 0
    bot.running = False
    cfg.state_file_path = str(tmp_path / "paper-session" / "state.json")
    bot.write_run_metadata = MagicMock()
    bot.show_banner = MagicMock()
    bot.check_exits = MagicMock()
    bot.process_symbol = MagicMock()
    bot.print_status = MagicMock()
    bot.risk_manager = type("Risk", (), {"open_positions": {
        "BTC/USDT": {"notional": 5.0, "side": "long"},
    }})()
    bot.position_targets = {}
    bot.position_entry_cycle = {}
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: pytest.fail("should not sleep"))

    bot.run()

    assert bot.cycle_count == 1
    assert bot.running is False
    persisted = main.json.loads((tmp_path / "paper-session" / "state.json").read_text())
    assert persisted["open_positions"]["BTC/USDT"]["notional"] == 5.0
    bot.print_status.assert_called_once()


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


def test_parity_without_capital_preserves_legacy_hundred_dollar_default():
    cfg = main.build_config_from_args(["--parity"], environ={})
    assert cfg.scalping.initial_capital == 100.0


def test_cost_flags_configure_paper_and_parity_runs():
    args = ["--maker-fee", "0.0002", "--taker-fee", "0.0005", "--slippage", "0.0003"]
    cfg = main.build_config_from_args(args, environ={})
    assert (cfg.backtest.maker_fee, cfg.backtest.taker_fee, cfg.backtest.slippage) == (
        0.0002, 0.0005, 0.0003,
    )
    parity = main.build_config_from_args(["--parity", *args], environ={})
    assert (parity.backtest.maker_fee, parity.backtest.taker_fee, parity.backtest.slippage) == (
        0.0002, 0.0005, 0.0003,
    )


def test_cost_flags_reject_negative_values():
    with pytest.raises(SystemExit, match="--maker-fee must be non-negative"):
        main.build_config_from_args(["--maker-fee", "-0.1"], environ={})


def test_parity_rejects_live_mode():
    with pytest.raises(SystemExit, match="--parity is available only in paper mode"):
        main.build_config_from_args(["live", "--parity"], environ={"CONFIRM_LIVE": "YES"})


def test_strategy_cli_selects_name_and_keeps_default():
    assert main.build_config_from_args([], environ={}).strategy.active_strategy == "ema_rsi_atr"
    cfg = main.build_config_from_args(["--strategy", "grid_scalper"], environ={})
    assert cfg.strategy.active_strategy == "grid_scalper"


def test_symbols_cli_accepts_space_and_comma_separators():
    cfg = main.build_config_from_args([
        "--symbols", "BTC/USDT", "ETH/USDT,XRP/USDT", "SOL/USDT",
    ], environ={})
    assert cfg.scalping.symbols == ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]


def test_fresh_is_paper_only_and_skips_position_restore(monkeypatch):
    cfg = main.build_config_from_args(["--fresh"], environ={})
    assert cfg.mode == "paper"
    assert cfg.fresh is True
    with pytest.raises(SystemExit, match="--fresh is available only in paper mode"):
        main.build_config_from_args(["live", "--fresh"], environ={"CONFIRM_LIVE": "YES"})

    monkeypatch.setattr(main, "PaperExchange", MagicMock())
    monkeypatch.setattr(main.state, "set_database", lambda _path: None)
    cfg.scalping.scan_all_coins = False
    load_state = MagicMock()
    monkeypatch.setattr(main.UnifiedCryptoBot, "load_state", load_state)
    bot = main.UnifiedCryptoBot(config=cfg)
    load_state.assert_not_called()
    assert bot.loaded_positions == []


@pytest.mark.parametrize("mode,expected_block", [("paper", False), ("live", True)])
def test_reconcile_mismatch_blocks_only_live_entries(monkeypatch, mode, expected_block):
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = BotConfig(mode=mode)
    bot.entries_blocked = not expected_block
    bot.loaded_positions = []
    bot.exchange = MagicMock()
    bot.exchange.get_positions.return_value = {}
    bot.risk_manager = MagicMock()
    monkeypatch.setattr("core.state.load_open_positions", lambda: [{"symbol": "BTC/USDT"}])

    bot.load_state()

    assert bot.entries_blocked is expected_block


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


@pytest.mark.parametrize("signal,long_only,expected_entries,expected_status", [
    (Signal.LONG, True, ["BTC/USDT"], "entered"),
    (Signal.SHORT, True, [], "skipped:long_only"),
])
def test_process_symbol_fetches_mtf_history_and_applies_long_only(
    monkeypatch, tmp_path, signal, long_only, expected_entries, expected_status,
):
    cfg = BotConfig()
    cfg.strategy.active_strategy = "mtf_trend_pullback"
    cfg.scalping.scan_all_coins = False
    cfg.execution.entry_mode = "market"
    cfg.long_only = long_only

    idx = pd.date_range("2026-01-01", periods=724, freq="5min")
    close = np.linspace(100.0, 110.0, len(idx))
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=idx)

    class FakeExchange:
        def __init__(self):
            self.requested_limits = {}
            self.positions = {}
            self.entries = []

        def fetch_ohlcv(self, symbol, timeframe, limit):
            self.requested_limits[timeframe] = limit
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
    bot.risk_manager.check_stop_loss_risk.return_value = (True, "OK", 0.01)
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
        return TradeSetup(signal, 110.0, 108.0, 114.0, 0.7, "mtf_trend_pullback")

    monkeypatch.setattr(bot.strategy, "analyze", synthetic_signal)
    monkeypatch.setattr(main.state, "save_position_at_entry", lambda *args, **kwargs: None)
    bot.process_symbol("BTC/USDT")

    assert exchange.requested_limits["5m"] >= 722
    assert seen_rows and max(seen_rows) >= 720
    assert exchange.entries == expected_entries
    rows = pd.read_csv(bot.decision_log_path)
    assert rows.loc[0, "status"] == expected_status
    if signal == Signal.SHORT:
        assert bot.risk_manager.register_position.call_count == 0
        return
    meta = bot.paper_trade_meta["BTC/USDT"]
    assert meta["entry_fee"] == pytest.approx(110.0 * 0.1 * cfg.backtest.taker_fee)
    assert meta["entry_slippage"] == pytest.approx(110.0 * 0.1 * cfg.backtest.slippage)
    assert rows.loc[0, "fill_price"] == pytest.approx(110.0)


def test_run_metadata_records_effective_costs_and_context(monkeypatch, tmp_path):
    cfg = BotConfig()
    cfg.exchange.name = "kucoin"
    cfg.scalping.symbols = ["BTC/USDT"]
    cfg.strategy.active_strategy = "ema_rsi_atr"
    cfg.backtest.maker_fee = 0.0002
    cfg.backtest.taker_fee = 0.0005
    cfg.backtest.slippage = 0.0003
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    monkeypatch.chdir(tmp_path)

    bot.write_run_metadata()

    metadata = __import__("json").loads((tmp_path / "logs" / "run_meta.json").read_text())
    assert metadata["exchange"] == "kucoin"
    assert metadata["symbols"] == ["BTC/USDT"]
    assert metadata["strategy"] == "ema_rsi_atr"
    assert metadata["maker_fee"] == 0.0002
    assert metadata["taker_fee"] == 0.0005
    assert metadata["slippage"] == 0.0003
    assert metadata["hold_minutes"] == cfg.risk.max_hold_time_minutes
    assert metadata["git_commit"]
    assert metadata["start_time"]


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
    bot.risk_manager.check_stop_loss_risk.return_value = (True, "OK", 0.01)
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
    assert list(rows.columns) == main.DECISION_FIELDS
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
