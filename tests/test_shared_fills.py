from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import main
from config.settings import BotConfig
from core.backtester import BacktestTrade, Backtester
from core.fills import FILL_MODEL_VERSION, exit_fee_rate, stop_fill, tp_fill
from core.risk_manager import RiskManager
from core.strategies import Signal, TradeSetup


@pytest.mark.parametrize("side,level,reference,slip,expected", [
    ("long", 95.0, 90.0, 0.001, 90.0 * 0.999),
    ("short", 105.0, 110.0, 0.001, 110.0 * 1.001),
    ("long", 95.0, 100.0, 0.001, 95.0 * 0.999),
    ("short", 105.0, 100.0, 0.001, 105.0 * 1.001),
])
def test_stop_fill_gap_and_no_gap_for_both_sides(side, level, reference, slip, expected):
    assert stop_fill(side, level, reference, slip) == pytest.approx(expected)


@pytest.mark.parametrize("side,tp,high,low,expected", [
    ("long", 110.0, 110.0, 109.0, None),
    ("short", 90.0, 91.0, 90.0, None),
    ("long", 110.0, 110.01, 109.0, 110.0),
    ("short", 90.0, 91.0, 89.99, 90.0),
])
def test_tp_fill_requires_strict_through_and_never_slips(side, tp, high, low, expected):
    assert tp_fill(side, tp, high, low) == expected


def test_tp_fee_role_switches_fee_and_stops_stay_taker():
    costs = SimpleNamespace(maker_fee=0.0002, taker_fee=0.0006, tp_fee_role="maker")
    assert exit_fee_rate(costs, "TP") == pytest.approx(0.0002)
    assert exit_fee_rate(costs, "TAKE PROFIT") == pytest.approx(0.0002)
    assert exit_fee_rate(costs, "SL") == pytest.approx(0.0006)
    costs.tp_fee_role = "taker"
    assert exit_fee_rate(costs, "TP") == pytest.approx(0.0006)


@pytest.mark.parametrize("side,signal,stop,target", [
    ("long", Signal.LONG, 95.0, 105.0),
    ("short", Signal.SHORT, 105.0, 95.0),
])
def test_backtester_same_bar_sl_and_tp_chooses_stop(side, signal, stop, target):
    cfg = BotConfig()
    cfg.backtest.slippage = 0.0
    bt = Backtester(cfg)
    cfg.risk.max_daily_loss_pct = 1.0
    cfg.risk.max_drawdown_pct = 1.0
    cfg.risk.circuit_breaker_enabled = False

    class OneSignal:
        lookback_bars = 1
        fired = False
        def analyze(self, _window):
            if self.fired:
                return TradeSetup(Signal.HOLD)
            self.fired = True
            return TradeSetup(signal, 100.0, stop, target, 1.0, "test")

    bt.strategy = OneSignal()
    index = pd.date_range("2026-01-01", periods=200, freq="5min")
    values = {key: [100.0] * len(index) for key in ("open", "high", "low", "close")}
    values["high"][61], values["low"][61] = 106.0, 94.0
    values["volume"] = [1000.0] * len(index)
    bt.run(pd.DataFrame(values, index=index), "X/USDT")
    assert len(bt.trades) == 1
    assert bt.trades[0].exit_reason == "SL"


@pytest.mark.parametrize("tp_fee_role,expected_rate", [
    ("maker", 0.0002), ("taker", 0.0006),
])
def test_backtester_take_profit_fee_uses_configured_role(tp_fee_role, expected_rate):
    cfg = BotConfig()
    cfg.backtest.maker_fee = 0.0002
    cfg.backtest.taker_fee = 0.0006
    cfg.backtest.tp_fee_role = tp_fee_role
    bt = Backtester(cfg)
    trade = BacktestTrade("X", "long", 100.0, 1.0, datetime(2026, 1, 1),
                          95.0, 110.0, "test")
    bt.open_trades["X"] = trade
    bt._close("X", 110.0, datetime(2026, 1, 1, 0, 5), "TP",
              reference_price=110.0, exit_slippage_rate=0.0)
    assert trade.exit_fee == pytest.approx(110.0 * expected_rate)


def _synthetic_bars(side, stop, target):
    index = pd.date_range("2026-01-01 00:00", periods=200, freq="5min")
    values = {key: [100.0] * len(index) for key in ("open", "high", "low", "close")}
    if side == "long":
        values["open"][61], values["high"][61] = 90.0, 91.0
        values["low"][61], values["close"][61] = 89.0, 90.0
    else:
        values["open"][61], values["high"][61] = 110.0, 111.0
        values["low"][61], values["close"][61] = 109.0, 110.0
    values["volume"] = [1000.0] * len(index)
    return pd.DataFrame(values, index=index)


@pytest.mark.parametrize("side,signal,stop,target", [
    ("long", Signal.LONG, 95.0, 110.0),
    ("short", Signal.SHORT, 105.0, 90.0),
])
def test_bot_and_backtester_share_stop_gap_fill_fee_and_pnl(
    side, signal, stop, target, monkeypatch
):
    cfg = BotConfig()
    cfg.strategy.active_strategy = "ema_rsi_atr"
    cfg.execution.entry_mode = "market"
    cfg.backtest.taker_fee = 0.0006
    cfg.backtest.maker_fee = 0.0002
    cfg.backtest.slippage = 0.001
    cfg.risk.trailing_stop_enabled = False
    cfg.risk.max_daily_loss_pct = 1.0
    cfg.risk.max_drawdown_pct = 1.0
    cfg.risk.circuit_breaker_enabled = False
    cfg.risk.reduce_size_on_loss = False
    bars = _synthetic_bars(side, stop, target)

    class OneSignal:
        lookback_bars = 1
        fired = False
        def analyze(self, _window):
            if self.fired:
                return TradeSetup(Signal.HOLD)
            self.fired = True
            return TradeSetup(signal, 100.0, stop, target, 1.0, "synthetic")

    backtester = Backtester(cfg)
    backtester.strategy = OneSignal()
    backtester.run(bars.copy(), "X/USDT")
    trade = next(item for item in backtester.trades if item.exit_reason == "SL")
    expected_exit_fee = trade.exit_reference_price * trade.size * cfg.backtest.taker_fee
    expected_slippage = abs(trade.exit_price - trade.exit_reference_price) * trade.size

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.clock = lambda: bars.index[-1].to_pydatetime()
    bot._clock_uses_local_timezone = False
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = {
        "entryPrice": 100.0, "contracts": trade.size, "side": side,
    }
    bot.exchange.fetch_ohlcv.return_value = bars
    bot.risk_manager = RiskManager(cfg, clock=bot.clock)
    bot.risk_manager.register_position("X/USDT", 100.0 * trade.size, side,
                                       quantity=trade.size, stop_price=stop)
    bot.risk_manager.update_position_price("X/USDT", 100.0)
    bot.position_targets = {"X/USDT": {
        "sl": stop, "tp": target,
        "signal_bar_open": bars.index[60].isoformat(),
        "timeframe": "5m", "strategy": "synthetic@5m",
    }}
    bot.position_entry_cycle = {"X/USDT": 0}
    bot.cycle_count = 1
    bot.active_decisions = {}
    bot.paper_trade_meta = {"X/USDT": {
        "entry_fee": trade.entry_fee, "entry_slippage": trade.entry_slip,
        "selected_leverage": 1.0,
    }}
    bot._prepare_poll_close = MagicMock()
    bot.persist_state = MagicMock()
    monkeypatch.setattr(main, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.state, "record_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.state, "close_position", lambda *args, **kwargs: None)

    bot.check_exits("X/USDT")

    assert bot.exchange.close_position.called
    actual_exit = bot.exchange.close_position.call_args.args[1]
    assert actual_exit == pytest.approx(trade.exit_price)
    exit_fee = expected_exit_fee
    entry_fee = trade.entry_fee
    entry_slip = trade.entry_slip
    pnl = bot.risk_manager.trade_log[-1]["pnl"]
    assert exit_fee == pytest.approx(trade.exit_fee)
    assert abs(actual_exit - trade.exit_reference_price) * trade.size == pytest.approx(expected_slippage)
    assert pnl == pytest.approx(trade.pnl)


@pytest.mark.parametrize("side,entry,trail,polled", [
    ("long", 100.0, 99.0, 98.0),
    ("short", 100.0, 101.0, 102.0),
])
def test_bot_trailing_poll_exit_uses_shared_stop_fill(side, entry, trail, polled):
    index = pd.date_range("2026-01-01 00:00", periods=5, freq="5min")
    if side == "long":
        high, low, target = [entry] * 5, [entry] * 4 + [polled], 120.0
    else:
        high, low, target = [entry] * 4 + [polled], [entry] * 5, 80.0
    frame = pd.DataFrame({"open": [entry] * 4 + [polled], "high": high, "low": low,
                          "close": [entry] * 4 + [polled], "volume": [1.0] * 5}, index=index)
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = BotConfig()
    bot.config.mode = "paper"
    bot.config.scalping.timeframe = "5m"
    bot.clock = lambda: index[-1].to_pydatetime()
    bot._clock_uses_local_timezone = False
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = {"entryPrice": entry, "contracts": 1.0, "side": side}
    bot.exchange.fetch_ohlcv.return_value = frame
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {"X": {
        "entry_price": entry, "side": side, "quantity": 1.0,
        "original_stop": 90.0 if side == "long" else 110.0,
        "trailing_stop": trail,
    }}
    bot.risk_manager.check_trailing_stop.return_value = (True, "trailing hit")
    bot.position_targets = {"X": {"sl": 90.0 if side == "long" else 110.0,
                                   "tp": target,
                                   "signal_bar_open": index[0].isoformat()}}
    bot.position_entry_cycle = {}
    bot.cycle_count = 0
    bot._prepare_poll_close = MagicMock()
    bot._close_position = MagicMock()

    bot.check_exits("X")

    args, kwargs = bot._close_position.call_args
    assert args[3].startswith("TRAILING STOP")
    assert args[2] == pytest.approx(
        stop_fill(side, trail, polled, bot.config.backtest.slippage))
    assert kwargs["reference_price"] == pytest.approx(polled)
    assert kwargs["exit_slippage_rate"] == pytest.approx(bot.config.backtest.slippage)


def test_every_new_evolve_ledger_entry_gets_fill_model_version(tmp_path):
    from evolve.ledger import append_entry, read_entries

    path = tmp_path / "ledger.jsonl"
    row = append_entry({"event": "holdout_attempt", "name": "trial"}, str(path))
    assert row["fill_model_version"] == FILL_MODEL_VERSION
    assert read_entries(str(path))[0]["fill_model_version"] == FILL_MODEL_VERSION
