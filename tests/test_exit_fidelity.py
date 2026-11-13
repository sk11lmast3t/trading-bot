from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from main import UnifiedCryptoBot


def bot_for_exit(frame, *, now=None, sl=95, tp=110, entry=100, side="long"):
    bot = UnifiedCryptoBot.__new__(UnifiedCryptoBot)
    bot.config = SimpleNamespace(
        mode="paper", scalping=SimpleNamespace(timeframe="5m", stop_loss_pct=.05,
                                               take_profit_pct=.10, max_hold_cycles=60),
        risk=SimpleNamespace(max_hold_time_minutes=60))
    bot.config.backtest = SimpleNamespace(slippage=.001, taker_fee=.0005)
    bot.clock = lambda: now or datetime(2026, 1, 1, 0, 20)
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = {"entryPrice": entry, "contracts": 1,
                                              "side": side}
    bot.exchange.fetch_ohlcv.return_value = frame
    bot.position_targets = {"X": {"sl": sl, "tp": tp,
                                   "signal_bar_open": "2026-01-01T00:00:00"}}
    bot.position_entry_cycle = {"X": 0}
    bot.cycle_count = 1
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {}
    bot.active_decisions = {}
    bot._close_position = MagicMock()
    return bot


def test_touch_sl_spike_closes_at_level_with_slippage():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="5min")
    frame = pd.DataFrame({"open": [100, 100, 100, 100], "high": [101, 105, 105, 105],
                          "low": [99, 94, 99, 99], "close": [100, 102, 103, 104]}, index=idx)
    bot = bot_for_exit(frame)
    bot.check_exits("X")
    bot._close_position.assert_called_once()
    args, kwargs = bot._close_position.call_args
    assert args[3] == "STOP LOSS"
    assert args[2] == 95 * (1 - .001)
    assert kwargs["reference_price"] == 95


def test_same_candle_sl_and_tp_prefers_sl():
    idx = pd.date_range("2026-01-01 00:00", periods=2, freq="5min")
    frame = pd.DataFrame({"open": [100, 100], "high": [101, 112],
                          "low": [99, 94], "close": [100, 105]}, index=idx)
    bot = bot_for_exit(frame)
    bot.check_exits("X")
    assert bot._close_position.call_args.args[3] == "STOP LOSS"


def test_signal_bar_and_earlier_candles_are_ignored():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="5min")
    frame = pd.DataFrame({"open": [100, 100, 100], "high": [111, 105, 105],
                          "low": [90, 99, 99], "close": [100, 104, 104]}, index=idx)
    bot = bot_for_exit(frame)
    bot.check_exits("X")
    bot._close_position.assert_not_called()


def test_untouched_levels_do_not_close_position():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="5min")
    frame = pd.DataFrame({"open": [100] * 3, "high": [104] * 3,
                          "low": [96] * 3, "close": [100] * 3}, index=idx)
    bot = bot_for_exit(frame)
    bot.risk_manager.check_trailing_stop.return_value = (False, "")
    bot.check_exits("X")
    bot._close_position.assert_not_called()


def test_long_take_profit_fills_at_limit_without_adverse_slippage():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="5min")
    frame = pd.DataFrame({"open": [100, 108, 108], "high": [101, 112, 112],
                          "low": [99, 107, 107], "close": [100, 111, 111]}, index=idx)
    bot = bot_for_exit(frame)
    bot.check_exits("X")
    args, kwargs = bot._close_position.call_args
    assert args[3] == "TAKE PROFIT"
    assert args[2] == 110  # resting limit, never improved by a favourable gap
    assert kwargs["reference_price"] == 110
    assert kwargs["exit_slippage_rate"] == 0.0


def test_short_take_profit_is_not_slipped_through_the_target():
    # Regression for the ETH short: entry 2015.15, TP 2013.4702, reported exit
    # 2014.074195 because adverse slippage was applied to a resting limit.
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="5min")
    frame = pd.DataFrame({"open": [2015, 2014, 2014], "high": [2016, 2016, 2016],
                          "low": [2014, 2012.5, 2012.5], "close": [2015, 2013.2, 2013.2]},
                         index=idx)
    bot = bot_for_exit(frame, entry=2015.15, sl=2020.0, tp=2013.4702, side="short")
    bot.check_exits("X")
    args, kwargs = bot._close_position.call_args
    assert args[3] == "TAKE PROFIT"
    assert args[2] == pytest.approx(2013.4702)
    assert kwargs["reference_price"] == pytest.approx(2013.4702)
    assert kwargs["exit_slippage_rate"] == 0.0


def test_stop_gap_fills_at_first_available_price_then_applies_slippage():
    idx = pd.date_range("2026-01-01 00:00", periods=2, freq="5min")
    frame = pd.DataFrame({"open": [100, 90], "high": [101, 91],
                          "low": [99, 89], "close": [100, 90]}, index=idx)
    bot = bot_for_exit(frame)
    bot.check_exits("X")
    args, kwargs = bot._close_position.call_args
    assert args[3] == "STOP LOSS"
    assert kwargs["reference_price"] == 90  # gap, not the 95 trigger
    assert args[2] == pytest.approx(90 * (1 - .001))


