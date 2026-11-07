from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

import main
from config.settings import BotConfig
from core.risk_manager import RiskManager
from core.strategies import Signal, TradeSetup
import scripts.parity_replay as replay


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_risk_manager_default_clock_remains_wall_clock():
    config = BotConfig()
    risk = RiskManager(config)
    before = datetime.now()
    risk.register_position("BTC/USDT", 100.0, "buy")
    after = datetime.now()

    assert before <= risk.open_positions["BTC/USDT"]["entry_time"] <= after


def test_risk_manager_uses_injected_clock_for_entries_and_trades():
    config = BotConfig()
    fixed = datetime(2020, 1, 2, 3, 4, 5)
    clock = FakeClock(fixed)
    risk = RiskManager(config, clock=clock)

    risk.register_position("BTC/USDT", 100.0, "buy")
    risk.record_trade(1.0, "BTC/USDT")

    assert risk.open_positions["BTC/USDT"]["entry_time"] == fixed
    assert risk.trade_log[-1]["timestamp"] == fixed.isoformat()
    assert risk.last_trade_time == fixed


def test_main_wall_clock_time_exit_uses_injected_clock():
    config = BotConfig()
    config.risk.max_hold_time_minutes = 30
    config.risk.trailing_stop_enabled = False
    now = datetime(2020, 1, 2, 3, 4, 5)
    risk = RiskManager(config, clock=FakeClock(now))
    risk.register_position("BTC/USDT", 100.0, "buy")
    risk.open_positions["BTC/USDT"]["entry_time"] = now - timedelta(minutes=31)

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = config
    bot.clock = FakeClock(now)
    bot.risk_manager = risk
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = {
        "entryPrice": 100.0, "contracts": 1.0, "side": "long",
    }
    bot.exchange.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0, 100.0]})
    bot.position_targets = {"BTC/USDT": {"sl": 90.0, "tp": 120.0}}
    bot.position_entry_cycle = {}
    bot.cycle_count = 10
    bot._close_position = MagicMock()

    bot.check_exits("BTC/USDT")

    bot._close_position.assert_called_once()
    assert bot._close_position.call_args.args[3] == "Time exit (wall-clock)"


def test_replay_runner_returns_the_known_bot_and_backtester_trades(monkeypatch):
    fire = pd.Timestamp("2024-01-01 05:00:00")
    index = pd.date_range("2024-01-01", periods=75, freq="5min")
    close = [100.0] * len(index)
    close[index.get_loc(fire) + 1] = 102.0
    frame = pd.DataFrame({"open": close, "high": close, "low": close,
                          "close": close, "volume": [1.0] * len(index)}, index=index)

    class OneSignal:
        lookback_bars = 30
        name = "synthetic"

        def __init__(self, config):
            pass

        def analyze(self, window):
            if window.index[-1] == fire:
                return TradeSetup(Signal.LONG, 100.0, 90.0, 101.0, 1.0, "synthetic")
            return TradeSetup(Signal.HOLD)

    monkeypatch.setattr(replay, "get_strategy", OneSignal)
    monkeypatch.setattr("core.backtester.get_strategy", OneSignal)
    cfg = replay.make_cfg("synthetic", "BTC/USDT", 10_000, 100.0,
                          (0.0002, 0.0005, 0.0003))
    bot_trades = replay.run_bot(frame, "BTC/USDT", cfg, fire, fire + pd.Timedelta(minutes=15))
    bt_trades, bt = replay.run_backtester(frame, "BTC/USDT", cfg, fire,
                                          fire + pd.Timedelta(minutes=15), 10_000)

    assert len(bot_trades) == 1
    assert bot_trades[0]["side"] == "long"
    assert bot_trades[0]["entry_time"] == (fire + pd.Timedelta(minutes=5)).to_pydatetime()
    assert bot_trades[0]["entry_price"] == 100.0
    assert bot_trades[0]["exit_time"] == (fire + pd.Timedelta(minutes=10)).to_pydatetime()
    assert bot_trades[0]["exit_price"] == 102.0
    assert len(bt_trades) == 1
    assert bt_trades[0].side == "long"
    assert bt_trades[0].entry_time == fire.to_pydatetime()
    assert bt_trades[0].entry_price == 100.0
    assert bt_trades[0].exit_time == (fire + pd.Timedelta(minutes=5)).to_pydatetime()
    assert bt_trades[0].exit_price == 101.0

    matched, only_bot, only_bt, details, _, histogram, exceptions = replay.compare(
        bot_trades, bt_trades, frame, bt)
    assert len(matched) == 1
    assert only_bot == only_bt == exceptions == []
    assert histogram == {"+5 min": 1}
    assert details[0]["entry_price_diff_bps"] == 0.0
    assert details[0]["exit_time_diff_bars"] == 1.0
    assert details[0]["exit_price_diff_bps"] > 0


def test_compare_keeps_nearest_pair_outside_five_minutes_unmatched():
    signal_bar = pd.Timestamp("2024-02-01 00:00:00")
    bot = replay.TradeRun([{
        "side": "long", "entry_time": (signal_bar + pd.Timedelta(minutes=10)).to_pydatetime(),
        "signal_bar_open": signal_bar + pd.Timedelta(minutes=5),
    }])
    bt_trade = type("Trade", (), {"side": "long", "entry_time": signal_bar})()
    frame = pd.DataFrame({"open": [100], "high": [101], "low": [99],
                          "close": [100]}, index=[signal_bar])

    matched, only_bot, only_bt, _, mismatches, histogram, exceptions = replay.compare(
        bot, [bt_trade], frame)

    assert matched == []
    assert len(only_bot) == len(only_bt) == 1
    assert histogram == {"+10 min": 1}
    assert len(exceptions) == 1
    assert exceptions[0][2] == 10.0
    assert any(row["cause"] == "entry timing offset" for row in mismatches)
