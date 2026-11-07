from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

import main
from config.settings import BotConfig
from core.risk_manager import RiskManager


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
