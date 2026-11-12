from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from main import UnifiedCryptoBot


def test_time_exit_uses_signal_close_and_includes_exact_limit():
    bot = UnifiedCryptoBot.__new__(UnifiedCryptoBot)
    bot.config = SimpleNamespace(
        mode="paper", scalping=SimpleNamespace(timeframe="5m", stop_loss_pct=.05,
                                               take_profit_pct=.10, max_hold_cycles=60),
        risk=SimpleNamespace(max_hold_time_minutes=30),
        backtest=SimpleNamespace(slippage=.001, taker_fee=.0005))
    bot.clock = lambda: datetime(2026, 1, 1, 0, 35)
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = {"entryPrice": 100, "contracts": 1, "side": "long"}
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="5min")
    bot.exchange.fetch_ohlcv.return_value = pd.DataFrame(
        {"open": [100] * 3, "high": [104] * 3, "low": [96] * 3, "close": [100] * 3}, index=idx)
    bot.position_targets = {"X": {"sl": 95, "tp": 110, "signal_bar_open": "2026-01-01T00:00:00"}}
    bot.risk_manager = MagicMock()
    bot.risk_manager.open_positions = {"X": {"entry_time": datetime(2026, 1, 1, 0, 5)}}
    bot.risk_manager.check_trailing_stop.return_value = (False, "")
    bot._close_position = MagicMock()
    bot.check_exits("X")
    assert bot._close_position.call_args.args[3] == "Time exit (wall-clock)"
