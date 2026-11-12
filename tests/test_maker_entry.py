import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch
import pandas as pd

from config.settings import BotConfig
from main import UnifiedCryptoBot


class DummySetup:
    def __init__(self, signal='LONG'):
        from core.strategies import Signal
        self.signal = Signal.LONG if signal == 'LONG' else Signal.SHORT
        self.entry_price = 100.0
        self.stop_loss = 99.0
        self.take_profit = 101.0
        self.strategy_name = 'test'
        self.confidence = 0.9


def make_bot(entry_mode='maker'):
    cfg = BotConfig()
    cfg.mode = 'paper'
    cfg.scalping.symbols = ['BTC/USDT']
    cfg.scalping.scan_all_coins = False
    cfg.execution.entry_mode = entry_mode
    cfg.risk.max_open_positions = 10
    bot = UnifiedCryptoBot.__new__(UnifiedCryptoBot)
    bot.config = cfg
    bot.risk_manager = MagicMock()
    bot.risk_manager.can_trade.return_value = (True, 'OK')
    bot.risk_manager.current_balance = 10000
    bot.risk_manager.open_positions = {}
    bot.risk_manager.calculate_position_size.return_value = 0.01
    bot.risk_manager.check_portfolio_heat.return_value = (True, 'OK')
    bot.risk_manager.register_position = MagicMock()
    bot.risk_manager.update_position_price = MagicMock()
    bot.exchange = MagicMock()
    bot.exchange.get_position.return_value = None
    bot.entries_blocked = False
    bot.cycle_count = 0
    bot.position_entry_cycle = {}
    bot.position_targets = {}
    bot.last_rejection_log = {}
    bot.gru_predictor = MagicMock()
    bot.gru_predictor.model = None
    bot.strategy = MagicMock()
    bot.strategy.analyze.return_value = DummySetup('LONG')
    return bot


def test_maker_entry_calls_create_limit_order(monkeypatch):
    bot = make_bot(entry_mode='maker')
    dates = pd.date_range('2023-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [100]*50, 'high': [101]*50, 'low': [99]*50, 'close': [100]*50, 'volume': [1000]*50
    }, index=dates)
    bot.exchange.fetch_ohlcv.return_value = df
    from core.indicators import Indicators
    monkeypatch.setattr(Indicators, "calculate_all", MagicMock(return_value=df))

    maker_order = {'id': 'paper_1', 'symbol': 'BTC/USDT', 'side': 'buy',
                   'amount': 0.01, 'price': 100.0, 'status': 'filled',
                   'filled': 0.01, 'timestamp': 'now'}
    bot.exchange.create_best_bid_ask_limit_order = MagicMock(return_value=maker_order)
    bot.exchange.check_limit_fill = MagicMock(return_value=maker_order)
    bot.exchange.cancel_all_orders = MagicMock()
    bot.risk_manager.register_position = MagicMock()

    bot.process_symbol('BTC/USDT')

    bot.exchange.create_best_bid_ask_limit_order.assert_called_once()
    bot.exchange.check_limit_fill.assert_called()
    bot.risk_manager.register_position.assert_called_once()


def test_market_entry_uses_create_market_order(monkeypatch):
    bot = make_bot(entry_mode='market')
    dates = pd.date_range('2023-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [100]*50, 'high': [101]*50, 'low': [99]*50, 'close': [100]*50, 'volume': [1000]*50
    }, index=dates)
    bot.exchange.fetch_ohlcv.return_value = df
    from core.indicators import Indicators
    monkeypatch.setattr(Indicators, "calculate_all", MagicMock(return_value=df))

    market_order = {'id': 'paper_1', 'symbol': 'BTC/USDT', 'side': 'buy',
                    'amount': 0.01, 'price': 100.0, 'status': 'closed',
                    'filled': 0.01, 'timestamp': 'now'}
    bot.exchange.create_market_order = MagicMock(return_value=market_order)
    bot.exchange.create_best_bid_ask_limit_order = MagicMock()
    bot.risk_manager.register_position = MagicMock()

    bot.process_symbol('BTC/USDT')

    bot.exchange.create_market_order.assert_called_once()
    bot.exchange.create_best_bid_ask_limit_order.assert_not_called()
    bot.risk_manager.register_position.assert_called_once()
