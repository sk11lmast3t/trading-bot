import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch
from core.exchange import OrderState, OrderRecord
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


def make_bot_live():
    cfg = BotConfig()
    cfg.mode = 'live'
    cfg.scalping.symbols = ['BTC/USDT']
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
    bot.exchange.set_leverage = MagicMock()
    bot.cycle_count = 0
    bot.position_entry_cycle = {}
    bot.position_targets = {}
    bot.last_rejection_log = {}
    bot.gru_predictor = MagicMock()
    bot.gru_predictor.model = None
    bot.strategy = MagicMock()
    bot.strategy.analyze.return_value = DummySetup('LONG')
    return bot


def test_entry_filled_stop_placed(monkeypatch):
    bot = make_bot_live()
    df_mock = MagicMock()
    df_mock.__len__.return_value = 100
    df_mock.iloc = MagicMock()
    df_mock.iloc.__getitem__ = lambda self, idx: MagicMock()
    # Simplify: patch fetch_ohlcv/indicators
    bot.exchange.fetch_ohlcv = MagicMock(return_value=MagicMock())
    import pandas as pd
    dates = pd.date_range('2023-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [100]*50, 'high': [101]*50, 'low': [99]*50, 'close': [100]*50, 'volume': [1000]*50
    }, index=dates)
    bot.exchange.fetch_ohlcv.return_value = df
    from core import indicators
    from core.indicators import Indicators
    monkeypatch.setattr(Indicators, "calculate_all", MagicMock(return_value=df))
    entry = OrderRecord(id='1', symbol='BTC/USDT', side='buy', type='market', amount=0.01, price=None)
    entry.state = OrderState.FILLED
    sl = OrderRecord(id='2', symbol='BTC/USDT', side='sell', type='stop_market', amount=0.01, price=None)
    sl.state = OrderState.OPEN
    bot.exchange.create_market_order = MagicMock(return_value=entry)
    bot.exchange.create_stop_loss = MagicMock(return_value=sl)
    bot.process_symbol('BTC/USDT')
    bot.risk_manager.register_position.assert_called_once()
    bot.exchange.create_stop_loss.assert_called_once()


def test_stop_rejected_flattens(monkeypatch):
    bot = make_bot_live()
    import pandas as pd
    dates = pd.date_range('2023-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [100]*50, 'high': [101]*50, 'low': [99]*50, 'close': [100]*50, 'volume': [1000]*50
    }, index=dates)
    bot.exchange.fetch_ohlcv = MagicMock(return_value=df)
    from core.indicators import Indicators
    monkeypatch.setattr(Indicators, "calculate_all", MagicMock(return_value=df))
    entry = OrderRecord(id='1', symbol='BTC/USDT', side='buy', type='market', amount=0.01, price=None)
    entry.state = OrderState.FILLED
    sl = OrderRecord(id='2', symbol='BTC/USDT', side='sell', type='stop_market', amount=0.01, price=None)
    sl.state = OrderState.REJECTED
    bot.exchange.create_market_order = MagicMock(return_value=entry)
    bot.exchange.create_stop_loss = MagicMock(return_value=sl)
    bot.exchange.close_position = MagicMock()
    bot.exchange.cancel_all_orders = MagicMock()
    bot.process_symbol('BTC/USDT')
    bot.risk_manager.register_position.assert_not_called()
    bot.exchange.close_position.assert_called()


def test_entry_none_or_rejected_nothing_registered(monkeypatch):
    bot = make_bot_live()
    import pandas as pd
    dates = pd.date_range('2023-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [100]*50, 'high': [101]*50, 'low': [99]*50, 'close': [100]*50, 'volume': [1000]*50
    }, index=dates)
    bot.exchange.fetch_ohlcv = MagicMock(return_value=df)
    from core.indicators import Indicators
    monkeypatch.setattr(Indicators, "calculate_all", MagicMock(return_value=df))
    bot.exchange.create_market_order = MagicMock(return_value=None)
    bot.process_symbol('BTC/USDT')
    bot.risk_manager.register_position.assert_not_called()
