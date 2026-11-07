import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from main import UnifiedCryptoBot
from config.settings import BotConfig


class TestMainExits(unittest.TestCase):
    def setUp(self):
        self.config = BotConfig(mode='paper')
        self.config.scalping.scan_all_coins = False
        self.bot = UnifiedCryptoBot(config=self.config)
        self.bot.exchange = MagicMock()

    def test_sl_hit_closes_position(self):
        self.bot.exchange.get_position.return_value = {
            'entryPrice': '100', 
            'contracts': '1', 
            'side': 'long'
        }
        df = pd.DataFrame({'close': [95, 90]})
        self.bot.exchange.fetch_ohlcv.return_value = df
        self.bot.position_targets['BTC/USDT'] = {'sl': 95, 'tp': 110}
        self.bot.risk_manager.record_trade = MagicMock()
        self.bot.risk_manager.release_position = MagicMock()
        self.bot.exchange.close_position = MagicMock()
        self.bot.check_exits('BTC/USDT')
        self.bot.exchange.close_position.assert_called()
        self.bot.risk_manager.record_trade.assert_called()
        args, kwargs = self.bot.risk_manager.record_trade.call_args
        pnl = args[0]
        self.assertLess(pnl, 0)

    def test_tp_hit(self):
        self.bot.exchange.get_position.return_value = {
            'entryPrice': '100', 
            'contracts': '1', 
            'side': 'long'
        }
        df = pd.DataFrame({'close': [105, 110]})
        self.bot.exchange.fetch_ohlcv.return_value = df
        self.bot.position_targets['BTC/USDT'] = {'sl': 95, 'tp': 105}
        self.bot.risk_manager.record_trade = MagicMock()
        self.bot.risk_manager.release_position = MagicMock()
        self.bot.exchange.close_position = MagicMock()
        self.bot.check_exits('BTC/USDT')
        self.bot.exchange.close_position.assert_called()
        self.bot.risk_manager.record_trade.assert_called()
        args, kwargs = self.bot.risk_manager.record_trade.call_args
        pnl = args[0]
        self.assertGreater(pnl, 0)

    def test_trailing_hit(self):
        self.bot.exchange.get_position.return_value = {
            'entryPrice': '100', 
            'contracts': '1', 
            'side': 'long'
        }
        df = pd.DataFrame({'close': [100, 100]})
        self.bot.exchange.fetch_ohlcv.return_value = df
        self.bot.position_targets['BTC/USDT'] = {'sl': 90, 'tp': 120}
        self.bot.risk_manager.check_trailing_stop = MagicMock(return_value=(True, 'trail'))
        self.bot.risk_manager.record_trade = MagicMock()
        self.bot.risk_manager.release_position = MagicMock()
        self.bot.exchange.close_position = MagicMock()
        self.bot.check_exits('BTC/USDT')
        self.bot.exchange.close_position.assert_called()

    def test_time_exit(self):
        self.bot.exchange.get_position.return_value = {
            'entryPrice': '100', 
            'contracts': '1', 
            'side': 'long'
        }
        df = pd.DataFrame({'close': [100, 100]})
        self.bot.exchange.fetch_ohlcv.return_value = df
        self.bot.position_targets['BTC/USDT'] = {'sl': 90, 'tp': 120}
        self.bot.position_entry_cycle['BTC/USDT'] = 0
        self.bot.cycle_count = 100
        self.bot.config.scalping.max_hold_cycles = 60
        self.bot.risk_manager.check_trailing_stop = MagicMock(return_value=(False, ''))
        self.bot.risk_manager.record_trade = MagicMock()
        self.bot.risk_manager.release_position = MagicMock()
        self.bot.exchange.close_position = MagicMock()
        self.bot.check_exits('BTC/USDT')
        self.bot.exchange.close_position.assert_called()


if __name__ == '__main__':
    unittest.main()
