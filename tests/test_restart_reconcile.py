import unittest
from unittest.mock import MagicMock, patch
from main import UnifiedCryptoBot
from config.settings import BotConfig


class TestRestartReconcile(unittest.TestCase):
    def setUp(self):
        self.config = BotConfig(mode='paper')
        self.config.scalping.scan_all_coins = False

    def test_paper_mismatch_does_not_block_entries(self):
        with patch('main.GRUPredictor'), patch('core.state.load_open_positions', return_value=[{'symbol':'BTC/USDT'}]):
            bot = UnifiedCryptoBot(config=self.config)
            bot.exchange = MagicMock()
            bot.exchange.get_positions.return_value = {}
            bot.load_state()
            self.assertFalse(bot.entries_blocked)

    def test_match_unblocks(self):
        with patch('main.GRUPredictor'), patch('core.state.load_open_positions', return_value=[{'symbol':'BTC/USDT','side':'long','size':1,'entry_price':100}]), patch.dict('os.environ', {'RECONCILE_CONFIRM':'1'}):
            bot = UnifiedCryptoBot(config=self.config)
            bot.exchange = MagicMock()
            bot.exchange.get_positions.return_value = {'BTC/USDT': {'entryPrice':'100','contracts':'1','side':'long'}}
            bot.load_state()
            self.assertFalse(bot.entries_blocked)
            self.assertIn('BTC/USDT', bot.risk_manager.open_positions)


if __name__ == '__main__':
    unittest.main()
