from config.settings import BotConfig
from core.risk_manager import RiskManager
config = BotConfig()
rm = RiskManager(config)
rm.register_position('BTC/USDT', 100.0, 'short')
rm.update_position_price('BTC/USDT', 100)
t,r = rm.check_trailing_stop('BTC/USDT', 99.75)
print(t,r, rm.open_positions['BTC/USDT'].get('trailing_stop'), rm.open_positions['BTC/USDT'].get('breakeven_armed'))