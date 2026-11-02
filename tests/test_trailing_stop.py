import pytest

from config.settings import BotConfig
from core.risk_manager import RiskManager


def test_short_trailing_exits_near_99_2():
    config = BotConfig()
    rm = RiskManager(config)
    rm.register_position('BTC/USDT', 100.0, 'short')
    rm.update_position_price('BTC/USDT', 99.75)
    triggered, reason = rm.check_trailing_stop('BTC/USDT', 99.75)
    assert triggered is False
    
    rm.update_position_price('BTC/USDT', 99.0)
    triggered, reason = rm.check_trailing_stop('BTC/USDT', 99.0)
    assert triggered is False
    
    rm.update_position_price('BTC/USDT', 99.3)
    triggered, reason = rm.check_trailing_stop('BTC/USDT', 99.3)
    assert triggered is True


def test_breakeven_fee_adjusted_long():
    config = BotConfig()
    rm = RiskManager(config)
    rm.register_position('BTC/USDT', 100.0, 'long')
    rm.open_positions['BTC/USDT']['entry_price'] = 100
    rm.open_positions['BTC/USDT']['high_watermark'] = 100
    rm.open_positions['BTC/USDT']['low_watermark'] = 100
    
    rm.update_position_price('BTC/USDT', 100.2)
    rm.check_trailing_stop('BTC/USDT', 100.2)
    assert rm.open_positions['BTC/USDT']['trailing_stop'] == 100 * (1 + 0.0025)


def test_trailing_tightens():
    config = BotConfig()
    rm = RiskManager(config)
    rm.register_position('BTC/USDT', 100.0, 'long')
    rm.open_positions['BTC/USDT']['entry_price'] = 100
    rm.open_positions['BTC/USDT']['high_watermark'] = 100
    rm.open_positions['BTC/USDT']['low_watermark'] = 100
    
    rm.update_position_price('BTC/USDT', 100.5)
    rm.check_trailing_stop('BTC/USDT', 100.5)
    rm.open_positions['BTC/USDT']['high_watermark'] = 101
    rm.check_trailing_stop('BTC/USDT', 100.5)
    assert rm.open_positions['BTC/USDT']['trailing_stop'] == 101 * (1 - 0.002)
