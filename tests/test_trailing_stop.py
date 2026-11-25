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
    assert rm.open_positions['BTC/USDT']['breakeven_armed'] is False

    rm.update_position_price('BTC/USDT', 100.3)
    rm.check_trailing_stop('BTC/USDT', 100.3)
    assert rm.open_positions['BTC/USDT']['breakeven_armed'] is True
    assert rm.open_positions['BTC/USDT']['trailing_stop'] == 100 * (1 + 0.0025)


def test_breakeven_covers_configured_fees_slippage_and_buffer_for_both_sides():
    config = BotConfig()
    config.backtest.taker_fee = 0.001
    config.backtest.slippage = 0.0003
    config.execution.cost_safety_buffer_pct = 0.0002
    rm = RiskManager(config)

    long_stop = rm.breakeven_stop_price(100.0, 'long')
    short_stop = rm.breakeven_stop_price(100.0, 'short')

    assert long_stop == pytest.approx(100.0 * (1.0015 / 0.9987))
    assert short_stop == pytest.approx(100.0 * (0.9985 / 1.0013))

    rm.register_position('LONG/USDT', 100.0, 'long')
    rm.open_positions['LONG/USDT']['entry_price'] = 100.0
    rm.update_position_price('LONG/USDT', 100.25)
    rm.check_trailing_stop('LONG/USDT', 100.25)
    assert rm.open_positions['LONG/USDT']['breakeven_armed'] is False
    rm.update_position_price('LONG/USDT', 100.35)
    rm.check_trailing_stop('LONG/USDT', 100.35)
    assert rm.open_positions['LONG/USDT']['trailing_stop'] == pytest.approx(long_stop)

    rm.register_position('SHORT/USDT', 100.0, 'short')
    rm.open_positions['SHORT/USDT']['entry_price'] = 100.0
    rm.update_position_price('SHORT/USDT', 99.75)
    rm.check_trailing_stop('SHORT/USDT', 99.75)
    assert rm.open_positions['SHORT/USDT']['breakeven_armed'] is False
    rm.update_position_price('SHORT/USDT', 99.65)
    rm.check_trailing_stop('SHORT/USDT', 99.65)
    assert rm.open_positions['SHORT/USDT']['trailing_stop'] == pytest.approx(short_stop)


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
