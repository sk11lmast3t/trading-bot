import pytest

from config.settings import BotConfig, ScalpingConfig
from core.risk_manager import RiskManager, normalize_side


def test_normalize_side_mappings():
    assert normalize_side('buy') == 'long'
    assert normalize_side('BUY') == 'long'
    assert normalize_side('long') == 'long'
    assert normalize_side('sell') == 'short'
    assert normalize_side('SELL') == 'short'
    assert normalize_side('short') == 'short'


def test_normalize_side_unknown():
    with pytest.raises(ValueError):
        normalize_side('foo')


def test_register_position_stores_normalized_side():
    config = BotConfig(scalping=ScalpingConfig(initial_capital=1000))
    risk_manager = RiskManager(config)
    risk_manager.register_position('BTC/USDT', 100.0, 'BUY')
    assert risk_manager.open_positions['BTC/USDT']['side'] == 'long'

    risk_manager.register_position('ETH/USDT', 50.0, 'sell')
    assert risk_manager.open_positions['ETH/USDT']['side'] == 'short'
