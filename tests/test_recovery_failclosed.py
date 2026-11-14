"""Restart recovery must fail closed.

Covers: a crash during reconcile blocks entries, RECONCILE_CONFIRM cannot clear
integrity blocks, a failed live exchange query is not treated as "no positions",
and the leverage cap is applied once (paper clamps, live keeps the real value).
No network or live exchange is used.
"""
from unittest.mock import MagicMock, patch

import pytest

import main
from config.settings import BotConfig

ROW = {'symbol': 'BTC/USDT', 'side': 'long', 'size': 1, 'entry_price': 100, 'sl': 99}
EXCHANGE_POS = {'BTC/USDT': {'entryPrice': '100', 'contracts': '1', 'side': 'long'}}


def _bot(tmp_path, monkeypatch, mode="paper"):
    monkeypatch.setattr(main.state, "DB_PATH", tmp_path / "paper.db")
    cfg = BotConfig(mode="paper")
    cfg.scalping.scan_all_coins = False
    bot = main.UnifiedCryptoBot(config=cfg)
    bot.config.mode = mode
    bot._state_path = lambda: str(tmp_path / "missing_snapshot.json")
    bot.exchange = MagicMock()
    bot.exchange.get_positions.return_value = dict(EXCHANGE_POS)
    return bot


def test_reconcile_crash_blocks_entries(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch)
    with patch('core.state.load_open_positions', return_value=[dict(ROW)]):
        bot._merge_restored_dynamic_state = MagicMock(side_effect=RuntimeError("boom"))
        bot.load_state()
    assert bot.entries_blocked is True


def test_confirm_does_not_clear_missing_stop_block(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch)
    row = dict(ROW)
    row.pop('sl')
    with patch('core.state.load_open_positions', return_value=[row]), \
            patch.dict('os.environ', {'RECONCILE_CONFIRM': '1'}):
        bot.load_state()
    assert bot.entries_blocked is True


def test_confirm_clears_plain_state_when_clean(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch)
    with patch('core.state.load_open_positions', return_value=[dict(ROW)]), \
            patch.dict('os.environ', {'RECONCILE_CONFIRM': '1'}):
        bot.load_state()
    assert bot.entries_blocked is False
    assert 'BTC/USDT' in bot.risk_manager.open_positions


def test_confirm_does_not_clear_invalid_side_block(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch)
    row = dict(ROW, side='sideways')
    with patch('core.state.load_open_positions', return_value=[row]), \
            patch.dict('os.environ', {'RECONCILE_CONFIRM': '1'}):
        bot.load_state()
    assert bot.entries_blocked is True


def test_live_exchange_query_failure_blocks_entries(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch, mode="live")
    bot.exchange.get_positions.side_effect = ConnectionError("exchange down")
    bot.exchange.get_position.side_effect = ConnectionError("exchange down")
    with patch('core.state.load_open_positions', return_value=[]), \
            patch.dict('os.environ', {'RECONCILE_CONFIRM': '1'}):
        bot.load_state()
    assert bot.entries_blocked is True


def test_paper_leverage_is_clamped_to_cap(tmp_path, monkeypatch):
    bot = _bot(tmp_path, monkeypatch)
    row = dict(ROW, leverage=5.0)
    with patch('core.state.load_open_positions', return_value=[row]):
        bot.load_state()
    assert bot.risk_manager.open_positions['BTC/USDT']['leverage'] == main.MAX_SAFE_LEVERAGE


@pytest.mark.parametrize("mode,raw,expected", [
    ("paper", 5.0, 3.0),
    ("paper", 2.0, 2.0),
    ("paper", None, 1.0),
    ("live", 5.0, 5.0),
    ("live", 2.0, 2.0),
])
def test_restored_leverage_helper(tmp_path, monkeypatch, mode, raw, expected):
    bot = _bot(tmp_path, monkeypatch, mode=mode)
    assert bot._restored_leverage('BTC/USDT', raw) == expected
