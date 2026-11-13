"""Paper restart recovery: dynamic risk state must survive a process restart.

Every test uses an isolated SQLite database and JSON snapshot under tmp_path.
SQLite owns position identity; the JSON snapshot only restores dynamic risk
fields. No live exchange or network call is made.
"""
from datetime import datetime
import json

import pandas as pd
import pytest
from unittest.mock import MagicMock

import main
from config.settings import BotConfig
from core.exchange import PaperExchange
from core.risk_manager import RiskManager

SYMBOL = "SOL/USDT"
ENTRY = 81.84
STOP = 82.22424457666315
TRAILING = 81.6354
QUANTITY = 0.06109481915933528
MARK = 81.49
TP = 80.0


def _cfg():
    cfg = BotConfig()
    cfg.mode = "paper"
    cfg.scalping.symbols = [SYMBOL]
    cfg.scalping.initial_capital = 1000.0
    cfg.scalping.timeframe = "1m"
    cfg.risk.trailing_stop_enabled = True
    cfg.risk.breakeven_activation_pct = 0.002
    cfg.risk.trailing_stop_activation_pct = 0.0035
    cfg.risk.trailing_stop_pct = 0.002
    cfg.risk.round_trip_cost_pct = 0.0025
    return cfg


def _seed_sqlite(monkeypatch, db_path):
    monkeypatch.setattr(main.state, "DB_PATH", db_path)
    main.state.save_position_at_entry(
        SYMBOL, "short", QUANTITY, ENTRY, TP, STOP, "grid_scalper@1m",
        entry_time="2026-11-12T11:26:04", leverage=1.0, margin=QUANTITY * ENTRY,
    )


def _snapshot_payload(**overrides):
    position = {
        "symbol": SYMBOL,
        "side": "short",
        "quantity": QUANTITY,
        "entry_price": ENTRY,
        "notional": QUANTITY * ENTRY,
        "margin": QUANTITY * ENTRY,
        "leverage": 1.0,
        "strategy": "grid_scalper@1m",
        "stop_price": STOP,
        "original_stop": STOP,
        "tp": TP,
        "trailing_stop": TRAILING,
        "breakeven_armed": True,
        "high_watermark": ENTRY,
        "low_watermark": MARK,
        "mark_price": MARK,
    }
    position.update(overrides)
    return {
        "open_positions": {SYMBOL: position},
        "position_targets": {
            SYMBOL: {
                "tp": TP,
                "sl": STOP,
                "strategy": "grid_scalper@1m",
                "timeframe": "1m",
                "signal_bar_open": "2026-11-12T11:25:00",
                "journal_entry": {
                    "entry_time": "2026-11-12T11:26:04",
                    "fill_price": ENTRY,
                    "quantity": QUANTITY,
                    "fee": 0.01,
                    "slippage": 0.0,
                    "strategy": "grid_scalper@1m",
                    "signal": "short",
                    "selected_leverage": 1.0,
                },
            }
        },
        "position_entry_cycle": {SYMBOL: 5},
        "cycle_count": 10,
        "account_balance": 1000.0,
    }


def _write_snapshot(tmp_path, payload=None, raw=None):
    path = tmp_path / "paper-session.state.json"
    if raw is not None:
        path.write_text(raw)
    else:
        path.write_text(json.dumps(payload if payload is not None else _snapshot_payload()))
    return path


def _make_bot(tmp_path, fresh_mark=81.60):
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = _cfg()
    bot.clock = lambda: datetime(2026, 11, 12, 11, 36)
    bot._clock_uses_local_timezone = False
    bot.exchange = PaperExchange.__new__(PaperExchange)
    bot.exchange.positions = {}
    bot.exchange.get_positions = MagicMock(return_value={})
    bot.exchange.get_position = lambda symbol: bot.exchange.positions.get(symbol)
    if fresh_mark is None:
        bot.exchange.fetch_ohlcv = MagicMock(side_effect=RuntimeError("offline test fixture"))
    else:
        bot.exchange.fetch_ohlcv = MagicMock(return_value=pd.DataFrame(
            {"open": [fresh_mark], "high": [fresh_mark], "low": [fresh_mark],
             "close": [fresh_mark], "volume": [1.0]},
            index=[pd.Timestamp("2026-11-12 11:35:00")],
        ))
    bot.risk_manager = RiskManager(bot.config)
    bot.loaded_positions = []
    bot.entries_blocked = False
    bot.position_targets = {}
    bot.position_entry_cycle = {}
    bot.cycle_count = 0
    bot.paper_trade_meta = {}
    bot._state_path = lambda: str(tmp_path / "paper-session.state.json")
    return bot


def test_restart_restores_trailing_breakeven_and_watermarks(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path)
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["trailing_stop"] == pytest.approx(TRAILING)
    assert pos["breakeven_armed"] is True
    assert pos["high_watermark"] == pytest.approx(ENTRY)
    assert pos["low_watermark"] == pytest.approx(MARK)
    # Historical mark is kept for reference; the active mark is the fresh one.
    assert pos["last_mark_price"] == pytest.approx(MARK)
    assert pos["mark_price"] == pytest.approx(81.60)
    # SQLite remains authoritative for identity and the original stop.
    assert pos["side"] == "short"
    assert pos["quantity"] == pytest.approx(QUANTITY)
    assert pos["entry_price"] == pytest.approx(ENTRY)
    assert pos["original_stop"] == pytest.approx(STOP)
    assert pos["stop_price"] == pytest.approx(STOP)
    assert pos["sl"] == pytest.approx(STOP)
    assert pos["tp"] == pytest.approx(TP)


def test_restored_trailing_stop_triggers_adverse_exit(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path)
    bot = _make_bot(tmp_path, fresh_mark=81.60)
    bot.load_state()

    trigger = pd.DataFrame(
        {"open": [81.65], "high": [81.72], "low": [81.60],
         "close": [81.70], "volume": [1.0]},
        index=[pd.Timestamp("2026-11-12 11:35:00")],
    )
    bot.exchange.fetch_ohlcv = MagicMock(return_value=trigger)
    bot._prepare_poll_close = MagicMock()
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_called_once()
    reason = bot._close_position.call_args.args[3]
    assert reason.startswith("TRAILING STOP")
    assert "Trailing stop hit" in reason


def test_original_stop_survives_without_a_snapshot(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    # No JSON file at all.
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["original_stop"] == pytest.approx(STOP)
    assert pos["stop_price"] == pytest.approx(STOP)
    assert pos["sl"] == pytest.approx(STOP)
    assert pos["trailing_stop"] is None
    assert pos["breakeven_armed"] is False
    assert bot.entries_blocked is False


def test_corrupt_json_does_not_erase_known_protection(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path, raw="{ this is not valid json")
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["original_stop"] == pytest.approx(STOP)
    assert pos["stop_price"] == pytest.approx(STOP)
    assert pos["trailing_stop"] is None
    assert pos["breakeven_armed"] is False


def test_armed_breakeven_with_corrupt_trailing_is_reconstructed(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path, payload=_snapshot_payload(
        trailing_stop="corrupt", breakeven_armed=True))
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    # entry * (1 - round_trip_cost_pct) for a short.
    assert pos["breakeven_armed"] is True
    assert pos["trailing_stop"] == pytest.approx(ENTRY * (1 - 0.0025))
    assert pos["original_stop"] == pytest.approx(STOP)


def test_mismatched_snapshot_keeps_sqlite_identity(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path, payload=_snapshot_payload(
        side="long", quantity=999.0, entry_price=1.0))
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["side"] == "short"
    assert pos["quantity"] == pytest.approx(QUANTITY)
    assert pos["entry_price"] == pytest.approx(ENTRY)
    # Dynamic state still restores from the same snapshot entry.
    assert pos["trailing_stop"] == pytest.approx(TRAILING)
    assert pos["breakeven_armed"] is True


def test_stale_snapshot_mark_is_not_used_when_refresh_fails(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path)
    bot = _make_bot(tmp_path, fresh_mark=None)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["last_mark_price"] == pytest.approx(MARK)
    assert pos["mark_price"] is None
    # Equity falls back to entry, not the stale 81.49 mark.
    assert bot.risk_manager.current_equity() == pytest.approx(
        bot.risk_manager.current_balance)


def test_restored_paper_exchange_agrees_with_risk_position(monkeypatch, tmp_path):
    _seed_sqlite(monkeypatch, tmp_path / "paper.db")
    _write_snapshot(tmp_path)
    bot = _make_bot(tmp_path, fresh_mark=81.60)

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    ex = bot.exchange.positions[SYMBOL]
    assert ex["side"] == pos["side"] == "short"
    assert ex["contracts"] == pytest.approx(pos["quantity"])
    assert ex["entryPrice"] == pytest.approx(pos["entry_price"])
    assert ex["leverage"] == pytest.approx(pos["leverage"])
    assert ex["margin"] == pytest.approx(pos["margin"])
    assert ex["sl"] == pytest.approx(pos["original_stop"])
    assert ex["tp"] == pytest.approx(pos["tp"])
    assert ex["trailing_stop"] == pytest.approx(pos["trailing_stop"])
    assert ex["breakeven_armed"] is pos["breakeven_armed"]
    assert ex["markPrice"] == pytest.approx(pos["mark_price"])


def test_portfolio_heat_uses_more_protective_trailing_stop():
    risk = RiskManager(_cfg())
    risk.register_position(
        SYMBOL, ENTRY * QUANTITY, "short", quantity=QUANTITY, stop_price=STOP)
    risk.update_position_price(SYMBOL, ENTRY)
    pos = risk.open_positions[SYMBOL]
    pos["original_stop"] = STOP

    original_heat = risk.stop_risk_for_position(pos)
    pos["trailing_stop"] = TRAILING
    trailing_heat = risk.stop_risk_for_position(pos)

    assert risk.effective_stop_price(pos) == pytest.approx(TRAILING)
    assert original_heat == pytest.approx(
        (STOP - ENTRY) * QUANTITY + QUANTITY * risk.stop_costs_per_unit(ENTRY, STOP))
    # The trailing stop is already beyond entry for a short, so only costs remain.
    assert trailing_heat == pytest.approx(
        QUANTITY * risk.stop_costs_per_unit(ENTRY, TRAILING))
    assert trailing_heat < original_heat


def test_long_effective_stop_prefers_higher_protective_level():
    risk = RiskManager(_cfg())
    pos = {
        "entry_price": 100.0, "quantity": 1.0, "side": "long",
        "original_stop": 95.0, "trailing_stop": 98.0,
    }
    assert risk.effective_stop_price(pos) == pytest.approx(98.0)
    pos["trailing_stop"] = 90.0
    assert risk.effective_stop_price(pos) == pytest.approx(95.0)


def test_missing_or_invalid_stop_is_not_zero_risk():
    risk = RiskManager(_cfg())
    for stop in (None, 0, -1, float("nan"), "bad"):
        pos = {
            "entry_price": 100.0, "quantity": 1.0, "side": "long",
            "original_stop": stop, "trailing_stop": stop,
        }
        assert risk.effective_stop_price(pos) is None
        assert risk.stop_risk_for_position(pos) is None
