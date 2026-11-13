"""Regression coverage for the second-stage safety audit and project-wide repair.

Every test is deterministic, offline and isolated: temporary SQLite/JSON files,
in-memory exchange doubles and no real credentials or network access.
"""
from datetime import datetime
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import main
import core.state as state
from config.settings import BotConfig
from core.exchange import PaperExchange
from core.risk_manager import RiskManager, validate_trailing_stop

SYMBOL = "BTC/USDT"
ENTRY = 100.0
NOW = datetime(2026, 1, 1, 0, 20)


def _cfg(mode="paper"):
    cfg = BotConfig()
    cfg.mode = mode
    cfg.scalping.scan_all_coins = False
    cfg.scalping.symbols = [SYMBOL]
    cfg.scalping.initial_capital = 1000.0
    cfg.scalping.timeframe = "5m"
    cfg.risk.trailing_stop_enabled = True
    cfg.risk.breakeven_activation_pct = 0.002
    cfg.risk.trailing_stop_activation_pct = 0.0035
    cfg.risk.trailing_stop_pct = 0.002
    cfg.risk.round_trip_cost_pct = 0.0025
    return cfg


def _make_bot(tmp_path, cfg=None, exchange=None):
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg or _cfg()
    bot.clock = lambda: NOW
    bot._clock_uses_local_timezone = False
    bot.exchange = exchange if exchange is not None else MagicMock()
    bot.risk_manager = RiskManager(bot.config)
    bot.position_targets = {}
    bot.position_entry_cycle = {}
    bot.cycle_count = 0
    bot.entries_blocked = False
    bot.entries_blocked_by_marks = False
    bot.mark_unavailable_symbols = set()
    bot.paper_trade_meta = {}
    bot.active_decisions = {}
    bot.loaded_positions = []
    bot.decision_log_path = str(tmp_path / "decisions.csv")
    bot._state_path = lambda: str(tmp_path / "state.json")
    return bot


def _frame(rows, start="2026-01-01 00:00", freq="5min"):
    index = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1.0] * len(rows)},
        index=index,
    )


def _seed(tmp_path, monkeypatch, symbol=SYMBOL, side="long", size=1.0,
          entry=ENTRY, tp=120.0, sl=95.0, leverage=1.0, margin=None):
    monkeypatch.setattr(state, "DB_PATH", tmp_path / "paper.db")
    state.save_position_at_entry(
        symbol, side, size, entry, tp, sl, "test", entry_time="2026-01-01T00:00:00",
        leverage=leverage, margin=margin if margin is not None else size * entry / leverage,
    )


def _paper_exchange(cfg=None):
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = cfg or _cfg()
    exchange.positions = {}
    exchange.orders = {}
    exchange.trade_history = []
    exchange.order_id = 0
    exchange._requested_leverage = {}
    exchange._balance_source = None
    exchange._balance = 1000.0
    exchange.latency_ms = 0
    exchange.base_slippage_pct = 0.0
    exchange.partial_fill_chance = 0.0
    exchange.reject_chance = 0.0
    exchange.get_positions = MagicMock(return_value={})
    exchange.get_position = lambda symbol: exchange.positions.get(symbol)
    return exchange


# --------------------------------------------------------------------------
# C1/C2: candidate and aggregate risk must fail closed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stop", [None, 0, -1, float("nan"), float("inf"), "bad", True])
def test_candidate_stop_that_cannot_be_priced_is_rejected(stop):
    risk = RiskManager(_cfg())
    allowed, reason, amount = risk.check_stop_loss_risk(ENTRY, stop, 1.0, "long")
    assert allowed is False
    assert "cannot price candidate" in reason
    assert math.isinf(amount)


@pytest.mark.parametrize("side", ["", "sideways", None, 42])
def test_unknown_candidate_side_is_rejected(side):
    risk = RiskManager(_cfg())
    allowed, reason, amount = risk.check_stop_loss_risk(ENTRY, 95.0, 1.0, side)
    assert allowed is False
    assert "unknown candidate position side" in reason
    assert math.isinf(amount)


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), -1, 0, None, "bad", True])
def test_non_finite_or_non_positive_quantity_is_rejected(quantity):
    risk = RiskManager(_cfg())
    allowed, _reason, amount = risk.check_stop_loss_risk(ENTRY, 95.0, quantity, "long")
    assert allowed is False
    assert math.isinf(amount)


def test_invalid_candidate_is_rejected_even_with_a_valid_open_position():
    risk = RiskManager(_cfg())
    risk.register_position(SYMBOL, 100.0, "long", quantity=1.0, stop_price=95.0)
    risk.update_position_price(SYMBOL, ENTRY)
    allowed, reason, amount = risk.check_stop_loss_risk(ENTRY, float("nan"), 1.0, "long")
    assert allowed is False
    assert "cannot price candidate" in reason
    assert math.isinf(amount)


def test_nan_position_never_passes_and_heat_stays_finite():
    risk = RiskManager(_cfg())
    risk.register_position(SYMBOL, 100.0, "long", quantity=1.0, stop_price=95.0)
    risk.update_position_price(SYMBOL, ENTRY)
    risk.open_positions[SYMBOL]["quantity"] = float("nan")
    stats = risk.get_stats()
    assert math.isfinite(stats["portfolio_heat"])
    assert stats["unpriced_stop_risk_positions"] == 1
    assert stats["unpriced_equity_positions"] >= 1


# --------------------------------------------------------------------------
# H1: restored original stop is authoritative in the exit path
# --------------------------------------------------------------------------

def test_exit_uses_restored_sqlite_stop_before_less_protective_targets(tmp_path):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=95.0)
    bot.risk_manager.update_position_price(SYMBOL, ENTRY)
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 101, 99, 100), (100, 101, 94, 96)])
    # A stale snapshot target that is much less protective than the SQLite stop.
    bot.position_targets[SYMBOL] = {"sl": 80.0, "tp": 120.0}
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_called_once()
    args, kwargs = bot._close_position.call_args
    assert args[3] == "STOP LOSS"
    assert kwargs["reference_price"] == pytest.approx(95.0)


def test_exit_take_profit_prefers_restored_sqlite_target(tmp_path):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=95.0)
    bot.risk_manager.open_positions[SYMBOL]["tp"] = 110.0
    bot.risk_manager.update_position_price(SYMBOL, ENTRY)
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 101, 99, 100), (100, 112, 105, 111)])
    bot.position_targets[SYMBOL] = {"sl": 95.0, "tp": 120.0}
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    args, kwargs = bot._close_position.call_args
    assert args[3] == "TAKE PROFIT"
    assert kwargs["reference_price"] == pytest.approx(110.0)


# --------------------------------------------------------------------------
# H2: malformed targets must not disable protective exits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("targets", [
    "not-a-dict",
    {"sl": "bad", "tp": "bad"},
    {"sl": float("nan"), "tp": float("inf")},
    {"sl": -5, "tp": 0},
])
def test_malformed_targets_do_not_abort_stop_loss(tmp_path, targets):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=95.0)
    bot.risk_manager.update_position_price(SYMBOL, ENTRY)
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 100.4, 99, 100), (100, 100.4, 94, 96)])
    bot.position_targets[SYMBOL] = targets
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_called_once()
    assert bot._close_position.call_args.args[3] == "STOP LOSS"


def test_malformed_targets_still_allow_trailing_exit(tmp_path):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=90.0)
    bot.risk_manager.update_position_price(SYMBOL, ENTRY)
    bot.risk_manager.open_positions[SYMBOL]["trailing_stop"] = 99.0
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 100.4, 99, 100), (100, 100.2, 98, 98)])
    bot.position_targets[SYMBOL] = "not-a-dict"
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_called_once()
    assert bot._close_position.call_args.args[3].startswith("TRAILING STOP")


# --------------------------------------------------------------------------
# H3: failed mark refresh blocks entries, retries, and clears on success
# --------------------------------------------------------------------------

def test_failed_mark_refresh_blocks_entries_and_success_unblocks(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    bot.exchange.fetch_ohlcv = MagicMock(side_effect=RuntimeError("offline"))

    bot.load_state()

    assert bot.entries_blocked is True
    assert SYMBOL in bot.mark_unavailable_symbols
    assert bot.risk_manager.open_positions[SYMBOL]["mark_price"] is None

    bot.exchange.fetch_ohlcv = MagicMock(return_value=_frame(
        [(100, 100, 100, 100)], start="2026-01-01 00:20"))
    bot.load_state()

    assert bot.entries_blocked is False
    assert bot.mark_unavailable_symbols == set()
    assert bot.risk_manager.open_positions[SYMBOL]["mark_price"] == pytest.approx(100.0)


def test_mark_refresh_retries_three_times(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    bot.exchange.fetch_ohlcv = MagicMock(side_effect=RuntimeError("offline"))

    bot.load_state()

    assert bot.exchange.fetch_ohlcv.call_count == 3


def test_repeated_load_state_is_idempotent(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    bot.exchange.fetch_ohlcv = MagicMock(return_value=_frame(
        [(100, 100, 100, 100)], start="2026-01-01 00:20"))

    bot.load_state()
    first_exposure = dict(bot.risk_manager.symbol_exposure)
    first_position = dict(bot.risk_manager.open_positions[SYMBOL])
    bot.load_state()

    assert bot.risk_manager.symbol_exposure == first_exposure == {SYMBOL: 100.0}
    assert len(bot.risk_manager.open_positions) == 1
    assert bot.risk_manager.open_positions[SYMBOL]["original_stop"] == first_position["original_stop"]
    assert bot.exchange.positions[SYMBOL]["contracts"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# H4/H5: watermark and trailing validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("high,low", [
    (1e9, 1e-9),
    (True, False),
    ("bad", "bad"),
    (50.0, 150.0),
])
def test_corrupt_watermarks_are_reset_not_used_for_trailing(tmp_path, high, low):
    bot = _make_bot(tmp_path)
    pos = {"entry_price": ENTRY, "side": "long", "quantity": 1.0,
           "original_stop": 95.0}
    snapshot = {"high_watermark": high, "low_watermark": low,
                "trailing_stop": None, "breakeven_armed": False}

    bot._merge_restored_dynamic_state(SYMBOL, pos, snapshot)

    assert pos["high_watermark"] == pytest.approx(ENTRY)
    assert pos["low_watermark"] == pytest.approx(ENTRY)
    assert pos["trailing_stop"] is None


@pytest.mark.parametrize("side,trailing", [
    ("long", True),
    ("long", 1e9),
    ("long", float("nan")),
    ("short", 1e-12),
    ("short", float("inf")),
])
def test_implausible_trailing_stop_falls_back_to_original(tmp_path, side, trailing):
    bot = _make_bot(tmp_path)
    original = 95.0 if side == "long" else 105.0
    pos = {"entry_price": ENTRY, "side": side, "quantity": 1.0,
           "original_stop": original}
    snapshot = {"high_watermark": ENTRY, "low_watermark": ENTRY,
                "trailing_stop": trailing, "breakeven_armed": False}

    bot._merge_restored_dynamic_state(SYMBOL, pos, snapshot)

    assert pos["trailing_stop"] is None
    # Portfolio heat must not collapse to costs only because of a corrupt value.
    risk = bot.risk_manager
    risk.open_positions[SYMBOL] = dict(pos)
    heat = risk.stop_risk_for_position(risk.open_positions[SYMBOL])
    assert heat is not None and heat > 0


def test_short_trailing_stop_above_entry_never_weakens_original(tmp_path):
    bot = _make_bot(tmp_path)
    pos = {"entry_price": ENTRY, "side": "short", "quantity": 1.0,
           "original_stop": 105.0}
    snapshot = {"high_watermark": ENTRY, "low_watermark": ENTRY,
                "trailing_stop": 110.0, "breakeven_armed": True}

    bot._merge_restored_dynamic_state(SYMBOL, pos, snapshot)

    assert pos["trailing_stop"] == pytest.approx(ENTRY * (1 - 0.0025))
    assert pos["trailing_stop"] <= 105.0


def test_trailing_validation_helpers_are_side_aware():
    assert validate_trailing_stop(98.0, ENTRY, "long", 101.0, 99.0) == pytest.approx(98.0)
    assert validate_trailing_stop(99.5, ENTRY, "short", 101.0, 99.0) == pytest.approx(99.5)
    assert validate_trailing_stop(1e9, ENTRY, "long", 101.0, 99.0) is None
    assert validate_trailing_stop(1e-12, ENTRY, "short", 101.0, 99.0) is None
    assert validate_trailing_stop(True, ENTRY, "long", 101.0, 99.0) is None


# --------------------------------------------------------------------------
# M1: breakeven reconstruction never weakens protection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("side,original,expected", [
    ("long", 101.0, 101.0),
    ("long", 90.0, ENTRY * (1 + 0.0025)),
    ("short", 99.0, 99.0),
    ("short", 110.0, ENTRY * (1 - 0.0025)),
])
def test_breakeven_reconstruction_keeps_tighter_stop(tmp_path, side, original, expected):
    bot = _make_bot(tmp_path)
    assert bot._breakeven_stop(ENTRY, side, original) == pytest.approx(expected)


# --------------------------------------------------------------------------
# M2: SQLite leverage/margin normalization
# --------------------------------------------------------------------------

def test_invalid_sqlite_leverage_and_margin_are_recomputed_consistently(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, leverage="bad", margin=-5.0)
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    bot.exchange.fetch_ohlcv = MagicMock(return_value=_frame(
        [(100, 100, 100, 100)], start="2026-01-01 00:20"))

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["leverage"] == pytest.approx(1.0)
    assert pos["margin"] == pytest.approx(ENTRY)
    assert bot.exchange.positions[SYMBOL]["leverage"] == pytest.approx(pos["leverage"])
    assert bot.exchange.positions[SYMBOL]["margin"] == pytest.approx(pos["margin"])


def test_capped_restored_leverage_recomputes_margin_consistently(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, leverage=10.0, margin=10.0)
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    bot.exchange.fetch_ohlcv = MagicMock(return_value=_frame(
        [(100, 100, 100, 100)], start="2026-01-01 00:20"))

    bot.load_state()

    pos = bot.risk_manager.open_positions[SYMBOL]
    assert pos["leverage"] == pytest.approx(3.0)
    assert pos["margin"] == pytest.approx(ENTRY / 3.0)
    assert bot.exchange.positions[SYMBOL]["leverage"] == pytest.approx(pos["leverage"])
    assert bot.exchange.positions[SYMBOL]["margin"] == pytest.approx(pos["margin"])


# --------------------------------------------------------------------------
# M3/M4: malformed snapshot fields are isolated
# --------------------------------------------------------------------------

def test_malformed_cycle_count_does_not_abort_account_restore(tmp_path):
    bot = _make_bot(tmp_path)
    bot.exchange.get_positions = MagicMock(return_value={})
    snapshot = {
        "cycle_count": "not-an-int",
        "account_balance": 900.0,
        "peak_balance": 1100.0,
        "daily_pnl": -5.0,
        "daily_trades": 7,
        "initial_portfolio_equity": 1000.0,
        "portfolio_profit_target_pct": 70.0,
    }
    (tmp_path / "state.json").write_text(json.dumps(snapshot))

    bot.load_state()

    assert bot.cycle_count == 0
    assert bot.risk_manager.current_balance == pytest.approx(900.0)
    assert bot.risk_manager.peak_balance == pytest.approx(1100.0)
    assert bot.risk_manager.daily_pnl == pytest.approx(-5.0)
    assert bot.risk_manager.daily_trades == 7
    assert bot.config.portfolio_profit_target_pct == pytest.approx(70.0)


def test_malformed_journal_for_one_symbol_does_not_block_others(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, symbol="AAA/USDT")
    _seed(tmp_path, monkeypatch, symbol="BBB/USDT")
    bot = _make_bot(tmp_path)
    bot.exchange.get_positions = MagicMock(return_value={})
    snapshot = {
        "open_positions": {},
        "position_targets": {
            "AAA/USDT": "not-a-dict",
            "BBB/USDT": {
                "tp": 120.0, "sl": 95.0, "strategy": "test",
                "journal_entry": {
                    "entry_time": "2026-01-01T00:00:00", "fill_price": 100.0,
                    "quantity": 1.0, "fee": 0.01, "slippage": 0.0,
                    "strategy": "test", "signal": "long", "selected_leverage": 1.0,
                },
            },
        },
        "account_balance": 950.0,
    }
    (tmp_path / "state.json").write_text(json.dumps(snapshot))

    bot.load_state()

    assert "BBB/USDT" in bot.paper_trade_meta
    assert "AAA/USDT" not in bot.paper_trade_meta
    assert bot.risk_manager.current_balance == pytest.approx(950.0)


# --------------------------------------------------------------------------
# M5: type-confusion helpers
# --------------------------------------------------------------------------

def test_helpers_reject_booleans_and_junk_flags():
    assert main._positive_float(True) is None
    assert main._positive_float(False) is None
    assert main._finite_float(True) is None
    assert main._bool_value(0.5) is False
    assert main._bool_value(2) is False
    assert main._bool_value(1) is True
    assert main._bool_value(1.0) is True
    assert main._bool_value("false") is False
    assert main._non_negative_int(True, 7) == 7
    assert main._non_negative_int(-3, 7) == 7


# --------------------------------------------------------------------------
# M6: stale and outlier candles are rejected
# --------------------------------------------------------------------------

def test_stale_candle_blocks_exits_and_entries(tmp_path):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=95.0)
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 101, 94, 96)], start="2020-01-01 00:00")
    bot.position_targets[SYMBOL] = {"sl": 95.0, "tp": 120.0}
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_not_called()
    assert bot.entries_blocked is True


def test_outlier_candle_blocks_exits_and_entries(tmp_path):
    bot = _make_bot(tmp_path)
    bot.risk_manager.register_position(SYMBOL, 100.0, "long", quantity=1.0,
                                       stop_price=95.0)
    bot.exchange.get_position.return_value = {
        "entryPrice": ENTRY, "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame(
        [(100, 101, 99, 1e9)], start="2026-01-01 00:20")
    bot.position_targets[SYMBOL] = {"sl": 95.0, "tp": 120.0}
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_not_called()
    assert bot.entries_blocked is True


# --------------------------------------------------------------------------
# M7: live mode never exposes a snapshot mark as a live price
# --------------------------------------------------------------------------

def test_live_restore_keeps_snapshot_mark_inactive(tmp_path):
    bot = _make_bot(tmp_path, cfg=_cfg("live"))
    pos = {"entry_price": ENTRY, "side": "long", "quantity": 1.0,
           "original_stop": 95.0}
    snapshot = {"high_watermark": ENTRY, "low_watermark": ENTRY,
                "trailing_stop": 98.0, "breakeven_armed": True,
                "mark_price": 50.0}

    bot._merge_restored_dynamic_state(SYMBOL, pos, snapshot)

    assert pos["mark_price"] is None
    assert pos["last_mark_price"] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# M8: SQLite original stops are validated
# --------------------------------------------------------------------------

@pytest.mark.parametrize("side,stop", [
    ("long", 1e9),
    ("long", 110.0),
    ("long", 0),
    ("long", float("nan")),
    ("short", 1e-9),
    ("short", 90.0),
    ("short", -1),
])
def test_implausible_sqlite_stop_is_not_trusted(tmp_path, side, stop):
    bot = _make_bot(tmp_path)
    row = {"symbol": SYMBOL, "entry_price": ENTRY, "side": side, "sl": stop}
    assert bot._restore_original_stop(row, None, None) is None


def test_plausible_sqlite_stop_is_kept(tmp_path):
    bot = _make_bot(tmp_path)
    assert bot._restore_original_stop(
        {"symbol": SYMBOL, "entry_price": ENTRY, "side": "long", "sl": 95.0},
        None, None) == pytest.approx(95.0)
    assert bot._restore_original_stop(
        {"symbol": SYMBOL, "entry_price": ENTRY, "side": "short", "sl": 105.0},
        None, None) == pytest.approx(105.0)


# --------------------------------------------------------------------------
# L1: exchange synchronization keeps the full metadata schema
# --------------------------------------------------------------------------

def test_exchange_sync_preserves_metadata_and_entry_time(tmp_path):
    bot = _make_bot(tmp_path, exchange=_paper_exchange())
    entry_time = datetime(2026, 1, 1, 0, 0)
    pos = {
        "entry_price": ENTRY, "side": "long", "quantity": 1.0,
        "notional": 100.0, "leverage": 1.0, "margin": 100.0,
        "strategy": "grid", "entry_time": entry_time, "original_stop": 95.0,
        "tp": 120.0, "mark_price": 101.0, "trailing_stop": None,
        "breakeven_armed": False, "high_watermark": 101.0, "low_watermark": 99.0,
    }

    bot._sync_exchange_position(SYMBOL, pos)

    ex = bot.exchange.positions[SYMBOL]
    assert ex["strategy"] == "grid"
    assert ex["notional"] == pytest.approx(100.0)
    assert ex["entry_time"] == entry_time
    assert ex["timestamp"] == entry_time.isoformat()
    assert ex["markPrice"] == pytest.approx(101.0)


# --------------------------------------------------------------------------
# L3: negative cost configuration cannot reduce estimated risk
# --------------------------------------------------------------------------

def test_negative_costs_do_not_reduce_stop_risk():
    cfg = _cfg()
    cfg.backtest.maker_fee = -0.01
    cfg.backtest.taker_fee = -0.01
    cfg.backtest.slippage = -0.01
    cfg.backtest.funding_rate = -0.01
    risk = RiskManager(cfg)
    cost = risk.stop_costs_per_unit(ENTRY, 95.0)
    assert cost >= 0
    risk.register_position(SYMBOL, 100.0, "long", quantity=1.0, stop_price=95.0)
    risk.update_position_price(SYMBOL, ENTRY)
    assert risk.stop_risk_for_position(risk.open_positions[SYMBOL]) >= 5.0


# --------------------------------------------------------------------------
# L5: invalid restored sides block rather than silently disabling trailing
# --------------------------------------------------------------------------

def test_invalid_restored_side_blocks_entries(tmp_path):
    bot = _make_bot(tmp_path)
    pos = {"entry_price": ENTRY, "side": "sideways", "quantity": 1.0,
           "original_stop": 95.0}
    snapshot = {"high_watermark": ENTRY, "low_watermark": ENTRY,
                "trailing_stop": 98.0, "breakeven_armed": True}

    bot._merge_restored_dynamic_state(SYMBOL, pos, snapshot)

    assert bot.entries_blocked is True
    assert pos["trailing_stop"] is None


# --------------------------------------------------------------------------
# Additional project-wide defects
# --------------------------------------------------------------------------

def test_paper_close_position_can_skip_second_slippage():
    exchange = _paper_exchange()
    exchange.base_slippage_pct = 0.05
    exchange.positions[SYMBOL] = {
        "entryPrice": 100.0, "contracts": 1.0, "side": "long"}

    pnl = exchange.close_position(SYMBOL, 110.0, apply_slippage=False)

    assert pnl == pytest.approx(10.0)


def test_check_limit_fill_records_full_position_metadata():
    cfg = _cfg()
    exchange = _paper_exchange()
    exchange.config = cfg
    exchange.fetch_ohlcv = MagicMock(return_value=_frame(
        [(100, 101, 99, 100), (100, 101, 99, 100)]))
    exchange._requested_leverage = {SYMBOL: 2.0}
    order = exchange.create_limit_order(SYMBOL, "buy", 1.0, 100.0)
    exchange.orders[order["id"]] = order

    filled = exchange.check_limit_fill(SYMBOL, order["id"], 100.0, "buy")

    assert filled["status"] == "filled"
    position = exchange.positions[SYMBOL]
    assert position["markPrice"] == pytest.approx(100.0)
    assert position["leverage"] == pytest.approx(2.0)
    assert position["margin"] == pytest.approx(50.0)
    assert position["notional"] == pytest.approx(100.0)


def test_persist_state_writes_atomically_and_leaves_no_temp_file(tmp_path):
    bot = _make_bot(tmp_path)
    bot.config = _cfg()
    bot.risk_manager.current_balance = 1010.0
    bot.risk_manager.peak_balance = 1010.0
    bot.risk_manager.peak_equity = 1010.0

    bot.persist_state()

    payload = json.loads((tmp_path / "state.json").read_text())
    assert payload["account_balance"] == pytest.approx(1010.0)
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]


def test_duplicate_open_rows_keep_the_latest(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, sl=95.0)
    state.save_position_at_entry(
        SYMBOL, "long", 1.0, ENTRY, 120.0, 90.0, "test",
        entry_time="2026-01-01T00:00:00", leverage=1.0, margin=100.0)

    rows = state.load_open_positions()

    assert len(rows) == 1
    assert rows[0]["sl"] == pytest.approx(90.0)


def test_malformed_exchange_position_blocks_instead_of_closing_at_zero(tmp_path):
    bot = _make_bot(tmp_path)
    bot.exchange.get_position.return_value = {
        "entryPrice": "bad", "contracts": 1.0, "side": "long"}
    bot.exchange.fetch_ohlcv.return_value = _frame([(100, 101, 94, 96)])
    bot._close_position = MagicMock()

    bot.check_exits(SYMBOL)

    bot._close_position.assert_not_called()
    assert bot.entries_blocked is True


def test_long_and_short_stop_risk_are_symmetric():
    cfg = _cfg()
    cfg.backtest.maker_fee = 0.0
    cfg.backtest.taker_fee = 0.0
    cfg.backtest.slippage = 0.0
    cfg.backtest.funding_rate = 0.0
    cfg.risk.round_trip_cost_pct = 0.0
    long_risk = RiskManager(cfg)
    short_risk = RiskManager(cfg)
    long_risk.register_position("LONG", 100.0, "long", quantity=1.0, stop_price=95.0)
    short_risk.register_position("SHORT", 100.0, "short", quantity=1.0, stop_price=105.0)
    long_risk.update_position_price("LONG", ENTRY)
    short_risk.update_position_price("SHORT", ENTRY)

    assert (long_risk.stop_risk_for_position(long_risk.open_positions["LONG"])
            == pytest.approx(
                short_risk.stop_risk_for_position(short_risk.open_positions["SHORT"])))
