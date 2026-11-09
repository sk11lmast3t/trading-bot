import csv
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import main
from config.settings import BotConfig
from core.indicators import Indicators
from core.strategies import Signal, TradeSetup


def candle_frame(index, close):
    close = pd.Series(close, index=index, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0,
    }, index=index)


def test_context_is_completed_bar_only_and_has_expected_values():
    signal_open = pd.Timestamp("2026-04-06 12:05:00", tz="UTC")
    five_idx = pd.date_range(end=signal_open, periods=300, freq="5min")
    five = candle_frame(five_idx, [100 + i * 0.1 for i in range(len(five_idx))])
    hourly_idx = pd.date_range("2026-04-05 14:00", "2026-04-06 12:00", freq="1h", tz="UTC")
    hourly = candle_frame(hourly_idx, [100 + i for i in range(len(hourly_idx))])
    half_idx = pd.date_range("2026-04-06 00:00", "2026-04-06 12:00", freq="30min", tz="UTC")
    half = candle_frame(half_idx, [90 + i * 0.5 for i in range(len(half_idx))])

    class FakeExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return {"1h": hourly, "30m": half}[timeframe]

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.exchange = FakeExchange()
    context = bot._decision_context("BTC/USDT", signal_open, five)

    completed_hour = hourly.loc[hourly.index + pd.Timedelta(hours=1) <= signal_open + pd.Timedelta(minutes=5)]
    hclose = completed_hour["close"]
    hatr = Indicators.atr(completed_hour.high, completed_hour.low, hclose, 14)
    hema = Indicators.ema(hclose, 20)
    expected_slope = (hema.iloc[-1] - hema.iloc[-2]) / hema.iloc[-2] * 100
    assert context["atr_1h_pct"] == pytest.approx(hatr.iloc[-1] / hclose.iloc[-1] * 100)
    assert context["ema20_1h_slope_sign"] == "up"
    assert context["ema20_1h_slope_pct"] == pytest.approx(expected_slope)
    assert context["trend_30m"] == "up"
    five_close = five.close
    expected_rsi = Indicators.rsi(five_close, 14).iloc[-1]
    expected_distance = ((five_close.iloc[-1] - Indicators.ema(five_close, 20).iloc[-1]) /
                         Indicators.atr(five.high, five.low, five_close, 14).iloc[-1])
    assert context["rsi_5m"] == pytest.approx(expected_rsi)
    assert context["ema_distance_atr_5m"] == pytest.approx(expected_distance)
    assert context["hour_utc"] == 12
    assert context["day_of_week_utc"] == 0
    assert context["rv_24h_bucket"] == "low"

    changed = hourly.copy()
    changed.loc[signal_open, ["open", "high", "low", "close"]] = [1, 1e9, 1, 1e9]
    bot.exchange.fetch_ohlcv = lambda symbol, timeframe, limit: {"1h": changed, "30m": half}[timeframe]
    assert bot._decision_context("BTC/USDT", signal_open, five) == context


def test_closed_trade_journals_excursions_r_and_stable_header(tmp_path, monkeypatch):
    path = tmp_path / "decisions.csv"
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.decision_log_path = str(path)
    bot.decision_log_lock = __import__("threading").Lock()
    bot.config = SimpleNamespace(
        mode="paper", backtest=SimpleNamespace(taker_fee=0.0, slippage=0.0),
    )
    bot.clock = lambda: datetime(2026, 4, 6, 12, 25)
    bot.exchange = MagicMock()
    bot.risk_manager = MagicMock()
    bot.risk_manager.current_balance = 1000.0
    bot.risk_manager.record_trade = MagicMock()
    bot.risk_manager.release_position = MagicMock()
    bot.position_entry_cycle = {"BTC/USDT": 1}
    bot.position_targets = {"BTC/USDT": {
        "sl": 95.0, "tp": 110.0, "signal_bar_open": "2026-04-06T12:00:00",
    }}
    bot.active_decisions = {}
    bot.paper_trade_meta = {"BTC/USDT": {"entry_fee": 0.0, "entry_slippage": 0.0}}
    bot.persist_state = MagicMock()
    monkeypatch.setattr(main.state, "record_trade", lambda *args, **kwargs: None)

    row = {
        "ts": "2026-04-06T12:00:00", "symbol": "BTC/USDT", "strategy": "synthetic",
        "signal": "LONG", "intended_entry": 100.0, "sl": 95.0, "tp": 110.0,
        "status": "entered", "fill_price": 100.0, "quantity": 1.0,
        "notional": 100.0, "planned_sl_bps": 500.0, "planned_tp_bps": 1000.0,
        "atr_1h_pct": 1.25, "ema20_1h_slope_sign": "up", "ema20_1h_slope_pct": 0.02,
        "trend_30m": "up", "rsi_5m": 60.0, "ema_distance_atr_5m": 0.5,
        "hour_utc": 12, "day_of_week_utc": 0, "rv_24h_bucket": "medium",
        "entry_fill_diff_bps": 0.0, "entry_time": datetime(2026, 4, 6, 12, 10),
    }
    bot._append_decision(row)
    bot.active_decisions["BTC/USDT"] = dict(row)
    frame_idx = pd.date_range("2026-04-06 12:00", periods=5, freq="5min")
    frame = pd.DataFrame({
        "high": [101, 103, 102, 100, 100],
        "low": [99, 98, 99, 97, 99],
        "close": [100, 101, 101, 100, 100],
    }, index=frame_idx)
    position = {"entryPrice": 100.0, "contracts": 1.0, "side": "long"}
    assert bot._exit_excursions("BTC/USDT", position, frame, row) == {
        "mfe_bps": 300.0, "mae_bps": 300.0, "bars_held": 4,
    }
    bot._prepare_poll_close("BTC/USDT", position, frame, 102.0)
    assert bot.active_decisions["BTC/USDT"]["mfe_bps"] == pytest.approx(300.0)
    bot._close_position("BTC/USDT", position, 101.0, "TAKE PROFIT", reference_price=102.0)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == main.DECISION_FIELDS
        rows = list(reader)
    trade = rows[0]
    assert trade["status"] == "closed", trade
    assert float(trade["mfe_bps"]) == pytest.approx(300.0)
    assert float(trade["mae_bps"]) == pytest.approx(300.0)
    assert int(trade["bars_held"]) == 4
    assert float(trade["r_multiple"]) == pytest.approx(0.4)
    assert float(trade["planned_sl_bps"]) == pytest.approx(500.0)
    assert float(trade["planned_tp_bps"]) == pytest.approx(1000.0)
    assert float(trade["exit_fill_diff_bps"]) == pytest.approx((101 / 102 - 1) * 10000)
    assert trade["close_source"] == "poll"
    assert len(rows[0]) == len(main.DECISION_FIELDS)


def test_skipped_signal_is_logged_with_context_columns(tmp_path, monkeypatch):
    cfg = BotConfig()
    cfg.scalping.scan_all_coins = False
    cfg.long_only = True
    index = pd.date_range("2026-04-06 10:00", periods=50, freq="5min")
    five = candle_frame(index, [100 + i * 0.1 for i in range(len(index))])
    hourly_idx = pd.date_range("2026-04-04", "2026-04-06 10:00", freq="1h", tz="UTC")
    hourly = candle_frame(hourly_idx, [100 + i * 0.1 for i in range(len(hourly_idx))])
    half_idx = pd.date_range("2026-04-05", "2026-04-06 10:00", freq="30min", tz="UTC")
    half = candle_frame(half_idx, [100 + i * 0.1 for i in range(len(half_idx))])

    class FakeExchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return {"5m": five, "1h": hourly, "30m": half}[timeframe].iloc[-limit:]

    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config = cfg
    bot.strategy = MagicMock()
    bot.strategy.lookback_bars = 30
    bot.strategy.analyze.return_value = TradeSetup(Signal.SHORT, 105, 106, 103, 0.8, "synthetic")
    bot.exchange = FakeExchange()
    bot.risk_manager = MagicMock()
    bot.risk_manager.current_balance = 1000
    bot.last_processed_bar = {}
    bot.entries_blocked = False
    bot.last_rejection_log = {}
    bot.decision_log_path = str(tmp_path / "decisions.csv")
    bot.decision_log_lock = __import__("threading").Lock()
    bot.clock = lambda: datetime(2026, 4, 6, 12, 0)
    monkeypatch.setattr(Indicators, "calculate_all", lambda frame, config: frame.assign(
        rsi=55.0, atr=1.0, ema_medium=frame.close.ewm(span=20, adjust=False).mean()))

    bot.process_symbol("BTC/USDT")

    with open(bot.decision_log_path, newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "skipped:long_only"
    assert row["hour_utc"] != ""
    assert row["day_of_week_utc"] != ""
    assert row["rsi_5m"] != ""
    assert row["rv_24h_bucket"] != ""
    assert row["trend_30m"] != ""


def test_exchange_flat_is_recorded_as_exchange_stop(tmp_path):
    path = tmp_path / "decisions.csv"
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.decision_log_path = str(path)
    bot.decision_log_lock = __import__("threading").Lock()
    bot.config = SimpleNamespace(mode="live", scalping=SimpleNamespace(timeframe="5m"))
    bot.clock = lambda: datetime(2026, 4, 6, 12, 25)
    idx = pd.date_range("2026-04-06 12:00", periods=5, freq="5min")
    bot.exchange = SimpleNamespace(fetch_ohlcv=lambda *args: pd.DataFrame({
        "high": [101, 103, 102, 100, 100], "low": [99, 98, 99, 97, 99],
        "close": [100, 101, 101, 100, 100],
    }, index=idx))
    entry = {
        "ts": "2026-04-06T12:00:00", "symbol": "BTC/USDT", "strategy": "synthetic",
        "signal": "LONG", "status": "entered", "fill_price": 100.0,
        "intended_entry": 100.0, "sl": 95.0, "tp": 110.0, "quantity": 1.0,
        "notional": 100.0, "entry_time": datetime(2026, 4, 6, 12, 10),
    }
    bot._append_decision(entry)
    bot.active_decisions = {"BTC/USDT": dict(entry)}
    bot.position_targets = {"BTC/USDT": {
        "signal_bar_open": "2026-04-06T12:00:00", "journal_entry": dict(entry),
    }}

    bot._record_exchange_stop_flat("BTC/USDT")

    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "closed"
    assert row["close_source"] == "exchange_stop_inferred"
    assert row["exit_reason"] == "exchange_stop_or_external_close"
    assert row["exit_price"] == ""


def test_legacy_decision_header_is_upgraded_before_append(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("ts,symbol,status\nold,BTC/USDT,skipped:hold\n", encoding="utf-8")
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.decision_log_path = str(path)
    bot._append_decision({"ts": "new", "symbol": "ETH/USDT", "status": "skipped:hold"})

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == main.DECISION_FIELDS
    assert rows[0]["ts"] == "old"
    assert rows[0]["atr_1h_pct"] == ""
    assert rows[1]["ts"] == "new"


