"""Network-free tests for strategy/timeframe discovery and arbitration."""
import numpy as np
import pandas as pd

import main
from core import state
from config.settings import BotConfig
from core.strategies import Signal, TradeSetup
from core.strategy_engine import (
    STRATEGIES, StrategyEngine, StrategyInfo, completed_frame, estimate_reward_cost,
    resolve_selection,
)


def candles(count=40, freq="5min"):
    index = pd.date_range("2026-01-01", periods=count, freq=freq)
    close = np.linspace(100, 104, count)
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(count, 100.0),
    }, index=index)


def test_registry_contains_all_registered_strategy_classes():
    assert set(STRATEGIES) == {
        "ema_rsi_atr", "momentum_scalper", "grid_scalper", "orderbook_imbalance",
        "kairi_disparity", "crypto_laggard", "combined", "mtf_trend_pullback",
    }


def test_timeframe_selection_all_explicit_and_unsupported():
    strategies, selected, warnings = resolve_selection(
        None, ["all"], None, ["all"], ["1m", "5m", "1h"])
    assert set(strategies) == set(STRATEGIES)
    assert selected == ["1m", "5m", "1h"]
    assert warnings == []
    assert resolve_selection("grid_scalper", None, "5m", ["2m"], ["5m"])[1:] == (
        [], ["unsupported timeframe: 2m"])
    assert resolve_selection(None, ["grid_scalper", "kairi_disparity"], None,
                             ["15m"], ["5m", "15m"])[0] == ["grid_scalper", "kairi_disparity"]


def test_only_completed_candles_are_retained():
    frame = candles(3, "5min")
    result = completed_frame(frame, "5m", now=pd.Timestamp("2026-01-01 00:11"))
    assert list(result.index) == list(frame.index[:2])


def test_strategy_failures_are_isolated_and_candle_fetch_is_cached(monkeypatch):
    class Broken:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            raise RuntimeError("fixture")

    class Hold:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.HOLD)

    monkeypatch.setitem(STRATEGIES, "fixture_broken", StrategyInfo("fixture_broken", Broken, (), 1, "both"))
    monkeypatch.setitem(STRATEGIES, "fixture_hold", StrategyInfo("fixture_hold", Hold, (), 1, "both"))

    class Exchange:
        calls = 0

        def fetch_ohlcv(self, symbol, timeframe, limit):
            self.calls += 1
            return candles()

    exchange = Exchange()
    cfg = BotConfig()
    now = pd.Timestamp("2026-01-01 03:20")
    engine = StrategyEngine(["fixture_broken", "fixture_hold"], ["5m"], cfg)
    setup, frame, timeframe, bar = engine.evaluate(exchange, "BTC/USDT", now=now)
    assert setup.signal == Signal.HOLD
    assert frame is not None and timeframe == "5m" and bar is not None
    assert "strategy error" in engine.last_support[("BTC/USDT", "fixture_broken", "5m")]
    engine.evaluate(exchange, "BTC/USDT", now=now + pd.Timedelta(seconds=1))
    assert exchange.calls == 1


def test_incremental_update_uses_since_and_reuses_indicator_frame(monkeypatch):
    class Exchange:
        def __init__(self):
            self.since_values = []

        def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
            if since is None:
                return candles()
            self.since_values.append(since)
            return None

    class Hold:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.HOLD)

    from core.strategy_engine import STRATEGIES, StrategyInfo
    monkeypatch.setitem(STRATEGIES, "fixture_hold", StrategyInfo("fixture_hold", Hold, (), 1, "both"))
    calls = []
    original = main.Indicators.calculate_all
    monkeypatch.setattr(main.Indicators, "calculate_all", staticmethod(lambda frame, cfg: (calls.append(1), original(frame, cfg))[1]))
    exchange = Exchange()
    engine = StrategyEngine(["fixture_hold"], ["5m"], BotConfig())
    first_now = pd.Timestamp("2026-01-01 03:24:50")
    engine.evaluate(exchange, "BTC/USDT", now=first_now)
    engine.evaluate(exchange, "BTC/USDT", now=first_now + pd.Timedelta(seconds=40))
    assert len(exchange.since_values) == 1
    assert len(calls) == 1


def test_signal_from_same_closed_bar_is_emitted_only_once(monkeypatch):
    class Long:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.LONG, 100, 98, 104, 0.8, "long")

    from core.strategy_engine import STRATEGIES, StrategyInfo
    monkeypatch.setitem(STRATEGIES, "fixture_long", StrategyInfo("fixture_long", Long, (), 1, "long"))

    class Exchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return candles()

    engine = StrategyEngine(["fixture_long"], ["5m"], BotConfig())
    now = pd.Timestamp("2026-01-01 03:20")
    first = engine.evaluate(Exchange(), "BTC/USDT", now=now)[0]
    second = engine.evaluate(Exchange(), "BTC/USDT", now=now + pd.Timedelta(seconds=1))[0]
    assert first.signal == Signal.LONG
    assert second.signal == Signal.HOLD


def test_conflicting_candidate_directions_are_rejected(monkeypatch):
    class Long:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.LONG, 100, 98, 104, 0.8, "long")

    class Short:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.SHORT, 100, 102, 96, 0.79, "short")

    monkeypatch.setitem(STRATEGIES, "fixture_long", StrategyInfo("fixture_long", Long, (), 1, "long"))
    monkeypatch.setitem(STRATEGIES, "fixture_short", StrategyInfo("fixture_short", Short, (), 1, "short"))

    class Exchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return candles()

    cfg = BotConfig()
    engine = StrategyEngine(["fixture_long", "fixture_short"], ["5m"], cfg)
    setup, _, _, _ = engine.evaluate(
        Exchange(), "BTC/USDT", now=pd.Timestamp("2026-01-01 03:20"))
    assert setup.signal == Signal.HOLD


def test_multi_selection_preview_is_paper_only_and_reports_matrix():
    cfg = main.build_config_from_args([
        "--exchange", "kucoin", "--strategies", "all", "--timeframes", "all",
        "--symbols", "BTC/USDT", "ETH/USDT", "SOL/USDT", "--capital", "10",
        "--adaptive-leverage", "--max-leverage", "3", "--max-margin-allocation-pct", "0.5",
        "--max-gross-exposure-pct", "1.5", "--max-open-positions", "1", "--show-config",
    ], environ={})
    summary = main.effective_config_summary(cfg)
    assert len(summary["SELECTED STRATEGIES"]) == 8
    assert "5m" in summary["SELECTED TIMEFRAMES"]
    assert not summary["TIMEFRAME STRATEGY COMPATIBILITY"]["mtf_trend_pullback"]["1m"]["compatible"]
    assert summary["REAL ORDER EXECUTION"] == "DISABLED"
    assert summary["EXCHANGE MARKET TYPE"] == "spot"
    assert summary["MAX LEVERAGE"] == 1.0
    assert summary["MAX MARGIN ALLOCATION"] == 5.0
    assert summary["MAX GROSS EXPOSURE"] == 15.0


def test_multi_strategy_mode_is_rejected_for_live_execution():
    import pytest
    with pytest.raises(SystemExit, match="paper-only"):
        main.build_config_from_args(
            ["live", "--strategies", "all"], environ={"CONFIRM_LIVE": "YES"})


def test_full_strategy_timeframe_matrix_is_explicit_and_keeps_genuine_combinations():
    assert len(STRATEGIES) == 8
    all_timeframes = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                      "6h", "8h", "12h", "1d", "1w", "1M")
    assert len(all_timeframes) == 14
    # Every registered strategy declares an explicit non-empty horizon set.
    for info in STRATEGIES.values():
        assert info.supported_timeframes
        assert set(info.supported_timeframes) <= set(all_timeframes)
    assert "1m" in STRATEGIES["grid_scalper"].supported_timeframes
    assert "1d" not in STRATEGIES["grid_scalper"].supported_timeframes
    assert STRATEGIES["mtf_trend_pullback"].supported_timeframes == ("5m",)


def test_unsupported_timeframe_is_skipped_with_reason_not_raised(monkeypatch):
    class Long:
        def __init__(self, config):
            pass

        def analyze(self, frame):
            return TradeSetup(Signal.LONG, 100, 98, 104, 0.8, "long")

    monkeypatch.setitem(STRATEGIES, "fixture_long", StrategyInfo(
        "fixture_long", Long, (), 1, "long", supported_timeframes=("5m",)))

    class Exchange:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return candles(40, "1h" if timeframe == "1h" else "5min")

    engine = StrategyEngine(["fixture_long"], ["5m", "1h"], BotConfig())
    setup, _, _, _ = engine.evaluate(
        Exchange(), "BTC/USDT", now=pd.Timestamp("2026-01-01 03:20"))
    assert setup.signal == Signal.LONG
    assert "incompatible" in engine.last_support[("BTC/USDT", "fixture_long", "1h")]
    assert "incompatible" in engine.last_skip_reasons["BTC/USDT"]["fixture_long@1h"]


def test_cost_filter_rejects_eth_style_tiny_target():
    cfg = BotConfig()
    cfg.backtest.taker_fee = 0.0005
    cfg.backtest.slippage = 0.0003
    cfg.execution.cost_safety_buffer_pct = 0.0002
    cfg.execution.minimum_expected_net_reward_pct = 0.0002
    setup = TradeSetup(Signal.SHORT, 2015.15, 2020.0, 2013.4702, 0.7, "grid_scalper@1m")

    estimate = estimate_reward_cost(setup, cfg)

    assert estimate["allowed"] is False
    assert estimate["gross_reward_pct"] < estimate["round_trip_cost_pct"]
    assert estimate["expected_net_reward_pct"] < 0


def test_cost_filter_rejects_target_on_wrong_side_of_entry():
    cfg = BotConfig()
    long_setup = TradeSetup(Signal.LONG, 100.0, 98.0, 99.0, 0.7, "x")
    short_setup = TradeSetup(Signal.SHORT, 100.0, 102.0, 101.0, 0.7, "x")

    assert estimate_reward_cost(long_setup, cfg)["allowed"] is False
    assert estimate_reward_cost(short_setup, cfg)["allowed"] is False


def test_cost_filter_allows_reward_that_covers_all_costs():
    cfg = BotConfig()
    setup = TradeSetup(Signal.LONG, 100.0, 98.0, 103.0, 0.7, "x")

    estimate = estimate_reward_cost(setup, cfg)

    assert estimate["allowed"] is True
    assert estimate["expected_net_reward_pct"] > 0


def test_orderbook_is_fetched_once_per_symbol_and_cycle_with_supported_limit():
    class Exchange:
        def __init__(self, book):
            self.book = book
            self.limits = []

        def fetch_ohlcv(self, symbol, timeframe, limit):
            freq = "15min" if timeframe == "15m" else "5min"
            end = pd.Timestamp("2026-01-01 10:00") - pd.Timedelta(minutes=15 if freq == "15min" else 5)
            index = pd.date_range(end=end, periods=40, freq=freq)
            close = np.linspace(100, 104, len(index))
            return pd.DataFrame({
                "open": close, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.full(len(index), 100.0),
            }, index=index)

        def fetch_orderbook(self, symbol, limit):
            self.limits.append(limit)
            return self.book

    cfg = BotConfig()
    cfg.strategy.orderbook_depth = 20
    exchange = Exchange(None)
    engine = StrategyEngine(["orderbook_imbalance"], ["5m", "15m"], cfg)

    engine.evaluate(exchange, "BTC/USDT", now=pd.Timestamp("2026-01-01 10:00"))

    assert exchange.limits == [20]
    for timeframe in ("5m", "15m"):
        assert "unavailable" in engine.last_skip_reasons["BTC/USDT"][
            f"orderbook_imbalance@{timeframe}"]


def test_strategy_performance_history_has_strict_entry_cutoff_and_isolated_label(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "DB_PATH", tmp_path / "engine.sqlite")
    state.record_trade(1.0, "BTC/USDT", "target", "2026-01-01T00:00:00",
                       strategy="ema_rsi_atr@5m", net_return=0.01, oos_validated=True)
    state.record_trade(-1.0, "BTC/USDT", "stop", "2026-01-01T00:10:00",
                       strategy="ema_rsi_atr@5m", net_return=-0.01, oos_validated=True)
    state.record_trade(2.0, "BTC/USDT", "target", "2026-01-01T00:00:00",
                       strategy="ema_rsi_atr@1h", net_return=0.02, oos_validated=True)

    rows = state.load_strategy_trades("ema_rsi_atr@5m", before="2026-01-01T00:05:00")
    performance = state.strategy_performance()
    assert len(rows) == 1
    assert performance["ema_rsi_atr@5m"]["completed"] == 2
    assert performance["ema_rsi_atr@5m"]["oos_validated"] == 2
    assert performance["ema_rsi_atr@1h"]["completed"] == 1
