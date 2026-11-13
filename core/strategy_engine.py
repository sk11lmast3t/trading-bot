"""Deterministic multi-strategy / multi-timeframe signal arbitration for paper mode."""
from dataclasses import dataclass
import math
import re
from pathlib import Path

import pandas as pd

from core.indicators import Indicators
from core import state
from core.adaptive_leverage import estimate_edge
from core.strategies import (
    CombinedStrategy, CryptoLaggardStrategy, EmaRsiAtrStrategy, GridScalperStrategy,
    KairiDisparityStrategy, MomentumScalperStrategy, MtfTrendPullbackStrategy,
    OrderbookStrategy, Signal, TradeSetup,
)

def _minutes_for_strategy(strategy_label, fallback):
    try:
        interval = str(strategy_label).rsplit("@", 1)[1]
        return timeframe_minutes(interval) or fallback
    except (IndexError, TypeError):
        return fallback


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    cls: type
    indicators: tuple[str, ...]
    min_bars: int
    direction: str
    market_data: tuple[str, ...] = ("ohlcv",)
    context: str = ""
    supported_timeframes: tuple[str, ...] = ()


STRATEGIES = {
    info.name: info for info in (
        StrategyInfo("ema_rsi_atr", EmaRsiAtrStrategy, ("ema_fast", "ema_medium", "ema_slow", "rsi", "atr", "vol_sma"), 30, "long/short",
                     supported_timeframes=("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h")),
        StrategyInfo("momentum_scalper", MomentumScalperStrategy, ("rsi", "macd_hist", "bb_upper", "bb_lower", "bb_middle", "atr"), 30, "long/short",
                     supported_timeframes=("1m", "3m", "5m", "15m", "30m")),
        StrategyInfo("grid_scalper", GridScalperStrategy, ("ema_medium", "rsi", "atr"), 30, "long/short",
                     supported_timeframes=("1m", "3m", "5m", "15m", "30m", "1h")),
        StrategyInfo("orderbook_imbalance", OrderbookStrategy, ("ema_fast", "ema_medium", "atr"), 20, "long/short", ("ohlcv", "orderbook"), "public order book required",
                     ("1m", "3m", "5m", "15m")),
        StrategyInfo("kairi_disparity", KairiDisparityStrategy, ("disparity_25", "rsi", "vol_sma"), 30, "long/short",
                     supported_timeframes=("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h")),
        StrategyInfo("crypto_laggard", CryptoLaggardStrategy, ("atr",), 20, "long", ("peer OHLCV", "BTC leader OHLCV"), "BTC leader candles required",
                     ("5m", "15m", "30m", "1h", "2h", "4h")),
        StrategyInfo("combined", CombinedStrategy, ("ema_fast", "ema_medium", "ema_slow", "rsi", "atr", "macd_hist", "bb_upper", "bb_lower", "disparity_25"), 30, "long/short", context="runs its four built-in constituent rules",
                     supported_timeframes=("1m", "3m", "5m", "15m", "30m", "1h")),
        StrategyInfo("mtf_trend_pullback", MtfTrendPullbackStrategy, ("ema_fast", "rsi", "atr"), 720, "long/short", context="5m entry bars with completed 30m and 1h context",
                     supported_timeframes=("5m",)),
    )
}


# Indicator availability alone does not make an interval appropriate for a
# scalping rule. These bounds express the rules' intended holding horizons.
# The registry above is the single source of truth; this mapping is retained
# for callers that import TIMEFRAME_RULES directly.
TIMEFRAME_RULES = {name: info.supported_timeframes for name, info in STRATEGIES.items()}


def supported_timeframes(name):
    """Return a strategy's explicit intervals, or () when it is unrestricted.

    Test doubles and third-party strategies may be registered without explicit
    rules; they are treated as compatible with every selected interval rather
    than raising KeyError.
    """
    info = STRATEGIES.get(name)
    if info is None:
        return ()
    return info.supported_timeframes or TIMEFRAME_RULES.get(name, ())


def is_timeframe_compatible(name, timeframe):
    allowed = supported_timeframes(name)
    return not allowed or timeframe in allowed


def estimate_reward_cost(setup, config):
    """Conservatively compare target distance with round-trip costs and buffer.

    A target on the wrong side of the entry is never a valid reward, and the
    expected net reward must clear both the configurable safety buffer and the
    configurable minimum. Costs are never reduced to make a trade fit.
    """
    entry, target = float(setup.entry_price or 0.0), float(setup.take_profit or 0.0)
    if not math.isfinite(entry) or not math.isfinite(target) or entry <= 0:
        return {"allowed": False, "reason": "entry/target is not a finite positive price",
                "gross_reward_pct": 0.0, "round_trip_cost_pct": math.inf,
                "expected_net_reward_pct": -math.inf}
    signal = getattr(setup, "signal", None)
    wrong_side = ((signal == Signal.LONG and target <= entry) or
                  (signal == Signal.SHORT and target >= entry))
    if wrong_side:
        return {"allowed": False, "reason": "take-profit is on the wrong side of entry",
                "gross_reward_pct": 0.0, "round_trip_cost_pct": math.inf,
                "expected_net_reward_pct": -math.inf}
    gross = abs(target - entry) / entry
    maker = config.execution.entry_mode == "maker"
    entry_fee = config.backtest.maker_fee if maker else config.backtest.taker_fee
    entry_slippage = 0.0 if maker else config.backtest.slippage
    exit_fee, exit_slippage = config.backtest.taker_fee, config.backtest.slippage
    funding = (abs(float(getattr(config.backtest, "funding_rate", 0.0) or 0.0)) *
               float(config.risk.max_hold_time_minutes) / 480.0)
    round_trip = entry_fee + entry_slippage + exit_fee + exit_slippage + funding
    buffer = float(getattr(config.execution, "cost_safety_buffer_pct", 0.0002))
    minimum = float(getattr(config.execution, "minimum_expected_net_reward_pct", 0.0002))
    net = gross - round_trip - buffer
    allowed = net >= minimum
    return {
        "allowed": allowed,
        "reason": "" if allowed else "expected net reward is below the configured minimum",
        "gross_reward_pct": gross * 100.0,
        "entry_fee_pct": entry_fee * 100.0,
        "exit_fee_pct": exit_fee * 100.0,
        "entry_slippage_pct": entry_slippage * 100.0,
        "exit_slippage_pct": exit_slippage * 100.0,
        "funding_cost_pct": funding * 100.0,
        "cost_safety_buffer_pct": buffer * 100.0,
        "round_trip_cost_pct": round_trip * 100.0,
        "expected_net_reward_pct": net * 100.0,
        "minimum_expected_net_reward_pct": minimum * 100.0,
    }


def timeframe_minutes(value):
    match = re.fullmatch(r"(\d+)([mhdwM])", str(value))
    if not match:
        return None
    n, unit = int(match.group(1)), match.group(2)
    return n * {"m": 1, "h": 60, "d": 1440, "w": 10080, "M": 43200}[unit]


def timeframe_delta(value):
    match = re.fullmatch(r"(\d+)([mhdwM])", str(value))
    if not match:
        return None
    n, unit = int(match.group(1)), match.group(2)
    if unit == "M":
        return pd.DateOffset(months=n)
    return pd.Timedelta(minutes=timeframe_minutes(value))


def completed_frame(frame, timeframe, now=None):
    """Drop candles whose close time is after now (CCXT timestamps are candle opens)."""
    if frame is None or frame.empty:
        return None
    minutes = timeframe_minutes(timeframe)
    if minutes is None:
        return None
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    out = frame.copy()
    index = pd.DatetimeIndex(out.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index
    out = out.loc[(out.index + timeframe_delta(timeframe)) <= now]
    return out if not out.empty else None


class StrategyEngine:
    """Caches completed OHLCV/indicator frames and chooses at most one setup per symbol."""

    def __init__(self, strategies, timeframes, config):
        unknown = sorted(set(strategies) - set(STRATEGIES))
        if unknown:
            raise ValueError("Unknown strategies: " + ", ".join(unknown))
        self.strategy_names = tuple(dict.fromkeys(strategies))
        self.timeframes = tuple(dict.fromkeys(timeframes))
        self.config = config
        self._cache = {}
        self._last_evaluated = {}
        self._orderbook_cache = {}
        self.last_support = {}
        self.last_rejection_reason = {}
        self.last_skip_reasons = {}

    def evaluate(self, exchange, symbol, now=None):
        now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
        data, changed = {}, False
        self.last_rejection_reason.pop(symbol, None)
        self.last_skip_reasons[symbol] = {}
        newest_update = None
        for timeframe in self.timeframes:
            for name in self.strategy_names:
                if not is_timeframe_compatible(name, timeframe):
                    self.last_support[(symbol, name, timeframe)] = "incompatible with strategy holding horizon"
                    self.last_skip_reasons[symbol][f"{name}@{timeframe}"] = self.last_support[(symbol, name, timeframe)]
        eligible_timeframes = [tf for tf in self.timeframes if any(
            is_timeframe_compatible(name, tf) for name in self.strategy_names)]
        for timeframe in eligible_timeframes:
            key = (symbol, timeframe)
            minutes = timeframe_minutes(timeframe)
            if not minutes:
                self.last_support[key] = "unsupported timeframe format"
                continue
            cached = self._cache.get(key)
            if cached and now < cached["refresh_at"]:
                if self._frame_is_fresh(cached["frame"], timeframe, now):
                    data[timeframe] = cached["frame"]
                else:
                    self.last_support[key] = "stale last completed candle"
                continue
            try:
                raw = self._fetch_raw(exchange, symbol, timeframe, cached)
                if raw is None or raw.empty:
                    if cached:
                        if self._frame_is_fresh(cached["frame"], timeframe, now):
                            data[timeframe] = cached["frame"]
                        else:
                            self.last_support[key] = "stale last completed candle; refresh failed"
                    else:
                        self.last_support[key] = "no candle data returned"
                    continue
                if cached:
                    raw = pd.concat([cached["frame"], raw]).loc[lambda x: ~x.index.duplicated(keep="last")]
                    raw = raw.iloc[-max(800, 722 if timeframe == "5m" else 300):]
                frame = completed_frame(raw, timeframe, now)
                if frame is None:
                    self.last_support[key] = "no completed candles available"
                    continue
                if not self._frame_is_fresh(frame, timeframe, now):
                    self.last_support[key] = "stale last completed candle"
                    continue
                last = frame.index[-1]
                if cached and cached["frame"].index[-1] == last:
                    ready = cached["frame"]
                else:
                    ready = Indicators.calculate_all(frame, self.config)
                    changed = True
                    if newest_update is None or last > newest_update[0]:
                        newest_update = (last, timeframe, ready)
                # Recheck shortly after the next expected candle boundary; prevents per-cycle downloads.
                boundary = last + timeframe_delta(timeframe) * 2
                refresh_at = max(now + pd.Timedelta(seconds=min(30, max(2, minutes * 6))),
                                 boundary + pd.Timedelta(seconds=2))
                self._cache[key] = {"frame": ready, "refresh_at": refresh_at}
                data[timeframe] = ready
            except Exception as exc:  # one failed timeframe must not stop other evaluations
                self.last_support[key] = f"fetch/evaluation data error: {type(exc).__name__}"

        candidates = []
        evaluated_keys = []
        orderbook_loaded = False
        orderbook_value = None
        for timeframe in eligible_timeframes:
            frame = data.get(timeframe)
            if frame is None:
                continue
            for name in self.strategy_names:
                info = STRATEGIES[name]
                key = (symbol, name, timeframe)
                if not is_timeframe_compatible(name, timeframe):
                    self.last_support[key] = "incompatible with strategy holding horizon"
                    continue
                if name == "crypto_laggard":
                    leader_symbol = self.config.strategy.laggard_leader
                    if symbol == leader_symbol:
                        self.last_support[key] = "BTC itself is the configured leader, so no peer rotation applies"
                        continue
                if len(frame) < info.min_bars:
                    self.last_support[key] = f"requires {info.min_bars} completed candles; found {len(frame)}"
                    continue
                bar = frame.index[-1]
                evaluated_keys.append((key, bar))
                if self._last_evaluated.get(key) == bar:
                    continue
                try:
                    strategy = info.cls(self.config)
                    if name == "crypto_laggard":
                        leader_frame = self._fetch_frame(exchange, leader_symbol, timeframe, now)
                        if leader_frame is None or len(leader_frame) < info.min_bars:
                            self.last_support[key] = "insufficient completed BTC leader candles"
                            continue
                        setup = strategy.analyze_with_leader(frame, leader_frame)
                    elif name == "orderbook_imbalance":
                        if not orderbook_loaded:
                            fetch_book = getattr(exchange, "fetch_orderbook", None)
                            orderbook_value = fetch_book(symbol, self.config.strategy.orderbook_depth) if callable(fetch_book) else None
                            orderbook_loaded = True
                        book = orderbook_value
                        if not book:
                            self.last_support[key] = "public order-book data unavailable"
                            self.last_rejection_reason[symbol] = "orderbook_imbalance skipped: public order-book data unavailable"
                            self.last_skip_reasons[symbol][f"{name}@{timeframe}"] = self.last_support[key]
                            continue
                        setup = strategy.analyze_with_orderbook(frame, book)
                    else:
                        setup = strategy.analyze(frame)
                    if setup.signal in (Signal.LONG, Signal.SHORT):
                        setup.strategy_name = f"{name}@{timeframe}"
                        costs = estimate_reward_cost(setup, self.config)
                        if costs["allowed"]:
                            candidates.append((setup, timeframe, frame))
                        else:
                            self.last_rejection_reason[symbol] = (
                                f"cost-aware filter: net reward {costs['expected_net_reward_pct']:.4f}% "
                                f"below minimum {costs['minimum_expected_net_reward_pct']:.4f}%"
                            )
                            self.last_skip_reasons[symbol][setup.strategy_name] = self.last_rejection_reason[symbol]
                    self.last_support[key] = "evaluated"
                except Exception as exc:
                    self.last_support[key] = f"strategy error: {type(exc).__name__}"
        for key, bar in evaluated_keys:
            self._last_evaluated[key] = bar
        if not candidates:
            if newest_update is None:
                return TradeSetup(Signal.HOLD), None, None, None
            bar, timeframe, frame = newest_update
            return TradeSetup(Signal.HOLD), frame, timeframe, bar

        ranked = [(self._score(item[0], item[2].index[-1]), item) for item in candidates]
        ranked.sort(key=lambda row: row[0], reverse=True)
        best_score, (best, timeframe, frame) = ranked[0]
        if not math.isfinite(best_score):
            return TradeSetup(Signal.HOLD), frame, timeframe, frame.index[-1]
        direction = "long" if best.signal == Signal.LONG else "short"
        higher_frames = sorted(
            (tf for tf in data if timeframe_minutes(tf) and timeframe_minutes(tf) > timeframe_minutes(timeframe)),
            key=timeframe_minutes,
        )
        confirmations = 0
        opposing_trend = 0
        for higher_tf in higher_frames:
            higher = data[higher_tf]
            if len(higher) < 22 or "ema_medium" not in higher:
                continue
            close = float(higher["close"].iloc[-1])
            ema_now = float(higher["ema_medium"].iloc[-1])
            ema_before = float(higher["ema_medium"].iloc[-2])
            trend = "long" if close > ema_now and ema_now >= ema_before else (
                "short" if close < ema_now and ema_now <= ema_before else "neutral")
            confirmations += int(trend == direction)
            opposing_trend += int(trend not in (direction, "neutral"))
        required_confirmations = int(getattr(self.config, "min_timeframe_confirmations", 0))
        if confirmations < required_confirmations:
            return TradeSetup(Signal.HOLD), frame, timeframe, frame.index[-1]
        if opposing_trend:
            best.confidence = max(0.0, best.confidence - min(0.20, 0.05 * opposing_trend))
        opposing = [item for item in candidates[1:] if item[0].signal != best.signal]
        agreeing = [item for item in candidates if item[0].signal == best.signal]
        if opposing:
            strongest_opposition = max(item[0].confidence for item in opposing)
            if strongest_opposition >= best.confidence * 0.85:
                return TradeSetup(Signal.HOLD), frame, timeframe, frame.index[-1]
            best.confidence = max(0.0, best.confidence - 0.10 * len(opposing))
        # Agreement is a score input only; it never modifies leverage.
        best.confidence = min(0.99, best.confidence + min(0.10, 0.025 * (len(agreeing) - 1)))
        return best, frame, timeframe, frame.index[-1]

    def _fetch_frame(self, exchange, symbol, timeframe, now):
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
        key = (symbol, timeframe)
        cached = self._cache.get(key)
        if cached and now < cached["refresh_at"]:
            return cached["frame"] if self._frame_is_fresh(cached["frame"], timeframe, now) else None
        raw = self._fetch_raw(exchange, symbol, timeframe, cached)
        if raw is None or raw.empty:
            return cached["frame"] if cached else None
        if cached:
            raw = pd.concat([cached["frame"], raw]).loc[lambda x: ~x.index.duplicated(keep="last")].iloc[-300:]
        frame = completed_frame(raw, timeframe, now)
        if frame is None or not self._frame_is_fresh(frame, timeframe, now):
            return None
        ready = Indicators.calculate_all(frame, self.config)
        mins = timeframe_minutes(timeframe)
        self._cache[key] = {
            "frame": ready,
            "refresh_at": max(now + pd.Timedelta(seconds=min(30, max(2, mins * 6))),
                              frame.index[-1] + timeframe_delta(timeframe) * 2 + pd.Timedelta(seconds=2)),
        }
        return ready

    @staticmethod
    def _frame_is_fresh(frame, timeframe, now):
        if frame is None or frame.empty:
            return False
        delta = timeframe_delta(timeframe)
        age = pd.Timestamp(now) - (pd.Timestamp(frame.index[-1]) + delta)
        maximum_age = pd.Timedelta(minutes=max(timeframe_minutes(timeframe) * 2, 5))
        return age <= maximum_age

    def _fetch_raw(self, exchange, symbol, timeframe, cached):
        if cached:
            since = int(pd.Timestamp(cached["frame"].index[-1]).timestamp() * 1000) + 1
            try:
                return exchange.fetch_ohlcv(symbol, timeframe, 200, since=since)
            except TypeError:
                # Compatibility with existing test doubles and older exchange wrappers.
                return exchange.fetch_ohlcv(symbol, timeframe, 800)
        return exchange.fetch_ohlcv(symbol, timeframe, 800)

    def _score(self, setup, entry_time):
        entry, stop, target = map(float, (setup.entry_price, setup.stop_loss, setup.take_profit))
        risk = abs(entry - stop) / entry if entry > 0 else math.inf
        reward = abs(target - entry) / entry if entry > 0 else 0.0
        rr = reward / risk if risk > 0 and math.isfinite(risk) else 0.0
        round_trip_cost = 2 * (float(self.config.backtest.taker_fee) + float(self.config.backtest.slippage))
        net_reward = max(0.0, reward - round_trip_cost)
        score = float(setup.confidence) * min(net_reward / risk if risk else 0.0, 5.0) * min(rr, 5.0)
        # Historical preference only uses pre-entry, out-of-sample trade outcomes.
        history = (state.load_strategy_trades(setup.strategy_name, 500, before=entry_time)
                   if Path(state.DB_PATH).exists() else [])
        estimate = estimate_edge(history)
        if estimate.sample_size >= 100 and estimate.expected_value is not None:
            if estimate.expected_value <= 0:
                return -math.inf
            score *= 1.0 + min(estimate.expected_value, 0.05)
        return score


def resolve_selection(strategy_arg, strategies_arg, timeframe_arg, timeframes_arg, supported):
    strategies = list(STRATEGIES)
    selected_strategies = strategies if strategies_arg == "all" or strategies_arg == ["all"] else list(strategies_arg or [])
    if strategies_arg is None:
        selected_strategies = [strategy_arg or "ema_rsi_atr"]
    selected_timeframes = list(supported) if timeframes_arg == ["all"] else list(timeframes_arg or [])
    if not selected_timeframes:
        selected_timeframes = [timeframe_arg or "5m"]
    valid = set(supported)
    warnings = [f"unsupported timeframe: {tf}" for tf in selected_timeframes if tf not in valid]
    return selected_strategies, [tf for tf in selected_timeframes if tf in valid], warnings
