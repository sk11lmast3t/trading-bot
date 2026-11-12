"""
Unified Strategies Engine
Includes:
1. EmaRsiAtrStrategy (Trend following)
2. MomentumScalperStrategy (Mean reversion Bollinger Bands + MACD)
3. GridScalperStrategy (Passivbot DCA Grid around EMA)
4. OrderbookStrategy (Hummingbot L2 Imbalance pressure)
5. KairiDisparityStrategy (Takashi Kotegawa / BNF 25-Period Panic Disparity Reversion)
6. CryptoLaggardStrategy (BNF Sector Laggard Rotation)
7. CombinedStrategy (Ensemble Voting Consensus)
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from enum import Enum
from core.indicators import Indicators


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


class TradeSetup:
    def __init__(self, signal: Signal, entry_price: float = 0.0,
                 stop_loss: float = 0.0, take_profit: float = 0.0,
                 confidence: float = 0.0, strategy_name: str = ""):
        self.signal = signal
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.strategy_name = strategy_name


class BaseStrategy:
    def __init__(self, config):
        self.config = config
        self.name = "base"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        raise NotImplementedError


class EmaRsiAtrStrategy(BaseStrategy):
    """Triple EMA + RSI + ATR Trend Scalping
    Fires on sustained trend alignment (no crossover requirement).
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "ema_rsi_atr"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        last = df.iloc[-1]
        price = last['close']
        atr = last.get('atr', price * 0.01)

        ema_fast = last['ema_fast']
        ema_medium = last['ema_medium']
        ema_slow = last['ema_slow']
        rsi = last['rsi']
        vol_ok = last['volume'] > (last['vol_sma'] * c.min_volume_multiplier) if 'vol_sma' in last and 'volume' in last else True

        # LONG: Fast > Med > Slow + RSI in widened bullish band + Vol
        if (ema_fast > ema_medium > ema_slow and vol_ok and
                rsi >= 35 and rsi <= 65):
            sl = price - (atr * c.atr_multiplier * 0.8)
            tp = price + (atr * c.atr_multiplier * 2.5)
            confidence = min(0.95, 0.6 + (0.15 if rsi < 55 else 0))
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # SHORT: Fast < Med < Slow + RSI in widened bearish band
        if (ema_fast < ema_medium < ema_slow and vol_ok and
                rsi >= 35 and rsi <= 65):
            sl = price + (atr * c.atr_multiplier * 0.8)
            tp = price - (atr * c.atr_multiplier * 2.5)
            confidence = min(0.95, 0.6 + (0.15 if rsi > 45 else 0))
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)


class MomentumScalperStrategy(BaseStrategy):
    """Mean Reversion Bollinger Bands + MACD Momentum Scalper"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "momentum_scalper"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last['close']
        atr = last.get('atr', price * 0.01)

        rsi = last['rsi']
        macd_hist = last['macd_hist']
        bb_upper = last['bb_upper']
        bb_lower = last['bb_lower']
        bb_middle = last['bb_middle']

        # LONG: Near lower BB + RSI oversold zone + MACD turning up
        if (price <= bb_lower * 1.005 and rsi < 35 and macd_hist > prev['macd_hist']):
            sl = price - (atr * 1.5)
            tp = bb_middle * 1.002
            confidence = 0.85 if rsi < c.rsi_oversold else 0.65
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # SHORT: Near upper BB + RSI overbought zone + MACD turning down
        if (price >= bb_upper * 0.995 and rsi > 65 and macd_hist < prev['macd_hist']):
            sl = price + (atr * 1.5)
            tp = bb_middle * 0.998
            confidence = 0.85 if rsi > c.rsi_overbought else 0.65
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)


class GridScalperStrategy(BaseStrategy):
    """Passivbot-Style Grid Scalper around EMA
    Uses ATR proximity band for mean-reversion entries.
    No strict trend gate — instead uses RSI confirmation and
    trend-aware TP adjustment to stay profitable in trends.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "grid_scalper"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        last = df.iloc[-1]
        price = last["close"]
        atr = last.get("atr", price * 0.01)
        ema = last["ema_medium"]
        rsi = last.get("rsi", 50)

        # Distance from EMA as ratio of ATR
        dist = abs(price - ema) / atr if atr > 0 else 0

        # Only trade in mean-reversion zone: within 2.0 ATR of EMA
        # but at least 0.3 ATR away (avoid noise right at EMA)
        if dist < 0.3 or dist > 2.0:
            return TradeSetup(Signal.HOLD)

        # Confidence scales with distance from EMA (farther = stronger reversion)
        confidence = min(0.85, 0.55 + dist * 0.15)

        # Grid LONG: price below EMA + RSI not extreme overbought
        if price < ema and rsi < 60:
            sl = price - min(atr * 2, price * 0.015)
            tp = ema + (atr * 1.0)  # target slightly past EMA
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # Grid SHORT: price above EMA + RSI not extreme oversold
        if price > ema and rsi > 40:
            sl = price + min(atr * 2, price * 0.015)
            tp = ema - (atr * 1.0)  # target slightly past EMA
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)


class OrderbookStrategy(BaseStrategy):
    """Hummingbot Orderbook Depth Imbalance Strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "orderbook_imbalance"

    def analyze_with_orderbook(self, df: pd.DataFrame, orderbook: dict) -> TradeSetup:
        if len(df) < 20 or not orderbook:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        last = df.iloc[-1]
        price = last['close']
        atr = last.get('atr', price * 0.01)

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        imbalance = Indicators.orderbook_imbalance(bids, asks, c.orderbook_depth)
        ema_trend = last['ema_fast'] > last['ema_medium']

        if imbalance > c.imbalance_threshold and ema_trend:
            sl = price - (atr * 1.5)
            tp = price + (atr * 2.0)
            confidence = min(0.88, 0.5 + abs(imbalance) * 0.45)
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        if imbalance < -c.imbalance_threshold and not ema_trend:
            sl = price + (atr * 1.5)
            tp = price - (atr * 2.0)
            confidence = min(0.88, 0.5 + abs(imbalance) * 0.45)
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        return TradeSetup(Signal.HOLD)


class KairiDisparityStrategy(BaseStrategy):
    """
    Takashi Kotegawa (BNF) 25-Period Disparity (Kairi-ritsu) Strategy
    Exploits irrational retail panic sell-offs and liquidations to capture sharp 3% to 7% relief bounces.
    Never averages down, enforces rigid stops, and exits quickly into the relief bounce.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "kairi_disparity"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        last = df.iloc[-1]
        price = last['close']
        disparity = last.get('disparity_25', 0.0)

        # Dynamic regime thresholding
        is_crash = disparity <= c.disparity_crash_threshold
        threshold = c.disparity_crash_threshold if is_crash else c.disparity_panic_threshold

        vol_spike = last['volume'] > (last['vol_sma'] * 1.25) if 'vol_sma' in last and 'volume' in last else True
        rsi_oversold = last.get('rsi', 50) < 35

        # PANIC BUY: Deep negative disparity + seller exhaustion signals
        if disparity <= threshold and (vol_spike or rsi_oversold):
            sl = price * (1 - c.disparity_sl_pct * 0.8)
            tp = price * (1 + c.disparity_tp_pct * 1.2)
            confidence = min(0.98, 0.75 + abs(disparity) * 0.025)
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # OVERHEATED BLOW-OFF TOP: Extreme positive disparity
        if disparity >= abs(threshold) * 1.5 and last.get('rsi', 50) > 75:
            sl = price * (1 + c.disparity_sl_pct)
            tp = price * (1 - c.disparity_tp_pct)
            confidence = min(0.88, 0.65 + disparity * 0.015)
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)


class CryptoLaggardStrategy(BaseStrategy):
    """
    BNF Sector Laggard Rotation Strategy (Junkan Busshoku)
    In trending environments, detects when the primary market leader (BTC/ETH) has surged
    while the correlated altcoin peer remains flat, anticipating capital rotation.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "crypto_laggard"

    def analyze_with_leader(self, df_peer: pd.DataFrame, df_leader: pd.DataFrame) -> TradeSetup:
        if len(df_peer) < 20 or len(df_leader) < 20:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        lookback = c.laggard_lookback
        leader_ret = (df_leader['close'].iloc[-1] / df_leader['close'].iloc[-lookback]) - 1.0
        peer_ret = (df_peer['close'].iloc[-1] / df_peer['close'].iloc[-lookback]) - 1.0

        price = df_peer['close'].iloc[-1]
        atr = df_peer.get('atr', price * 0.01).iloc[-1]

        if leader_ret >= c.laggard_leader_surge and abs(peer_ret) <= c.laggard_peer_max_move:
            sl = price - (atr * 1.5)
            tp = price + (atr * 3.0)
            confidence = min(0.90, 0.65 + leader_ret * 5.0)
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        return TradeSetup(Signal.HOLD)


class CombinedStrategy(BaseStrategy):
    """
    Consensus Voting Ensemble
    Executes when 2 or more strategies agree on direction, or a high-conviction panic/disparity setup triggers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "combined"
        self.strategies = [
            EmaRsiAtrStrategy(config),
            MomentumScalperStrategy(config),
            GridScalperStrategy(config),
            KairiDisparityStrategy(config),
        ]

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        signals = []
        for strategy in self.strategies:
            setup = strategy.analyze(df)
            if setup.signal != Signal.HOLD:
                signals.append(setup)

        if not signals:
            return TradeSetup(Signal.HOLD)

        long_signals = [s for s in signals if s.signal == Signal.LONG]
        short_signals = [s for s in signals if s.signal == Signal.SHORT]

        # BNF Panic Override: If KairiDisparity fires with high confidence, grant immediate execution
        disparity_signals = [s for s in long_signals if s.strategy_name == "kairi_disparity"]
        if disparity_signals and disparity_signals[0].confidence >= 0.85:
            return disparity_signals[0]

        # Multi-strategy consensus
        if len(long_signals) >= 2:
            best = max(long_signals, key=lambda x: x.confidence)
            avg_conf = sum(s.confidence for s in long_signals) / len(long_signals)
            best.confidence = min(0.96, avg_conf + 0.1)
            best.strategy_name = f"combined({','.join(s.strategy_name for s in long_signals)})"
            return best

        if len(short_signals) >= 2:
            best = max(short_signals, key=lambda x: x.confidence)
            avg_conf = sum(s.confidence for s in short_signals) / len(short_signals)
            best.confidence = min(0.96, avg_conf + 0.1)
            best.strategy_name = f"combined({','.join(s.strategy_name for s in short_signals)})"
            return best

        # High confidence single signal
        best_single = max(signals, key=lambda x: x.confidence)
        if best_single.confidence >= 0.65:
            return best_single

        return TradeSetup(Signal.HOLD)


def completed_bars(window: pd.DataFrame, rule: str, minutes: int, base_minutes: int = 5) -> pd.DataFrame:
    """Resample 5m bars (indexed by OPEN time) to `rule`, keeping only COMPLETED higher-timeframe bars.

    The newest bucket is dropped unless the window's last 5m bar is the final one of that bucket.
    Keeping a partial bucket would leak the future into the trend filter.
    """
    ohlc = window.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(ohlc) == 0:
        return ohlc
    bucket_end = ohlc.index[-1] + pd.Timedelta(minutes=minutes)
    if window.index[-1] + pd.Timedelta(minutes=base_minutes) < bucket_end:
        ohlc = ohlc.iloc[:-1]
    return ohlc


class MtfTrendPullbackStrategy(BaseStrategy):
    """30m + 1h trend bias, 5m pullback-and-resume entry. ONE fixed rule set (pre-registered, not tuned).

    Trend (completed bars only): LONG bias when the 1h close is above its EMA(20), the 1h EMA(20) is
    rising over 3 bars, and the 30m close is above its EMA(20). SHORT bias is the mirror image.
    Entry (5m): LONG when RSI fell to <= 40 within the last 6 bars and price closes back above the 5m
    EMA(fast) after closing at/below it; SHORT is the mirror (RSI >= 60, close back below the EMA).
    Exits: stop 1.5 x ATR(14) of the 1h bars, target 3 x that (2R); the time exit comes from the
    backtester (use the `name@MINUTES` label in the report script).
    """
    lookback_bars = 720            # 60h of 5m bars: enough for EMA(20)/ATR(14) on 1h bars

    HTF_EMA = 20
    SLOPE_BARS = 3
    ATR_N = 14
    SL_ATR = 1.5
    TP_ATR = 3.0
    PULLBACK_BARS = 6
    PULLBACK_RSI_LONG = 40.0
    PULLBACK_RSI_SHORT = 60.0
    MIN_HTF_BARS = 25

    def __init__(self, config):
        super().__init__(config)
        self.name = "mtf_trend_pullback"

    def _trigger(self, df):
        """Cheap 5m check first, so the resampling only runs on the few bars that could trade."""
        if len(df) < self.PULLBACK_BARS + 2:
            return None
        last, prev = df.iloc[-1], df.iloc[-2]
        rsi = df["rsi"].iloc[-self.PULLBACK_BARS:]
        if rsi.isna().any() or pd.isna(last["ema_fast"]) or pd.isna(prev["ema_fast"]):
            return None
        if rsi.min() <= self.PULLBACK_RSI_LONG and last["close"] > last["ema_fast"] \
                and prev["close"] <= prev["ema_fast"]:
            return "long"
        if rsi.max() >= self.PULLBACK_RSI_SHORT and last["close"] < last["ema_fast"] \
                and prev["close"] >= prev["ema_fast"]:
            return "short"
        return None

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        side = self._trigger(df)
        if side is None:
            return TradeSetup(Signal.HOLD)
        h1 = completed_bars(df, "1h", 60)
        h30 = completed_bars(df, "30min", 30)
        if len(h1) < self.MIN_HTF_BARS or len(h30) < self.MIN_HTF_BARS:
            return TradeSetup(Signal.HOLD)
        ema1 = Indicators.ema(h1["close"], self.HTF_EMA)
        ema30 = Indicators.ema(h30["close"], self.HTF_EMA)
        rising = ema1.iloc[-1] > ema1.iloc[-1 - self.SLOPE_BARS]
        falling = ema1.iloc[-1] < ema1.iloc[-1 - self.SLOPE_BARS]
        up = h1["close"].iloc[-1] > ema1.iloc[-1] and rising and h30["close"].iloc[-1] > ema30.iloc[-1]
        down = h1["close"].iloc[-1] < ema1.iloc[-1] and falling and h30["close"].iloc[-1] < ema30.iloc[-1]
        atr = float(Indicators.atr(h1["high"], h1["low"], h1["close"], self.ATR_N).iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return TradeSetup(Signal.HOLD)
        price = float(df["close"].iloc[-1])
        if side == "long" and up:
            return TradeSetup(Signal.LONG, price, price - self.SL_ATR * atr,
                              price + self.TP_ATR * atr, 0.7, self.name)
        if side == "short" and down:
            return TradeSetup(Signal.SHORT, price, price + self.SL_ATR * atr,
                              price - self.TP_ATR * atr, 0.7, self.name)
        return TradeSetup(Signal.HOLD)


def get_strategy(config) -> BaseStrategy:
    name = config.strategy.active_strategy
    if name.startswith("candidate:"):
        from evolve.evaluate import load_candidate
        candidate = name.split(":", 1)[1]
        with open("config/paper_candidates.json", encoding="utf-8") as stream:
            registered = __import__("json").load(stream).get("paper_only", [])
        if config.mode != "paper" or candidate not in registered:
            raise ValueError("Candidate strategies are registered for paper mode only")
        return load_candidate(candidate, config)[0]
    strategies = {
        "mtf_trend_pullback": MtfTrendPullbackStrategy,
        "ema_rsi_atr": EmaRsiAtrStrategy,
        "momentum_scalper": MomentumScalperStrategy,
        "grid_scalper": GridScalperStrategy,
        "orderbook_imbalance": OrderbookStrategy,
        "kairi_disparity": KairiDisparityStrategy,
        "crypto_laggard": CryptoLaggardStrategy,
        "combined": CombinedStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")
    return strategies[name](config)



