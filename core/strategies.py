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
    """Triple EMA + RSI + ATR Trend Scalping"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "ema_rsi_atr"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        c = self.config.strategy
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last['close']
        atr = last.get('atr', price * 0.01)

        ema_fast = last['ema_fast']
        ema_medium = last['ema_medium']
        ema_slow = last['ema_slow']
        rsi = last['rsi']
        vol_ok = last['volume'] > (last['vol_sma'] * c.min_volume_multiplier) if 'vol_sma' in last and 'volume' in last else True

        # LONG: Fast > Med > Slow + RSI in bullish scalp band + Vol confirmation
        if (ema_fast > ema_medium > ema_slow and vol_ok and
                rsi >= 40 and rsi <= 60 and
                prev['ema_fast'] <= prev['ema_medium']):
            sl = price - (atr * c.atr_multiplier * 0.8)
            tp = price + (atr * c.atr_multiplier * 2.5)
            confidence = min(0.95, 0.6 + (0.15 if rsi < 50 else 0))
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # SHORT: Fast < Med < Slow + RSI in bearish scalp band
        if (ema_fast < ema_medium < ema_slow and vol_ok and
                rsi >= 40 and rsi <= 60 and
                prev['ema_fast'] >= prev['ema_medium']):
            sl = price + (atr * c.atr_multiplier * 0.8)
            tp = price - (atr * c.atr_multiplier * 2.5)
            confidence = min(0.95, 0.6 + (0.15 if rsi > 50 else 0))
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

        # LONG: Bounce off lower BB + RSI oversold + MACD turning up
        if (price <= bb_lower * 1.002 and rsi < 30 and macd_hist > prev['macd_hist']):
            sl = price - (atr * 1.5)
            tp = bb_middle * 1.002
            confidence = 0.85 if rsi < c.rsi_oversold else 0.65
            return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # SHORT: Reject from upper BB + RSI overbought + MACD turning down
        if (price >= bb_upper * 0.998 and rsi > 70 and macd_hist < prev['macd_hist']):
            sl = price + (atr * 1.5)
            tp = bb_middle * 0.998
            confidence = 0.85 if rsi > c.rsi_overbought else 0.65
            return TradeSetup(Signal.SHORT, price, sl, tp, confidence, self.name)

        return TradeSetup(Signal.HOLD)


class GridScalperStrategy(BaseStrategy):
    """Passivbot-Style Grid Scalper around EMA"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "grid_scalper"

    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        if len(df) < 30:
            return TradeSetup(Signal.HOLD)
        last = df.iloc[-1]
        price = last['close']
        atr = last.get('atr', price * 0.01)
        sc = self.config.scalping
        ema = last['ema_medium']
        rsi = last['rsi']
        ema_fast = last['ema_fast']
        ema_medium = last['ema_medium']

        # Grid LONG: Price below EMA, only in uptrend (fast > medium)
        if price < ema * (1 - sc.grid_spacing_pct * 1.5) and ema_fast > ema_medium:
            levels = min(sc.grid_levels, int((ema - price) / (price * sc.grid_spacing_pct)))
            if levels >= 1 and rsi < 45:
                sl = price - (atr * 2)
                tp = ema
                confidence = min(0.85, 0.5 + (levels * 0.15))
                return TradeSetup(Signal.LONG, price, sl, tp, confidence, self.name)

        # Grid SHORT: Price above EMA, only in downtrend (fast < medium)
        if price > ema * (1 + sc.grid_spacing_pct * 1.5) and ema_fast < ema_medium:
            levels = min(sc.grid_levels, int((price - ema) / (price * sc.grid_spacing_pct)))
            if levels >= 1 and rsi > 55:
                sl = price + (atr * 2)
                tp = ema
                confidence = min(0.85, 0.5 + (levels * 0.15))
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
        if best_single.confidence >= 0.85:
            return best_single

        return TradeSetup(Signal.HOLD)


def get_strategy(config) -> BaseStrategy:
    strategies = {
        "ema_rsi_atr": EmaRsiAtrStrategy,
        "momentum_scalper": MomentumScalperStrategy,
        "grid_scalper": GridScalperStrategy,
        "orderbook_imbalance": OrderbookStrategy,
        "kairi_disparity": KairiDisparityStrategy,
        "crypto_laggard": CryptoLaggardStrategy,
        "combined": CombinedStrategy,
    }
    name = config.strategy.active_strategy
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")
    return strategies[name](config)
