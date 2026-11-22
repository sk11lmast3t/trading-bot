import numpy as np
import pandas as pd
from core.indicators import Indicators

class Candidate(BaseStrategy):
    lookback_bars = 100
    EMA_FAST = 20
    EMA_SLOW = 50
    RSI_PERIOD = 14
    RSI_PULLBACK = 40
    VOLUME_MULT = 1.5
    STOP_PCT = 0.02
    TARGET_PCT = 0.05

    def analyze(self, df):
        if len(df) < self.lookback_bars:
            return TradeSetup(Signal.HOLD)

        close = pd.Series(df['close'].values)
        high = pd.Series(df['high'].values)
        low = pd.Series(df['low'].values)
        volume = pd.Series(df['volume'].values)

        ema_fast = Indicators.ema(close, self.EMA_FAST)
        ema_slow = Indicators.ema(close, self.EMA_SLOW)
        rsi = Indicators.rsi(close, self.RSI_PERIOD)
        avg_volume = Indicators.sma(volume, 20)

        if len(ema_fast) < 2 or len(ema_slow) < 2 or len(rsi) < 2 or len(avg_volume) < 2:
            return TradeSetup(Signal.HOLD)

        curr_close = close.iloc[-1]
        curr_ema_fast = ema_fast.iloc[-1]
        curr_ema_slow = ema_slow.iloc[-1]
        prev_ema_fast = ema_fast.iloc[-2]
        prev_ema_slow = ema_slow.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        curr_volume = volume.iloc[-1]
        curr_avg_vol = avg_volume.iloc[-1]

        if curr_avg_vol <= 0:
            return TradeSetup(Signal.HOLD)

        uptrend = curr_ema_fast > curr_ema_slow and prev_ema_fast > prev_ema_slow
        pullback = curr_rsi < self.RSI_PULLBACK and prev_rsi >= self.RSI_PULLBACK
        volume_spike = curr_volume > self.VOLUME_MULT * curr_avg_vol
        price_near_fast = abs(curr_close - curr_ema_fast) / curr_close < 0.01

        if uptrend and pullback and volume_spike and price_near_fast:
            entry = curr_close
            stop = entry * (1 - self.STOP_PCT)
            target = entry * (1 + self.TARGET_PCT)
            return TradeSetup(Signal.LONG, entry, stop, target, 0.7, "evolve_0010")

        return TradeSetup(Signal.HOLD)
