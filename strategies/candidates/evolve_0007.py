import numpy as np
import pandas as pd
from core.indicators import Indicators

class Candidate(BaseStrategy):
    lookback_bars = 200
    FAST_EMA = 20
    SLOW_EMA = 50
    RSI_PERIOD = 14
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    STOP_PCT = 0.02
    TARGET_PCT = 0.06

    def analyze(self, df):
        if len(df) < self.lookback_bars:
            return TradeSetup(Signal.HOLD)

        close = df['close']
        high = df['high']
        low = df['low']

        fast_ema = Indicators.ema(close, self.FAST_EMA)
        slow_ema = Indicators.ema(close, self.SLOW_EMA)
        rsi = Indicators.rsi(close, self.RSI_PERIOD)

        fast_curr = fast_ema.iloc[-1]
        fast_prev = fast_ema.iloc[-2]
        slow_curr = slow_ema.iloc[-1]
        slow_prev = slow_ema.iloc[-2]
        rsi_curr = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]

        price = float(close.iloc[-1])

        trend_up = fast_curr > slow_curr and fast_prev > slow_prev
        trend_down = fast_curr < slow_curr and fast_prev < slow_prev

        rsi_rising = rsi_curr > rsi_prev
        rsi_falling = rsi_curr < rsi_prev

        if trend_up and rsi_curr < self.RSI_OVERSOLD and rsi_rising:
            stop_loss = price * (1 - self.STOP_PCT)
            take_profit = price * (1 + self.TARGET_PCT)
            return TradeSetup(Signal.LONG, price, stop_loss, take_profit, 0.7, "evolve_0007")

        if trend_down and rsi_curr > self.RSI_OVERBOUGHT and rsi_falling:
            stop_loss = price * (1 + self.STOP_PCT)
            take_profit = price * (1 - self.TARGET_PCT)
            return TradeSetup(Signal.SHORT, price, stop_loss, take_profit, 0.7, "evolve_0007")

        return TradeSetup(Signal.HOLD)
