import numpy as np
import pandas as pd
from core.indicators import Indicators

class Candidate(BaseStrategy):
    lookback_bars = 100
    FAST_MA = 20
    SLOW_MA = 50
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    STOP_PCT = 0.02
    TARGET_PCT = 0.05

    def analyze(self, df):
        if len(df) < self.SLOW_MA + 5:
            return TradeSetup(Signal.HOLD)

        close = df['close']
        high = df['high']
        low = df['low']

        fast_ma = Indicators.sma(close, self.FAST_MA)
        slow_ma = Indicators.sma(close, self.SLOW_MA)
        rsi = Indicators.rsi(close, self.RSI_PERIOD)

        if len(fast_ma) < 2 or len(slow_ma) < 2 or len(rsi) < 2:
            return TradeSetup(Signal.HOLD)

        fast_curr = fast_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_curr = slow_ma.iloc[-1]
        slow_prev = slow_ma.iloc[-2]
        rsi_curr = rsi.iloc[-1]

        entry_price = float(close.iloc[-1])
        stop_loss = entry_price * (1 - self.STOP_PCT)
        take_profit = entry_price * (1 + self.TARGET_PCT)

        if fast_prev <= slow_prev and fast_curr > slow_curr and rsi_curr < self.RSI_OVERBOUGHT:
            return TradeSetup(Signal.LONG, entry_price, stop_loss, take_profit, 0.7, "evolve_0009")

        return TradeSetup(Signal.HOLD)
