import numpy as np
import pandas as pd
from core.indicators import Indicators

class Candidate(BaseStrategy):
    lookback_bars = 100
    
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MA_PERIOD = 20
    ATR_PERIOD = 14
    STOP_PCT = 0.02
    TARGET_PCT = 0.05
    
    def analyze(self, df):
        if len(df) < max(self.RSI_PERIOD, self.MA_PERIOD, self.ATR_PERIOD) + 5:
            return TradeSetup(Signal.HOLD)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        rsi = Indicators.rsi(close, self.RSI_PERIOD)
        ma = Indicators.sma(close, self.MA_PERIOD)
        atr = Indicators.atr(high, low, close, self.ATR_PERIOD)
        
        current_rsi = rsi[-1]
        current_ma = ma[-1]
        current_atr = atr[-1]
        current_price = close[-1]
        prev_price = close[-2]
        
        if np.isnan(current_rsi) or np.isnan(current_ma) or np.isnan(current_atr):
            return TradeSetup(Signal.HOLD)
        
        trend_up = current_price > current_ma and prev_price > ma[-2]
        
        if trend_up and current_rsi < self.RSI_OVERSOLD:
            entry_price = current_price
            stop_loss = entry_price * (1 - self.STOP_PCT)
            take_profit = entry_price * (1 + self.TARGET_PCT)
            
            if take_profit - entry_price < entry_price * self.TARGET_PCT:
                take_profit = entry_price * (1 + self.TARGET_PCT)
            
            confidence = min(0.8, (self.RSI_OVERSOLD - current_rsi) / self.RSI_OVERSOLD + 0.3)
            
            return TradeSetup(
                Signal.LONG,
                entry_price,
                stop_loss,
                take_profit,
                confidence,
                "evolve_0005"
            )
        
        return TradeSetup(Signal.HOLD)
