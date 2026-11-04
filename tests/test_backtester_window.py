import pandas as pd
import numpy as np
from core.indicators import Indicators
from core.strategies import get_strategy
from config.settings import BotConfig as Config

def test_signals_equal_window():
    cfg = Config()
    strat = get_strategy(cfg)
    df = pd.DataFrame({
        "open": np.random.rand(500)*10+100,
        "high": np.random.rand(500)*10+105,
        "low": np.random.rand(500)*10+95,
        "close": np.random.rand(500)*10+100,
        "volume": np.random.rand(500)*1000,
    })
    df.index = pd.date_range("2024-01-01", periods=500, freq="1min")
    df = Indicators.calculate_all(df, cfg)
    full = strat.analyze(df)
    bounded = strat.analyze(df.iloc[-300:].copy())
    assert full.signal == bounded.signal
