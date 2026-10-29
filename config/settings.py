"""
Unified Crypto Scalping & Mean-Reversion Bot - Settings
Fuses Freqtrade, Hummingbot, Passivbot, Jesse AI, and the Takashi Kotegawa (BNF) Disparity Strategy.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExchangeConfig:
    name: str = os.getenv("EXCHANGE_NAME", "binance")
    api_key: str = os.getenv("API_KEY", "")
    api_secret: str = os.getenv("API_SECRET", "")
    testnet: bool = True
    demo: bool = os.getenv("DEMO", "false").lower() == "true"
    rate_limit: int = 1200


@dataclass
class ScalpingConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe: str = "5m"
    higher_timeframe: str = "15m"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01          # 1% risk per trade
    max_positions: int = 3
    leverage: int = 1                     # 1x for cash/BNF discipline, up to 5x-20x for scalping
    take_profit_pct: float = 0.003
    stop_loss_pct: float = 0.002
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.001
    use_grid: bool = True
    grid_levels: int = 3
    grid_spacing_pct: float = 0.0005
    cooldown_seconds: int = 30


@dataclass
class StrategyConfig:
    active_strategy: str = "combined"     # combined, kairi_disparity, crypto_laggard, ema_rsi_atr, momentum_scalper, grid_scalper, orderbook_imbalance
    
    # EMA / Momentum parameters
    ema_fast: int = 8
    ema_medium: int = 21
    ema_slow: int = 55
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_scalp_upper: float = 65.0
    rsi_scalp_lower: float = 35.0
    atr_period: int = 14
    atr_multiplier: float = 1.5
    min_atr_threshold: float = 0.0005
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    volume_sma_period: int = 20
    min_volume_multiplier: float = 1.2
    orderbook_depth: int = 10
    imbalance_threshold: float = 0.6

    # BNF 25-Period Disparity (Kairi-ritsu) Strategy
    disparity_period: int = 25
    disparity_panic_threshold: float = -5.0      # -5% dislocation for liquid crypto pairs on intraday
    disparity_stable_threshold: float = -3.5     # -3.5% for high-cap anchors (BTC/ETH)
    disparity_crash_threshold: float = -15.0     # -15% for systemic flash crashes
    disparity_tp_pct: float = 0.04               # +3% to +7% relief bounce target (4% default)
    disparity_sl_pct: float = 0.02               # 2% rigid stop loss (never average down)
    
    # BNF Sector Laggard Rotation (Junkan Busshoku)
    laggard_leader: str = "BTC/USDT"
    laggard_lookback: int = 5
    laggard_leader_surge: float = 0.015          # Leader surge > 1.5%
    laggard_peer_max_move: float = 0.003         # Peer unmoved < 0.3%

    # GRU Deep Learning Predictor
    use_gru_filter: bool = False
    gru_model_path: str = "models/crypto_gru_model.h5"
    gru_scaler_path: str = "models/crypto_gru_model_scaler.pkl"
    gru_lookback: int = 20


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.05
    max_consecutive_losses: int = 5
    daily_trade_limit: int = 100
    pause_after_loss_streak: int = 300
    max_position_size_pct: float = 0.1
    max_position_per_symbol_pct: float = 0.15
    max_portfolio_heat_pct: float = 0.60
    reduce_size_on_loss: bool = True
    size_reduction_factor: float = 0.5
    # Volatility-adjusted position sizing
    vol_adjust_sizing: bool = True
    atr_multiplier_sizing: float = 2.0
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    circuit_breaker_drawdown_pct: float = 0.02
    circuit_breaker_time_window_min: int = 15
    circuit_breaker_error_streak: int = 5
    # Trailing stop & breakeven
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_pct: float = 0.01
    breakeven_activation_pct: float = 0.01


@dataclass
class ExecutionConfig:
    # Order retry
    max_order_retries: int = 3
    retry_backoff_base_ms: int = 500
    retry_backoff_max_ms: int = 5000
    # Slippage
    slippage_model_enabled: bool = True
    max_slippage_pct: float = 0.003          # reject if estimated slippage > 0.3%
    slippage_depth_levels: int = 10
    # Order flags
    post_only: bool = False                  # maker-only limit orders
    reduce_only_stops: bool = True           # stop-losses must be reduce-only
    # TWAP execution
    twap_enabled: bool = False
    twap_slices: int = 5
    twap_interval_sec: int = 30
    # Paper realism
    paper_latency_ms: int = 50
    paper_slippage_pct: float = 0.001        # 0.1% simulated slippage
    paper_partial_fill_chance: float = 0.1   # 10% chance of partial fill
    paper_reject_chance: float = 0.02        # 2% chance of order rejection


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    scalping: ScalpingConfig = field(default_factory=ScalpingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"
    mode: str = "paper"
