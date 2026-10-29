"""
Unified Crypto Scalper & Disparity Bot - Main Entry Point
Integrates:
- Real-time technical scalping (Freqtrade / Jesse AI)
- High-frequency OrderBook microstructure & trade delta flow (Hummingbot)
- Anti-drawdown risk management & grid accumulation (Passivbot)
- Takashi Kotegawa (BNF) 25-Period Disparity (Kairi-ritsu) panic buying
- Deep learning GRU price trend validation
"""
import time
import logging
import os
import sys
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import BotConfig
from core.indicators import Indicators
from core.strategies import get_strategy, Signal
from core.exchange import ExchangeConnector, PaperExchange
from core.risk_manager import RiskManager
from core.gru_predictor import GRUPredictor

console = Console()
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bot.log')
    ]
)
logger = logging.getLogger(__name__)


class UnifiedCryptoBot:
    def __init__(self, config=None):
        self.config = config or BotConfig()
        self.strategy = get_strategy(self.config)
        self.risk_manager = RiskManager(self.config)
        self.gru_predictor = GRUPredictor(
            model_path=self.config.strategy.gru_model_path,
            scaler_path=self.config.strategy.gru_scaler_path,
            lookback=self.config.strategy.gru_lookback
        )
        self.running = False
        self.cycle_count = 0
        self.position_entry_cycle = {}

        if self.config.mode == "live":
            self.exchange = ExchangeConnector(self.config).connect()
        else:
            self.exchange = PaperExchange(self.config)

    def show_banner(self):
        console.print("[bold cyan]====================================================[/bold cyan]")
        console.print("[bold green]  UNIFIED CRYPTO SCALPER & DISPARITY BOT (100% PYTHON) [/bold green]")
        console.print("[bold cyan]====================================================[/bold cyan]")
        console.print(f"Mode: [yellow]{self.config.mode.upper()}[/yellow] | Strategy: [magenta]{self.config.strategy.active_strategy}[/magenta]")
        console.print(f"Symbols: {', '.join(self.config.scalping.symbols)}")
        console.print(f"Timeframe: {self.config.scalping.timeframe} | Leverage: {self.config.scalping.leverage}x")
        console.print(f"Disparity Period: {self.config.strategy.disparity_period} (Panic Thresh: {self.config.strategy.disparity_panic_threshold}%)")
        console.print("[bold cyan]----------------------------------------------------[/bold cyan]")

    def process_symbol(self, symbol: str):
        try:
            # 1. Fetch OHLCV
            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, 200)
            if df is None or len(df) < 30:
                return

            # 2. Vectorized Indicators & Disparity
            df = Indicators.calculate_all(df, self.config)

            # 3. Strategy Evaluation
            setup = self.strategy.analyze(df)
            if setup.signal == Signal.HOLD:
                return

            # 4. Optional GRU Deep Learning Directional Filter
            if self.config.strategy.use_gru_filter and self.gru_predictor.model is not None:
                predicted_price = self.gru_predictor.predict_next_price(df)
                if predicted_price is not None:
                    curr_price = df['close'].iloc[-1]
                    if setup.signal == Signal.LONG and predicted_price < curr_price:
                        logger.info(f"[{symbol}] LONG signal filtered out by GRU forecast (${predicted_price:.2f} < ${curr_price:.2f})")
                        return
                    elif setup.signal == Signal.SHORT and predicted_price > curr_price:
                        logger.info(f"[{symbol}] SHORT signal filtered out by GRU forecast (${predicted_price:.2f} > ${curr_price:.2f})")
                        return

            # 5. Risk Governor Verification
            now = datetime.now()
            can_trade, reason = self.risk_manager.can_trade(now)
            if not can_trade:
                logger.info(f"[{symbol}] {setup.signal.value} rejected by RiskManager: {reason}")
                return

            # 6. Check existing positions (No Averaging Down)
            if self.exchange.get_position(symbol):
                return

            # 7. Position Sizing
            atr = df.get('atr', pd.Series([0])).iloc[-1] if 'atr' in df.columns else None
            size = self.risk_manager.calculate_position_size(
                self.risk_manager.current_balance, setup.entry_price, setup.stop_loss, atr
            )
            if size <= 0:
                return

            # 8. Portfolio Heat Check
            notional_value = setup.entry_price * size
            heat_ok, heat_reason = self.risk_manager.check_portfolio_heat(symbol, notional_value)
            if not heat_ok:
                logger.info(f"[{symbol}] {setup.signal.value} rejected by Portfolio Heat: {heat_reason}")
                return

            side = 'buy' if setup.signal == Signal.LONG else 'sell'
            logger.info(
                f"[{symbol}] [EXECUTE {setup.signal.value}] Strategy: {setup.strategy_name} | "
                f"Entry: ${setup.entry_price:.4f} | SL: ${setup.stop_loss:.4f} | "
                f"TP: ${setup.take_profit:.4f} | Size: {size:.6f} | Notional: ${notional_value:.2f} | Conf: {setup.confidence:.2f}"
            )

            # 9. Order Placement with retry & slippage checks
            if self.config.mode == "live":
                self.exchange.set_leverage(symbol, self.config.scalping.leverage)
                entry_order = self.exchange.create_market_order(symbol, side, size)
                if entry_order and entry_order.state != 'filled':
                    logger.error(f"[{symbol}] Entry order not filled: state={entry_order.state}, error={entry_order.last_error}")
                    return
                opp_side = 'sell' if side == 'buy' else 'buy'
                sl_order = self.exchange.create_stop_loss(symbol, opp_side, size, setup.stop_loss)
                if sl_order and sl_order.state == 'rejected':
                    logger.error(f"[{symbol}] Stop-loss order rejected: {sl_order.last_error}")
            else:
                entry_order = self.exchange.create_market_order(symbol, side, size, {'price': setup.entry_price})
                if isinstance(entry_order, dict) and entry_order.get('status') == 'rejected':
                    logger.error(f"[{symbol}] Entry order rejected: {entry_order.get('reject_reason')}")
                    return

            # 10. Register position for heat & trailing stop tracking
            self.risk_manager.register_position(symbol, notional_value, side)
            self.risk_manager.update_position_price(symbol, setup.entry_price)
            self.position_entry_cycle[symbol] = self.cycle_count

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def check_exits(self, symbol: str):
        try:
            pos = self.exchange.get_position(symbol)
            if not pos:
                return

            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, 5)
            if df is None or len(df) == 0:
                return

            price = df.iloc[-1]['close']
            entry = float(pos.get('entryPrice', 0.0))
            side = pos.get('side', 'long')
            sc = self.config.scalping

            # Max hold time exit
            entry_cycle = self.position_entry_cycle.get(symbol, self.cycle_count)
            max_hold = getattr(sc, 'max_hold_cycles', 60)
            if self.cycle_count - entry_cycle > max_hold:
                pnl = (price - entry) * float(pos.get('contracts', 0.0)) if side == 'long' else (entry - price) * float(pos.get('contracts', 0.0))
                logger.info(f"[{symbol}] Closed {side.upper()} position: TIME EXIT | PnL: ${pnl:.4f}")
                if self.config.mode == "live":
                    opp_side = 'sell' if side == 'long' else 'buy'
                    self.exchange.create_market_order(symbol, opp_side, float(pos.get('contracts', 0.0)))
                else:
                    self.exchange.close_position(symbol, price)
                self.risk_manager.record_trade(pnl, symbol)
                self.risk_manager.release_position(symbol)
                self.position_entry_cycle.pop(symbol, None)
                self.position_entry_cycle.pop(symbol, None)
                return

            # Update trailing stop watermark
            self.risk_manager.update_position_price(symbol, price)

            # Check trailing stop / breakeven
            trail_hit, trail_reason = self.risk_manager.check_trailing_stop(symbol, price)
            if trail_hit:
                reason = f"TRAILING STOP: {trail_reason}"
                logger.info(f"[{symbol}] Closed {side.upper()} position: {reason} | PnL: ${pnl:.4f}")
                if self.config.mode == "live":
                    opp_side = 'sell' if side == 'long' else 'buy'
                    self.exchange.create_market_order(symbol, opp_side, float(pos.get('contracts', 0.0)))
                else:
                    self.exchange.close_position(symbol, price)
                self.risk_manager.record_trade(pnl, symbol)
                self.risk_manager.release_position(symbol)
                self.position_entry_cycle.pop(symbol, None)
                self.position_entry_cycle.pop(symbol, None)
                return

            hit_tp = False
            hit_sl = False

            if side == 'long':
                hit_tp = price >= entry * (1 + sc.take_profit_pct)
                hit_sl = price <= entry * (1 - sc.stop_loss_pct)
                pnl = (price - entry) * float(pos.get('contracts', 0.0))
            else:
                hit_tp = price <= entry * (1 - sc.take_profit_pct)
                hit_sl = price >= entry * (1 + sc.stop_loss_pct)
                pnl = (entry - price) * float(pos.get('contracts', 0.0))

            if hit_tp or hit_sl:
                reason = "TAKE PROFIT" if hit_tp else "STOP LOSS"
                logger.info(f"[{symbol}] Closed {side.upper()} position: {reason} | PnL: ${pnl:.4f}")
                if self.config.mode == "live":
                    opp_side = 'sell' if side == 'long' else 'buy'
                    self.exchange.create_market_order(symbol, opp_side, float(pos.get('contracts', 0.0)))
                else:
                    self.exchange.close_position(symbol, price)
                self.risk_manager.record_trade(pnl, symbol)
                self.risk_manager.release_position(symbol)
                self.position_entry_cycle.pop(symbol, None)

        except Exception as e:
            logger.error(f"Error checking exits for {symbol}: {e}")

    def print_status(self):
        stats = self.risk_manager.get_stats()
        table = Table(title="[bold yellow]Unified Bot Live Telemetry[/bold yellow]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in stats.items():
            table.add_row(k.replace('_', ' ').title(), f"{v:.4f}" if isinstance(v, float) else str(v))
        table.add_row("Cycle Iteration", str(self.cycle_count))
        console.print(table)

    def run(self):
        self.show_banner()
        self.running = True
        logger.info("Bot execution cycle initialized!")

        try:
            while self.running:
                self.cycle_count += 1
                for sym in self.config.scalping.symbols:
                    self.check_exits(sym)
                    self.process_symbol(sym)

                if self.cycle_count % 10 == 0:
                    self.print_status()

                time.sleep(5)  # 5-second polling loop
        except KeyboardInterrupt:
            logger.info("Bot manually terminated by user.")
            self.print_status()


if __name__ == "__main__":
    mode = "paper"
    demo_flag = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--demo":
                demo_flag = True
                continue
            if arg.lower() in ("live", "paper"):
                mode = arg.lower()
                continue
            if arg.lower().startswith("--mode="):
                mode = arg.lower().split("=", 1)[1]
                continue

    cfg = BotConfig()
    if demo_flag:
        cfg.exchange.demo = True
        cfg.exchange.testnet = False
        mode = "live"
    cfg.mode = mode
    if cfg.mode == "live":
        if not cfg.exchange.api_key or not cfg.exchange.api_secret:
            logger.error("API key or secret missing for live mode")
            sys.exit(1)
    bot = UnifiedCryptoBot(cfg)
    bot.run()
