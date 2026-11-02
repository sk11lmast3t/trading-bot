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
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import BotConfig
from core.indicators import Indicators
from core.strategies import get_strategy, Signal
from core.exchange import ExchangeConnector, PaperExchange, OrderState
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
os.makedirs('logs', exist_ok=True)
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    ah = logging.FileHandler('logs/audit.log')
    ah.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    audit_logger.addHandler(ah)
def audit(symbol, side, qty, price, order_id, action='send'):
    audit_logger.info(f"{action} symbol={symbol} side={side} qty={qty} price={price} order_id={order_id}")


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
        self.position_targets = {}
        self.last_rejection_log = {}
        self.trade_lock = threading.Lock()
        self.last_symbol_refresh = 0

        if self.config.mode == "live":
            self.exchange = ExchangeConnector(self.config).connect()
        else:
            self.exchange = PaperExchange(self.config)

        if self.config.scalping.scan_all_coins:
            self.refresh_symbols()

        self.load_state()

    def refresh_symbols(self):
        try:
            logger.info("Scanning market for active liquid trading pairs...")
            discovered = self.exchange.fetch_active_symbols(
                limit=self.config.scalping.max_symbols_to_scan,
                min_volume=self.config.scalping.min_24h_volume_usd,
                exclude=self.config.scalping.exclude_coins
            )
            if discovered:
                self.config.scalping.symbols = discovered
                self.last_symbol_refresh = time.time()
                logger.info(f"Market scan complete. Dynamically tracking {len(discovered)} coins: {', '.join(discovered[:8])}...")
        except Exception as e:
            logger.warning(f"Error during market symbol refresh: {e}")

    def show_banner(self):
        console.print("[bold cyan]====================================================[/bold cyan]")
        console.print("[bold green]  UNIFIED CRYPTO SCALPER & DISPARITY BOT (100% PYTHON) [/bold green]")
        console.print("[bold cyan]====================================================[/bold cyan]")
        console.print(f"Mode: [yellow]{self.config.mode.upper()}[/yellow] | Strategy: [magenta]{self.config.strategy.active_strategy}[/magenta]")
        sym_list = self.config.scalping.symbols
        display_syms = ', '.join(sym_list[:8]) + (f" ... (+{len(sym_list)-8} more)" if len(sym_list) > 8 else "")
        console.print(f"Symbols ({len(sym_list)} total): {display_syms}")
        console.print(f"Timeframe: {self.config.scalping.timeframe} | Leverage: {self.config.scalping.leverage}x | Workers: {self.config.scalping.parallel_workers}")
        console.print(f"Disparity Period: {self.config.strategy.disparity_period} (Panic Thresh: {self.config.strategy.disparity_panic_threshold}%)")
        console.print("[bold cyan]----------------------------------------------------[/bold cyan]")

    def process_symbol(self, symbol: str):
        try:
            # 1. Fetch OHLCV
            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, 201)
            if df is None or len(df) < 30:
                return
            df = df.iloc[:-1]

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
                        last_time = self.last_rejection_log.get((symbol, "gru_long"), 0)
                        if time.time() - last_time > 300:
                            logger.info(f"[{symbol}] LONG signal filtered out by GRU forecast (${predicted_price:.2f} < ${curr_price:.2f})")
                            self.last_rejection_log[(symbol, "gru_long")] = time.time()
                        return
                    elif setup.signal == Signal.SHORT and predicted_price > curr_price:
                        last_time = self.last_rejection_log.get((symbol, "gru_short"), 0)
                        if time.time() - last_time > 300:
                            logger.info(f"[{symbol}] SHORT signal filtered out by GRU forecast (${predicted_price:.2f} > ${curr_price:.2f})")
                            self.last_rejection_log[(symbol, "gru_short")] = time.time()
                        return

            # 5. Risk Governor Verification
            now = datetime.now()
            can_trade, reason = self.risk_manager.can_trade(now)
            if not can_trade:
                last_time = self.last_rejection_log.get((symbol, reason), 0)
                if time.time() - last_time > 300:  # Log at most once per 5 minutes per reason
                    logger.info(f"[{symbol}] {setup.signal.value} rejected by RiskManager: {reason}")
                    self.last_rejection_log[(symbol, reason)] = time.time()
                return
            else:
                self.last_rejection_log.pop((symbol, reason), None)

            # 6. Check existing positions (No Averaging Down)
            if self.exchange.get_position(symbol):
                return

            # 6b. Max positions enforcement
            if len(self.risk_manager.open_positions) >= self.config.risk.max_open_positions:
                logger.info(f"[{symbol}] {setup.signal.value} skipped: max open positions reached")
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
                audit(symbol, side, size, setup.entry_price, getattr(entry_order,'id',None) or getattr(entry_order,'order_id',None))
                if entry_order is None or getattr(entry_order, 'state') != OrderState.FILLED:
                    logger.error(f"[{symbol}] Entry order not filled: state={getattr(entry_order,'state',None)}, error={getattr(entry_order,'last_error',None)}")
                    return
                opp_side = 'sell' if side == 'buy' else 'buy'
                sl_order = self.exchange.create_stop_loss(symbol, opp_side, size, setup.stop_loss)
                audit(symbol, opp_side, size, setup.stop_loss, getattr(sl_order,'id',None) or getattr(sl_order,'order_id',None), 'send')
                if sl_order is None or getattr(sl_order, 'state') == OrderState.REJECTED:
                    logger.error(f"[{symbol}] Stop-loss order rejected: {getattr(sl_order,'last_error',None)}")
                    try:
                        self.exchange.close_position(symbol)
                    except Exception:
                        try:
                            self.exchange.create_market_order(symbol, opp_side, size)
                        except Exception:
                            pass
                    try:
                        self.exchange.cancel_all_orders(symbol)
                    except Exception:
                        pass
                    return
            else:
                entry_order = self.exchange.create_market_order(symbol, side, size, {'price': setup.entry_price})
                audit(symbol, side, size, setup.entry_price, getattr(entry_order,'id',None) or getattr(entry_order,'order_id',None) or (entry_order.get('id') if isinstance(entry_order,dict) else None))
                if entry_order is None:
                    logger.error(f"[{symbol}] Entry order rejected: None")
                    return
                if isinstance(entry_order, dict) and entry_order.get('status') == 'rejected':
                    logger.error(f"[{symbol}] Entry order rejected: {entry_order.get('reject_reason')}")
                    return

            # 10. Register position for heat & trailing stop tracking
            self.risk_manager.register_position(symbol, notional_value, side)
            self.risk_manager.update_position_price(symbol, setup.entry_price)
            self.position_entry_cycle[symbol] = self.cycle_count
            self.position_targets[symbol] = {
                'tp': setup.take_profit,
                'sl': setup.stop_loss,
                'strategy': setup.strategy_name
            }

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def _close_position(self, symbol, pos, price, reason):
        entry = float(pos.get('entryPrice', 0.0))
        contracts = float(pos.get('contracts', 0.0))
        side = pos.get('side', 'long')
        if side == 'long':
            pnl = (price - entry) * contracts
        else:
            pnl = (entry - price) * contracts
        logger.info(f"[{symbol}] Closed {side.upper()} position: {reason} | PnL: ${pnl:.4f}")
        if self.config.mode == "live":
            opp_side = 'sell' if side == 'long' else 'buy'
            co = self.exchange.create_market_order(symbol, opp_side, contracts)
            audit(symbol, opp_side, contracts, price, getattr(co,'id',None) or getattr(co,'order_id',None))
            try:
                self.exchange.cancel_all_orders(symbol)
                audit(symbol, 'cancel', contracts, price, None, 'cancel')
            except Exception:
                pass
        else:
            self.exchange.close_position(symbol, price)
            audit(symbol, 'close', contracts, price, None, 'fill')
        self.risk_manager.record_trade(pnl, symbol)
        self.risk_manager.release_position(symbol)
        self.position_entry_cycle.pop(symbol, None)
        self.position_targets.pop(symbol, None)
        self.persist_state()

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
            targets = self.position_targets.get(symbol, {})
            tp_price = targets.get('tp')
            sl_price = targets.get('sl')

            self.risk_manager.update_position_price(symbol, price)

            # 1. Hard SL (never skipped)
            if side == 'long':
                hit_sl = (price <= sl_price) if (sl_price and sl_price < entry) else (price <= entry * (1 - sc.stop_loss_pct))
            else:
                hit_sl = (price >= sl_price) if (sl_price and sl_price > entry) else (price >= entry * (1 + sc.stop_loss_pct))
            if hit_sl:
                self._close_position(symbol, pos, price, "STOP LOSS")
                return

            # 2. TP
            if side == 'long':
                hit_tp = (price >= tp_price) if (tp_price and tp_price > entry) else (price >= entry * (1 + sc.take_profit_pct))
            else:
                hit_tp = (price <= tp_price) if (tp_price and tp_price < entry) else (price <= entry * (1 - sc.take_profit_pct))
            if hit_tp:
                self._close_position(symbol, pos, price, "TAKE PROFIT")
                return

            # 3. Trailing / breakeven (isolated so failure can't skip SL above)
            try:
                trail_hit, trail_reason = self.risk_manager.check_trailing_stop(symbol, price)
                if trail_hit:
                    self._close_position(symbol, pos, price, f"TRAILING STOP: {trail_reason}")
                    return
            except Exception as e:
                logger.error(f"Trailing check error for {symbol}: {e}")

            # 4. Time exit (wall-clock)
            rm_pos = self.risk_manager.open_positions.get(symbol, {})
            entry_time = rm_pos.get('entry_time')
            if entry_time is not None:
                hold_min = (datetime.now() - entry_time).total_seconds() / 60.0
                if hold_min > self.config.risk.max_hold_time_minutes:
                    self._close_position(symbol, pos, price, "Time exit (wall-clock)")
                    return
            else:
                entry_cycle = self.position_entry_cycle.get(symbol, self.cycle_count)
                max_hold = getattr(sc, 'max_hold_cycles', 60)
                if self.cycle_count - entry_cycle > max_hold:
                    self._close_position(symbol, pos, price, "TIME EXIT")
                    return

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

    def _state_path(self):
        return 'state.json'

    def persist_state(self):
        try:
            data = {
                'open_positions': {},
                'position_targets': self.position_targets,
                'position_entry_cycle': self.position_entry_cycle,
            }
            for sym, pos in getattr(self.risk_manager, 'open_positions', {}).items():
                p = dict(pos)
                if isinstance(p.get('entry_time'), datetime):
                    p['entry_time'] = p['entry_time'].isoformat()
                data['open_positions'][sym] = p
            with open(self._state_path(), 'w') as f:
                json.dump(data, f, default=str)
            logger.info("State persisted")
        except Exception as e:
            logger.error(f"persist_state failed: {e}")

    def load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"load_state failed: {e}")
            return
        try:
            rm = self.risk_manager
            for sym, pos in data.get('open_positions', {}).items():
                p = dict(pos)
                if p.get('entry_time'):
                    try:
                        p['entry_time'] = datetime.fromisoformat(p['entry_time'])
                    except Exception:
                        pass
                rm.open_positions[sym] = p
        except Exception as e:
            logger.error(f"restore positions failed: {e}")
        self.position_targets = data.get('position_targets', self.position_targets)
        self.position_entry_cycle = data.get('position_entry_cycle', self.position_entry_cycle)
        for sym in list(self.risk_manager.open_positions.keys()):
            try:
                ex_pos = self.exchange.get_position(sym)
            except Exception:
                ex_pos = None
            if not ex_pos:
                logger.warning(f"State mismatch for {sym}: not found on exchange, skipping")
                self.risk_manager.open_positions.pop(sym, None)
                self.position_targets.pop(sym, None)
                self.position_entry_cycle.pop(sym, None)
        logger.info("State loaded")

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
            self.persist_state()
        finally:
            try:
                self.persist_state()
            except Exception:
                pass


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
