"""
Unified Crypto Scalper & Disparity Bot - Main Entry Point
Integrates:
- Real-time technical scalping (Freqtrade / Jesse AI)
- High-frequency OrderBook microstructure & trade delta flow (Hummingbot)
- Anti-drawdown risk management & grid accumulation (Passivbot)
- Takashi Kotegawa (BNF) 25-Period Disparity (Kairi-ritsu) panic buying
- Deep learning GRU price trend validation
"""
import argparse
import csv
import time
import logging
import os
import json
import sys
import threading
import subprocess
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
from core import state

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


DECISION_FIELDS = [
    "ts", "symbol", "strategy", "signal", "intended_entry", "sl", "tp",
    "status", "fill_price", "fee", "slippage", "exit_price", "exit_reason",
    "hold_minutes", "net_pnl", "balance_after", "quantity", "notional",
    "atr_1h_pct", "ema20_1h_slope_sign", "ema20_1h_slope_pct", "trend_30m",
    "rsi_5m", "ema_distance_atr_5m", "hour_utc", "day_of_week_utc",
    "rv_24h_bucket", "mfe_bps", "mae_bps", "bars_held", "planned_sl_bps",
    "planned_tp_bps", "r_multiple", "entry_fill_diff_bps",
    "intended_exit", "exit_fill_diff_bps", "fee_bps", "slippage_bps", "close_source",
]


def _utc_datetime_index(frame):
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if idx.isna().any():
        return None
    return idx


def _completed_bars(frame, minutes, asof):
    idx = _utc_datetime_index(frame)
    if idx is None:
        return frame.iloc[0:0] if frame is not None else pd.DataFrame()
    out = frame.copy()
    out.index = idx
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    return out.loc[out.index + pd.Timedelta(minutes=minutes) <= asof]


def _number_or_none(value):
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def _order_fill_price(order, fallback):
    if isinstance(order, dict):
        value = order.get("average") or order.get("average_price") or order.get("price")
    else:
        value = getattr(order, "average_price", None) or getattr(order, "price", None)
    number = _number_or_none(value)
    return float(number if number else fallback)


def _order_fee(order):
    raw = order.get("fee") if isinstance(order, dict) else getattr(order, "fee", None)
    if isinstance(raw, dict):
        raw = raw.get("cost")
    number = _number_or_none(raw)
    return float(number or 0.0)


class UnifiedCryptoBot:
    def __init__(self, config=None, clock=None):
        self.config = config or BotConfig()
        state.set_mode(self.config.mode)
        clock_source = clock or getattr(self.config, "clock", None)
        self.clock = clock_source or datetime.now
        self._clock_uses_local_timezone = clock_source is None
        self.strategy = get_strategy(self.config)
        self.risk_manager = RiskManager(self.config, clock=clock_source)
        self.gru_predictor = None
        if self.config.strategy.use_gru_filter:
            from core.gru_predictor import GRUPredictor
            self.gru_predictor = GRUPredictor(
                model_path=self.config.strategy.gru_model_path,
                scaler_path=self.config.strategy.gru_scaler_path,
                lookback=self.config.strategy.gru_lookback
            )
        self.running = False
        self.cycle_count = 0
        self.position_entry_cycle = {}
        self.position_targets = {}
        self.entries_blocked = False
        self.loaded_positions = []
        self.last_rejection_log = {}
        self.last_processed_bar = {}
        self.active_decisions = {}
        self.decision_log_path = os.path.join("logs", "decisions.csv")
        self.decision_log_lock = threading.Lock()
        self.trade_lock = threading.Lock()
        self.last_symbol_refresh = 0

        if self.config.mode == "live":
            self.exchange = ExchangeConnector(self.config).connect()
        else:
            self.exchange = PaperExchange(self.config)
            self.exchange.bind_balance_source(self.risk_manager)
            # Paper accounting charges fixed slippage separately, as the backtester does.
            self.exchange.base_slippage_pct = 0.0
        self.paper_trade_meta = {}

        if self.config.scalping.scan_all_coins:
            self.refresh_symbols()

        if getattr(self.config, "fresh", False) and self.config.mode == "paper":
            self.loaded_positions = []
            logger.info("Fresh paper session: skipping persisted position restore")
        else:
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

    def _append_decision(self, decision):
        path = getattr(self, "decision_log_path", None)
        if not path:
            return
        fields = DECISION_FIELDS
        lock = getattr(self, "decision_log_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.decision_log_lock = lock
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with lock:
                write_header = not os.path.exists(path) or os.path.getsize(path) == 0
                if not write_header:
                    with open(path, newline="", encoding="utf-8") as existing:
                        reader = csv.DictReader(existing)
                        old_fields = reader.fieldnames
                        old_rows = list(reader)
                    if old_fields != fields:
                        with open(path, "w", newline="", encoding="utf-8") as upgraded:
                            writer = csv.DictWriter(upgraded, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows({field: row.get(field, "") for field in fields}
                                             for row in old_rows)
                with open(path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if write_header:
                        writer.writeheader()
                    writer.writerow({field: decision.get(field, "") for field in fields})
        except Exception as e:
            logger.error("Could not append decision log %s: %s", path, e)

    def _update_decision_exit(self, context, values):
        path = getattr(self, "decision_log_path", None)
        if not path or not os.path.exists(path):
            return
        fields = DECISION_FIELDS
        lock = getattr(self, "decision_log_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.decision_log_lock = lock
        try:
            with lock:
                with open(path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                for row in reversed(rows):
                    if (row.get("ts") == context.get("ts")
                            and row.get("symbol") == context.get("symbol")
                            and row.get("strategy") == context.get("strategy")
                            and row.get("status") == "entered"):
                        row.update(values)
                        break
                else:
                    return
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        except Exception as e:
            logger.error("Could not update decision log %s: %s", path, e)

    def _decision_context(self, symbol, signal_bar_open, five_minute_bars):
        """Build observation fields from candles that had closed by signal time."""
        context = {
            "atr_1h_pct": "", "ema20_1h_slope_sign": "",
            "ema20_1h_slope_pct": "", "trend_30m": "",
            "rsi_5m": "", "ema_distance_atr_5m": "",
            "hour_utc": "", "day_of_week_utc": "", "rv_24h_bucket": "",
        }
        try:
            signal_ts = pd.Timestamp(signal_bar_open)
            signal_ts = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts.tz_convert("UTC")
        except (TypeError, ValueError):
            return context
        asof = signal_ts + pd.Timedelta(minutes=5)

        try:
            hourly_raw = self.exchange.fetch_ohlcv(symbol, "1h", 100)
            hourly = _completed_bars(hourly_raw, 60, asof)
            if len(hourly) >= 21:
                close = hourly["close"].astype(float)
                ema20 = Indicators.ema(close, 20)
                atr = Indicators.atr(hourly["high"].astype(float), hourly["low"].astype(float), close, 14)
                context["atr_1h_pct"] = float(atr.iloc[-1] / close.iloc[-1] * 100.0)
                slope = float((ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2] * 100.0)
                context["ema20_1h_slope_sign"] = "up" if slope > 0 else "down" if slope < 0 else "flat"
                context["ema20_1h_slope_pct"] = slope
        except Exception:
            pass

        try:
            half_hour_raw = self.exchange.fetch_ohlcv(symbol, "30m", 100)
            half_hour = _completed_bars(half_hour_raw, 30, asof)
            if len(half_hour) >= 21:
                close = half_hour["close"].astype(float)
                ema20 = Indicators.ema(close, 20)
                slope = ema20.iloc[-1] - ema20.iloc[-2]
                if close.iloc[-1] > ema20.iloc[-1] and slope > 0:
                    context["trend_30m"] = "up"
                elif close.iloc[-1] < ema20.iloc[-1] and slope < 0:
                    context["trend_30m"] = "down"
                else:
                    context["trend_30m"] = "sideways"
        except Exception:
            pass

        try:
            five = _completed_bars(five_minute_bars, 5, asof)
            if len(five) >= 20:
                close = five["close"].astype(float)
                ema20 = Indicators.ema(close, 20)
                rsi = Indicators.rsi(close, 14)
                atr = Indicators.atr(five["high"].astype(float), five["low"].astype(float), close, 14)
                if pd.notna(rsi.iloc[-1]):
                    context["rsi_5m"] = float(rsi.iloc[-1])
                if pd.notna(atr.iloc[-1]) and atr.iloc[-1] != 0:
                    context["ema_distance_atr_5m"] = float((close.iloc[-1] - ema20.iloc[-1]) / atr.iloc[-1])
            if len(five) >= 13:
                returns = five["close"].astype(float).pct_change().dropna().tail(288)
                if len(returns) >= 12:
                    rv_24h_pct = float(returns.std(ddof=0) * (len(returns) ** 0.5) * 100.0)
                    context["rv_24h_bucket"] = "low" if rv_24h_pct < 2 else "medium" if rv_24h_pct < 5 else "high"
        except Exception:
            pass

        context["hour_utc"] = int(asof.hour)
        context["day_of_week_utc"] = int(asof.day_of_week)
        return context

    def _clock_utc(self):
        now = pd.Timestamp(getattr(self, "clock", datetime.now)())
        if now.tzinfo is not None:
            return now.tz_convert("UTC")
        if getattr(self, "_clock_uses_local_timezone", False):
            local_zone = datetime.now().astimezone().tzinfo
            return now.tz_localize(local_zone).tz_convert("UTC")
        return now.tz_localize("UTC")

    def process_symbol(self, symbol: str):
        decision = None
        decision_written = False

        def finish_decision(status, **values):
            nonlocal decision_written
            if decision is None or decision_written:
                return
            decision["status"] = status
            decision.update(values)
            self._append_decision(decision)
            decision_written = True

        if getattr(self, "entries_blocked", False) and symbol not in getattr(self.risk_manager, "open_positions", {}):
            try:
                self.check_exits(symbol)
            except Exception:
                pass
            return
        try:
            # 1. Fetch OHLCV
            lookback = int(getattr(self.strategy, "lookback_bars", 300))
            fetch_limit = max(201, lookback + 2)
            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, fetch_limit)
            if df is None:
                logger.error("[%s] Candle fetch failed; skipping this bar", symbol)
                return
            if len(df) < 30:
                logger.info("[%s] Only %s candles available; skipping this bar", symbol, len(df))
                return
            df = df.iloc[:-1]
            completed_bar = df.index[-1]
            last_processed = getattr(self, "last_processed_bar", {})
            if last_processed.get(symbol) == completed_bar:
                return
            last_processed[symbol] = completed_bar
            self.last_processed_bar = last_processed

            # 2. Vectorized Indicators & Disparity
            df = Indicators.calculate_all(df, self.config)

            # 3. Strategy Evaluation
            setup = self.strategy.analyze(df)
            decision = {
                "ts": completed_bar.isoformat(),
                "symbol": symbol,
                "strategy": setup.strategy_name or getattr(self.strategy, "name", ""),
                "signal": setup.signal.value,
                "intended_entry": setup.entry_price,
                "sl": setup.stop_loss,
                "tp": setup.take_profit,
                "balance_after": self.risk_manager.current_balance,
            }
            decision.update(self._decision_context(symbol, completed_bar, df))
            entry_reference = float(setup.entry_price or 0.0)
            decision["planned_sl_bps"] = (abs(entry_reference - float(setup.stop_loss)) / entry_reference * 10000.0
                                           if entry_reference else "")
            decision["planned_tp_bps"] = (abs(entry_reference - float(setup.take_profit)) / entry_reference * 10000.0
                                           if entry_reference else "")
            if setup.signal == Signal.HOLD:
                finish_decision("skipped:hold")
                return
            if getattr(self.config, "long_only", False) and setup.signal == Signal.SHORT:
                finish_decision("skipped:long_only")
                return

            # 4. Optional GRU Deep Learning Directional Filter
            if self.config.strategy.use_gru_filter and self.gru_predictor is not None and self.gru_predictor.model is not None:
                predicted_price = self.gru_predictor.predict_next_price(df)
                if predicted_price is not None:
                    curr_price = df['close'].iloc[-1]
                    if setup.signal == Signal.LONG and predicted_price < curr_price:
                        last_time = self.last_rejection_log.get((symbol, "gru_long"), 0)
                        if time.time() - last_time > 300:
                            logger.info(f"[{symbol}] LONG signal filtered out by GRU forecast (${predicted_price:.2f} < ${curr_price:.2f})")
                            self.last_rejection_log[(symbol, "gru_long")] = time.time()
                        finish_decision("skipped:gru_filter:long")
                        return
                    elif setup.signal == Signal.SHORT and predicted_price > curr_price:
                        last_time = self.last_rejection_log.get((symbol, "gru_short"), 0)
                        if time.time() - last_time > 300:
                            logger.info(f"[{symbol}] SHORT signal filtered out by GRU forecast (${predicted_price:.2f} > ${curr_price:.2f})")
                            self.last_rejection_log[(symbol, "gru_short")] = time.time()
                        finish_decision("skipped:gru_filter:short")
                        return

            # 5. Risk Governor Verification
            now = getattr(self, "clock", datetime.now)()
            can_trade, reason = self.risk_manager.can_trade(now)
            if not can_trade:
                last_time = self.last_rejection_log.get((symbol, reason), 0)
                if time.time() - last_time > 300:  # Log at most once per 5 minutes per reason
                    logger.info(f"[{symbol}] {setup.signal.value} rejected by RiskManager: {reason}")
                    self.last_rejection_log[(symbol, reason)] = time.time()
                finish_decision(f"skipped:risk:{reason}")
                return
            else:
                self.last_rejection_log.pop((symbol, reason), None)

            # 6. Check existing positions (No Averaging Down)
            if self.exchange.get_position(symbol):
                finish_decision("skipped:position_open")
                return

            # 6b. Max positions enforcement
            if len(self.risk_manager.open_positions) >= self.config.risk.max_open_positions:
                logger.info(f"[{symbol}] {setup.signal.value} skipped: max open positions reached")
                finish_decision("skipped:max_positions")
                return

            # 7. Position Sizing
            atr = df.get('atr', pd.Series([0])).iloc[-1] if 'atr' in df.columns else None
            size = self.risk_manager.calculate_position_size(
                self.risk_manager.current_balance, setup.entry_price, setup.stop_loss, atr
            )
            if size <= 0:
                finish_decision("skipped:zero_size")
                return

            # 8. Portfolio Heat Check
            notional_value = setup.entry_price * size
            heat_ok, heat_reason = self.risk_manager.check_portfolio_heat(symbol, notional_value)
            if not heat_ok:
                logger.info(f"[{symbol}] {setup.signal.value} rejected by Portfolio Heat: {heat_reason}")
                finish_decision(f"skipped:portfolio_heat:{heat_reason}")
                return

            side = 'buy' if setup.signal == Signal.LONG else 'sell'
            logger.info(
                f"[{symbol}] [EXECUTE {setup.signal.value}] Strategy: {setup.strategy_name} | "
                f"Entry: ${setup.entry_price:.4f} | SL: ${setup.stop_loss:.4f} | "
                f"TP: ${setup.take_profit:.4f} | Size: {size:.6f} | Notional: ${notional_value:.2f} | Conf: {setup.confidence:.2f}"
            )

            # 9. Order Placement with retry & slippage checks
            entry_mode = self.config.execution.entry_mode
            if entry_mode == "maker":
                # Post-only limit at best bid/ask; track fill rate; cancel after timeout.
                if self.config.mode == "live":
                    self.exchange.set_leverage(symbol, self.config.scalping.leverage)
                    entry_order = self.exchange.create_best_bid_ask_limit_order(
                        symbol, side, size, post_only=True
                    )
                else:
                    entry_order = self.exchange.create_best_bid_ask_limit_order(
                        symbol, side, size, post_only=True
                    )
                entry_id = (getattr(entry_order, 'id', None) or
                            getattr(entry_order, 'order_id', None) or
                            (entry_order.get('id') if isinstance(entry_order, dict) else None))
                audit(symbol, side, size, setup.entry_price, entry_id)
                if entry_order is None:
                    logger.error(f"[{symbol}] Maker entry order rejected: None")
                    finish_decision("skipped:maker_rejected")
                    return
                if isinstance(entry_order, dict) and entry_order.get('status') == 'rejected':
                    logger.error(f"[{symbol}] Maker entry order rejected: {entry_order.get('reject_reason')}")
                    finish_decision(f"skipped:maker_rejected:{entry_order.get('reject_reason', '')}")
                    return
                # Track fill rate and cancel after timeout if unfilled.
                filled = False
                cancel_deadline = time.time() + self.config.execution.maker_cancel_seconds
                while time.time() < cancel_deadline:
                    if self.config.mode == "live":
                        checked = self.exchange.check_maker_fill(symbol, entry_id, setup.entry_price, side)
                    else:
                        checked = self.exchange.check_limit_fill(symbol, entry_id, setup.entry_price, side)
                    status = getattr(checked, 'state', None) or (checked.get('status') if isinstance(checked, dict) else None)
                    if status in ('filled', 'closed') or (isinstance(checked, dict) and checked.get('filled', 0) >= size):
                        filled = True
                        entry_order = checked
                        break
                    if status in ('cancelled', 'canceled', 'rejected'):
                        break
                    time.sleep(1)
                if not filled:
                    logger.info(f"[{symbol}] Maker entry not filled within {self.config.execution.maker_cancel_seconds}s; cancelling")
                    try:
                        self.exchange.cancel_all_orders(symbol)
                        audit(symbol, 'cancel', size, setup.entry_price, entry_id, 'cancel')
                    except Exception:
                        pass
                    finish_decision("skipped:maker_not_filled")
                    return
            elif self.config.mode == "live":
                self.exchange.set_leverage(symbol, self.config.scalping.leverage)
                entry_order = self.exchange.create_market_order(symbol, side, size)
                audit(symbol, side, size, setup.entry_price, getattr(entry_order,'id',None) or getattr(entry_order,'order_id',None))
                if entry_order is None or getattr(entry_order, 'state') != OrderState.FILLED:
                    logger.error(f"[{symbol}] Entry order not filled: state={getattr(entry_order,'state',None)}, error={getattr(entry_order,'last_error',None)}")
                    finish_decision("skipped:entry_not_filled")
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
                    finish_decision("skipped:stop_loss_rejected")
                    return
            else:
                entry_order = self.exchange.create_market_order(symbol, side, size, {'price': setup.entry_price})
                audit(symbol, side, size, setup.entry_price, getattr(entry_order,'id',None) or getattr(entry_order,'order_id',None) or (entry_order.get('id') if isinstance(entry_order,dict) else None))
                if entry_order is None:
                    logger.error(f"[{symbol}] Entry order rejected: None")
                    finish_decision("skipped:entry_rejected")
                    return
                if isinstance(entry_order, dict) and entry_order.get('status') == 'rejected':
                    logger.error(f"[{symbol}] Entry order rejected: {entry_order.get('reject_reason')}")
                    finish_decision(f"skipped:entry_rejected:{entry_order.get('reject_reason', '')}")
                    return

            # 10. Register actual fill size and track paper costs until the trade closes.
            filled_amount = float(
                entry_order.get("filled", size) if isinstance(entry_order, dict)
                else getattr(entry_order, "filled", None) or size
            )
            fill_price = float(
                (entry_order.get("price") or setup.entry_price) if isinstance(entry_order, dict)
                else getattr(entry_order, "price", None) or setup.entry_price
            )
            observed_entry_fill = float(
                _order_fill_price(entry_order, fill_price)
            )
            observed_entry_fee = _order_fee(entry_order)
            notional_value = fill_price * filled_amount
            journal_notional = observed_entry_fill * filled_amount
            if self.config.mode != "live":
                fee_rate = (self.config.backtest.maker_fee if entry_mode == "maker"
                            else self.config.backtest.taker_fee)
                slip_rate = 0.0 if entry_mode == "maker" else self.config.backtest.slippage
                meta = getattr(self, "paper_trade_meta", {})
                meta[symbol] = {
                    "entry_price": fill_price,
                    "quantity": filled_amount,
                    "entry_fee": fill_price * filled_amount * fee_rate,
                    "entry_slippage": fill_price * filled_amount * slip_rate,
                    "entry_time": getattr(self, "clock", datetime.now)(),
                    "strategy": setup.strategy_name,
                    "signal": setup.signal.value,
                }
                self.paper_trade_meta = meta

            self.risk_manager.register_position(symbol, notional_value, side)
            state.save_position_at_entry(
                symbol, side, filled_amount, fill_price, setup.take_profit, setup.stop_loss,
                setup.strategy_name, entry_time=getattr(self, "clock", datetime.now)().isoformat(),
            )
            self.risk_manager.update_position_price(symbol, fill_price)
            self.position_entry_cycle[symbol] = self.cycle_count
            signal_bar_open = pd.Timestamp(decision["ts"])
            self.position_targets[symbol] = {
                'tp': setup.take_profit,
                'sl': setup.stop_loss,
                'strategy': setup.strategy_name,
                'signal_bar_open': signal_bar_open.isoformat(),
                'journal_entry': dict(decision),
            }
            active = getattr(self, "active_decisions", {})
            active[symbol] = {
                **decision,
                "entry_time": getattr(self, "clock", datetime.now)(),
                "entry_fill_price": observed_entry_fill,
                "quantity": filled_amount,
                "notional": journal_notional,
                "entry_live_fee": observed_entry_fee,
            }
            self.active_decisions = active
            meta = getattr(self, "paper_trade_meta", {}).get(symbol, {})
            finish_decision(
                "entered", fill_price=observed_entry_fill,
                fee=meta.get("entry_fee", ""),
                slippage=meta.get("entry_slippage", ""),
                quantity=filled_amount,
                notional=journal_notional,
                entry_fill_diff_bps=((observed_entry_fill - float(setup.entry_price)) / float(setup.entry_price) * 10000.0
                                     if setup.entry_price else ""),
                balance_after=self.risk_manager.current_balance,
            )
            self.position_targets[symbol]['journal_entry'].update(decision)
            self.position_targets[symbol]['journal_entry'].update({
                "fill_price": observed_entry_fill, "quantity": filled_amount, "notional": journal_notional,
                "entry_time": getattr(self, "clock", datetime.now)().isoformat(),
                "entry_live_fee": observed_entry_fee,
            })
            self.active_decisions[symbol].update({
                "fill_price": observed_entry_fill, "quantity": filled_amount, "notional": journal_notional,
                "entry_fill_diff_bps": decision.get("entry_fill_diff_bps", ""),
            })

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            finish_decision(f"skipped:error:{e}")

    def _exit_excursions(self, symbol, pos, frame, context):
        targets = getattr(self, "position_targets", {}).get(symbol, {})
        signal_open = targets.get("signal_bar_open") or context.get("ts")
        if not signal_open or frame is None:
            return {"mfe_bps": "", "mae_bps": "", "bars_held": ""}
        try:
            idx = _utc_datetime_index(frame)
            if idx is None:
                return {"mfe_bps": "", "mae_bps": "", "bars_held": ""}
            observed = frame.copy()
            observed.index = idx
            signal_ts = pd.Timestamp(signal_open)
            signal_ts = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts.tz_convert("UTC")
            observed = observed.loc[(observed.index > signal_ts) & (observed.index <= self._clock_utc())]
            if observed.empty:
                return {"mfe_bps": "", "mae_bps": "", "bars_held": 0}
            entry = float(context.get("fill_price") or context.get("entry_fill_price") or pos.get("entryPrice", 0.0))
            if entry <= 0:
                return {"mfe_bps": "", "mae_bps": "", "bars_held": len(observed)}
            side = pos.get("side", "long")
            high = observed["high"].astype(float).max()
            low = observed["low"].astype(float).min()
            if side == "long":
                mfe = max((high - entry) / entry * 10000.0, 0.0)
                mae = max((entry - low) / entry * 10000.0, 0.0)
            else:
                mfe = max((entry - low) / entry * 10000.0, 0.0)
                mae = max((high - entry) / entry * 10000.0, 0.0)
            return {"mfe_bps": float(mfe), "mae_bps": float(mae), "bars_held": int(len(observed))}
        except Exception:
            return {"mfe_bps": "", "mae_bps": "", "bars_held": ""}

    def _prepare_poll_close(self, symbol, pos, frame, intended_exit):
        targets = getattr(self, "position_targets", {}).get(symbol, {})
        context = getattr(self, "active_decisions", {}).get(symbol) or targets.get("journal_entry")
        if context is None:
            return
        try:
            context.update(self._exit_excursions(symbol, pos, frame, context))
            context["close_source"] = "poll"
            context["intended_exit"] = intended_exit
        except Exception as exc:
            logger.debug("Could not record exit observations for %s: %s", symbol, exc)

    def _record_exchange_stop_flat(self, symbol):
        if getattr(self.config, "mode", "paper") != "live":
            return
        targets = getattr(self, "position_targets", {}).get(symbol, {})
        context = getattr(self, "active_decisions", {}).get(symbol) or targets.get("journal_entry")
        if not context:
            return
        signal_open = targets.get("signal_bar_open") or context.get("ts")
        candles = None
        try:
            candles = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, 300)
        except Exception:
            pass
        position = {
            "entryPrice": context.get("fill_price", context.get("entry_fill_price", 0)),
            "contracts": context.get("quantity", 0),
            "side": "short" if context.get("signal") == "SHORT" else "long",
        }
        observed = self._exit_excursions(symbol, position, candles, context)
        hold_minutes = ""
        entry_time = context.get("entry_time")
        if isinstance(entry_time, datetime):
            hold_minutes = max((getattr(self, "clock", datetime.now)() - entry_time).total_seconds() / 60.0, 0.0)
        elif context.get("entry_time"):
            try:
                entered = datetime.fromisoformat(context["entry_time"])
                hold_minutes = max((getattr(self, "clock", datetime.now)() - entered).total_seconds() / 60.0, 0.0)
            except (TypeError, ValueError):
                pass
        self._update_decision_exit(context, {
            "status": "closed", "exit_price": "", "exit_reason": "exchange_stop_or_external_close",
            "intended_exit": targets.get("sl", ""),
            "hold_minutes": hold_minutes, "close_source": "exchange_stop_inferred",
            "exit_fill_diff_bps": "", "fee_bps": "", "slippage_bps": "",
            "r_multiple": "", **observed,
        })
        getattr(self, "active_decisions", {}).pop(symbol, None)

    def _close_position(self, symbol, pos, price, reason, reference_price=None):
        reference_price = float(price if reference_price is None else reference_price)
        entry = float(pos.get('entryPrice', 0.0))
        contracts = float(pos.get('contracts', 0.0))
        side = pos.get('side', 'long')
        decision_context = getattr(self, "active_decisions", {}).pop(symbol, None)
        targets = getattr(self, "position_targets", {}).get(symbol, {})
        if decision_context is None:
            decision_context = targets.get("journal_entry")
        hold_minutes = ""
        if decision_context and isinstance(decision_context.get("entry_time"), datetime):
            hold_minutes = max((getattr(self, "clock", datetime.now)() - decision_context["entry_time"]).total_seconds() / 60.0, 0.0)
        if side == 'long':
            gross_pnl = (reference_price - entry) * contracts
        else:
            gross_pnl = (entry - reference_price) * contracts
        pnl = gross_pnl
        journal_exit_fill = float(price)
        live_exit_fee = None
        if self.config.mode == "live":
            opp_side = 'sell' if side == 'long' else 'buy'
            co = self.exchange.create_market_order(symbol, opp_side, contracts)
            audit(symbol, opp_side, contracts, price, getattr(co,'id',None) or getattr(co,'order_id',None))
            if co is not None:
                journal_exit_fill = _order_fill_price(co, price)
                live_exit_fee = _order_fee(co)
            try:
                self.exchange.cancel_all_orders(symbol)
                audit(symbol, 'cancel', contracts, price, None, 'cancel')
            except Exception:
                pass
        else:
            self.exchange.close_position(symbol, price)
            audit(symbol, 'close', contracts, price, None, 'fill')
            meta = getattr(self, "paper_trade_meta", {}).pop(symbol, {})
            exit_fee = reference_price * contracts * self.config.backtest.taker_fee
            exit_slippage = reference_price * contracts * self.config.backtest.slippage
            decision_fee = meta.get("entry_fee", 0.0) + exit_fee
            decision_slippage = meta.get("entry_slippage", 0.0) + exit_slippage
            pnl = (gross_pnl - meta.get("entry_fee", 0.0)
                   - meta.get("entry_slippage", 0.0) - exit_fee - exit_slippage)
        logger.info(f"[{symbol}] Closed {side.upper()} position: {reason} | PnL: ${pnl:.4f}")
        self.risk_manager.record_trade(pnl, symbol)
        state.record_trade(pnl, symbol, reason)
        if decision_context:
            entry_fill = float(decision_context.get("fill_price") or decision_context.get("entry_fill_price") or entry or 0.0)
            notional = float(decision_context.get("notional") or entry_fill * contracts)
            planned_risk = abs(entry_fill - float(targets.get("sl", decision_context.get("sl", 0.0)) or 0.0)) * contracts
            journal_intended_exit = float(decision_context.get("intended_exit", reference_price) or reference_price)
            if self.config.mode != "live":
                fee_bps = decision_fee / notional * 10000.0 if notional else ""
            else:
                entry_fee = _number_or_none(decision_context.get("entry_live_fee"))
                fee_bps = ((entry_fee + live_exit_fee) / notional * 10000.0
                           if entry_fee is not None and live_exit_fee is not None and notional else "")
            if self.config.mode != "live" and notional:
                slip_bps = decision_slippage / notional * 10000.0
            else:
                entry_diff = float(decision_context.get("entry_fill_diff_bps") or 0.0)
                exit_diff = ((journal_exit_fill - journal_intended_exit) / journal_intended_exit * 10000.0
                             if journal_intended_exit else 0.0)
                slip_bps = abs(entry_diff) + abs(exit_diff)
            self._update_decision_exit(decision_context, {
                "status": "closed",
                "exit_price": journal_exit_fill,
                "exit_reason": reason,
                "intended_exit": journal_intended_exit,
                "hold_minutes": hold_minutes,
                "net_pnl": pnl,
                "balance_after": self.risk_manager.current_balance,
                "mfe_bps": decision_context.get("mfe_bps", ""),
                "mae_bps": decision_context.get("mae_bps", ""),
                "bars_held": decision_context.get("bars_held", ""),
                "exit_fill_diff_bps": ((journal_exit_fill - journal_intended_exit) / journal_intended_exit * 10000.0
                                        if journal_intended_exit else ""),
                "fee_bps": fee_bps,
                "slippage_bps": slip_bps,
                "r_multiple": pnl / planned_risk if planned_risk else "",
                "close_source": decision_context.get("close_source", "poll"),
                **({"fee": decision_fee, "slippage": decision_slippage}
                   if self.config.mode != "live" else {}),
            })
        self.risk_manager.release_position(symbol)
        self.position_entry_cycle.pop(symbol, None)
        self.position_targets.pop(symbol, None)
        self.persist_state()

    def check_exits(self, symbol: str):
        try:
            pos = self.exchange.get_position(symbol)
            if not pos:
                self._record_exchange_stop_flat(symbol)
                return

            targets = self.position_targets.get(symbol, {})
            signal_bar_open = targets.get('signal_bar_open')
            limit = 5
            if signal_bar_open:
                try:
                    elapsed = max((getattr(self, "clock", datetime.now)() -
                                   pd.Timestamp(signal_bar_open).to_pydatetime()).total_seconds(), 0)
                    limit = max(limit, int(elapsed // 300) + 3)
                except (TypeError, ValueError):
                    pass
            df = self.exchange.fetch_ohlcv(symbol, self.config.scalping.timeframe, limit)
            if df is None or len(df) == 0:
                return

            price = df.iloc[-1]['close']
            entry = float(pos.get('entryPrice', 0.0))
            side = pos.get('side', 'long')
            sc = self.config.scalping
            tp_price = targets.get('tp')
            sl_price = targets.get('sl')

            self.risk_manager.update_position_price(symbol, price)

            # Check every post-signal candle, including the currently forming one.
            fallback_sl = entry * (1 - sc.stop_loss_pct) if side == 'long' else entry * (1 + sc.stop_loss_pct)
            fallback_tp = entry * (1 + sc.take_profit_pct) if side == 'long' else entry * (1 - sc.take_profit_pct)
            sl_price = float(sl_price if sl_price is not None else fallback_sl)
            tp_price = float(tp_price if tp_price is not None else fallback_tp)
            candles = df
            if signal_bar_open and isinstance(df.index, pd.DatetimeIndex):
                candles = df.loc[df.index > pd.Timestamp(signal_bar_open)]
            elif len(df) > 1:
                # Compatibility with exchanges/tests that return no candle timestamps.
                candles = df.iloc[1:]
            for _, candle in candles.iterrows():
                high = float(candle.get('high', candle['close']))
                low = float(candle.get('low', candle['close']))
                hit_sl = low <= sl_price if side == 'long' else high >= sl_price
                hit_tp = high >= tp_price if side == 'long' else low <= tp_price
                if hit_sl or hit_tp:
                    is_sl = hit_sl  # adverse-first when both levels lie inside the bar
                    level = sl_price if is_sl else tp_price
                    fill = level
                    if self.config.mode != "live":
                        slip = float(self.config.backtest.slippage)
                        fill = level * (1 - slip if side == 'long' else 1 + slip)
                    self._prepare_poll_close(symbol, pos, df, level)
                    self._close_position(symbol, pos, fill,
                                         "STOP LOSS" if is_sl else "TAKE PROFIT",
                                         reference_price=level if self.config.mode != "live" else fill)
                    return

            # 3. Trailing / breakeven (isolated so failure can't skip SL above)
            try:
                trail_hit, trail_reason = self.risk_manager.check_trailing_stop(symbol, price)
                if trail_hit:
                    self._prepare_poll_close(symbol, pos, df, price)
                    self._close_position(symbol, pos, price, f"TRAILING STOP: {trail_reason}")
                    return
            except Exception as e:
                logger.error(f"Trailing check error for {symbol}: {e}")

            # 4. Time exit (wall-clock)
            rm_pos = self.risk_manager.open_positions.get(symbol, {})
            entry_time = rm_pos.get('entry_time')
            if entry_time is not None:
                hold_min = (getattr(self, "clock", datetime.now)() - entry_time).total_seconds() / 60.0
                signal_close = None
                if signal_bar_open:
                    signal_close = pd.Timestamp(signal_bar_open).to_pydatetime() + pd.Timedelta(minutes=5)
                elapsed = (getattr(self, "clock", datetime.now)() - signal_close).total_seconds() / 60.0 if signal_close else hold_min
                if elapsed >= self.config.risk.max_hold_time_minutes:
                    self._prepare_poll_close(symbol, pos, df, price)
                    self._close_position(symbol, pos, price, "Time exit (wall-clock)")
                    return
            else:
                entry_cycle = self.position_entry_cycle.get(symbol, self.cycle_count)
                max_hold = getattr(sc, 'max_hold_cycles', 60)
                if self.cycle_count - entry_cycle > max_hold:
                    self._prepare_poll_close(symbol, pos, df, price)
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
        import core.state as cstate
        try:
            self.loaded_positions = cstate.load_open_positions()
        except Exception as e:
            logger.error(f"load_state failed: {e}")
            self.loaded_positions = []
        try:
            ex_positions = {}
            try:
                ex_positions = self.exchange.get_positions() or {}
            except Exception:
                for pp in self.loaded_positions:
                    sym = pp.get('symbol')
                    if sym:
                        try:
                            ep = self.exchange.get_position(sym)
                            if ep:
                                ex_positions[sym] = ep
                        except Exception:
                            pass
            loaded_syms = set(p.get('symbol') for p in self.loaded_positions if p.get('symbol'))
            ex_syms = set(ex_positions.keys()) if isinstance(ex_positions, dict) else set()
            if loaded_syms != ex_syms:
                self.entries_blocked = self.config.mode == "live"
                logger.warning(f"Restart reconcile mismatch: loaded={loaded_syms} exchange={ex_syms}")
                return
            rm = self.risk_manager
            for pp in self.loaded_positions:
                sym = pp.get('symbol')
                if not sym:
                    continue
                pos = dict(pp)
                if pos.get('entry_time'):
                    try:
                        pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
                    except Exception:
                        pass
                rm.open_positions[sym] = pos
            path = self._state_path()
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    self.position_targets = data.get('position_targets', self.position_targets)
                    self.position_entry_cycle = data.get('position_entry_cycle', self.position_entry_cycle)
                except Exception:
                    pass
            if os.environ.get('RECONCILE_CONFIRM') == '1':
                self.entries_blocked = False
            logger.info("State loaded")
        except Exception as e:
            logger.error(f"reconcile failed: {e}")

    def run(self):
        self.write_run_metadata()
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

    def write_run_metadata(self):
        """Record the resolved run settings before trading starts."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                check=True, timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
        metadata = {
            "exchange": self.config.exchange.name,
            "symbols": list(self.config.scalping.symbols),
            "strategy": self.config.strategy.active_strategy,
            "maker_fee": self.config.backtest.maker_fee,
            "taker_fee": self.config.backtest.taker_fee,
            "slippage": self.config.backtest.slippage,
            "hold_minutes": self.config.risk.max_hold_time_minutes,
            "git_commit": commit,
            "start_time": getattr(self, "clock", datetime.now)().isoformat(),
        }
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        logger.info(
            "Run costs: maker_fee=%.6f taker_fee=%.6f slippage=%.6f",
            metadata["maker_fee"], metadata["taker_fee"], metadata["slippage"],
        )


def build_config_from_args(argv=None, environ=None):
    parser = argparse.ArgumentParser(description="Unified Crypto Scalper")
    parser.add_argument("mode_arg", nargs="?", choices=("paper", "live"), default="paper")
    parser.add_argument("--mode", dest="mode_option", choices=("paper", "live"))
    parser.add_argument("--demo", action="store_true", help="Run in paper mode")
    parser.add_argument("--parity", action="store_true", help="Use deterministic paper/backtest parity settings")
    parser.add_argument("--strategy", help="Select the strategy by registered name")
    parser.add_argument("--symbols", nargs="+", help="Fixed symbols separated by spaces or commas")
    parser.add_argument("--fresh", action="store_true", help="Start paper mode without restoring persisted positions")
    parser.add_argument("--hold-minutes", type=float, default=30.0)
    parser.add_argument("--maker-fee", type=float, default=0.0002,
                        help="Maker fee as a fraction of notional (default: 0.0002)")
    parser.add_argument("--taker-fee", type=float, default=0.0005,
                        help="Taker fee as a fraction of notional (default: 0.0005)")
    parser.add_argument("--slippage", type=float, default=0.0003,
                        help="Slippage as a fraction of notional (default: 0.0003)")
    parser.add_argument("--long-only", action="store_true", help="Skip SHORT signals")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--exchange", help="Pin candle data to this CCXT exchange")
    args = parser.parse_args(argv)

    requested_mode = args.mode_option or args.mode_arg
    if args.demo:
        requested_mode = "paper"
    if args.fresh and requested_mode != "paper":
        raise SystemExit("--fresh is available only in paper mode.")
    if args.parity and requested_mode != "paper":
        raise SystemExit("--parity is available only in paper mode.")
    elif requested_mode == "live":
        live_arg = args.mode_arg == "live" or args.mode_option == "live"
        env = os.environ if environ is None else environ
        if not live_arg or env.get("CONFIRM_LIVE") != "YES":
            raise SystemExit("Live mode requires the explicit 'live' argument and CONFIRM_LIVE=YES.")

    cfg = BotConfig()
    cfg.mode = requested_mode
    cfg.fresh = args.fresh
    for name in ("maker_fee", "taker_fee", "slippage"):
        value = getattr(args, name)
        if value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")
        setattr(cfg.backtest, name, value)
    cfg.long_only = args.long_only
    if args.strategy:
        cfg.strategy.active_strategy = args.strategy
    if args.exchange:
        cfg.exchange.name = args.exchange.lower()
    if args.symbols:
        cfg.scalping.symbols = [symbol.strip() for value in args.symbols
                                for symbol in value.split(",") if symbol.strip()]
    if args.parity:
        if args.capital <= 0 or args.hold_minutes <= 0:
            raise SystemExit("--capital and --hold-minutes must be greater than zero in parity mode.")
        cfg.parity = True
        cfg.risk.trailing_stop_enabled = False
        cfg.risk.breakeven_activation_pct = 0.0
        cfg.risk.trailing_stop_activation_pct = 0.0
        cfg.scalping.scan_all_coins = False
        cfg.scalping.symbols = list(cfg.scalping.symbols)
        cfg.risk.max_hold_time_minutes = args.hold_minutes
        cfg.execution.paper_partial_fill_chance = 0.0
        cfg.execution.paper_reject_chance = 0.0
        cfg.scalping.initial_capital = args.capital
    if cfg.mode == "live" and (not cfg.exchange.api_key or not cfg.exchange.api_secret):
        raise SystemExit("API key or secret missing for live mode")
    return cfg


if __name__ == "__main__":
    bot = UnifiedCryptoBot(build_config_from_args())
    bot.run()



