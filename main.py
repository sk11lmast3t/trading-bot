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
import math
import os
import json
import re
import sys
import threading
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import BotConfig
from core.indicators import Indicators
from core.strategies import get_strategy, Signal
from core.exchange import ExchangeConnector, PaperExchange, OrderState
from core.risk_manager import (
    DEFAULT_MAX_STOP_DISTANCE_PCT,
    DEFAULT_RECOVERY_BAND_PCT,
    RiskManager,
    finite_float,
    normalize_side,
    plausible_protective_stop,
    validate_trailing_stop,
)
from core import state
from core.adaptive_leverage import build_paper_trade_plan, portfolio_snapshot
from core.strategy_engine import (
    STRATEGIES, StrategyEngine, estimate_reward_cost,
    is_timeframe_compatible, resolve_selection, timeframe_minutes,
)
import ccxt

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
    "ts", "symbol", "strategy", "timeframe", "signal", "intended_entry", "sl", "tp",
    "status", "fill_price", "fee", "slippage", "exit_price", "exit_reason",
    "hold_minutes", "net_pnl", "balance_after", "quantity", "notional",
    "atr_1h_pct", "ema20_1h_slope_sign", "ema20_1h_slope_pct", "trend_30m",
    "rsi_5m", "ema_distance_atr_5m", "hour_utc", "day_of_week_utc",
    "rv_24h_bucket", "mfe_bps", "mae_bps", "bars_held", "planned_sl_bps",
    "planned_tp_bps", "r_multiple", "entry_fill_diff_bps",
    "intended_exit", "trigger_level", "exit_fill_diff_bps", "fee_bps", "slippage_bps", "close_source",
    "signal_confidence", "gross_reward_pct", "round_trip_cost_pct",
    "expected_net_reward_pct", "minimum_expected_net_reward_pct",
    "gross_pnl", "win_probability", "probability_lower_bound",
    "historical_sample_size", "break_even_probability", "expected_value_after_costs",
    "selected_leverage", "account_equity", "required_margin", "planned_stop_loss_risk",
    "remaining_portfolio_exposure", "used_margin", "available_margin",
    "available_account_equity", "remaining_margin_capacity", "portfolio_heat",
    "portfolio_heat_pct", "total_margin_used",
    "gross_exposure_pct", "strategy_skip_reasons",
    "total_gross_exposure", "unrealized_pnl", "funding_cost",
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
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if pd.notna(number) and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _finite_float(value):
    """Return any finite float (including zero/negative), or None."""
    return _number_or_none(value)


def _positive_float(value):
    """Return a finite positive float, or None for missing/invalid input."""
    number = _number_or_none(value)
    return number if number is not None and number > 0 else None


def _non_negative_int(value, default=0):
    """Return a non-negative integer, rejecting booleans and junk."""
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _bool_value(value):
    """Interpret snapshot booleans without treating junk or 0.5 as True."""
    parsed = _bool_or_none(value)
    return parsed if parsed is not None else False


def _bool_or_none(value):
    """Return a real boolean for recognised flags, or None when unusable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, float):
        return value == 1.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return None


def _most_protective_stop(side, *stops):
    """Return the tightest valid protective stop for the given side."""
    valid = [stop for stop in stops if stop is not None]
    if not valid:
        return None
    return max(valid) if side == "long" else min(valid)


def _plausible_take_profit(value, entry, side, band=0.5):
    """Validate a restored take-profit against entry, side and a wide band."""
    target = _positive_float(value)
    entry_v = _positive_float(entry)
    if target is None or entry_v is None or side not in ("long", "short"):
        return None
    limit = _finite_float(band)
    if limit is None or limit <= 0 or limit >= 1:
        limit = 0.5
    if side == "long":
        return target if entry_v < target <= entry_v * (1.0 + limit) else None
    return target if entry_v * (1.0 - limit) <= target < entry_v else None


def _position_side(position):
    """Return 'long'/'short' for a position, or None when the side is unusable."""
    try:
        return normalize_side(position.get("side") if isinstance(position, dict) else None)
    except (TypeError, ValueError):
        return None


def _candle_price(frame, symbol, *, now, timeframe, reference=None,
                  max_outlier_pct=0.35):
    """Return a trustworthy last close, or None for stale/outlier data.

    A stale timestamp (older than ``max(3 bars, 30 minutes)``) or a close that
    is implausibly far from the known reference price is rejected so a single
    bad candle cannot move trailing stops, equity or portfolio heat.
    """
    if frame is None or len(frame) == 0:
        return None
    try:
        price = _positive_float(frame.iloc[-1]["close"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if price is None:
        logger.error("[%s] Rejected candle with a non-finite or non-positive close", symbol)
        return None
    idx = _utc_datetime_index(frame)
    if idx is not None:
        minutes = timeframe_minutes(timeframe) or 5
        max_age = pd.Timedelta(minutes=max(3.0 * minutes, 30.0))
        try:
            now_ts = pd.Timestamp(now)
            now_ts = (now_ts.tz_localize("UTC") if now_ts.tzinfo is None
                      else now_ts.tz_convert("UTC"))
            last_ts = idx[-1]
        except (TypeError, ValueError):
            return None
        if now_ts - last_ts > max_age:
            logger.error("[%s] Rejected stale candle at %s (age %s)", symbol,
                         last_ts, now_ts - last_ts)
            return None
        if last_ts - now_ts > pd.Timedelta(minutes=minutes):
            logger.error("[%s] Rejected future candle at %s", symbol, last_ts)
            return None
    ref = _positive_float(reference)
    if ref is not None:
        limit = _finite_float(max_outlier_pct)
        if limit is None or limit <= 0 or limit >= 1:
            limit = 0.35
        if not (ref * (1.0 - limit) <= price <= ref * (1.0 + limit)):
            logger.error(
                "[%s] Rejected outlier candle close %.8f (reference %.8f, band %.0f%%)",
                symbol, price, ref, limit * 100.0)
            return None
    return price


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
        session_db = getattr(self.config, "state_db_path", None)
        if session_db and self.config.mode == "paper":
            state.set_database(session_db)
        clock_source = clock or getattr(self.config, "clock", None)
        self.clock = clock_source or datetime.now
        self._clock_uses_local_timezone = clock_source is None
        self.strategy = get_strategy(self.config)
        selection = getattr(self.config, "strategy_names", None)
        self.strategy_engine = (
            StrategyEngine(selection, self.config.timeframes, self.config)
            if selection else None
        )
        for warning in getattr(self.config, "timeframe_warnings", []):
            logger.warning("Timeframe selection: %s", warning)
        self.risk_manager = RiskManager(self.config, clock=clock_source)
        self.initial_portfolio_equity = float(self.config.scalping.initial_capital)
        self.portfolio_target_status = (
            "ACTIVE" if self.config.portfolio_profit_target_enabled else "DISABLED"
        )
        self.trading_halted = False
        self.gru_predictor = None
        if self.config.strategy.use_gru_filter:
            from core.gru_predictor import GRUPredictor
            self.gru_predictor = GRUPredictor(
                model_path=self.config.strategy.gru_model_path,
                scaler_path=self.config.strategy.gru_scaler_path,
                lookback=self.config.strategy.gru_lookback
            )
        self.running = False
        self.max_cycles = getattr(self.config, "max_cycles", None)
        self.cycle_count = 0
        self.position_entry_cycle = {}
        self.position_targets = {}
        self.entries_blocked = False
        self.entries_blocked_by_marks = False
        self.mark_unavailable_symbols = set()
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
        if self.config.mode == "paper":
            self.persist_state()

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
        if getattr(self, "strategy_engine", None):
            console.print(
                f"Strategy engine: {len(self.strategy_engine.strategy_names)} strategies x "
                f"{len(self.strategy_engine.timeframes)} intervals | "
                f"{len(self.strategy_engine.strategy_names) * len(self.strategy_engine.timeframes)} combinations"
            )
            for label, stats in state.strategy_performance().items():
                console.print(
                    f"{label}: n={stats['completed']} OOS={stats['oos_validated']} "
                    f"win={stats['win_rate']:.1%} expectancy={stats['net_expectancy']:.4%} "
                    f"PF={stats['profit_factor']:.2f}"
                )
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

    def _signal_timestamp_utc(self, value):
        """Signal-bar timestamps are always UTC (CCXT candle opens)."""
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def _entry_timestamp_utc(self, value):
        """Entry times come from self.clock(), which may be naive local time."""
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC")
        if getattr(self, "_clock_uses_local_timezone", False):
            local_zone = datetime.now().astimezone().tzinfo
            return ts.tz_localize(local_zone).tz_convert("UTC")
        return ts.tz_localize("UTC")

    def _elapsed_minutes(self, start, *, signal=False):
        if start is None:
            return None
        start_utc = (self._signal_timestamp_utc(start) if signal
                     else self._entry_timestamp_utc(start))
        return max((self._clock_utc() - start_utc).total_seconds() / 60.0, 0.0)

    def process_symbol(self, symbol: str):
        if self.config.mode == "paper" and getattr(self, "trading_halted", False):
            return
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
            # 1-3. Fetch completed bars, calculate indicators once, and arbitrate
            # selected strategy/timeframe signals. The legacy single strategy path remains.
            selected_timeframe = self.config.scalping.timeframe
            strategy_engine = getattr(self, "strategy_engine", None)
            if strategy_engine:
                setup, df, selected_timeframe, completed_bar = strategy_engine.evaluate(
                    self.exchange, symbol)
                if df is None or completed_bar is None:
                    return
            else:
                lookback = int(getattr(self.strategy, "lookback_bars", 300))
                fetch_limit = max(201, lookback + 2)
                df = self.exchange.fetch_ohlcv(symbol, selected_timeframe, fetch_limit)
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
                df = Indicators.calculate_all(df, self.config)
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
                "signal_confidence": setup.confidence,
            }
            if not strategy_engine:
                decision.update(self._decision_context(symbol, completed_bar, df))
            decision["timeframe"] = selected_timeframe
            if strategy_engine:
                decision["strategy_skip_reasons"] = json.dumps(
                    strategy_engine.last_skip_reasons.get(symbol, {}), sort_keys=True)
            entry_reference = float(setup.entry_price or 0.0)
            decision["planned_sl_bps"] = (abs(entry_reference - float(setup.stop_loss)) / entry_reference * 10000.0
                                           if entry_reference else "")
            decision["planned_tp_bps"] = (abs(entry_reference - float(setup.take_profit)) / entry_reference * 10000.0
                                           if entry_reference else "")
            if setup.signal == Signal.HOLD:
                skip_reason = (getattr(strategy_engine, "last_rejection_reason", {}).get(symbol)
                               if strategy_engine else None)
                finish_decision(f"skipped:{skip_reason}" if skip_reason else "skipped:hold")
                return
            if getattr(self.config, "long_only", False) and setup.signal == Signal.SHORT:
                finish_decision("skipped:long_only")
                return
            if setup.signal in (Signal.LONG, Signal.SHORT):
                cost_estimate = estimate_reward_cost(setup, self.config)
                decision.update(cost_estimate)
                if not cost_estimate["allowed"]:
                    finish_decision(
                        f"skipped:cost:net_reward_{cost_estimate['expected_net_reward_pct']:.4f}%"
                    )
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
            open_position_limit = self.config.risk.max_open_positions
            if (self.config.adaptive_leverage.enabled or
                    self.config.portfolio_profit_target_enabled):
                open_position_limit = min(open_position_limit, 1)
            if len(self.risk_manager.open_positions) >= open_position_limit:
                logger.info(f"[{symbol}] {setup.signal.value} skipped: max open positions reached")
                finish_decision("skipped:max_positions")
                return

            # 7. Position Sizing
            atr = df.get('atr', pd.Series([0])).iloc[-1] if 'atr' in df.columns else None
            adaptive_plan = None
            selected_leverage = 1.0
            if getattr(self.config.adaptive_leverage, "enabled", False):
                strategy_name = setup.strategy_name or getattr(self.strategy, "name", "")
                history = state.load_strategy_trades(
                    strategy_name, self.config.adaptive_leverage.history_window,
                    before=completed_bar)
                adaptive_plan = build_paper_trade_plan(
                    self.config, self.risk_manager, history,
                    strategy=strategy_name, entry=setup.entry_price,
                    stop=setup.stop_loss,
                    side="long" if setup.signal == Signal.LONG else "short",
                    signal_confidence=setup.confidence,
                )
                if not adaptive_plan.get("allowed"):
                    finish_decision(f"skipped:adaptive_risk:{adaptive_plan.get('reason')}")
                    return
                size = adaptive_plan["quantity"]
                selected_leverage = adaptive_plan["leverage"]
                portfolio = adaptive_plan["portfolio"]
                estimate = adaptive_plan["estimate"]
                decision.update({
                    "win_probability": (estimate.win_probability
                                        if estimate.win_probability is not None else 0.5),
                    "probability_lower_bound": estimate.probability_lower_bound,
                    "historical_sample_size": estimate.sample_size,
                    "break_even_probability": estimate.break_even_probability,
                    "expected_value_after_costs": estimate.expected_value,
                    "selected_leverage": selected_leverage,
                    "account_equity": portfolio.equity,
                    "required_margin": adaptive_plan["required_margin"],
                    "planned_stop_loss_risk": adaptive_plan["planned_stop_risk"],
                    "total_margin_used": portfolio.used_margin,
                    "available_account_equity": portfolio.available_balance,
                    "remaining_margin_capacity": max(0.0, portfolio.maximum_margin - portfolio.used_margin),
                    "remaining_portfolio_exposure": portfolio.remaining_exposure,
                    "used_margin": portfolio.used_margin,
                    "available_margin": portfolio.available_margin,
                    "total_gross_exposure": portfolio.gross_exposure,
                    "unrealized_pnl": portfolio.unrealized_pnl,
                })
            else:
                size = self.risk_manager.calculate_position_size(
                    self.risk_manager.current_balance, setup.entry_price, setup.stop_loss, atr
                )
            if size <= 0:
                finish_decision("skipped:zero_size")
                return

            # 8. Portfolio Heat Check
            notional_value = setup.entry_price * size
            max_position_pct = float(self.config.risk.max_position_size_pct)
            max_position_notional = self.risk_manager.current_balance * max_position_pct
            if notional_value > max_position_notional + 1e-9:
                logger.info("[%s] %s skipped: position notional %.8f exceeds configured maximum %.8f",
                            symbol, setup.signal.value, notional_value, max_position_notional)
                finish_decision("skipped:max_position_notional")
                return
            if self.config.mode == "paper":
                stop_ok, stop_reason, planned_stop_risk = self.risk_manager.check_stop_loss_risk(
                    setup.entry_price, setup.stop_loss, size,
                    "long" if setup.signal == Signal.LONG else "short")
                decision["planned_stop_loss_risk"] = planned_stop_risk
                if not stop_ok:
                    finish_decision(f"skipped:stop_risk:{stop_reason}")
                    return
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
                f" | Leverage: {selected_leverage:.2f}x"
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
                if self.config.mode == "live" and getattr(entry_order, "state", None) == OrderState.UNKNOWN:
                    self._block_unknown_order(symbol, "maker entry")
                    finish_decision("unknown:maker_entry_order")
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
                    if self.config.mode == "live" and status == OrderState.UNKNOWN:
                        self._block_unknown_order(symbol, "maker entry")
                        finish_decision("unknown:maker_entry_order")
                        return
                    if self.config.mode == "live":
                        status = getattr(status, "value", status)
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
                if entry_order is not None and getattr(entry_order, "state", None) == OrderState.UNKNOWN:
                    self._block_unknown_order(symbol, "entry")
                    finish_decision("unknown:entry_order")
                    return
                if entry_order is None or getattr(entry_order, 'state') != OrderState.FILLED:
                    logger.error(f"[{symbol}] Entry order not filled: state={getattr(entry_order,'state',None)}, error={getattr(entry_order,'last_error',None)}")
                    finish_decision("skipped:entry_not_filled")
                    return
                opp_side = 'sell' if side == 'buy' else 'buy'
                sl_order = self.exchange.create_stop_loss(symbol, opp_side, size, setup.stop_loss)
                audit(symbol, opp_side, size, setup.stop_loss, getattr(sl_order,'id',None) or getattr(sl_order,'order_id',None), 'send')
                if sl_order is not None and getattr(sl_order, "state", None) == OrderState.UNKNOWN:
                    self._block_unknown_order(symbol, "stop")
                    finish_decision("unknown:stop_order")
                    return
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
                order_params = {'price': setup.entry_price}
                if adaptive_plan is not None:
                    order_params["leverage"] = selected_leverage
                entry_order = self.exchange.create_market_order(symbol, side, size, order_params)
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
                    "adaptive_plan": adaptive_plan,
                    "selected_leverage": selected_leverage,
                }
                self.paper_trade_meta = meta

            actual_margin = notional_value / selected_leverage
            if self.config.mode == "live":
                self.risk_manager.register_position(symbol, notional_value, side)
                state.save_position_at_entry(
                    symbol, side, filled_amount, fill_price, setup.take_profit,
                    setup.stop_loss, setup.strategy_name,
                    entry_time=getattr(self, "clock", datetime.now)().isoformat(),
                )
            else:
                self.risk_manager.register_position(
                    symbol, notional_value, side, leverage=selected_leverage,
                    margin=actual_margin, quantity=filled_amount,
                    strategy=setup.strategy_name, stop_price=setup.stop_loss,
                )
                state.save_position_at_entry(
                    symbol, side, filled_amount, fill_price, setup.take_profit,
                    setup.stop_loss, setup.strategy_name,
                    entry_time=getattr(self, "clock", datetime.now)().isoformat(),
                    leverage=selected_leverage, margin=actual_margin,
                )
            self.risk_manager.update_position_price(symbol, fill_price)
            self.position_entry_cycle[symbol] = self.cycle_count
            signal_bar_open = pd.Timestamp(decision["ts"])
            self.position_targets[symbol] = {
                'tp': setup.take_profit,
                'sl': setup.stop_loss,
                'strategy': setup.strategy_name,
                'timeframe': selected_timeframe,
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
            if self.config.mode == "paper":
                self.persist_state()

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
        all_targets = getattr(self, "position_targets", {})
        targets = all_targets.get(symbol, {}) if isinstance(all_targets, dict) else {}
        if not isinstance(targets, dict):
            targets = {}
        active = getattr(self, "active_decisions", {})
        context = (active.get(symbol) if isinstance(active, dict) else None)
        if context is None:
            context = targets.get("journal_entry")
        if not isinstance(context, dict):
            return
        try:
            context.update(self._exit_excursions(symbol, pos, frame, context))
            context["close_source"] = "poll"
            context["intended_exit"] = intended_exit
            context["trigger_level"] = intended_exit
        except Exception as exc:
            logger.debug("Could not record exit observations for %s: %s", symbol, exc)

    def _record_exchange_stop_flat(self, symbol):
        if getattr(self.config, "mode", "paper") != "live":
            return
        all_targets = getattr(self, "position_targets", {})
        targets = all_targets.get(symbol, {}) if isinstance(all_targets, dict) else {}
        if not isinstance(targets, dict):
            targets = {}
        active = getattr(self, "active_decisions", {})
        context = (active.get(symbol) if isinstance(active, dict) else None)
        if not isinstance(context, dict):
            context = targets.get("journal_entry")
        if not isinstance(context, dict) or not context:
            position = getattr(self.risk_manager, "open_positions", {}).get(symbol, {})
            context = {
                "ts": position.get("entry_time", ""), "symbol": symbol,
                "strategy": position.get("strategy", ""),
                "signal": "SHORT" if position.get("side") == "short" else "LONG",
                "fill_price": position.get("entryPrice", position.get("entry_price", "")),
                "quantity": position.get("contracts", position.get("size", "")),
                "entry_time": position.get("entry_time", ""),
            }
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
            hold_minutes = self._elapsed_minutes(entry_time) or 0.0
        elif context.get("entry_time"):
            try:
                entered = datetime.fromisoformat(context["entry_time"])
                hold_minutes = self._elapsed_minutes(entered) or 0.0
            except (TypeError, ValueError):
                pass
        values = {
            "status": "closed", "exit_price": "", "exit_reason": "exchange_stop_or_external_close",
            "intended_exit": targets.get("sl", ""),
            "hold_minutes": hold_minutes, "close_source": "exchange_stop_inferred",
            "exit_fill_diff_bps": "", "fee_bps": "", "slippage_bps": "",
            "r_multiple": "", **observed,
        }
        if context.get("ts") and context.get("strategy"):
            self._update_decision_exit(context, values)
        else:
            self._append_decision({**context, "ts": context.get("ts") or self._clock_utc().isoformat(),
                                   "symbol": symbol, **values})
        getattr(self, "active_decisions", {}).pop(symbol, None)

    def _block_unknown_order(self, symbol, label):
        self.entries_blocked = True
        logger.error("[%s] %s order state UNKNOWN; blocking entries until position reconciliation", symbol, label)
        try:
            self.exchange.get_position(symbol)
        except Exception as exc:
            logger.error("[%s] Position check after unknown %s order failed: %s", symbol, label, exc)

    def _close_position(self, symbol, pos, price, reason, reference_price=None,
                        exit_slippage_rate=None):
        exit_price = _positive_float(price)
        reference_price = _positive_float(
            price if reference_price is None else reference_price)
        if exit_price is None or reference_price is None:
            logger.error("[%s] Refusing to close with invalid price=%r reference=%r; "
                         "blocking new entries", symbol, price, reference_price)
            self.entries_blocked = True
            return
        price = exit_price
        if not isinstance(pos, dict):
            logger.error("[%s] Refusing to close a malformed position %r", symbol, pos)
            self.entries_blocked = True
            return
        if self.config.mode == "live":
            # An exchange stop may have filled between the poll and this close.
            fresh = self.exchange.get_position(symbol)
            if not fresh:
                if getattr(self.exchange, "position_check_error", None):
                    logger.error("[%s] Position recheck failed; leaving state reserved and blocking entries", symbol)
                    self.entries_blocked = True
                    return
                self._record_exchange_stop_flat(symbol)
                self.risk_manager.release_position(symbol)
                self.position_entry_cycle.pop(symbol, None)
                self.position_targets.pop(symbol, None)
                self.persist_state()
                return
            pos = fresh
        entry = _positive_float(pos.get('entryPrice') or pos.get('entry_price'))
        contracts = _positive_float(pos.get('contracts') or pos.get('quantity'))
        side = _position_side(pos)
        if entry is None or contracts is None or side is None:
            logger.error("[%s] Refusing to close malformed position (entry=%r contracts=%r "
                         "side=%r); blocking new entries", symbol,
                         pos.get('entryPrice'), pos.get('contracts'), pos.get('side'))
            self.entries_blocked = True
            return
        decision_context = getattr(self, "active_decisions", {}).pop(symbol, None)
        all_targets = getattr(self, "position_targets", {})
        targets = all_targets.get(symbol, {}) if isinstance(all_targets, dict) else {}
        if not isinstance(targets, dict):
            targets = {}
        if decision_context is not None and not isinstance(decision_context, dict):
            logger.warning("[%s] Ignoring malformed journal decision context %r",
                           symbol, decision_context)
            decision_context = None
        if decision_context is None:
            decision_context = targets.get("journal_entry")
        hold_minutes = ""
        if decision_context and isinstance(decision_context.get("entry_time"), datetime):
            hold_minutes = self._elapsed_minutes(decision_context["entry_time"]) or 0.0
        if side == 'long':
            gross_pnl = (reference_price - entry) * contracts
        else:
            gross_pnl = (entry - reference_price) * contracts
        pnl = gross_pnl
        journal_exit_fill = float(price)
        live_exit_fee = None
        if self.config.mode == "live":
            opp_side = 'sell' if side == 'long' else 'buy'
            co = self.exchange.create_market_order(symbol, opp_side, contracts, reduce_only=True)
            audit(symbol, opp_side, contracts, price, getattr(co,'id',None) or getattr(co,'order_id',None))
            if co is not None and getattr(co, "state", None) == OrderState.REJECTED and any(
                    marker in str(getattr(co, "last_error", "")).lower()
                    for marker in ("position is zero", "position is flat", "no position", "position not found",
                                   "position does not exist", "position size is zero")):
                self._record_exchange_stop_flat(symbol)
                self.risk_manager.release_position(symbol)
                self.position_entry_cycle.pop(symbol, None)
                self.position_targets.pop(symbol, None)
                self.persist_state()
                return
            if co is None or getattr(co, "state", None) != OrderState.FILLED:
                logger.error("[%s] Close fill is not confirmed; blocking entries until position reconciliation", symbol)
                self.entries_blocked = True
                try:
                    self.exchange.get_position(symbol)
                except Exception:
                    pass
                return
            if co is not None:
                journal_exit_fill = _order_fill_price(co, price)
                live_exit_fee = _order_fee(co)
            try:
                self.exchange.cancel_all_orders(symbol)
                audit(symbol, 'cancel', contracts, price, None, 'cancel')
            except Exception:
                pass
        else:
            # Slippage and fees are accounted explicitly below; the simulator
            # must not apply a second random slippage to an already-slipped fill.
            self.exchange.close_position(symbol, price, apply_slippage=False)
            audit(symbol, 'close', contracts, price, None, 'fill')
            meta = getattr(self, "paper_trade_meta", {}).pop(symbol, {})
            if not isinstance(meta, dict):
                meta = {}
            taker_fee = max(0.0, _finite_float(self.config.backtest.taker_fee) or 0.0)
            exit_fee = reference_price * contracts * taker_fee
            if exit_slippage_rate is None:
                exit_slippage_rate = self.config.backtest.slippage
            exit_slippage_rate = max(0.0, _finite_float(exit_slippage_rate) or 0.0)
            exit_slippage = reference_price * contracts * exit_slippage_rate
            entry_fee = _finite_float(meta.get("entry_fee"))
            entry_fee = entry_fee if entry_fee is not None and entry_fee >= 0 else 0.0
            entry_slippage = _finite_float(meta.get("entry_slippage"))
            entry_slippage = (entry_slippage if entry_slippage is not None
                              and entry_slippage >= 0 else 0.0)
            decision_fee = entry_fee + exit_fee
            decision_slippage = entry_slippage + exit_slippage
            if hold_minutes == "" and meta.get("entry_time"):
                hold_minutes = self._elapsed_minutes(meta["entry_time"]) or 0.0
            hold_for_funding = float(hold_minutes or 0.0)
            funding_rate = abs(_finite_float(
                getattr(self.config.backtest, "funding_rate", 0.0)) or 0.0)
            funding_cost = (entry * contracts * funding_rate *
                            hold_for_funding / 480.0)
            pnl = (gross_pnl - entry_fee - entry_slippage - exit_fee
                   - exit_slippage - funding_cost)
            if _finite_float(pnl) is None:
                logger.error("[%s] Refusing to record non-finite close PnL %r",
                             symbol, pnl)
                self.entries_blocked = True
                return
        logger.info(
            f"[{symbol}] Closed {side.upper()} position: {reason} | "
            f"gross PnL: ${gross_pnl:.4f} | net PnL: ${pnl:.4f}"
        )
        plan = meta.get("adaptive_plan") if self.config.mode != "live" else None
        adaptive_estimate = plan.get("estimate") if isinstance(plan, dict) else None
        if self.config.mode == "live":
            self.risk_manager.record_trade(pnl, symbol)
            state.record_trade(pnl, symbol, reason)
        else:
            net_return = pnl / (entry * contracts) if entry * contracts else None
            self.risk_manager.record_trade(
                pnl, symbol, strategy=meta.get("strategy", ""), net_return=net_return,
                selected_leverage=meta.get("selected_leverage", 1.0),
                expected_value=(adaptive_estimate.expected_value if adaptive_estimate else None),
            )
            state.record_trade(
                pnl, symbol, reason, strategy=meta.get("strategy", ""),
                notional=(entry * contracts if entry * contracts else None),
                net_return=net_return,
                prediction_probability=(adaptive_estimate.win_probability if adaptive_estimate else None),
                probability_lower_bound=(adaptive_estimate.probability_lower_bound if adaptive_estimate else None),
                break_even_probability=(adaptive_estimate.break_even_probability if adaptive_estimate else None),
                expected_value=(adaptive_estimate.expected_value if adaptive_estimate else None),
                selected_leverage=_positive_float(meta.get("selected_leverage")) or 1.0,
                required_margin=(entry * contracts /
                                 max(_positive_float(meta.get("selected_leverage")) or 1.0, 1.0)),
                funding_cost=funding_cost,
                fees=decision_fee,
                slippage=decision_slippage,
                gross_pnl=gross_pnl,
                oos_validated=bool(adaptive_estimate is not None),
            )
            state.close_position(symbol)
        if decision_context:
            entry_fill = (_positive_float(decision_context.get("fill_price"))
                          or _positive_float(decision_context.get("entry_fill_price"))
                          or entry)
            notional = (_positive_float(decision_context.get("notional"))
                        or entry_fill * contracts)
            stop_reference = (_positive_float(targets.get("sl"))
                              or _positive_float(decision_context.get("sl")))
            planned_risk = (abs(entry_fill - stop_reference) * contracts
                            if stop_reference is not None else 0.0)
            journal_intended_exit = (_positive_float(decision_context.get("intended_exit"))
                                     or reference_price)
            if self.config.mode != "live":
                fee_bps = decision_fee / notional * 10000.0 if notional else ""
            else:
                entry_fee = _number_or_none(decision_context.get("entry_live_fee"))
                fee_bps = ((entry_fee + live_exit_fee) / notional * 10000.0
                           if entry_fee is not None and live_exit_fee is not None and notional else "")
            if self.config.mode != "live" and notional:
                slip_bps = decision_slippage / notional * 10000.0
            else:
                entry_diff = _finite_float(decision_context.get("entry_fill_diff_bps")) or 0.0
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
                "gross_pnl": gross_pnl,
                "funding_cost": funding_cost if self.config.mode != "live" else "",
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
            if not isinstance(pos, dict):
                logger.error("[%s] Exchange returned a malformed position %r; "
                             "blocking new entries", symbol, pos)
                self.entries_blocked = True
                return

            all_targets = getattr(self, 'position_targets', {})
            if not isinstance(all_targets, dict):
                logger.error("[%s] position_targets container is malformed (%s); "
                             "ignoring all targets", symbol, type(all_targets).__name__)
                all_targets = {}
            targets = all_targets.get(symbol, {})
            if not isinstance(targets, dict):
                logger.error("[%s] position_targets entry is malformed (%r); "
                             "continuing with protective exits", symbol, targets)
                targets = {}

            rm_pos = self.risk_manager.open_positions.get(symbol, {})
            if not isinstance(rm_pos, dict):
                logger.error("[%s] Risk position is malformed (%r); treating it as empty",
                             symbol, rm_pos)
                rm_pos = {}

            side = _position_side(pos) or _position_side(rm_pos)
            if side is None:
                logger.error("[%s] Cannot evaluate exits with invalid side=%r",
                             symbol, pos.get('side'))
                self.entries_blocked = True
                return
            entry = _positive_float(pos.get('entryPrice')
                                    or pos.get('entry_price')
                                    or rm_pos.get('entry_price'))
            if entry is None:
                logger.error("[%s] Cannot evaluate exits with invalid entry price %r; "
                             "blocking new entries", symbol,
                             pos.get('entryPrice', pos.get('entry_price')))
                self.entries_blocked = True
                return

            signal_bar_open = targets.get('signal_bar_open')
            exit_timeframe = targets.get("timeframe")
            if not isinstance(exit_timeframe, str) or not timeframe_minutes(exit_timeframe):
                strategy_label = targets.get("strategy")
                strategy_label = strategy_label if isinstance(strategy_label, str) else ""
                suffix = strategy_label.rsplit("@", 1)
                exit_timeframe = (suffix[1] if len(suffix) == 2 and timeframe_minutes(suffix[1])
                                  else self.config.scalping.timeframe)
            interval_minutes = timeframe_minutes(exit_timeframe) or 5
            limit = 5
            if signal_bar_open:
                try:
                    elapsed = max((self._clock_utc() -
                                   self._signal_timestamp_utc(signal_bar_open)).total_seconds(), 0)
                    limit = max(limit, int(elapsed // (interval_minutes * 60)) + 3)
                except (TypeError, ValueError, OverflowError):
                    logger.warning("[%s] Ignoring malformed signal_bar_open=%r",
                                   symbol, signal_bar_open)
            df = self.exchange.fetch_ohlcv(symbol, exit_timeframe, limit)
            if df is None or len(df) == 0:
                return

            price = _candle_price(df, symbol, now=self._clock_utc(),
                                  timeframe=exit_timeframe, reference=entry)
            if price is None:
                # A stale or corrupt candle must never move stops or equity.
                self.entries_blocked = True
                return
            update_mark = getattr(self.exchange, "update_mark_price", None)
            if callable(update_mark):
                update_mark(symbol, float(price))
            sc = self.config.scalping
            self.risk_manager.update_position_price(symbol, price)

            band, max_distance = self._recovery_limits()
            # The restored SQLite original stop is authoritative. Snapshot
            # targets are only a fallback, and the configuration percentage is
            # the last resort; neither may weaken a valid restored level.
            original = None
            for key in ('original_stop', 'stop_price', 'sl'):
                original = plausible_protective_stop(
                    rm_pos.get(key), entry, side, max_distance)
                if original is not None:
                    break
            target_sl = plausible_protective_stop(
                targets.get('sl'), entry, side, max_distance)
            if original is not None:
                sl_price = original
            elif target_sl is not None:
                sl_price = target_sl
            else:
                sl_price = (entry * (1 - sc.stop_loss_pct) if side == 'long'
                            else entry * (1 + sc.stop_loss_pct))
                logger.warning("[%s] No valid restored or target stop; using the "
                               "%.4f%% configuration fallback at %.8f",
                               symbol, sc.stop_loss_pct * 100.0, sl_price)

            rm_tp = _plausible_take_profit(rm_pos.get('tp'), entry, side)
            target_tp = _plausible_take_profit(targets.get('tp'), entry, side)
            if rm_tp is not None:
                tp_price = rm_tp
            elif target_tp is not None:
                tp_price = target_tp
            else:
                tp_price = (entry * (1 + sc.take_profit_pct) if side == 'long'
                            else entry * (1 - sc.take_profit_pct))
                logger.warning("[%s] No valid restored or target take-profit; using "
                               "the %.4f%% configuration fallback at %.8f",
                               symbol, sc.take_profit_pct * 100.0, tp_price)
            if _positive_float(sl_price) is None or _positive_float(tp_price) is None:
                logger.error("[%s] Cannot evaluate exits: stop=%r take-profit=%r",
                             symbol, sl_price, tp_price)
                self.entries_blocked = True
                return

            candle_index = _utc_datetime_index(df)
            if candle_index is not None:
                candles = df.copy()
                candles.index = candle_index
                if signal_bar_open:
                    try:
                        candles = candles.loc[
                            candles.index > self._signal_timestamp_utc(signal_bar_open)]
                    except (TypeError, ValueError):
                        logger.warning("[%s] Could not filter candles by signal_bar_open",
                                       symbol)
            elif len(df) > 1:
                # Compatibility with exchanges/tests that return no candle timestamps.
                candles = df.iloc[1:]
            else:
                candles = df
            for _, candle in candles.iterrows():
                try:
                    close = float(candle.get('close', candle['close']))
                    high = float(candle.get('high', close))
                    low = float(candle.get('low', close))
                    bar_open = float(candle.get('open', close))
                except (TypeError, ValueError):
                    logger.warning("[%s] Skipping candle with malformed OHLC values", symbol)
                    continue
                if not all(math.isfinite(v) and v > 0
                           for v in (close, high, low, bar_open)):
                    logger.warning("[%s] Skipping candle with non-finite OHLC values", symbol)
                    continue
                if not (entry * 0.5 <= high <= entry * 1.5
                        and entry * 0.5 <= low <= entry * 1.5):
                    logger.warning("[%s] Skipping implausible candle high=%.8f low=%.8f "
                                   "(entry=%.8f)", symbol, high, low, entry)
                    continue
                hit_sl = low <= sl_price if side == 'long' else high >= sl_price
                hit_tp = high >= tp_price if side == 'long' else low <= tp_price
                if hit_sl or hit_tp:
                    is_sl = hit_sl  # adverse-first when both levels lie inside the bar
                    level = sl_price if is_sl else tp_price
                    # Stop gaps fill at the first available bar price. A target
                    # gap is conservatively filled at its limit, never improved.
                    base_fill = level
                    if is_sl and side == 'long' and bar_open < sl_price:
                        base_fill = bar_open
                    elif is_sl and side != 'long' and bar_open > sl_price:
                        base_fill = bar_open
                    fill = base_fill
                    exit_slippage_rate = 0.0
                    if self.config.mode != "live" and is_sl:
                        # Stops are market orders and pay adverse slippage. A
                        # take-profit is a resting limit: it fills at its price
                        # and is never improved by a favourable gap.
                        slip = max(0.0, _finite_float(self.config.backtest.slippage) or 0.0)
                        exit_slippage_rate = slip
                        fill = base_fill * (1 - slip if side == 'long' else 1 + slip)
                    self._prepare_poll_close(symbol, pos, df, level)
                    self._close_position(
                        symbol, pos, fill,
                        "STOP LOSS" if is_sl else "TAKE PROFIT",
                        reference_price=base_fill if self.config.mode != "live" else fill,
                        exit_slippage_rate=exit_slippage_rate,
                    )
                    return

                # Stop and target checks run first. Liquidation is only a last
                # resort if a bar crosses the maintenance threshold first.
                if (self.config.mode == "paper" and
                        getattr(getattr(self.config, "adaptive_leverage", None),
                                "enabled", False)):
                    liquidation_price = getattr(self.exchange, "liquidation_price", lambda _s: None)(symbol)
                    if liquidation_price is not None:
                        liq_hit = (low <= liquidation_price if side == 'long'
                                   else high >= liquidation_price)
                        if liq_hit:
                            self._prepare_poll_close(symbol, pos, df, liquidation_price)
                            self._close_position(symbol, pos, liquidation_price,
                                                 "Paper liquidation", reference_price=liquidation_price)
                            return

            # 3. Trailing / breakeven (isolated so failure can't skip SL above)
            try:
                trail_result = self.risk_manager.check_trailing_stop(symbol, price)
                if isinstance(trail_result, tuple) and len(trail_result) == 2:
                    trail_hit, trail_reason = trail_result
                else:
                    logger.error("[%s] Trailing check returned a malformed result %r; "
                                 "ignoring it", symbol, trail_result)
                    trail_hit, trail_reason = False, ""
                if trail_hit:
                    self._prepare_poll_close(symbol, pos, df, price)
                    self._close_position(symbol, pos, price, f"TRAILING STOP: {trail_reason}")
                    return
            except Exception as e:
                logger.error(f"Trailing check error for {symbol}: {e}")

            # 4. Time exit (wall-clock)
            entry_time = rm_pos.get('entry_time')
            if entry_time is not None:
                hold_min = self._elapsed_minutes(entry_time) or 0.0
                signal_close = None
                if signal_bar_open:
                    try:
                        signal_close = (self._signal_timestamp_utc(signal_bar_open) +
                                        pd.Timedelta(minutes=interval_minutes))
                    except (TypeError, ValueError):
                        signal_close = None
                elapsed = (max((self._clock_utc() - signal_close).total_seconds() / 60.0, 0.0)
                           if signal_close is not None else hold_min)
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
        stats.update(self.portfolio_target_metrics())
        table = Table(title="[bold yellow]Unified Bot Live Telemetry[/bold yellow]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        labels = {
            "portfolio_heat": "Portfolio heat: stop-loss exposure incl. costs ($)",
            "portfolio_heat_pct": "Portfolio heat (% of equity)",
            "total_margin_used": "Margin used ($)",
            "available_account_equity": "Free account equity after margin ($)",
            "remaining_margin_capacity": "Remaining margin-allocation capacity ($)",
            "available_margin": "Available margin under both caps ($)",
            "total_gross_exposure": "Gross exposure ($)",
            "gross_exposure_pct": "Gross exposure (% of equity)",
            "max_drawdown_pct": "Maximum drawdown (%)",
            "stop_risk_budget": "Stop-risk budget ($)",
            "unpriced_stop_risk_positions": "Open positions missing stop-risk data",
        }
        for k, v in stats.items():
            table.add_row(labels.get(k, k.replace('_', ' ').title()),
                          f"{v:.4f}" if isinstance(v, float) else str(v))
        table.add_row("Cycle Iteration", str(self.cycle_count))
        console.print(table)

    def portfolio_target_metrics(self):
        stats = self.risk_manager.get_stats()
        initial = _positive_float(getattr(self, "initial_portfolio_equity",
                                           self.config.scalping.initial_capital))
        if initial is None:
            initial = _positive_float(self.config.scalping.initial_capital) or 0.0
        current = _finite_float(stats.get("portfolio_equity"))
        if current is None:
            current = _finite_float(self.risk_manager.current_balance) or 0.0
        enabled = bool(self.config.portfolio_profit_target_enabled)
        target_pct = None
        if enabled:
            target_pct = _finite_float(self.config.portfolio_profit_target_pct)
            if target_pct is None:
                enabled = False
        target_equity = initial * (1.0 + target_pct / 100.0) if enabled else None
        balance = _finite_float(self.risk_manager.current_balance) or 0.0
        unrealized = _finite_float(stats.get("unrealized_pnl")) or 0.0
        return {
            "initial_portfolio_equity": initial,
            "current_portfolio_equity": current,
            "realized_pnl_from_start": balance - initial,
            "unrealized_pnl": unrealized,
            "net_portfolio_return_pct": ((current - initial) / initial * 100.0
                                          if initial > 0 else 0.0),
            "portfolio_profit_target_pct": target_pct,
            "remaining_profit_required": max(0.0, target_equity - current)
                if target_equity is not None else None,
            "target_portfolio_equity": target_equity,
            "portfolio_target_status": getattr(self, "portfolio_target_status", "ACTIVE"),
            "trading_halted": bool(getattr(self, "trading_halted", False)),
        }

    def _portfolio_target_report_path(self):
        state_path = self._state_path()
        if state_path.endswith(".state.json"):
            return state_path[:-len(".state.json")] + ".portfolio_report.json"
        root, _ = os.path.splitext(state_path)
        return root + ".portfolio_report.json"

    def _write_portfolio_target_report(self):
        metrics = self.portfolio_target_metrics()
        report = {
            **metrics,
            "target_status": self.portfolio_target_status,
            "target_reached": self.portfolio_target_status == "TARGET_REACHED",
            "generated_at": self._clock_utc().isoformat(),
            "open_paper_positions": len(getattr(self.exchange, "positions", {})),
        }
        path = self._portfolio_target_report_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
        logger.info("Final portfolio performance report: %s", path)
        logger.info("Portfolio target status: %s | final equity: $%.4f | target equity: $%.4f",
                    self.portfolio_target_status,
                    self.risk_manager.current_balance,
                    metrics["target_portfolio_equity"] or 0.0)

    def check_portfolio_profit_target(self):
        """Halt and liquidate paper positions once marked equity reaches its objective."""
        if self.config.mode != "paper" or not self.config.portfolio_profit_target_enabled:
            return False
        if not hasattr(self, "initial_portfolio_equity") or not hasattr(self, "portfolio_target_status"):
            return False
        terminal = {"TARGET_REACHED", "TARGET_NOT_REACHED_AFTER_LIQUIDATION"}
        if self.portfolio_target_status in terminal:
            return True
        metrics = self.portfolio_target_metrics()
        target_equity = metrics["target_portfolio_equity"]
        marked_equity = metrics["current_portfolio_equity"]
        if self.portfolio_target_status != "LIQUIDATING" and marked_equity < target_equity:
            return False

        self.trading_halted = True
        self.portfolio_target_status = "LIQUIDATING"
        self.persist_state()
        cancel_entries = getattr(self.exchange, "cancel_entry_orders", None)
        if callable(cancel_entries):
            cancel_entries()

        positions = dict(getattr(self.exchange, "positions", {}))
        for symbol, position in positions.items():
            if not isinstance(position, dict):
                logger.error("Cannot liquidate %s for portfolio target: malformed position", symbol)
                continue
            entry_price = _positive_float(position.get("entryPrice"))
            mark_price = _positive_float(position.get("markPrice")) or entry_price
            if entry_price is None or mark_price is None:
                logger.error("Cannot liquidate %s for portfolio target: missing valid paper mark", symbol)
                continue
            self._prepare_poll_close(symbol, position, None, mark_price)
            self._close_position(symbol, position, mark_price,
                                 "PORTFOLIO PROFIT TARGET LIQUIDATION",
                                 reference_price=mark_price)
            cancel_all = getattr(self.exchange, "cancel_all_orders", None)
            if callable(cancel_all):
                cancel_all(symbol)

        remaining_positions = getattr(self.exchange, "positions", {})
        if remaining_positions or self.risk_manager.open_positions:
            logger.error("Portfolio target liquidation incomplete; entries remain halted")
            self.persist_state()
            return False

        final_equity = float(self.risk_manager.current_balance)
        self.portfolio_target_status = (
            "TARGET_REACHED" if final_equity >= target_equity
            else "TARGET_NOT_REACHED_AFTER_LIQUIDATION"
        )
        self.trading_halted = True
        self.persist_state()
        self._write_portfolio_target_report()
        return True

    def _state_path(self):
        return getattr(self.config, "state_file_path", 'state.json')

    def persist_state(self):
        try:
            data = {
                'open_positions': {},
                'position_targets': self.position_targets,
                'position_entry_cycle': self.position_entry_cycle,
            }
            if self.config.mode == "paper":
                risk = self.risk_manager
                breaker = getattr(risk, 'circuit_breaker', None)
                data['peak_equity'] = getattr(
                    risk, 'peak_equity', getattr(risk, 'peak_balance', 0.0),
                )
                data['max_drawdown_pct'] = getattr(risk, 'max_drawdown_pct', 0.0)
                data.update({
                    'cycle_count': int(getattr(self, 'cycle_count', 0)),
                    'account_balance': getattr(risk, 'current_balance',
                                               self.config.scalping.initial_capital),
                    'peak_balance': getattr(risk, 'peak_balance',
                                            self.config.scalping.initial_capital),
                    'initial_portfolio_equity': getattr(
                        self, 'initial_portfolio_equity', self.config.scalping.initial_capital),
                    'portfolio_profit_target_pct': self.config.portfolio_profit_target_pct,
                    'portfolio_profit_target_enabled': self.config.portfolio_profit_target_enabled,
                    'portfolio_target_status': getattr(self, 'portfolio_target_status', 'ACTIVE'),
                    'trading_halted': getattr(self, 'trading_halted', False),
                    'consecutive_losses': getattr(risk, 'consecutive_losses', 0),
                    'daily_pnl': getattr(risk, 'daily_pnl', 0.0),
                    'daily_trades': getattr(risk, 'daily_trades', 0),
                    'circuit_breaker': {
                        'halted': getattr(breaker, 'halted', False),
                        'halt_reason': getattr(breaker, 'halt_reason', None),
                        'error_count': getattr(breaker, 'error_count', 0),
                        'window_start': (breaker.window_start.isoformat()
                                         if getattr(breaker, 'window_start', None) else None),
                        'window_low': getattr(breaker, 'window_low', None),
                        'peak_balance': getattr(breaker, 'peak_balance',
                                                self.config.scalping.initial_capital),
                    },
                })
            for sym, pos in getattr(self.risk_manager, 'open_positions', {}).items():
                p = dict(pos)
                if isinstance(p.get('entry_time'), datetime):
                    p['entry_time'] = p['entry_time'].isoformat()
                data['open_positions'][sym] = p
            path = self._state_path()
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            # Write to a sibling temp file and atomically replace the snapshot so
            # a crash mid-write cannot corrupt the only copy of the state.
            fd, temp_path = tempfile.mkstemp(prefix=".state-", suffix=".tmp",
                                             dir=directory)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            logger.info("State persisted")
        except Exception as e:
            logger.error(f"persist_state failed: {e}")

    def _recovery_limits(self):
        risk_cfg = getattr(self.config, "risk", None)
        band = finite_float(getattr(risk_cfg, "recovery_price_band_pct", None))
        if band is None or band <= 0 or band >= 1:
            band = DEFAULT_RECOVERY_BAND_PCT
        distance = finite_float(getattr(risk_cfg, "max_stop_distance_pct", None))
        if distance is None or distance <= 0 or distance >= 1:
            distance = DEFAULT_MAX_STOP_DISTANCE_PCT
        return band, distance

    def _restore_original_stop(self, row, snapshot_position, targets):
        """Pick the authoritative original protective stop and keep it separate.

        The SQLite stop wins when it is plausible for the restored side and
        entry. A corrupt SQLite value is logged and skipped rather than trusted;
        the snapshot/target values are validated the same way. When no level is
        usable the position is left as unpriced risk so new entries are blocked.
        """
        symbol = row.get('symbol') if isinstance(row, dict) else None
        entry = _positive_float((row or {}).get('entry_price')
                                or (row or {}).get('entryPrice'))
        side = _position_side(row) if isinstance(row, dict) else None
        _, max_distance = self._recovery_limits()
        candidates = [("sqlite.sl", (row or {}).get('sl'))]
        if isinstance(snapshot_position, dict):
            candidates.extend([
                ("snapshot.original_stop", snapshot_position.get('original_stop')),
                ("snapshot.stop_price", snapshot_position.get('stop_price')),
                ("snapshot.sl", snapshot_position.get('sl')),
            ])
        if isinstance(targets, dict):
            candidates.append(("position_targets.sl", targets.get('sl')))

        original, source = None, None
        for label, value in candidates:
            if value is None or value == "":
                continue
            if entry is not None and side is not None:
                stop = plausible_protective_stop(value, entry, side, max_distance)
            else:
                stop = _positive_float(value)
            if stop is None:
                logger.warning(
                    "[%s] Ignoring implausible protective stop %s=%r for entry=%r side=%r",
                    symbol, label, value, entry, side)
                continue
            original, source = stop, label
            break
        if original is None:
            if any(value is not None and value != "" for _, value in candidates):
                logger.error(
                    "[%s] Protective stop values exist but none is valid for entry=%r "
                    "side=%r: %s; the position will be treated as unpriced risk",
                    symbol, entry, side, candidates)
            else:
                logger.error(
                    "[%s] No protective stop in SQLite or the JSON snapshot; "
                    "the position will be treated as unpriced risk", symbol)
            return None
        if isinstance(snapshot_position, dict):
            for label, value in candidates[1:3]:
                other = _positive_float(value)
                if other is not None and abs(other - original) > max(1e-12, original * 1e-9):
                    logger.warning(
                        "[%s] Snapshot %s=%.8f differs from authoritative %s "
                        "stop=%.8f; keeping %s", symbol, label, other, source,
                        original, source)
                    break
        logger.info("[%s] Restored original protective stop %.8f from %s",
                    symbol, original, source)
        return original

    def _breakeven_stop(self, entry, side, original=None):
        entry = _positive_float(entry)
        if entry is None:
            return None
        offset = _number_or_none(
            getattr(getattr(self.risk_manager, "rc", None), "round_trip_cost_pct", 0.0))
        offset = max(0.0, float(offset or 0.0))
        breakeven = entry * (1 + offset) if side == "long" else entry * (1 - offset)
        return _most_protective_stop(side, _positive_float(breakeven),
                                     _positive_float(original))

    def _merge_restored_dynamic_state(self, symbol, pos, snapshot_position):
        """Restore trailing/breakeven/watermark state, trusting SQLite identity."""
        entry = _positive_float(pos.get('entry_price')) or 0.0
        side = _position_side(pos)
        if side is None:
            logger.error(
                "[%s] Refusing to restore trailing state with invalid side %r; "
                "the position is treated as unpriced risk and entries are blocked",
                symbol, pos.get('side'))
            self.entries_blocked = True
            pos['trailing_stop'] = None
            pos['breakeven_armed'] = False
            pos['high_watermark'] = None
            pos['low_watermark'] = None
            pos['mark_price'] = None
            pos['last_mark_price'] = None
            return
        quantity = _positive_float(pos.get('quantity')) or 0.0
        band, _ = self._recovery_limits()
        original = _positive_float(pos.get('original_stop'))
        if not isinstance(snapshot_position, dict):
            logger.warning(
                "[%s] No matching JSON snapshot; keeping the SQLite original stop "
                "and starting trailing/breakeven state fresh", symbol)
            pos['high_watermark'] = entry
            pos['low_watermark'] = entry
            pos['trailing_stop'] = None
            pos['breakeven_armed'] = False
            pos['mark_price'] = None
            pos['last_mark_price'] = None
            return

        snap_side = _position_side(snapshot_position)
        if snap_side and snap_side != side:
            logger.warning("[%s] Snapshot side=%s disagrees with SQLite side=%s; keeping SQLite",
                           symbol, snap_side, side)
        snap_qty = _positive_float(snapshot_position.get('quantity'))
        if snap_qty is not None and quantity > 0 and abs(snap_qty - quantity) > max(1e-12, quantity * 1e-9):
            logger.warning(
                "[%s] Snapshot quantity=%.12f disagrees with SQLite quantity=%.12f; keeping SQLite",
                symbol, snap_qty, quantity)
        snap_entry = _positive_float(snapshot_position.get('entry_price'))
        if snap_entry is not None and entry > 0 and abs(snap_entry - entry) > max(1e-12, entry * 1e-9):
            logger.warning(
                "[%s] Snapshot entry=%.8f disagrees with SQLite entry=%.8f; keeping SQLite",
                symbol, snap_entry, entry)

        # Watermarks: a valid high is at or above entry and inside the recovery
        # band; a valid low is at or below entry and inside the band. This
        # rejects booleans, 1e-9 / 1e9 corruption and wrong-direction values.
        raw_high = snapshot_position.get('high_watermark')
        raw_low = snapshot_position.get('low_watermark')
        high = _positive_float(raw_high)
        low = _positive_float(raw_low)
        high_ok = (high is not None and entry > 0
                   and entry <= high <= entry * (1.0 + band))
        low_ok = (low is not None and entry > 0
                  and entry * (1.0 - band) <= low <= entry)
        if not high_ok:
            if raw_high is not None:
                logger.warning(
                    "[%s] Snapshot high_watermark=%r is implausible for %s entry=%.8f; "
                    "resetting to entry", symbol, raw_high, side, entry)
            high = entry
        if not low_ok:
            if raw_low is not None:
                logger.warning(
                    "[%s] Snapshot low_watermark=%r is implausible for %s entry=%.8f; "
                    "resetting to entry", symbol, raw_low, side, entry)
            low = entry
        if entry > 0:
            high = max(high or entry, entry)
            low = min(low or entry, entry)
        pos['high_watermark'] = high
        pos['low_watermark'] = low

        armed = _bool_value(snapshot_position.get('breakeven_armed'))
        raw_trailing = snapshot_position.get('trailing_stop')
        cost = _number_or_none(
            getattr(getattr(self.risk_manager, "rc", None), "round_trip_cost_pct", 0.0)) or 0.0
        trailing = validate_trailing_stop(
            raw_trailing, entry, side, high, low, band_pct=band, cost_pct=cost)
        if trailing is None and raw_trailing not in (None, '', 0, 0.0):
            logger.warning("[%s] Snapshot trailing_stop=%r is implausible for %s "
                           "entry=%.8f; ignoring it", symbol, raw_trailing, side, entry)
        if trailing is not None:
            # A restored trailing level must never weaken the original stop.
            tighter = _most_protective_stop(side, trailing, original)
            if tighter is not None and tighter != trailing:
                logger.warning(
                    "[%s] Restored trailing stop %.8f is less protective than the "
                    "original %.8f; using %.8f", symbol, trailing, original, tighter)
                trailing = tighter
        if armed and trailing is None:
            trailing = self._breakeven_stop(entry, side, original)
            if trailing is not None:
                logger.warning(
                    "[%s] Breakeven was armed but the trailing stop was missing or "
                    "corrupt; reconstructed protective stop at %.8f", symbol, trailing)
            else:
                logger.warning(
                    "[%s] Breakeven was armed but entry is invalid; the protective "
                    "stop could not be reconstructed", symbol)
        pos['trailing_stop'] = trailing
        pos['breakeven_armed'] = bool(armed)

        raw_mark = snapshot_position.get('mark_price')
        mark = _positive_float(raw_mark)
        if raw_mark is not None and mark is None:
            logger.warning("[%s] Snapshot mark_price=%r is malformed; ignoring the stale value",
                           symbol, raw_mark)
        pos['last_mark_price'] = mark
        # A live position must not expose a snapshot mark as a live price. The
        # active mark stays unavailable until a trustworthy live mark arrives.
        pos['mark_price'] = None if self.config.mode == "live" else mark
        logger.info(
            "[%s] Restored dynamic risk state: trailing=%s breakeven=%s high=%.8f "
            "low=%.8f last_mark=%s", symbol, trailing, armed, high, low, mark)

    def _sync_exchange_position(self, symbol, pos):
        """Make the paper simulator agree with the reconstructed risk position."""
        if self.config.mode != "paper" or not isinstance(self.exchange, PaperExchange):
            return
        positions = getattr(self.exchange, 'positions', None)
        if positions is None:
            return
        existing = positions.get(symbol)
        if not isinstance(existing, dict):
            existing = {}
        side = _position_side(pos)
        if side is None:
            logger.error("[%s] Not syncing paper exchange with invalid side %r",
                         symbol, pos.get('side'))
            return
        original = (_positive_float(pos.get('original_stop'))
                    or _positive_float(pos.get('stop_price'))
                    or _positive_float(pos.get('sl')))
        entry = _positive_float(pos.get('entry_price')) or 0.0
        quantity = _positive_float(pos.get('quantity')) or 0.0
        leverage = _positive_float(pos.get('leverage')) or 1.0
        if leverage > 3.0:
            logger.warning("[%s] Clamping restored paper leverage %.4f to the 3x safe cap",
                           symbol, leverage)
            leverage = 3.0
        notional = _positive_float(pos.get('notional'))
        if notional is None:
            notional = entry * quantity
        margin = _positive_float(pos.get('margin'))
        if (margin is None or (notional > 0 and margin > notional * (1.0 + 1e-9))
                or _positive_float(pos.get('leverage')) != leverage):
            margin = notional / leverage if notional else 0.0
            logger.warning("[%s] Recomputed invalid restored margin as %.8f", symbol, margin)

        # Log reconciliation disagreements instead of silently overwriting.
        comparisons = {
            'side': side, 'contracts': quantity, 'entryPrice': entry,
            'leverage': leverage, 'margin': margin, 'sl': original,
            'trailing_stop': pos.get('trailing_stop'),
            'markPrice': _positive_float(pos.get('mark_price')),
        }
        for key, value in comparisons.items():
            prior = existing.get(key)
            if prior is None or value is None:
                continue
            try:
                differs = abs(float(prior) - float(value)) > max(1e-12, abs(float(value)) * 1e-9)
            except (TypeError, ValueError):
                differs = prior != value
            if differs:
                logger.warning("[%s] Paper exchange %s=%r differs from restored risk "
                               "state %r; overwriting", symbol, key, prior, value)

        existing.update({
            'symbol': symbol,
            'side': side,
            'contracts': quantity,
            'entryPrice': entry,
            'notional': notional,
            'strategy': pos.get('strategy', ''),
            'leverage': leverage,
            'margin': margin,
            'tp': _positive_float(pos.get('tp')),
            'sl': original,
            'original_stop': original,
            'trailing_stop': pos.get('trailing_stop'),
            'breakeven_armed': bool(pos.get('breakeven_armed')),
            'high_watermark': pos.get('high_watermark'),
            'low_watermark': pos.get('low_watermark'),
        })
        entry_time = pos.get('entry_time')
        if entry_time is not None:
            existing['entry_time'] = entry_time
            existing['entryTime'] = entry_time
            existing['timestamp'] = (entry_time.isoformat()
                                     if isinstance(entry_time, datetime) else entry_time)
        mark = _positive_float(pos.get('mark_price'))
        if mark is not None:
            existing['markPrice'] = mark
            existing['mark_price'] = mark
        positions[symbol] = existing
        logger.info(
            "[%s] Reconciled paper exchange with risk state (side=%s qty=%.12f "
            "entry=%.8f original_stop=%s trailing=%s)", symbol, side, quantity,
            entry, original, pos.get('trailing_stop'))

    def _fetch_fresh_mark(self, symbol, pos, timeframe):
        """Fetch a trustworthy public mark with bounded retries."""
        reference = _positive_float(pos.get('entry_price'))
        last_error = None
        for attempt in range(3):
            try:
                df = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)
                price = _candle_price(
                    df, symbol, now=self._clock_utc(), timeframe=timeframe,
                    reference=reference)
                if price is not None:
                    return price, None
                last_error = ValueError("no trustworthy public candle")
            except Exception as exc:  # network/parsing failures must not escape
                last_error = exc
            logger.warning("[%s] Mark refresh attempt %d/3 failed: %s",
                           symbol, attempt + 1, last_error)
        return None, last_error

    def _refresh_restored_marks(self):
        """Replace restored snapshot marks with fresh public prices, never stale."""
        if self.config.mode != "paper" or not isinstance(self.exchange, PaperExchange):
            return
        unavailable = set()
        self.mark_unavailable_symbols = getattr(self, 'mark_unavailable_symbols', set())
        jobs = []
        for symbol, pos in list(self.risk_manager.open_positions.items()):
            targets = getattr(self, 'position_targets', {}).get(symbol, {})
            if not isinstance(targets, dict):
                targets = {}
            timeframe = targets.get('timeframe')
            if not isinstance(timeframe, str) or not timeframe_minutes(timeframe):
                timeframe = self.config.scalping.timeframe
            jobs.append((symbol, pos, timeframe))

        # Fetch independently, but apply to shared risk state sequentially.
        fetched = {}
        if jobs:
            workers = min(4, len(jobs))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._fetch_fresh_mark, symbol, pos, timeframe): symbol
                    for symbol, pos, timeframe in jobs
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        fetched[symbol] = future.result()
                    except Exception as exc:
                        fetched[symbol] = (None, exc)

        for symbol, pos, _timeframe in jobs:
            price, error = fetched.get(symbol, (None, ValueError("mark not fetched")))
            if price is None:
                # Do not let a stale snapshot price stand in for a fresh mark.
                pos['mark_price'] = None
                self._sync_exchange_position(symbol, pos)
                unavailable.add(symbol)
                logger.error(
                    "[%s] Could not obtain a trustworthy mark (%s); entries stay "
                    "blocked until risk can be evaluated safely", symbol, error)
                continue
            self.risk_manager.update_position_price(symbol, price)
            update_mark = getattr(self.exchange, 'update_mark_price', None)
            if callable(update_mark):
                update_mark(symbol, price)
            pos['mark_price'] = price
            self._sync_exchange_position(symbol, pos)
            logger.info("[%s] Refreshed restored mark from public candles at %.8f",
                        symbol, price)

        self.mark_unavailable_symbols = unavailable
        if unavailable:
            self.entries_blocked = True
            self.entries_blocked_by_marks = True
        elif getattr(self, 'entries_blocked_by_marks', False):
            # Only the mark failure is cleared here; a live reconciliation
            # mismatch keeps its own block.
            self.entries_blocked = False
            self.entries_blocked_by_marks = False

    def _restore_snapshot_account_state(self, snapshot):
        """Restore account-level snapshot fields independently.

        A malformed scalar must not abort balance, PnL, trade-count, drawdown,
        circuit-breaker or journal restoration for the rest of the session.
        """
        rm = self.risk_manager
        data = snapshot

        targets = data.get('position_targets')
        if isinstance(targets, dict):
            self.position_targets = targets
        elif targets is not None:
            logger.warning("State snapshot position_targets is malformed (%s); ignoring it",
                           type(targets).__name__)
        entry_cycle = data.get('position_entry_cycle')
        if isinstance(entry_cycle, dict):
            self.position_entry_cycle = entry_cycle
        elif entry_cycle is not None:
            logger.warning("State snapshot position_entry_cycle is malformed (%s); ignoring it",
                           type(entry_cycle).__name__)

        if self.config.mode != "paper":
            return

        def restore_number(key, apply, *, positive=False, allow_zero=True, label=None):
            if key not in data:
                return
            value = _finite_float(data.get(key))
            valid = value is not None and (value > 0 if positive else
                                           (value >= 0 if allow_zero else True))
            if not valid:
                logger.warning("Ignoring malformed snapshot %s=%r; keeping %s",
                               key, data.get(key), label or key)
                return
            apply(value)

        def restore_int(key, apply, label=None):
            if key not in data:
                return
            value = _non_negative_int(data.get(key), None)
            if value is None:
                logger.warning("Ignoring malformed snapshot %s=%r; keeping %s",
                               key, data.get(key), label or key)
                return
            apply(value)

        if 'cycle_count' in data:
            restore_int('cycle_count', lambda v: setattr(self, 'cycle_count', v))
        restore_number('account_balance', lambda v: setattr(rm, 'current_balance', v))
        restore_number('peak_balance',
                       lambda v: setattr(rm, 'peak_balance', max(rm.peak_balance, v)),
                       positive=True)
        restore_number('peak_equity',
                       lambda v: setattr(rm, 'peak_equity', max(rm.peak_equity, v)),
                       positive=True)
        if 'max_drawdown_pct' in data:
            dd = _finite_float(data.get('max_drawdown_pct'))
            if dd is None or dd < 0 or dd > 1:
                logger.warning("Ignoring malformed snapshot max_drawdown_pct=%r",
                               data.get('max_drawdown_pct'))
            else:
                rm.max_drawdown_pct = max(rm.max_drawdown_pct, dd)
        restore_int('consecutive_losses',
                    lambda v: setattr(rm, 'consecutive_losses', v))
        restore_number('daily_pnl', lambda v: setattr(rm, 'daily_pnl', v),
                       allow_zero=False)
        restore_int('daily_trades', lambda v: setattr(rm, 'daily_trades', v))
        restore_number('initial_portfolio_equity',
                       lambda v: setattr(self, 'initial_portfolio_equity', v),
                       positive=True)
        restore_number('portfolio_profit_target_pct',
                       lambda v: setattr(self.config, 'portfolio_profit_target_pct', v))

        if 'portfolio_profit_target_enabled' in data:
            enabled_flag = _bool_or_none(data.get('portfolio_profit_target_enabled'))
            if enabled_flag is None:
                logger.warning(
                    "Ignoring malformed snapshot portfolio_profit_target_enabled=%r; "
                    "keeping %s", data.get('portfolio_profit_target_enabled'),
                    self.config.portfolio_profit_target_enabled)
            else:
                self.config.portfolio_profit_target_enabled = enabled_flag
        if 'portfolio_target_status' in data:
            status = data.get('portfolio_target_status')
            if isinstance(status, str) and status:
                self.portfolio_target_status = status
            else:
                logger.warning("Ignoring malformed snapshot portfolio_target_status=%r", status)
        if 'trading_halted' in data:
            halted_flag = _bool_or_none(data.get('trading_halted'))
            if halted_flag is None:
                # A corrupt halt flag must fail closed.
                logger.warning("Snapshot trading_halted=%r is unusable; halting trading",
                               data.get('trading_halted'))
                self.trading_halted = True
            else:
                self.trading_halted = halted_flag

        breaker_data = data.get('circuit_breaker')
        breaker = getattr(rm, 'circuit_breaker', None)
        if breaker_data is not None and not isinstance(breaker_data, dict):
            logger.warning("Ignoring malformed snapshot circuit_breaker=%r", breaker_data)
        elif isinstance(breaker_data, dict) and breaker is not None:
            if 'halted' in breaker_data:
                halted_flag = _bool_or_none(breaker_data.get('halted'))
                if halted_flag is None:
                    logger.warning(
                        "Snapshot circuit_breaker.halted=%r is unusable; halting trading",
                        breaker_data.get('halted'))
                    breaker.halted = True
                else:
                    breaker.halted = halted_flag
            if 'halt_reason' in breaker_data:
                reason = breaker_data.get('halt_reason')
                breaker.halt_reason = reason if isinstance(reason, str) or reason is None else str(reason)
            if 'error_count' in breaker_data:
                count = _non_negative_int(breaker_data.get('error_count'), None)
                if count is None:
                    logger.warning("Ignoring malformed snapshot circuit_breaker.error_count=%r",
                                   breaker_data.get('error_count'))
                else:
                    breaker.error_count = count
            if breaker_data.get('window_start'):
                try:
                    breaker.window_start = datetime.fromisoformat(str(breaker_data['window_start']))
                except (TypeError, ValueError):
                    logger.warning("Ignoring malformed snapshot circuit_breaker.window_start=%r",
                                   breaker_data.get('window_start'))
            if 'window_low' in breaker_data:
                low = _positive_float(breaker_data.get('window_low'))
                if low is None:
                    logger.warning("Ignoring malformed snapshot circuit_breaker.window_low=%r",
                                   breaker_data.get('window_low'))
                else:
                    breaker.window_low = low
            if 'peak_balance' in breaker_data:
                peak = _positive_float(breaker_data.get('peak_balance'))
                if peak is None:
                    logger.warning("Ignoring malformed snapshot circuit_breaker.peak_balance=%r",
                                   breaker_data.get('peak_balance'))
                else:
                    breaker.peak_balance = peak

        restored_meta = getattr(self, "paper_trade_meta", None)
        if not isinstance(restored_meta, dict):
            restored_meta = {}
        for sym, sym_targets in self.position_targets.items():
            try:
                if not isinstance(sym, str) or not isinstance(sym_targets, dict):
                    continue
                journal = sym_targets.get("journal_entry")
                if not isinstance(journal, dict) or not journal:
                    continue
                if sym not in rm.open_positions:
                    continue
                pos = rm.open_positions[sym]
                fill = (_positive_float(journal.get("fill_price"))
                        or _positive_float(pos.get("entry_price")) or 0.0)
                qty = (_positive_float(journal.get("quantity"))
                       or _positive_float(pos.get("quantity")) or 0.0)
                fee = _finite_float(journal.get("fee"))
                fee = fee if fee is not None and fee >= 0 else 0.0
                slippage = _finite_float(journal.get("slippage"))
                slippage = slippage if slippage is not None and slippage >= 0 else 0.0
                leverage = _positive_float(journal.get("selected_leverage")) or 1.0
                entry_time = journal.get("entry_time")
                try:
                    entry_time = (datetime.fromisoformat(entry_time)
                                  if isinstance(entry_time, str) and entry_time else None)
                except (TypeError, ValueError):
                    entry_time = None
                strategy = journal.get("strategy")
                if not isinstance(strategy, str):
                    strategy = (sym_targets.get("strategy")
                                if isinstance(sym_targets.get("strategy"), str) else "")
                signal = journal.get("signal")
                signal = signal if isinstance(signal, str) else ""
                restored_meta[sym] = {
                    "entry_price": fill,
                    "quantity": qty,
                    "entry_fee": fee,
                    "entry_slippage": slippage,
                    "entry_time": entry_time,
                    "strategy": strategy,
                    "signal": signal,
                    "selected_leverage": leverage,
                    "adaptive_plan": None,
                }
            except Exception as exc:
                logger.warning("[%s] Could not restore paper journal metadata: %s", sym, exc)
        self.paper_trade_meta = restored_meta

    def load_state(self):
        import core.state as cstate
        # Mark availability is recomputed on every restoration attempt; do not
        # let a previous attempt's block survive a successful refresh.
        if getattr(self, 'entries_blocked_by_marks', False):
            self.entries_blocked = False
        self.entries_blocked_by_marks = False
        self.mark_unavailable_symbols = set()
        try:
            raw_loaded = cstate.load_open_positions()
        except Exception as e:
            logger.error(f"load_state failed: {e}")
            raw_loaded = []
        path = self._state_path()
        snapshot = None
        if os.path.exists(path):
            try:
                with open(path) as f:
                    snapshot = json.load(f)
                if not isinstance(snapshot, dict):
                    logger.warning(
                        "State snapshot at %s is not a JSON object; ignoring dynamic state", path)
                    snapshot = None
            except Exception as e:
                logger.warning(
                    "State snapshot at %s is unreadable (%s); keeping SQLite original "
                    "stops and skipping dynamic trailing state", path, e)
                snapshot = None
        else:
            logger.warning(
                "No state snapshot at %s; keeping SQLite original stops and starting "
                "dynamic trailing state fresh", path)
        snapshot_positions = {}
        if snapshot is not None:
            raw_positions = snapshot.get('open_positions')
            if isinstance(raw_positions, dict):
                snapshot_positions = raw_positions
            elif raw_positions is not None:
                logger.warning(
                    "State snapshot open_positions is malformed (%s); ignoring it",
                    type(raw_positions).__name__)
        snapshot_targets = snapshot.get('position_targets', {}) if snapshot is not None else {}
        if not isinstance(snapshot_targets, dict):
            logger.warning("State snapshot position_targets is malformed; ignoring it")
            snapshot_targets = {}

        # Validate and de-duplicate open rows. The latest row per symbol wins so
        # repeated saves cannot produce nondeterministic restoration.
        valid_rows = []
        malformed_rows = 0
        seen = {}
        for pp in raw_loaded if isinstance(raw_loaded, list) else []:
            if not isinstance(pp, dict):
                malformed_rows += 1
                logger.error("Ignoring non-object persisted position row: %r", pp)
                continue
            sym = pp.get('symbol')
            if not sym:
                malformed_rows += 1
                logger.error("Ignoring persisted position row without a symbol: %r", pp)
                continue
            amount = _positive_float(pp.get('size') if pp.get('size') is not None
                                     else pp.get('contracts'))
            entry = _positive_float(pp.get('entry_price') if pp.get('entry_price') is not None
                                    else pp.get('entryPrice'))
            if amount is None or entry is None:
                malformed_rows += 1
                logger.error(
                    "[%s] Ignoring malformed open position row (size=%r entry=%r)",
                    sym, pp.get('size'), pp.get('entry_price'))
                continue
            if sym in seen:
                logger.warning("[%s] Duplicate open position rows; keeping the latest row", sym)
            seen[sym] = (pp, amount, entry)
        valid_rows = list(seen.values())
        self.loaded_positions = [pp for pp, _, _ in valid_rows]

        if malformed_rows and not valid_rows:
            # Nothing restorable exists; preserve the historical paper behaviour
            # (no live reconciliation needed) but surface the corruption.
            self.entries_blocked = self.config.mode == "live"
            logger.warning(
                "Restart reconcile: %d malformed open row(s) and no valid position",
                malformed_rows)
            return

        try:
            ex_positions = {}
            try:
                ex_positions = self.exchange.get_positions() or {}
            except Exception:
                for pp, _amount, _entry in valid_rows:
                    sym = pp.get('symbol')
                    if sym:
                        try:
                            ep = self.exchange.get_position(sym)
                            if ep:
                                ex_positions[sym] = ep
                        except Exception:
                            pass
            loaded_syms = set(sym for sym, _, _ in
                              ((pp.get('symbol'), a, e) for pp, a, e in valid_rows))
            ex_syms = set(ex_positions.keys()) if isinstance(ex_positions, dict) else set()
            side_unknown = False
            if loaded_syms != ex_syms:
                if self.config.mode == "paper" and isinstance(self.exchange, PaperExchange):
                    # The in-memory simulator is empty after process restart;
                    # the selected paper database is authoritative for paper positions.
                    for pp, amount, entry in valid_rows:
                        symbol = pp.get('symbol')
                        side = _position_side(pp)
                        if side is None:
                            side_unknown = True
                            logger.error(
                                "[%s] Persisted side=%r is invalid; the paper exchange "
                                "cannot be reconciled safely", symbol, pp.get('side'))
                            continue
                        leverage = _positive_float(pp.get('leverage')) or 1.0
                        if leverage > 3.0:
                            logger.warning(
                                "[%s] Clamping persisted paper leverage %.4f to the 3x safe cap",
                                symbol, leverage)
                            leverage = 3.0
                        notional = amount * entry
                        margin = _positive_float(pp.get('margin'))
                        if (margin is None or margin > notional * (1.0 + 1e-9)
                                or _positive_float(pp.get('leverage')) != leverage):
                            margin = notional / leverage if leverage else notional
                        self.exchange.positions[symbol] = {
                            'symbol': symbol,
                            'side': side,
                            'contracts': amount,
                            'entryPrice': entry,
                            'markPrice': entry,
                            'notional': notional,
                            'leverage': leverage,
                            'margin': margin,
                            'timestamp': pp.get('entry_time'),
                        }
                    ex_syms = set(self.exchange.positions)
                if loaded_syms != ex_syms:
                    logger.warning(f"Restart reconcile mismatch: loaded={loaded_syms} exchange={ex_syms}")
                    if self.config.mode == "live":
                        self.entries_blocked = True
                        return
                    # Paper mode: keep restoring the rows that are valid, but do
                    # not allow new entries while a real position is unreconciled.
                    self.entries_blocked = True
            if side_unknown:
                self.entries_blocked = True

            rm = self.risk_manager
            for pp, amount, entry in valid_rows:
                sym = pp.get('symbol')
                pos = dict(pp)
                side = _position_side(pos)
                if side is None:
                    logger.error(
                        "[%s] Invalid persisted side=%r; restoring as unpriced risk and "
                        "blocking new entries", sym, pos.get('side'))
                    self.entries_blocked = True
                if pos.get('entry_time'):
                    try:
                        pos['entry_time'] = datetime.fromisoformat(str(pos['entry_time']))
                    except (TypeError, ValueError):
                        logger.warning("[%s] Ignoring malformed entry_time=%r",
                                       sym, pos.get('entry_time'))
                        pos['entry_time'] = None
                notional = amount * entry
                leverage = _positive_float(pos.get('leverage')) or 1.0
                if leverage > 3.0:
                    logger.warning(
                        "[%s] Clamping restored leverage %.4f to the 3x safe cap",
                        sym, leverage)
                    leverage = 3.0
                margin = _finite_float(pos.get('margin'))
                if (margin is None or margin < 0
                        or (notional > 0 and margin > notional * (1.0 + 1e-9))
                        or _positive_float(pos.get('leverage')) != leverage):
                    margin = notional / leverage if leverage else 0.0
                    logger.warning("[%s] Recomputed invalid persisted margin as %.8f",
                                   sym, margin)
                pos['quantity'] = amount
                pos['entry_price'] = entry
                pos['notional'] = notional
                pos['leverage'] = leverage
                pos['margin'] = margin
                pos['side'] = side if side is not None else pos.get('side')
                pos['strategy'] = (pos.get('strategy')
                                   if isinstance(pos.get('strategy'), str) else "")
                # SQLite owns identity/side/quantity/entry/TP/original SL,
                # strategy, leverage and margin. The JSON snapshot only
                # restores the dynamic risk-management fields.
                pos['original_stop'] = self._restore_original_stop(
                    pp, snapshot_positions.get(sym), snapshot_targets.get(sym))
                if pos['original_stop'] is None:
                    # Missing or corrupt protective levels are unpriced risk, not
                    # zero risk. Block new entries until a human reconciles it.
                    logger.error("[%s] No usable protective stop after restore; "
                                 "blocking new entries", sym)
                    self.entries_blocked = True
                pos['stop_price'] = pos['original_stop']
                pos['sl'] = pos['original_stop']
                tp = _plausible_take_profit(pos.get('tp'), entry, side)
                if tp is None:
                    tp = _plausible_take_profit(
                        (snapshot_targets.get(sym) or {}).get('tp')
                        if isinstance(snapshot_targets.get(sym), dict) else None,
                        entry, side)
                if pos.get('tp') is not None and tp is None:
                    logger.warning("[%s] Ignoring implausible persisted tp=%r for %s "
                                   "entry=%.8f", sym, pos.get('tp'), side, entry)
                pos['tp'] = tp
                pos.setdefault('mark_price', None)
                rm.symbol_exposure[sym] = notional
                self._merge_restored_dynamic_state(sym, pos, snapshot_positions.get(sym))
                rm.open_positions[sym] = pos
                self._sync_exchange_position(sym, pos)
            rm._refresh_equity_drawdown()
            if snapshot is not None:
                self._restore_snapshot_account_state(snapshot)
            self._refresh_restored_marks()
            rm._refresh_equity_drawdown()
            if (os.environ.get('RECONCILE_CONFIRM') == '1'
                    and not getattr(self, 'mark_unavailable_symbols', set())):
                self.entries_blocked = False
                self.entries_blocked_by_marks = False
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
                if self.max_cycles is not None and self.cycle_count >= self.max_cycles:
                    self.running = False
                    logger.info(
                        "Cycle limit already reached; preserving %d open position(s) for resume.",
                        len(getattr(self.risk_manager, "open_positions", {})),
                    )
                    self.print_status()
                    self.persist_state()
                    break
                self.cycle_count += 1
                for sym in self.config.scalping.symbols:
                    self.check_exits(sym)

                if self.config.mode == "paper" and self.check_portfolio_profit_target():
                    self.running = False
                    self.print_status()
                    self.persist_state()
                    logger.info("Paper session stopped after portfolio target liquidation.")
                    break

                if not getattr(self, "trading_halted", False):
                    for sym in self.config.scalping.symbols:
                        self.process_symbol(sym)

                if self.cycle_count % 10 == 0:
                    self.print_status()

                if self.max_cycles is not None and self.cycle_count >= self.max_cycles:
                    self.running = False
                    logger.info(
                        "Maximum cycle count reached; stopping gracefully with %d open position(s) persisted.",
                        len(getattr(self.risk_manager, "open_positions", {})),
                    )
                    self.print_status()
                    self.persist_state()
                    break

                if self.running:
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
            "paper_session_id": getattr(self.config, "paper_session_id", None),
            "state_database": getattr(self.config, "state_db_path", "data/paper.db"),
            "portfolio_profit_target_enabled": self.config.portfolio_profit_target_enabled,
            "portfolio_profit_target_pct": self.config.portfolio_profit_target_pct,
            "max_cycles": getattr(self, "max_cycles", None),
            "initial_portfolio_equity": getattr(
                self, "initial_portfolio_equity", self.config.scalping.initial_capital),
        }
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        logger.info(
            "Run costs: maker_fee=%.6f taker_fee=%.6f slippage=%.6f",
            metadata["maker_fee"], metadata["taker_fee"], metadata["slippage"],
        )
        if self.config.mode == "paper" and getattr(self.config, "paper_session_id", None):
            logger.info("Paper session id=%s database=%s",
                        self.config.paper_session_id, self.config.state_db_path)


def build_config_from_args(argv=None, environ=None):
    parser = argparse.ArgumentParser(description="Unified Crypto Scalper")
    parser.add_argument("mode_arg", nargs="?", choices=("paper", "live"), default="paper")
    parser.add_argument("--mode", dest="mode_option", choices=("paper", "live"))
    parser.add_argument("--demo", action="store_true", help="Run in paper mode")
    parser.add_argument("--parity", action="store_true", help="Use deterministic paper/backtest parity settings")
    parser.add_argument("--strategy", help="Select the strategy by registered name")
    parser.add_argument("--strategies", nargs="+", help="Select strategy names or 'all'")
    parser.add_argument("--timeframe", help="Select one candle interval (legacy single-timeframe option)")
    parser.add_argument("--timeframes", nargs="+", help="Select candle intervals or 'all'")
    parser.add_argument("--min-timeframe-confirmations", type=int, default=0,
                        help="Minimum higher-timeframe trend confirmations before entry (default: 0)")
    parser.add_argument("--orderbook-limit", type=int, choices=(20, 100), default=20,
                        help="Supported KuCoin public order-book depth (20 or 100; default: 20)")
    parser.add_argument("--cost-safety-buffer-pct", type=float, default=0.02,
                        help="Additional reward buffer in percentage points (default: 0.02)")
    parser.add_argument("--min-expected-net-reward-pct", type=float, default=0.02,
                        help="Minimum net target reward in percentage points (default: 0.02)")
    parser.add_argument("--symbols", nargs="+", help="Fixed symbols separated by spaces or commas")
    parser.add_argument("--fresh", action="store_true", help="Start paper mode without restoring persisted positions")
    parser.add_argument("--paper-session-id",
                        help="Create or resume an isolated paper session by this stable identifier")
    parser.add_argument("--portfolio-profit-target-pct", type=float, default=70.0,
                        help="Net portfolio return objective from original starting equity (default: 70)")
    parser.add_argument("--disable-portfolio-profit-target", action="store_true",
                        help="Disable the portfolio-level take-profit objective")
    parser.add_argument("--show-config", action="store_true",
                        help="Print effective settings without starting the bot")
    parser.add_argument("--scan-all-coins", action="store_true",
                        help="Discover symbols even when --symbols is provided")
    parser.add_argument("--max-position-notional-pct", type=float,
                        help="Maximum position notional as a fraction of current equity")
    parser.add_argument("--max-open-positions", type=int,
                        help="Maximum simultaneous open positions")
    parser.add_argument("--max-cycles", type=int,
                        help="Stop gracefully after this many trading cycles")
    parser.add_argument("--adaptive-leverage", action="store_true",
                        help="Enable statistically gated leverage in paper mode only")
    parser.add_argument("--min-leverage", type=float, default=1.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--max-margin-allocation-pct", type=float, default=0.50)
    parser.add_argument("--max-gross-exposure-pct", type=float)
    parser.add_argument("--min-oos-trades", type=int, default=100)
    parser.add_argument("--leverage-edge-small", type=float, default=0.02)
    parser.add_argument("--leverage-edge-moderate", type=float, default=0.05)
    parser.add_argument("--leverage-edge-strong", type=float, default=0.08)
    parser.add_argument("--leverage-edge-exceptional", type=float, default=0.12)
    parser.add_argument("--leverage-reduce-drawdown-pct", type=float, default=0.02)
    parser.add_argument("--leverage-min-only-drawdown-pct", type=float, default=0.04)
    parser.add_argument("--hold-minutes", type=float, default=30.0)
    parser.add_argument("--maker-fee", type=float, default=0.0002,
                        help="Maker fee as a fraction of notional (default: 0.0002)")
    parser.add_argument("--taker-fee", type=float, default=0.0005,
                        help="Taker fee as a fraction of notional (default: 0.0005)")
    parser.add_argument("--slippage", type=float, default=0.0003,
                        help="Slippage as a fraction of notional (default: 0.0003)")
    parser.add_argument("--long-only", action="store_true", help="Skip SHORT signals")
    parser.add_argument("--capital", type=float,
                        help="Starting virtual capital (paper mode)")
    parser.add_argument("--exchange", help="Pin candle data to this CCXT exchange")
    args = parser.parse_args(argv)

    requested_mode = args.mode_option or args.mode_arg
    if args.demo:
        requested_mode = "paper"
    if args.fresh and requested_mode != "paper":
        raise SystemExit("--fresh is available only in paper mode.")
    if args.paper_session_id and requested_mode != "paper":
        raise SystemExit("--paper-session-id is available only in paper mode.")
    if args.parity and requested_mode != "paper":
        raise SystemExit("--parity is available only in paper mode.")
    elif requested_mode == "live":
        live_arg = args.mode_arg == "live" or args.mode_option == "live"
        env = os.environ if environ is None else environ
        if not live_arg or env.get("CONFIRM_LIVE") != "YES":
            raise SystemExit("Live mode requires the explicit 'live' argument and CONFIRM_LIVE=YES.")
    if args.adaptive_leverage and requested_mode != "paper":
        raise SystemExit("--adaptive-leverage is available only in paper mode.")
    if (args.strategies or args.timeframes) and requested_mode != "paper":
        raise SystemExit("Multi-strategy/multi-timeframe execution is paper-only.")

    cfg = BotConfig()
    cfg.mode = requested_mode
    cfg.fresh = args.fresh
    if args.paper_session_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.paper_session_id):
        raise SystemExit("--paper-session-id must contain 1-64 letters, digits, underscores, or hyphens.")
    cfg.paper_session_id = args.paper_session_id
    cfg.portfolio_profit_target_pct = args.portfolio_profit_target_pct
    if (not math.isfinite(cfg.portfolio_profit_target_pct) or
            cfg.portfolio_profit_target_pct <= 0):
        raise SystemExit("--portfolio-profit-target-pct must be finite and greater than zero.")
    cfg.portfolio_profit_target_enabled = not args.disable_portfolio_profit_target
    cfg.show_config = args.show_config
    if args.exchange:
        cfg.exchange.name = args.exchange.lower()
    exchange_cls = getattr(ccxt, cfg.exchange.name, None)
    if exchange_cls:
        try:
            exchange_instance = exchange_cls({"enableRateLimit": True})
        except TypeError:
            exchange_instance = exchange_cls()
        exchange_timeframes = list(getattr(exchange_instance, "timeframes", {}) or {})
    else:
        exchange_timeframes = []
    cfg.exchange_timeframes = exchange_timeframes or [cfg.scalping.timeframe]
    selected_strategies, selected_timeframes, warnings = resolve_selection(
        args.strategy, args.strategies, args.timeframe, args.timeframes,
        exchange_timeframes or [cfg.scalping.timeframe],
    )
    if args.strategies and args.strategies != ["all"]:
        unknown = sorted(set(args.strategies) - set(STRATEGIES))
        if unknown:
            raise SystemExit("Unknown strategy name(s): " + ", ".join(unknown))
    cfg.strategy_names = selected_strategies if (args.strategies or args.timeframes) else None
    cfg.timeframes = selected_timeframes
    if args.timeframe and not args.timeframes:
        cfg.scalping.timeframe = args.timeframe
    cfg.timeframe_warnings = warnings
    if args.min_timeframe_confirmations < 0 or args.min_timeframe_confirmations > 13:
        raise SystemExit("--min-timeframe-confirmations must be between 0 and 13.")
    cfg.min_timeframe_confirmations = args.min_timeframe_confirmations
    if (not math.isfinite(args.cost_safety_buffer_pct) or args.cost_safety_buffer_pct < 0 or
            not math.isfinite(args.min_expected_net_reward_pct) or args.min_expected_net_reward_pct < 0):
        raise SystemExit("Cost buffer and minimum expected net reward must be finite and non-negative.")
    cfg.execution.cost_safety_buffer_pct = args.cost_safety_buffer_pct / 100.0
    cfg.execution.minimum_expected_net_reward_pct = args.min_expected_net_reward_pct / 100.0
    cfg.strategy.orderbook_depth = args.orderbook_limit
    cfg.max_cycles = args.max_cycles
    for name in ("maker_fee", "taker_fee", "slippage"):
        value = getattr(args, name)
        if value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")
        setattr(cfg.backtest, name, value)
    cfg.long_only = args.long_only
    if args.strategy:
        if args.strategy.startswith("candidate:") and requested_mode != "paper":
            raise SystemExit("Candidate strategies are available only in paper mode.")
        cfg.strategy.active_strategy = args.strategy
    if args.symbols:
        symbols = [symbol.strip() for value in args.symbols
                   for symbol in value.split(",") if symbol.strip()]
        if not symbols:
            raise SystemExit("--symbols must include at least one symbol.")
        cfg.scalping.symbols = symbols
        if not args.scan_all_coins:
            cfg.scalping.scan_all_coins = False
    elif args.scan_all_coins:
        cfg.scalping.scan_all_coins = True

    if args.capital is not None:
        if not math.isfinite(args.capital) or args.capital <= 0:
            raise SystemExit("--capital must be finite and greater than zero.")
        if requested_mode == "paper":
            cfg.scalping.initial_capital = args.capital
    if args.max_open_positions is not None:
        if args.max_open_positions < 1:
            raise SystemExit("--max-open-positions must be greater than zero.")
        cfg.risk.max_open_positions = args.max_open_positions
        cfg.scalping.max_positions = args.max_open_positions
    else:
        cfg.scalping.max_positions = cfg.risk.max_open_positions
    if args.max_cycles is not None and args.max_cycles < 1:
        raise SystemExit("--max-cycles must be greater than zero.")

    adaptive = cfg.adaptive_leverage
    adaptive.enabled = bool(args.adaptive_leverage)
    adaptive.min_leverage = args.min_leverage
    adaptive.max_leverage = args.max_leverage
    adaptive.max_margin_allocation_pct = args.max_margin_allocation_pct
    if (not math.isfinite(adaptive.min_leverage) or adaptive.min_leverage < 1.0 or
            not math.isfinite(adaptive.max_leverage) or adaptive.max_leverage < adaptive.min_leverage or
            adaptive.max_leverage > 3.0):
        raise SystemExit("--min-leverage and --max-leverage must be finite, with 1 <= min <= max <= 3.")
    if (not math.isfinite(adaptive.max_margin_allocation_pct) or
            not 0 < adaptive.max_margin_allocation_pct <= 0.50):
        raise SystemExit("--max-margin-allocation-pct must be finite and in (0, 0.50].")
    adaptive.max_gross_exposure_pct = (
        args.max_gross_exposure_pct if args.max_gross_exposure_pct is not None else
        adaptive.max_margin_allocation_pct * adaptive.max_leverage
    )
    if (not math.isfinite(adaptive.max_gross_exposure_pct) or
            not 0 < adaptive.max_gross_exposure_pct <= 1.50):
        raise SystemExit("--max-gross-exposure-pct must be finite and in (0, 1.50].")
    # The configured KuCoin adapter is spot. Spot candles alone cannot justify
    # leveraged futures sizing, so the effective paper leverage remains 1x.
    cfg.market_type = "futures" if cfg.exchange.name.lower() in {
        "kucoinfutures", "binanceusdm", "binancecoinm", "okx"
    } else "spot"
    cfg.market_leverage_evidence = False
    if adaptive.enabled and not cfg.market_leverage_evidence:
        adaptive.requested_max_leverage = adaptive.max_leverage
        adaptive.max_leverage = 1.0
    if args.min_oos_trades < 1:
        raise SystemExit("--min-oos-trades must be greater than zero.")
    adaptive.min_oos_trades = args.min_oos_trades
    edge_values = (args.leverage_edge_small, args.leverage_edge_moderate,
                   args.leverage_edge_strong, args.leverage_edge_exceptional)
    if (any(not math.isfinite(value) or not 0 < value < 1 for value in edge_values) or
            tuple(sorted(edge_values)) != edge_values):
        raise SystemExit("Leverage edge thresholds must be finite, in (0, 1), and strictly increasing.")
    (adaptive.edge_small, adaptive.edge_moderate, adaptive.edge_strong,
     adaptive.edge_exceptional) = edge_values
    adaptive.reduce_drawdown_pct = args.leverage_reduce_drawdown_pct
    adaptive.min_only_drawdown_pct = args.leverage_min_only_drawdown_pct
    if (not math.isfinite(adaptive.reduce_drawdown_pct) or
            not math.isfinite(adaptive.min_only_drawdown_pct) or
            not 0 < adaptive.reduce_drawdown_pct < adaptive.min_only_drawdown_pct < 1):
        raise SystemExit("Leverage drawdown thresholds must be finite, positive, and increasing.")

    if args.max_position_notional_pct is not None:
        pct = args.max_position_notional_pct
        upper = adaptive.max_gross_exposure_pct if adaptive.enabled else 1.0
        if not math.isfinite(pct) or not 0 < pct <= upper:
            raise SystemExit(
                "--max-position-notional-pct must be finite and in (0, 1] unless "
                "adaptive leverage is enabled, then it cannot exceed max gross exposure."
            )
        cfg.risk.max_position_size_pct = pct
        cfg.risk.max_position_per_symbol_pct = pct
        cfg.risk.max_portfolio_heat_pct = pct

    if adaptive.enabled:
        if args.max_open_positions is not None and args.max_open_positions > 1:
            raise SystemExit("Adaptive/targeted paper sessions are limited to one simultaneous open position.")
        cfg.risk.max_open_positions = 1
        cfg.scalping.max_positions = 1
        # These are independent from the per-position notional cap above.
        cfg.risk.max_position_per_symbol_pct = adaptive.max_gross_exposure_pct
        cfg.risk.max_portfolio_heat_pct = adaptive.max_gross_exposure_pct
    elif requested_mode == "paper" and cfg.portfolio_profit_target_enabled:
        if args.max_open_positions is not None and args.max_open_positions > 1:
            raise SystemExit("Targeted paper sessions are limited to one simultaneous open position.")
        cfg.risk.max_open_positions = 1
        cfg.scalping.max_positions = 1

    if requested_mode == "paper":
        # Global paper leverage stays at 1x; selected leverage is attached to
        # individual paper orders only when the opt-in adaptive planner allows it.
        cfg.scalping.leverage = 1
        if args.fresh:
            session_id = args.paper_session_id or uuid.uuid4().hex
            session_dir = os.path.join("data", "paper_sessions")
            cfg.state_db_path = os.path.join(session_dir, f"paper_{session_id}.db")
            cfg.state_file_path = os.path.join(session_dir, f"paper_{session_id}.state.json")
            if args.paper_session_id and (os.path.exists(cfg.state_db_path) or
                                          os.path.exists(cfg.state_file_path)):
                raise SystemExit("That paper session already exists; omit --fresh to resume it.")
            cfg.paper_session_id = session_id
        elif args.paper_session_id:
            session_dir = os.path.join("data", "paper_sessions")
            cfg.state_db_path = os.path.join(session_dir, f"paper_{args.paper_session_id}.db")
            cfg.state_file_path = os.path.join(session_dir, f"paper_{args.paper_session_id}.state.json")
            if not os.path.exists(cfg.state_db_path) and not os.path.exists(cfg.state_file_path):
                raise SystemExit("Paper session state not found; use --fresh to create it.")
    if args.parity:
        if (args.capital is not None and (not math.isfinite(args.capital) or args.capital <= 0)) or args.hold_minutes <= 0:
            raise SystemExit("--capital and --hold-minutes must be greater than zero in parity mode.")
        cfg.parity = True
        cfg.risk.trailing_stop_enabled = False
        cfg.risk.breakeven_activation_pct = 0.0
        cfg.risk.trailing_stop_activation_pct = 0.0
        cfg.scalping.scan_all_coins = bool(args.scan_all_coins)
        cfg.scalping.symbols = list(cfg.scalping.symbols)
        cfg.risk.max_hold_time_minutes = args.hold_minutes
        cfg.execution.paper_partial_fill_chance = 0.0
        cfg.execution.paper_reject_chance = 0.0
        cfg.scalping.initial_capital = args.capital if args.capital is not None else 100.0
    if cfg.mode == "live" and (not cfg.exchange.api_key or not cfg.exchange.api_secret):
        raise SystemExit("API key or secret missing for live mode")
    return cfg


def effective_config_summary(cfg):
    pct = cfg.risk.max_position_size_pct
    strategies = getattr(cfg, "strategy_names", None) or [cfg.strategy.active_strategy]
    timeframes = getattr(cfg, "timeframes", None) or [cfg.scalping.timeframe]
    matrix = {}
    for name in strategies:
        info = STRATEGIES.get(name)
        matrix[name] = {
            tf: ({
                "compatible": is_timeframe_compatible(name, tf),
                "minimum_completed_bars": info.min_bars if info else None,
                "directions": info.direction if info else "unknown",
                "required_market_data": list(info.market_data) if info else [],
                "context": (info.context if is_timeframe_compatible(name, tf) else
                            "outside this strategy's intended holding horizon"),
            } if info else {"compatible": False, "context": "unknown strategy"})
            for tf in timeframes
        }
    local_data = []
    data_dir = os.path.join("data_" + cfg.exchange.name, "*")
    for path in __import__("glob").glob(data_dir):
        base = os.path.basename(path)
        if any(base.startswith(symbol.replace("/", "_")) for symbol in cfg.scalping.symbols):
            local_data.append(base)
    return {
        "MODE": cfg.mode,
        "EXCHANGE": cfg.exchange.name,
        "CAPITAL": float(cfg.scalping.initial_capital),
        "SYMBOLS": list(cfg.scalping.symbols),
        "SELECTED STRATEGIES": strategies,
        "SELECTED TIMEFRAMES": timeframes,
        "MIN HIGHER TIMEFRAME CONFIRMATIONS": int(getattr(cfg, "min_timeframe_confirmations", 0)),
        "TIMEFRAME STRATEGY COMPATIBILITY": matrix,
        "EXCHANGE MARKET TYPE": getattr(cfg, "market_type", "spot"),
        "FUTURES LEVERAGE EVIDENCE": bool(getattr(cfg, "market_leverage_evidence", False)),
        "EXCHANGE AVAILABLE TIMEFRAMES": sorted(getattr(cfg, "exchange_timeframes", []) or [cfg.scalping.timeframe]),
        "LOCAL DATA FILES": local_data,
        "DATA COVERAGE NOTE": "Preview does not download candles; live coverage/warm-up is checked safely at runtime.",
        "ORDERBOOK LIMIT": int(cfg.strategy.orderbook_depth),
        "COST SAFETY BUFFER PCT": float(cfg.execution.cost_safety_buffer_pct) * 100.0,
        "MIN EXPECTED NET REWARD PCT": float(cfg.execution.minimum_expected_net_reward_pct) * 100.0,
        "TIMEFRAME WARNINGS": list(getattr(cfg, "timeframe_warnings", [])),
        "AUTO SCAN": bool(cfg.scalping.scan_all_coins),
        "MAX OPEN POSITIONS": int(cfg.risk.max_open_positions),
        "STOP RISK BUDGET": float(cfg.scalping.initial_capital * cfg.risk.risk_per_trade),
        "STOP RISK BUDGET PCT": float(cfg.risk.risk_per_trade) * 100.0,
        "MAX POSITION NOTIONAL": float(cfg.scalping.initial_capital * pct),
        "MAX POSITION NOTIONAL PCT": float(pct),
        "LEVERAGE": float(cfg.scalping.leverage),
        "ADAPTIVE LEVERAGE": bool(cfg.adaptive_leverage.enabled),
        "MIN LEVERAGE": float(cfg.adaptive_leverage.min_leverage),
        "MAX LEVERAGE": float(cfg.adaptive_leverage.max_leverage),
        "REQUESTED MAX LEVERAGE": float(getattr(cfg.adaptive_leverage, "requested_max_leverage", None) or cfg.adaptive_leverage.max_leverage),
        "MAX MARGIN ALLOCATION": float(cfg.scalping.initial_capital * cfg.adaptive_leverage.max_margin_allocation_pct),
        "MAX MARGIN ALLOCATION PCT": float(cfg.adaptive_leverage.max_margin_allocation_pct),
        "MAX GROSS EXPOSURE": float(cfg.scalping.initial_capital * cfg.adaptive_leverage.max_gross_exposure_pct),
        "MAX GROSS EXPOSURE PCT": float(cfg.adaptive_leverage.max_gross_exposure_pct),
        "MIN OOS TRADES": int(cfg.adaptive_leverage.min_oos_trades),
        "REAL ORDER EXECUTION": "DISABLED" if cfg.mode == "paper" else "ENABLED",
        "MAX CYCLES": getattr(cfg, "max_cycles", None),
        "STATE DATABASE": getattr(cfg, "state_db_path", "data/paper.db"),
        "PAPER SESSION ID": getattr(cfg, "paper_session_id", None),
        "PORTFOLIO PROFIT TARGET ENABLED": bool(cfg.portfolio_profit_target_enabled),
        "PORTFOLIO PROFIT TARGET PCT": float(cfg.portfolio_profit_target_pct)
            if cfg.portfolio_profit_target_enabled else None,
        "TARGET PORTFOLIO EQUITY": float(cfg.scalping.initial_capital * (1.0 + cfg.portfolio_profit_target_pct / 100.0))
            if cfg.portfolio_profit_target_enabled else None,
    }


if __name__ == "__main__":
    config = build_config_from_args()
    if config.show_config:
        print(json.dumps(effective_config_summary(config), indent=2))
    else:
        bot = UnifiedCryptoBot(config)
        bot.run()



