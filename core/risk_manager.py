"""
Unified Risk Management Engine
Includes Passivbot-style anti-drawdown logic and BNF strict no-averaging-down discipline.
Enhancements:
- Portfolio heat & per-symbol position limits
- Volatility-adjusted (ATR-normalized) position sizing
- Circuit breakers (flash-crash drawdown, error streaks)
- Trailing stop & breakeven activation
"""
import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def valid_stop_price(value) -> Optional[float]:
    """Return a finite positive stop price, or None when unusable."""
    try:
        stop = float(value)
    except (TypeError, ValueError):
        return None
    return stop if math.isfinite(stop) and stop > 0 else None


def normalize_side(side: str) -> str:
    s = str(side).strip().lower()
    if s in ('buy', 'long'):
        return 'long'
    if s in ('sell', 'short'):
        return 'short'
    raise ValueError(f"Unknown side: {side!r}")

class CircuitBreaker:
    """Tracks rapid drawdown and error streaks to halt trading automatically."""

    def __init__(self, risk_config, initial_capital: float):
        self.enabled = risk_config.circuit_breaker_enabled
        self.drawdown_pct = risk_config.circuit_breaker_drawdown_pct
        self.time_window_min = risk_config.circuit_breaker_time_window_min
        self.error_streak_limit = risk_config.circuit_breaker_error_streak
        self.peak_balance = initial_capital
        self.window_start: Optional[datetime] = None
        self.window_low: Optional[float] = None
        self.error_count = 0
        self.halted = False
        self.halt_reason: Optional[str] = None

    def check_drawdown(self, current_balance: float, now: datetime) -> Tuple[bool, str]:
        if not self.enabled:
            return False, ""
        if self.window_start is None or now - self.window_start > timedelta(minutes=self.time_window_min):
            self.window_start = now
            self.window_low = current_balance
            self.peak_balance = max(self.peak_balance, current_balance)

        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        if current_balance < (self.window_low or current_balance):
            self.window_low = current_balance

        dd = (self.peak_balance - current_balance) / self.peak_balance
        if dd >= self.drawdown_pct:
            self.halted = True
            self.halt_reason = f"Circuit breaker: {dd*100:.1f}% drawdown in {self.time_window_min}min"
            return True, self.halt_reason
        return False, ""

    def record_error(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, ""
        self.error_count += 1
        if self.error_count >= self.error_streak_limit:
            self.halted = True
            self.halt_reason = f"Circuit breaker: {self.error_count} consecutive exchange errors"
            return True, self.halt_reason
        return False, ""

    def reset_errors(self):
        self.error_count = 0

    def is_halted(self) -> Tuple[bool, Optional[str]]:
        return self.halted, self.halt_reason


class RiskManager:
    def __init__(self, config, clock=None):
        self.config = config
        self.rc = config.risk
        self.sc = config.scalping
        self.clock = clock or getattr(config, "clock", None) or datetime.now
        self.daily_pnl = 0.0
        self.peak_balance = config.scalping.initial_capital
        self.peak_equity = config.scalping.initial_capital
        self.max_drawdown_pct = 0.0
        self.current_balance = config.scalping.initial_capital
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_time = None
        self.pause_until = None
        self.trade_log: List[Dict] = []
        self._current_day = (self.clock().date() if clock is not None or getattr(config, "clock", None)
                             else datetime.utcnow().date())
        # Portfolio heat: per-symbol notional exposure
        self.symbol_exposure: Dict[str, float] = {}
        self.open_positions: Dict[str, Dict] = {}
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(config.risk, config.scalping.initial_capital)

    def _check_daily_reset(self, now = None):
        if now is None:
            now = datetime.utcnow()
        current_day = now.date() if hasattr(now, 'date') else now
        if self._current_day is None:
            self._current_day = current_day
        elif current_day != self._current_day:
            self.reset_daily()
            self._current_day = current_day

    def can_trade(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        now = now or self.clock()
        self._check_daily_reset(now)

        # Circuit breaker check
        halted, reason = self.circuit_breaker.is_halted()
        if halted:
            return False, reason

        if self.pause_until and now < self.pause_until:
            wait_s = int((self.pause_until - now).total_seconds())
            return False, f"Paused after loss streak ({wait_s}s remaining)"
        if self.rc.daily_trade_limit > 0 and self.daily_trades >= self.rc.daily_trade_limit:
            return False, f"Daily trade limit reached ({self.daily_trades}/{self.rc.daily_trade_limit})"
        if self.daily_pnl <= -(self.current_balance * self.rc.max_daily_loss_pct):
            return False, f"Max daily loss reached ({self.daily_pnl:.2f})"
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown >= self.rc.max_drawdown_pct:
            return False, f"Max drawdown hit ({drawdown*100:.1f}%)"
        if self.consecutive_losses >= self.rc.max_consecutive_losses:
            self.pause_until = now + timedelta(seconds=self.rc.pause_after_loss_streak)
            self.consecutive_losses = 0
            return False, "Loss streak pause triggered"
        if self.last_trade_time:
            cooldown = timedelta(seconds=self.sc.cooldown_seconds)
            if now - self.last_trade_time < cooldown:
                return False, "Cooldown between trades"
        return True, "OK"

    def check_portfolio_heat(self, symbol: str, notional_value: float) -> Tuple[bool, str]:
        """Reject trade if per-symbol or portfolio heat would exceed limits."""
        current_symbol_exposure = self.symbol_exposure.get(symbol, 0.0)
        new_symbol_exposure = current_symbol_exposure + notional_value
        max_symbol_notional = self.current_balance * self.rc.max_position_per_symbol_pct
        if new_symbol_exposure > max_symbol_notional:
            return False, f"Per-symbol heat limit for {symbol} ({new_symbol_exposure:.2f} > {max_symbol_notional:.2f})"

        total_heat = sum(self.symbol_exposure.values()) + notional_value
        max_portfolio_notional = self.current_balance * self.rc.max_portfolio_heat_pct
        if total_heat > max_portfolio_notional:
            return False, f"Portfolio heat limit ({total_heat:.2f} > {max_portfolio_notional:.2f})"
        return True, "OK"

    def register_position(self, symbol: str, notional_value: float, side: str,
                          leverage: float = 1.0, margin: float = 0.0,
                          quantity: float = 0.0, strategy: str = "",
                          stop_price: Optional[float] = None):
        side = normalize_side(side)
        self.symbol_exposure[symbol] = self.symbol_exposure.get(symbol, 0.0) + notional_value
        self.open_positions[symbol] = {
            'notional': notional_value,
            'margin': margin or notional_value / max(float(leverage), 1.0),
            'leverage': float(leverage),
            'quantity': float(quantity),
            'strategy': strategy,
            # The original protective stop is kept separate from the active
            # trailing stop so recovery never loses the exchange-level stop.
            'stop_price': stop_price,
            'original_stop': stop_price,
            'side': side,
            'entry_time': self.clock(),
            'entry_price': None,
            'mark_price': None,
            'last_mark_price': None,
            'high_watermark': None,
            'low_watermark': None,
            'trailing_stop': None,
            'breakeven_armed': False,
        }

    def update_position_price(self, symbol: str, price: float):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        pos['entry_price'] = pos.get('entry_price') or price
        pos['mark_price'] = float(price)
        pos['high_watermark'] = max(pos.get('high_watermark') or price, price)
        pos['low_watermark'] = min(pos.get('low_watermark') or price, price)
        self._refresh_equity_drawdown()

    def _refresh_equity_drawdown(self):
        if getattr(self.config, "mode", "paper") != "paper":
            return
        unrealized = 0.0
        for position in self.open_positions.values():
            entry = float(position.get('entry_price') or 0.0)
            mark = float(position.get('mark_price') or entry)
            quantity = float(position.get('quantity') or 0.0)
            side = position.get('side', 'long')
            if entry > 0 and mark > 0 and quantity > 0:
                unrealized += (mark - entry) * quantity * (1.0 if side == 'long' else -1.0)
        equity = self.current_balance + unrealized
        self.peak_equity = max(getattr(self, 'peak_equity', self.peak_balance), equity)
        drawdown = ((self.peak_equity - equity) / self.peak_equity
                    if self.peak_equity > 0 else 0.0)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown)

    def check_trailing_stop(self, symbol: str, price: float) -> Tuple[bool, str]:
        """Return (should_exit, reason) if trailing stop or breakeven is hit."""
        pos = self.open_positions.get(symbol)
        if not pos or not self.rc.trailing_stop_enabled:
            return False, ""

        entry = pos['entry_price'] or price
        side = normalize_side(pos.get('side', ''))
        high = pos['high_watermark'] or price
        low = pos['low_watermark'] or price

        # Breakeven: arm at breakeven_activation_pct, offset stop by round_trip_cost_pct
        if not pos.get('breakeven_armed', False) and self.rc.breakeven_activation_pct > 0:
            if side == 'long' and price >= entry * (1 + self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                if pos.get('trailing_stop') is None:
                    pos['trailing_stop'] = entry * (1 + self.rc.round_trip_cost_pct)
                logger.info(f"[{symbol}] Breakeven armed at {pos.get('trailing_stop'):.4f}")
            elif side == 'short' and price <= entry * (1 - self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                if pos.get('trailing_stop') is None:
                    pos['trailing_stop'] = entry * (1 - self.rc.round_trip_cost_pct)
                logger.info(f"[{symbol}] Breakeven armed at {pos.get('trailing_stop'):.4f}")

        # Trailing stop activation: arm when price moves trailing_stop_activation_pct in favor
        if pos.get('trailing_stop') is None and self.rc.trailing_stop_activation_pct > 0:
            if side == 'long' and price >= entry * (1 + self.rc.trailing_stop_activation_pct):
                pos['trailing_stop'] = high * (1 - self.rc.trailing_stop_pct)
                logger.info(f"[{symbol}] Trailing armed at {pos['trailing_stop']:.4f}")
            elif side == 'short' and price <= entry * (1 - self.rc.trailing_stop_activation_pct):
                pos['trailing_stop'] = low * (1 + self.rc.trailing_stop_pct)
                logger.info(f"[{symbol}] Trailing armed at {pos['trailing_stop']:.4f}")

        # Trailing stop update & hit check
        if pos.get('trailing_stop') is not None:
            if side == 'long':
                new_trail = high * (1 - self.rc.trailing_stop_pct)
                if new_trail > pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if price <= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
            else:
                new_trail = low * (1 + self.rc.trailing_stop_pct)
                if new_trail < pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if price >= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
        return False, ""


    def release_position(self, symbol: str):
        if symbol in self.symbol_exposure:
            del self.symbol_exposure[symbol]
        self.open_positions.pop(symbol, None)
        self._refresh_equity_drawdown()

    def calculate_position_size(self, capital: float, entry: float, stop_loss: float,
                                atr: Optional[float] = None) -> float:
        risk_pct = self.rc.risk_per_trade
        # Anti-martingale size decay on losses
        if self.rc.reduce_size_on_loss and self.consecutive_losses > 0:
            risk_pct *= self.rc.size_reduction_factor ** self.consecutive_losses

        # Volatility-adjusted sizing: scale risk by ATR if enabled
        if self.rc.vol_adjust_sizing and atr is not None and atr > 0:
            vol_risk = atr * self.rc.atr_multiplier_sizing
            price_risk = max(abs(entry - stop_loss), vol_risk)
        else:
            price_risk = abs(entry - stop_loss)

        if price_risk == 0:
            return 0.0

        max_size_value = capital * self.rc.max_position_size_pct
        costs_per_unit = self.stop_costs_per_unit(entry, stop_loss)
        effective_risk_per_unit = price_risk + costs_per_unit
        size_value = min((capital * risk_pct) / effective_risk_per_unit * entry, max_size_value)
        return size_value / entry

    def stop_costs_per_unit(self, entry: float, stop: float) -> float:
        """Estimated entry/closing costs and funding per unit at the planned stop."""
        costs = getattr(self.config, "backtest", None)
        execution = getattr(self.config, "execution", None)
        if costs is None:
            return 0.0
        maker = getattr(execution, "entry_mode", "market") == "maker"
        entry_fee = costs.maker_fee if maker else costs.taker_fee
        entry_slip = 0.0 if maker else costs.slippage
        exit_fee, exit_slip = costs.taker_fee, costs.slippage
        hold_minutes = float(self.rc.max_hold_time_minutes)
        funding = abs(float(getattr(costs, "funding_rate", 0.0) or 0.0)) * hold_minutes / 480.0
        return (float(entry) * (entry_fee + entry_slip + funding) +
                float(stop) * (exit_fee + exit_slip))

    def effective_stop_price(self, position: dict) -> Optional[float]:
        """Return the protective stop that would actually cap the loss.

        The original stop and the active trailing stop are both honoured: a
        long keeps the higher (more protective) level, a short the lower one.
        A missing or invalid level is never treated as zero risk; if neither
        level is usable this returns None so callers can block instead.
        """
        try:
            side = normalize_side(position.get("side", "long"))
        except (TypeError, ValueError):
            return None
        original = valid_stop_price(position.get("original_stop"))
        if original is None:
            original = valid_stop_price(position.get("stop_price"))
        if original is None:
            original = valid_stop_price(position.get("sl"))
        trailing = valid_stop_price(position.get("trailing_stop"))
        if original is None:
            return trailing
        if trailing is None:
            return original
        return max(original, trailing) if side == "long" else min(original, trailing)

    def stop_risk_for_position(self, position: dict) -> Optional[float]:
        try:
            entry = float(position.get("entry_price") or position.get("entryPrice") or 0.0)
            quantity = float(position.get("quantity") or position.get("contracts") or position.get("size") or 0.0)
            side = normalize_side(position.get("side", "long"))
        except (TypeError, ValueError):
            return None
        stop = self.effective_stop_price(position)
        if stop is None or not math.isfinite(entry) or entry <= 0 or quantity <= 0:
            return None
        distance = max(0.0, entry - stop) if side == "long" else max(0.0, stop - entry)
        return distance * quantity + quantity * self.stop_costs_per_unit(entry, stop)

    def current_equity(self) -> float:
        """Realized balance plus open-position unrealized PnL (paper mark)."""
        unrealized = 0.0
        for position in self.open_positions.values():
            entry = float(position.get('entry_price') or 0.0)
            mark = float(position.get('mark_price') or entry)
            quantity = float(position.get('quantity') or 0.0)
            side = position.get('side', 'long')
            if entry > 0 and mark > 0 and quantity > 0:
                unrealized += (mark - entry) * quantity * (1.0 if side == 'long' else -1.0)
        return self.current_balance + unrealized

    def check_stop_loss_risk(self, entry: float, stop: float, quantity: float,
                             side: str = "long") -> Tuple[bool, str, float]:
        active_risk = 0.0
        for position in self.open_positions.values():
            risk = self.stop_risk_for_position(position)
            if risk is None:
                return False, "cannot price existing position stop risk", float("inf")
            active_risk += risk
        candidate = self.stop_risk_for_position({
            "entry_price": entry, "stop_price": stop, "quantity": quantity, "side": side
        })
        total = active_risk + (candidate or 0.0)
        budget = self.current_equity() * min(float(self.rc.risk_per_trade), 0.01)
        if total > budget + max(1e-9, budget * 1e-9):
            return False, f"aggregate stop risk ${total:.6f} exceeds 1% budget ${budget:.6f}", total
        return True, "OK", total

    def record_trade(self, pnl: float, symbol: str = "", trade_time: Optional[datetime] = None,
                     *, strategy: str = "", net_return: Optional[float] = None,
                     selected_leverage: float = 1.0, expected_value: Optional[float] = None):
        t_time = trade_time or self.clock()
        self.daily_pnl += pnl
        self.current_balance += pnl
        self.daily_trades += 1
        self.last_trade_time = t_time
        if pnl >= 0:
            self.consecutive_losses = 0
            self.circuit_breaker.reset_errors()
        else:
            self.consecutive_losses += 1
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance
                    if self.peak_balance > 0 else 0.0)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown)
        if not self.open_positions:
            self.peak_equity = max(self.peak_equity, self.current_balance)
            equity_drawdown = ((self.peak_equity - self.current_balance) / self.peak_equity
                               if self.peak_equity > 0 else 0.0)
            self.max_drawdown_pct = max(self.max_drawdown_pct, equity_drawdown)
        self.trade_log.append({
            'timestamp': t_time.isoformat() if hasattr(t_time, 'isoformat') else str(t_time),
            'symbol': symbol,
            'pnl': pnl,
            'balance': self.current_balance,
            'strategy': strategy,
            'net_return': net_return,
            'selected_leverage': float(selected_leverage),
            'expected_value': expected_value,
        })
        tag = "WIN" if pnl >= 0 else "LOSS"
        logger.info(f"[{tag}] PnL: ${pnl:.4f} | New Bal: ${self.current_balance:.2f} | Streak: {self.consecutive_losses}")

    def record_error(self):
        halted, reason = self.circuit_breaker.record_error()
        if halted:
            logger.error(f"CIRCUIT BREAKER HALT: {reason}")

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def get_stats(self) -> Dict:
        total = len(self.trade_log)
        wins = sum(1 for t in self.trade_log if t['pnl'] > 0)
        losses = total - wins
        total_pnl = sum(t['pnl'] for t in self.trade_log)
        wr = (wins / total * 100) if total > 0 else 0
        current_unrealized = sum(
            (float(p.get('mark_price') or p.get('entry_price') or 0.0) -
             float(p.get('entry_price') or 0.0)) * float(p.get('quantity') or 0.0) *
            (1.0 if p.get('side', 'long') == 'long' else -1.0)
            for p in self.open_positions.values()
            if float(p.get('entry_price') or 0.0) > 0 and float(p.get('quantity') or 0.0) > 0
        )
        current_equity = self.current_balance + current_unrealized
        if getattr(self.config, "mode", "paper") == "paper":
            current_dd = ((self.peak_equity - current_equity) / self.peak_equity * 100
                          if self.peak_equity > 0 else 0.0)
            dd = max(self.max_drawdown_pct * 100.0, current_dd)
        else:
            dd = ((self.peak_balance - self.current_balance) / self.peak_balance * 100
                  if self.peak_balance > 0 else 0.0)
        halted, halt_reason = self.circuit_breaker.is_halted()
        unrealized = used_margin = gross_exposure = portfolio_heat = 0.0
        unpriced_stop_risk_positions = 0
        open_leverages = []
        for position in self.open_positions.values():
            entry = float(position.get('entry_price') or 0.0)
            mark = float(position.get('mark_price') or entry)
            quantity = float(position.get('quantity') or 0.0)
            side = position.get('side', 'long')
            if quantity > 0 and entry > 0 and mark > 0:
                unrealized += (mark - entry) * quantity * (1.0 if side == 'long' else -1.0)
                gross_exposure += mark * quantity
            else:
                gross_exposure += float(position.get('notional') or 0.0)
            used_margin += float(position.get('margin') or 0.0)
            stop_risk = self.stop_risk_for_position(position)
            if stop_risk is None:
                unpriced_stop_risk_positions += 1
            else:
                portfolio_heat += stop_risk
            open_leverages.append(float(position.get('leverage') or 1.0))
        equity = self.current_balance + unrealized
        adaptive = getattr(self.config, 'adaptive_leverage', None)
        # The aggregate margin-allocation cap applies whether or not adaptive
        # leverage is enabled; disabling adaptive mode must not silently raise
        # the allocation ceiling to 100% of equity.
        margin_pct = (float(getattr(adaptive, 'max_margin_allocation_pct', 0.50))
                      if adaptive is not None else 0.50)
        margin_pct = min(max(margin_pct, 0.0), 1.0)
        available_account_equity = max(0.0, equity - used_margin)
        remaining_margin_capacity = max(0.0, equity * margin_pct - used_margin)
        available_margin = min(available_account_equity, remaining_margin_capacity)
        leverage_values = [float(t.get('selected_leverage') or 1.0) for t in self.trade_log]
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'total_pnl': total_pnl,
            'max_drawdown': dd,
            'max_drawdown_pct': dd,  # percentage points, matching the UI label
            'max_drawdown_fraction': dd / 100.0,
            'balance': self.current_balance,
            'portfolio_equity': equity,
            'realized_pnl': total_pnl,
            'unrealized_pnl': unrealized,
            'current_leverage': max(open_leverages, default=1.0),
            'average_leverage': (sum(leverage_values) / len(leverage_values)
                                 if leverage_values else 1.0),
            'maximum_leverage_used': max(leverage_values + open_leverages, default=1.0),
            'total_margin_used': used_margin,
            'available_margin': available_margin,
            'available_account_equity': available_account_equity,
            'remaining_margin_capacity': remaining_margin_capacity,
            'margin_allocation_cap': equity * margin_pct,
            'total_gross_exposure': gross_exposure,
            'gross_exposure_pct': gross_exposure / equity * 100.0 if equity > 0 else 0.0,
            'peak': self.peak_balance,
            'portfolio_heat': portfolio_heat,
            'portfolio_heat_pct': portfolio_heat / equity * 100.0 if equity > 0 else 0.0,
            'unpriced_stop_risk_positions': unpriced_stop_risk_positions,
            'stop_risk_budget': self.current_equity() * min(float(self.rc.risk_per_trade), 0.01),
            'stop_risk_budget_pct': min(float(self.rc.risk_per_trade), 0.01) * 100.0,
            'circuit_breaker_halted': halted,
            'circuit_breaker_reason': halt_reason,
        }
