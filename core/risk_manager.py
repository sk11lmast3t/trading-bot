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
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


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
    def __init__(self, config):
        self.config = config
        self.rc = config.risk
        self.sc = config.scalping
        self.daily_pnl = 0.0
        self.peak_balance = config.scalping.initial_capital
        self.current_balance = config.scalping.initial_capital
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_time = None
        self.pause_until = None
        self.trade_log: List[Dict] = []
        self._current_day = datetime.utcnow().date()
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
        now = now or datetime.now()
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

    def register_position(self, symbol: str, notional_value: float, side: str):
        self.symbol_exposure[symbol] = self.symbol_exposure.get(symbol, 0.0) + notional_value
        self.open_positions[symbol] = {
            'notional': notional_value,
            'side': side,
            'entry_time': datetime.now(),
            'entry_price': None,
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
        pos['high_watermark'] = max(pos.get('high_watermark') or price, price)
        pos['low_watermark'] = min(pos.get('low_watermark') or price, price)

    def check_trailing_stop(self, symbol: str, price: float) -> Tuple[bool, str]:
        """Return (should_exit, reason) if trailing stop or breakeven is hit."""
        pos = self.open_positions.get(symbol)
        if not pos or not self.rc.trailing_stop_enabled:
            return False, ""

        entry = pos['entry_price'] or price
        side = pos['side']
        high = pos['high_watermark'] or price

        # Breakeven: move stop to entry after activation pct
        if not pos['breakeven_armed'] and self.rc.breakeven_activation_pct > 0:
            if side == 'long' and price >= entry * (1 + self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                pos['trailing_stop'] = entry
                logger.info(f"[{symbol}] Breakeven armed at {entry:.4f}")
            elif side == 'short' and price <= entry * (1 - self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                pos['trailing_stop'] = entry
                logger.info(f"[{symbol}] Breakeven armed at {entry:.4f}")

        # Trailing stop
        if pos['trailing_stop'] is not None:
            if side == 'long':
                new_trail = high * (1 - self.rc.trailing_stop_pct)
                if new_trail > pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if price <= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
            else:
                new_trail = high * (1 + self.rc.trailing_stop_pct)
                if new_trail < pos['trailing_stop'] or pos['trailing_stop'] == entry:
                    pos['trailing_stop'] = new_trail
                if price >= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
        return False, ""

    def release_position(self, symbol: str):
        if symbol in self.symbol_exposure:
            del self.symbol_exposure[symbol]
        self.open_positions.pop(symbol, None)

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
        size_value = min((capital * risk_pct) / price_risk * entry, max_size_value)
        return size_value / entry

    def record_trade(self, pnl: float, symbol: str = "", trade_time: Optional[datetime] = None):
        t_time = trade_time or datetime.now()
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
        self.trade_log.append({
            'timestamp': t_time.isoformat() if hasattr(t_time, 'isoformat') else str(t_time),
            'symbol': symbol,
            'pnl': pnl,
            'balance': self.current_balance
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
        dd = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        halted, halt_reason = self.circuit_breaker.is_halted()
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'total_pnl': total_pnl,
            'max_drawdown': dd,
            'balance': self.current_balance,
            'peak': self.peak_balance,
            'portfolio_heat': sum(self.symbol_exposure.values()),
            'circuit_breaker_halted': halted,
            'circuit_breaker_reason': halt_reason,
        }
