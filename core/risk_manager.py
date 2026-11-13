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

# A recovered stop may sit slightly on the profit side of entry (breakeven
# locks in round-trip costs), so allow a small margin beyond entry while still
# rejecting nonsense such as 1e-12 or 1e9.
RECOVERY_STOP_MARGIN = 0.001
DEFAULT_RECOVERY_BAND_PCT = 0.25
DEFAULT_MAX_STOP_DISTANCE_PCT = 0.25


def finite_float(value) -> Optional[float]:
    """Return a finite float, or None. Booleans are not prices."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_stop_price(value) -> Optional[float]:
    """Return a finite positive stop price, or None when unusable."""
    stop = finite_float(value)
    return stop if stop is not None and stop > 0 else None


def finite_non_negative(value, default: float = 0.0) -> float:
    """Clamp a cost/rate input to a finite non-negative number."""
    number = finite_float(value)
    return number if number is not None and number >= 0 else float(default)


def plausible_protective_stop(stop, entry, side, max_distance_pct=None) -> Optional[float]:
    """Validate an *original* stop against entry and side.

    A long stop must be below entry; a short stop must be above entry. Stops
    farther than ``max_distance_pct`` from entry are treated as corrupt rather
    than silently trusted.
    """
    stop_v = valid_stop_price(stop)
    entry_v = valid_stop_price(entry)
    if stop_v is None or entry_v is None:
        return None
    try:
        side_v = normalize_side(side)
    except (TypeError, ValueError):
        return None
    limit = finite_float(max_distance_pct)
    if limit is None or limit <= 0:
        limit = DEFAULT_MAX_STOP_DISTANCE_PCT
    if side_v == "long":
        if stop_v >= entry_v or stop_v < entry_v * (1.0 - limit):
            return None
    else:
        if stop_v <= entry_v or stop_v > entry_v * (1.0 + limit):
            return None
    return stop_v


def validate_trailing_stop(value, entry, side, high=None, low=None,
                           band_pct=None, cost_pct=None) -> Optional[float]:
    """Validate a restored trailing stop against entry, side and watermarks.

    Legitimate trailing stops may be far above entry (a long after a large
    run-up) or far below it (a short), so the watermark excursion extends the
    allowed range. Values outside that range, booleans and wrong-side extremes
    are rejected and callers fall back to the original stop.
    """
    trailing = valid_stop_price(value)
    entry_v = valid_stop_price(entry)
    if trailing is None or entry_v is None:
        return None
    try:
        side_v = normalize_side(side)
    except (TypeError, ValueError):
        return None
    band = finite_float(band_pct)
    if band is None or band <= 0:
        band = DEFAULT_RECOVERY_BAND_PCT
    margin = finite_non_negative(cost_pct) + RECOVERY_STOP_MARGIN
    high_v = valid_stop_price(high)
    low_v = valid_stop_price(low)
    if side_v == "long":
        lower = entry_v * (1.0 - band)
        upper = max(entry_v, high_v or entry_v) * (1.0 + margin)
    else:
        lower = min(entry_v, low_v or entry_v) * (1.0 - margin)
        upper = entry_v * (1.0 + margin)
    return trailing if lower <= trailing <= upper else None


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
        balance = finite_float(current_balance)
        if balance is None or balance <= 0:
            self.halted = True
            self.halt_reason = "Circuit breaker: non-finite or non-positive balance"
            return True, self.halt_reason
        if self.window_start is None or now - self.window_start > timedelta(minutes=self.time_window_min):
            self.window_start = now
            self.window_low = balance
            self.peak_balance = max(self.peak_balance, balance)

        if balance > self.peak_balance:
            self.peak_balance = balance
        if balance < (self.window_low or balance):
            self.window_low = balance

        if self.peak_balance <= 0:
            self.halted = True
            self.halt_reason = "Circuit breaker: peak balance is not positive"
            return True, self.halt_reason
        dd = (self.peak_balance - balance) / self.peak_balance
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

        # A non-finite risk state must never silently pass a limit check.
        for label, value in (("balance", self.current_balance),
                             ("daily_pnl", self.daily_pnl),
                             ("peak_balance", self.peak_balance)):
            if finite_float(value) is None:
                return False, f"Risk state is not finite ({label})"

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
        if self.peak_balance <= 0:
            return False, "Peak balance is not positive; risk state is invalid"
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if not math.isfinite(drawdown):
            return False, "Drawdown is not finite"
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
        notional = finite_float(notional_value)
        balance = finite_float(self.current_balance)
        if notional is None or notional < 0:
            return False, f"Invalid notional for {symbol}: {notional_value!r}"
        if balance is None or balance <= 0:
            return False, "Account balance is not finite and positive"
        current_symbol_exposure = finite_float(self.symbol_exposure.get(symbol, 0.0))
        if current_symbol_exposure is None:
            return False, f"Stored exposure for {symbol} is not finite"
        new_symbol_exposure = current_symbol_exposure + notional
        max_symbol_notional = balance * self.rc.max_position_per_symbol_pct
        if new_symbol_exposure > max_symbol_notional:
            return False, f"Per-symbol heat limit for {symbol} ({new_symbol_exposure:.2f} > {max_symbol_notional:.2f})"

        exposures = []
        for name, value in self.symbol_exposure.items():
            if finite_float(value) is None:
                return False, f"Stored exposure for {name} is not finite"
            exposures.append(value)
        total_heat = sum(exposures) + notional
        max_portfolio_notional = balance * self.rc.max_portfolio_heat_pct
        if total_heat > max_portfolio_notional:
            return False, f"Portfolio heat limit ({total_heat:.2f} > {max_portfolio_notional:.2f})"
        return True, "OK"

    def register_position(self, symbol: str, notional_value: float, side: str,
                          leverage: float = 1.0, margin: float = 0.0,
                          quantity: float = 0.0, strategy: str = "",
                          stop_price: Optional[float] = None):
        side = normalize_side(side)
        notional = finite_float(notional_value)
        leverage_v = finite_float(leverage)
        margin_v = finite_float(margin)
        quantity_v = finite_float(quantity)
        if notional is None or notional < 0:
            raise ValueError(f"Invalid position notional: {notional_value!r}")
        if leverage_v is None or leverage_v <= 0:
            raise ValueError(f"Invalid position leverage: {leverage!r}")
        if margin_v is None or margin_v < 0:
            margin_v = notional / leverage_v
        if quantity_v is None or quantity_v < 0:
            raise ValueError(f"Invalid position quantity: {quantity!r}")
        prior_exposure = finite_float(self.symbol_exposure.get(symbol, 0.0))
        if prior_exposure is None:
            raise ValueError(f"Stored exposure for {symbol} is not finite")
        self.symbol_exposure[symbol] = prior_exposure + notional
        self.open_positions[symbol] = {
            'notional': notional,
            'margin': margin_v or notional / max(leverage_v, 1.0),
            'leverage': leverage_v,
            'quantity': quantity_v,
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
        value = valid_stop_price(price)
        if value is None:
            logger.warning("[%s] Ignoring non-finite or non-positive mark %r", symbol, price)
            return
        pos['entry_price'] = valid_stop_price(pos.get('entry_price')) or value
        pos['mark_price'] = value
        high = valid_stop_price(pos.get('high_watermark')) or value
        low = valid_stop_price(pos.get('low_watermark')) or value
        pos['high_watermark'] = max(high, value)
        pos['low_watermark'] = min(low, value)
        self._refresh_equity_drawdown()

    def _equity_components(self):
        """Return (unrealized, unpriced_count) with malformed positions counted."""
        unrealized = 0.0
        unpriced = 0
        for position in self.open_positions.values():
            try:
                entry = finite_float(position.get('entry_price'))
                mark = finite_float(position.get('mark_price'))
                quantity = finite_float(position.get('quantity'))
                side = position.get('side', 'long')
            except (TypeError, ValueError):
                unpriced += 1
                continue
            if entry is None or mark is None or quantity is None:
                unpriced += 1
                continue
            if quantity == 0:
                continue
            if entry <= 0 or mark <= 0 or quantity < 0:
                unpriced += 1
                continue
            try:
                side_v = normalize_side(side)
            except (TypeError, ValueError):
                unpriced += 1
                continue
            unrealized += (mark - entry) * quantity * (1.0 if side_v == 'long' else -1.0)
        return unrealized, unpriced

    def _refresh_equity_drawdown(self):
        if getattr(self.config, "mode", "paper") != "paper":
            return
        unrealized, unpriced = self._equity_components()
        self.unpriced_equity_positions = unpriced
        previous = getattr(self, '_last_unpriced_equity', None)
        if unpriced and unpriced != previous:
            logger.warning(
                "Equity excludes %d malformed position(s); risk state is incomplete",
                unpriced)
        self._last_unpriced_equity = unpriced
        balance = self.current_balance
        if not math.isfinite(balance):
            logger.error("Cannot refresh equity: balance is not finite")
            return
        equity = balance + unrealized
        if not math.isfinite(equity):
            logger.error("Cannot refresh equity: computed equity is not finite")
            return
        peak = getattr(self, 'peak_equity', self.peak_balance)
        if not math.isfinite(peak) or peak <= 0:
            peak = max(balance, equity)
        self.peak_equity = max(peak, equity)
        drawdown = ((self.peak_equity - equity) / self.peak_equity
                    if self.peak_equity > 0 else 0.0)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown)

    def check_trailing_stop(self, symbol: str, price: float) -> Tuple[bool, str]:
        """Return (should_exit, reason) if trailing stop or breakeven is hit."""
        pos = self.open_positions.get(symbol)
        if not pos or not self.rc.trailing_stop_enabled:
            return False, ""
        price_v = valid_stop_price(price)
        if price_v is None:
            logger.warning("[%s] Ignoring trailing check with invalid price %r", symbol, price)
            return False, ""
        try:
            side = normalize_side(pos.get('side', ''))
        except (TypeError, ValueError):
            logger.error("[%s] Cannot evaluate trailing stop with invalid side %r",
                         symbol, pos.get('side'))
            return False, ""
        entry = valid_stop_price(pos.get('entry_price')) or price_v
        band = getattr(self.rc, "recovery_price_band_pct", DEFAULT_RECOVERY_BAND_PCT)
        cost = getattr(self.rc, "round_trip_cost_pct", 0.0)
        high = valid_stop_price(pos.get('high_watermark'))
        low = valid_stop_price(pos.get('low_watermark'))
        # Reject corrupt watermarks; a legitimate large run is preserved because
        # the band is multiplicative and deliberately generous.
        if high is None or high < entry or high > entry * (1.0 + band):
            high = max(entry, price_v)
        if low is None or low > entry or low < entry / (1.0 + band):
            low = min(entry, price_v)
        high = max(high, price_v)
        low = min(low, price_v)
        pos['high_watermark'] = high
        pos['low_watermark'] = low

        raw_trailing = pos.get('trailing_stop')
        trailing = validate_trailing_stop(raw_trailing, entry, side, high, low,
                                          band_pct=band, cost_pct=cost)
        if raw_trailing is not None and trailing is None:
            logger.warning("[%s] Ignoring implausible trailing stop %r", symbol, raw_trailing)
            pos['trailing_stop'] = None
        elif trailing is not None:
            pos['trailing_stop'] = trailing

        # Breakeven: arm at breakeven_activation_pct, offset stop by round_trip_cost_pct
        if not pos.get('breakeven_armed', False) and self.rc.breakeven_activation_pct > 0:
            if side == 'long' and price_v >= entry * (1 + self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                if pos.get('trailing_stop') is None:
                    pos['trailing_stop'] = entry * (1 + self.rc.round_trip_cost_pct)
                logger.info(f"[{symbol}] Breakeven armed at {pos.get('trailing_stop'):.4f}")
            elif side == 'short' and price_v <= entry * (1 - self.rc.breakeven_activation_pct):
                pos['breakeven_armed'] = True
                if pos.get('trailing_stop') is None:
                    pos['trailing_stop'] = entry * (1 - self.rc.round_trip_cost_pct)
                logger.info(f"[{symbol}] Breakeven armed at {pos.get('trailing_stop'):.4f}")

        # Trailing stop activation: arm when price moves trailing_stop_activation_pct in favor
        if pos.get('trailing_stop') is None and self.rc.trailing_stop_activation_pct > 0:
            if side == 'long' and price_v >= entry * (1 + self.rc.trailing_stop_activation_pct):
                pos['trailing_stop'] = high * (1 - self.rc.trailing_stop_pct)
                logger.info(f"[{symbol}] Trailing armed at {pos['trailing_stop']:.4f}")
            elif side == 'short' and price_v <= entry * (1 - self.rc.trailing_stop_activation_pct):
                pos['trailing_stop'] = low * (1 + self.rc.trailing_stop_pct)
                logger.info(f"[{symbol}] Trailing armed at {pos['trailing_stop']:.4f}")

        # Trailing stop update & hit check
        if pos.get('trailing_stop') is not None:
            if side == 'long':
                new_trail = high * (1 - self.rc.trailing_stop_pct)
                if new_trail > pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if price_v <= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
            else:
                new_trail = low * (1 + self.rc.trailing_stop_pct)
                if new_trail < pos['trailing_stop']:
                    pos['trailing_stop'] = new_trail
                if price_v >= pos['trailing_stop']:
                    return True, f"Trailing stop hit at {pos['trailing_stop']:.4f}"
        return False, ""


    def release_position(self, symbol: str):
        if symbol in self.symbol_exposure:
            del self.symbol_exposure[symbol]
        self.open_positions.pop(symbol, None)
        self._refresh_equity_drawdown()

    def calculate_position_size(self, capital: float, entry: float, stop_loss: float,
                                atr: Optional[float] = None) -> float:
        risk_pct = finite_float(self.rc.risk_per_trade)
        capital = finite_float(capital)
        entry = finite_float(entry)
        stop_loss = finite_float(stop_loss)
        if capital is None or entry is None or stop_loss is None:
            return 0.0
        if risk_pct is None:
            return 0.0
        if capital <= 0 or entry <= 0 or stop_loss <= 0:
            return 0.0
        # Anti-martingale size decay on losses
        if self.rc.reduce_size_on_loss and self.consecutive_losses > 0:
            risk_pct *= self.rc.size_reduction_factor ** self.consecutive_losses
        if not math.isfinite(risk_pct) or risk_pct <= 0:
            return 0.0

        # Volatility-adjusted sizing: scale risk by ATR if enabled
        atr_v = finite_float(atr)
        if self.rc.vol_adjust_sizing and atr_v is not None and atr_v > 0:
            vol_risk = atr_v * self.rc.atr_multiplier_sizing
            price_risk = max(abs(entry - stop_loss), vol_risk)
        else:
            price_risk = abs(entry - stop_loss)

        if not math.isfinite(price_risk) or price_risk <= 0:
            return 0.0

        max_size_value = capital * self.rc.max_position_size_pct
        costs_per_unit = self.stop_costs_per_unit(entry, stop_loss)
        effective_risk_per_unit = price_risk + costs_per_unit
        if not math.isfinite(effective_risk_per_unit) or effective_risk_per_unit <= 0:
            return 0.0
        size_value = min((capital * risk_pct) / effective_risk_per_unit * entry, max_size_value)
        size = size_value / entry
        return size if math.isfinite(size) and size > 0 else 0.0

    def stop_costs_per_unit(self, entry: float, stop: float) -> float:
        """Estimated entry/closing costs and funding per unit at the planned stop."""
        costs = getattr(self.config, "backtest", None)
        execution = getattr(self.config, "execution", None)
        if costs is None:
            return 0.0
        entry_v = finite_float(entry)
        stop_v = finite_float(stop)
        if entry_v is None or stop_v is None or entry_v <= 0 or stop_v <= 0:
            return 0.0
        maker = getattr(execution, "entry_mode", "market") == "maker"
        # Negative fees/slippage/funding must never reduce the risk estimate.
        entry_fee = finite_non_negative(costs.maker_fee if maker else costs.taker_fee)
        entry_slip = 0.0 if maker else finite_non_negative(costs.slippage)
        exit_fee = finite_non_negative(costs.taker_fee)
        exit_slip = finite_non_negative(costs.slippage)
        hold_minutes = finite_non_negative(self.rc.max_hold_time_minutes)
        funding = abs(finite_float(getattr(costs, "funding_rate", 0.0)) or 0.0) * hold_minutes / 480.0
        cost = (entry_v * (entry_fee + entry_slip + funding) +
                stop_v * (exit_fee + exit_slip))
        return cost if math.isfinite(cost) and cost >= 0 else 0.0

    def effective_stop_price(self, position: dict) -> Optional[float]:
        """Return the protective stop that would actually cap the loss.

        The original stop and the active trailing stop are both honoured: a
        long keeps the higher (more protective) level, a short the lower one.
        A missing, malformed or implausible level is never treated as zero
        risk; if neither level is usable this returns None so callers block.
        """
        try:
            side = normalize_side(position.get("side", "long"))
        except (TypeError, ValueError):
            return None
        entry = valid_stop_price(position.get("entry_price")
                                 or position.get("entryPrice"))
        band = getattr(self.rc, "recovery_price_band_pct", DEFAULT_RECOVERY_BAND_PCT)
        cost = getattr(self.rc, "round_trip_cost_pct", 0.0)
        max_distance = getattr(self.rc, "max_stop_distance_pct",
                               DEFAULT_MAX_STOP_DISTANCE_PCT)
        original = None
        if entry is not None:
            for key in ("original_stop", "stop_price", "sl"):
                candidate = plausible_protective_stop(
                    position.get(key), entry, side, max_distance)
                if candidate is not None:
                    original = candidate
                    break
        else:
            for key in ("original_stop", "stop_price", "sl"):
                candidate = valid_stop_price(position.get(key))
                if candidate is not None:
                    original = candidate
                    break
        trailing = validate_trailing_stop(
            position.get("trailing_stop"), entry, side,
            high=position.get("high_watermark"),
            low=position.get("low_watermark"),
            band_pct=band, cost_pct=cost,
        )
        if original is None:
            return trailing
        if trailing is None:
            return original
        return max(original, trailing) if side == "long" else min(original, trailing)

    def stop_risk_for_position(self, position: dict) -> Optional[float]:
        try:
            side = normalize_side(position.get("side", "long"))
        except (TypeError, ValueError):
            return None
        entry = finite_float(position.get("entry_price")
                             or position.get("entryPrice"))
        quantity = finite_float(position.get("quantity")
                                or position.get("contracts")
                                or position.get("size"))
        if entry is None or entry <= 0 or quantity is None or quantity <= 0:
            return None
        stop = self.effective_stop_price(position)
        if stop is None:
            return None
        distance = max(0.0, entry - stop) if side == "long" else max(0.0, stop - entry)
        risk = distance * quantity + quantity * self.stop_costs_per_unit(entry, stop)
        return risk if math.isfinite(risk) else None

    def current_equity(self) -> float:
        """Realized balance plus open-position unrealized PnL (paper mark)."""
        unrealized, unpriced = self._equity_components()
        balance = self.current_balance
        if not math.isfinite(balance):
            return float("nan")
        equity = balance + unrealized
        if unpriced and math.isfinite(equity):
            # The value is conservative (malformed positions contribute zero
            # unrealized) but the count is exposed so callers cannot mistake it
            # for a complete measurement.
            self.unpriced_equity_positions = unpriced
        return equity

    def check_stop_loss_risk(self, entry: float, stop: float, quantity: float,
                             side: str = "long") -> Tuple[bool, str, float]:
        try:
            side = normalize_side(side)
        except (TypeError, ValueError):
            return False, f"unknown candidate position side: {side!r}", float("inf")
        active_risk = 0.0
        for position in self.open_positions.values():
            risk = self.stop_risk_for_position(position)
            if risk is None or not math.isfinite(risk):
                return False, "cannot price existing position stop risk", float("inf")
            active_risk += risk
        candidate = self.stop_risk_for_position({
            "entry_price": entry, "stop_price": stop, "quantity": quantity, "side": side
        })
        if candidate is None or not math.isfinite(candidate):
            return False, "cannot price candidate stop risk", float("inf")
        total = active_risk + candidate
        if not math.isfinite(total):
            return False, "aggregate stop risk is not finite", float("inf")
        risk_pct = finite_float(self.rc.risk_per_trade)
        if risk_pct is None or risk_pct <= 0:
            return False, "stop-risk budget is not finite and positive", float("inf")
        budget = self.current_equity() * min(risk_pct, 0.01)
        if not math.isfinite(budget) or budget <= 0:
            return False, "stop-risk budget is not finite and positive", float("inf")
        if total > budget + max(1e-9, budget * 1e-9):
            return False, f"aggregate stop risk ${total:.6f} exceeds 1% budget ${budget:.6f}", total
        return True, "OK", total

    def record_trade(self, pnl: float, symbol: str = "", trade_time: Optional[datetime] = None,
                     *, strategy: str = "", net_return: Optional[float] = None,
                     selected_leverage: float = 1.0, expected_value: Optional[float] = None):
        pnl_v = finite_float(pnl)
        if pnl_v is None:
            logger.error("[%s] Refusing to record non-finite PnL: %r", symbol, pnl)
            return False
        pnl = pnl_v
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
        leverage_v = finite_float(selected_leverage)
        if leverage_v is None or leverage_v <= 0:
            leverage_v = 1.0
        self.trade_log.append({
            'timestamp': t_time.isoformat() if hasattr(t_time, 'isoformat') else str(t_time),
            'symbol': symbol,
            'pnl': pnl,
            'balance': self.current_balance,
            'strategy': strategy,
            'net_return': net_return,
            'selected_leverage': leverage_v,
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
        wins = sum(1 for t in self.trade_log if (finite_float(t.get('pnl')) or 0.0) > 0)
        losses = total - wins
        total_pnl = sum(finite_float(t.get('pnl')) or 0.0 for t in self.trade_log)
        wr = (wins / total * 100) if total > 0 else 0
        current_unrealized, unpriced_equity = self._equity_components()
        balance = self.current_balance if math.isfinite(self.current_balance) else 0.0
        current_equity = balance + current_unrealized
        if not math.isfinite(current_equity):
            current_equity = balance
        peak_equity = finite_float(self.peak_equity) or 0.0
        peak_balance = finite_float(self.peak_balance) or 0.0
        max_drawdown = finite_float(self.max_drawdown_pct) or 0.0
        if getattr(self.config, "mode", "paper") == "paper":
            current_dd = ((peak_equity - current_equity) / peak_equity * 100
                          if peak_equity > 0 else 0.0)
            dd = max(max_drawdown * 100.0, current_dd)
        else:
            dd = ((peak_balance - balance) / peak_balance * 100
                  if peak_balance > 0 else 0.0)
        if not math.isfinite(dd):
            dd = max_drawdown * 100.0
        halted, halt_reason = self.circuit_breaker.is_halted()
        used_margin = gross_exposure = portfolio_heat = 0.0
        unpriced_stop_risk_positions = 0
        open_leverages = []
        for position in self.open_positions.values():
            if not isinstance(position, dict):
                unpriced_stop_risk_positions += 1
                continue
            entry = finite_float(position.get('entry_price'))
            mark = finite_float(position.get('mark_price'))
            quantity = finite_float(position.get('quantity'))
            if mark is None:
                mark = entry
            if (entry is not None and mark is not None and quantity is not None
                    and entry > 0 and mark > 0 and quantity > 0):
                gross_exposure += mark * quantity
            else:
                notional = finite_float(position.get('notional')) or 0.0
                gross_exposure += max(0.0, notional)
            margin = finite_float(position.get('margin'))
            if margin is not None and margin >= 0:
                used_margin += margin
            stop_risk = self.stop_risk_for_position(position)
            if stop_risk is None or not math.isfinite(stop_risk):
                unpriced_stop_risk_positions += 1
            else:
                portfolio_heat += stop_risk
            leverage = finite_float(position.get('leverage'))
            open_leverages.append(leverage if leverage is not None and leverage > 0 else 1.0)
        if not math.isfinite(portfolio_heat):
            portfolio_heat = 0.0
            unpriced_stop_risk_positions = max(unpriced_stop_risk_positions, 1)
        adaptive = getattr(self.config, 'adaptive_leverage', None)
        # The aggregate margin-allocation cap applies whether or not adaptive
        # leverage is enabled; disabling adaptive mode must not silently raise
        # the allocation ceiling to 100% of equity.
        margin_pct = finite_float(getattr(adaptive, 'max_margin_allocation_pct', None))
        if margin_pct is None:
            margin_pct = 0.50
        margin_pct = min(max(margin_pct, 0.0), 1.0)
        available_account_equity = max(0.0, current_equity - used_margin)
        remaining_margin_capacity = max(0.0, current_equity * margin_pct - used_margin)
        available_margin = min(available_account_equity, remaining_margin_capacity)
        leverage_values = [finite_float(t.get('selected_leverage')) or 1.0
                           for t in self.trade_log]
        risk_per_trade = finite_float(self.rc.risk_per_trade)
        if risk_per_trade is None:
            risk_per_trade = 0.0
        risk_per_trade = min(max(risk_per_trade, 0.0), 0.01)
        stop_risk_budget = (current_equity * risk_per_trade
                            if math.isfinite(current_equity) else 0.0)
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
            'portfolio_equity': current_equity,
            'realized_pnl': total_pnl,
            'unrealized_pnl': current_unrealized,
            'current_leverage': max(open_leverages, default=1.0),
            'average_leverage': (sum(leverage_values) / len(leverage_values)
                                 if leverage_values else 1.0),
            'maximum_leverage_used': max(leverage_values + open_leverages, default=1.0),
            'total_margin_used': used_margin,
            'available_margin': available_margin,
            'available_account_equity': available_account_equity,
            'remaining_margin_capacity': remaining_margin_capacity,
            'margin_allocation_cap': current_equity * margin_pct,
            'total_gross_exposure': gross_exposure,
            'gross_exposure_pct': (gross_exposure / current_equity * 100.0
                                   if current_equity > 0 else 0.0),
            'peak': peak_balance,
            'portfolio_heat': portfolio_heat,
            'portfolio_heat_pct': (portfolio_heat / current_equity * 100.0
                                   if current_equity > 0 else 0.0),
            'unpriced_stop_risk_positions': unpriced_stop_risk_positions,
            'unpriced_equity_positions': unpriced_equity,
            'stop_risk_budget': stop_risk_budget,
            'stop_risk_budget_pct': risk_per_trade * 100.0,
            'circuit_breaker_halted': halted,
            'circuit_breaker_reason': halt_reason,
        }
