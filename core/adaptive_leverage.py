"""Paper-only, walk-forward leverage decisions and portfolio sizing."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class EdgeEstimate:
    sample_size: int
    win_probability: float | None
    probability_lower_bound: float | None
    average_win: float | None
    average_loss: float | None
    break_even_probability: float | None
    expected_value: float | None
    edge: float | None
    recent_expected_value: float | None
    prior_expected_value: float | None
    deteriorating: bool


@dataclass(frozen=True)
class PortfolioSnapshot:
    equity: float
    available_balance: float
    used_margin: float
    available_margin: float
    unrealized_pnl: float
    gross_exposure: float
    maximum_gross_exposure: float
    remaining_exposure: float
    maximum_margin: float


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Two-sided 95% Wilson lower bound for an observed Bernoulli win rate."""
    if total <= 0:
        return 0.0
    p = min(max(float(wins) / total, 0.0), 1.0)
    z2 = float(z) ** 2
    denom = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    spread = float(z) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return min(max((center - spread) / denom, 0.0), 1.0)


def estimate_edge(trades: Iterable[dict], min_samples: int = 100,
                  rolling_window: int = 500, z: float = 1.96) -> EdgeEstimate:
    """Estimate net expectancy from completed prequential OOS outcomes.

    Each stored outcome was scored against a probability snapshot made before
    that trade opened. The current decision only sees outcomes closed earlier,
    in chronological order. Returns are net of fees, slippage, and funding.
    """
    clean = []
    for row in trades:
        if not row or not bool(row.get("oos_validated", True)):
            continue
        try:
            value = float(row["net_return"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    clean = clean[-max(1, int(rolling_window)):]
    wins = [value for value in clean if value > 0]
    losses = [-value for value in clean if value < 0]
    count = len(clean)
    if count < int(min_samples) or not wins or not losses:
        return EdgeEstimate(count, None, None, None, None, None, None, None,
                            None, None, False)

    probability = len(wins) / count
    lower = wilson_lower_bound(len(wins), count, z)
    avg_win, avg_loss = sum(wins) / len(wins), sum(losses) / len(losses)
    break_even = avg_loss / (avg_win + avg_loss)
    expected = lower * avg_win - (1.0 - lower) * avg_loss
    edge = lower - break_even

    # Compare the latest 25 completed outcomes with the preceding 25. A recent
    # negative mean forces 1x; a meaningful decline caps the choice one tier.
    recent_ev = prior_ev = None
    deteriorating = False
    if count >= 50:
        prior = clean[-50:-25]
        recent = clean[-25:]
        prior_ev = sum(prior) / len(prior)
        recent_ev = sum(recent) / len(recent)
        deteriorating = recent_ev < prior_ev - 0.0025
    return EdgeEstimate(count, probability, lower, avg_win, avg_loss,
                        break_even, expected, edge, recent_ev, prior_ev,
                        deteriorating)


def portfolio_snapshot(realized_balance: float, positions: dict,
                       max_margin_allocation_pct: float,
                       max_gross_exposure_pct: float) -> PortfolioSnapshot:
    """Compute equity, margin reservation, unrealized P&L, and exposure."""
    balance = float(realized_balance)
    gross = used_margin = unrealized = 0.0
    for position in (positions or {}).values():
        try:
            entry = float(position.get("entry_price") or position.get("entryPrice") or 0.0)
            quantity = float(position.get("quantity") or position.get("contracts") or 0.0)
            mark = float(position.get("mark_price") or position.get("markPrice") or entry)
            leverage = max(float(position.get("leverage") or 1.0), 1.0)
            notional = abs(mark * quantity) if quantity > 0 and mark > 0 else float(position.get("notional") or 0.0)
            side = str(position.get("side") or "long").lower()
        except (TypeError, ValueError):
            continue
        if not math.isfinite(notional) or notional < 0:
            continue
        gross += notional
        used_margin += float(position.get("margin") or notional / leverage)
        if quantity > 0 and entry > 0 and mark > 0:
            unrealized += (mark - entry) * quantity * (1.0 if side in ("long", "buy") else -1.0)

    equity = balance + unrealized
    max_margin = max(0.0, equity * float(max_margin_allocation_pct))
    max_gross = max(0.0, equity * float(max_gross_exposure_pct))
    available_balance = max(0.0, equity - used_margin)
    available_margin = max(0.0, min(available_balance, max_margin - used_margin))
    return PortfolioSnapshot(
        equity=equity, available_balance=available_balance,
        used_margin=used_margin, available_margin=available_margin,
        unrealized_pnl=unrealized, gross_exposure=gross,
        maximum_gross_exposure=max_gross,
        remaining_exposure=max(0.0, max_gross - gross),
        maximum_margin=max_margin,
    )


def _lower_tier(value: float, maximum: float) -> float:
    tiers = sorted({1.0, min(1.25, maximum), min(1.5, maximum),
                    min(2.0, maximum), maximum})
    eligible = [tier for tier in tiers if tier < value - 1e-9]
    return eligible[-1] if eligible else 1.0


def select_leverage(estimate: EdgeEstimate, *, max_leverage: float = 3.0,
                    min_leverage: float = 1.0, min_samples: int = 100,
                    edge_small: float = 0.02, edge_moderate: float = 0.05,
                    edge_strong: float = 0.08, edge_exceptional: float = 0.12,
                    consecutive_losses: int = 0, drawdown: float = 0.0,
                    reduce_drawdown: float = 0.02, min_only_drawdown: float = 0.04,
                    circuit_breaker_active: bool = False) -> tuple[float, str]:
    """Choose a tier from conservative net edge, then apply defensive caps."""
    maximum = float(max_leverage)
    floor = max(1.0, float(min_leverage))
    if (circuit_breaker_active or estimate.sample_size < min_samples or
            estimate.expected_value is None or estimate.edge is None):
        return 1.0, "circuit_breaker" if circuit_breaker_active else "insufficient_oos_evidence"
    if estimate.expected_value <= 0 or estimate.edge <= 0:
        return 1.0, "non_positive_conservative_expectancy"
    if estimate.recent_expected_value is not None and estimate.recent_expected_value <= 0:
        return 1.0, "recent_performance_non_positive"

    edge = estimate.edge
    if edge >= edge_exceptional:
        selected = maximum
        reason = "exceptional_validated_edge"
    elif edge >= edge_strong:
        selected = min(2.0, maximum)
        reason = "strong_validated_edge"
    elif edge >= edge_moderate:
        selected = min(1.5, maximum)
        reason = "moderate_validated_edge"
    elif edge >= edge_small:
        selected = min(1.25, maximum)
        reason = "small_validated_edge"
    else:
        return 1.0, "edge_below_first_tier"
    selected = min(maximum, max(floor, selected))

    if estimate.deteriorating:
        selected = _lower_tier(selected, maximum)
        reason = "reduced_for_deteriorating_performance"
    for _ in range(max(0, int(consecutive_losses))):
        selected = _lower_tier(selected, maximum)
    if consecutive_losses:
        reason = "reduced_for_consecutive_losses"
    if drawdown >= min_only_drawdown:
        return 1.0, "drawdown_minimum_only"
    if drawdown >= reduce_drawdown:
        selected = _lower_tier(selected, maximum)
        reason = "reduced_for_drawdown"
    return max(1.0, selected), reason


def stop_loss_risk_per_unit(entry: float, stop: float, *, entry_fee: float,
                            exit_fee: float, entry_slippage: float,
                            exit_slippage: float, funding_rate: float = 0.0,
                            hold_minutes: float = 0.0) -> float:
    """Worst-case per-unit stop loss including round-trip execution costs."""
    entry, stop = float(entry), float(stop)
    funding_cost = entry * abs(float(funding_rate)) * max(float(hold_minutes), 0.0) / 480.0
    return (abs(entry - stop) + entry * (entry_fee + entry_slippage) +
            stop * (exit_fee + exit_slippage) + funding_cost)


def build_paper_trade_plan(config, risk_manager, history, *, strategy: str,
                           entry: float, stop: float, side: str,
                           signal_confidence: float = 0.0) -> dict:
    """Size one paper entry under edge, stop-risk, margin, and exposure caps."""
    adaptive = config.adaptive_leverage
    entry, stop = float(entry), float(stop)
    if (float(adaptive.max_leverage) > 3.0 or
            float(adaptive.max_margin_allocation_pct) > 0.50 or
            float(adaptive.max_gross_exposure_pct) > 1.50):
        return {"allowed": False, "reason": "adaptive_configuration_exceeds_paper_safety_caps"}
    if (not math.isfinite(entry) or not math.isfinite(stop) or
            entry <= 0 or stop <= 0 or entry == stop):
        return {"allowed": False, "reason": "invalid_entry_or_stop"}
    snapshot = portfolio_snapshot(
        risk_manager.current_balance, risk_manager.open_positions,
        adaptive.max_margin_allocation_pct, adaptive.max_gross_exposure_pct,
    )
    estimate = estimate_edge(history, adaptive.min_oos_trades,
                             adaptive.history_window)
    peak_equity = float(getattr(risk_manager, "peak_equity", risk_manager.peak_balance))
    drawdown = ((peak_equity - snapshot.equity) / peak_equity
                if peak_equity > 0 else 0.0)
    halted, _ = risk_manager.circuit_breaker.is_halted()
    leverage, reason = select_leverage(
        estimate, max_leverage=adaptive.max_leverage,
        min_leverage=adaptive.min_leverage, min_samples=adaptive.min_oos_trades,
        edge_small=adaptive.edge_small, edge_moderate=adaptive.edge_moderate,
        edge_strong=adaptive.edge_strong, edge_exceptional=adaptive.edge_exceptional,
        consecutive_losses=risk_manager.consecutive_losses, drawdown=drawdown,
        reduce_drawdown=adaptive.reduce_drawdown_pct,
        min_only_drawdown=adaptive.min_only_drawdown_pct,
        circuit_breaker_active=halted,
    )

    entry_mode = config.execution.entry_mode
    if entry_mode != "market" and leverage > 1.0:
        leverage = 1.0
        reason = "maker_execution_is_limited_to_1x"
    entry_fee = (config.backtest.maker_fee if entry_mode == "maker"
                 else config.backtest.taker_fee)
    entry_slippage = 0.0 if entry_mode == "maker" else config.backtest.slippage
    exit_fee, exit_slippage = config.backtest.taker_fee, config.backtest.slippage
    funding_rate = config.backtest.funding_rate
    risk_per_unit = stop_loss_risk_per_unit(
        entry, stop, entry_fee=entry_fee, exit_fee=exit_fee,
        entry_slippage=entry_slippage, exit_slippage=exit_slippage,
        funding_rate=funding_rate,
        hold_minutes=config.risk.max_hold_time_minutes,
    )
    risk_pct = min(float(config.risk.risk_per_trade), 0.01)
    if config.risk.reduce_size_on_loss and risk_manager.consecutive_losses > 0:
        risk_pct *= (float(config.risk.size_reduction_factor) **
                     risk_manager.consecutive_losses)
    risk_budget = snapshot.equity * risk_pct
    max_position_notional = snapshot.equity * float(config.risk.max_position_size_pct)

    # The stop must be reached before isolated liquidation. Lower leverage one
    # tier at a time if the configured stop is too close to a liquidation price.
    while leverage > 1.0:
        maintenance = adaptive.maintenance_margin_rate
        liq = (entry * (1.0 - 1.0 / leverage + maintenance)
               if str(side).lower() in ("long", "buy") else
               entry * (1.0 + 1.0 / leverage - maintenance))
        stop_before_liquidation = (stop > liq if str(side).lower() in ("long", "buy")
                                   else stop < liq)
        if stop_before_liquidation:
            break
        leverage = _lower_tier(leverage, adaptive.max_leverage)
        reason = "reduced_to_keep_stop_ahead_of_liquidation"

    risk_sized_notional = (risk_budget / risk_per_unit * entry
                           if risk_per_unit > 0 else 0.0)
    margin_sized_notional = snapshot.available_margin * leverage
    notional = min(risk_sized_notional, max_position_notional,
                   snapshot.remaining_exposure, margin_sized_notional)
    if not math.isfinite(notional) or notional <= 0:
        return {"allowed": False, "reason": "no_margin_exposure_or_risk_capacity",
                "leverage": leverage, "portfolio": snapshot, "estimate": estimate,
                "risk_budget": risk_budget}
    quantity = notional / entry
    planned_risk = quantity * risk_per_unit
    required_margin = notional / leverage
    if planned_risk > risk_budget + max(1e-9, risk_budget * 1e-9):
        return {"allowed": False, "reason": "planned_stop_loss_exceeds_risk_budget",
                "leverage": leverage, "portfolio": snapshot, "estimate": estimate,
                "risk_budget": risk_budget, "planned_stop_risk": planned_risk}
    if required_margin > snapshot.available_margin + 1e-9:
        return {"allowed": False, "reason": "required_margin_exceeds_available_margin",
                "leverage": leverage, "portfolio": snapshot, "estimate": estimate,
                "risk_budget": risk_budget, "planned_stop_risk": planned_risk}
    return {
        "allowed": True, "reason": reason, "strategy": strategy,
        "signal_confidence": float(signal_confidence),
        "win_probability": estimate.win_probability,
        "probability_lower_bound": estimate.probability_lower_bound,
        "historical_sample_size": estimate.sample_size,
        "break_even_probability": estimate.break_even_probability,
        "expected_value": estimate.expected_value,
        "edge": estimate.edge, "leverage": leverage,
        "quantity": quantity, "position_notional": notional,
        "required_margin": required_margin,
        "planned_stop_risk": planned_risk, "risk_budget": risk_budget,
        "portfolio": snapshot, "estimate": estimate,
        "per_unit_stop_risk": risk_per_unit,
        "entry_fee_rate": entry_fee, "entry_slippage_rate": entry_slippage,
        "exit_fee_rate": exit_fee, "exit_slippage_rate": exit_slippage,
        "funding_rate": funding_rate,
    }

