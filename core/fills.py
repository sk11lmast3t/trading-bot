"""Pure exit-fill helpers shared by paper execution and the backtester."""
from __future__ import annotations

import math


FILL_MODEL_VERSION = "shared-stop-gap-strict-tp-v1"


def _side(value: str) -> str:
    side = str(value).strip().lower()
    if side in ("long", "buy"):
        return "long"
    if side in ("short", "sell"):
        return "short"
    raise ValueError(f"Unknown position side: {value!r}")


def _positive_price(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive price")
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive price") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"{name} must be a finite positive price")
    return price


def stop_fill(side: str, level: float, ref_price: float, slippage: float) -> float:
    """Fill a triggered stop at the worse of its level and reference price.

    For a long stop, adverse slippage lowers the fill; for a short stop it raises
    it. ``ref_price`` is the candle open for bar checks and the latest polled
    price for poll-based stop checks.
    """
    side = _side(side)
    level = _positive_price("level", level)
    ref_price = _positive_price("ref_price", ref_price)
    if isinstance(slippage, bool):
        raise ValueError("slippage must be finite and in [0, 1)")
    try:
        slippage = float(slippage)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("slippage must be finite and in [0, 1)") from exc
    if not math.isfinite(slippage) or not 0 <= slippage < 1:
        raise ValueError("slippage must be finite and in [0, 1)")

    reference_fill = stop_reference_fill(side, level, ref_price)
    return reference_fill * (1.0 - slippage if side == "long" else 1.0 + slippage)


def stop_reference_fill(side: str, level: float, ref_price: float) -> float:
    """Return the gap-aware stop fill before adverse slippage."""
    side = _side(side)
    level = _positive_price("level", level)
    ref_price = _positive_price("ref_price", ref_price)
    return min(level, ref_price) if side == "long" else max(level, ref_price)


def tp_fill(side: str, tp: float, bar_high: float, bar_low: float) -> float | None:
    """Return the resting TP limit fill only when price trades strictly through it."""
    side = _side(side)
    tp = _positive_price("tp", tp)
    bar_high = _positive_price("bar_high", bar_high)
    bar_low = _positive_price("bar_low", bar_low)
    if bar_low > bar_high:
        raise ValueError("bar_low cannot exceed bar_high")
    through = bar_high > tp if side == "long" else bar_low < tp
    return tp if through else None


def exit_fee_rate(backtest_config, reason: str) -> float:
    """Return the configured fee rate; stops and other exits remain taker."""
    label = str(reason).strip().upper()
    is_tp = label in ("TP", "TAKE PROFIT")
    role = getattr(backtest_config, "tp_fee_role", "taker") if is_tp else "taker"
    if not isinstance(role, str) or role.lower() not in ("maker", "taker"):
        raise ValueError(f"tp_fee_role must be 'maker' or 'taker', got {role!r}")
    role = role.lower()
    raw_rate = getattr(backtest_config, f"{role}_fee", None)
    if isinstance(raw_rate, bool):
        raise ValueError(f"{role}_fee must be finite and non-negative")
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{role}_fee must be finite and non-negative") from exc
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(f"{role}_fee must be finite and non-negative")
    return rate
