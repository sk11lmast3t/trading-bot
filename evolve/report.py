"""Human-readable, holdout-free reports for evolution cycles."""
from __future__ import annotations

from datetime import date
import os

DEFAULT_MAX_CANDIDATES = 3
DEFAULT_TOKEN_BUDGET = 12000


def render_cycle(records, settings=None, max_candidates=DEFAULT_MAX_CANDIDATES,
                 token_budget=DEFAULT_TOKEN_BUDGET):
    settings = settings or {"maker_fee": 0.0002, "taker_fee": 0.0005,
                            "slippage": 0.0003, "long_only": False,
                            "stop_pct": 0.02, "min_target_pct": 0.05,
                            "max_leverage": 1.0}
    lines = [
        "# Strategy evolution cycle", "",
        ("Effective settings: "
         f"maker_fee={settings['maker_fee']}, taker_fee={settings['taker_fee']}, "
         f"slippage={settings['slippage']}, long_only={settings['long_only']}, "
         f"stop_pct={settings.get('stop_pct', 0.02)}, "
         f"min_target_pct={settings.get('min_target_pct', 0.05)}, "
         f"max_leverage={settings.get('max_leverage', 1.0)}"),
        (f"Cycle limits: --max-candidates={max_candidates} "
         f"(default {DEFAULT_MAX_CANDIDATES}); --token-budget={token_budget} "
         f"(default {DEFAULT_TOKEN_BUDGET})"),
        "", "Search and validation results only.", "",
    ]
    for row in records:
        lines += [f"## {row['name']}", "", f"Tried: {row.get('what', 'one fixed candidate rule set')}", ""]
        for part in ("search", "validation"):
            result = row.get(part, {})
            base = result.get("base", {})
            lines.append(f"- {part.title()}: trades={base.get('trades', 0)}, net bps/trade={base.get('net_bps_per_trade')}, adjusted CI={base.get('adjusted_ci')}")
        metrics = (row.get("validation", {}).get("base") or
                   row.get("search", {}).get("base") or {})
        stop = metrics.get("median_stop_pct")
        target = metrics.get("median_target_pct")
        hit_rate = metrics.get("target_before_stop_rate")
        if stop is not None and target is not None:
            costs_pct = 2 * (float(settings.get("taker_fee", 0.0005)) +
                             float(settings.get("slippage", 0.0003))) * 100.0
            denom = float(stop) + float(target)
            break_even = (float(stop) + costs_pct) / denom if denom else None
        else:
            break_even = None
        lines.append(f"- Median stop: {f'{float(stop):.3f}%' if stop is not None else 'n/a'}")
        lines.append(f"- Median target: {f'{float(target):.3f}%' if target is not None else 'n/a'}")
        lines.append(f"- Target before stop hit rate: {f'{float(hit_rate):.1%}' if hit_rate is not None else 'n/a'}")
        lines.append(f"- Break-even hit rate at configured costs: {f'{break_even:.1%}' if break_even is not None else 'n/a'}")
        lines += [f"- Verdict: {row.get('verdict')}", f"- Reason: {'; '.join(row.get('reason_failed', [])) or 'passed all validation gates'}", ""]
    return "\n".join(lines)


def write_cycle(records, out_dir="reports", today=None, settings=None,
                max_candidates=DEFAULT_MAX_CANDIDATES, token_budget=DEFAULT_TOKEN_BUDGET):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evolve_{today or date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(render_cycle(records, settings, max_candidates, token_budget))
    return path
