"""Human-readable, holdout-free reports for evolution cycles."""
from __future__ import annotations

from datetime import date
import os

DEFAULT_MAX_CANDIDATES = 3
DEFAULT_TOKEN_BUDGET = 12000


def render_cycle(records, settings=None, max_candidates=DEFAULT_MAX_CANDIDATES,
                 token_budget=DEFAULT_TOKEN_BUDGET):
    settings = settings or {"maker_fee": 0.0002, "taker_fee": 0.0005,
                            "slippage": 0.0003, "long_only": False}
    lines = [
        "# Strategy evolution cycle", "",
        ("Effective settings: "
         f"maker_fee={settings['maker_fee']}, taker_fee={settings['taker_fee']}, "
         f"slippage={settings['slippage']}, long_only={settings['long_only']}"),
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
        lines += [f"- Verdict: {row.get('verdict')}", f"- Reason: {'; '.join(row.get('reason_failed', [])) or 'passed all validation gates'}", ""]
    return "\n".join(lines)


def write_cycle(records, out_dir="reports", today=None, settings=None,
                max_candidates=DEFAULT_MAX_CANDIDATES, token_budget=DEFAULT_TOKEN_BUDGET):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evolve_{today or date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(render_cycle(records, settings, max_candidates, token_budget))
    return path
