"""Human-readable, holdout-free reports for evolution cycles."""
from __future__ import annotations

from datetime import date
import os


def render_cycle(records):
    lines = ["# Strategy evolution cycle", "", "Search and validation results only.", ""]
    for row in records:
        lines += [f"## {row['name']}", "", f"Tried: {row.get('what', 'one fixed candidate rule set')}", ""]
        for part in ("search", "validation"):
            result = row.get(part, {})
            base = result.get("base", {})
            lines.append(f"- {part.title()}: trades={base.get('trades', 0)}, net bps/trade={base.get('net_bps_per_trade')}, adjusted CI={base.get('adjusted_ci')}")
        lines += [f"- Verdict: {row.get('verdict')}", f"- Reason: {'; '.join(row.get('reason_failed', [])) or 'passed all validation gates'}", ""]
    return "\n".join(lines)


def write_cycle(records, out_dir="reports", today=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evolve_{today or date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(render_cycle(records))
    return path
