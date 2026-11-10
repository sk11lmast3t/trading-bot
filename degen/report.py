"""Descriptive cohort statistics; deliberately makes no recommendations."""
from __future__ import annotations

SNAPSHOTS = ((60, "1m"), (300, "5m"), (900, "15m"), (3600, "1h"),
             (21600, "6h"), (86400, "24h"))


def render_report(db):
    total = db.execute("SELECT COUNT(*) FROM launches").fetchone()[0]
    lines = ["# Degen token cohort report", "", f"Launches recorded: {total}", ""]
    for seconds, label in SNAPSHOTS:
        alive = db.execute("SELECT COUNT(*) FROM snapshots WHERE offset_seconds=? AND alive=1", (seconds,)).fetchone()[0]
        n = db.execute("SELECT COUNT(*) FROM snapshots WHERE offset_seconds=?", (seconds,)).fetchone()[0]
        pct = (100 * alive / n) if n else 0
        lines.append(f"- Alive at {label}: {alive}/{n} ({pct:.1f}%)")
    multiples = db.execute("SELECT l.mint,l.ts,l.launch_price,t.ts,t.price FROM launches l JOIN trades t ON t.mint=l.mint WHERE l.launch_price>0 ORDER BY l.mint,t.ts").fetchall()
    peaks = {}
    for row in multiples:
        multiple = float(row[4]) / float(row[2]) if row[2] else 0.0
        old = peaks.get(row[0])
        if old is None or multiple > old[1]:
            from datetime import datetime
            elapsed = (datetime.fromisoformat(row[3]) - datetime.fromisoformat(row[1])).total_seconds()
            peaks[row[0]] = (elapsed, multiple)
    vals = sorted(value for _, value in peaks.values())
    lines += ["", f"Tokens with priced snapshots: {len(vals)}"]
    if vals:
        def quantile(p): return vals[round((len(vals) - 1) * p)]
        peak_offsets = sorted(offset for offset, _ in peaks.values())
        lines += [f"Max multiple distribution (n={len(vals)}): min={vals[0]:.3f}x, median={quantile(.5):.3f}x, p90={quantile(.9):.3f}x, max={vals[-1]:.3f}x",
                  f"Time to peak (n={len(peak_offsets)}): median={peak_offsets[len(peak_offsets)//2]:.0f} seconds"]
    for label, cutoff in (("+2x", 2.0), ("+5x", 5.0)):
        reached = sum(value >= cutoff for value in vals)
        lines.append(f"- Reached {label} before death (snapshot proxy): {reached}/{len(vals)} ({(100*reached/len(vals)) if vals else 0:.1f}%)")
    if total < 100:
        lines += ["", f"**WARNING: sample bucket has {total} tokens; fewer than 100.**"]
    priced_n = len(vals)
    if priced_n < 100:
        lines += [f"**WARNING: max-multiple/time-to-peak bucket has {priced_n} tokens; fewer than 100.**"]
    return "\n".join(lines) + "\n"
