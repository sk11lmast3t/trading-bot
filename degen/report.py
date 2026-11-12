"""Descriptive cohort statistics; deliberately makes no recommendations."""
from __future__ import annotations

SNAPSHOTS = ((60, "1m"), (300, "5m"), (900, "15m"), (3600, "1h"),
             (21600, "6h"), (86400, "24h"))


def render_report(db):
    total = db.execute("SELECT COUNT(*) FROM launches").fetchone()[0]
    lines = ["# Degen token cohort report", "", f"Launches recorded: {total}", ""]
    warnings = []
    for seconds, label in SNAPSHOTS:
        alive = db.execute("SELECT COUNT(*) FROM snapshots WHERE offset_seconds=? AND alive=1", (seconds,)).fetchone()[0]
        n = db.execute("SELECT COUNT(*) FROM snapshots WHERE offset_seconds=?", (seconds,)).fetchone()[0]
        pct = (100 * alive / n) if n else 0
        dead = n - alive
        lines.append(f"- Alive at {label}: {alive}/{n} ({pct:.1f}%); dead: {dead}/{n}")
        if alive < 100:
            warnings.append(f"alive at {label} bucket has {alive} tokens")
        if dead < 100:
            warnings.append(f"dead at {label} bucket has {dead} tokens")
    multiples = db.execute("SELECT l.mint,l.ts,l.launch_price,t.ts,t.price FROM launches l JOIN trades t ON t.mint=l.mint WHERE l.launch_price>0 ORDER BY l.mint,t.ts").fetchall()
    peaks = {}
    previous = {}
    for row in multiples:
        from datetime import datetime
        launch_time = datetime.fromisoformat(row[1])
        trade_time = datetime.fromisoformat(row[3])
        prior = previous.get(row[0])
        if prior is not None and (trade_time - prior).total_seconds() >= 1800:
            continue
        previous[row[0]] = trade_time
        multiple = float(row[4]) / float(row[2]) if row[2] else 0.0
        old = peaks.get(row[0])
        if old is None or multiple > old[1]:
            elapsed = (trade_time - launch_time).total_seconds()
            peaks[row[0]] = (elapsed, multiple)
    vals = sorted(value for _, value in peaks.values())
    lines += ["", f"Tokens with pre-death trade prices (n={len(vals)})"]
    if vals:
        def quantile(p): return vals[round((len(vals) - 1) * p)]
        peak_offsets = sorted(offset for offset, _ in peaks.values())
        def time_quantile(p): return peak_offsets[round((len(peak_offsets) - 1) * p)]
        lines += [f"Max multiple distribution (n={len(vals)}): min={vals[0]:.3f}x, median={quantile(.5):.3f}x, p90={quantile(.9):.3f}x, max={vals[-1]:.3f}x",
                  f"Time to peak distribution (n={len(peak_offsets)}): min={peak_offsets[0]:.0f}s, median={time_quantile(.5):.0f}s, p90={time_quantile(.9):.0f}s, max={peak_offsets[-1]:.0f}s"]
    for label, cutoff in (("+2x", 2.0), ("+5x", 5.0)):
        reached = sum(value >= cutoff for value in vals)
        lines.append(f"- Reached {label} before first 30-minute inactivity death: {reached}/{total} tokens ({(100*reached/total) if total else 0:.1f}%; priced sample n={len(vals)})")
        if reached < 100:
            warnings.append(f"{label} outcome bucket has {reached} tokens")
    if total < 100:
        warnings.append(f"launch cohort has {total} tokens")
    priced_n = len(vals)
    if priced_n < 100:
        warnings.append(f"max-multiple/time-to-peak bucket has {priced_n} tokens")
    if warnings:
        lines += ["", "**WARNING: buckets under 100 tokens:**"]
        lines.extend(f"- {warning} (<100)" for warning in warnings)
    return "\n".join(lines) + "\n"
