"""Read-only report for a paper run: skips by reason, trades, costs, net PnL.

Reads logs/decisions.csv and a paper session SQLite database. It never writes to
either (the database is opened with SQLite's read-only mode, without calling
core.state, which creates tables). The only file it creates is the report text
file, and it refuses to overwrite an existing one.

Timestamps are UTC. In decisions.csv, ``ts`` is the candle (bar) time, not the
wall-clock time of the decision.
"""
import argparse
import csv
import math
import os
import sqlite3
import sys
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

GATE_BUCKETS = ("risk/reward filter", "volume filter", "net-reward (cost) filter")


def parse_time(value):
    """Parse an ISO-like timestamp to a naive UTC datetime, or None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bucket_status(status):
    """Group a decisions.csv status string into a short reason label."""
    status = str(status or "").strip()
    if not status.startswith("skipped:"):
        return status or "(empty)"
    reason = status[len("skipped:"):]
    if reason == "hold":
        return "no signal (hold)"
    if reason.startswith("risk/reward"):
        return "risk/reward filter"
    if reason.startswith("volume filter"):
        return "volume filter"
    if reason.startswith("cost-aware filter") or reason.startswith("cost:"):
        return "net-reward (cost) filter"
    return reason.split(":", 1)[0][:40] or "(other)"


def read_decisions(path, since, until):
    """Return (rows, note). Rows are dicts inside [since, until]."""
    if not path or not os.path.exists(path):
        return [], f"decisions file not found: {path}"
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            moment = parse_time(row.get("ts"))
            if moment is None:
                continue
            if since is not None and moment < since:
                continue
            if until is not None and moment > until:
                continue
            rows.append(row)
    return rows, ""


def open_readonly(db_path):
    uri = "file:" + urllib.parse.quote(str(Path(db_path).resolve()).replace("\\", "/")) + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_database(db_path, since, until):
    """Return (trades, open_positions, note) from a session database."""
    if not db_path or not os.path.exists(db_path):
        return [], [], f"session database not found: {db_path}"
    conn = open_readonly(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        trades, positions = [], []
        if "trades" in tables:
            for row in conn.execute("SELECT * FROM trades ORDER BY id"):
                item = dict(row)
                moment = parse_time(item.get("exit_time"))
                if moment is not None and ((since and moment < since) or (until and moment > until)):
                    continue
                trades.append(item)
        if "positions" in tables:
            positions = [dict(r) for r in conn.execute(
                "SELECT * FROM positions WHERE status='open' ORDER BY id")]
        return trades, positions, ""
    finally:
        conn.close()


def pct(value):
    return "n/a" if value is None else f"{value * 100:.3f}%"


def money(value):
    return "n/a" if value is None else f"${value:,.4f}"


def trade_section(trades):
    lines = ["TRADES (closed, from the session database; PnL is net of fees, slippage and funding)"]
    if not trades:
        lines.append("  No closed trades in this window.")
        return lines
    pnls = [number(t.get("pnl")) or 0.0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    returns = [(number(t.get("net_return")), p) for t, p in zip(trades, pnls)]
    win_returns = [r for r, p in returns if r is not None and p > 0]
    loss_returns = [r for r, p in returns if r is not None and p < 0]
    gross = [number(t.get("gross_pnl")) for t in trades]
    fees = sum(number(t.get("fees")) or 0.0 for t in trades)
    slip = sum(number(t.get("slippage")) or 0.0 for t in trades)
    funding = sum(number(t.get("funding_cost")) or 0.0 for t in trades)
    net = sum(pnls)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else None
    lines += [
        f"  Trades: {len(trades)}  wins: {len(wins)}  losses: {len(losses)}  "
        f"flat: {len(trades) - len(wins) - len(losses)}",
        f"  Win rate: {len(wins) / len(trades) * 100:.1f}%",
        f"  Net PnL: {money(net)}",
        f"  Average win: {pct(sum(win_returns) / len(win_returns)) if win_returns else 'n/a'} of notional",
        f"  Average loss: {pct(sum(loss_returns) / len(loss_returns)) if loss_returns else 'n/a'} of notional",
        "  Profit factor: " + (f"{profit_factor:.2f}" if profit_factor is not None
                               else ("n/a (no losing trades)" if wins else "n/a")),
        f"  Fees: {money(fees)}  slippage: {money(slip)}  funding: {money(funding)}",
    ]
    if all(g is not None for g in gross):
        implied = sum(gross) - fees - slip - funding
        diff = implied - net
        flag = "OK" if abs(diff) <= 0.01 else "MISMATCH - check the cost accounting"
        lines.append(f"  Cost check: gross {money(sum(gross))} - costs = {money(implied)} "
                     f"vs recorded net {money(net)} (diff {diff:+.4f}) {flag}")
    else:
        lines.append("  Cost check: skipped (some trades have no gross PnL recorded).")
    notional = sum(number(t.get("notional")) or 0.0 for t in trades)
    if notional > 0:
        lines.append(f"  Average cost per trade: {(fees + slip + funding) / notional * 10000:.1f} bps of notional")
    reasons = Counter(str(t.get("exit_reason") or "(none)") for t in trades)
    lines.append("  Exit reasons: " + ", ".join(f"{k}: {v}" for k, v in reasons.most_common()))
    by_symbol = defaultdict(list)
    for t, p in zip(trades, pnls):
        by_symbol[str(t.get("symbol") or "?")].append(p)
    for symbol, values in sorted(by_symbol.items()):
        lines.append(f"  {symbol}: {len(values)} trades, net {money(sum(values))}")
    if len(trades) < 30:
        lines.append(f"  NOTE: {len(trades)} trades is far too few to judge a strategy. "
                     "This is not evidence of profitability or of an edge.")
    return lines


def decision_section(rows, near_miss_limit):
    lines = ["DECISIONS (from logs/decisions.csv; ts is the candle time, UTC)"]
    if not rows:
        lines.append("  No decision rows in this window.")
        return lines
    buckets = Counter(bucket_status(r.get("status")) for r in rows)
    lines.append(f"  Rows: {len(rows)}  first: {min(r['ts'] for r in rows)}  last: {max(r['ts'] for r in rows)}")
    for label, count in buckets.most_common():
        lines.append(f"  {count:6d}  {label}")
    near = [r for r in rows if bucket_status(r.get("status")) in GATE_BUCKETS]
    lines.append(f"  Signals rejected by the size/volume gates: {len(near)}")
    per = Counter((r.get("symbol", "?"), r.get("timeframe", "?")) for r in near)
    for (symbol, timeframe), count in sorted(per.items()):
        lines.append(f"    {symbol} {timeframe}: {count}")
    for r in near[:near_miss_limit]:
        lines.append(f"    {r.get('ts')}  {r.get('symbol')}  {r.get('timeframe')}  "
                     f"{str(r.get('status'))[:110]}")
    if len(near) > near_miss_limit:
        lines.append(f"    ... {len(near) - near_miss_limit} more not shown")
    return lines


def position_section(positions):
    lines = ["OPEN POSITIONS (not included in the trade totals; no unrealized PnL is computed)"]
    if not positions:
        lines.append("  None.")
        return lines
    for p in positions:
        lines.append(f"  {p.get('symbol')} {p.get('side')} size {p.get('size')} entry {p.get('entry_price')} "
                     f"tp {p.get('tp')} sl {p.get('sl')} since {p.get('entry_time')} ({p.get('strategy')})")
    return lines


def build_report(args):
    since, until = parse_time(args.since), parse_time(args.until)
    db_path = args.db or (f"data/paper_sessions/paper_{args.session_id}.db" if args.session_id else None)
    rows, decisions_note = read_decisions(args.decisions, since, until)
    trades, positions, db_note = read_database(db_path, since, until)
    lines = [
        "PAPER RUN REPORT (read-only)",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Window: since {args.since or '(start)'}  until {args.until or '(now)'}",
        f"Decisions file: {args.decisions}",
        f"Session database: {db_path or '(not given)'}",
    ]
    for note in (decisions_note, db_note):
        if note:
            lines.append(f"WARNING: {note}")
    lines += [""] + decision_section(rows, args.near_miss_limit)
    lines += [""] + trade_section(trades)
    lines += [""] + position_section(positions)
    lines += ["", "This report describes what happened. It does not show that the strategy has an edge."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="Start of the window, UTC (e.g. 2026-11-25T20:53)")
    parser.add_argument("--until", help="End of the window, UTC")
    parser.add_argument("--session-id", help="Paper session id (uses data/paper_sessions/paper_<id>.db)")
    parser.add_argument("--db", help="Explicit session database path")
    parser.add_argument("--decisions", default="logs/decisions.csv")
    parser.add_argument("--out", help="Report file to create (default: paper_report_<UTC time>.txt); never overwritten")
    parser.add_argument("--near-miss-limit", type=int, default=30)
    args = parser.parse_args(argv)
    for name in ("since", "until"):
        value = getattr(args, name)
        if value and parse_time(value) is None:
            raise SystemExit(f"--{name} is not a valid timestamp: {value}")
    out = args.out or f"paper_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    if os.path.exists(out):
        raise SystemExit(f"Refusing to overwrite existing file: {out}")
    text = build_report(args)
    with open(out, "x", encoding="utf-8") as stream:
        stream.write(text)
    sys.stdout.write(text)
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
