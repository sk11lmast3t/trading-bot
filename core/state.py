import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


_MODE = "paper"
DB_PATH = Path("data") / "paper.db"


def set_mode(mode):
    """Select isolated persistence for a bot mode; honor explicit test overrides."""
    global DB_PATH, _MODE
    mode = str(mode).lower()
    if mode not in ("paper", "live"):
        raise ValueError(f"Unsupported state mode: {mode}")
    if Path(DB_PATH) == Path("data") / f"{_MODE}.db":
        DB_PATH = Path("data") / f"{mode}.db"
    _MODE = mode


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init(conn)
    return conn


def _init(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        side TEXT,
        size REAL,
        entry_price REAL,
        tp REAL,
        sl REAL,
        strategy TEXT,
        entry_time TEXT,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pnl REAL,
        symbol TEXT,
        exit_reason TEXT,
        exit_time TEXT
    );
    CREATE TABLE IF NOT EXISTS risk_state (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        type TEXT,
        payload TEXT
    );
    """)
    conn.commit()


def save_position_at_entry(symbol, side, size, entry_price, tp, sl, strategy, entry_time=None, status="open"):
    if entry_time is None:
        entry_time = datetime.utcnow().isoformat()
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO positions (symbol, side, size, entry_price, tp, sl, strategy, entry_time, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (symbol, side, size, entry_price, tp, sl, strategy, entry_time, status),
        )
        conn.commit()
    finally:
        conn.close()


def close_position(symbol, status="closed"):
    conn = _conn()
    try:
        conn.execute("UPDATE positions SET status=? WHERE symbol=? AND status='open'", (status, symbol))
        conn.commit()
    finally:
        conn.close()


def record_trade(pnl, symbol, exit_reason, exit_time=None):
    if exit_time is None:
        exit_time = datetime.utcnow().isoformat()
    conn = _conn()
    try:
        conn.execute("INSERT INTO trades (pnl, symbol, exit_reason, exit_time) VALUES (?,?,?,?)", (pnl, symbol, exit_reason, exit_time))
        conn.commit()
    finally:
        conn.close()


def record_event(event_type, payload, ts=None):
    if ts is None:
        ts = datetime.utcnow().isoformat()
    conn = _conn()
    try:
        conn.execute("INSERT INTO events (ts, type, payload) VALUES (?,?,?)", (ts, event_type, payload if isinstance(payload, str) else json.dumps(payload)))
        conn.commit()
    finally:
        conn.close()


def load_open_positions():
    conn = _conn()
    try:
        rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_risk_state(key, value):
    conn = _conn()
    try:
        conn.execute("INSERT INTO risk_state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value if isinstance(value, str) else json.dumps(value)))
        conn.commit()
    finally:
        conn.close()


def load_risk_state():
    conn = _conn()
    try:
        rows = conn.execute("SELECT key, value FROM risk_state").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out
    finally:
        conn.close()


def report(days=7):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    conn = _conn()
    try:
        events = conn.execute("SELECT COUNT(*) c FROM events WHERE ts>=?", (since,)).fetchone()["c"]
        trades = conn.execute("SELECT COUNT(*) c, COALESCE(SUM(pnl),0) s FROM trades WHERE exit_time>=?", (since,)).fetchone()
        positions = conn.execute("SELECT COUNT(*) c FROM positions WHERE entry_time>=?", (since,)).fetchone()["c"]
        open_pos = conn.execute("SELECT COUNT(*) c FROM positions WHERE status='open'").fetchone()["c"]
        print(f"Last {days} days: events={events} trades={trades['c']} pnl={trades['s']:.4f} positions={positions} open={open_pos}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    r = sub.add_parser("report")
    r.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    if args.cmd == "report":
        report(args.days)


if __name__ == "__main__":
    main()
