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


def set_database(path):
    """Use an explicit persistence file for an isolated paper session."""
    global DB_PATH
    DB_PATH = Path(path)


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
        status TEXT,
        leverage REAL NOT NULL DEFAULT 1.0,
        margin REAL NOT NULL DEFAULT 0.0
    );
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pnl REAL,
        gross_pnl REAL,
        symbol TEXT,
        exit_reason TEXT,
        exit_time TEXT,
        strategy TEXT,
        notional REAL,
        net_return REAL,
        prediction_probability REAL,
        probability_lower_bound REAL,
        break_even_probability REAL,
        expected_value REAL,
        selected_leverage REAL NOT NULL DEFAULT 1.0,
        required_margin REAL,
        funding_cost REAL,
        fees REAL NOT NULL DEFAULT 0.0,
        slippage REAL NOT NULL DEFAULT 0.0,
        oos_validated INTEGER NOT NULL DEFAULT 0
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
    migrations = {
        "positions": {
            "leverage": "REAL NOT NULL DEFAULT 1.0",
            "margin": "REAL NOT NULL DEFAULT 0.0",
        },
        "trades": {
            "strategy": "TEXT", "notional": "REAL", "net_return": "REAL",
            "gross_pnl": "REAL",
            "prediction_probability": "REAL", "probability_lower_bound": "REAL",
            "break_even_probability": "REAL", "expected_value": "REAL",
            "selected_leverage": "REAL NOT NULL DEFAULT 1.0",
            "required_margin": "REAL", "funding_cost": "REAL",
            "fees": "REAL NOT NULL DEFAULT 0.0", "slippage": "REAL NOT NULL DEFAULT 0.0",
            "oos_validated": "INTEGER NOT NULL DEFAULT 0",
        },
    }
    for table, columns in migrations.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, declaration in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    conn.commit()


def save_position_at_entry(symbol, side, size, entry_price, tp, sl, strategy,
                           entry_time=None, status="open", leverage=1.0, margin=0.0):
    if entry_time is None:
        entry_time = datetime.utcnow().isoformat()
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO positions (symbol, side, size, entry_price, tp, sl, strategy, entry_time, status, leverage, margin) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, side, size, entry_price, tp, sl, strategy, entry_time, status, leverage, margin),
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


def record_trade(pnl, symbol, exit_reason, exit_time=None, *, strategy="", notional=None,
                 net_return=None, prediction_probability=None,
                 probability_lower_bound=None, break_even_probability=None,
                 expected_value=None, selected_leverage=1.0, required_margin=None,
                 funding_cost=0.0, fees=0.0, slippage=0.0, gross_pnl=None,
                 oos_validated=False):
    if exit_time is None:
        exit_time = datetime.utcnow().isoformat()
    conn = _conn()
    try:
        conn.execute(
            """INSERT INTO trades (
                pnl, gross_pnl, symbol, exit_reason, exit_time, strategy, notional, net_return,
                prediction_probability, probability_lower_bound,
                break_even_probability, expected_value, selected_leverage,
                required_margin, funding_cost, fees, slippage, oos_validated
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pnl, gross_pnl, symbol, exit_reason, exit_time, strategy, notional, net_return,
             prediction_probability, probability_lower_bound, break_even_probability,
             expected_value, selected_leverage, required_margin, funding_cost, fees, slippage,
             int(bool(oos_validated))),
        )
        conn.commit()
    finally:
        conn.close()


def record_event(event_type, payload, ts=None):
    """Insert one lifecycle event.

    NOTE: as of this audit the bot does not call this function during normal
    operation, so the ``events`` table stays empty unless a caller or the
    ``stats`` report queries it. It is kept as an explicit, tested API for
    future telemetry rather than being claimed as populated.
    """
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
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY id ASC"
        ).fetchall()
        # A symbol can accumulate duplicate open rows if a process died between
        # the insert and the close update. The latest row is authoritative, and
        # ordering by id makes the choice deterministic across runs.
        ordered = []
        index = {}
        for row in rows:
            entry = dict(row)
            symbol = entry.get("symbol")
            if not symbol:
                ordered.append(entry)
                continue
            if symbol in index:
                ordered[index[symbol]] = entry
            else:
                index[symbol] = len(ordered)
                ordered.append(entry)
        return ordered
    finally:
        conn.close()


def save_risk_state(key, value):
    """Upsert a key/value risk marker.

    NOTE: currently unused by the bot lifecycle; risk telemetry is persisted
    in the JSON state snapshot. This API is retained for isolated callers and
    is not evidence that the ``risk_state`` table is populated in a session.
    """
    conn = _conn()
    try:
        conn.execute("INSERT INTO risk_state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value if isinstance(value, str) else json.dumps(value)))
        conn.commit()
    finally:
        conn.close()


def load_risk_state():
    """Read all ``risk_state`` key/value pairs.

    NOTE: the bot does not currently read this table during recovery; the JSON
    snapshot is authoritative for dynamic position state.
    """
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


def load_strategy_trades(strategy, limit=500, before=None):
    """Load validated completed outcomes, optionally closed strictly before an entry time."""
    conn = _conn()
    try:
        cutoff = before.isoformat() if hasattr(before, "isoformat") else before
        cutoff_sql = " AND exit_time < ?" if cutoff is not None else ""
        rows = conn.execute(
            f"""SELECT id, exit_time, strategy, net_return, prediction_probability,
                      probability_lower_bound, break_even_probability, expected_value,
                      selected_leverage, oos_validated
               FROM trades WHERE strategy=? AND oos_validated=1 AND net_return IS NOT NULL{cutoff_sql}
               ORDER BY id DESC LIMIT ?""",
            ((str(strategy), cutoff, max(1, int(limit))) if cutoff is not None
             else (str(strategy), max(1, int(limit)))),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]
    finally:
        conn.close()


def strategy_performance():
    """Summarize realized paper outcomes by stored strategy@timeframe label."""
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT strategy, symbol, net_return, pnl, fees, slippage, oos_validated FROM trades
               WHERE strategy IS NOT NULL AND strategy != '' AND net_return IS NOT NULL
               ORDER BY id"""
        ).fetchall()
    finally:
        conn.close()
    grouped = {}
    for row in rows:
        item = grouped.setdefault(row["strategy"], {"returns": [], "oos": 0, "symbols": {}})
        value = float(row["net_return"])
        item["returns"].append(value)
        item["oos"] += int(bool(row["oos_validated"]))
        symbol = item["symbols"].setdefault(row["symbol"], {"returns": [], "oos": 0})
        symbol["returns"].append(value)
        symbol["oos"] += int(bool(row["oos_validated"]))
    result = {}
    for strategy, item in grouped.items():
        returns = item["returns"]
        wins = [value for value in returns if value > 0]
        losses = [-value for value in returns if value < 0]
        equity = peak = 1.0
        max_drawdown = 0.0
        loss_streak = max_loss_streak = 0
        for value in returns:
            equity *= 1.0 + value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
            loss_streak = loss_streak + 1 if value < 0 else 0
            max_loss_streak = max(max_loss_streak, loss_streak)
        symbol_metrics = {}
        for symbol_name, symbol_stats in item["symbols"].items():
            values = symbol_stats["returns"]
            symbol_metrics[symbol_name] = {
                "completed": len(values), "oos_validated": symbol_stats["oos"],
                "win_rate": sum(v > 0 for v in values) / len(values),
                "net_expectancy": sum(values) / len(values),
            }
        result[strategy] = {
            "completed": len(returns), "oos_validated": item["oos"],
            "win_rate": len(wins) / len(returns) if returns else 0.0,
            "average_win": sum(wins) / len(wins) if wins else 0.0,
            "average_loss": sum(losses) / len(losses) if losses else 0.0,
            "net_expectancy": sum(returns) / len(returns) if returns else 0.0,
            "profit_factor": sum(wins) / sum(losses) if losses else (float("inf") if wins else 0.0),
            "fees": sum(float(row["fees"] or 0) for row in rows if row["strategy"] == strategy),
            "slippage": sum(float(row["slippage"] or 0) for row in rows if row["strategy"] == strategy),
            "max_drawdown": max_drawdown, "max_consecutive_losses": max_loss_streak,
            "wilson_win_rate_lower_bound": __import__(
                "core.adaptive_leverage", fromlist=["wilson_lower_bound"]
            ).wilson_lower_bound(len(wins), len(returns)),
            "symbols": symbol_metrics,
        }
    return result


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
