import csv
import hashlib
import os

import pytest

import main
from core import state
from scripts import paper_report


def make_db(tmp_path, monkeypatch, trades=None, positions=True):
    path = tmp_path / "paper_test.db"
    monkeypatch.setattr(state, "DB_PATH", path)
    # (pnl, symbol, reason, exit_time, gross, fees, slippage, notional, net_return)
    trades = trades if trades is not None else [
        (4.7, "ETH/USDT", "Take profit", "2026-11-26T01:00:00.123", 5.0, 0.2, 0.1, 100.0, 0.047),
        (2.7, "SOL/USDT", "Time exit", "2026-11-26T03:00:00.456", 3.0, 0.2, 0.1, 100.0, 0.027),
        (-2.3, "ETH/USDT", "Stop loss", "2026-11-26T05:00:00.789", -2.0, 0.2, 0.1, 100.0, -0.023),
        (9.9, "BTC/USDT", "Take profit", "2026-11-24T05:00:00.000", 10.0, 0.05, 0.05, 100.0, 0.099),
    ]
    for pnl, symbol, reason, when, gross, fees, slip, notional, net_return in trades:
        state.record_trade(pnl, symbol, reason, when, strategy="ema_rsi_atr", notional=notional,
                           net_return=net_return, fees=fees, slippage=slip, funding_cost=0.0,
                           gross_pnl=gross)
    if positions:
        state.save_position_at_entry("SOL/USDT", "long", 1.0, 150.0, 160.0, 145.0, "ema_rsi_atr",
                                     entry_time="2026-11-26T05:00:00")
    return path


def make_decisions(tmp_path):
    path = tmp_path / "decisions.csv"
    rows = [
        ("2026-11-24T10:00:00", "BTC/USDT", "15m", "skipped:hold"),            # before the window
        ("2026-11-26T00:00:00", "BTC/USDT", "15m", "skipped:hold"),
        ("2026-11-26T00:15:00", "BTC/USDT", "15m", "skipped:hold"),
        ("2026-11-26T00:30:00", "ETH/USDT", "15m",
         "skipped:volume filter: last closed bar volume is 1.40x its 20-bar average, below 1.50x"),
        ("2026-11-26T00:45:00", "BTC/USDT", "15m",
         "skipped:risk/reward after costs 1.94 is below the configured minimum 2.00"),
        ("2026-11-26T04:00:00", "BTC/USDT", "15m",
         "skipped:cost-aware filter: net reward 1.2991% below minimum 1.3000%"),
        ("2026-11-26T05:00:00", "SOL/USDT", "1h", "entered"),
        ("2026-11-26T05:30:00", "ETH/USDT", "1h", "closed"),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=main.DECISION_FIELDS, restval="")
        writer.writeheader()
        for ts, symbol, timeframe, status in rows:
            writer.writerow({"ts": ts, "symbol": symbol, "timeframe": timeframe, "status": status})
    return path


def run(tmp_path, *extra):
    out = tmp_path / "report.txt"
    code = paper_report.main(["--out", str(out), *extra])
    assert code == 0
    return out.read_text(encoding="utf-8")


def test_report_summarizes_trades_costs_and_profit_factor(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch)
    text = run(tmp_path, "--db", str(db), "--decisions", str(make_decisions(tmp_path)),
               "--since", "2026-11-25T20:53")
    assert "Trades: 3  wins: 2  losses: 1" in text      # the Nov 24 trade is outside the window
    assert "Win rate: 66.7%" in text
    assert "Net PnL: $5.1000" in text
    assert "Average win: 3.700% of notional" in text
    assert "Average loss: -2.300% of notional" in text
    assert "Profit factor: 3.22" in text                 # 7.4 / 2.3
    assert "Fees: $0.6000  slippage: $0.3000  funding: $0.0000" in text
    assert "(diff +0.0000) OK" in text
    assert "Average cost per trade: 30.0 bps" in text
    assert "ETH/USDT: 2 trades" in text
    assert "far too few to judge" in text
    assert "edge" in text.lower()


def test_report_buckets_decisions_and_lists_gate_near_misses(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch)
    text = run(tmp_path, "--db", str(db), "--decisions", str(make_decisions(tmp_path)),
               "--since", "2026-11-25T20:53")
    assert "Rows: 7" in text                             # the Nov 24 row is filtered out
    assert "2  no signal (hold)" in text
    assert "1  entered" in text and "1  closed" in text
    assert "1  risk/reward filter" in text
    assert "1  volume filter" in text
    assert "1  net-reward (cost) filter" in text
    assert "Signals rejected by the size/volume gates: 3" in text
    assert "BTC/USDT 15m: 2" in text and "ETH/USDT 15m: 1" in text
    assert "1.2991%" in text


def test_open_positions_are_listed_separately_from_trade_totals(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch)
    text = run(tmp_path, "--db", str(db), "--decisions", str(make_decisions(tmp_path)))
    assert "OPEN POSITIONS" in text and "SOL/USDT long size 1.0 entry 150.0" in text
    assert "no unrealized PnL is computed" in text


def test_report_never_modifies_inputs_or_creates_other_files(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch)
    decisions = make_decisions(tmp_path)

    def snapshot():
        return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in tmp_path.iterdir()}

    before = snapshot()
    run(tmp_path, "--db", str(db), "--decisions", str(decisions))
    after = snapshot()
    assert set(after) - set(before) == {"report.txt"}
    for name, value in before.items():
        assert after[name] == value


def test_missing_database_is_reported_and_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "paper_x.db"
    text = run(tmp_path, "--db", str(missing), "--decisions", str(make_decisions(tmp_path)))
    assert "WARNING: session database not found" in text
    assert "No closed trades in this window." in text
    assert not missing.exists() and not missing.parent.exists()


def test_session_id_selects_the_session_database_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "paper_sessions").mkdir(parents=True)
    monkeypatch.setattr(state, "DB_PATH", tmp_path / "data" / "paper_sessions" / "paper_swing-1.db")
    state.record_trade(1.0, "BTC/USDT", "Take profit", "2026-11-26T01:00:00", strategy="s",
                       notional=100.0, net_return=0.01, fees=0.0, slippage=0.0, gross_pnl=1.0)
    text = run(tmp_path, "--session-id", "swing-1", "--decisions", str(tmp_path / "none.csv"))
    assert "Trades: 1  wins: 1  losses: 0" in text
    assert "Profit factor: n/a (no losing trades)" in text
    assert "WARNING: decisions file not found" in text


def test_cost_check_flags_a_net_that_does_not_match_gross_minus_costs(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch, trades=[
        (9.0, "BTC/USDT", "Take profit", "2026-11-26T01:00:00", 5.0, 0.2, 0.1, 100.0, 0.09)],
        positions=False)
    text = run(tmp_path, "--db", str(db), "--decisions", str(make_decisions(tmp_path)))
    assert "MISMATCH" in text
    assert "OPEN POSITIONS" in text and "None." in text


def test_refuses_to_overwrite_and_rejects_bad_timestamps(tmp_path, monkeypatch):
    db = make_db(tmp_path, monkeypatch)
    existing = tmp_path / "report.txt"
    existing.write_text("keep me")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        paper_report.main(["--out", str(existing), "--db", str(db)])
    assert existing.read_text() == "keep me"
    with pytest.raises(SystemExit, match="not a valid timestamp"):
        paper_report.main(["--out", str(tmp_path / "x.txt"), "--since", "yesterday"])


def test_bucket_status_groups_known_reasons():
    assert paper_report.bucket_status("skipped:hold") == "no signal (hold)"
    assert paper_report.bucket_status("skipped:risk:Max daily loss reached") == "risk"
    assert paper_report.bucket_status("skipped:cost:net_reward_0.1000%") == "net-reward (cost) filter"
    assert paper_report.bucket_status("entered") == "entered"
    assert paper_report.bucket_status("") == "(empty)"
