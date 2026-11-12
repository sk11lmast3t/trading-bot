import csv

import main
from scripts.journal_report import report


def test_journal_report_warns_on_small_buckets_without_recommendations(tmp_path, capsys):
    path = tmp_path / "journal.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=main.DECISION_FIELDS)
        writer.writeheader()
        writer.writerow({
            "ts": "2026-04-06T12:00:00", "symbol": "BTC/USDT", "strategy": "synthetic",
            "status": "closed", "net_pnl": 1.0, "notional": 100.0,
            "atr_1h_pct": 1.0, "ema20_1h_slope_sign": "up", "ema20_1h_slope_pct": 0.02,
            "trend_30m": "up", "rsi_5m": 60, "ema_distance_atr_5m": 0.5,
            "hour_utc": 12, "day_of_week_utc": 0, "rv_24h_bucket": "medium",
        })

    report(str(path))

    output = capsys.readouterr().out
    assert "closed: 1" in output
    assert "100.00 bps" in output
    assert "TOO FEW TO CONCLUDE" in output
    assert "best rule" not in output.lower()
    assert "recommendation" not in output.lower()
