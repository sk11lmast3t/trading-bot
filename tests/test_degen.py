from datetime import datetime, timedelta, timezone
import sqlite3
import pytest

from degen.collector import Collector, SNAPSHOTS
from degen.source import Event, FakeSource
from degen.store import DegenStore


def test_every_launch_including_immediately_dead_has_all_offsets(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = [Event("launch", "quiet", start, creator="c", name="Quiet", symbol="Q")]
    # Advance source past every scheduled horizon with an unrelated launch.
    events.append(Event("launch", "next", start + timedelta(days=2), creator="c"))
    store = DegenStore(str(tmp_path / "degen.db"))
    assert Collector(FakeSource(events), store).collect() == 2
    assert store.connection.execute("SELECT COUNT(*) FROM launches").fetchone()[0] == 2
    rows = store.connection.execute("SELECT offset_seconds,alive FROM snapshots WHERE mint='quiet' ORDER BY offset_seconds").fetchall()
    assert [row[0] for row in rows] == [seconds for seconds, _ in SNAPSHOTS]
    assert rows[0][1] == 1
    assert rows[-1][1] == 0


def test_snapshot_offsets_and_trade_metrics(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = [
        Event("launch", "mint", start, name="T"),
        Event("trade", "mint", start + timedelta(seconds=30), side="buy", sol_amount=2, token_amount=10, wallet="a"),
        Event("trade", "mint", start + timedelta(seconds=120), side="sell", sol_amount=3, token_amount=10, wallet="b"),
        Event("launch", "advance", start + timedelta(days=2)),
    ]
    store = DegenStore(str(tmp_path / "degen.db"))
    Collector(FakeSource(events, {"mint": .42}), store).collect()
    row = store.connection.execute("SELECT trade_count,buy_sell_ratio,unique_buyers,top10_holder_share,price_vs_launch,sol_volume FROM snapshots WHERE mint='mint' AND offset_seconds=300").fetchone()
    assert tuple(row[:4]) == (2, 1.0, 1, .42)
    assert row[4] == pytest.approx(1.5)
    assert row[5] == 5.0


def test_untrusted_text_is_truncated_and_never_executed(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    marker = tmp_path / "executed"
    evil = f"__import__('pathlib').Path({str(marker)!r}).touch()" + "x" * 300
    store = DegenStore(str(tmp_path / "degen.db"))
    Collector(FakeSource([Event("launch", "mint", start, name=evil, symbol=evil,
                                description=evil, url=evil), Event("launch", "next", start+timedelta(days=2))]), store).collect()
    row = store.connection.execute("SELECT name,symbol,description,url FROM launches WHERE mint='mint'").fetchone()
    assert all(len(value) == 200 for value in row)
    assert not marker.exists()


def test_degen_sources_contain_no_signing_imports():
    import pathlib
    root = pathlib.Path(__file__).parents[1] / "degen"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "solders.keypair" not in source.lower()
    assert "solana.keypair" not in source.lower()
    assert "Keypair" not in source
