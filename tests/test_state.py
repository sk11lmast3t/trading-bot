import os, tempfile
from pathlib import Path
import core.state as st

def test_save_at_entry_persists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st.save_position_at_entry("BTC/USDT","long",1.0,100.0,110.0,95.0,"scalp")
    rows = st.load_open_positions()
    assert len(rows)==1
    assert rows[0]["symbol"]=="BTC/USDT"


def test_paper_state_never_touches_live_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(st, "DB_PATH", Path("data") / "paper.db")
    monkeypatch.setattr(st, "_MODE", "paper")
    Path("data").mkdir()
    live_db = Path("data") / "live.db"
    live_db.write_bytes(b"live database sentinel")
    st.set_mode("paper")
    st.save_position_at_entry("BTC/USDT", "long", 1, 100, 110, 95, "test")
    assert (Path("data") / "paper.db").exists()
    assert live_db.read_bytes() == b"live database sentinel"
    assert not (Path("data") / "bot.db").exists()
