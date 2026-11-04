import os, tempfile
from pathlib import Path
import core.state as st

def test_save_at_entry_persists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st.save_position_at_entry("BTC/USDT","long",1.0,100.0,110.0,95.0,"scalp")
    rows = st.load_open_positions()
    assert len(rows)==1
    assert rows[0]["symbol"]=="BTC/USDT"
