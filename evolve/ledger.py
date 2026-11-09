"""Append-only JSONL audit ledger for candidate evaluation and holdout use."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


LEDGER_PATH = os.path.join("evolve", "ledger.jsonl")


def read_entries(path=LEDGER_PATH):
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def append_entry(entry, path=LEDGER_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **entry}
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def latest_candidate(name, path=LEDGER_PATH):
    rows = [row for row in read_entries(path)
            if row.get("name") == name and row.get("event", "candidate") == "candidate"]
    return rows[-1] if rows else None


def holdout_attempted(name, path=LEDGER_PATH):
    return any(row.get("name") == name and row.get("event") == "holdout_attempt"
               for row in read_entries(path))


def prompt_entries(path=LEDGER_PATH, limit=10):
    """Return recent ledger context with all holdout content removed."""
    safe = []
    for row in read_entries(path):
        if row.get("event") != "candidate":
            continue
        clean = {key: value for key, value in row.items()
                 if key not in {"holdout", "holdout_result", "holdout_metrics"}}
        safe.append(clean)
    return safe[-max(0, int(limit)):]

