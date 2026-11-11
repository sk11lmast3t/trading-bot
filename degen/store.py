"""Append-only SQLite storage for degen observation data."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager


SCHEMA = """
CREATE TABLE IF NOT EXISTS launches (
 mint TEXT PRIMARY KEY, ts TEXT NOT NULL, creator TEXT NOT NULL,
 name TEXT NOT NULL, symbol TEXT NOT NULL, description TEXT NOT NULL,
 url TEXT NOT NULL, launch_price REAL
);
CREATE TABLE IF NOT EXISTS trades (
 id INTEGER PRIMARY KEY AUTOINCREMENT, mint TEXT NOT NULL, ts TEXT NOT NULL,
 side TEXT NOT NULL, sol_amount REAL NOT NULL, token_amount REAL NOT NULL,
 wallet TEXT NOT NULL, price REAL NOT NULL,
 FOREIGN KEY(mint) REFERENCES launches(mint)
);
CREATE INDEX IF NOT EXISTS trades_mint_ts ON trades(mint, ts);
CREATE TABLE IF NOT EXISTS snapshots (
 mint TEXT NOT NULL, offset_seconds INTEGER NOT NULL, ts TEXT NOT NULL,
 trade_count INTEGER NOT NULL, buy_sell_ratio REAL, unique_buyers INTEGER NOT NULL,
 top10_holder_share REAL, price_vs_launch REAL, sol_volume REAL NOT NULL,
 alive INTEGER NOT NULL, PRIMARY KEY(mint, offset_seconds),
 FOREIGN KEY(mint) REFERENCES launches(mint)
);
"""


class DegenStore:
    """One-connection, single-writer store. Rows are never deleted by this API."""

    def __init__(self, path="data/degen.db"):
        self.path = path
        self._connection = None
        self._writing = False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @contextmanager
    def writer(self):
        if self._writing:
            raise RuntimeError("DegenStore allows one writer at a time")
        self._writing = True
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            self._writing = False

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
