"""Event sources for the observation-only collector.

The production adapter is intentionally a documented stub. Fill it only with
an approved, read-only market data source; this package has no wallet or order
interfaces. All descriptive strings are untrusted display data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Event:
    kind: str
    mint: str
    ts: datetime
    creator: str = ""
    name: str = ""
    symbol: str = ""
    side: str = ""
    sol_amount: float = 0.0
    token_amount: float = 0.0
    wallet: str = ""
    description: str = ""
    url: str = ""


class EventSource(ABC):
    """Read-only source of normalized ``launch`` and ``trade`` events."""

    @abstractmethod
    def events(self) -> Iterable[Event]:
        """Yield events ordered by event time."""

    def top_holder_share(self, mint: str, ts: datetime):
        """Return optional top-ten holder share (0..1), if the source provides it."""
        return None


class ApprovedSourceStub(EventSource):
    """Documented integration point; implement with an approved read-only API."""

    def events(self) -> Iterator[Event]:
        raise NotImplementedError("Add an approved observation-only data adapter")
        yield  # pragma: no cover


class FakeSource(EventSource):
    """Deterministic in-memory source for tests; never performs network I/O."""

    def __init__(self, events=(), holder_shares=None):
        self._events = tuple(sorted(events, key=lambda event: event.ts))
        self._holder_shares = dict(holder_shares or {})

    def events(self):
        return iter(self._events)

    def top_holder_share(self, mint, ts):
        value = self._holder_shares.get(mint)
        return value(ts) if callable(value) else value
