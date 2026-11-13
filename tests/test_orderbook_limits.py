"""KuCoin order-book depth validation, caching, and graceful unavailability."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config.settings import BotConfig
from core.exchange import ExchangeConnector, PaperExchange


class FakeBookExchange:
    def __init__(self, book=None):
        self.book = book or {"bids": [[99.0, 1.0]], "asks": [[101.0, 1.0]]}
        self.limits = []

    def fetch_order_book(self, symbol, limit=None):
        self.limits.append(limit)
        return self.book


def paper_exchange(book_exchange):
    exchange = PaperExchange.__new__(PaperExchange)
    exchange.config = BotConfig()
    exchange._current_public = book_exchange
    exchange._orderbook_cache = {}
    exchange._orderbook_last_warning = {}
    return exchange


def test_paper_orderbook_rejects_unsupported_limit_without_calling_venue():
    venue = FakeBookExchange()
    exchange = paper_exchange(venue)

    assert exchange.fetch_orderbook("BTC/USDT", limit=10) is None
    assert venue.limits == []
    assert any("unsupported configured limit" in key[1]
               for key in exchange._orderbook_last_warning)


def test_paper_orderbook_defaults_to_supported_depth_and_caches():
    venue = FakeBookExchange()
    exchange = paper_exchange(venue)

    first = exchange.fetch_orderbook("BTC/USDT")
    second = exchange.fetch_orderbook("BTC/USDT")

    assert first is second
    assert venue.limits == [20]


def test_paper_orderbook_unavailable_is_graceful():
    exchange = paper_exchange(None)

    assert exchange.fetch_orderbook("BTC/USDT") is None
    assert any("endpoint unavailable" in key[1]
               for key in exchange._orderbook_last_warning)


def live_connector(venue):
    connector = ExchangeConnector.__new__(ExchangeConnector)
    connector.config = BotConfig()
    connector.exec_cfg = connector.config.execution
    connector.exchange = venue
    return connector


def test_live_connector_defaults_to_twenty_and_validates_before_calling():
    venue = FakeBookExchange()
    connector = live_connector(venue)

    assert connector.fetch_orderbook("BTC/USDT") is not None
    assert connector.fetch_orderbook("BTC/USDT", limit=50) is None
    assert venue.limits == [20]


def test_live_best_bid_ask_uses_supported_depth():
    venue = FakeBookExchange()
    connector = live_connector(venue)
    connector.create_limit_order = MagicMock(return_value="order")

    assert connector.create_best_bid_ask_limit_order("BTC/USDT", "buy", 1.0) == "order"
    assert venue.limits == [20]
    connector.create_limit_order.assert_called_once()


def test_live_slippage_check_uses_supported_depth(monkeypatch):
    venue = FakeBookExchange()
    connector = live_connector(venue)
    connector.orders = {}
    connector.retry = MagicMock()
    connector.slippage_model = MagicMock()
    connector.slippage_model.estimate.return_value = 0.0
    connector.config.strategy.orderbook_depth = 20
    venue.fetch_ticker = lambda _symbol: {"bid": 99.0, "ask": 101.0, "last": 100.0}
    connector.exchange.create_order = MagicMock(return_value={"id": "1"})
    connector.exchange.fetch_order = MagicMock(return_value={
        "filled": 1.0, "remaining": 0.0, "average": 100.0, "status": "closed", "fee": {"cost": 0.0},
    })

    connector.create_market_order("BTC/USDT", "buy", 1.0)

    assert venue.limits == [20]


def test_slippage_depth_config_is_not_used_as_a_venue_limit():
    # The order-book request limit must be 20/100 even though the internal
    # depth-level model defaults to 10 levels.
    cfg = BotConfig()
    assert cfg.execution.slippage_depth_levels == 10
    venue = FakeBookExchange()
    connector = live_connector(venue)
    connector.orders = {}
    connector.retry = MagicMock()
    connector.slippage_model = MagicMock()
    connector.slippage_model.estimate.return_value = 0.0
    venue.fetch_ticker = lambda _symbol: {"bid": 99.0, "ask": 101.0, "last": 100.0}
    connector.exchange.create_order = MagicMock(return_value={"id": "1"})
    connector.exchange.fetch_order = MagicMock(return_value={
        "filled": 1.0, "remaining": 0.0, "average": 100.0, "status": "closed", "fee": {"cost": 0.0},
    })

    connector.create_market_order("BTC/USDT", "buy", 1.0)

    assert venue.limits == [20]
