from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config.settings import BotConfig
from core.exchange import ExchangeConnector, OrderRecord, OrderState, RetryHandler, SlippageModel
from main import UnifiedCryptoBot


def _bot():
    cfg = BotConfig(mode="live")
    cfg.scalping.scan_all_coins = False
    bot = UnifiedCryptoBot(config=cfg)
    bot.exchange = MagicMock()
    bot.exchange.position_check_error = None
    bot.active_decisions = {}
    bot.position_targets = {"BTC/USDT": {}}
    bot.position_entry_cycle = {"BTC/USDT": 1}
    bot.persist_state = MagicMock()
    bot.risk_manager.release_position = MagicMock()
    bot.risk_manager.record_trade = MagicMock()
    bot.entries_blocked = False
    return bot


def test_stop_already_filled_sends_zero_orders():
    bot = _bot()
    bot.exchange.get_position.return_value = None
    bot._record_exchange_stop_flat = MagicMock()
    bot._close_position("BTC/USDT", {"entryPrice": 100, "contracts": 3, "side": "long"}, 95, "stop")
    bot.exchange.create_market_order.assert_not_called()
    bot._record_exchange_stop_flat.assert_called_once_with("BTC/USDT")
    bot.risk_manager.release_position.assert_called_once_with("BTC/USDT")


def test_exchange_closed_without_journal_context_is_still_recorded():
    bot = _bot()
    bot.risk_manager.open_positions = {"BTC/USDT": {"entryPrice": 100, "contracts": 1, "side": "long"}}
    bot._append_decision = MagicMock()
    bot._record_exchange_stop_flat("BTC/USDT")
    row = bot._append_decision.call_args.args[0]
    assert row["status"] == "closed"
    assert row["close_source"] == "exchange_stop_inferred"


def test_close_uses_fresh_contract_size_and_reduce_only():
    bot = _bot()
    bot.exchange.get_position.return_value = {"entryPrice": "100", "contracts": "1.25", "side": "short"}
    bot.exchange.create_market_order.return_value = OrderRecord("close", "BTC/USDT", "buy", "market", 1.25,
                                                               None, state=OrderState.FILLED,
                                                               filled=1.25, average_price=101)
    with patch("main.state.record_trade"), patch("main.audit"):
        bot._close_position("BTC/USDT", {"entryPrice": 100, "contracts": 99, "side": "long"}, 101, "manual")
    bot.exchange.create_market_order.assert_called_once_with("BTC/USDT", "buy", 1.25, reduce_only=True)
    bot.exchange.cancel_all_orders.assert_called_once_with("BTC/USDT")


def test_flat_reduce_only_rejection_is_recorded_without_retry():
    bot = _bot()
    bot.exchange.get_position.return_value = {"entryPrice": 100, "contracts": 2, "side": "long"}
    bot.exchange.create_market_order.return_value = OrderRecord(
        "close", "BTC/USDT", "sell", "market", 2, None,
        state=OrderState.REJECTED, last_error="ReduceOnly order rejected: position is zero")
    bot._record_exchange_stop_flat = MagicMock()
    bot._close_position("BTC/USDT", {"entryPrice": 100, "contracts": 2, "side": "long"}, 98, "stop")
    bot.exchange.create_market_order.assert_called_once()
    bot._record_exchange_stop_flat.assert_called_once_with("BTC/USDT")
    bot.risk_manager.release_position.assert_called_once_with("BTC/USDT")


def _connector(exchange):
    connector = ExchangeConnector.__new__(ExchangeConnector)
    connector.exchange = exchange
    connector.exec_cfg = SimpleNamespace(slippage_model_enabled=False, max_order_retries=0,
                                         retry_backoff_base_ms=0, retry_backoff_max_ms=0)
    connector.retry = RetryHandler(max_retries=0)
    connector.slippage_model = SlippageModel()
    connector.orders = {}
    return connector


def test_sparse_create_response_reads_fill_from_fetch_order():
    exchange = MagicMock()
    exchange.create_order.return_value = {"id": "x", "filled": None, "average": None,
                                          "remaining": None, "fee": None}
    exchange.fetch_order.return_value = {"id": "x", "status": "closed", "filled": 2.0,
                                         "average": 101.5, "remaining": 0.0, "fee": {"cost": 0.1}}
    order = _connector(exchange).create_order_with_retry("BTC/USDT", "market", "buy", 2,
                                                          check_slippage=False)
    assert order.state == OrderState.FILLED
    assert order.filled == 2.0
    assert order.average_price == 101.5
    exchange.fetch_order.assert_called_once_with("x", "BTC/USDT")


def test_fetch_failure_returns_unknown_and_blocks_entries_before_stop():
    exchange = MagicMock()
    exchange.create_order.return_value = {"id": "x", "filled": None, "average": None,
                                          "remaining": None, "fee": None}
    exchange.fetch_order.side_effect = RuntimeError("private endpoint unavailable")
    order = _connector(exchange).create_order_with_retry("BTC/USDT", "market", "buy", 2,
                                                          check_slippage=False)
    assert order.state == OrderState.UNKNOWN
    assert order.last_error

    bot = _bot()
    bot.exchange.create_market_order.return_value = order
    bot.exchange.get_position.return_value = {"contracts": 2}
    bot._block_unknown_order("BTC/USDT", "entry")
    assert bot.entries_blocked is True
    bot.exchange.get_position.assert_called_once_with("BTC/USDT")
    bot.exchange.create_stop_loss.assert_not_called()


def test_ambiguous_submission_timeout_is_unknown_without_retry():
    exchange = MagicMock()
    exchange.create_order.side_effect = TimeoutError("network timeout after submit")
    connector = _connector(exchange)
    order = connector.create_order_with_retry("BTC/USDT", "market", "buy", 2,
                                              check_slippage=False)
    assert order.state == OrderState.UNKNOWN
    exchange.create_order.assert_called_once()
