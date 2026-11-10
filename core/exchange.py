"""
Unified Exchange Connector & Paper Trading Simulator
Supports CCXT live futures/spot connectivity as well as offline paper trading.
Execution Quality Enhancements (P1):
- Order state machine (pending -> open/filled/partial/rejected/cancelled)
- Retry with exponential backoff on transient errors
- Slippage estimation from orderbook depth
- Post-only / reduce-only order flags
- Paper trading realism: latency, slippage, partial fills, rejects
"""
import ccxt
import pandas as pd
import logging
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class OrderRecord:
    id: str
    symbol: str
    side: str
    type: str
    amount: float
    price: Optional[float]
    state: OrderState = OrderState.PENDING
    filled: float = 0.0
    remaining: float = 0.0
    fee: float = 0.0
    average_price: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_error: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        self.remaining = self.amount


class SlippageModel:
    """Estimates slippage from orderbook depth for a given notional size."""

    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels

    def estimate(self, orderbook: Dict, side: str, amount: float, mid_price: float) -> float:
        """Return estimated slippage percentage (0.0 to 1.0)."""
        if not orderbook or amount <= 0 or mid_price <= 0:
            return 0.0

        book_side = orderbook.get('asks' if side in ('buy', 'long') else 'bids', [])
        if not book_side:
            return 0.01  # conservative fallback

        cumulative = 0.0
        weighted_price_sum = 0.0
        for level in book_side[:self.depth_levels]:
            price = float(level[0])
            qty = float(level[1])
            take = min(qty, amount - cumulative)
            weighted_price_sum += price * take
            cumulative += take
            if cumulative >= amount:
                break

        if cumulative == 0:
            return 0.01

        avg_fill_price = weighted_price_sum / cumulative
        if side in ('buy', 'long'):
            slippage = (avg_fill_price - mid_price) / mid_price
        else:
            slippage = (mid_price - avg_fill_price) / mid_price

        return max(0.0, slippage)


class RetryHandler:
    """Exponential backoff retry for transient exchange errors."""

    def __init__(self, max_retries: int = 3, base_ms: int = 500, max_ms: int = 5000):
        self.max_retries = max_retries
        self.base_ms = base_ms
        self.max_ms = max_ms

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        msg = str(error).lower()
        transient = ('network' in msg or 'timeout' in msg or '502' in msg or
                     '503' in msg or '504' in msg or 'rate limit' in msg or
                     'temporarily' in msg or 'unavailable' in msg)
        return transient

    def backoff_ms(self, attempt: int) -> int:
        return min(self.base_ms * (2 ** attempt), self.max_ms)


class ExchangeConnector:
    def __init__(self, config):
        self.config = config
        self.exchange = None
        self.exec_cfg = config.execution
        self.retry = RetryHandler(
            max_retries=self.exec_cfg.max_order_retries,
            base_ms=self.exec_cfg.retry_backoff_base_ms,
            max_ms=self.exec_cfg.retry_backoff_max_ms,
        )
        self.slippage_model = SlippageModel(depth_levels=self.exec_cfg.slippage_depth_levels)
        self.orders: Dict[str, OrderRecord] = {}

    def connect(self):
        ec = self.config.exchange
        exchange_class = getattr(ccxt, ec.name, ccxt.binance)
        params = {
            'apiKey': ec.api_key,
            'secret': ec.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        }
        if ec.testnet:
            params['sandbox'] = True
        self.exchange = exchange_class(params)
        logger.info(f"Connected to {ec.name} (sandbox={ec.testnet})")
        return self

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_orderbook(self, symbol: str, limit: int = 10) -> Dict:
        return self.exchange.fetch_order_book(symbol, limit)

    def fetch_balance(self) -> Dict:
        return self.exchange.fetch_balance()

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            positions = self.exchange.fetch_positions([symbol])
            self.position_check_error = None
            for pos in positions:
                if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                    return pos
        except Exception as e:
            self.position_check_error = str(e)
            logger.error(f"Error fetching position for {symbol}: {e}")
        return None

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            logger.warning(f"Leverage error on {symbol}: {e}")

    def fetch_active_symbols(self, limit: int = 50, min_volume: float = 5_000_000.0,
                             exclude: Optional[List[str]] = None) -> List[str]:
        """Fetch all active USDT pairs ranked by 24h trading volume."""
        exclude_set = set(exclude or [
            'USDC/USDT', 'TUSD/USDT', 'BUSD/USDT', 'FDUSD/USDT', 'USDP/USDT',
            'EUR/USDT', 'USD1/USDT', 'DAI/USDT', 'USTC/USDT', 'RLUSD/USDT'
        ])
        try:
            tickers = self.exchange.fetch_tickers()
            valid = []
            for sym, t in tickers.items():
                if not sym.endswith('/USDT') or sym in exclude_set:
                    continue
                vol = t.get('quoteVolume', 0) or 0
                if vol >= min_volume:
                    valid.append((sym, vol))
            valid.sort(key=lambda x: x[1], reverse=True)
            symbols = [s[0] for s in valid[:limit]]
            return symbols if symbols else ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        except Exception as e:
            logger.warning(f"Error fetching active symbols: {e}. Falling back to default list.")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    def _build_params(self, params: dict, post_only: bool = False,
                      reduce_only: bool = False) -> dict:
        p = dict(params or {})
        if post_only:
            p['postOnly'] = True
        if reduce_only:
            p['reduceOnly'] = True
        return p

    def create_order_with_retry(self, symbol: str, order_type: str, side: str,
                                amount: float, price: Optional[float] = None,
                                params: dict = None,
                                post_only: bool = False,
                                reduce_only: bool = False,
                                check_slippage: bool = True) -> Optional[OrderRecord]:
        """Create an order with retry, slippage check, and state tracking."""
        order_id = f"ord_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        order = OrderRecord(
            id=order_id, symbol=symbol, side=side, type=order_type,
            amount=amount, price=price,
        )
        self.orders[order_id] = order

        # Slippage check for market orders
        if check_slippage and self.exec_cfg.slippage_model_enabled and order_type == 'market':
            try:
                orderbook = self.fetch_orderbook(symbol, limit=self.exec_cfg.slippage_depth_levels)
                ticker = self.exchange.fetch_ticker(symbol)
                mid_price = (ticker.get('bid', 0) + ticker.get('ask', 0)) / 2 or ticker.get('last', 0)
                if mid_price > 0:
                    est_slippage = self.slippage_model.estimate(orderbook, side, amount, mid_price)
                    if est_slippage > self.exec_cfg.max_slippage_pct:
                        order.state = OrderState.REJECTED
                        order.last_error = f"Estimated slippage {est_slippage*100:.2f}% exceeds max {self.exec_cfg.max_slippage_pct*100:.2f}%"
                        logger.warning(f"[{symbol}] Order rejected: {order.last_error}")
                        return order
            except Exception as e:
                logger.warning(f"[{symbol}] Slippage check failed: {e}")

        # Retry loop
        attempt = 0
        while True:
            try:
                order.state = OrderState.OPEN
                p = self._build_params(params, post_only=post_only, reduce_only=reduce_only)
                result = self.exchange.create_order(symbol, order_type, side, amount, price, p)
                if result.get('id'):
                    order.id = str(result['id'])
                # Query the exchange once: create_order responses are often
                # acknowledgements with missing or null fill fields.
                try:
                    fetched = self.exchange.fetch_order(order.id, symbol)
                except Exception as fetch_error:
                    order.state = OrderState.UNKNOWN
                    order.last_error = f"Order accepted but fill status is unknown: {fetch_error}"
                    logger.error("[%s] %s", symbol, order.last_error)
                    return order
                data = fetched or result or {}
                raw_filled = data.get('filled')
                raw_remaining = data.get('remaining')
                raw_average = data.get('average')
                order.filled = float(raw_filled) if raw_filled is not None else 0.0
                order.remaining = float(raw_remaining) if raw_remaining is not None else max(0.0, amount - order.filled)
                order.average_price = (float(raw_average) if raw_average is not None
                                       else (float(price) if price is not None else None))
                fee = data.get('fee') or {}
                raw_fee = fee.get('cost') if isinstance(fee, dict) else None
                order.fee = float(raw_fee) if raw_fee is not None else 0.0
                status = str(data.get('status') or '').lower()
                if order.filled > 0 and order.remaining <= 0 or status in ('closed', 'filled'):
                    order.state = OrderState.FILLED
                    if order.filled <= 0:
                        order.filled = amount
                elif order.filled > 0:
                    order.state = OrderState.PARTIAL
                elif status in ('rejected', 'canceled', 'cancelled', 'expired'):
                    order.state = {'rejected': OrderState.REJECTED, 'canceled': OrderState.CANCELLED,
                                   'cancelled': OrderState.CANCELLED, 'expired': OrderState.EXPIRED}[status]
                else:
                    order.state = OrderState.UNKNOWN
                    order.last_error = f"Order {order.id} has no confirmed fill (status={status or 'missing'})"
                    logger.error("[%s] %s", symbol, order.last_error)
                    return order
                logger.info(
                    f"[{symbol}] Order filled: {side} {order.filled}/{amount} @ "
                    f"{order.average_price} (fee: {order.fee})"
                )
                return order
            except Exception as e:
                order.last_error = str(e)
                flat_markers = ("position is zero", "position is flat", "no position", "position not found",
                                "position does not exist", "position size is zero")
                if reduce_only and any(marker in str(e).lower() for marker in flat_markers):
                    order.state = OrderState.REJECTED
                    logger.warning("[%s] Close rejected because position is already flat: %s", symbol, e)
                    return order
                ambiguous_markers = ("network", "timeout", "timed out", "connection", "unavailable", "gateway")
                if any(marker in str(e).lower() for marker in ambiguous_markers):
                    order.state = OrderState.UNKNOWN
                    order.last_error = f"Order submission may have reached exchange; status unknown: {e}"
                    logger.error("[%s] %s", symbol, order.last_error)
                    return order
                if self.retry.should_retry(e, attempt):
                    attempt += 1
                    order.retry_count = attempt
                    delay_ms = self.retry.backoff_ms(attempt - 1)
                    logger.warning(
                        f"[{symbol}] Order attempt {attempt} failed: {e}. "
                        f"Retrying in {delay_ms}ms..."
                    )
                    time.sleep(delay_ms / 1000.0)
                    continue
                order.state = OrderState.REJECTED
                logger.error(f"[{symbol}] Order rejected after {attempt+1} attempts: {e}")
                return order

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict = None,
                           reduce_only: bool = False):
        return self.create_order_with_retry(
            symbol, 'market', side, amount, None, params,
            reduce_only=reduce_only, check_slippage=True,
        )

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                           params: dict = None, post_only: bool = False):
        return self.create_order_with_retry(
            symbol, 'limit', side, amount, price, params,
            post_only=self.exec_cfg.post_only or post_only,
            check_slippage=False,
        )

    def create_best_bid_ask_limit_order(self, symbol: str, side: str, amount: float,
                                        post_only: bool = True, params: dict = None):
        """Place a post-only limit order at the best bid (buy) or best ask (sell)."""
        try:
            orderbook = self.fetch_orderbook(symbol, limit=1)
        except Exception as e:
            logger.warning(f'[{symbol}] Failed to fetch orderbook for best-bid/ask: {e}')
            return None
        if not orderbook:
            return None
        if side in ('buy', 'long'):
            book = orderbook.get('bids') or []
        else:
            book = orderbook.get('asks') or []
        if not book:
            return None
        best_price = float(book[0][0])
        return self.create_limit_order(symbol, side, amount, best_price,
                                       params=params, post_only=post_only)

    def check_maker_fill(self, symbol: str, order_id: str, limit_price: float, side: str):
        """Poll a live maker limit order; return updated OrderRecord (or None)."""
        order = self.orders.get(order_id)
        if order is None:
            return None
        try:
            raw = self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.warning(f'[{symbol}] fetch_order {order_id} failed: {e}')
            return order
        filled = float(raw.get('filled', 0.0))
        status = raw.get('status', '').lower()
        if filled > 0:
            order.filled = filled
            order.remaining = order.amount - filled
            order.average_price = float(raw.get('average', limit_price))
            order.state = OrderState.FILLED if filled >= order.amount else OrderState.PARTIAL
        elif status in ('closed', 'filled'):
            order.state = OrderState.FILLED
            order.filled = order.amount
        elif status in ('canceled', 'cancelled'):
            order.state = OrderState.CANCELLED
        elif status in ('rejected',):
            order.state = OrderState.REJECTED
        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float):
        params = {'stopPrice': stop_price}
        if self.exec_cfg.reduce_only_stops:
            params['reduceOnly'] = True
        return self.create_order_with_retry(
            symbol, 'stop_market', side, amount, None, params,
            reduce_only=self.exec_cfg.reduce_only_stops,
            check_slippage=True,
        )

    def cancel_all_orders(self, symbol: str):
        self.exchange.cancel_all_orders(symbol)

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        return self.orders.get(order_id)

    def get_orders(self, symbol: Optional[str] = None) -> List[OrderRecord]:
        if symbol:
            return [o for o in self.orders.values() if o.symbol == symbol]
        return list(self.orders.values())


class PaperExchange:
    def __init__(self, config):
        self.config = config
        self.exec_cfg = config.execution
        self._balance = config.scalping.initial_capital
        self._balance_source = None
        self.positions = {}
        self.trade_history = []
        self.order_id = 0
        self.orders = {}
        self.latency_ms = self.exec_cfg.paper_latency_ms
        self.base_slippage_pct = self.exec_cfg.paper_slippage_pct
        self.partial_fill_chance = self.exec_cfg.paper_partial_fill_chance
        self.reject_chance = self.exec_cfg.paper_reject_chance
        self._public_exchanges = []
        self._current_public = None
        self._init_public_exchanges()

    @property
    def balance(self):
        if self._balance_source is not None:
            return float(self._balance_source.current_balance)
        return self._balance

    @balance.setter
    def balance(self, value):
        if self._balance_source is not None:
            self._balance_source.current_balance = float(value)
        else:
            self._balance = float(value)

    def bind_balance_source(self, source):
        """Use the risk manager as the balance authority for bot-managed paper trades."""
        self._balance_source = source

    def _init_public_exchanges(self):
        name = str(self.config.exchange.name).lower()
        ex_class = getattr(ccxt, name, None)
        if ex_class is None:
            raise ValueError(f"Unknown candle exchange: {name}")
        try:
            self._current_public = ex_class({'enableRateLimit': True})
        except Exception as e:
            logger.error("failed to initialize candle exchange %s: %s", name, e)
            raise
        self._public_exchanges = [self._current_public]
        logger.info("Pinned candle source to %s", getattr(self._current_public, 'id', name))

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 200):
        ex = self._current_public
        if ex is None:
            logger.error("no pinned candle exchange for %s %s; skipping", symbol, timeframe)
            return None
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error("pinned candle exchange %s failed for %s %s: %s; skipping bar",
                         getattr(ex, 'id', self.config.exchange.name), symbol, timeframe, e)
            return None


    def _simulate_latency(self):
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    def _simulate_slippage(self, price: float, side: str) -> float:
        slippage = random.uniform(0, self.base_slippage_pct)
        if side in ('buy', 'long'):
            return price * (1 + slippage)
        return price * (1 - slippage)

    def _simulate_outcome(self) -> Optional[str]:
        """Returns None if order proceeds, or a reject reason."""
        if random.random() < self.reject_chance:
            return "Simulated order rejection"
        return None

    def fetch_balance(self) -> Dict:
        self._simulate_latency()
        return {'USDT': {'free': self.balance, 'total': self.balance}}

    def get_position(self, symbol: str) -> Optional[Dict]:
        self._simulate_latency()
        return self.positions.get(symbol)

    def set_leverage(self, symbol: str, leverage: int):
        pass

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict = None) -> Dict:
        self._simulate_latency()
        reject_reason = self._simulate_outcome()
        if reject_reason:
            self.order_id += 1
            order = {
                'id': f"paper_{self.order_id}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': 0.0,
                'status': 'rejected',
                'timestamp': datetime.now().isoformat(),
                'reject_reason': reject_reason,
            }
            self.trade_history.append(order)
            return order

        # Partial fill simulation
        fill_ratio = 1.0
        if random.random() < self.partial_fill_chance:
            fill_ratio = random.uniform(0.5, 0.95)

        filled_amount = amount * fill_ratio
        price = self._simulate_slippage(params.get('price', 0.0) if params else 0.0, side)

        self.order_id += 1
        order = {
            'id': f"paper_{self.order_id}",
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'filled': filled_amount,
            'price': price,
            'status': 'partial' if fill_ratio < 1.0 else 'closed',
            'timestamp': datetime.now().isoformat(),
        }
        self.orders[order['id']] = order

        if side in ['buy', 'long']:
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'long',
                'contracts': filled_amount,
                'entryPrice': price,
                'timestamp': datetime.now().isoformat(),
            }
        else:
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'short',
                'contracts': filled_amount,
                'entryPrice': price,
                'timestamp': datetime.now().isoformat(),
            }
        self.trade_history.append(order)
        return order

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                           params: dict = None, post_only: bool = False) -> Dict:
        self._simulate_latency()
        reject_reason = self._simulate_outcome()
        if reject_reason:
            self.order_id += 1
            return {
                'id': f"paper_{self.order_id}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'rejected',
                'timestamp': datetime.now().isoformat(),
                'reject_reason': reject_reason,
            }
        # Limit orders fill at limit price (simplified)
        self.order_id += 1
        order = {
            'id': f"paper_{self.order_id}",
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'timestamp': datetime.now().isoformat(),
            'post_only': post_only,
        }
        self.orders[order['id']] = order
        self.trade_history.append(order)
        return order

    def create_best_bid_ask_limit_order(self, symbol: str, side: str, amount: float,
                                          post_only: bool = True, params: dict = None) -> Dict:
        """Place a post-only limit order at the best bid (buy) or best ask (sell)."""
        self._simulate_latency()
        reject_reason = self._simulate_outcome()
        if reject_reason:
            self.order_id += 1
            return {
                'id': f"paper_{self.order_id}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': 0.0,
                'status': 'rejected',
                'timestamp': datetime.now().isoformat(),
                'reject_reason': reject_reason,
            }
        best_price = self._get_best_price(symbol, side)
        if best_price is None:
            self.order_id += 1
            return {
                'id': f"paper_{self.order_id}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': 0.0,
                'status': 'rejected',
                'timestamp': datetime.now().isoformat(),
                'reject_reason': 'No orderbook price available',
            }
        return self.create_limit_order(symbol, side, amount, best_price,
                                       params=params, post_only=post_only)

    def _get_best_price(self, symbol: str, side: str):
        """Best bid for buys, best ask for sells from the public orderbook."""
        if self._current_public is None:
            return None
        try:
            orderbook = self._current_public.fetch_order_book(symbol, limit=1)
        except Exception as e:
            logger.warning('orderbook fetch failed for %s: %s' % (symbol, e))
            return None
        if not orderbook:
            return None
        if side in ('buy', 'long'):
            book = orderbook.get('bids') or []
        else:
            book = orderbook.get('asks') or []
        if not book:
            return None
        try:
            return float(book[0][0])
        except (TypeError, IndexError, ValueError):
            return None

    def check_limit_fill(self, symbol: str, order_id: str, limit_price: float, side: str) -> Dict:
        """Adverse-selection fill model: fill only if price trades through the limit on the next bar."""
        self._simulate_latency()
        order = self.orders.get(order_id)
        if not order or order.get('status') != 'open':
            return order
        df = self.fetch_ohlcv(symbol, self.config.scalping.timeframe, 2)
        if df is None or len(df) < 2:
            return order
        cur = df.iloc[-1]
        high = float(cur['high'])
        low = float(cur['low'])
        # Price traded through the limit price during the bar.
        traded_through = (low <= limit_price <= high)
        if traded_through:
            order['status'] = 'filled'
            order['filled'] = order['amount']
            order['price'] = limit_price
            if side in ('buy', 'long'):
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': 'long',
                    'contracts': order['amount'],
                    'entryPrice': limit_price,
                    'timestamp': datetime.now().isoformat(),
                }
            else:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': 'short',
                    'contracts': order['amount'],
                    'entryPrice': limit_price,
                    'timestamp': datetime.now().isoformat(),
                }
            self.trade_history.append(order)
        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict:
        self._simulate_latency()
        reject_reason = self._simulate_outcome()
        if reject_reason:
            self.order_id += 1
            return {
                'id': f"paper_{self.order_id}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'stopPrice': stop_price,
                'status': 'rejected',
                'timestamp': datetime.now().isoformat(),
                'reject_reason': reject_reason,
            }
        self.order_id += 1
        order = {
            'id': f"paper_{self.order_id}",
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'stopPrice': stop_price,
            'status': 'open',
            'timestamp': datetime.now().isoformat(),
            'reduceOnly': True,
        }
        self.orders[order['id']] = order
        self.trade_history.append(order)
        return order

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        self._simulate_latency()
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        entry = pos['entryPrice']
        amt = pos['contracts']
        side = pos['side']
        # Apply slippage on exit
        exit_price = self._simulate_slippage(exit_price, 'sell' if side == 'long' else 'buy')
        pnl = (exit_price - entry) * amt if side == 'long' else (entry - exit_price) * amt
        if self._balance_source is None:
            self.balance += pnl
        return pnl

    def cancel_all_orders(self, symbol: str):
        self._simulate_latency()
        to_cancel = [k for k,v in self.orders.items() if v.get('symbol') == symbol]
        self.orders = {k: v for k, v in self.orders.items() if v.get('symbol') != symbol}
        return len(to_cancel)

    def get_order(self, order_id: str) -> Optional[Dict]:
        return self.orders.get(order_id)

    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol:
            return [o for o in self.orders.values() if o.get('symbol') == symbol]
        return list(self.orders.values())

    def fetch_active_symbols(self, limit: int = 50, min_volume: float = 5_000_000.0,
                             exclude: Optional[List[str]] = None) -> List[str]:
        """Fetch all active USDT pairs ranked by 24h trading volume for Paper trading."""
        exclude_set = set(exclude or [
            'USDC/USDT', 'TUSD/USDT', 'BUSD/USDT', 'FDUSD/USDT', 'USDP/USDT',
            'EUR/USDT', 'USD1/USDT', 'DAI/USDT', 'USTC/USDT', 'RLUSD/USDT',
            'AEUR/USDT', 'EURI/USDT'
        ])
        tries = []
        if self._current_public is not None:
            tries.append(self._current_public)
        for ex in self._public_exchanges:
            if ex not in tries:
                tries.append(ex)

        for ex in tries:
            try:
                tickers = ex.fetch_tickers()
                valid = []
                for sym, t in tickers.items():
                    if not sym.endswith('/USDT') or sym in exclude_set:
                        continue
                    vol = t.get('quoteVolume', 0) or 0
                    if vol >= min_volume:
                        valid.append((sym, vol))
                valid.sort(key=lambda x: x[1], reverse=True)
                symbols = [s[0] for s in valid[:limit]]
                if symbols:
                    logger.info(f"Dynamically discovered {len(symbols)} active coins from {getattr(ex, 'id', 'exchange')}")
                    return symbols
            except Exception as e:
                logger.warning(f"Error fetching active symbols from {getattr(ex, 'id', 'exchange')}: {e}")
                continue

        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

