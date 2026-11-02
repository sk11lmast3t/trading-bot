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
            for pos in positions:
                if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                    return pos
        except Exception as e:
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
                order.state = OrderState.FILLED
                order.filled = float(result.get('filled', amount))
                order.remaining = float(result.get('remaining', 0.0))
                order.average_price = float(result.get('average', price or 0))
                order.fee = float(result.get('fee', {}).get('cost', 0.0))
                if result.get('id'):
                    order.id = str(result['id'])
                logger.info(
                    f"[{symbol}] Order filled: {side} {order.filled}/{amount} @ "
                    f"{order.average_price} (fee: {order.fee})"
                )
                return order
            except Exception as e:
                order.last_error = str(e)
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

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict = None):
        return self.create_order_with_retry(
            symbol, 'market', side, amount, None, params,
            check_slippage=True,
        )

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                           params: dict = None, post_only: bool = False):
        return self.create_order_with_retry(
            symbol, 'limit', side, amount, price, params,
            post_only=self.exec_cfg.post_only or post_only,
            check_slippage=False,
        )

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
        self.balance = config.scalping.initial_capital
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

    def _init_public_exchanges(self):
        candidates = []
        for name in ['binance', 'bybit', 'okx', 'kucoin', 'kraken', 'coinbase']:
            try:
                ex_class = getattr(ccxt, name)
                ex = ex_class({'enableRateLimit': True})
                try:
                    ex.fetch_ohlcv('BTC/USDT', '5m', limit=1)
                except Exception:
                    pass
                candidates.append(ex)
                if self._current_public is None:
                    self._current_public = ex
                logger.info('init %s' % getattr(ex,'id',name))
            except Exception as e:
                logger.warning('fail %s: %s' % (name,e))
        self._public_exchanges = candidates
        if self._current_public is None and candidates:
            self._current_public = candidates[0]
        if self._current_public is None:
            for name in ['binance', 'bybit', 'okx', 'kucoin', 'kraken', 'coinbase']:
                try:
                    ex_class = getattr(ccxt,name)
                    self._current_public = ex_class({'enableRateLimit':True})
                    self._public_exchanges = [self._current_public]
                    break
                except Exception:
                    continue
            if self._current_public is None:
                try:
                    self._current_public = ccxt.okx({'enableRateLimit':True})
                except Exception as e:
                    logger.error('no pub: %s' % e)

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 200):
        tries = []
        if self._current_public is not None:
            tries.append(self._current_public)
        for ex in self._public_exchanges:
            if ex not in tries:
                tries.append(ex)
        if not tries:
            self._init_public_exchanges()
            if self._current_public is not None:
                tries.append(self._current_public)
            for ex in self._public_exchanges:
                if ex not in tries:
                    tries.append(ex)
        last=None
        for ex in tries:
            try:
                ohlcv = ex.fetch_ohlcv(symbol,timeframe,limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
                df.set_index('timestamp',inplace=True)
                self._current_public = ex
                return df
            except Exception as e:
                last=e
                logger.warning('fail %s: %s' % (getattr(ex,'id','?'),e))
                continue
        logger.error('all fail %s %s: %s' % (symbol,timeframe,last))
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
        self.balance += pnl
        return pnl

    def cancel_all_orders(self, symbol: str):
        self._simulate_latency()
        self.orders = {k: v for k, v in self.orders.items() if k.get('symbol') != symbol}

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

