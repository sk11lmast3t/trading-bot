"""
Ultra Scalping Bot - Backtester Engine
Complete backtesting with historical data, metrics & equity curve
"""
import ccxt
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict
from core.indicators import Indicators
from core.strategies import get_strategy, Signal
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class DataDownloader:
    """Download historical OHLCV data via CCXT"""
    @staticmethod
    def download(exchange_name, symbol, timeframe, start_date, end_date):
        exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        all_ohlcv = []
        logger.info(f"Downloading {symbol} {timeframe} {start_date} to {end_date}")
        while since < end_ts:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        logger.info(f"Downloaded {len(df)} candles")
        return df

    @staticmethod
    def save(df, path):
        df.to_csv(path)

    @staticmethod
    def load(path):
        return pd.read_csv(path, index_col="timestamp", parse_dates=True)


VALID_ENTRY_MODES = ("market", "maker", "maker_fallback")


class BacktestTrade:
    """One simulated trade. All cost fields are in quote currency (USDT).

    entry_price / exit_price are REFERENCE prices (before slippage). Costs are
    recorded separately, so:  pnl (net) = gross_pnl - costs
    where costs = entry_fee + entry_slip + exit_fee + exit_slip + funding.
    """
    def __init__(self, symbol, side, entry, size, time, sl, tp, strat):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry
        self.size = size
        self.entry_time = time
        self.sl = sl
        self.tp = tp
        self.strategy = strat
        self.exit_price = 0
        self.exit_time = None
        self.pnl = 0                 # NET pnl after every cost
        self.exit_reason = ""
        self.entry_bar = None
        self.entry_how = "market"    # how the entry really filled: market / maker
        self.gross_pnl = 0.0
        self.entry_fee = 0.0
        self.entry_slip = 0.0
        self.exit_fee = 0.0
        self.exit_slip = 0.0
        self.funding = 0.0

    @property
    def costs(self):
        return (self.entry_fee + self.entry_slip + self.exit_fee
                + self.exit_slip + self.funding)

    @property
    def notional(self):
        return self.entry_price * self.size


class Backtester:
    """Backtesting engine. Single cost source: every trade carries its own
    gross pnl and itemised costs, and the balance, the risk manager and any
    report all read those same numbers.

    entry_mode (config.execution.entry_mode):
      market          enter at the signal-bar close; taker fee + slippage.
      maker           post-only limit at the signal-bar close. Fills ONLY if the
                      next bar trades THROUGH the limit by backtest.maker_penetration
                      (low < limit*(1-pen) for a long, high > limit*(1+pen) for a
                      short); otherwise the signal is skipped. Maker fee, no
                      slippage on entry. The penetration is a queue-position
                      assumption, not a measured fill rate.
      maker_fallback  as maker, but an unfilled order is replaced by a market
                      entry at the next bar's open (taker fee + slippage).
    Exits are always market orders (taker fee + slippage), as in main.py.
    Exit order: SL first, then TP (pessimistic when both are touched in one
    bar), then time exit (config.risk.max_hold_time_minutes) at the bar close.
    NOT modelled (the live bot has them): trailing stop and breakeven.
    """
    def __init__(self, config, max_hold_minutes=None):
        self.config = config
        self.strategy = get_strategy(config)
        self.risk_manager = RiskManager(config)
        self.trades = []
        self.open_trades = {}
        self.equity_curve = []
        self.balance = config.backtest.initial_balance
        self.peak = self.balance
        self.stats = defaultdict(int)
        if max_hold_minutes is None:
            max_hold_minutes = getattr(config.risk, "max_hold_time_minutes", 0) or 0
        self.max_hold_minutes = float(max_hold_minutes)

    @staticmethod
    def _bar_minutes(index):
        if len(index) < 2:
            return 5.0
        step = pd.Series(index).diff().dropna().median()
        return max(float(step.total_seconds()) / 60.0, 1e-9)

    def run(self, df, symbol="BTC/USDT", trade_from=None):
        """Run over df. If trade_from is given, bars before it only warm up the
        indicators: no new trades are opened before that timestamp."""
        mode = getattr(self.config.execution, "entry_mode", "market")
        if mode not in VALID_ENTRY_MODES:
            raise ValueError(f"entry_mode must be one of {VALID_ENTRY_MODES}, got {mode!r}")
        logger.info(f"Backtest: {len(df)} candles, ${self.balance}, entry_mode={mode}")
        df = Indicators.calculate_all(df, self.config)
        bar_min = self._bar_minutes(df.index)
        # bars of history handed to the strategy; multi-timeframe strategies ask for more (default 300)
        lookback = int(getattr(self.strategy, "lookback_bars", 300))
        max_hold_bars = (int(round(self.max_hold_minutes / bar_min))
                         if self.max_hold_minutes > 0 else None)
        for i in range(60, len(df)):
            cur = df.iloc[i]
            price, high, low = float(cur["close"]), float(cur["high"]), float(cur["low"])
            ts = df.index[i]
            # Exits: SL first, then TP, then time exit
            for sym in list(self.open_trades.keys()):
                t = self.open_trades[sym]
                if t.side == "long":
                    sl_hit, tp_hit = low <= t.sl, high >= t.tp
                else:
                    sl_hit, tp_hit = high >= t.sl, low <= t.tp
                if sl_hit:
                    self._close(sym, t.sl, ts, "SL")
                elif tp_hit:
                    self._close(sym, t.tp, ts, "TP")
                elif max_hold_bars is not None and i - t.entry_bar >= max_hold_bars:
                    self._close(sym, price, ts, "TIME")
            # New signals (no `continue` here: the equity curve gets one point per bar)
            if symbol not in self.open_trades and (trade_from is None or ts >= trade_from):
                sim_time = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                can, why = self.risk_manager.can_trade(sim_time)
                if not can:
                    self.stats["risk_blocked_bars"] += 1
                    self.stats["risk_blocked:" + str(why).split(" (")[0]] += 1
                else:
                    window = df.iloc[max(0, i - lookback + 1):i + 1].copy()
                    setup = self.strategy.analyze(window)
                    if setup.signal in (Signal.LONG, Signal.SHORT):
                        self.stats["signals"] += 1
                        self._try_enter(df, i, symbol, setup, mode, cur)
            self.equity_curve.append({"ts": ts, "bal": self._equity(price)})
        # Close remaining
        fp = float(df.iloc[-1]["close"])
        for sym in list(self.open_trades.keys()):
            self._close(sym, fp, df.index[-1], "END")
        return self._metrics()

    def _try_enter(self, df, i, symbol, setup, mode, cur):
        bt = self.config.backtest
        side = "long" if setup.signal == Signal.LONG else "short"
        ref, fill_bar, maker = float(cur["close"]), i, False
        if mode in ("maker", "maker_fallback"):
            self.stats["maker_attempts"] += 1
            if i + 1 >= len(df):
                return
            nxt = df.iloc[i + 1]
            pen = float(getattr(bt, "maker_penetration", 0.0))
            if side == "long":
                through = float(nxt["low"]) < ref * (1.0 - pen)
            else:
                through = float(nxt["high"]) > ref * (1.0 + pen)
            if through:
                fill_bar, maker = i + 1, True
                self.stats["maker_fills"] += 1
            elif mode == "maker":
                self.stats["maker_unfilled"] += 1
                return
            else:
                ref, fill_bar = float(nxt["open"]), i + 1
                self.stats["maker_fallbacks"] += 1
        atr = cur.get("atr") if "atr" in cur else None
        try:
            atr = float(atr) if atr is not None else None
        except (TypeError, ValueError):
            atr = None
        size = self.risk_manager.calculate_position_size(
            self.balance, ref, setup.stop_loss, atr=atr)
        if size <= 0:
            self.stats["size_zero"] += 1
            return
        if (size * ref) / self.config.scalping.leverage > self.balance * 0.95:
            self.stats["insufficient_balance"] += 1
            return
        t = BacktestTrade(symbol, side, ref, size, df.index[fill_bar],
                          setup.stop_loss, setup.take_profit, setup.strategy_name)
        t.entry_bar = fill_bar
        t.entry_how = "maker" if maker else "market"
        t.entry_fee = ref * size * (bt.maker_fee if maker else bt.taker_fee)
        t.entry_slip = 0.0 if maker else ref * size * bt.slippage
        self.balance -= t.entry_fee + t.entry_slip   # entry costs hit the balance now
        self.open_trades[symbol] = t
        self.stats["entries_" + t.entry_how] += 1

    def _close(self, symbol, exit_price, exit_time, reason):
        bt = self.config.backtest
        t = self.open_trades.pop(symbol)
        t.exit_price = exit_price
        t.exit_time = exit_time
        t.exit_reason = reason
        if t.side == "long":
            t.gross_pnl = (exit_price - t.entry_price) * t.size
        else:
            t.gross_pnl = (t.entry_price - exit_price) * t.size
        t.exit_fee = exit_price * t.size * bt.taker_fee
        t.exit_slip = exit_price * t.size * bt.slippage
        hold_min = max((exit_time - t.entry_time).total_seconds() / 60.0, 0.0)
        # funding_rate is per 8h; positive means longs pay shorts
        t.funding = (t.notional * getattr(bt, "funding_rate", 0.0) * hold_min / 480.0
                     * (1 if t.side == "long" else -1))
        t.pnl = t.gross_pnl - t.costs
        # entry costs were already taken from the balance at entry
        self.balance += t.gross_pnl - t.exit_fee - t.exit_slip - t.funding
        sim_exit = exit_time.to_pydatetime() if hasattr(exit_time, "to_pydatetime") else exit_time
        self.risk_manager.record_trade(t.pnl, symbol, sim_exit)
        if self.balance > self.peak:
            self.peak = self.balance
        self.trades.append(t)

    def _equity(self, price):
        eq = self.balance
        for t in self.open_trades.values():
            if t.side == "long":
                eq += (price - t.entry_price) * t.size
            else:
                eq += (t.entry_price - price) * t.size
        return eq

    def _metrics(self):
        if not self.trades:
            return {"error": "No trades"}
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        pnl = sum(t.pnl for t in self.trades)
        gp = sum(t.pnl for t in wins) if wins else 0
        gl = abs(sum(t.pnl for t in losses)) if losses else 0.01
        eq = pd.Series([e["bal"] for e in self.equity_curve])
        dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
        durs = [(t.exit_time - t.entry_time).total_seconds()/60
                for t in self.trades if t.exit_time and t.entry_time]
        init = self.config.backtest.initial_balance
        eq_df = pd.DataFrame({"bal": [e["bal"] for e in self.equity_curve]}, index=pd.to_datetime([e["ts"] for e in self.equity_curve]))
        daily = eq_df["bal"].resample("1D").last().pct_change().dropna()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        return {
            "total_trades": len(self.trades), "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins)/len(self.trades)*100,
            "total_pnl": pnl, "roi_pct": (self.balance-init)/init*100,
            "gross_pnl": sum(t.gross_pnl for t in self.trades),
            "total_costs": sum(t.costs for t in self.trades),
            "profit_factor": gp/gl, "max_drawdown_pct": dd,
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0,
            "best_trade": max(t.pnl for t in self.trades),
            "worst_trade": min(t.pnl for t in self.trades),
            "avg_duration_min": np.mean(durs) if durs else 0,
            "sharpe_ratio": sharpe,
            "final_balance": self.balance,
        }

    def print_results(self, m):
        print("\n" + "="*55)
        print("         BACKTEST RESULTS")
        print("="*55)
        for k, v in m.items():
            print(f"  {k:.<35} {v:>12.4f}" if isinstance(v, float) else f"  {k:.<35} {v:>12}")
        print("="*55)
