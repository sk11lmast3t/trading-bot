"""Backtester cost accounting: one cost source, entry slippage, strict maker fills."""
import numpy as np
import pandas as pd
import pytest

from config.settings import BotConfig
from core.backtester import Backtester, BacktestTrade
from core.strategies import Signal, TradeSetup


def make_df(n=6000, seed=1, vol=0.0012):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[close[0], close[:-1]]
    r2 = np.random.default_rng(seed + 1)
    hi = np.maximum(open_, close) * (1 + np.abs(r2.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(r2.normal(0, vol / 2, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": r2.uniform(100, 1000, n)}, index=idx)


def cfg(entry_mode="market", maker=0.0002, taker=0.0005, slip=0.0003, strat="grid_scalper"):
    c = BotConfig()
    c.strategy.active_strategy = strat
    c.execution.entry_mode = entry_mode
    c.backtest.maker_fee, c.backtest.taker_fee, c.backtest.slippage = maker, taker, slip
    # measurement mode: no risk-limit halts
    c.risk.max_drawdown_pct = 1.0
    c.risk.max_daily_loss_pct = 1.0
    c.risk.max_consecutive_losses = 10 ** 9
    c.risk.daily_trade_limit = 0
    c.risk.circuit_breaker_enabled = False
    c.risk.reduce_size_on_loss = False
    return c


def run(c, df=None):
    bt = Backtester(c)
    bt.run((df if df is not None else make_df()).copy(), "X/USDT")
    return bt


@pytest.mark.parametrize("mode", ["market", "maker", "maker_fallback"])
def test_balance_equals_initial_plus_sum_of_net_pnl(mode):
    bt = run(cfg(entry_mode=mode))
    assert len(bt.trades) > 20
    assert bt.balance == pytest.approx(1000.0 + sum(t.pnl for t in bt.trades), abs=1e-6)
    # risk manager must see the same (net) numbers as the balance
    assert bt.risk_manager.current_balance == pytest.approx(bt.balance, abs=1e-6)


def test_net_is_gross_minus_itemised_costs():
    bt = run(cfg())
    for t in bt.trades:
        assert t.pnl == pytest.approx(t.gross_pnl - t.costs, abs=1e-9)
        assert t.costs >= 0


def test_entry_slippage_is_charged_on_market_entries():
    bt = run(cfg(maker=0, taker=0, slip=0.001))
    assert bt.trades
    for t in bt.trades:
        assert t.entry_slip == pytest.approx(t.entry_price * t.size * 0.001)
        assert t.exit_slip == pytest.approx(t.exit_price * t.size * 0.001)


def test_zero_costs_means_net_equals_gross():
    bt = run(cfg(maker=0, taker=0, slip=0))
    assert sum(t.pnl for t in bt.trades) == pytest.approx(sum(t.gross_pnl for t in bt.trades))


def test_maker_fills_are_strictly_through_and_filter_signals():
    df = make_df()
    mk, kt = run(cfg("maker"), df), run(cfg("market"), df)
    assert mk.stats["maker_attempts"] > 0
    assert mk.stats["maker_unfilled"] > 0                    # some orders must NOT fill
    assert mk.stats["maker_fills"] + mk.stats["maker_unfilled"] + mk.stats["maker_no_next_bar"] == mk.stats["maker_attempts"]
    assert all(t.entry_how == "maker" and t.entry_slip == 0 for t in mk.trades)


def test_maker_fill_rule_unit():
    """A long limit at the close does not fill when the next bar only touches it."""
    class OneShot:
        name = "one"
        def __init__(self): self.fired = False
        def analyze(self, window):
            if not self.fired:
                self.fired = True
                p = float(window.iloc[-1]["close"])
                return TradeSetup(Signal.LONG, p, p * 0.9, p * 1.1, 0.9, "one")
            return TradeSetup(Signal.HOLD)

    def build(next_low):
        df = make_df(n=200)
        df["open"] = df["close"] = df["high"] = df["low"] = 100.0
        df["volume"] = 500.0
        return df, next_low

    def go(next_low):
        df = make_df(n=200)
        for c in ("open", "close", "high", "low"):
            df[c] = 100.0
        df["volume"] = 500.0
        c = cfg("maker")
        bt = Backtester(c)
        bt.strategy = OneShot()
        # next bar after the signal bar (signal fires on the first evaluated bar, i=60)
        df.iloc[61, df.columns.get_loc("low")] = next_low
        bt.run(df, "X/USDT")
        return bt

    assert go(100.0).stats["maker_fills"] == 0        # touch only -> no fill
    assert go(99.995).stats["maker_fills"] == 0       # 0.5 bp through < 1 bp penetration -> no fill
    assert go(99.9).stats["maker_fills"] == 1         # 10 bp through -> fill


def test_maker_fallback_always_enters_when_maker_misses():
    df = make_df()
    fb, mk = run(cfg("maker_fallback"), df), run(cfg("maker"), df)
    assert len(fb.trades) >= len(mk.trades)
    assert fb.stats["maker_fallbacks"] > 0
    assert {t.entry_how for t in fb.trades} <= {"maker", "market"}


def test_time_exit_fires():
    c = cfg()
    bt = Backtester(c, max_hold_minutes=15)       # 3 bars on 5m data
    bt.run(make_df(), "X/USDT")
    reasons = {t.exit_reason for t in bt.trades}
    assert "TIME" in reasons
    for t in bt.trades:
        if t.exit_reason == "TIME":
            held = (t.exit_time - t.entry_time).total_seconds() / 60
            assert held == pytest.approx(15)


def test_equity_curve_has_a_point_for_every_bar_even_when_blocked():
    c = cfg()
    c.risk.daily_trade_limit = 1                    # blocks most bars
    df = make_df(n=3000)
    bt = Backtester(c)
    bt.run(df.copy(), "X/USDT")
    assert len(bt.equity_curve) == len(df) - 60
    assert bt.stats["risk_blocked_bars"] > 0


def test_trade_from_blocks_entries_before_cutoff():
    df = make_df(n=4000)
    cutoff = df.index[2500]
    bt = Backtester(cfg())
    bt.run(df.copy(), "X/USDT", trade_from=cutoff)
    assert bt.trades
    assert all(t.entry_time >= cutoff for t in bt.trades)


def test_invalid_entry_mode_rejected():
    c = cfg()
    c.execution.entry_mode = "typo"
    with pytest.raises(ValueError):
        Backtester(c).run(make_df(n=300), "X/USDT")


def test_funding_sign_long_pays_when_positive():
    c = cfg(maker=0, taker=0, slip=0)
    c.backtest.funding_rate = 0.0008               # 0.08% per 8h (exaggerated)
    bt = run(c)
    longs = [t for t in bt.trades if t.side == "long"]
    shorts = [t for t in bt.trades if t.side == "short"]
    assert longs and shorts
    assert all(t.funding >= 0 for t in longs)
    assert all(t.funding <= 0 for t in shorts)


class FinalBarOnly:
    """Fires one LONG signal, only on the very last candle."""
    name = "final"
    def __init__(self, last_ts): self.last_ts = last_ts
    def analyze(self, window):
        if window.index[-1] == self.last_ts:
            p = float(window.iloc[-1]["close"])
            return TradeSetup(Signal.LONG, p, p * 0.9, p * 1.1, 0.9, "final")
        return TradeSetup(Signal.HOLD)


@pytest.mark.parametrize("mode", ["maker", "maker_fallback"])
def test_maker_signal_on_final_candle_is_counted_separately(mode):
    df = make_df(n=200)
    for c in ("open", "close", "high", "low"):
        df[c] = 100.0
    df["volume"] = 500.0
    bt = Backtester(cfg(mode))
    bt.strategy = FinalBarOnly(df.index[-1])
    bt.run(df, "X/USDT")
    assert bt.stats["maker_attempts"] == 1
    assert bt.stats["maker_no_next_bar"] == 1
    assert bt.stats["maker_fills"] == 0
    assert bt.stats["maker_unfilled"] == 0
    assert bt.stats["maker_fallbacks"] == 0
    assert not bt.trades


@pytest.mark.parametrize("mode", ["maker", "maker_fallback"])
def test_every_maker_order_has_exactly_one_outcome(mode):
    for seed in (1, 2, 3):
        bt = run(cfg(mode), make_df(n=3000, seed=seed))
        s = bt.stats
        assert s["maker_attempts"] == (s["maker_fills"] + s["maker_unfilled"]
                                       + s["maker_fallbacks"] + s["maker_no_next_bar"])
