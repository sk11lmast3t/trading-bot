"""Compare main.py's paper execution path with the bar-based backtester.

Replay timing follows candle opens: the clock is advanced to each bar's close,
so that bar is complete and may drive a decision. ReplayExchange also appends a
flat forming candle whose OHLC equals the latest completed close. process_symbol
drops that final row before analysis; check_exits reads its close as the current
mark. Its high/low are deliberately not used by main.py's exit checks.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from config.settings import BotConfig
from core.backtester import Backtester
from core.indicators import Indicators
from core.risk_manager import RiskManager
from core.strategies import Signal, TradeSetup, get_strategy
import main


class ReplayClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class ReplayExchange:
    """Small PaperExchange-shaped adapter backed by a timestamped CSV."""
    def __init__(self, frame, clock, symbol, decision_bar=None):
        self.frame = frame.sort_index()
        self.clock = clock
        self.symbol = symbol
        self.decision_bar = decision_bar
        self.positions = {}
        self._seq = 0

    def fetch_ohlcv(self, symbol, timeframe="5m", limit=200):
        if symbol != self.symbol:
            raise ValueError(f"Replay is pinned to {self.symbol}, requested {symbol}")
        # A candle is complete only when its five-minute interval has ended.
        complete = self.frame[self.frame.index + pd.Timedelta(minutes=5) <= self.clock()]
        rows = complete.tail(max(0, limit - 2)).copy()
        # This forming row is consumed only by main.check_exits as a close mark;
        # process_symbol removes it before indicators/strategy analysis.
        if len(rows):
            mark = float(rows.iloc[-1]["close"])
            forming_ts = self.clock().replace(second=0, microsecond=0)
            forming_ts = forming_ts - timedelta(minutes=forming_ts.minute % 5)
            forming = pd.DataFrame([[mark, mark, mark, mark, 0.0]],
                index=[forming_ts], columns=["open", "high", "low", "close", "volume"])
            rows = pd.concat([rows, forming])
        return rows.tail(limit)

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def create_market_order(self, symbol, side, amount, params=None):
        if side in ("buy", "sell"):
            self._seq += 1
            price = float((params or {}).get("price", self.frame.loc[self.decision_bar, "close"]))
            self.positions[symbol] = {"entryPrice": price, "contracts": float(amount),
                                      "side": "long" if side == "buy" else "short",
                                      "id": str(self._seq)}
            return {"id": str(self._seq), "filled": float(amount), "price": price}

    def close_position(self, symbol, exit_price):
        return self.positions.pop(symbol, None)


def make_cfg(strategy, symbol, hold_minutes, capital, fees):
    c = BotConfig()
    c.mode = "paper"
    c.strategy.active_strategy = strategy
    c.scalping.symbols = [symbol]
    c.scalping.scan_all_coins = False
    c.scalping.initial_capital = capital
    c.scalping.cooldown_seconds = 0
    c.risk.max_hold_time_minutes = float(hold_minutes)
    c.risk.trailing_stop_enabled = False
    c.risk.breakeven_activation_pct = 0
    c.risk.trailing_stop_activation_pct = 0
    c.risk.max_drawdown_pct = 1.0
    c.risk.max_daily_loss_pct = 1.0
    c.risk.max_consecutive_losses = 10 ** 9
    c.risk.daily_trade_limit = 0
    c.risk.circuit_breaker_enabled = False
    c.risk.reduce_size_on_loss = False
    c.backtest.initial_balance = capital
    c.backtest.maker_fee, c.backtest.taker_fee, c.backtest.slippage = fees
    c.execution.entry_mode = "market"
    c.execution.paper_partial_fill_chance = 0
    c.execution.paper_reject_chance = 0
    return c


def run_bot(frame, symbol, cfg, start, end):
    clock = ReplayClock(start.to_pydatetime() - timedelta(minutes=5))
    bot = main.UnifiedCryptoBot.__new__(main.UnifiedCryptoBot)
    bot.config, bot.clock = cfg, clock
    bot.strategy = get_strategy(cfg)
    bot.risk_manager = RiskManager(cfg, clock=clock)
    bot.exchange = ReplayExchange(frame, clock, symbol)
    bot.position_targets, bot.position_entry_cycle = {}, {}
    bot.last_processed_bar, bot.active_decisions = {}, {}
    bot.paper_trade_meta, bot.cycle_count = {}, 0
    bot.entries_blocked = False
    bot.last_rejection_log = {}
    bot.decision_log_path = None
    bot.persist_state = lambda: None
    old_save, old_record = main.state.save_position_at_entry, main.state.record_trade
    main.state.save_position_at_entry = lambda *a, **k: None
    main.state.record_trade = lambda *a, **k: None
    completed = []
    original_close = bot._close_position
    def capture_close(sym, pos, price, reason):
        before = dict(pos)
        entry_time = bot.active_decisions.get(sym, {}).get("entry_time")
        targets = bot.position_targets.get(sym, {})
        meta = dict(bot.paper_trade_meta.get(sym, {}))
        original_close(sym, pos, price, reason)
        log = bot.risk_manager.trade_log[-1]
        completed.append({"side": before["side"], "entry_time": entry_time,
            "entry_price": float(before["entryPrice"]), "exit_price": float(price),
            "exit_time": clock(), "exit_reason": reason, "net": log["pnl"],
            "quantity": float(before["contracts"]), "sl": targets.get("sl"),
            "tp": targets.get("tp"),
            "costs": (float(meta.get("entry_fee", 0.0)) +
                float(meta.get("entry_slippage", 0.0)) +
                float(price) * float(before["contracts"]) *
                (cfg.backtest.taker_fee + cfg.backtest.slippage))})
    bot._close_position = capture_close
    try:
        for ts in frame.index[(frame.index >= start) & (frame.index < end)]:
            clock.now = ts.to_pydatetime() + timedelta(minutes=5)
            bot.exchange.decision_bar = ts
            bot.cycle_count += 1
            bot.check_exits(symbol)
            bot.process_symbol(symbol)
    finally:
        main.state.save_position_at_entry, main.state.record_trade = old_save, old_record
    return completed


def run_backtester(frame, symbol, cfg, start, end, hold):
    strategy = get_strategy(cfg)
    warm = max(300, int(getattr(strategy, "lookback_bars", 300)))
    i0, i1 = frame.index.searchsorted(start), frame.index.searchsorted(end)
    bt = Backtester(cfg, max_hold_minutes=hold)
    bt.run(frame.iloc[max(0, i0 - warm):i1].copy(), symbol, trade_from=start)
    return [t for t in bt.trades if t.exit_reason != "END"], bt


def compare(bot_trades, bt_trades, frame):
    def key(t):
        return (t["side"], pd.Timestamp(t["entry_time"]))
    bot_map = {key(t): t for t in bot_trades}
    bt_map = {key({"side": t.side, "entry_time": t.entry_time}): t for t in bt_trades}
    common = sorted(bot_map.keys() & bt_map.keys(), key=lambda x: x[1])
    only_bot, only_bt = sorted(bot_map.keys() - bt_map.keys()), sorted(bt_map.keys() - bot_map.keys())
    mismatch_rows = []
    for k in only_bot:
        near = min((t for t in bt_trades if t.side == k[0]),
                   key=lambda t: abs((pd.Timestamp(t.entry_time) - k[1]).total_seconds()), default=None)
        evidence = "No backtester trade with same side and entry timestamp"
        cause = "bot-only entry"
        if near and abs((pd.Timestamp(near.entry_time) - k[1]).total_seconds()) <= 300:
            cause = "entry timing: bot enters at signal-bar close"
            evidence += f"; nearest backtester entry={near.entry_time}, bot entry={k[1]} (<=5m offset)"
        mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": cause, "evidence": evidence})
    for k in only_bt:
        near = min((t for t in bot_trades if t["side"] == k[0]),
                   key=lambda t: abs((pd.Timestamp(t["entry_time"]) - k[1]).total_seconds()), default=None)
        evidence = "No bot trade with same side and entry timestamp"
        cause = "backtester-only entry"
        if near and abs((pd.Timestamp(near["entry_time"]) - k[1]).total_seconds()) <= 300:
            cause = "entry timing: backtester timestamps signal bar"
            evidence += f"; nearest bot entry={near['entry_time']}, backtester entry={k[1]} (<=5m offset)"
        mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": cause, "evidence": evidence})
    matches = []
    for k in common:
        b, t = bot_map[k], bt_map[k]
        pnl_bps = (b["net"] - t.pnl) / t.notional * 10000 if t.notional else 0
        reason_ok = (b["exit_reason"].lower().replace("take profit", "tp").replace("stop loss", "sl")
                     == t.exit_reason.lower())
        row = {"side": k[0], "entry_time": str(k[1]),
            "entry_diff": b["entry_price"] - t.entry_price,
            "sl_diff": (b["sl"] - t.sl) if b["sl"] is not None else "n/a",
            "tp_diff": (b["tp"] - t.tp) if b["tp"] is not None else "n/a",
            "exit_reason_agrees": reason_ok,
            "exit_time_diff_minutes": (b["exit_time"] - t.exit_time).total_seconds() / 60,
            "net_pnl_diff_bps": pnl_bps}
        matches.append(row)
        causes = []
        if not reason_ok:
            at = pd.Timestamp(t.exit_time)
            if at in frame.index:
                bar = frame.loc[at]
                if t.side == "long" and (bar.low <= t.sl or bar.high >= t.tp):
                    causes.append("bot close vs backtester high/low")
                elif t.side == "short" and (bar.high >= t.sl or bar.low <= t.tp):
                    causes.append("bot close vs backtester high/low")
        if abs(row["exit_time_diff_minutes"]) >= 5:
            causes.append("entry/exit timing")
        if abs(b["quantity"] - t.size) > max(1e-10, abs(t.size) * 1e-6):
            causes.append("position sizing")
        if abs(b["costs"] - t.costs) > max(1e-8, abs(t.costs) * 1e-6):
            causes.append("fees/slippage formula")
        if causes:
            mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": "; ".join(causes), "evidence": str(row)})
    return common, only_bot, only_bt, matches, mismatch_rows


def main_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--strategy", default="mtf_trend_pullback")
    p.add_argument("--hold-minutes", type=float, default=240)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--maker-fee", type=float, default=.0002)
    p.add_argument("--taker-fee", type=float, default=.0005)
    p.add_argument("--slippage", type=float, default=.0003)
    p.add_argument("--scenario", choices=["market"], default="market")
    p.add_argument("--capital", type=float, default=100)
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    frame = pd.read_csv(a.csv, index_col="timestamp", parse_dates=True).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    if end <= start:
        p.error("--end must be after --start")
    fees = (a.maker_fee, a.taker_fee, a.slippage)
    cfg = make_cfg(a.strategy, a.symbol, a.hold_minutes, a.capital, fees)
    bot = run_bot(frame, a.symbol, cfg, start, end)
    bt_trades, bt = run_backtester(frame, a.symbol, cfg, start, end, a.hold_minutes)
    matched, only_bot, only_bt, details, mismatches = compare(bot, bt_trades, frame)
    os.makedirs(a.out, exist_ok=True)
    date = f"{start:%Y%m%d}_{end:%Y%m%d}"
    stem = f"parity_{a.symbol.replace('/', '_')}_{date}"
    sides = lambda rows, side: sum(1 for t in rows if (t["side"] if isinstance(t, dict) else t.side) == side)
    causes = {}
    for m in mismatches:
        for cause in m["cause"].split("; "):
            causes[cause] = causes.get(cause, 0) + 1
    with open(os.path.join(a.out, stem + "_mismatches.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["side", "entry_time", "cause", "evidence"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(mismatches)
    lines = [f"# Paper vs backtester parity: {a.symbol}", "", f"Window: {start} through {end} (end exclusive); strategy `{a.strategy}`; scenario `{a.scenario}`.",
        "", "Replay rule: clock advances to candle close; only completed candles are available for strategy analysis. A flat forming row duplicates the last completed close solely because `check_exits` reads the last fetched row's close; `process_symbol` drops that row before analysis.", "",
        "## Trade counts", "", "| Engine | Long | Short | Total |", "|---|---:|---:|---:|",
        f"| Bot | {sides(bot,'long')} | {sides(bot,'short')} | {len(bot)} |",
        f"| Backtester | {sides(bt_trades,'long')} | {sides(bt_trades,'short')} | {len(bt_trades)} |",
        f"| Matched | {sum(1 for k in matched if k[0]=='long')} | {sum(1 for k in matched if k[0]=='short')} | {len(matched)} |",
        f"| Bot only | {sum(1 for k in only_bot if k[0]=='long')} | {sum(1 for k in only_bot if k[0]=='short')} | {len(only_bot)} |",
        f"| Backtester only | {sum(1 for k in only_bt if k[0]=='long')} | {sum(1 for k in only_bt if k[0]=='short')} | {len(only_bt)} |", "",
        f"Matched trades: {len(matched)}; only in bot: {len(only_bot)}; only in backtester: {len(only_bt)}.", "",
        "## Matched trade differences", "", "| Side | Entry time | Entry Δ | SL Δ | TP Δ | Exit reason agrees | Exit Δ min | Net PnL Δ bps |", "|---|---|---:|---:|---:|---|---:|---:|"]
    for d in details:
        lines.append(f"| {d['side']} | {d['entry_time']} | {d['entry_diff']:.8g} | {d['sl_diff']} | {d['tp_diff']} | {d['exit_reason_agrees']} | {d['exit_time_diff_minutes']:.2f} | {d['net_pnl_diff_bps']:.3f} |")
    lines += ["", "## Mismatch causes", "", "| Cause | Count |", "|---|---:|"]
    lines.extend(f"| {cause} | {count} |" for cause, count in sorted(causes.items()))
    lines += ["", f"Mismatch detail CSV: `{stem}_mismatches.csv`.", "", "Classification is evidence-based for exact entry-key mismatches and matched exits. Unmatched entries are classified as timing/entry disagreement; net differences with matched reference entries can reflect the fee/slippage formula or size and need the row evidence for attribution."]
    with open(os.path.join(a.out, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:21]))


if __name__ == "__main__":
    main_cli()
