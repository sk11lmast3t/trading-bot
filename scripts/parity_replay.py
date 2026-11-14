"""Compare main.py's paper execution path with the bar-based backtester.

Replay timing follows candle opens: the clock is advanced to each bar's close,
so that bar is complete and may drive a decision. ReplayExchange appends a flat
forming candle as the sentinel for process_symbol's final-row drop. check_exits
uses OHLC on post-signal bars; the replay sentinel repeats the latest completed
close and carries no future range information.
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
        # process_symbol drops this sentinel so it can analyze the newest completed
        # bar; check_exits reads its close as the mark. It is flat to avoid lookahead.
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

    def close_position(self, symbol, exit_price, apply_slippage=True):
        # Replay prices already include the modelled fill; the flag exists for
        # parity with PaperExchange.close_position.
        return self.positions.pop(symbol, None)


class TradeRun(list):
    """List of closed paper trades plus the bot's decision trace."""
    decision_rows = ()


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
    bot.decision_rows = []
    bot._append_decision = lambda row: bot.decision_rows.append(dict(row))
    bot.persist_state = lambda: None
    old_save, old_record = main.state.save_position_at_entry, main.state.record_trade
    main.state.save_position_at_entry = lambda *a, **k: None
    main.state.record_trade = lambda *a, **k: None
    completed = []
    original_close = bot._close_position
    def capture_close(sym, pos, price, reason, **kwargs):
        before = dict(pos)
        entry_time = bot.active_decisions.get(sym, {}).get("entry_time")
        targets = bot.position_targets.get(sym, {})
        meta = dict(bot.paper_trade_meta.get(sym, {}))
        original_close(sym, pos, price, reason, **kwargs)
        log = bot.risk_manager.trade_log[-1]
        reference = float(kwargs.get("reference_price", price))
        exit_fee = reference * float(before["contracts"]) * cfg.backtest.taker_fee
        # Mirror main._close_position: a take-profit is a resting limit and pays
        # no exit slippage, while stops/time exits pay the configured rate.
        slip_rate = kwargs.get("exit_slippage_rate")
        if slip_rate is None:
            slip_rate = cfg.backtest.slippage
        exit_slippage = reference * float(before["contracts"]) * max(0.0, float(slip_rate))
        completed.append({"side": before["side"], "entry_time": entry_time,
            "signal_bar_open": entry_time - timedelta(minutes=5),
            "entry_price": float(before["entryPrice"]), "exit_price": float(price),
            "exit_time": clock(), "exit_reason": reason, "net": log["pnl"],
            "quantity": float(before["contracts"]), "sl": targets.get("sl"),
            "tp": targets.get("tp"),
            "entry_fee": float(meta.get("entry_fee", 0.0)),
            "exit_fee": exit_fee,
            "entry_slippage": float(meta.get("entry_slippage", 0.0)),
            "exit_slippage": exit_slippage,
            "costs": (float(meta.get("entry_fee", 0.0)) + exit_fee +
                float(meta.get("entry_slippage", 0.0)) + exit_slippage)})
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
    result = TradeRun(completed)
    result.decision_rows = tuple(bot.decision_rows)
    return result


def run_backtester(frame, symbol, cfg, start, end, hold):
    strategy = get_strategy(cfg)
    warm = max(300, int(getattr(strategy, "lookback_bars", 300)))
    i0, i1 = frame.index.searchsorted(start), frame.index.searchsorted(end)
    bt = Backtester(cfg, max_hold_minutes=hold)
    bt.decision_signals = {}
    original_analyze = bt.strategy.analyze
    def capture_signal(window):
        setup = original_analyze(window)
        ts = pd.Timestamp(window.index[-1])
        bt.decision_signals[ts] = setup
        return setup
    bt.strategy.analyze = capture_signal
    bt.run(frame.iloc[max(0, i0 - warm):i1].copy(), symbol, trade_from=start)
    bt.all_trades = list(bt.trades)
    return [t for t in bt.trades if t.exit_reason != "END"], bt


def compare(bot_trades, bt_trades, frame, bt_obj=None):
    def key_bot(t):
        bar_open = t.get("signal_bar_open", pd.Timestamp(t["entry_time"]) - pd.Timedelta(minutes=5))
        return (t["side"], pd.Timestamp(bar_open))
    def key_bt(t):
        return (t.side, pd.Timestamp(t.entry_time))
    bot_map = {key_bot(t): t for t in bot_trades}
    bt_map = {key_bt(t): t for t in bt_trades}
    common = sorted(bot_map.keys() & bt_map.keys(), key=lambda x: x[1])
    only_bot, only_bt = sorted(bot_map.keys() - bt_map.keys()), sorted(bt_map.keys() - bot_map.keys())
    # Histogram uses a one-to-one nearest same-side pairing, independent of match keys.
    remaining = set(range(len(bt_trades)))
    nearest_pairs = []
    for bi, b in enumerate(sorted(bot_trades, key=lambda x: pd.Timestamp(x["entry_time"]))):
        candidates = [j for j in remaining if bt_trades[j].side == b["side"]]
        if not candidates:
            continue
        bj = min(candidates, key=lambda j: abs((pd.Timestamp(b["entry_time"]) -
                    pd.Timestamp(bt_trades[j].entry_time)).total_seconds()))
        remaining.remove(bj)
        delta = (pd.Timestamp(b["entry_time"]) - pd.Timestamp(bt_trades[bj].entry_time)).total_seconds() / 60
        nearest_pairs.append((b, bt_trades[bj], delta))
    histogram = {}
    exceptions = []
    for b, t, delta in nearest_pairs:
        label = f"{delta:+g} min"
        histogram[label] = histogram.get(label, 0) + 1
        if abs(delta - 5.0) > 1e-9:
            exceptions.append((b, t, delta))

    mismatch_rows = []
    bot_decisions = {pd.Timestamp(d["ts"]): d for d in getattr(bot_trades, "decision_rows", ())}
    bt_decisions = getattr(bt_obj, "decision_signals", {})
    for k in only_bot:
        observed = bt_decisions.get(k[1])
        if observed is not None:
            other_saw = f"strategy signal={observed.signal.value}, entry={observed.entry_price:g}, SL={observed.stop_loss:g}, TP={observed.take_profit:g}"
        else:
            active = next((t for t in getattr(bt_obj, "all_trades", ())
                           if t.entry_time <= k[1] < (t.exit_time or k[1])), None)
            other_saw = f"strategy not evaluated; backtester position open from {active.entry_time} through {active.exit_time}" if active else "strategy evaluation not captured (could not identify a matching decision)"
        mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": "other: no normalized backtester trade",
            "evidence": f"signal bar {k[1]}; close/high/low={_ohlc_evidence(frame, k[1])}; {other_saw}"})
    for k in only_bt:
        observed = bot_decisions.get(k[1])
        bot_saw = (f"signal={observed.get('signal')}, status={observed.get('status')}, entry={observed.get('intended_entry')}, SL={observed.get('sl')}, TP={observed.get('tp')}"
                   if observed else "bot did not record a decision at this bar")
        mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": "other: no normalized bot trade",
            "evidence": f"signal bar {k[1]}; close/high/low={_ohlc_evidence(frame, k[1])}; {bot_saw}"})
    matches = []
    mismatch_rows.extend({"side": b["side"], "entry_time": str(b["entry_time"]),
        "cause": "entry timing offset", "evidence": f"bot entry={b['entry_time']}, backtester entry={t.entry_time}; delta={delta:+g} minutes (expected +5)"}
        for b, t, delta in exceptions)
    for k in common:
        b, t = bot_map[k], bt_map[k]
        pnl_bps = (b["net"] - t.pnl) / t.notional * 10000 if t.notional else 0
        reason_ok = _normalize_reason(b["exit_reason"]) == _normalize_reason(t.exit_reason)
        row = {"side": k[0], "entry_time": str(k[1]),
            "bot_net": b["net"], "bot_entry": b["entry_price"], "bot_size": b["quantity"],
            "bt_net": t.pnl, "bt_entry": t.entry_price, "bt_size": t.size,
            "entry_price_diff_bps": _bps_diff(b["entry_price"], t.entry_price),
            "sl_diff_bps": _bps_diff(b["sl"], t.sl),
            "tp_diff_bps": _bps_diff(b["tp"], t.tp),
            "size_diff_pct": (b["quantity"] - t.size) / t.size * 100 if t.size else 0,
            "exit_reason_agrees": reason_ok,
            "bot_exit_reason": _normalize_reason(b["exit_reason"]),
            "bt_exit_reason": _normalize_reason(t.exit_reason),
            # Bot timestamps the completed signal candle close; BT timestamps its open.
            # Remove that known +1 bar convention to isolate exit timing residual.
            "exit_time_diff_bars": (b["exit_time"] - t.exit_time).total_seconds() / 300 - 1,
            "exit_price_diff_bps": _bps_diff(b["exit_price"], t.exit_price),
            "entry_fee_diff": b["entry_fee"] - t.entry_fee,
            "exit_fee_diff": b["exit_fee"] - t.exit_fee,
            "entry_slippage_diff": b["entry_slippage"] - t.entry_slip,
            "exit_slippage_diff": b["exit_slippage"] - t.exit_slip,
            "slippage_diff": (b["entry_slippage"] + b["exit_slippage"] - t.entry_slip - t.exit_slip),
            "net_pnl_diff_bps": pnl_bps}
        matches.append(row)
        causes = []
        bot_reason, bt_reason = row["bot_exit_reason"], row["bt_exit_reason"]
        at = pd.Timestamp(t.exit_time)
        bar = frame.loc[at] if at in frame.index else None
        intrabar_hit = bool(bar is not None and (
            (t.side == "long" and ((bt_reason == "SL" and bar.low <= t.sl) or (bt_reason == "TP" and bar.high >= t.tp))) or
            (t.side == "short" and ((bt_reason == "SL" and bar.high >= t.sl) or (bt_reason == "TP" and bar.low <= t.tp)))))
        if not reason_ok:
            if intrabar_hit and bot_reason != bt_reason:
                causes.append("bot close vs backtester high/low")
            elif "TIME" in (bt_reason, bot_reason):
                causes.append("time-exit timing")
            else:
                causes.append("other exit-reason mismatch")
        if abs(row["exit_price_diff_bps"]) > 1e-9:
            if intrabar_hit:
                if "bot close vs backtester high/low" not in causes:
                    causes.append("bot close vs backtester high/low")
            else:
                causes.append("time-exit timing" if "TIME" in (bt_reason, bot_reason) else "other exit-price mismatch")
        if abs(row["exit_time_diff_bars"]) > 1e-9 and "TIME" in (bt_reason, bot_reason):
            causes.append("time-exit timing")
        if abs(row["size_diff_pct"]) > 1e-9:
            causes.append("sizing")
        rate_diffs = (
            _rate_diff(b["entry_fee"], b["entry_price"] * b["quantity"],
                       t.entry_fee, t.entry_price * t.size),
            _rate_diff(b["exit_fee"], b["exit_price"] * b["quantity"],
                       t.exit_fee, t.exit_price * t.size),
            _rate_diff(b["entry_slippage"], b["entry_price"] * b["quantity"],
                       t.entry_slip, t.entry_price * t.size),
            _rate_diff(b["exit_slippage"], b["exit_price"] * b["quantity"],
                       t.exit_slip, t.exit_price * t.size),
        )
        if any(abs(diff) > 1e-10 for diff in rate_diffs):
            # A take-profit is a resting limit in the paper bot (no exit
            # slippage) but a market fill in the backtester (pays slippage).
            # Label that deliberate model difference instead of calling it a
            # formula bug. Any other cost-rate difference stays generic.
            tp_limit_only = (
                bot_reason == bt_reason == "TP"
                and all(abs(diff) <= 1e-10 for diff in rate_diffs[:3])
                and abs(rate_diffs[3]) > 1e-10
            )
            causes.append("TP limit vs market slippage model" if tp_limit_only
                          else "slippage/fee formula")
        if abs(pnl_bps) > 1e-9 and not causes:
            causes.append("other pnl mismatch")
        causes = list(dict.fromkeys(causes))
        if causes:
            mismatch_rows.append({"side": k[0], "entry_time": str(k[1]), "cause": "; ".join(causes), "evidence": str(row)})
    return common, only_bot, only_bt, matches, mismatch_rows, histogram, exceptions


def _bps_diff(bot_value, bt_value):
    return (float(bot_value) - float(bt_value)) / abs(float(bt_value)) * 10000 if bt_value else 0.0


def _rate_diff(bot_cost, bot_notional, bt_cost, bt_notional):
    bot_rate = float(bot_cost) / float(bot_notional) if bot_notional else 0.0
    bt_rate = float(bt_cost) / float(bt_notional) if bt_notional else 0.0
    return bot_rate - bt_rate


def _normalize_reason(reason):
    value = str(reason).upper()
    if "STOP LOSS" in value: return "SL"
    if "TAKE PROFIT" in value: return "TP"
    if "TIME" in value: return "TIME"
    return value


def _ohlc_evidence(frame, ts):
    ts = pd.Timestamp(ts)
    if ts not in frame.index:
        return "bar missing from CSV"
    bar = frame.loc[ts]
    return f"O={bar.open:g},H={bar.high:g},L={bar.low:g},C={bar.close:g}"


def metric_summary(rows, field, tol=1e-9):
    values = pd.Series([float(row[field]) for row in rows if row.get(field) is not None], dtype=float)
    if values.empty:
        return {"count": 0, "mean": "n/a", "median": "n/a", "p95": "n/a", "max": "n/a", "nonzero": 0}
    return {"count": len(values), "mean": values.mean(), "median": values.median(),
            "p95": values.abs().quantile(.95), "max": values.abs().max(),
            "nonzero": int((values.abs() > tol).sum())}


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
    matched, only_bot, only_bt, details, mismatches, histogram, exceptions = compare(bot, bt_trades, frame, bt)
    os.makedirs(a.out, exist_ok=True)
    date = f"{start:%Y%m%d}_{end:%Y%m%d}"
    stem = f"parity_compare_{a.symbol.replace('/', '_')}_{date}"
    sides = lambda rows, side: sum(1 for t in rows if (t["side"] if isinstance(t, dict) else t.side) == side)
    causes = {}
    for m in mismatches:
        for cause in m["cause"].split("; "):
            causes[cause] = causes.get(cause, 0) + 1
    for cause in ("bot close vs backtester high/low", "time-exit timing",
                  "slippage/fee formula", "sizing", "other exit-reason mismatch",
                  "other exit-price mismatch", "other pnl mismatch",
                  "other: no normalized backtester trade", "other: no normalized bot trade",
                  "entry timing offset"):
        causes.setdefault(cause, 0)
    with open(os.path.join(a.out, stem + "_mismatches.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["side", "entry_time", "cause", "evidence"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(mismatches)
    metric_fields = [
        ("Entry price diff (bot - backtester)", "entry_price_diff_bps", "bps"),
        ("SL diff (bot - backtester)", "sl_diff_bps", "bps"), ("TP diff (bot - backtester)", "tp_diff_bps", "bps"),
        ("Position size diff (bot - backtester)", "size_diff_pct", "%"),
        ("Residual exit time diff (bot - backtester; after -1 bar)", "exit_time_diff_bars", "bars"),
        ("Exit price diff (bot - backtester)", "exit_price_diff_bps", "bps"),
        ("Entry fee diff (bot - backtester)", "entry_fee_diff", "quote"),
        ("Exit fee diff (bot - backtester)", "exit_fee_diff", "quote"),
        ("Entry slippage diff (bot - backtester)", "entry_slippage_diff", "quote"),
        ("Exit slippage diff (bot - backtester)", "exit_slippage_diff", "quote"),
        ("Slippage diff (bot - backtester)", "slippage_diff", "quote"),
        ("Net PnL diff (bot - backtester)", "net_pnl_diff_bps", "bps of BT notional"),
    ]
    def fmt(value):
        return f"{value:+.8f}" if isinstance(value, (int, float)) else str(value)
    reason_labels = sorted({_normalize_reason(x["exit_reason"]) for x in bot} |
                          {_normalize_reason(t.exit_reason) for t in bt_trades})
    reason_matrix = {(br, tr): 0 for br in reason_labels for tr in reason_labels}
    for d in details:
        reason_matrix[(d["bot_exit_reason"], d["bt_exit_reason"])] += 1
    histogram_rows = [f"| {delta} | {count} |" for delta, count in sorted(histogram.items())]
    lines = [f"# Paper/backtester comparison: {a.symbol}", "",
        f"Window: {start} through {end} (end exclusive); strategy `{a.strategy}`; scenario `{a.scenario}`.",
        "", "Replay clock advances to each five-minute candle close. Only completed CSV candles are used. A flat forming row repeats the latest completed close as the sentinel that `process_symbol` drops before analysis and as the close mark `check_exits` reads. It contains no future OHLC information.", "",
        "## Nearest same-side entry offset histogram", "", "| Bot entry time minus Backtester entry time | Pairs |", "|---|---:|"] + histogram_rows
    if not histogram_rows:
        lines.append("| none | 0 |")
    lines += ["", "Expected offset is exactly +5 minutes. Matching uses `(side, bot_entry_time - 5 minutes)` against `(side, backtester_entry_time)`.", "", "### Nearest pairs not exactly +5 minutes", ""]
    if exceptions:
        lines.extend(f"- {b['side']} bot={b['entry_time']}, backtester={t.entry_time}, offset={delta:+g} minutes; not normalized into a match." for b,t,delta in exceptions)
    else:
        lines.append("None.")
    bot_net_bps = [t["net"] / (t["entry_price"] * t["quantity"]) * 10000 for t in bot if t["entry_price"] and t["quantity"]]
    bt_net_bps = [t.pnl / t.notional * 10000 for t in bt_trades if t.notional]
    matched_bot = [d["bot_net"] / (d["bot_entry"] * d["bot_size"]) * 10000 for d in details if d["bot_entry"] and d["bot_size"]]
    matched_bt = [d["bt_net"] / (d["bt_entry"] * d["bt_size"]) * 10000 for d in details if d["bt_entry"] and d["bt_size"]]
    lines += ["", "All difference columns use bot minus Backtester. Exit-time residual subtracts the known +1 bar entry-stamp convention.", "", "## Trade totals", "", "| Path | Trades | All trades net bps/trade | Matched-only net bps/trade | Total net PnL |", "|---|---:|---:|---:|---:|",
        f"| Bot | {len(bot)} | {sum(bot_net_bps)/len(bot_net_bps) if bot_net_bps else 0:+.5f} | {sum(matched_bot)/len(matched_bot) if matched_bot else 0:+.5f} | {sum(t['net'] for t in bot):+.8f} |",
        f"| Backtester | {len(bt_trades)} | {sum(bt_net_bps)/len(bt_net_bps) if bt_net_bps else 0:+.5f} | {sum(matched_bt)/len(matched_bt) if matched_bt else 0:+.5f} | {sum(t.pnl for t in bt_trades):+.8f} |", "",
        "## Counts by side", "", "| Path | Long | Short | Total |", "|---|---:|---:|---:|",
        f"| Bot | {sides(bot,'long')} | {sides(bot,'short')} | {len(bot)} |",
        f"| Backtester | {sides(bt_trades,'long')} | {sides(bt_trades,'short')} | {len(bt_trades)} |",
        f"| Matched | {sum(1 for k in matched if k[0]=='long')} | {sum(1 for k in matched if k[0]=='short')} | {len(matched)} |",
        f"| Bot unmatched | {sum(1 for k in only_bot if k[0]=='long')} | {sum(1 for k in only_bot if k[0]=='short')} | {len(only_bot)} |",
        f"| Backtester unmatched | {sum(1 for k in only_bt if k[0]=='long')} | {sum(1 for k in only_bt if k[0]=='short')} | {len(only_bt)} |", "",
        "## Matched trade difference summary", "",
        "Mean and median retain the sign. P95 and max are absolute difference magnitudes. `Nonzero` counts values above a small numeric tolerance.",
        "", "| Metric | Unit | Count | Mean | Median | P95 abs | Max abs | Nonzero |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, field, unit in metric_fields:
        stats = metric_summary(details, field)
        lines.append(f"| {label} | {unit} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p95'])} | {fmt(stats['max'])} | {stats['nonzero']} |")
    lines += ["", "## Exit reason agreement matrix", "", "Rows are bot exit reason; columns are Backtester exit reason.", "",
        "| Bot \\ Backtester | " + " | ".join(reason_labels) + " |",
        "|---|" + "|".join("---:" for _ in reason_labels) + "|"]
    for br in reason_labels:
        lines.append("| " + br + " | " + " | ".join(str(reason_matrix[(br,tr)]) for tr in reason_labels) + " |")
    lines += ["", "## Unmatched trade explanations", ""]
    if only_bot or only_bt:
        for row in mismatches:
            if row["cause"].startswith("other: no normalized"):
                lines.append(f"- {row['side']} {row['entry_time']}: {row['evidence']}")
    else:
        lines.append("None.")
    lines += ["", "## Mismatch causes", "", "| Cause | Count |", "|---|---:|"]
    lines.extend(f"| {cause} | {count} |" for cause, count in sorted(causes.items()))
    lines += ["", f"Mismatch detail CSV: `{stem}_mismatches.csv`.", "", "For each unmatched entry the report records the signal bar, its OHLC values, and the other path's captured decision or explains that the decision was not captured. Exit mismatch causes use the backtest exit bar's high/low and separate time exits, size, and cost differences."]
    with open(os.path.join(a.out, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    preview = ["Entry offset histogram: " + ", ".join(f"{k}={v}" for k,v in sorted(histogram.items())),
               f"Parity comparison {a.symbol}: bot={len(bot)} backtester={len(bt_trades)} matched={len(matched)} unmatched_bot={len(only_bot)} unmatched_backtester={len(only_bt)}"]
    preview.extend(f"Non +5m pair: {b['side']} bot={b['entry_time']} backtester={t.entry_time} offset={delta:+g}m" for b,t,delta in exceptions)
    print("\n".join(preview).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main_cli()
