"""
Generate the baseline walk-forward + backtest report.

Runs every strategy x fee scenario through the walk-forward engine
(train 3mo / test 1mo windows) and writes reports/baseline_<date>.md.

No strategy parameters are tuned; config defaults are used as-is.

Fee scenarios (exit is always a taker fill in the backtester):
  all-maker  : entry_mode="maker" (post-only limit, modeled next-bar fill),
               maker fee on entry, taker fee on exit, light slippage
  mixed      : entry_mode="market", blended entry fee (maker+taker)/2,
               taker fee on exit, medium slippage
  all-taker  : entry_mode="market", taker fee on entry and exit,
               heavy slippage
"""
import os
import sys
import datetime as dt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import BotConfig
from core.backtester import Backtester, DataDownloader
from core.walk_forward import WalkForwardEngine

STRATEGIES = ["grid_scalper", "ema_rsi_atr", "momentum_scalper", "kairi_disparity", "combined"]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
START = "2026-06-01"
END = "2026-11-02"
TIMEFRAME = "5m"
EXCHANGE = "binance"

INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.01            # config default risk_per_trade
RISK_DOLLAR = INITIAL_BALANCE * RISK_PER_TRADE   # 1R = $10.00

FEE_SCENARIOS = [
    {
        "label": "all_maker",
        "name": "All-maker (modeled fills)",
        "entry_mode": "maker",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "slippage": 0.0001,
    },
    {
        "label": "mixed",
        "name": "Mixed",
        "entry_mode": "market",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "slippage": 0.0003,
    },
    {
        "label": "all_taker",
        "name": "All-taker",
        "entry_mode": "market",
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "slippage": 0.0005,
    },
]

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def entry_fee_rate(scenario):
    """Effective entry fee rate per scenario."""
    if scenario["label"] == "all_maker":
        return scenario["maker_fee"]
    if scenario["label"] == "mixed":
        return (scenario["maker_fee"] + scenario["taker_fee"]) / 2.0
    return scenario["taker_fee"]


def make_cfg(strat_name, scenario):
    cfg = BotConfig()
    cfg.strategy.active_strategy = strat_name
    cfg.execution.entry_mode = scenario["entry_mode"]
    cfg.backtest.initial_balance = INITIAL_BALANCE
    cfg.backtest.maker_fee = scenario["maker_fee"]
    cfg.backtest.taker_fee = scenario["taker_fee"]
    cfg.backtest.slippage = scenario["slippage"]
    return cfg


def run_backtest(test_df, symbol, strat_name, scenario):
    cfg = make_cfg(strat_name, scenario)
    bt = Backtester(cfg)
    m = bt.run(test_df.copy(), symbol)
    trades = []
    for t in bt.trades:
        trades.append({
            "symbol": t.symbol,
            "side": t.side,
            "entry": float(t.entry_price),
            "exit": float(t.exit_price),
            "size": float(t.size),
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "exit_reason": t.exit_reason,
            "strategy": t.strategy,
        })
    return m, trades


def trade_pnl(trade, entry_fee_rate, exit_fee_rate, slip_rate):
    if trade["side"] == "long":
        gross = (trade["exit"] - trade["entry"]) * trade["size"]
    else:
        gross = (trade["entry"] - trade["exit"]) * trade["size"]
    entry_cost = trade["entry"] * trade["size"] * entry_fee_rate
    exit_cost = trade["exit"] * trade["size"] * exit_fee_rate
    slip = trade["exit"] * trade["size"] * slip_rate
    net = gross - entry_cost - exit_cost - slip
    return gross, net, entry_cost + exit_cost + slip


def summarize(trades, scenario, days):
    if not trades:
        return None
    efr = entry_fee_rate(scenario)
    processed = [trade_pnl(t, efr, scenario["taker_fee"], scenario["slippage"]) for t in trades]
    nets = [p[1] for p in processed]
    grosss = [p[0] for p in processed]
    costs = sum(p[2] for p in processed)
    n = len(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    win_rate = len(wins) / n * 100
    avg_win = float(np.mean([g for g, n_ in zip(grosss, nets) if n_ > 0])) if wins else 0.0
    avg_loss = float(np.mean([n_ for n_ in nets if n_ <= 0])) if losses else 0.0
    gross_win_sum = sum(g for g, n_ in zip(grosss, nets) if n_ > 0)
    net_expectancy = float(np.mean(nets))
    gross_expectancy = (len(wins) / n) * avg_win
    total_pnl = float(sum(nets))
    bal = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0.0
    for net in nets:
        bal += net
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    c2t = (costs / gross_win_sum) if gross_win_sum > 0 else float("inf")
    trades_per_day = n / days if days else 0.0

    def pct(x):
        return x / INITIAL_BALANCE * 100

    def bps(x):
        return pct(x) * 100

    def r(x):
        return x / RISK_DOLLAR

    exit_reasons = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t, net in zip(trades, nets):
        b = exit_reasons[t["exit_reason"]]
        b["n"] += 1
        b["pnl"] += net

    return {
        "trades": n,
        "win_rate": win_rate,
        "net_exp_$": net_expectancy,
        "net_exp_%": pct(net_expectancy),
        "net_exp_bps": bps(net_expectancy),
        "net_exp_R": r(net_expectancy),
        "gross_exp_$": gross_expectancy,
        "gross_exp_%": pct(gross_expectancy),
        "gross_exp_bps": bps(gross_expectancy),
        "gross_exp_R": r(gross_expectancy),
        "avg_win_$": avg_win,
        "avg_win_%": pct(avg_win),
        "avg_win_bps": bps(avg_win),
        "avg_win_R": r(avg_win),
        "avg_loss_$": avg_loss,
        "avg_loss_%": pct(avg_loss),
        "avg_loss_bps": bps(avg_loss),
        "avg_loss_R": r(avg_loss),
        "total_pnl_$": total_pnl,
        "roi_%": pct(total_pnl),
        "profit_factor": pf,
        "max_dd_%": max_dd,
        "cost_to_target": c2t,
        "cost_to_target_%": c2t * 100 if c2t != float("inf") else float("inf"),
        "trades_per_day": trades_per_day,
        "total_costs_$": costs,
        "gross_profit_$": gross_win_sum,
        "exit_reasons": dict(exit_reasons),
    }


def fmt_exit_breakdown(er):
    lines = []
    total = sum(v["n"] for v in er.values())
    for reason in ("SL", "TP", "END"):
        v = er.get(reason, {"n": 0, "pnl": 0.0})
        share = v["n"] / total * 100 if total else 0
        avg = v["pnl"] / v["n"] if v["n"] else 0
        lines.append(f"| {reason} | {v['n']} | {share:5.1f}% | ${avg:9.2f} |")
    if not lines:
        lines.append("| - | - | - | - |")
    return "\n".join(lines)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Downloading data...")
    data = {}
    for sym in SYMBOLS:
        try:
            df = DataDownloader.download(EXCHANGE, sym, TIMEFRAME, START, END)
            data[sym] = df.sort_index()
            print(f"  {sym}: {len(df)} candles")
        except Exception as e:
            print(f"  {sym} FAILED: {e}")

    if not data:
        print("No data downloaded; aborting.")
        return 1

    wf = WalkForwardEngine(train_months=3, test_months=1, step_months=1)
    windows = wf.windows(next(iter(data.values())))
    window_count = len(windows)
    test_days = sum(
        (w["test"].index.max() - w["test_start"]).days for w in windows
    ) or 1

    results = {}  # (strat, scenario_label) -> pooled metrics + per symbol
    for strat in STRATEGIES:
        for scenario in FEE_SCENARIOS:
            pooled_trades = []
            per_symbol = {}
            for sym in SYMBOLS:
                df = data[sym]

                def evaluate(train_df, test_df, _sym=sym, _strat=strat, _scen=scenario):
                    _, trades = run_backtest(test_df, _sym, _strat, _scen)
                    return trades

                out = wf.run(df, evaluate)
                sym_trades = []
                for res in out["oos"]:
                    sym_trades.extend(res or [])
                per_symbol[sym] = sym_trades
                pooled_trades.extend(sym_trades)
                print(f"  {strat:16s} {scenario['label']:10s} {sym:9s} trades={len(sym_trades)} windows={out['windows']}")

            m = summarize(pooled_trades, scenario, test_days)
            if m is None:
                m = {"trades": 0, "win_rate": 0.0, "net_exp_$": 0.0, "net_exp_%": 0.0,
                     "net_exp_bps": 0.0, "net_exp_R": 0.0, "gross_exp_$": 0.0,
                     "gross_exp_%": 0.0, "gross_exp_bps": 0.0, "gross_exp_R": 0.0,
                     "avg_win_$": 0.0, "avg_win_%": 0.0, "avg_win_bps": 0.0, "avg_win_R": 0.0,
                     "avg_loss_$": 0.0, "avg_loss_%": 0.0, "avg_loss_bps": 0.0, "avg_loss_R": 0.0,
                     "total_pnl_$": 0.0, "roi_%": 0.0, "profit_factor": 0.0,
                     "max_dd_%": 0.0, "cost_to_target": float("inf"),
                     "cost_to_target_%": 0.0, "trades_per_day": 0.0,
                     "total_costs_$": 0.0, "gross_profit_$": 0.0, "exit_reasons": {}}
            m["per_symbol"] = {}
            for sym, sym_trades in per_symbol.items():
                m["per_symbol"][sym] = summarize(sym_trades, scenario, test_days)
            results[(strat, scenario["label"])] = m

    report_date = dt.date.today().isoformat()
    path = os.path.join(REPORT_DIR, f"baseline_{report_date}.md")
    lines = []
    lines.append(f"# Baseline Report — Unified Crypto Scalper ({report_date})")
    lines.append("")
    lines.append("Generated by `scripts/generate_baseline_report.py`. No strategy parameters were tuned;")
    lines.append("config defaults (`config/settings.py`) are used as-is.")
    lines.append("")
    lines.append("## Data & setup")
    lines.append(f"- Data range: {START} to {END} ({TIMEFRAME} candles, {EXCHANGE})")
    lines.append(f"- Symbol count: {len(SYMBOLS)} — {', '.join(SYMBOLS)}")
    lines.append(f"- Walk-forward windows: {window_count} (train 3mo / test 1mo, step 1mo), out-of-sample test period only")
    lines.append(f"- Initial balance: ${INITIAL_BALANCE:,.2f}, leverage 1x")
    lines.append(f"- Risk per trade: {RISK_PER_TRADE*100:.2f}% of capital = ${RISK_DOLLAR:,.2f} (1R)")
    lines.append("")
    lines.append("## Fee scenarios")
    lines.append("| Scenario | Entry | Entry fee | Exit fee | Slippage | Notes |")
    lines.append("|----------|-------|-----------|----------|----------|-------|")
    for s in FEE_SCENARIOS:
        efr = entry_fee_rate(s)
        lines.append(
            f"| {s['name']} | {s['entry_mode']} | {efr*100:5.3f}% | "
            f"{s['taker_fee']*100:5.3f}% | {s['slippage']*100:5.3f}% | "
            f"{'modeled next-bar fill' if s['entry_mode']=='maker' else 'blended maker/taker entry cost' if s['label']=='mixed' else 'aggressive'} |"
        )
    lines.append("")
    lines.append("Exit fees are taker fees in all scenarios (the backtester exits at market).")
    lines.append("Expectancy per trade: net = mean(net PnL/trade); gross = win rate × avg win.")
    lines.append("Percent/bps are vs initial balance ($1,000); R is vs $10.00 risk per trade.")
    lines.append("Cost-to-target = total transaction costs / gross profit from winning trades.")
    lines.append("")
    lines.append("## Summary by strategy × fee scenario (pooled across symbols & walk-forward windows)")
    lines.append("")
    lines.append("| Strategy | Scenario | Trades | Trades/day | Win % | Net exp ($ / % / bps / R) | Gross exp ($ / % / bps / R) | Avg win ($) | Avg loss ($) | Cost→target | Max DD % |")
    lines.append("|----------|----------|--------|------------|-------|---------------------------|-----------------------------|-------------|--------------|-------------|----------|")
    for strat in STRATEGIES:
        for scenario in FEE_SCENARIOS:
            m = results[(strat, scenario["label"])]
            net = m["net_exp_$"]
            gross = m["gross_exp_$"]
            lines.append(
                f"| {strat} | {scenario['name']} | {m['trades']} | {m['trades_per_day']:6.2f} | "
                f"{m['win_rate']:5.2f} | ${net:9.2f} / {m['net_exp_%']:6.3f}% / {m['net_exp_bps']:7.1f} / {m['net_exp_R']:7.3f} | "
                f"${gross:9.2f} / {m['gross_exp_%']:6.3f}% / {m['gross_exp_bps']:7.1f} / {m['gross_exp_R']:7.3f} | "
                f"${m['avg_win_$']:9.2f} | ${m['avg_loss_$']:9.2f} | "
                f"{m['cost_to_target'] if m['cost_to_target'] != float('inf') else 'n/a':>5} "
                f"({m['cost_to_target_%'] if m['cost_to_target_%'] != float('inf') else 0:5.1f}%) | {m['max_dd_%']:6.2f} |"
            )
    lines.append("")

    for scenario in FEE_SCENARIOS:
        lines.append(f"## {scenario['name']}")
        lines.append("")
        lines.append("### Exit-reason breakdown (pooled across symbols & windows)")
        lines.append("")
        lines.append("| Reason | Trades | Share | Avg PnL ($) |")
        lines.append("|--------|--------|-------|-------------|")
        any_trades = False
        for strat in STRATEGIES:
            m = results[(strat, scenario["label"])]
            er = m["exit_reasons"]
            total = sum(v["n"] for v in er.values())
            if total == 0:
                continue
            any_trades = True
            for reason in ("SL", "TP", "END"):
                v = er.get(reason, {"n": 0, "pnl": 0.0})
                if v["n"] == 0:
                    continue
                share = v["n"] / total * 100
                avg = v["pnl"] / v["n"]
                lines.append(f"| {strat} / {reason} | {v['n']} | {share:5.1f}% | ${avg:9.2f} |")
        if not any_trades:
            lines.append("| - | - | - | - |")
        lines.append("")
        lines.append("### Per-symbol detail")
        lines.append("")
        lines.append("| Symbol | Trades | Trades/day | Win % | Net exp/trade ($) | Net exp (R) | Avg win ($) | Avg loss ($) | Cost→target (%) | Max DD % |")
        lines.append("|--------|--------|------------|-------|-------------------|-------------|-------------|--------------|-----------------|----------|")
        for strat in STRATEGIES:
            for sym in SYMBOLS:
                m = results[(strat, scenario["label"])]["per_symbol"][sym]
                if m is None:
                    continue
                c2t = m["cost_to_target_%"] if m["cost_to_target_%"] != float("inf") else 0.0
                lines.append(
                    f"| {sym} | {m['trades']} | {m['trades_per_day']:6.2f} | {m['win_rate']:5.2f} | "
                    f"${m['net_exp_$']:9.2f} | {m['net_exp_R']:7.3f} | ${m['avg_win_$']:9.2f} | "
                    f"${m['avg_loss_$']:9.2f} | {c2t:6.2f}% | {m['max_dd_%']:6.2f} |"
                )
        lines.append("")

    lines.append("## Notes")
    lines.append("- All figures are out-of-sample walk-forward results (no training-window leakage into reported metrics).")
    lines.append("- Maker entries use modeled next-bar fills: unposted limit orders that do not trade through are skipped.")
    lines.append("- Expectancy in R uses $10.00 per trade (1% of $1,000 capital).")
    lines.append("- Drawdown is computed on the trade-close balance curve per run.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
