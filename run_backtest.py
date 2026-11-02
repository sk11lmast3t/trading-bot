import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from config.settings import BotConfig
from core.backtester import Backtester, DataDownloader

STRATEGIES = ["grid_scalper", "ema_rsi_atr", "momentum_scalper", "combined"]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
START = "2026-09-29"
END = "2026-10-29"


def run_one(strat_name, symbol, df, commission, slippage, entry_mode="market"):
    cfg = BotConfig()
    cfg.strategy.active_strategy = strat_name
    cfg.execution.entry_mode = entry_mode
    cfg.backtest.initial_balance = 1000.0
    cfg.backtest.taker_fee = commission
    cfg.backtest.maker_fee = commission * 0.66
    cfg.backtest.slippage = slippage
    bt = Backtester(cfg)
    m = bt.run(df.copy(), symbol)
    return m, bt.trades


def main():
    print("Downloading data...")
    data = {}
    for sym in SYMBOLS:
        try:
            df = DataDownloader.download("binance", sym, "5m", START, END)
            data[sym] = df
            print(f"  {sym}: {len(df)} candles")
        except Exception as e:
            print(f"  {sym} FAILED: {e}")

    rows = []
    for fee_label, (comm, slip) in {"REAL_FEES": (0.001, 0.0005), "ZERO_FEES": (0.0, 0.0)}.items():
        for strat in STRATEGIES:
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                m, trades = run_one(strat, sym, data[sym], comm, slip, entry_mode="market")
                if "error" in m:
                    print(f"{fee_label} {strat} {sym}: NO TRADES")
                    continue
                rows.append((fee_label, strat, sym, m["total_trades"], m["win_rate"], m["total_pnl"], m["sharpe_ratio"], m["max_drawdown_pct"]))
                print(f"{fee_label} {strat:18s} {sym:9s} trades={m['total_trades']:4d} win={m['win_rate']:6.2f}% pnl={m['total_pnl']:9.2f} sharpe={m['sharpe_ratio']:7.2f} dd={m['max_drawdown_pct']:7.2f}%")

    print("\n" + "="*100)
    print("AGGREGATE BY STRATEGY + FEE (summed across symbols)")
    print("="*100)
    from collections import defaultdict
    agg = defaultdict(lambda: {"trades":0,"pnl":0.0})
    for fee_label, strat, sym, tr, wr, pnl, sh, dd in rows:
        a = agg[(fee_label, strat)]
        a["trades"] += tr
        a["pnl"] += pnl
    for (fee_label, strat), a in sorted(agg.items()):
        print(f"{fee_label:10s} {strat:18s} trades={a['trades']:5d} total_pnl={a['pnl']:10.2f}")

    print("\n" + "="*100)
    print("BEST BY WIN RATE (per symbol, real fees)")
    print("="*100)
    best = {}
    for fee_label, strat, sym, tr, wr, pnl, sh, dd in rows:
        if fee_label != "REAL_FEES":
            continue
        if sym not in best or wr > best[sym][4]:
            best[sym] = (fee_label, strat, sym, tr, wr, pnl, sh, dd)
    for sym, r in best.items():
        print(f"{sym:9s} -> {r[1]:18s} win={r[4]:6.2f}% pnl={r[5]:9.2f} trades={r[3]}")


if __name__ == "__main__":
    main()
