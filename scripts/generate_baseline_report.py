"""Generate reports/baseline_<date>.md: do any of the bot's strategies have a NET edge?

Prerequisite (on your machine, needs network):   python scripts/download_data.py
Then:                                            python scripts/generate_baseline_report.py --jobs 4

What it does
  * Reads data/*_5m.csv, cuts every symbol into consecutive calendar-month test windows.
    Nothing is fitted, so every window is out-of-sample by construction (--skip-months N
    drops the first N months if you later add a tuning step and want a real train period).
  * Runs each strategy x cost scenario through core.backtester.Backtester, which records
    gross pnl and every cost on every trade. This script never recomputes costs: it reads
    those fields, so report, balance and risk manager share ONE cost model.
  * Reports gross edge, cost and NET edge per trade in bps of notional and in R (R uses
    each trade's own stop distance), a bootstrap 95% CI, the share of profitable months,
    exit reasons, maker fill rates, signal diagnostics and a plain verdict per strategy.
Risk-limit halts are switched off for measurement (otherwise one 5% drawdown stops a test
window for good), so this measures the SIGNAL, not the risk layer.
"""
import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

STRATEGIES = ["grid_scalper", "ema_rsi_atr", "momentum_scalper", "kairi_disparity", "combined"]
SCENARIOS = [  # label, title, backtester entry_mode
    ("all_maker", "All-maker (strict fills, skipped if unfilled)", "maker"),
    ("mixed", "Mixed (maker first, taker fallback)", "maker_fallback"),
    ("all_taker", "All-taker (market entries, as main.py trades today)", "market"),
]
REALISTIC = "all_taker"
WARMUP_BARS = 300
INITIAL_BALANCE = 1000.0
TRADE_COLS = ["strategy", "scenario", "symbol", "window", "side", "entry_how", "exit_reason",
              "notional", "gross", "costs", "net", "risk", "minutes"]


# --------------------------------------------------------------------------- config
def make_cfg(strategy, entry_mode, p):
    """BotConfig for one run. Risk-limit halts OFF: we measure the signal, not the risk layer."""
    from config.settings import BotConfig
    c = BotConfig()
    c.strategy.active_strategy = strategy
    c.execution.entry_mode = entry_mode
    c.backtest.initial_balance = INITIAL_BALANCE
    c.backtest.maker_fee = p["maker_fee"]
    c.backtest.taker_fee = p["taker_fee"]
    c.backtest.slippage = p["slippage"]
    c.backtest.funding_rate = p["funding"]
    c.backtest.maker_penetration = p["penetration"]
    c.risk.max_drawdown_pct = 1.0
    c.risk.max_daily_loss_pct = 1.0
    c.risk.max_consecutive_losses = 10 ** 9
    c.risk.daily_trade_limit = 0
    c.risk.circuit_breaker_enabled = False
    c.risk.reduce_size_on_loss = False
    return c


# --------------------------------------------------------------------------- windows
def month_windows(index, test_months=1, skip_months=0):
    """Consecutive calendar-month windows fully covered by the data: [(start, end), ...]."""
    if len(index) == 0:
        return []
    first, last = index.min(), index.max()
    start = first.normalize().replace(day=1)
    if start < first.normalize():
        start = start + pd.DateOffset(months=1)
    wins = []
    while True:
        end = start + pd.DateOffset(months=test_months)
        if end > last + pd.Timedelta(minutes=5):      # month must be fully covered (bars are 5m)
            break
        wins.append((start, end))
        start = end
    return wins[skip_months:]


def load_csv(path):
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True).sort_index()
    return df[~df.index.duplicated(keep="last")]


def symbol_from_path(path):
    stem = os.path.basename(path).rsplit("_", 1)[0]          # BTC_USDT_5m.csv -> BTC_USDT
    return stem.replace("_", "/", 1)


# --------------------------------------------------------------------------- one task
def run_task(task):
    """All windows for one (symbol, strategy, scenario). Returns (rows, run_rows, stats, max_acct_err)."""
    from core.backtester import Backtester
    path, symbol, strategy, label, mode, p = task
    df = load_csv(path)
    if p["last_months"]:
        df = df[df.index >= df.index.max() - pd.DateOffset(months=p["last_months"])]
    rows, runs, stats, acct_err = [], [], {}, 0.0
    for wi, (ws, we) in enumerate(month_windows(df.index, p["test_months"], p["skip_months"])):
        i0, i1 = df.index.searchsorted(ws), df.index.searchsorted(we)
        if i1 - i0 < 500:
            continue
        seg = df.iloc[max(0, i0 - WARMUP_BARS):i1]
        bt = Backtester(make_cfg(strategy, mode, p))
        m = bt.run(seg.copy(), symbol, trade_from=ws)
        acct_err = max(acct_err, abs(bt.balance - (INITIAL_BALANCE + sum(t.pnl for t in bt.trades))))
        window_label = f"{ws:%Y-%m}"
        for t in bt.trades:
            if t.exit_reason == "END":                        # truncated by the window edge
                stats["end_excluded"] = stats.get("end_excluded", 0) + 1
                continue
            rows.append((strategy, label, symbol, window_label, t.side, t.entry_how, t.exit_reason,
                         t.notional, t.gross_pnl, t.costs, t.pnl,
                         abs(t.entry_price - t.sl) * t.size,
                         (t.exit_time - t.entry_time).total_seconds() / 60.0))
        runs.append({"strategy": strategy, "scenario": label, "symbol": symbol, "window": window_label,
                     "days": (we - ws).days, "dd": float(m.get("max_drawdown_pct", 0.0)) if "error" not in m else 0.0})
        for k, v in bt.stats.items():
            stats[k] = stats.get(k, 0) + v
    return rows, runs, stats, acct_err


# --------------------------------------------------------------------------- metrics
def add_derived(t):
    t = t.copy()
    n = t["notional"].replace(0, np.nan)
    t["gross_bps"] = t["gross"] / n * 1e4
    t["cost_bps"] = t["costs"] / n * 1e4
    t["net_bps"] = t["net"] / n * 1e4
    risk = t["risk"].replace(0, np.nan)
    t["net_R"] = t["net"] / risk
    t["gross_R"] = t["gross"] / risk
    return t


def cluster_bootstrap_ci(t, value="net_bps", reps=2000, seed=7):
    """95% CI of the pooled mean, resampling (symbol, window) clusters (trades inside a
    cluster are not independent: same market, same month)."""
    g = t.groupby(["symbol", "window"])[value].agg(["sum", "count"])
    if len(g) < 2:
        return (np.nan, np.nan)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(t, symbol_days, reps=2000, seed=7):
    """One row of metrics. t = derived trades df for one strategy x scenario."""
    if len(t) == 0:
        return None
    win = t["net"] > 0
    losses = t.loc[~win, "net"]
    lo, hi = cluster_bootstrap_ci(t, "net_bps", reps, seed)
    month_means = t.groupby("window")["net_bps"].mean()
    return {
        "n": len(t),
        "per_day": len(t) / symbol_days if symbol_days else np.nan,
        "win_pct": float(win.mean() * 100),
        "gross_bps": float(t["gross_bps"].mean()),                 # mean over ALL trades
        "cost_bps": float(t["cost_bps"].mean()),
        "net_bps": float(t["net_bps"].mean()),
        "ci_lo": lo, "ci_hi": hi,
        "net_R": float(t["net_R"].mean()),
        "net_usd": float(t["net"].mean()),
        "avg_win_bps": float(t.loc[win, "net_bps"].mean()) if win.any() else 0.0,
        "avg_loss_bps": float(t.loc[~win, "net_bps"].mean()) if (~win).any() else 0.0,
        "pf": float(t.loc[win, "net"].sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf,
        "pos_months": int((month_means > 0).sum()),
        "months": int(len(month_means)),
        "pos_symbols": int((t.groupby("symbol")["net_bps"].mean() > 0).sum()),
        "symbols": int(t["symbol"].nunique()),
    }


def verdict(s):
    """Plain-language call. 'Edge candidate' needs a positive CI AND a majority of positive months."""
    if s is None:
        return "NO TRADES"
    if np.isnan(s["ci_lo"]):
        return "TOO FEW WINDOWS"
    if s["ci_lo"] > 0 and s["pos_months"] > s["months"] / 2:
        return "EDGE CANDIDATE (confirm on fresh data + paper)"
    if s["ci_hi"] < 0:
        return "LOSES MONEY (CI below 0)"
    return "NO EDGE DEMONSTRATED (CI includes 0)"


# --------------------------------------------------------------------------- formatting
def f(x, d=1, signed=False):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "n/a"
    if isinstance(x, float) and np.isinf(x):
        return "inf"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def git_rev():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- report
def kairi_diagnostics(files, last_months):
    from core.indicators import Indicators
    from config.settings import BotConfig
    thr = BotConfig().strategy.disparity_panic_threshold
    crash = BotConfig().strategy.disparity_crash_threshold
    rows = []
    for path in files:
        df = load_csv(path)
        if last_months:
            df = df[df.index >= df.index.max() - pd.DateOffset(months=last_months)]
        d = Indicators.disparity(df["close"], 25).dropna()
        rows.append((symbol_from_path(path), len(d), f(d.min(), 2), f(d.quantile(0.001), 2),
                     f(d.quantile(0.01), 2), int((d <= thr).sum()), int((d <= crash).sum())))
    return thr, crash, rows


def build_report(trades, runs, stats, files, args, params, acct_err, scen_filter):
    t = add_derived(trades) if len(trades) else trades
    runs = pd.DataFrame(runs, columns=["strategy", "scenario", "symbol", "window", "days", "dd"])
    L = []
    L.append(f"# Baseline Report - {args.date}")
    L.append("")
    L.append(f"Generated by `scripts/generate_baseline_report.py` at commit `{git_rev()}`. "
             "No strategy parameter was tuned: `config/settings.py` defaults are used as-is.")
    L.append("")
    # ---- data
    L.append("## 1. Data and method")
    d_rows = []
    for path in files:
        df = load_csv(path)
        exp = int((df.index[-1] - df.index[0]) / pd.Timedelta(minutes=5)) + 1
        d_rows.append((symbol_from_path(path), f"{df.index[0]:%Y-%m-%d}", f"{df.index[-1]:%Y-%m-%d}",
                       f"{len(df):,}", f"{len(df) / exp:.2%}"))
    L.append(md_table(["Symbol", "From", "To", "5m bars", "Coverage"], d_rows))
    L.append("")
    n_windows = runs["window"].nunique() if len(runs) else 0
    sym_days = (runs.drop_duplicates(["symbol", "window"])["days"].sum() if len(runs) else 0)
    L.append(f"- **{len(files)} symbols x {n_windows} test months** ({sym_days:,} symbol-days). "
             f"Each month is tested on bars the strategy never saw; nothing is fitted, so there is no look-ahead "
             f"(indicators are warmed on the {WARMUP_BARS} bars before each month, with no trading during warm-up).")
    L.append(f"- Symbols are ranked by **current** volume, so the set is biased towards coins that are big today "
             f"(survivorship). Treat results as optimistic for alts.")
    L.append(f"- **Risk-limit halts are off** for measurement (drawdown, daily loss, loss-streak, circuit breaker); "
             f"sizing is the bot's own (1% risk, position capped at 10% of capital), so dollar pnl is small. "
             f"Compare strategies in **bps of notional** and **R** (R = pnl / that trade's own stop distance).")
    L.append(f"- Trades cut off by a month boundary are excluded ({stats.get('end_excluded', 0)} of them).")
    L.append(f"- Accounting self-check (balance == start + sum of net trade pnl, every run): "
             f"**{'PASS' if acct_err < 1e-6 else 'FAIL'}** (max error {acct_err:.2e}).")
    L.append("")
    L.append("### Cost assumptions (edit with CLI flags)")
    L.append(md_table(["Maker fee", "Taker fee", "Slippage / market leg", "Maker fill rule", "Funding"], [(
        f"{params['maker_fee'] * 1e4:.1f} bps", f"{params['taker_fee'] * 1e4:.1f} bps",
        f"{params['slippage'] * 1e4:.1f} bps",
        f"limit at signal close; next bar must trade {params['penetration'] * 1e4:.1f} bps through",
        f"{params['funding'] * 100:.4f}% per 8h")]))
    L.append("")
    L.append("Defaults are Binance USD-M futures VIP0 (maker 2 bps / taker 5 bps); Binance spot is 10 / 10 bps. "
             "**Use your real fee tier** (`--maker-fee`, `--taker-fee`). Every exit is a market order "
             "(taker fee + slippage), as in `main.py`. Maker entries pay no slippage.")
    L.append("")
    L.append(md_table(["Scenario", "Entry"], [(title, f"`{mode}`") for lab, title, mode in SCENARIOS if lab in scen_filter]))
    L.append("")
    # ---- verdicts
    sums, rows = {}, []
    for strat in args.strategies:
        for lab, title, _ in SCENARIOS:
            if lab not in scen_filter:
                continue
            sub = t[(t["strategy"] == strat) & (t["scenario"] == lab)] if len(t) else t
            days = runs[(runs["strategy"] == strat) & (runs["scenario"] == lab)].drop_duplicates(["symbol", "window"])["days"].sum() if len(runs) else 0
            sums[(strat, lab)] = summarize(sub, days, args.bootstrap, args.seed) if len(sub) else None
    L.append("## 2. Verdict")
    L.append("An **edge candidate** needs the 95% CI of NET expectancy above zero *and* a majority of profitable months. "
             "Anything else is *not demonstrated*, which is weaker than *proven absent*.")
    L.append("")
    v_rows = []
    for strat in args.strategies:
        for lab, title, _ in SCENARIOS:
            if lab not in scen_filter:
                continue
            s = sums[(strat, lab)]
            if s is None:
                v_rows.append((f"`{strat}`", lab, 0, "-", "-", "-", "NO TRADES"))
                continue
            v_rows.append((f"`{strat}`", lab, f"{s['n']:,}", f(s["net_bps"], 2, True),
                           f"[{f(s['ci_lo'], 2, True)}, {f(s['ci_hi'], 2, True)}]",
                           f"{s['pos_months']}/{s['months']}", verdict(s)))
    L.append(md_table(["Strategy", "Scenario", "Trades", "Net bps/trade", "95% CI", "Profitable months", "Verdict"], v_rows))
    L.append("")
    # ---- per scenario
    L.append("## 3. Results by scenario")
    for lab, title, mode in SCENARIOS:
        if lab not in scen_filter:
            continue
        L.append(f"### {title}")
        rows = []
        for strat in args.strategies:
            s = sums[(strat, lab)]
            if s is None:
                rows.append((f"`{strat}`", 0, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"))
                continue
            dd = runs[(runs["strategy"] == strat) & (runs["scenario"] == lab)]["dd"]
            rows.append((f"`{strat}`", f"{s['n']:,}", f(s["per_day"], 1), f(s["win_pct"], 1),
                         f(s["gross_bps"], 2, True), f(s["cost_bps"], 2), f(s["net_bps"], 2, True),
                         f(s["net_R"], 3, True), f(s["net_usd"], 3, True),
                         f"{f(s['avg_win_bps'], 1, True)} / {f(s['avg_loss_bps'], 1, True)}",
                         f(s["pf"], 2), f(float(dd.min()), 1)))
        L.append(md_table(["Strategy", "Trades", "/symbol-day", "Win %", "Gross bps", "Cost bps", "**Net bps**",
                           "Net R", "Net $", "Avg win / loss bps", "Profit factor", "Worst-month DD %"], rows))
        L.append("")
        L.append("Gross = mean over ALL trades before costs. Net = gross - cost. A strategy needs gross bps "
                 "comfortably above cost bps before it can pay for itself.")
        L.append("")
        # exit reasons
        sub = t[t["scenario"] == lab] if len(t) else t
        if len(sub):
            er = (sub.groupby(["strategy", "exit_reason"])
                  .agg(n=("net", "size"), net_bps=("net_bps", "mean")).reset_index())
            tot = er.groupby("strategy")["n"].transform("sum")
            er["share"] = er["n"] / tot * 100
            L.append("Exit reasons")
            L.append(md_table(["Strategy", "Exit", "Trades", "Share %", "Mean net bps"],
                              [(f"`{r.strategy}`", r.exit_reason, f"{r.n:,}", f(r.share, 1), f(r.net_bps, 2, True))
                               for r in er.itertuples()]))
            L.append("")
    # ---- pivots
    L.append("## 4. Where it works and where it does not")
    for lab, title, _ in SCENARIOS:
        if lab not in scen_filter or not len(t):
            continue
        sub = t[t["scenario"] == lab]
        piv = sub.pivot_table(index="symbol", columns="strategy", values="net_bps", aggfunc="mean")
        cnt = sub.pivot_table(index="symbol", columns="strategy", values="net_bps", aggfunc="size")
        cols = [s for s in args.strategies if s in piv.columns]
        L.append(f"### Net bps per trade by symbol - {lab}")
        L.append(md_table(["Symbol"] + [f"`{c}`" for c in cols],
                          [(sym, *[f"{f(piv.loc[sym, c], 1, True)} ({int(cnt.loc[sym, c]) if not np.isnan(cnt.loc[sym, c]) else 0})"
                                    if not np.isnan(piv.loc[sym, c]) else "-" for c in cols]) for sym in piv.index]))
        L.append("")
        pm = sub.pivot_table(index="window", columns="strategy", values="net_bps", aggfunc="mean")
        L.append(f"### Net bps per trade by month - {lab}")
        L.append(md_table(["Month"] + [f"`{c}`" for c in cols],
                          [(w, *[f(pm.loc[w, c], 1, True) if c in pm.columns and not np.isnan(pm.loc[w, c]) else "-" for c in cols])
                           for w in pm.index]))
        L.append("")
    # ---- maker
    L.append("## 5. Maker-entry diagnostics")
    L.append("Fill behaviour of the `maker` / `maker_fallback` scenarios. The fill rule is an assumption "
             "(see cost assumptions): **measure the real fill rate in paper/live before relying on maker fees.**")
    L.append("")
    mrows = []
    for key, lab in (("maker", "all_maker"), ("maker_fallback", "mixed")):
        if lab not in scen_filter:
            continue
        a, fl, un = stats["by"].get(lab, {}).get("maker_attempts", 0), stats["by"].get(lab, {}).get("maker_fills", 0), \
            stats["by"].get(lab, {}).get("maker_unfilled", 0)
        fb = stats["by"].get(lab, {}).get("maker_fallbacks", 0)
        mrows.append((lab, f"{a:,}", f"{fl:,}", f"{fl / a:.1%}" if a else "n/a", f"{un:,}", f"{fb:,}"))
    L.append(md_table(["Scenario", "Orders", "Filled as maker", "Fill rate", "Skipped (maker)", "Taker fallbacks"], mrows))
    L.append("")
    # ---- signals
    L.append("## 6. Signal diagnostics")
    srows = []
    for strat in args.strategies:
        sig = stats["by_strat"].get(strat, {}).get("signals", 0)
        days = runs[(runs["strategy"] == strat) & (runs["scenario"] == stats["sig_scen"])].drop_duplicates(["symbol", "window"])["days"].sum() if len(runs) else 0
        srows.append((f"`{strat}`", f"{sig:,}", f(sig / days, 2) if days else "n/a"))
    L.append(f"Signals seen in the `{stats['sig_scen']}` run (before position-size and balance checks):")
    L.append("")
    L.append(md_table(["Strategy", "Signals", "Per symbol-day"], srows))
    L.append("")
    if "kairi_disparity" in args.strategies:
        thr, crash, krow = kairi_diagnostics(files, args.last_months)
        L.append(f"`kairi_disparity` buys when 25-bar disparity <= **{thr}%** (crash regime {crash}%). "
                 f"On 5m bars of liquid pairs that depth is rare, which is why it can show 0 trades. "
                 f"Disparity (%) of the closing price vs its 25-bar SMA:")
        L.append("")
        L.append(md_table(["Symbol", "Bars", "Min", "0.1th pct", "1st pct", f"Bars <= {thr}%", f"Bars <= {crash}%"], krow))
        L.append("")
    # ---- limitations
    L.append("## 7. What this report does NOT prove")
    for line in [
        "**Backtester vs live.** The backtester exits on stop, target or the 30 min time limit only. The live bot also "
        "arms a breakeven stop and a trailing stop (`risk_manager.py`), which are NOT simulated here. They can change results either way.",
        "**Intrabar order.** If a bar touches both stop and target, the stop is assumed first (pessimistic). Gaps through a stop fill at the stop price (optimistic).",
        "**Maker fills** use a simple queue-penetration rule. Real fill rates and adverse selection must be measured live.",
        "**Fees and slippage** are assumptions (flags above), identical for every symbol. Small alts slip more than BTC.",
        "**Per-symbol independence.** Each symbol is simulated alone on a fresh $1,000; portfolio caps, margin and "
        "exchange limits are not modelled.",
        "**One market history.** Months here are the months that happened. A regime not present in the data is not tested.",
        "**Survivorship** in the symbol list (ranked by today's volume).",
    ]:
        L.append(f"- {line}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--scenarios", nargs="*", default=[s[0] for s in SCENARIOS],
                    choices=[s[0] for s in SCENARIOS])
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0003)
    ap.add_argument("--penetration", type=float, default=0.0001)
    ap.add_argument("--funding", type=float, default=0.0, help="per-8h funding rate, longs pay when positive")
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--skip-months", type=int, default=0, help="drop the first N test months (train period)")
    ap.add_argument("--last-months", type=int, default=0, help="only use the most recent N months (quick runs)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save-trades", action="store_true", help="also write the trade list as .csv.gz")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*_5m.csv")))
    if not files:
        sys.exit(f"No *_5m.csv files in {args.data}. Run: python scripts/download_data.py")
    params = {"maker_fee": args.maker_fee, "taker_fee": args.taker_fee, "slippage": args.slippage,
              "penetration": args.penetration, "funding": args.funding,
              "test_months": args.test_months, "skip_months": args.skip_months, "last_months": args.last_months}
    scen = [s for s in SCENARIOS if s[0] in args.scenarios]
    tasks = [(p, symbol_from_path(p), st, lab, mode, params)
             for p in files for st in args.strategies for lab, _, mode in scen]
    print(f"{len(files)} symbols x {len(args.strategies)} strategies x {len(scen)} scenarios = {len(tasks)} tasks, jobs={args.jobs}")

    all_rows, all_runs, acct_err = [], [], 0.0
    stats = {"end_excluded": 0, "by": {}, "by_strat": {},
             "sig_scen": REALISTIC if REALISTIC in args.scenarios else args.scenarios[0]}
    def absorb(res):
        nonlocal acct_err
        rows, runs, st, err = res
        all_rows.extend(rows)
        all_runs.extend(runs)
        acct_err = max(acct_err, err)
        stats["end_excluded"] += st.get("end_excluded", 0)
        if runs:
            lab, strat = runs[0]["scenario"], runs[0]["strategy"]
            for k, v in st.items():
                stats["by"].setdefault(lab, {}).setdefault(k, 0)
                stats["by"][lab][k] += v
                if lab == stats["sig_scen"]:
                    stats["by_strat"].setdefault(strat, {}).setdefault(k, 0)
                    stats["by_strat"][strat][k] += v

    done = 0
    if args.jobs <= 1:
        for task in tasks:
            absorb(run_task(task))
            done += 1
            print(f"  [{done}/{len(tasks)}] {task[1]} {task[2]} {task[3]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_task, tk): tk for tk in tasks}
            for fu in as_completed(futs):
                absorb(fu.result())
                done += 1
                tk = futs[fu]
                print(f"  [{done}/{len(tasks)}] {tk[1]} {tk[2]} {tk[3]}", flush=True)

    trades = pd.DataFrame(all_rows, columns=TRADE_COLS)
    os.makedirs(args.out, exist_ok=True)
    report = build_report(trades, all_runs, stats, files, args, params, acct_err, set(args.scenarios))
    path = os.path.join(args.out, f"baseline_{args.date}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    if args.save_trades:
        trades.to_csv(os.path.join(args.out, f"baseline_{args.date}_trades.csv.gz"), index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
