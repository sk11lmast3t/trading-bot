"""scripts/generate_baseline_report.py: metric math, windows, verdicts, end-to-end run."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "gen_report", os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_baseline_report.py"))
gr = importlib.util.module_from_spec(SPEC)
sys.modules["gen_report"] = gr          # needed so worker processes / pickling can import it
SPEC.loader.exec_module(gr)


def trades(gross, costs=0.1, notional=100.0, risk=1.0, clusters=None):
    n = len(gross)
    clusters = clusters or [("A/USDT", "2026-01"), ("A/USDT", "2026-02")] * (n // 2 + 1)
    return pd.DataFrame({
        "strategy": "s", "scenario": "x",
        "symbol": [c[0] for c in clusters[:n]], "window": [c[1] for c in clusters[:n]],
        "side": "long", "entry_how": "market", "exit_reason": "TP",
        "notional": notional, "gross": gross, "costs": costs,
        "net": np.array(gross) - costs, "risk": risk, "minutes": 5.0,
    })


def test_expectancy_is_mean_over_all_trades_not_winrate_times_avg_win():
    t = gr.add_derived(trades([2.0, -1.0, 3.0, -1.0]))
    s = gr.summarize(t, symbol_days=1, reps=200)
    assert s["gross_bps"] == pytest.approx(75.0)         # (200-100+300-100)/4 over ALL trades
    assert s["cost_bps"] == pytest.approx(10.0)
    assert s["net_bps"] == pytest.approx(s["gross_bps"] - s["cost_bps"])
    assert s["net_bps"] == pytest.approx(65.0)
    # the OLD buggy formula (win rate x average winning move) would give a different, wrong number
    old_formula = 0.5 * np.mean([200.0, 300.0])
    assert old_formula == pytest.approx(125.0) and abs(old_formula - s["gross_bps"]) > 1
    assert s["win_pct"] == pytest.approx(50.0)
    assert s["net_R"] == pytest.approx((1.9 - 1.1 + 2.9 - 1.1) / 4)
    assert s["per_day"] == pytest.approx(4.0)


def test_R_uses_each_trades_own_risk():
    t = trades([1.0, 1.0], costs=0.0)
    t["risk"] = [0.5, 2.0]
    d = gr.add_derived(t)
    assert list(d["net_R"]) == pytest.approx([2.0, 0.5])


def test_bootstrap_ci_brackets_mean_and_separates_from_zero():
    rng = np.random.default_rng(0)
    cl = [("A/USDT", f"2026-{m:02d}") for m in range(1, 7) for _ in range(50)]
    pos = gr.add_derived(trades(list(rng.normal(0.5, 0.3, 300)), costs=0.0, clusters=cl))
    lo, hi = gr.cluster_bootstrap_ci(pos, reps=1000)
    assert lo < pos["net_bps"].mean() < hi and lo > 0
    zero = gr.add_derived(trades(list(rng.normal(0.0, 0.3, 300)), costs=0.0, clusters=cl))
    lo0, hi0 = gr.cluster_bootstrap_ci(zero, reps=1000)
    assert lo0 < 0 < hi0


def base(**kw):
    s = {"ci_lo": 1.0, "ci_hi": 5.0, "pos_months": 8, "months": 10}
    s.update(kw)
    return s


def test_verdicts():
    assert gr.verdict(None) == "NO TRADES"
    assert gr.verdict(base(ci_lo=np.nan)) == "TOO FEW WINDOWS"
    assert gr.verdict(base()).startswith("EDGE CANDIDATE")
    assert gr.verdict(base(pos_months=4)).startswith("NO EDGE")           # CI>0 but minority of months
    assert gr.verdict(base(ci_lo=-2.0)).startswith("NO EDGE")             # CI includes 0
    assert gr.verdict(base(ci_lo=-9.0, ci_hi=-1.0)).startswith("LOSES")


def test_month_windows():
    idx = pd.date_range("2026-01-01", "2026-04-30 23:55", freq="5min")
    w = gr.month_windows(idx)
    assert [f"{a:%Y-%m}" for a, _ in w] == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert gr.month_windows(idx, skip_months=2)[0][0] == pd.Timestamp("2026-03-01")
    mid = pd.date_range("2026-01-15", "2026-04-30 23:55", freq="5min")
    assert gr.month_windows(mid)[0][0] == pd.Timestamp("2026-02-01")      # partial first month skipped
    short = pd.date_range("2026-01-01", "2026-03-20", freq="5min")
    assert len(gr.month_windows(short)) == 2                              # partial last month skipped


def test_symbol_from_path():
    assert gr.symbol_from_path("data/BTC_USDT_5m.csv") == "BTC/USDT"
    assert gr.symbol_from_path("data/1000SHIB_USDT_5m.csv") == "1000SHIB/USDT"


def synth(n, seed, vol=0.0012):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[close[0], close[:-1]]
    r2 = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * (1 + np.abs(r2.normal(0, vol / 2, n))),
        "low": np.minimum(open_, close) * (1 - np.abs(r2.normal(0, vol / 2, n))),
        "volume": r2.uniform(100, 1000, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="5min", name="timestamp"))


def test_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    synth(26000, 3).to_csv(data / "AAA_USDT_5m.csv")          # ~3 months
    synth(26000, 5).to_csv(data / "BBB_USDT_5m.csv")
    out = tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", [
        "gen", "--data", str(data), "--out", str(out), "--strategies", "grid_scalper", "kairi_disparity",
        "--scenarios", "all_taker", "all_maker", "--jobs", "1", "--bootstrap", "200", "--date", "2026-01-01"])
    gr.main()
    text = (out / "baseline_2026-01-01.md").read_text(encoding="utf-8")
    for needle in ["## 1. Data and method", "## 2. Verdict", "## 3. Results by scenario",
                   "## 4. Where it works", "## 5. Maker-entry diagnostics", "## 6. Signal diagnostics",
                   "## 7. What this report does NOT prove", "AAA/USDT", "BBB/USDT",
                   "Accounting self-check", "**PASS**", "0.1th pct"]:
        assert needle in text, needle
    # per-symbol tables carry a strategy column
    assert "| Symbol | `grid_scalper` |" in text
    # no unformatted raw floats like 0.41110263877445086
    assert not any(len(tok.split(".")[-1]) > 8 for tok in text.replace("|", " ").split()
                   if tok.replace(".", "", 1).replace("-", "", 1).isdigit() and "." in tok)
