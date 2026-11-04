import pandas as pd
import numpy as np
from core.walk_forward import WalkForwardEngine


def make_df(months=6):
    idx = pd.date_range("2024-01-01", periods=months*30*24, freq="1h")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


def test_window_count():
    df = make_df(6)
    eng = WalkForwardEngine(train_months=3, test_months=1, step_months=1)
    wins = eng.windows(df)
    assert len(wins) >= 2
    for w in wins:
        assert (w["test"].index.min() - w["train"].index.min()).days >= 80


def test_oos_aggregation():
    df = make_df(6)
    eng = WalkForwardEngine(train_months=3, test_months=1, step_months=1)
    def evaluate(train, test):
        return {"n": len(test)}
    res = eng.run(df, evaluate)
    assert res["windows"] == len(res["oos"])
    assert "aggregate" in res
    assert "n" in res["aggregate"]


def test_cached_csv(tmp_path):
    df = make_df(4)
    p = tmp_path / "cached.csv"
    df.to_csv(p)
    eng = WalkForwardEngine(train_months=2, test_months=1, step_months=1)
    res = eng.run(None, lambda t, e: {"n": len(e)}, cache_path=str(p))
    assert res["windows"] >= 1
