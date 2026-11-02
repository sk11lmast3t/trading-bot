"""
Walk-forward evaluation engine.
"""
import os
import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """Walk-forward evaluation with train (3mo) / test (1mo) windows."""

    def __init__(self, train_months: int = 3, test_months: int = 1, step_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def windows(self, df: pd.DataFrame) -> List[Dict]:
        df = df.sort_index()
        start = df.index.min()
        end = df.index.max()
        out = []
        cur = start
        while True:
            train_start = cur
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            if test_end > end:
                break
            train_df = df.loc[(df.index >= train_start) & (df.index < train_end)]
            test_df = df.loc[(df.index >= test_start) & (df.index < test_end)]
            if len(train_df) > 0 and len(test_df) > 0:
                out.append({"train": train_df, "test": test_df,
                            "train_start": train_start, "test_start": test_start})
            cur = cur + pd.DateOffset(months=self.step_months)
        return out

    def run(self, df: pd.DataFrame, evaluate: Callable, cache_path: Optional[str] = None) -> Dict:
        if (df is None or (hasattr(df, "empty") and df.empty)) and cache_path and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        oos_results = []
        for w in self.windows(df):
            res = evaluate(w["train"], w["test"])
            oos_results.append(res)
        agg = {}
        if oos_results:
            keys = list(oos_results[0].keys()) if isinstance(oos_results[0], dict) else []
            for k in keys:
                vals = [r[k] for r in oos_results if isinstance(r, dict) and k in r]
                if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                    agg[k] = sum(vals) / len(vals)
                else:
                    agg[k] = vals
        return {"windows": len(oos_results), "oos": oos_results, "aggregate": agg}

    @staticmethod
    def load_cached(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0, parse_dates=True)
