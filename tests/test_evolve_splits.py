import csv

import pandas as pd

from evolve.splits import discover_splits, load_partition


def _write_months(path, months=16):
    idx = pd.date_range("2024-01-01", periods=months, freq="MS")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for i, ts in enumerate(idx):
            writer.writerow([ts.isoformat(), 100 + i, 101 + i, 99 + i, 100 + i, 10])


def test_split_months_are_search_then_validation_then_last_two_holdout(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_months(data_dir / "BTC_USDT_5m.csv", 16)

    split = discover_splits(str(data_dir))["BTC/USDT"]

    assert len(split.search) == 10
    assert len(split.validation) == 2
    assert len(split.holdout) == 2
    assert split.search[0][0] == pd.Timestamp("2024-01-01")
    assert split.validation[0][0] == pd.Timestamp("2024-11-01")
    assert split.holdout[0][0] == pd.Timestamp("2025-02-01")


def test_search_and_validation_partitions_exclude_holdout_rows(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "BTC_USDT_5m.csv"
    _write_months(path, 16)
    split = discover_splits(str(data_dir))["BTC/USDT"]

    search = load_partition(split, "search", warmup_bars=0)
    validation = load_partition(split, "validation", warmup_bars=0)

    assert search.index.max() < split.holdout[0][0]
    assert validation.index.max() < split.holdout[0][0]
    assert set(search.index).isdisjoint(set(split.holdout[0]))
