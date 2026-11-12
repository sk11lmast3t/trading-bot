"""Calendar-month data splits with an isolated, explicit holdout reader."""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    symbol: str
    path: str
    search: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    validation: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    holdout: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]


def month_windows(index, test_months=1):
    """Match scripts/generate_baseline_report.py's fully covered months."""
    if len(index) == 0:
        return []
    index = pd.DatetimeIndex(index)
    first, last = index.min(), index.max()
    start = first.normalize().replace(day=1)
    if start < first.normalize():
        start += pd.DateOffset(months=1)
    result = []
    while True:
        end = start + pd.DateOffset(months=test_months)
        if end > last + pd.Timedelta(minutes=5):
            break
        result.append((start, end))
        start = end
    return result


def _symbol(path):
    return os.path.basename(path).rsplit("_", 1)[0].replace("_", "/", 1)


def _timestamps(path):
    # Read timestamps only to establish the calendar. OHLCV values in the
    # holdout are not loaded by discovery, cycle, or evaluation.
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "timestamp" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a timestamp column")
        values = [row["timestamp"] for row in reader if row.get("timestamp")]
    return pd.to_datetime(values, errors="raise")


def discover_splits(data_dir="data"):
    paths = sorted(glob.glob(os.path.join(data_dir, "*_5m.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_5m.csv files found in {data_dir}")
    splits = {}
    for path in paths:
        windows = month_windows(_timestamps(path))
        if len(windows) < 14:
            # Partial-history symbols cannot support all three prescribed
            # partitions and are excluded from the experiment.
            continue
        splits[_symbol(path)] = DataSplit(
            _symbol(path), path, tuple(windows[:10]), tuple(windows[10:12]), tuple(windows[-2:]),
        )
    if not splits:
        raise ValueError(f"No *_5m.csv files in {data_dir} have 14 fully covered months")
    return splits


def windows_for(split: DataSplit, partition: str):
    if partition == "search":
        return split.search
    if partition == "validation":
        return split.validation
    if partition == "holdout":
        return split.holdout
    raise ValueError(f"Unknown data partition: {partition}")


def load_partition(split: DataSplit, partition: str, warmup_bars=300):
    """Load OHLCV for a partition and its preceding non-holdout warmup bars."""
    windows = windows_for(split, partition)
    if partition != "holdout":
        first = windows[0][0]
        last = windows[-1][1]
    else:
        first, last = windows[0][0], windows[-1][1]
    timestamps = _timestamps(split.path)
    end_index = timestamps.searchsorted(first)
    start_index = max(0, end_index - int(warmup_bars))
    stop_index = timestamps.searchsorted(last)
    if partition != "holdout":
        # Search/validation callers never receive holdout rows.
        hold_start = split.holdout[0][0]
        stop_index = min(stop_index, timestamps.searchsorted(hold_start))
    # pandas skips all OHLCV fields outside the allowed row interval; timestamp
    # discovery above reads only the index column, never held-out price values.
    data_start, data_stop = start_index, max(start_index, stop_index)
    frame = pd.read_csv(
        split.path, index_col="timestamp", parse_dates=True,
        skiprows=lambda line: line != 0 and not (data_start < line <= data_stop),
        nrows=max(0, data_stop - data_start),
    ).sort_index()
    return frame[~frame.index.duplicated(keep="last")].copy()


def partition_month_labels(split, partition):
    return [start.strftime("%Y-%m") for start, _ in windows_for(split, partition)]

