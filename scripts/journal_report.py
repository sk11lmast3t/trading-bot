"""Summarize observational trade journal CSV data without ranking strategies."""
import argparse
import csv
import math
from collections import Counter, defaultdict
from statistics import mean


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bucket(column, row):
    value = row.get(column, "")
    number = _number(value)
    if column == "atr_1h_pct":
        if number is None:
            return None
        return "low (<0.5%)" if number < 0.5 else "medium (0.5–1.5%)" if number < 1.5 else "high (>=1.5%)"
    if column == "ema20_1h_slope_sign":
        return value or None
    if column == "ema20_1h_slope_pct":
        if number is None:
            return None
        magnitude = abs(number)
        size = "small (<0.01%)" if magnitude < 0.01 else "medium (0.01–0.05%)" if magnitude < 0.05 else "large (>=0.05%)"
        return f"{'negative' if number < 0 else 'positive' if number > 0 else 'flat'} / {size}"
    if column == "trend_30m" or column == "rv_24h_bucket":
        return value or None
    if column == "rsi_5m":
        if number is None:
            return None
        return "oversold (<30)" if number < 30 else "neutral (30–70)" if number < 70 else "overbought (>=70)"
    if column == "ema_distance_atr_5m":
        if number is None:
            return None
        return "below EMA (<-1 ATR)" if number < -1 else "near EMA (-1 to 1 ATR)" if number <= 1 else "above EMA (>1 ATR)"
    if column == "hour_utc":
        return f"{int(number):02d}:00 UTC" if number is not None else None
    if column == "day_of_week_utc":
        names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        return names[int(number)] if number is not None and 0 <= int(number) < 7 else None
    return None


def report(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    print(f"Journal rows: {len(rows)}")
    counts = Counter(row.get("status", "") or "(blank)" for row in rows)
    print("Counts by status:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")

    trades = []
    for row in rows:
        pnl = _number(row.get("net_pnl"))
        notional = _number(row.get("notional"))
        if row.get("status") == "closed" and pnl is not None and notional and notional > 0:
            trades.append((row, pnl / notional * 10000.0))

    print(f"Closed trades with net bps: {len(trades)}")
    print("Net bps per trade:")
    for row, bps in trades:
        print(f"  {row.get('ts', '')} {row.get('symbol', '')} {row.get('strategy', '')}: {bps:.2f} bps")

    contexts = (
        "atr_1h_pct", "ema20_1h_slope_sign", "ema20_1h_slope_pct", "trend_30m",
        "rsi_5m", "ema_distance_atr_5m", "hour_utc", "day_of_week_utc", "rv_24h_bucket",
    )
    print("Net bps by context bucket (closed trades only):")
    grouped = defaultdict(list)
    for row, bps in trades:
        for column in contexts:
            bucket = _bucket(column, row)
            if bucket is not None:
                grouped[(column, bucket)].append(bps)
    for column in contexts:
        print(f"  {column}:")
        buckets = sorted((bucket, values) for (name, bucket), values in grouped.items() if name == column)
        if not buckets:
            print("    no measured buckets")
        for bucket, values in buckets:
            print(f"    {bucket}: mean={mean(values):.2f} bps, trades={len(values)}")
            if len(values) < 100:
                print(f"    !!! TOO FEW TO CONCLUDE: {len(values)} trades; fewer than 100 !!!")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="logs/decisions.csv")
    args = parser.parse_args(argv)
    report(args.csv_path)


if __name__ == "__main__":
    main()
