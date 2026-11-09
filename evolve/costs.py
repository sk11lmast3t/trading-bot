"""Estimate observed paper execution costs without fitting strategy rules."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean


def _num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def learn_costs(journal_path="logs/journal.csv", output_path="config/measured_costs.json",
                minimum_trades=50, assumed=None):
    assumed = assumed or {"maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0003}
    grouped = defaultdict(dict)
    if os.path.exists(journal_path):
        with open(journal_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                symbol = row.get("symbol")
                if not symbol:
                    continue
                key = (row.get("ts", ""), row.get("strategy", ""))
                grouped[symbol][key] = row
    measured = {}
    print("Measured execution costs versus assumptions:")
    for symbol in sorted(grouped):
        rows = list(grouped[symbol].values())
        closed = [row for row in rows if row.get("status") == "closed"]
        attempts = [row for row in rows if row.get("status") in {
            "closed", "entered", "skipped:entry_not_filled", "unknown:entry_order",
        }]
        slippage_values = []
        for row in closed:
            value = _num(row.get("slippage_bps"))
            if value is None:
                entry = _num(row.get("entry_fill_diff_bps"))
                exit_ = _num(row.get("exit_fill_diff_bps"))
                parts = [abs(x) for x in (entry, exit_) if x is not None]
                value = sum(parts) if parts else None
            if value is not None:
                slippage_values.append(abs(value))
        if len(closed) < minimum_trades:
            print(f"  {symbol}: {len(closed)} trades; too few for measurement, assumptions unchanged")
            continue
        fill_rate = sum(row.get("status") in ("entered", "closed") for row in attempts) / max(1, len(attempts))
        avg_slippage = mean(slippage_values) if slippage_values else None
        record = {"trades": len(closed), "fill_rate": fill_rate,
                  "measured_slippage_bps": avg_slippage}
        if avg_slippage is not None:
            record["slippage"] = avg_slippage / 10000.0
        measured[symbol] = record
        assumed_bps = float(assumed.get("slippage", 0.0003)) * 10000.0
        diff = "unavailable" if avg_slippage is None else f"{avg_slippage - assumed_bps:+.2f} bps"
        print(f"  {symbol}: trades={len(closed)}, fill_rate={fill_rate:.1%}, "
              f"slippage={avg_slippage if avg_slippage is not None else 'n/a'} bps "
              f"(assumed {assumed_bps:.2f}; diff {diff})")
    payload = {**assumed, "symbols": measured}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return payload
