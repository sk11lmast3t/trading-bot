"""Command line entry point for the paper-only strategy evolution workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from evolve.costs import learn_costs
from evolve.evaluate import evaluate_candidate, load_costs, validate_candidate_source
from evolve.gates import validation_verdict
from evolve.ledger import append_entry, holdout_attempted, latest_candidate, read_entries
from evolve.propose import propose_candidate
from evolve.report import (DEFAULT_MAX_CANDIDATES, DEFAULT_TOKEN_BUDGET,
                           write_cycle)


def _safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Candidate name may contain only letters, numbers, underscore, and hyphen")
    return name


def _settings(maker_fee=None, taker_fee=None, slippage=None, long_only=False):
    costs = load_costs(maker_fee, taker_fee, slippage)
    return {"maker_fee": costs["maker_fee"], "taker_fee": costs["taker_fee"],
            "slippage": costs["slippage"], "long_only": bool(long_only)}


def _row_settings(row):
    params = row.get("params") or {}
    costs = params.get("costs") or {}
    def value(key, fallback):
        raw = row.get(key)
        if raw is None:
            raw = costs.get(key, fallback)
        return float(raw)
    return {
        "maker_fee": value("maker_fee", 0.0002),
        "taker_fee": value("taker_fee", 0.0005),
        "slippage": value("slippage", 0.0003),
        "long_only": bool(row.get("long_only", params.get("long_only", False))),
    }


def _trial_count(entries, settings):
    return sum(1 for row in entries
               if row.get("event", "candidate") == "candidate"
               and _row_settings(row) == settings)


def cycle(max_candidates=DEFAULT_MAX_CANDIDATES, token_budget=DEFAULT_TOKEN_BUDGET,
          llm_client=None, data_dir="data", maker_fee=None, taker_fee=None,
          slippage=None, long_only=False, stop_pct=0.02,
          min_target_pct=0.05, max_leverage=1.0):
    if max_candidates < 1 or max_candidates > 3:
        raise ValueError("A cycle may create between 1 and 3 candidates")
    learn_costs()
    settings = _settings(maker_fee, taker_fee, slippage, long_only)
    costs = {key: settings[key] for key in ("maker_fee", "taker_fee", "slippage")}
    print("Cycle settings: " + ", ".join(f"{key}={value}" for key, value in settings.items())
          + f", max_candidates={max_candidates}, token_budget={token_budget}")
    records = []
    spent = 0
    entries = read_entries()
    for offset in range(max_candidates):
        name = f"evolve_{len([e for e in entries if e.get('event', 'candidate') == 'candidate']) + offset + 1:04d}"
        row = {"event": "candidate", "name": name, "parent": "baseline",
               "code_hash": None,
               **settings,
               "params": {"costs": costs, "long_only": settings["long_only"],
                          "stop_pct": stop_pct, "min_target_pct": min_target_pct,
                          "max_leverage": max_leverage,
                          "partitions": ["search", "validation"]},
               "search": {}, "validation": {}}
        try:
            path, used, _ = propose_candidate(name, llm_client=llm_client,
                                               token_budget=token_budget, already_spent=spent,
                                               stop_pct=stop_pct, min_target_pct=min_target_pct,
                                               max_leverage=max_leverage)
            spent += used
            with open(path, encoding="utf-8") as stream:
                source = stream.read()
            row["code_hash"] = hashlib.sha256(source.encode()).hexdigest()
            trials = _trial_count(entries, settings) + 1
            row["search"] = evaluate_candidate(name, "search", data_dir=data_dir, fees=costs,
                                               trials=trials, stress_factor=1.5,
                                               long_only=settings["long_only"])
            row["validation"] = evaluate_candidate(name, "validation", data_dir=data_dir, fees=costs,
                                                   trials=trials, stress_factor=1.5,
                                                   long_only=settings["long_only"])
            gate = validation_verdict(row["search"], row["validation"], trials)
            row.update(gate)
            row["what"] = "one fixed candidate rule set evaluated on SEARCH, then VALIDATION"
        except Exception as exc:
            row.update(verdict="failed", reason_failed=[f"cycle error: {type(exc).__name__}: {exc}"])
        append_entry(row)
        entries.append(row)
        records.append(row)
    report_path = write_cycle(records, settings=settings, max_candidates=max_candidates,
                              token_budget=token_budget)
    print(f"Wrote {report_path}")
    return records


def holdout(name, data_dir="data", maker_fee=None, taker_fee=None,
            slippage=None, long_only=False):
    name = _safe_name(name)
    settings = _settings(maker_fee, taker_fee, slippage, long_only)
    fees = {key: settings[key] for key in ("maker_fee", "taker_fee", "slippage")}
    if holdout_attempted(name):
        append_entry({"event": "holdout_refused", "name": name,
                      **settings, "params": {"costs": fees, "long_only": settings["long_only"]},
                      "reason": "a holdout attempt is already recorded"})
        raise RuntimeError(f"Holdout already attempted for {name}; second run refused")
    candidate = latest_candidate(name)
    if not candidate or candidate.get("verdict") != "passed":
        raise RuntimeError("Candidate must pass validation before its one holdout run")
    append_entry({"event": "holdout_attempt", "name": name, **settings,
                  "params": {"costs": fees, "long_only": settings["long_only"]}})
    try:
        result = evaluate_candidate(name, "holdout", data_dir=data_dir,
                                    fees=fees, trials=max(1, _trial_count(read_entries(), settings)),
                                    long_only=settings["long_only"])
        append_entry({"event": "holdout_result", "name": name, **settings,
                      "params": {"costs": fees, "long_only": settings["long_only"]},
                      "result": result})
        print(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        append_entry({"event": "holdout_failure", "name": name, **settings,
                      "params": {"costs": fees, "long_only": settings["long_only"]},
                      "reason": f"{type(exc).__name__}: {exc}"})
        raise


def promote(name):
    name = _safe_name(name)
    candidate = latest_candidate(name)
    if not candidate or candidate.get("verdict") != "passed":
        raise RuntimeError("Promotion refused: candidate has not passed validation")
    if not holdout_attempted(name):
        raise RuntimeError("Promotion refused: candidate has not had its single holdout run")
    outcomes = [r for r in read_entries() if r.get("name") == name and r.get("event") == "holdout_result"]
    if not outcomes:
        raise RuntimeError("Promotion refused: holdout run has no recorded result")
    os.makedirs("config", exist_ok=True)
    path = os.path.join("config", "paper_candidates.json")
    try:
        with open(path, encoding="utf-8") as stream:
            registry = json.load(stream)
    except FileNotFoundError:
        registry = {"paper_only": []}
    if name not in registry.setdefault("paper_only", []):
        registry["paper_only"].append(name)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(registry, stream, indent=2)
        stream.write("\n")
    print(f"Registered {name} for paper mode only. Run: python main.py --strategy candidate:{name}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evolve")
    sub = parser.add_subparsers(dest="command", required=True)
    cyc = sub.add_parser("cycle")
    cyc.add_argument("--max-candidates", type=int, default=3)
    cyc.add_argument("--token-budget", type=int, default=12000)
    cyc.add_argument("--data", default="data")
    cyc.add_argument("--maker-fee", type=float)
    cyc.add_argument("--taker-fee", type=float)
    cyc.add_argument("--slippage", type=float)
    cyc.add_argument("--long-only", action="store_true")
    cyc.add_argument("--stop-pct", type=float, default=0.02)
    cyc.add_argument("--min-target-pct", type=float, default=0.05)
    cyc.add_argument("--max-leverage", type=float, default=1.0)
    hold = sub.add_parser("holdout")
    hold.add_argument("name")
    hold.add_argument("--data", default="data")
    hold.add_argument("--maker-fee", type=float)
    hold.add_argument("--taker-fee", type=float)
    hold.add_argument("--slippage", type=float)
    hold.add_argument("--long-only", action="store_true")
    prom = sub.add_parser("promote")
    prom.add_argument("name")
    args = parser.parse_args(argv)
    if args.command == "cycle":
        cycle(args.max_candidates, args.token_budget, data_dir=args.data,
              maker_fee=args.maker_fee, taker_fee=args.taker_fee,
              slippage=args.slippage, long_only=args.long_only,
              stop_pct=args.stop_pct, min_target_pct=args.min_target_pct,
              max_leverage=args.max_leverage)
    elif args.command == "holdout":
        holdout(args.name, args.data, maker_fee=args.maker_fee,
                taker_fee=args.taker_fee, slippage=args.slippage,
                long_only=args.long_only)
    else:
        promote(args.name)


if __name__ == "__main__":
    main()
