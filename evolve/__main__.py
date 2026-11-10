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
from evolve.report import write_cycle


def _safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Candidate name may contain only letters, numbers, underscore, and hyphen")
    return name


def cycle(max_candidates=3, token_budget=12000, llm_client=None, data_dir="data"):
    if max_candidates < 1 or max_candidates > 3:
        raise ValueError("A cycle may create between 1 and 3 candidates")
    learn_costs()
    costs = load_costs()
    records = []
    spent = 0
    entries = read_entries()
    for offset in range(max_candidates):
        name = f"evolve_{len([e for e in entries if e.get('event', 'candidate') == 'candidate']) + offset + 1:04d}"
        row = {"event": "candidate", "name": name, "parent": "baseline",
               "code_hash": None,
               "params": {"costs": costs, "partitions": ["search", "validation"]},
               "search": {}, "validation": {}}
        try:
            path, used, _ = propose_candidate(name, llm_client=llm_client,
                                               token_budget=token_budget, already_spent=spent)
            spent += used
            with open(path, encoding="utf-8") as stream:
                source = stream.read()
            row["code_hash"] = hashlib.sha256(source.encode()).hexdigest()
            row["search"] = evaluate_candidate(name, "search", data_dir=data_dir, fees=costs,
                                               trials=len(entries) + 1, stress_factor=1.5)
            row["validation"] = evaluate_candidate(name, "validation", data_dir=data_dir, fees=costs,
                                                   trials=len(entries) + 1, stress_factor=1.5)
            gate = validation_verdict(row["search"], row["validation"], len(entries) + 1)
            row.update(gate)
            row["what"] = "one fixed candidate rule set evaluated on SEARCH, then VALIDATION"
        except Exception as exc:
            row.update(verdict="failed", reason_failed=[f"cycle error: {type(exc).__name__}: {exc}"])
        append_entry(row)
        entries.append(row)
        records.append(row)
    report_path = write_cycle(records)
    print(f"Wrote {report_path}")
    return records


def holdout(name, data_dir="data"):
    name = _safe_name(name)
    if holdout_attempted(name):
        append_entry({"event": "holdout_refused", "name": name,
                      "reason": "a holdout attempt is already recorded"})
        raise RuntimeError(f"Holdout already attempted for {name}; second run refused")
    candidate = latest_candidate(name)
    if not candidate or candidate.get("verdict") != "passed":
        raise RuntimeError("Candidate must pass validation before its one holdout run")
    append_entry({"event": "holdout_attempt", "name": name})
    try:
        result = evaluate_candidate(name, "holdout", data_dir=data_dir,
                                    fees=load_costs(), trials=len(read_entries()))
        append_entry({"event": "holdout_result", "name": name, "result": result})
        print(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        append_entry({"event": "holdout_failure", "name": name,
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
    hold = sub.add_parser("holdout")
    hold.add_argument("name")
    hold.add_argument("--data", default="data")
    prom = sub.add_parser("promote")
    prom.add_argument("name")
    args = parser.parse_args(argv)
    if args.command == "cycle":
        cycle(args.max_candidates, args.token_budget, data_dir=args.data)
    elif args.command == "holdout":
        holdout(args.name, args.data)
    else:
        promote(args.name)


if __name__ == "__main__":
    main()
