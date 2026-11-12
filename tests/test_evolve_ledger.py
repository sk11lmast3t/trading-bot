import json

from evolve.gates import validation_verdict
from evolve.ledger import append_entry, holdout_attempted, latest_candidate, prompt_entries, read_entries


def _metrics():
    return {"base": {"trades": 400, "adjusted_ci": [1.0, 4.0], "months": 2,
                     "profitable_months": 2, "symbols": 4, "profitable_symbols": 3,
                     "net_bps_per_trade": 2.0},
            "stress": {"net_bps_per_trade": .2}}


def test_ledger_is_jsonl_append_only_and_keeps_failure_reasons(tmp_path):
    path = tmp_path / "ledger.jsonl"
    candidate = {"event": "candidate", "name": "trial1", "parent": "baseline",
                 "code_hash": "abc", "params": {}, "search": {}, "validation": {},
                 "verdict": "failed", "reason_failed": ["CI lower bound is negative"]}
    append_entry(candidate, str(path))
    append_entry({"event": "holdout_attempt", "name": "trial1"}, str(path))

    rows = read_entries(str(path))
    assert len(rows) == 2
    assert latest_candidate("trial1", str(path))["reason_failed"] == ["CI lower bound is negative"]
    assert holdout_attempted("trial1", str(path))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_prompt_ledger_removes_holdout_results(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_entry({"event": "candidate", "name": "trial1", "search": {}, "validation": {},
                  "holdout": {"secret_metric": 999}}, str(path))
    append_entry({"event": "holdout_attempt", "name": "trial1", "result": {"secret_metric": 999}}, str(path))

    result = prompt_entries(str(path))

    assert len(result) == 1
    assert "holdout" not in result[0]
    assert "secret_metric" not in json.dumps(result)


def test_validation_gate_fails_without_ci_sample_and_stress_proof():
    result = validation_verdict({"base": {"net_bps_per_trade": -1}},
                                {"base": {"trades": 12, "adjusted_ci": [-3, 1],
                                           "months": 2, "profitable_months": 1,
                                           "symbols": 2, "profitable_symbols": 1},
                                 "stress": {"net_bps_per_trade": -4}}, 5)

    assert result["verdict"] == "failed"
    assert len(result["reason_failed"]) >= 4


def test_validation_gate_pass_requires_all_thresholds():
    result = validation_verdict(_metrics(), _metrics(), 2)

    assert result["verdict"] == "passed"
    assert result["reason_failed"] == []
