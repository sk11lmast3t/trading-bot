import pytest

import evolve.__main__ as evolve_cli
from evolve.report import render_cycle
from core.fills import FILL_MODEL_VERSION


def test_trial_counts_are_isolated_by_effective_settings():
    entries = [
        {"event": "candidate", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": False},  # Unversioned legacy row.
        {"event": "candidate", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": False,
         "fill_model_version": FILL_MODEL_VERSION},
        {"event": "candidate", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": True,
         "fill_model_version": FILL_MODEL_VERSION},
        {"event": "holdout_result", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": False},
    ]
    assert evolve_cli._trial_count(entries, {
        "maker_fee": .0002, "taker_fee": .0005, "slippage": .0003,
        "long_only": False, "fill_model_version": FILL_MODEL_VERSION,
    }) == 1


def test_legacy_candidate_cannot_enter_current_model_holdout_or_promotion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evolve").mkdir()
    evolve_cli.append_entry({"event": "candidate", "name": "legacy_trial",
                             "verdict": "passed", "fill_model_version": "legacy"})

    with pytest.raises(RuntimeError, match="different fill model"):
        evolve_cli.holdout("legacy_trial")
    with pytest.raises(RuntimeError, match="different fill model"):
        evolve_cli.promote("legacy_trial")


def test_cycle_report_starts_with_effective_settings_and_default_limits():
    report = render_cycle([], {"maker_fee": .0002, "taker_fee": .0005,
                               "slippage": .0003, "long_only": True})
    assert report.index("Effective settings:") < report.index("Search and validation")
    assert "--max-candidates=3 (default 3)" in report
    assert "--token-budget=12000 (default 12000)" in report
    assert "long_only=True" in report


def test_cycle_report_prints_break_even_hit_rate_for_configured_costs():
    report = render_cycle([{
        "name": "trial", "verdict": "failed", "reason_failed": [],
        "search": {"base": {}},
        "validation": {"base": {"trades": 2, "net_bps_per_trade": 0.0,
                                 "adjusted_ci": [None, None],
                                 "median_stop_pct": 2.0, "median_target_pct": 5.0,
                                 "target_before_stop_rate": 0.5}},
    }], {"maker_fee": .0002, "taker_fee": .0005, "slippage": .0003,
         "long_only": False})
    assert "Break-even hit rate at configured costs:" in report


def test_holdout_is_single_attempt_and_promotion_needs_validation_and_holdout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evolve").mkdir()
    with pytest.raises(RuntimeError, match="passed validation"):
        evolve_cli.promote("trial")
    evolve_cli.append_entry({"event": "candidate", "name": "trial", "verdict": "passed"})
    with pytest.raises(RuntimeError, match="single holdout"):
        evolve_cli.promote("trial")
    monkeypatch.setattr(evolve_cli, "evaluate_candidate", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(evolve_cli, "load_costs", lambda *args: {
        "maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0003,
    })
    evolve_cli.holdout("trial")
    with pytest.raises(RuntimeError, match="already attempted"):
        evolve_cli.holdout("trial")
    assert any(row.get("event") == "holdout_refused" for row in evolve_cli.read_entries())
    evolve_cli.promote("trial")
    assert "trial" in (tmp_path / "config" / "paper_candidates.json").read_text()



def _seed_holdout(monkeypatch, tmp_path, params=None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evolve").mkdir()
    row = {"event": "candidate", "name": "trial", "verdict": "passed"}
    if params is not None:
        row["params"] = params
    evolve_cli.append_entry(row)
    calls = []

    def fake_eval(*args, **kwargs):
        calls.append(kwargs)
        return {"pass": True}

    monkeypatch.setattr(evolve_cli, "evaluate_candidate", fake_eval)
    monkeypatch.setattr(evolve_cli, "load_costs", lambda *args: {
        "maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0003,
    })
    return calls


def test_holdout_refuses_a_different_hold_than_validation(tmp_path, monkeypatch):
    calls = _seed_holdout(monkeypatch, tmp_path, {"hold_minutes": 45.0})
    with pytest.raises(RuntimeError, match="differs"):
        evolve_cli.holdout("trial", hold_minutes=30)
    assert calls == []
    assert not any(row.get("event") == "holdout_attempt"
                   for row in evolve_cli.read_entries())


def test_holdout_uses_the_hold_recorded_on_the_candidate(tmp_path, monkeypatch):
    calls = _seed_holdout(monkeypatch, tmp_path, {"hold_minutes": 45.0})
    evolve_cli.holdout("trial")
    assert len(calls) == 1
    assert calls[0]["hold_minutes"] == 45.0
    attempts = [row for row in evolve_cli.read_entries()
                if row.get("event") == "holdout_attempt"]
    assert len(attempts) == 1
    assert attempts[0]["params"]["hold_minutes"] == 45.0
    results = [row for row in evolve_cli.read_entries()
               if row.get("event") == "holdout_result"]
    assert results[0]["params"]["hold_minutes"] == 45.0


def test_holdout_treats_a_row_without_hold_as_30_minutes(tmp_path, monkeypatch):
    calls = _seed_holdout(monkeypatch, tmp_path)
    evolve_cli.holdout("trial")
    assert calls[0]["hold_minutes"] == 30.0



def test_report_shows_real_run_settings_including_hold():
    report = render_cycle([], {"maker_fee": .001, "taker_fee": .001, "slippage": .0003,
                               "long_only": True, "stop_pct": 0.03,
                               "min_target_pct": 0.04, "max_leverage": 1.0,
                               "hold_minutes": 45.0})
    assert "stop_pct=0.03" in report
    assert "min_target_pct=0.04" in report
    assert "hold_minutes=45.0" in report
