import pytest

import evolve.__main__ as evolve_cli
from evolve.report import render_cycle


def test_trial_counts_are_isolated_by_effective_settings():
    entries = [
        {"event": "candidate", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": False},
        {"event": "candidate", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": True},
        {"event": "holdout_result", "maker_fee": .0002, "taker_fee": .0005,
         "slippage": .0003, "long_only": False},
    ]
    assert evolve_cli._trial_count(entries, {
        "maker_fee": .0002, "taker_fee": .0005, "slippage": .0003,
        "long_only": False,
    }) == 1


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
