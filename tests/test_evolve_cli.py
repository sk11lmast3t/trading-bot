import pytest

import evolve.__main__ as evolve_cli


def test_holdout_is_single_attempt_and_promotion_needs_validation_and_holdout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evolve").mkdir()
    with pytest.raises(RuntimeError, match="passed validation"):
        evolve_cli.promote("trial")
    evolve_cli.append_entry({"event": "candidate", "name": "trial", "verdict": "passed"})
    with pytest.raises(RuntimeError, match="single holdout"):
        evolve_cli.promote("trial")
    monkeypatch.setattr(evolve_cli, "evaluate_candidate", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(evolve_cli, "load_costs", lambda: {})
    evolve_cli.holdout("trial")
    with pytest.raises(RuntimeError, match="already attempted"):
        evolve_cli.holdout("trial")
    assert any(row.get("event") == "holdout_refused" for row in evolve_cli.read_entries())
    evolve_cli.promote("trial")
    assert "trial" in (tmp_path / "config" / "paper_candidates.json").read_text()
