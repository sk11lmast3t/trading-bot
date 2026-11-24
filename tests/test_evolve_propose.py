import os

import pytest

import evolve.evaluate as evaluate
import evolve.propose as propose
import evolve.__main__ as evolve_cli


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, *, prompt, max_tokens):
        self.calls.append((prompt, max_tokens))
        return self.response, {"total_tokens": 120}


def _valid_code():
    return """class Candidate(BaseStrategy):
    def analyze(self, df):
        return TradeSetup(Signal.HOLD)
"""


def test_build_prompt_documents_candidate_runtime_restrictions(tmp_path, monkeypatch):
    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")

    prompt = propose.build_prompt("trial_test", ledger_path=str(ledger))

    normalized = " ".join(prompt.split())
    assert "class header MUST be exactly `class Candidate(BaseStrategy):`" in normalized
    assert "A class with no base or with any other base is rejected by the validator." in normalized
    assert "must NOT define __init__ or any other dunder name" in normalized
    assert "__init__ is provided by the evaluation harness; do not define it" in prompt
    assert "BaseStrategy, TradeSetup, and Signal are already defined in the module" in normalized
    assert "do NOT import them" in prompt
    for module in ("numpy", "pandas", "core.indicators"):
        assert module in prompt
    assert "import numpy as np" in prompt
    assert "import pandas as pd" in prompt
    assert "from core.indicators import Indicators" in prompt
    assert (
        "abs, all, any, bool, dict, enumerate, Exception, float, int, isinstance,"
        " len, list, max, min, object, range, set, str, sum, super, tuple,"
        " ValueError, zip"
    ) in normalized
    assert "round, sorted, print" in prompt


def test_proposal_uses_fake_client_and_writes_one_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "search journal summary")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "| 2025-01 | search only |")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    client = FakeLLM("```python\n" + _valid_code() + "```\n")

    path, spent, prompt = propose.propose_candidate(
        "trial_test", llm_client=client, token_budget=20000, ledger_path=str(ledger),
    )

    assert os.path.isfile(path)
    assert not (tmp_path / "evolve" / "rejected" / "trial_test.txt").exists()
    assert os.listdir(evaluate.CANDIDATE_DIR) == ["trial_test.py"]
    assert spent == 120
    assert len(client.calls) == 1
    assert client.calls[0][1] > 0
    assert "search journal summary" in prompt
    assert "search only" in prompt
    assert "DESIGN CONSTRAINTS" in prompt
    assert "2.0000%" in prompt
    assert "5.0000%" in prompt
    assert "1x" in prompt
    assert "completed bars only" in prompt
    assert "force-closed 30 minutes after entry" in prompt
    assert "Holds up to several days" not in prompt


def test_rejected_proposal_saves_raw_response_and_reraises_candidate_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    raw_response = "```python\nimport core.types\n```\n"
    client = FakeLLM(raw_response)

    with pytest.raises(evaluate.CandidateError, match="Candidate import is not allowed: core.types"):
        propose.propose_candidate("rejected_trial", llm_client=client, token_budget=20000,
                                  ledger_path=str(ledger))

    rejected = tmp_path / "evolve" / "rejected" / "rejected_trial.txt"
    assert rejected.read_text(encoding="utf-8") == raw_response


def test_api_key_is_read_only_from_llm_env_file(tmp_path):
    env_file = tmp_path / ".llm_env"

    with pytest.raises(RuntimeError, match="only in evolve/.llm_env"):
        propose.llm_config(str(env_file), {"LLM_API_KEY": "ambient-secret",
                                           "LLM_BASE_URL": "http://localhost/v1",
                                           "LLM_MODEL": "fake"})

    env_file.write_text("LLM_API_KEY=file-secret\n", encoding="utf-8")
    result = propose.llm_config(str(env_file), {"LLM_BASE_URL": "http://localhost/v1",
                                                 "LLM_MODEL": "fake"})
    assert result["api_key"] == "file-secret"


def test_build_prompt_reads_the_data_dir_it_is_given(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(propose, "discover_splits", lambda path: seen.append(path) or {})
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")

    propose.build_prompt("trial_test", ledger_path=str(ledger), data_dir="some_data_dir")

    assert seen == ["some_data_dir"]


def test_propose_candidate_forwards_data_dir_to_build_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(propose, "build_prompt", lambda name, **kwargs: seen.update(kwargs) or "tiny")

    with pytest.raises(RuntimeError, match="hard cycle token budget"):
        propose.propose_candidate("trial_test", llm_client=object(), token_budget=100,
                                  data_dir="some_data_dir")

    assert seen.get("data_dir") == "some_data_dir"


def test_proposal_refuses_budget_overrun_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "build_prompt", lambda *_args, **_kwargs: "tiny")
    client = FakeLLM(_valid_code())

    with pytest.raises(RuntimeError, match="hard cycle token budget"):
        propose.propose_candidate("trial_test", llm_client=client, token_budget=100,
                                  already_spent=0)

    assert not os.path.exists(evaluate.CANDIDATE_DIR)


def test_second_holdout_is_refused_and_promotion_requires_pass_and_holdout(tmp_path, monkeypatch):
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
    evolve_cli.promote("trial")
    assert "trial" in (tmp_path / "config" / "paper_candidates.json").read_text()


def _write_baseline(path, tag):
    rows = ["### Net bps per trade by month - all_taker",
            "| Month | `" + tag + "` |", "|---|---|", "| 2025-01 | -1.0 |", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_baseline_table_uses_only_the_named_report(tmp_path, monkeypatch):
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_baseline(reports / "baseline_kucoin_real.md", "named_report")
    _write_baseline(reports / "baseline_zzz_later.md", "later_report")

    text = propose._baseline_search_tables({"2025-01"})

    assert "Source: baseline_kucoin_real.md" in text
    assert "named_report" in text
    assert "later_report" not in text


def test_baseline_table_has_no_fallback_to_another_report(tmp_path, monkeypatch):
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_baseline(reports / "baseline_zzz_later.md", "later_report")

    text = propose._baseline_search_tables({"2025-01"})

    assert text == "No checked-in baseline monthly table was available."


def test_build_prompt_states_the_hold_time_and_not_multi_day_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")

    prompt = propose.build_prompt("trial_test", ledger_path=str(ledger), hold_minutes=45)

    assert "45 minutes after entry" in prompt
    assert "Holds up to several days" not in prompt


def test_propose_candidate_forwards_hold_minutes_to_build_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(propose, "build_prompt", lambda name, **kwargs: seen.update(kwargs) or "tiny")

    with pytest.raises(RuntimeError, match="hard cycle token budget"):
        propose.propose_candidate("trial_test", llm_client=object(), token_budget=100,
                                  hold_minutes=45)

    assert seen.get("hold_minutes") == 45


def _stub_prompt_inputs(monkeypatch):
    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")


def _empty_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    return str(ledger)


def test_build_prompt_includes_research_idea_as_untrusted_data(tmp_path, monkeypatch):
    _stub_prompt_inputs(monkeypatch)

    prompt = propose.build_prompt("trial_test", ledger_path=_empty_ledger(tmp_path),
                                  research_idea="Buy pullbacks in an uptrend.")

    assert "Buy pullbacks in an uptrend." in prompt
    assert "untrusted data" in prompt
    assert "DESIGN CONSTRAINTS win" in prompt
    assert prompt.index("DESIGN CONSTRAINTS\n") < prompt.index("Buy pullbacks")


def test_build_prompt_without_idea_is_unchanged(tmp_path, monkeypatch):
    _stub_prompt_inputs(monkeypatch)
    ledger = _empty_ledger(tmp_path)

    base = propose.build_prompt("trial_test", ledger_path=ledger)
    explicit_none = propose.build_prompt("trial_test", ledger_path=ledger, research_idea=None)
    with_idea = propose.build_prompt("trial_test", ledger_path=ledger, research_idea="x")

    assert explicit_none == base
    assert "RESEARCH IDEA" not in base
    assert with_idea.replace(propose._idea_block("x"), "", 1) == base


def test_load_research_idea_rejects_empty_and_oversized(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    big = tmp_path / "big.txt"
    big.write_text("a" * (propose.MAX_IDEA_CHARS + 1), encoding="utf-8")
    exact = tmp_path / "exact.txt"
    exact.write_text("a" * propose.MAX_IDEA_CHARS, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        propose.load_research_idea(str(empty))
    with pytest.raises(ValueError, match="longer than"):
        propose.load_research_idea(str(big))
    assert propose.load_research_idea(str(exact)) == "a" * propose.MAX_IDEA_CHARS


def test_research_idea_still_hits_the_holdout_safety_check(tmp_path, monkeypatch):
    _stub_prompt_inputs(monkeypatch)

    with pytest.raises(RuntimeError, match="forbidden holdout content"):
        propose.build_prompt("trial_test", ledger_path=_empty_ledger(tmp_path),
                             research_idea="use holdout_result here")


def test_propose_candidate_forwards_research_idea_only_when_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(propose, "build_prompt", lambda name, **kwargs: seen.update(kwargs) or "tiny")

    with pytest.raises(RuntimeError, match="hard cycle token budget"):
        propose.propose_candidate("trial_test", llm_client=object(), token_budget=100,
                                  research_idea="an idea")
    assert seen.get("research_idea") == "an idea"

    seen.clear()
    with pytest.raises(RuntimeError, match="hard cycle token budget"):
        propose.propose_candidate("trial_test", llm_client=object(), token_budget=100)
    assert "research_idea" not in seen


def test_cycle_rejects_oversized_idea_file_before_any_llm_call(tmp_path, monkeypatch):
    big = tmp_path / "idea.txt"
    big.write_text("a" * (propose.MAX_IDEA_CHARS + 1), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("LLM proposal must not be reached")

    monkeypatch.setattr(evolve_cli, "propose_candidate", fail)

    with pytest.raises(ValueError, match="longer than"):
        evolve_cli.main(["cycle", "--max-candidates", "1", "--idea-file", str(big)])
