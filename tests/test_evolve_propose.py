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
    return """class Trial(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
    def analyze(self, df):
        return TradeSetup(Signal.HOLD)
"""


def test_proposal_uses_fake_client_and_writes_one_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "_search_months", lambda: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "search journal summary")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "| 2025-01 | search only |")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    client = FakeLLM("```python\n" + _valid_code() + "```\n")

    path, spent, prompt = propose.propose_candidate(
        "trial_test", llm_client=client, token_budget=20000, ledger_path=str(ledger),
    )

    assert os.path.isfile(path)
    assert os.listdir(evaluate.CANDIDATE_DIR) == ["trial_test.py"]
    assert spent == 120
    assert len(client.calls) == 1
    assert client.calls[0][1] > 0
    assert "search journal summary" in prompt
    assert "search only" in prompt


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
    monkeypatch.setattr(evolve_cli, "load_costs", lambda: {})
    evolve_cli.holdout("trial")
    with pytest.raises(RuntimeError, match="already attempted"):
        evolve_cli.holdout("trial")
    evolve_cli.promote("trial")
    assert "trial" in (tmp_path / "config" / "paper_candidates.json").read_text()
