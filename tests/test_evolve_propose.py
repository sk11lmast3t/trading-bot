import os

import numpy as np
import pandas as pd
import pytest

import evolve.evaluate as evaluate
import evolve.propose as propose
import evolve.__main__ as evolve_cli
from config.settings import BotConfig
from core.indicators import Indicators


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
    monkeypatch.setattr(propose, "_search_months", lambda: ({"2025-01"}, {}))
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
    assert "All Indicators methods take pandas Series inputs" in normalized
    assert "Never use .values or NumPy arrays with Indicators" in normalized
    assert ".iloc[-1] or .iloc[-2]" in normalized
    assert "never use [-1] on a pandas Series" in normalized
    for method in (
        "ema", "sma", "rsi", "atr", "macd", "bollinger_bands", "momentum",
        "volume_sma", "vwap", "stochastic_rsi", "disparity",
    ):
        assert f"Indicators.{method}(" in prompt


def test_prompt_rsi_sma_example_validates_and_runs_with_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(propose, "_search_months", lambda: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")

    prompt = propose.build_prompt("trial_example", ledger_path=str(ledger))
    marker = "Runnable RSI + SMA example:\n```python\n"
    assert marker in prompt
    example = prompt.split(marker, 1)[1].split("\n```", 1)[0]
    assert evaluate.validate_candidate_source(example) == "Candidate"

    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    (candidate_dir / "trial_example.py").write_text(example + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(candidate_dir))
    strategy, loaded_source = evaluate.load_candidate("trial_example", BotConfig())

    index = pd.date_range("2026-01-01", periods=40, freq="5min", name="timestamp")
    close = pd.Series(np.linspace(100.0, 104.0, len(index)), index=index)
    raw = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": pd.Series(np.full(len(index), 100.0), index=index),
    }, index=index)
    frame = Indicators.calculate_all(raw, BotConfig())

    setup = strategy.analyze(frame.iloc[-30:].copy())

    assert loaded_source.rstrip("\n") == example
    assert setup is not None


def test_proposal_uses_fake_client_and_writes_one_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
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
    assert "Holds up to several days are allowed" in prompt


def test_rejected_proposal_saves_raw_response_and_reraises_candidate_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(tmp_path / "candidates"))
    monkeypatch.setattr(propose, "ROOT", str(tmp_path))
    monkeypatch.setattr(propose, "_search_months", lambda: ({"2025-01"}, {}))
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
