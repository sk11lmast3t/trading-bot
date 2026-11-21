import pytest

import evolve.__main__ as evolve_cli
import evolve.evaluate as evaluate
import evolve.ledger as ledger
import evolve.propose as propose


VALID_CANDIDATE = """class Candidate(BaseStrategy):
    def analyze(self, df):
        return TradeSetup(Signal.HOLD)
"""


class StubLLM:
    def __init__(self, interrupt=False):
        self.interrupt = interrupt

    def complete(self, *, prompt, max_tokens):
        if self.interrupt:
            raise KeyboardInterrupt
        return VALID_CANDIDATE, {"total_tokens": 10}


@pytest.fixture
def isolated_cycle(tmp_path, monkeypatch):
    candidate_dir = tmp_path / "strategies" / "candidates"
    candidate_dir.mkdir(parents=True)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    reports = []
    evaluation_calls = []
    learn_cost_calls = []

    monkeypatch.setattr(evaluate, "CANDIDATE_DIR", str(candidate_dir))
    monkeypatch.setattr(evolve_cli, "read_entries",
                        lambda: ledger.read_entries(str(ledger_path)))
    monkeypatch.setattr(evolve_cli, "append_entry",
                        lambda row: ledger.append_entry(row, str(ledger_path)))
    monkeypatch.setattr(evolve_cli, "learn_costs", lambda: learn_cost_calls.append(True))
    monkeypatch.setattr(evolve_cli, "load_costs", lambda *args: {
        "maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0003,
    })
    def fake_evaluate_candidate(name, partition, **kwargs):
        evaluation_calls.append({"name": name, "partition": partition, **kwargs})
        return {}
    monkeypatch.setattr(evolve_cli, "evaluate_candidate", fake_evaluate_candidate)
    monkeypatch.setattr(evolve_cli, "validation_verdict", lambda *args: {
        "verdict": "failed", "reason_failed": [],
    })
    monkeypatch.setattr(evolve_cli, "constraint_failures", lambda *args: [])
    monkeypatch.setattr(evolve_cli, "write_cycle", lambda rows, **kwargs: (
        reports.append(rows) or str(tmp_path / "cycle-report.md")))

    monkeypatch.setattr(propose, "_search_months", lambda data_dir=None: ({"2025-01"}, {}))
    monkeypatch.setattr(propose, "_journal_summary", lambda *_: "")
    monkeypatch.setattr(propose, "_baseline_search_tables", lambda *_: "")

    def use_stub_client(client):
        def propose_with_stub(name, **kwargs):
            kwargs["llm_client"] = client
            kwargs["ledger_path"] = str(ledger_path)
            return propose.propose_candidate(name, **kwargs)
        monkeypatch.setattr(evolve_cli, "propose_candidate", propose_with_stub)

    return candidate_dir, ledger_path, reports, use_stub_client, evaluation_calls, learn_cost_calls


def test_cycle_skips_preexisting_candidate_file(isolated_cycle, capsys):
    candidate_dir, ledger_path, reports, use_stub_client, _, _ = isolated_cycle
    ledger.append_entry({"event": "candidate", "name": "older"}, str(ledger_path))
    (candidate_dir / "evolve_0002.py").write_text("occupied", encoding="utf-8")
    use_stub_client(StubLLM())

    records = evolve_cli.cycle(max_candidates=1, token_budget=1000)

    assert [row["name"] for row in records] == ["evolve_0003"]
    assert (candidate_dir / "evolve_0003.py").is_file()
    assert "Skipping candidate name evolve_0002" in capsys.readouterr().out
    assert len(reports) == 1


def test_multi_candidate_cycle_keeps_names_unique_around_existing_file(isolated_cycle):
    candidate_dir, _ledger_path, _reports, use_stub_client, _, _ = isolated_cycle
    (candidate_dir / "evolve_0001.py").write_text("occupied", encoding="utf-8")
    use_stub_client(StubLLM())

    records = evolve_cli.cycle(max_candidates=3, token_budget=3000)

    names = [row["name"] for row in records]
    assert names == ["evolve_0002", "evolve_0003", "evolve_0004"]
    assert len(names) == len(set(names))
    assert all((candidate_dir / f"{name}.py").is_file() for name in names)


def test_cycle_interrupt_records_failure_and_reraises_without_report(isolated_cycle):
    _candidate_dir, ledger_path, reports, use_stub_client, _, _ = isolated_cycle
    use_stub_client(StubLLM(interrupt=True))

    with pytest.raises(KeyboardInterrupt):
        evolve_cli.cycle(max_candidates=1, token_budget=1000)

    rows = ledger.read_entries(str(ledger_path))
    assert len(rows) == 1
    assert rows[0]["name"] == "evolve_0001"
    assert rows[0]["verdict"] == "failed"
    assert rows[0]["reason_failed"] == [
        "cycle error: KeyboardInterrupt: run interrupted"
    ]
    assert reports == []


def test_cycle_passes_data_dir_to_the_proposer(isolated_cycle, monkeypatch):
    _, _, _, use_stub_client, _, _ = isolated_cycle
    use_stub_client(StubLLM())
    seen = []
    real_propose_candidate = evolve_cli.propose_candidate

    def spy(name, **kwargs):
        seen.append(kwargs.get("data_dir"))
        return real_propose_candidate(name, **kwargs)

    monkeypatch.setattr(evolve_cli, "propose_candidate", spy)
    evolve_cli.cycle(max_candidates=1, token_budget=1000, data_dir="some_data_dir")

    assert seen == ["some_data_dir"]


def test_cycle_default_eval_timeout_is_passed_to_search_and_validation(isolated_cycle):
    _candidate_dir, _ledger_path, _reports, use_stub_client, evaluation_calls, _ = isolated_cycle
    use_stub_client(StubLLM())

    evolve_cli.cycle(max_candidates=1, token_budget=1000)

    assert [call["partition"] for call in evaluation_calls] == ["search", "validation"]
    assert [call["timeout"] for call in evaluation_calls] == [1800, 1800]


def test_configured_eval_timeout_is_passed_to_both_evaluations_and_cli(isolated_cycle,
                                                                       monkeypatch):
    _candidate_dir, _ledger_path, _reports, use_stub_client, evaluation_calls, _ = isolated_cycle
    use_stub_client(StubLLM())

    evolve_cli.cycle(max_candidates=1, token_budget=1000, eval_timeout=14400)

    assert [call["timeout"] for call in evaluation_calls] == [14400, 14400]

    forwarded = []
    monkeypatch.setattr(evolve_cli, "cycle", lambda *args, **kwargs: forwarded.append(kwargs))
    evolve_cli.main(["cycle"])
    evolve_cli.main(["cycle", "--eval-timeout", "14400"])
    assert [call["eval_timeout"] for call in forwarded] == [1800, 14400]


@pytest.mark.parametrize("eval_timeout", [0, -5])
def test_invalid_eval_timeout_fails_before_work_or_ledger_write(isolated_cycle, eval_timeout):
    _candidate_dir, ledger_path, _reports, _use_stub_client, _evaluation_calls, learn_calls = isolated_cycle

    with pytest.raises(ValueError, match="eval_timeout must be a positive integer"):
        evolve_cli.cycle(max_candidates=1, eval_timeout=eval_timeout)

    assert learn_calls == []
    assert ledger.read_entries(str(ledger_path)) == []


def test_hold_minutes_is_passed_to_both_evaluations_and_recorded(isolated_cycle):
    _candidate_dir, _ledger_path, _reports, use_stub_client, evaluation_calls, _ = isolated_cycle
    use_stub_client(StubLLM())

    records = evolve_cli.cycle(max_candidates=1, token_budget=1000, hold_minutes=120)

    assert [call["hold_minutes"] for call in evaluation_calls] == [120.0, 120.0]
    assert records[0]["params"]["hold_minutes"] == 120.0


def test_default_hold_minutes_is_30(isolated_cycle):
    _candidate_dir, _ledger_path, _reports, use_stub_client, evaluation_calls, _ = isolated_cycle
    use_stub_client(StubLLM())

    records = evolve_cli.cycle(max_candidates=1, token_budget=1000)

    assert [call["hold_minutes"] for call in evaluation_calls] == [30.0, 30.0]
    assert records[0]["params"]["hold_minutes"] == 30.0


@pytest.mark.parametrize("hold_minutes", [0, -5, float("nan"), float("inf"), True])
def test_invalid_hold_minutes_fails_before_any_work(isolated_cycle, hold_minutes):
    _candidate_dir, _ledger_path, _reports, _use_stub_client, evaluation_calls, learn_calls = isolated_cycle

    with pytest.raises(ValueError, match="hold_minutes must be a positive finite number"):
        evolve_cli.cycle(max_candidates=1, hold_minutes=hold_minutes)

    assert learn_calls == []
    assert evaluation_calls == []


def test_hold_minutes_does_not_change_the_trial_settings():
    base = {"event": "candidate", "maker_fee": 0.001, "taker_fee": 0.001, "slippage": 0.0003,
            "long_only": True, "fill_model_version": "v"}
    one = {**base, "params": {"hold_minutes": 60.0}}
    two = {**base, "params": {"hold_minutes": 120.0}}

    assert evolve_cli._row_settings(one) == evolve_cli._row_settings(two)


def test_cycle_passes_hold_minutes_to_the_proposer(isolated_cycle, monkeypatch):
    _, _, _, use_stub_client, _, _ = isolated_cycle
    use_stub_client(StubLLM())
    seen = []
    real_propose_candidate = evolve_cli.propose_candidate

    def spy(name, **kwargs):
        seen.append(kwargs.get("hold_minutes"))
        return real_propose_candidate(name, **kwargs)

    monkeypatch.setattr(evolve_cli, "propose_candidate", spy)
    evolve_cli.cycle(max_candidates=1, token_budget=1000, hold_minutes=90)

    assert seen == [90.0]
