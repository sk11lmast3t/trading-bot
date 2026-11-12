import csv
import json

from evolve.costs import learn_costs
from evolve.evaluate import load_costs


def test_cost_learner_requires_fifty_closed_trades_and_keeps_assumption(tmp_path, capsys):
    journal = tmp_path / "journal.csv"
    output = tmp_path / "measured_costs.json"
    with journal.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ts", "symbol", "strategy", "status", "slippage_bps"])
        writer.writeheader()
        for i in range(49):
            writer.writerow({"ts": str(i), "symbol": "BTC/USDT", "strategy": "x",
                             "status": "closed", "slippage_bps": "4"})

    result = learn_costs(str(journal), str(output), assumed={"maker_fee": .0002, "taker_fee": .0005, "slippage": .0003})

    assert result["symbols"] == {}
    assert result["slippage"] == .0003
    assert "too few" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["symbols"] == {}


def test_cost_learner_measures_slippage_and_fill_rate_at_fifty(tmp_path):
    journal = tmp_path / "journal.csv"
    output = tmp_path / "measured_costs.json"
    with journal.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ts", "symbol", "strategy", "status", "slippage_bps"])
        writer.writeheader()
        for i in range(55):
            writer.writerow({"ts": str(i), "symbol": "BTC/USDT", "strategy": "x",
                             "status": "closed" if i < 50 else "skipped:entry_not_filled",
                             "slippage_bps": "5"})

    result = learn_costs(str(journal), str(output))

    assert result["symbols"]["BTC/USDT"]["measured_slippage_bps"] == 5.0
    assert result["symbols"]["BTC/USDT"]["fill_rate"] == 50 / 55
    assert load_costs(costs_path=str(output))["slippage"] == .0005
