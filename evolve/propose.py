"""Construct a holdout-free prompt and request one candidate from an LLM."""
from __future__ import annotations

import csv
import json
import os
import re
import urllib.request
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO

from evolve.ledger import prompt_entries
from evolve.splits import discover_splits


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_ENV_PATH = os.path.join(os.path.dirname(__file__), ".llm_env")


def _read_llm_file(path=LLM_ENV_PATH):
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def llm_config(path=LLM_ENV_PATH, environ=None):
    file_values = _read_llm_file(path)
    environ = os.environ if environ is None else environ
    key = file_values.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY must be set only in evolve/.llm_env")
    base_url = environ.get("LLM_BASE_URL") or file_values.get("LLM_BASE_URL")
    model = environ.get("LLM_MODEL") or file_values.get("LLM_MODEL")
    if not base_url or not model:
        raise RuntimeError("Set LLM_BASE_URL and LLM_MODEL in the environment or evolve/.llm_env")
    return {"base_url": base_url.rstrip("/"), "model": model, "api_key": key}


def _search_months():
    splits = discover_splits(os.path.join(ROOT, "data"))
    month_sets = [{start.strftime("%Y-%m") for start, _ in split.search}
                  for split in splits.values()]
    # A shared baseline table aggregates symbols. Include only months that are
    # SEARCH for every symbol so a shorter-history symbol cannot leak later data.
    return (set.intersection(*month_sets) if month_sets else set()), splits


def _baseline_search_tables(months):
    reports = sorted(
        os.path.join(ROOT, "reports", name) for name in os.listdir(os.path.join(ROOT, "reports"))
        if name.startswith("baseline_") and name.endswith(".md")
    ) if os.path.isdir(os.path.join(ROOT, "reports")) else []
    if not reports:
        return "No checked-in baseline monthly table was available."
    for path in reversed(reports):
        with open(path, encoding="utf-8") as stream:
            lines = stream.readlines()
        out = [f"Source: {os.path.basename(path)} (SEARCH months only)"]
        keep = False
        found = False
        for line in lines:
            if line.startswith("### Net bps per trade by month - all_taker"):
                keep = True
                found = True
                out.append(line.rstrip())
                continue
            if keep:
                if line.startswith("### ") or (line.strip() == "" and len(out) > 1):
                    break
                if line.startswith("|"):
                    cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
                    if len(cells) and cells[0] in months or (len(cells) and cells[0] == "Month") or (len(cells) and set(cells[0]) <= {"-", ":"}):
                        out.append(line.rstrip())
        if found:
            return "\n".join(out)
    return "No all-taker baseline monthly table was available."


def _journal_summary(months, splits):
    journal = os.path.join(ROOT, "logs", "journal.csv")
    if not os.path.isfile(journal):
        journal = os.path.join(ROOT, "logs", "decisions.csv")
    if not os.path.isfile(journal):
        return "No trade journal is available."
    # Do not expose journal rows whose timestamps fall into any validation or
    # holdout period; only use dated SEARCH observations.
    split_by_symbol = {symbol.replace("_", "/", 1): split for symbol, split in splits.items()}
    with open(journal, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if row.get("ts")]
    filtered = []
    import pandas as pd
    for row in rows:
        try:
            ts = pd.Timestamp(row["ts"])
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        symbol = (row.get("symbol") or "").replace("_", "/", 1)
        split = split_by_symbol.get(symbol)
        if split and any(start <= ts < end for start, end in split.search):
            filtered.append(row)
    if not filtered:
        return "No journal observations fall inside the SEARCH date range."
    tmp = os.path.join(ROOT, "evolve", ".journal_search_context.csv")
    fields = list(filtered[0])
    with open(tmp, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(filtered)
    try:
        from scripts.journal_report import report
        captured = StringIO()
        with redirect_stdout(captured):
            report(tmp)
        return captured.getvalue().strip()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def build_prompt(name, ledger_path=None, stop_pct=0.02, min_target_pct=0.05,
                 max_leverage=1.0):
    months, splits = _search_months()
    ledger = prompt_entries(ledger_path or os.path.join(ROOT, "evolve", "ledger.jsonl"), 10)
    # Explicitly serialize only the search/validation fields. Holdout results
    # are omitted even if future ledger event formats add them.
    safe_ledger = [{k: v for k, v in entry.items() if k not in {
        "holdout", "holdout_result", "holdout_metrics", "holdout_ci",
    }} for entry in ledger]
    prompt = f"""Propose exactly one new Python candidate named {name}.

Return only source code for that one file. Do not include markdown fences.
The candidate file may import only numpy, pandas, and core.indicators. The
evaluation harness injects BaseStrategy, TradeSetup, and Signal. Define exactly
one BaseStrategy subclass with analyze(df) -> TradeSetup and optional
lookback_bars. Use one fixed rule set. Use no look-ahead and completed bars only.
Do not access the filesystem, environment, network, or subprocesses.

DESIGN CONSTRAINTS
- Stop distance should be about {stop_pct:.4%} of entry price.
- Target distance must be at least {min_target_pct:.4%} of entry price.
- Do not use leverage above {max_leverage:g}x.
- Use completed bars only.
- Use one fixed rule set.
- Holds up to several days are allowed.

Interface:
    class BaseStrategy:
        def __init__(self, config): self.config = config
        def analyze(self, df): return TradeSetup(Signal.HOLD)
TradeSetup(Signal.LONG|Signal.SHORT|Signal.HOLD, entry_price, stop_loss,
           take_profit, confidence, strategy_name)

Trade journal context summaries (SEARCH dates only):
{_journal_summary(months, splits)}

Baseline all-taker report table (SEARCH months only):
{_baseline_search_tables(months)}

Last 10 trial ledger entries (holdout material removed):
{json.dumps(safe_ledger, sort_keys=True)}
"""
    if any(month in prompt for month in ("holdout_result", "secret_metric")):
        raise RuntimeError("Prompt safety check detected forbidden holdout content")
    return prompt


def estimate_tokens(text):
    return max(1, (len(text) + 3) // 4)


def _http_complete(config, prompt, max_tokens, timeout=90):
    url = config["base_url"]
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {"model": config["model"], "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "max_tokens": int(max_tokens)}
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + config["api_key"],
                 "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def _extract_code(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def propose_candidate(name, llm_client=None, token_budget=12000, already_spent=0,
                      ledger_path=None, config_path=LLM_ENV_PATH, stop_pct=0.02,
                      min_target_pct=0.05, max_leverage=1.0):
    prompt = build_prompt(name, ledger_path=ledger_path, stop_pct=stop_pct,
                          min_target_pct=min_target_pct, max_leverage=max_leverage)
    remaining = int(token_budget) - int(already_spent) - estimate_tokens(prompt)
    if remaining < 256:
        raise RuntimeError("hard cycle token budget leaves fewer than 256 response tokens")
    max_tokens = min(4096, remaining)
    if llm_client is None:
        response, usage = _http_complete(llm_config(config_path), prompt, max_tokens)
    else:
        result = llm_client.complete(prompt=prompt, max_tokens=max_tokens)
        if isinstance(result, tuple):
            response, usage = result
        else:
            response, usage = result, {}
    spent = int(usage.get("total_tokens", 0)) if isinstance(usage, dict) else 0
    actual = spent or estimate_tokens(prompt) + estimate_tokens(response)
    if int(already_spent) + actual > int(token_budget):
        raise RuntimeError("LLM response exceeded the hard cycle token budget")
    source = _extract_code(response)
    from evolve.evaluate import CANDIDATE_DIR, validate_candidate_source
    validate_candidate_source(source)
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    path = os.path.join(CANDIDATE_DIR, name + ".py")
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(source.rstrip() + "\n")
    return path, actual, prompt

