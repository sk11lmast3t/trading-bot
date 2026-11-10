"""Isolated, cost-aware candidate evaluation through the existing Backtester."""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from evolve.splits import DataSplit, discover_splits, load_partition


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_DIR = os.path.join(ROOT, "strategies", "candidates")
DEFAULT_COSTS = {"maker_fee": 0.0002, "taker_fee": 0.0005, "slippage": 0.0003}
ALLOWED_IMPORTS = {"numpy", "pandas", "core.indicators"}
LOOKAHEAD_CALLS = {"bfill", "backfill"}
BLOCKED_CALLS = {"getattr", "setattr", "delattr", "globals", "locals", "vars",
                 "type", "dir", "breakpoint", "input"}
PANDAS_IO_CALLS = {"to_csv", "to_pickle", "to_parquet", "to_json", "to_excel"}
NUMPY_IO_CALLS = {"load", "save", "savez", "loadtxt", "fromfile", "tofile", "memmap"}


class CandidateError(ValueError):
    pass


def candidate_path(name):
    if not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise CandidateError("Candidate name may contain only letters, numbers, underscore, and hyphen")
    path = os.path.realpath(os.path.join(CANDIDATE_DIR, name + ".py"))
    if os.path.commonpath([os.path.realpath(CANDIDATE_DIR), path]) != os.path.realpath(CANDIDATE_DIR):
        raise CandidateError("Candidate path escapes strategies/candidates")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def validate_candidate_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CandidateError(f"Candidate syntax error: {exc}") from exc
    class_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateError(f"Candidate cannot access dunder name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise CandidateError(f"Candidate cannot access dunder attribute: {node.attr}")
            if node.attr.startswith("read_") or node.attr in PANDAS_IO_CALLS | NUMPY_IO_CALLS:
                raise CandidateError(f"Candidate I/O attribute is not allowed: {node.attr}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("__"):
            raise CandidateError(f"Candidate cannot define dunder name: {node.name}")
        if isinstance(node, ast.arg) and node.arg.startswith("__"):
            raise CandidateError(f"Candidate cannot define dunder argument: {node.arg}")
        if isinstance(node, ast.alias):
            for alias_name in (node.name, node.asname or ""):
                if alias_name.startswith("__"):
                    raise CandidateError(f"Candidate cannot import dunder name: {alias_name}")
        if isinstance(node, ast.keyword) and node.arg and node.arg.startswith("__"):
            raise CandidateError(f"Candidate cannot use dunder keyword: {node.arg}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            is_dunder = value.startswith("__")
            is_io_name = value.startswith("read_") or value in PANDAS_IO_CALLS | NUMPY_IO_CALLS
            if value in BLOCKED_CALLS or is_dunder or is_io_name:
                raise CandidateError(f"Candidate string constant is not allowed: {value}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(" as ", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise CandidateError(f"Candidate import is not allowed: {root}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise CandidateError(f"Candidate import is not allowed: {node.module}")
        elif isinstance(node, ast.ClassDef):
            if any(isinstance(base, ast.Name) and base.id == "BaseStrategy" for base in node.bases):
                class_names.add(node.name)
        elif isinstance(node, ast.Call):
            called_name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if called_name in BLOCKED_CALLS:
                raise CandidateError(f"Candidate call is not allowed: {called_name}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in LOOKAHEAD_CALLS:
                raise CandidateError(f"Potential look-ahead operation is not allowed: {node.func.attr}")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
                if node.args and isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub):
                    raise CandidateError("Potential look-ahead operation shift(-n) is not allowed")
                for kw in node.keywords:
                    if kw.arg == "periods" and isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                        raise CandidateError("Potential look-ahead operation shift(periods=-n) is not allowed")
                    if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        raise CandidateError("Centered rolling windows can look ahead")
        elif isinstance(node, ast.Name) and node.id in {"open", "eval", "exec", "compile", *BLOCKED_CALLS}:
            if isinstance(node.ctx, ast.Load):
                raise CandidateError(f"Candidate cannot access {node.id}")
    if len(class_names) != 1:
        raise CandidateError("Candidate must define exactly one BaseStrategy subclass")
    return next(iter(class_names))


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"Candidate import is not allowed: {name}")
    return __import__(name, globals, locals, fromlist, 0)


def load_candidate(name, config):
    path = candidate_path(name)
    with open(path, encoding="utf-8") as stream:
        source = stream.read()
    class_name = validate_candidate_source(source)
    from core.strategies import BaseStrategy, Signal, TradeSetup
    import builtins
    safe_builtins = {
        key: getattr(builtins, key) for key in (
            "__build_class__", "abs", "all", "any", "bool", "dict", "enumerate",
            "Exception", "float", "int", "isinstance", "len", "list", "max",
            "min", "object", "range", "set", "str", "sum", "super", "tuple",
            "ValueError", "zip",
        )
    }
    safe_builtins["__import__"] = _safe_import
    namespace = {
        "__builtins__": safe_builtins, "__name__": "evolve_candidate",
        "BaseStrategy": BaseStrategy, "TradeSetup": TradeSetup, "Signal": Signal,
    }
    exec(compile(source, path, "exec"), namespace, namespace)
    cls = namespace[class_name]
    strategy = cls(config)
    if not callable(getattr(strategy, "analyze", None)):
        raise CandidateError("Candidate strategy must implement analyze(df)")
    return strategy, source


class _LongOnlyStrategy:
    """Thin signal filter for evaluator runs; leaves Backtester unchanged."""

    def __init__(self, wrapped, signal, setup_type):
        self.wrapped = wrapped
        self.signal = signal
        self.setup_type = setup_type
        self.name = getattr(wrapped, "name", "candidate")
        self.lookback_bars = getattr(wrapped, "lookback_bars", 300)

    def analyze(self, frame):
        setup = self.wrapped.analyze(frame)
        if setup.signal == self.signal.SHORT:
            return self.setup_type(self.signal.HOLD)
        return setup


def long_only_strategy(strategy):
    from core.strategies import Signal, TradeSetup
    return _LongOnlyStrategy(strategy, Signal, TradeSetup)


def _setup_signature(setup):
    signal = getattr(getattr(setup, "signal", None), "value", getattr(setup, "signal", None))
    return (str(signal),) + tuple(
        None if getattr(setup, key, None) is None else float(getattr(setup, key))
        for key in ("entry_price", "stop_loss", "take_profit", "confidence")
    )


def check_truncated_overlap(strategy_factory, frame, lookback=300, max_points=200):
    """Require identical signals for shared timestamps in full and truncated inputs."""
    if len(frame) < 3:
        return True
    cutoff = max(2, int(len(frame) * 0.8))
    truncated = frame.iloc[:cutoff]
    start = max(1, cutoff - max_points)

    def trace(source):
        strategy = strategy_factory()
        result = {}
        first = max(0, start - max(lookback, 1))
        for end in range(first + 1, len(source) + 1):
            window = source.iloc[max(0, end - lookback):end].copy()
            setup = strategy.analyze(window)
            ts = source.index[end - 1]
            if end > start:
                result[ts] = _setup_signature(setup)
        return result

    full_trace, prefix_trace = trace(frame), trace(truncated)
    common = set(full_trace) & set(prefix_trace)
    return bool(common) and all(full_trace[ts] == prefix_trace[ts] for ts in common)


def _metrics(trades, reps=2000, seed=7, alpha=0.05):
    values = np.asarray([float(t.pnl) / float(t.notional) * 10000.0
                         for t in trades if float(t.notional) > 0], dtype=float)
    if not len(values):
        return {"trades": 0, "net_bps_per_trade": None, "bootstrap_95_ci": [None, None],
                "adjusted_ci": [None, None], "profitable_months": 0, "months": 0,
                "profitable_symbols": 0, "symbols": 0, "exit_reasons": {}}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    alpha = min(max(float(alpha), 1e-6), 0.5)
    low, high = np.quantile(samples, [0.025, 0.975])
    adjusted_low, adjusted_high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    month_values = defaultdict(list)
    symbol_values = defaultdict(list)
    reasons = Counter()
    for trade in trades:
        if float(trade.notional) <= 0:
            continue
        bps = float(trade.pnl) / float(trade.notional) * 10000.0
        month_values[pd.Timestamp(trade.entry_time).strftime("%Y-%m")].append(bps)
        symbol_values[str(trade.symbol)].append(bps)
        reasons[str(trade.exit_reason)] += 1
    month_means = {key: float(np.mean(val)) for key, val in sorted(month_values.items())}
    symbol_means = {key: float(np.mean(val)) for key, val in sorted(symbol_values.items())}
    return {
        "trades": len(values), "net_bps_per_trade": float(values.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "adjusted_ci": [float(adjusted_low), float(adjusted_high)],
        "profitable_months": sum(value > 0 for value in month_means.values()),
        "months": len(month_means), "month_net_bps": month_means,
        "profitable_symbols": sum(value > 0 for value in symbol_means.values()),
        "symbols": len(symbol_means), "symbol_net_bps": symbol_means,
        "per_symbol": {key: {"trades": len(symbol_values[key]), "net_bps_per_trade": symbol_means[key]}
                       for key in symbol_means},
        "exit_reasons": dict(sorted(reasons.items())),
    }


def load_costs(maker_fee=None, taker_fee=None, slippage=None, costs_path="config/measured_costs.json"):
    costs = dict(DEFAULT_COSTS)
    if os.path.isfile(costs_path):
        try:
            with open(costs_path, encoding="utf-8") as stream:
                measured = json.load(stream)
            for key in costs:
                if key in measured:
                    costs[key] = float(measured[key])
            symbol_costs = measured.get("symbols", {})
            weighted = [(float(item["slippage"]), int(item.get("trades", 0)))
                        for item in symbol_costs.values()
                        if isinstance(item, dict) and item.get("slippage") is not None
                        and int(item.get("trades", 0)) > 0]
            if weighted:
                costs["slippage"] = sum(value * count for value, count in weighted) / sum(count for _, count in weighted)
        except (OSError, ValueError, TypeError):
            pass
    for key, value in (("maker_fee", maker_fee), ("taker_fee", taker_fee), ("slippage", slippage)):
        if value is not None:
            costs[key] = float(value)
    if any(not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError("Fees and slippage must be finite non-negative fractions")
    return costs


def evaluate_candidate(name, partition="search", data_dir="data", fees=None,
                       hold_minutes=30.0, timeout=1800, trials=1, stress_factor=1.0,
                       long_only=False):
    """Evaluate in a network-disabled child process with a hard timeout."""
    if partition not in ("search", "validation", "holdout"):
        raise ValueError(partition)
    path = candidate_path(name)
    with open(path, encoding="utf-8") as stream:
        code = stream.read()
    validate_candidate_source(code)
    splits = discover_splits(data_dir)
    payload = {
        "name": name, "partition": partition, "data_dir": os.path.abspath(data_dir),
        "splits": {symbol: {"path": split.path,
                  "search": [[str(a), str(b)] for a, b in split.search],
                  "validation": [[str(a), str(b)] for a, b in split.validation],
                  "holdout": [[str(a), str(b)] for a, b in split.holdout]}
                  for symbol, split in splits.items()},
        "fees": fees or DEFAULT_COSTS, "hold_minutes": float(hold_minutes),
        "trials": max(1, int(trials)), "stress_factor": float(stress_factor),
        "long_only": bool(long_only),
    }
    with tempfile.TemporaryDirectory(prefix="evolve_eval_") as td:
        payload_path = os.path.join(td, "payload.json")
        with open(payload_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        worker = (
            "import os,socket,runpy,sys; "
            "os.environ['PYTHON_DOTENV_DISABLED']='1'; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled')); "
            "sys.argv=['evolve.evaluate','--worker',sys.argv[1]]; "
            "runpy.run_module('evolve.evaluate',run_name='__main__')"
        )
        env = {k: v for k, v in os.environ.items()
               if k not in {"LLM_API_KEY", "API_KEY", "API_SECRET"}}
        env["PYTHON_DOTENV_DISABLED"] = "1"
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                [sys.executable, "-c", worker, payload_path], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Candidate evaluation exceeded {timeout}s") from exc
        if result.returncode:
            raise CandidateError(f"Candidate worker failed: {result.stderr[-2000:]}")
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise CandidateError(f"Candidate worker returned invalid output: {result.stdout[-1000:]}") from exc


def _worker(payload_path):
    # Network is blocked before candidate import/execution.
    import socket
    def denied(*_args, **_kwargs):
        raise RuntimeError("network disabled during candidate evaluation")
    socket.socket.connect = denied
    socket.create_connection = denied
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    with open(payload_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    from config.settings import BotConfig
    from core.backtester import Backtester
    from core.strategies import Signal

    all_results = {}
    trials = payload["trials"]
    adjusted_alpha = 0.05 / trials
    for cost_multiplier, label in ((1.0, "base"), (payload["stress_factor"], "stress")):
        fees = payload["fees"]
        cfg = BotConfig()
        cfg.execution.entry_mode = "market"
        cfg.backtest.initial_balance = 1000.0
        cfg.backtest.maker_fee = float(fees["maker_fee"]) * cost_multiplier
        cfg.backtest.taker_fee = float(fees["taker_fee"]) * cost_multiplier
        cfg.backtest.slippage = float(fees["slippage"]) * cost_multiplier
        cfg.risk.max_drawdown_pct = 1.0
        cfg.risk.max_daily_loss_pct = 1.0
        cfg.risk.max_consecutive_losses = 10 ** 9
        cfg.risk.daily_trade_limit = 0
        cfg.risk.circuit_breaker_enabled = False
        cfg.risk.reduce_size_on_loss = False
        cfg.risk.max_hold_time_minutes = payload["hold_minutes"]
        cfg.strategy.active_strategy = "ema_rsi_atr"
        merged_trades = []
        for symbol, spec in payload["splits"].items():
            partition_windows = [tuple(pd.Timestamp(x) for x in pair) for pair in spec[payload["partition"]]]
            split = DataSplit(
                symbol, spec["path"],
                tuple(tuple(pd.Timestamp(x) for x in pair) for pair in spec["search"]),
                tuple(tuple(pd.Timestamp(x) for x in pair) for pair in spec["validation"]),
                tuple(tuple(pd.Timestamp(x) for x in pair) for pair in spec["holdout"]),
            )
            frame = load_partition(split, payload["partition"])
            probe, _ = load_candidate(payload["name"], cfg)
            warm = max(300, int(getattr(probe, "lookback_bars", 0)))
            if not check_truncated_overlap(lambda: load_candidate(payload["name"], cfg)[0], frame, warm):
                raise CandidateError(f"Look-ahead check failed on shared {symbol} signal bars")
            for start, end in partition_windows:
                i0, i1 = frame.index.searchsorted(start), frame.index.searchsorted(end)
                segment = frame.iloc[max(0, i0 - warm):i1].copy()
                bt = Backtester(cfg, max_hold_minutes=payload["hold_minutes"])
                bt.strategy, _ = load_candidate(payload["name"], cfg)
                if payload.get("long_only"):
                    bt.strategy = long_only_strategy(bt.strategy)
                bt.run(segment, symbol=symbol, trade_from=start)
                merged_trades.extend(t for t in bt.trades if t.exit_reason != "END")
        all_results[label] = _metrics(merged_trades, alpha=adjusted_alpha)
    return {"candidate": payload["name"], "partition": payload["partition"],
            "base": all_results["base"], "stress": all_results["stress"],
            "costs": payload["fees"], "stress_factor": payload["stress_factor"]}


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    print(json.dumps(_worker(args.worker), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()
