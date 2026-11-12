"""Candidate validation thresholds with a trial-count Bonferroni adjustment."""


def validation_verdict(search, validation, trial_count):
    failures = []
    base = validation.get("base", {})
    stress = validation.get("stress", {})
    if int(base.get("trades", 0)) < 300:
        failures.append(f"validation has {base.get('trades', 0)} trades; requires 300+")
    ci = base.get("adjusted_ci") or [None, None]
    if ci[0] is None or float(ci[0]) <= 0:
        failures.append(f"Bonferroni-adjusted CI lower bound is not above zero (trials={trial_count})")
    months, profitable_months = int(base.get("months", 0)), int(base.get("profitable_months", 0))
    if not months or profitable_months <= months / 2:
        failures.append(f"positive in {profitable_months}/{months} validation months; requires a majority")
    symbols, profitable_symbols = int(base.get("symbols", 0)), int(base.get("profitable_symbols", 0))
    if not symbols or profitable_symbols <= symbols / 2:
        failures.append(f"positive in {profitable_symbols}/{symbols} validation symbols; requires a majority")
    stress_net = stress.get("net_bps_per_trade")
    if stress_net is None or float(stress_net) <= 0:
        failures.append("validation expectancy is not positive at 1.5x costs")
    search_net = (search.get("base") or {}).get("net_bps_per_trade")
    if search_net is None or float(search_net) <= 0:
        failures.append("search expectancy is not positive")
    return {"verdict": "passed" if not failures else "failed", "reason_failed": failures}

