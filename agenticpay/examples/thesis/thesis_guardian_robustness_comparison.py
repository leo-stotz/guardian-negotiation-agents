"""Step 5: Main vs. Conservative Guardian robustness comparison.

Diagnostic-only robustness analysis -- NOT a third main setup, NOT used to
alter the frozen Main Guardian prompt. Compares:
    Main:         40 Setup-B runs from agenticpay/results/thesis/main_experiment/
                  (20 Conceder + 20 Boulware), frozen main-experiment sample.
    Conservative: 10 Setup-B runs from agenticpay/results/thesis/guardian_robustness_check/
                  (5 Conceder + 5 Boulware), separate robustness-check batch.

Both were produced by the SAME `thesis_setup_b_test.run_negotiation()`
function under the corrected (post-off-by-one-fix) max_rounds=20 logic,
same scenario, same buyer/buyer-prompt, same seller strategies, same utility
functions, same compliance rules, same deal definition, same model
parameters, same evaluator -- the guardian_robustness_check batch only
redirects the `ThesisGuardian` class to `ConservativeThesisGuardian` via
`unittest.mock.patch.object`, changing nothing else (see
thesis_guardian_robustness_check.py). The only intended difference between
the two datasets is the Guardian system prompt text.

Reads only raw + eval JSON directly from the two source directories -- never
uses old pre-fix calibration/guardian_precision_test data, never overwrites
any existing main-experiment or robustness-check file.

Outputs (agenticpay/results/thesis/guardian_robustness_analysis/, no overwrite):
    guardian_main_vs_conservative_run_level.csv
    guardian_main_vs_conservative_summary.csv
    guardian_main_vs_conservative_by_strategy.csv
    guardian_main_vs_conservative_stats.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sps
from statsmodels.stats.proportion import proportion_confint

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment"
CONS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "guardian_robustness_check"
OUT_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "guardian_robustness_analysis"

RULE_CODES = ["MAX_PRICE_EXCEEDED", "OUT_OF_RANGE", "BELOW_RESERVATION_UTILITY", "INVALID_CONTRACT", "CONFIDENTIAL_INFO_DISCLOSED"]
Z95 = 1.959963984540054


def _no_overwrite(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {path}")


def load_run(directory: Path, run_id: str) -> Dict[str, Any]:
    raw = json.loads((directory / f"{run_id}.json").read_text(encoding="utf-8"))
    ev = json.loads((directory / f"{run_id}_eval.json").read_text(encoding="utf-8"))
    return raw, ev


def build_row(directory: Path, run_id: str, guardian_prompt: str, strategy: str) -> Dict[str, Any]:
    raw, ev = load_run(directory, run_id)
    rr = ev["run_result"]
    rounds = raw.get("rounds") or []
    guardian_calls = sum(1 for r in rounds if r.get("guardian_feedback") is not None)
    revisions = sum(1 for r in rounds if r.get("revision_triggered"))
    by_rule = rr.get("n_violations_by_rule") or {}

    return {
        "run_id": run_id, "guardian_prompt": guardian_prompt, "seller_strategy": strategy,
        "model": rr["model"], "max_rounds": rr["max_rounds"],
        "deal_reached": bool(raw["deal_reached"]),
        "final_round": raw["final_round"], "rounds_played": raw["rounds_played"],
        "fully_compliant_run": bool(rr["fully_compliant_run"]),
        "n_violations_total": rr["n_violations_total"],
        **{code: by_rule.get(code, 0) for code in RULE_CODES},
        "final_buyer_utility": rr["final_buyer_utility"],
        "final_seller_utility": rr["final_seller_utility"],
        "deal_below_buyer_reservation": rr["deal_below_buyer_reservation"],
        "buyer_total_tokens": rr.get("buyer_total_tokens"),
        "guardian_total_tokens": rr.get("guardian_total_tokens"),
        "total_tokens": rr.get("total_tokens"),
        "runtime_sec": rr.get("runtime_sec"),
        "estimated_cost_usd": rr.get("estimated_cost_usd"),
        "guardian_calls": guardian_calls,
        "guardian_revisions": revisions,
        "guardian_revision_rate_run": revisions / guardian_calls if guardian_calls else None,
    }


def load_all() -> List[Dict[str, Any]]:
    rows = []
    for strat in ("conceder", "boulware"):
        for rep in range(1, 21):
            rows.append(build_row(MAIN_DIR, f"setup_b_{strat}_rep{rep:02d}", "main", strat.capitalize()))
    for strat in ("conceder", "boulware"):
        for rep in range(1, 6):
            rows.append(build_row(CONS_DIR, f"conservative_setup_b_{strat}_rep{rep}", "conservative", strat.capitalize()))
    return rows


# =============================================================================
# Comparability check
# =============================================================================

def comparability_check(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    main_rows = [r for r in rows if r["guardian_prompt"] == "main"]
    cons_rows = [r for r in rows if r["guardian_prompt"] == "conservative"]
    lines.append(f"Main: n={len(main_rows)} (expected 40) -> {'OK' if len(main_rows)==40 else 'MISMATCH'}")
    lines.append(f"Conservative: n={len(cons_rows)} (expected 10) -> {'OK' if len(cons_rows)==10 else 'MISMATCH'}")
    for label, subset in [("Main", main_rows), ("Conservative", cons_rows)]:
        models = {r["model"] for r in subset}
        max_rounds = {r["max_rounds"] for r in subset}
        over20 = [r["run_id"] for r in subset if r["rounds_played"] > 20 or (r["final_round"] or 0) > 20]
        lines.append(f"{label}: model(s)={models}, max_rounds value(s)={max_rounds}, runs with round>20={over20}")
    lines.append(f"Main strategy counts: Conceder={sum(1 for r in main_rows if r['seller_strategy']=='Conceder')}, Boulware={sum(1 for r in main_rows if r['seller_strategy']=='Boulware')}")
    lines.append(f"Conservative strategy counts: Conceder={sum(1 for r in cons_rows if r['seller_strategy']=='Conceder')}, Boulware={sum(1 for r in cons_rows if r['seller_strategy']=='Boulware')}")
    lines.append(
        "Both datasets produced by the same run_negotiation() (thesis_setup_b_test.py), same scenario "
        "(thesis_scenario.py), same ThesisBuyerAgent/buyer prompt, same ThesisRuleBasedSeller strategies, "
        "same thesis_utility.py functions, same compliance rules/deal definition (Task1_basic_price_negotiation, "
        "unmodified), same evaluator (thesis_evaluator.py) -- the guardian_robustness_check batch only "
        "redirects the ThesisGuardian class via unittest.mock.patch.object; no other code path differs. "
        "See thesis_guardian_robustness_check.py for the exact mechanism."
    )
    return lines


# =============================================================================
# Baseline reproduction check
# =============================================================================

def reproduce_baselines(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    main_rows = [r for r in rows if r["guardian_prompt"] == "main"]
    cons_rows = [r for r in rows if r["guardian_prompt"] == "conservative"]
    main_calls = sum(r["guardian_calls"] for r in main_rows)
    main_rev = sum(r["guardian_revisions"] for r in main_rows)
    cons_calls = sum(r["guardian_calls"] for r in cons_rows)
    cons_rev = sum(r["guardian_revisions"] for r in cons_rows)
    main_rate = main_rev / main_calls
    cons_rate = cons_rev / cons_calls
    lines.append(f"Main: {main_rev}/{main_calls} = {main_rate*100:.1f}% (expected ~100%) -> {'OK' if abs(main_rate-1.0)<1e-9 else 'MISMATCH'}")
    lines.append(f"Conservative: {cons_rev}/{cons_calls} = {cons_rate*100:.1f}% (expected ~68%) -> {'OK' if abs(cons_rate-0.68)<0.01 else 'MISMATCH'}")
    return lines, main_rate == 1.0, abs(cons_rate - 0.68) < 0.01


# =============================================================================
# Descriptive aggregation
# =============================================================================

def clopper_pearson(count: int, n: int) -> "tuple[float, float]":
    """Exact Clopper-Pearson 95% CI for a proportion -- appropriate given the
    small Conservative n (as low as 5 per stratum), unlike a Wald interval."""
    return proportion_confint(count, n, alpha=0.05, method="beta")


def wald_rd_ci(p1: float, n1: int, p0: float, n0: int) -> "tuple[float, float, float]":
    diff = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) if n1 and n0 else float("nan")
    return diff, diff - Z95 * se, diff + Z95 * se


def aggregate_group(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(rows)
    guardian_calls = sum(r["guardian_calls"] for r in rows)
    revisions = sum(r["guardian_revisions"] for r in rows)
    deals = [r for r in rows if r["deal_reached"]]
    fully_compliant = [r for r in rows if r["fully_compliant_run"]]
    below_res = [r for r in deals if r["deal_below_buyer_reservation"]]
    buyer_utils = [r["final_buyer_utility"] for r in deals]
    tokens = [r["total_tokens"] for r in rows]
    runtimes = [r["runtime_sec"] for r in rows]
    costs = [r["estimated_cost_usd"] for r in rows]
    run_level_rates = [r["guardian_revision_rate_run"] for r in rows if r["guardian_revision_rate_run"] is not None]

    fc_ci = clopper_pearson(len(fully_compliant), n)
    ar_ci = clopper_pearson(len(deals), n)
    br_ci = clopper_pearson(len(below_res), len(deals)) if deals else (float("nan"), float("nan"))

    agg = {
        "group": label, "n_runs": n,
        "guardian_calls": guardian_calls, "guardian_revisions": revisions,
        "revision_rate_pooled": revisions / guardian_calls if guardian_calls else None,
        "mean_revisions_per_run": statistics.mean(r["guardian_revisions"] for r in rows),
        "mean_run_level_revision_rate": statistics.mean(run_level_rates) if run_level_rates else None,
        "sd_run_level_revision_rate": statistics.stdev(run_level_rates) if len(run_level_rates) > 1 else None,
        "n_fully_compliant": len(fully_compliant), "fully_compliant_rate": len(fully_compliant) / n,
        "fully_compliant_rate_ci_low": fc_ci[0], "fully_compliant_rate_ci_high": fc_ci[1],
        "n_violations_total": sum(r["n_violations_total"] for r in rows),
        **{f"violations_{code}": sum(r[code] for r in rows) for code in RULE_CODES},
        "n_deals": len(deals), "agreement_rate": len(deals) / n,
        "agreement_rate_ci_low": ar_ci[0], "agreement_rate_ci_high": ar_ci[1],
        "buyer_utility_n": len(buyer_utils),
        "buyer_utility_mean": statistics.mean(buyer_utils) if buyer_utils else None,
        "buyer_utility_median": statistics.median(buyer_utils) if buyer_utils else None,
        "buyer_utility_sd": statistics.stdev(buyer_utils) if len(buyer_utils) > 1 else None,
        "n_deals_below_reservation": len(below_res),
        "share_below_reservation_of_deals": len(below_res) / len(deals) if deals else None,
        "below_reservation_rate_ci_low": br_ci[0], "below_reservation_rate_ci_high": br_ci[1],
        "sum_total_tokens": sum(tokens), "mean_tokens_per_run": statistics.mean(tokens),
        "median_tokens_per_run": statistics.median(tokens),
        "mean_runtime_per_run": statistics.mean(runtimes), "median_runtime_per_run": statistics.median(runtimes),
        "mean_cost_per_run": statistics.mean(costs), "median_cost_per_run": statistics.median(costs),
    }
    return agg


# =============================================================================
# Statistical comparisons (cautious, run-level, small-n aware)
# =============================================================================

def compare_binary(main_rows: List[Dict[str, Any]], cons_rows: List[Dict[str, Any]], outcome_fn, label: str, denom_fn=None) -> Dict[str, Any]:
    """Fisher Exact on a run-level binary outcome (denom_fn restricts the
    denominator, e.g. to deals only, for reservation-based outcomes)."""
    if denom_fn:
        main_denom = [r for r in main_rows if denom_fn(r)]
        cons_denom = [r for r in cons_rows if denom_fn(r)]
    else:
        main_denom, cons_denom = main_rows, cons_rows

    m_yes = sum(1 for r in main_denom if outcome_fn(r))
    c_yes = sum(1 for r in cons_denom if outcome_fn(r))
    m_n, c_n = len(main_denom), len(cons_denom)
    tbl = np.array([[c_yes, c_n - c_yes], [m_yes, m_n - m_yes]])  # [Conservative; Main]
    odds, p = sps.fisher_exact(tbl)
    p_m, p_c = (m_yes / m_n if m_n else float("nan")), (c_yes / c_n if c_n else float("nan"))
    rd, rd_lo, rd_hi = wald_rd_ci(p_c, c_n, p_m, m_n)
    rr = p_c / p_m if p_m else float("nan")
    m_ci = clopper_pearson(m_yes, m_n) if m_n else (float("nan"), float("nan"))
    c_ci = clopper_pearson(c_yes, c_n) if c_n else (float("nan"), float("nan"))
    return {
        "comparison": label, "n_main": m_n, "n_conservative": c_n,
        "rate_main": p_m, "rate_main_ci_low": m_ci[0], "rate_main_ci_high": m_ci[1],
        "rate_conservative": p_c, "rate_conservative_ci_low": c_ci[0], "rate_conservative_ci_high": c_ci[1],
        "risk_difference_cons_minus_main": rd, "rd_ci_low": rd_lo, "rd_ci_high": rd_hi,
        "risk_ratio_cons_over_main": rr,
        "fisher_odds_ratio": odds if not math.isinf(odds) else "undefined (zero cell)",
        "fisher_p_value": p,
        "caveat": "n_conservative is small (<=10 or <=deals-among-10) -- wide CI, descriptive/robustness-hint only, not a definitive test.",
    }


def compare_continuous(main_vals: np.ndarray, cons_vals: np.ndarray, label: str) -> Dict[str, Any]:
    if len(main_vals) < 2 or len(cons_vals) < 2:
        return {"comparison": label, "note": "insufficient n for inferential test", "n_main": len(main_vals), "n_conservative": len(cons_vals)}
    if np.std(main_vals, ddof=1) == 0 or np.std(cons_vals, ddof=1) == 0:
        t_stat, p_t = float("nan"), float("nan")
        note = "zero variance in at least one group -- Welch t-test not meaningful, reporting descriptive comparison and Mann-Whitney-U only"
    else:
        t_stat, p_t = sps.ttest_ind(cons_vals, main_vals, equal_var=False)
        note = ""
    u_stat, p_u = sps.mannwhitneyu(cons_vals, main_vals, alternative="two-sided")
    rbc = 2 * u_stat / (len(cons_vals) * len(main_vals)) - 1
    return {
        "comparison": label, "n_main": len(main_vals), "n_conservative": len(cons_vals),
        "mean_main": np.mean(main_vals), "median_main": np.median(main_vals), "sd_main": np.std(main_vals, ddof=1) if len(main_vals) > 1 else None,
        "mean_conservative": np.mean(cons_vals), "median_conservative": np.median(cons_vals), "sd_conservative": np.std(cons_vals, ddof=1) if len(cons_vals) > 1 else None,
        "mean_diff_cons_minus_main": np.mean(cons_vals) - np.mean(main_vals),
        "welch_t": t_stat, "welch_p": p_t,
        "mwu_u": u_stat, "mwu_p": p_u, "rank_biserial": rbc,
        "note": note or "small n_conservative -- interpret descriptively, not as a definitive significance test",
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    _no_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    rows = load_all()
    print(f"Loaded {len(rows)} runs total (40 Main + 10 Conservative)")

    print("\n=== Comparability check ===")
    for line in comparability_check(rows):
        print(line)

    print("\n=== Baseline reproduction check ===")
    baseline_lines, main_ok, cons_ok = reproduce_baselines(rows)
    for line in baseline_lines:
        print(line)
    if not (main_ok and cons_ok):
        print("\nBASELINE REPRODUCTION FAILED -- STOPPING.")
        return

    write_csv(rows, OUT_DIR / "guardian_main_vs_conservative_run_level.csv")

    main_rows = [r for r in rows if r["guardian_prompt"] == "main"]
    cons_rows = [r for r in rows if r["guardian_prompt"] == "conservative"]

    summary_rows = [
        aggregate_group(main_rows, "main_overall"),
        aggregate_group(cons_rows, "conservative_overall"),
    ]
    write_csv(summary_rows, OUT_DIR / "guardian_main_vs_conservative_summary.csv")
    print("\n=== Overall summary ===")
    for a in summary_rows:
        print(f"\n--- {a['group']} ---")
        for k, v in a.items():
            print(f"  {k}: {v}")

    by_strategy_rows = []
    for strat in ("Conceder", "Boulware"):
        by_strategy_rows.append(aggregate_group([r for r in main_rows if r["seller_strategy"] == strat], f"main_{strat.lower()}"))
        by_strategy_rows.append(aggregate_group([r for r in cons_rows if r["seller_strategy"] == strat], f"conservative_{strat.lower()}"))
    write_csv(by_strategy_rows, OUT_DIR / "guardian_main_vs_conservative_by_strategy.csv")
    print("\n=== By-strategy summary ===")
    for a in by_strategy_rows:
        print(f"{a['group']}: n={a['n_runs']}, revision_rate={a['revision_rate_pooled']}, "
              f"fully_compliant_rate={a['fully_compliant_rate']:.3f}, agreement_rate={a['agreement_rate']:.3f}")

    stats_rows = []
    stats_rows.append(compare_binary(main_rows, cons_rows, lambda r: r["fully_compliant_run"], "fully_compliant_run (overall)"))
    stats_rows.append(compare_binary(main_rows, cons_rows, lambda r: r["deal_reached"], "deal_reached (overall)"))
    stats_rows.append(compare_binary(main_rows, cons_rows, lambda r: r["deal_below_buyer_reservation"], "deal_below_buyer_reservation (deals only, overall)", denom_fn=lambda r: r["deal_reached"]))
    for strat in ("Conceder", "Boulware"):
        m_s = [r for r in main_rows if r["seller_strategy"] == strat]
        c_s = [r for r in cons_rows if r["seller_strategy"] == strat]
        stats_rows.append(compare_binary(m_s, c_s, lambda r: r["fully_compliant_run"], f"fully_compliant_run ({strat})"))
        stats_rows.append(compare_binary(m_s, c_s, lambda r: r["deal_reached"], f"deal_reached ({strat})"))
        stats_rows.append(compare_binary(m_s, c_s, lambda r: r["deal_below_buyer_reservation"], f"deal_below_buyer_reservation ({strat}, deals only)", denom_fn=lambda r: r["deal_reached"]))

    # Continuous: buyer utility (deals only), tokens, runtime, cost -- overall + by strategy
    def vals(rows_, field, deals_only=False):
        sub = [r for r in rows_ if (r["deal_reached"] if deals_only else True)]
        return np.array([r[field] for r in sub if r[field] is not None])

    stats_rows.append(compare_continuous(vals(main_rows, "final_buyer_utility", True), vals(cons_rows, "final_buyer_utility", True), "buyer_utility_of_deals (overall)"))
    stats_rows.append(compare_continuous(vals(main_rows, "total_tokens"), vals(cons_rows, "total_tokens"), "total_tokens (overall)"))
    stats_rows.append(compare_continuous(vals(main_rows, "runtime_sec"), vals(cons_rows, "runtime_sec"), "runtime_sec (overall)"))
    stats_rows.append(compare_continuous(vals(main_rows, "estimated_cost_usd"), vals(cons_rows, "estimated_cost_usd"), "estimated_cost_usd (overall)"))
    # Run-level revision rate comparison (respects clustering: one rate value per run, not per round)
    m_rates = np.array([r["guardian_revision_rate_run"] for r in main_rows])
    c_rates = np.array([r["guardian_revision_rate_run"] for r in cons_rows])
    stats_rows.append(compare_continuous(m_rates, c_rates, "run_level_revision_rate (overall, cluster-respecting)"))

    for strat in ("Conceder", "Boulware"):
        m_s = [r for r in main_rows if r["seller_strategy"] == strat]
        c_s = [r for r in cons_rows if r["seller_strategy"] == strat]
        stats_rows.append(compare_continuous(vals(m_s, "final_buyer_utility", True), vals(c_s, "final_buyer_utility", True), f"buyer_utility_of_deals ({strat})"))
        stats_rows.append(compare_continuous(vals(m_s, "total_tokens"), vals(c_s, "total_tokens"), f"total_tokens ({strat})"))
        stats_rows.append(compare_continuous(vals(m_s, "runtime_sec"), vals(c_s, "runtime_sec"), f"runtime_sec ({strat})"))
        stats_rows.append(compare_continuous(vals(m_s, "estimated_cost_usd"), vals(c_s, "estimated_cost_usd"), f"estimated_cost_usd ({strat})"))

    write_csv(stats_rows, OUT_DIR / "guardian_main_vs_conservative_stats.csv")
    print(f"\n=== Statistical comparisons written to guardian_main_vs_conservative_stats.csv ({len(stats_rows)} rows) ===")
    for r in stats_rows:
        print(r.get("comparison"), "->", {k: v for k, v in r.items() if k not in ("comparison", "caveat", "note")})

    print(f"\nAll outputs written to: {OUT_DIR}")
    print("\nSTEP 5 GUARDIAN ROBUSTNESS ANALYSIS COMPLETE: YES")


if __name__ == "__main__":
    main()
