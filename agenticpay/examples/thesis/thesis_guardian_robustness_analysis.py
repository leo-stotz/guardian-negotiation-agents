"""Descriptive run-level analysis of the 10 conservative-Guardian robustness-check
runs (see `thesis_guardian_robustness_check.py`; documented in the thesis as the
Guardian robustness check with a conservative prompt variant, n=10).

Reads only the already-produced run/eval JSON under
`agenticpay/results/thesis/guardian_robustness_check/` -- does not touch any
main-experiment, calibration, or verification data. Descriptive only: this
script deliberately does NOT compute a main-vs-conservative comparison (no
old pre-fix pilot data is loaded here) -- that primary comparison is reserved
for after the main experiment, against the final Setup-B main-Guardian runs
under the corrected max_rounds logic, per the user's explicit instruction.

Outputs:
    run_level.csv        -- one row per conservative run, all requested fields.
    aggregated_summary.csv -- conceder / boulware / overall aggregates, all
                              guardian_prompt=conservative (no main-guardian
                              rows -- there is nothing to compare against yet
                              under matching conditions).
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBUSTNESS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "guardian_robustness_check"

STRATEGIES = ["conceder", "boulware"]
REPS = range(1, 6)


def load_run(strategy: str, rep: int) -> Dict[str, Any]:
    run_id = f"conservative_setup_b_{strategy}_rep{rep}"
    raw = json.loads((ROBUSTNESS_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    evald = json.loads((ROBUSTNESS_DIR / f"{run_id}_eval.json").read_text(encoding="utf-8"))
    rr = evald["run_result"]
    violations = evald["violations"]

    rounds = raw["rounds"]
    guardian_calls = sum(1 for r in rounds if r.get("guardian_feedback") is not None)
    revisions = sum(1 for r in rounds if r.get("revision_triggered"))
    revision_rate = revisions / guardian_calls if guardian_calls else None

    violation_types: Dict[str, int] = {}
    for v in violations:
        violation_types[v["rule"]] = violation_types.get(v["rule"], 0) + 1

    return {
        "run_id": run_id,
        "guardian_prompt": "conservative",
        "setup": "setup_b",
        "seller_strategy": strategy,
        "rep": rep,
        "deal": rr["deal_reached"],
        "final_round": rr["final_round"],
        "max_rounds": rr["max_rounds"],
        "rounds_played": len(rounds),
        "buyer_utility": rr["final_buyer_utility"],
        "seller_utility": rr["final_seller_utility"],
        "deal_below_buyer_reservation": rr["deal_below_buyer_reservation"],
        "fully_compliant_run": rr["fully_compliant_run"],
        "n_violations_total": rr["n_violations_total"],
        "violation_types": violation_types,
        "guardian_calls": guardian_calls,
        "revisions": revisions,
        "revision_rate": revision_rate,
        "buyer_total_tokens": rr["buyer_total_tokens"],
        "guardian_total_tokens": rr["guardian_total_tokens"],
        "total_tokens": rr["total_tokens"],
        "runtime_sec": rr["runtime_sec"],
        "api_cost_usd": rr["estimated_cost_usd"],
    }


def write_run_level_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_id", "guardian_prompt", "setup", "seller_strategy", "deal", "final_round",
        "max_rounds", "rounds_played", "buyer_utility", "seller_utility",
        "deal_below_buyer_reservation", "fully_compliant_run", "n_violations_total",
        "violation_types", "guardian_calls", "revisions", "revision_rate",
        "buyer_total_tokens", "guardian_total_tokens", "total_tokens", "runtime_sec", "api_cost_usd",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            row = dict(r)
            row["violation_types"] = json.dumps(row["violation_types"])
            row.pop("rep", None)
            w.writerow(row)


def aggregate(runs: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(runs)
    deals = [r for r in runs if r["deal"]]
    n_deals = len(deals)
    fully_compliant = [r for r in runs if r["fully_compliant_run"]]
    below_reservation_deals = [r for r in deals if r["deal_below_buyer_reservation"]]
    all_violation_types: Dict[str, int] = {}
    for r in runs:
        for k, v in r["violation_types"].items():
            all_violation_types[k] = all_violation_types.get(k, 0) + v

    total_guardian_calls = sum(r["guardian_calls"] for r in runs)
    total_revisions = sum(r["revisions"] for r in runs)

    return {
        "group": label,
        "n_runs": n,
        "n_deals": n_deals,
        "agreement_rate": n_deals / n if n else None,
        "n_fully_compliant_runs": len(fully_compliant),
        "fully_compliant_share": len(fully_compliant) / n if n else None,
        "n_violations_total": sum(r["n_violations_total"] for r in runs),
        "violation_types_breakdown": all_violation_types,
        "n_deals_below_buyer_reservation": len(below_reservation_deals),
        "mean_buyer_utility_of_deals": statistics.mean([r["buyer_utility"] for r in deals]) if deals else None,
        "median_buyer_utility_of_deals": statistics.median([r["buyer_utility"] for r in deals]) if deals else None,
        "total_guardian_calls": total_guardian_calls,
        "total_revisions": total_revisions,
        "revision_rate_overall": total_revisions / total_guardian_calls if total_guardian_calls else None,
        "mean_revisions_per_run": statistics.mean([r["revisions"] for r in runs]) if runs else None,
        "mean_guardian_calls_per_run": statistics.mean([r["guardian_calls"] for r in runs]) if runs else None,
        "sum_total_tokens": sum(r["total_tokens"] for r in runs),
        "mean_total_tokens_per_run": statistics.mean([r["total_tokens"] for r in runs]) if runs else None,
        "sum_runtime_sec": sum(r["runtime_sec"] for r in runs),
        "mean_runtime_sec_per_run": statistics.mean([r["runtime_sec"] for r in runs]) if runs else None,
        "sum_api_cost_usd": sum(r["api_cost_usd"] for r in runs),
        "mean_api_cost_usd_per_run": statistics.mean([r["api_cost_usd"] for r in runs]) if runs else None,
    }


def write_aggregated_csv(aggs: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = list(aggs[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in aggs:
            row = dict(a)
            row["violation_types_breakdown"] = json.dumps(row["violation_types_breakdown"])
            w.writerow(row)


def print_agg(a: Dict[str, Any]) -> None:
    print(f"\n--- {a['group']} (n={a['n_runs']}) ---")
    print(f"  Agreement Rate: {a['n_deals']}/{a['n_runs']} = {a['agreement_rate']*100:.1f}%")
    print(f"  Fully compliant runs: {a['n_fully_compliant_runs']}/{a['n_runs']} = {a['fully_compliant_share']*100:.1f}%")
    print(f"  Total compliance violations: {a['n_violations_total']} -- breakdown: {a['violation_types_breakdown']}")
    print(f"  Deals below buyer reservation utility: {a['n_deals_below_buyer_reservation']}")
    if a['mean_buyer_utility_of_deals'] is not None:
        print(f"  Buyer utility of deals: mean={a['mean_buyer_utility_of_deals']:.4f}, median={a['median_buyer_utility_of_deals']:.4f}")
    print(f"  Guardian calls total: {a['total_guardian_calls']} (mean/run: {a['mean_guardian_calls_per_run']:.1f})")
    print(f"  Revisions total: {a['total_revisions']} (mean/run: {a['mean_revisions_per_run']:.1f})")
    print(f"  Revision rate (revisions / guardian calls): {a['revision_rate_overall']*100:.1f}%")
    print(f"  Tokens: sum={a['sum_total_tokens']}, mean/run={a['mean_total_tokens_per_run']:.0f}")
    print(f"  Runtime: sum={a['sum_runtime_sec']:.1f}s, mean/run={a['mean_runtime_sec_per_run']:.1f}s")
    print(f"  API cost: sum=${a['sum_api_cost_usd']:.4f}, mean/run=${a['mean_api_cost_usd_per_run']:.4f}")


def main() -> None:
    runs = [load_run(s, r) for s in STRATEGIES for r in REPS]

    print("=== Run-level detail (guardian_prompt=conservative, setup_b only) ===")
    for r in runs:
        print(
            f"  {r['run_id']:35s} deal={r['deal']!s:5} round={r['final_round']:>2}/{r['max_rounds']} "
            f"U_buyer={r['buyer_utility']:.4f} fully_compliant={r['fully_compliant_run']!s:5} "
            f"n_violations={r['n_violations_total']} guardian_calls={r['guardian_calls']:>2} "
            f"revisions={r['revisions']:>2} rev_rate={r['revision_rate']*100:5.1f}% "
            f"tokens={r['total_tokens']:>7} runtime={r['runtime_sec']:6.1f}s cost=${r['api_cost_usd']:.4f}"
        )

    run_csv = ROBUSTNESS_DIR / "run_level.csv"
    write_run_level_csv(runs, run_csv)
    print(f"\nWrote run-level CSV ({len(runs)} rows) to: {run_csv}")

    conceder_runs = [r for r in runs if r["seller_strategy"] == "conceder"]
    boulware_runs = [r for r in runs if r["seller_strategy"] == "boulware"]

    aggs = [
        aggregate(runs, "conservative_all (setup_b, conceder+boulware pooled)"),
        aggregate(conceder_runs, "conservative_conceder (setup_b)"),
        aggregate(boulware_runs, "conservative_boulware (setup_b)"),
    ]
    for a in aggs:
        print_agg(a)

    agg_csv = ROBUSTNESS_DIR / "aggregated_summary.csv"
    write_aggregated_csv(aggs, agg_csv)
    print(f"\nWrote aggregated summary CSV ({len(aggs)} rows) to: {agg_csv}")

    # Explicit round-cap sanity check (off-by-one fix verification for this batch)
    print("\n=== Round-cap sanity check (post max_rounds fix) ===")
    max_final_round = max(r["final_round"] for r in runs)
    max_rounds_played = max(r["rounds_played"] for r in runs)
    print(f"  max(final_round) across all 10 runs = {max_final_round} (must be <= 20)")
    print(f"  max(rounds_played) across all 10 runs = {max_rounds_played} (must be <= 20)")
    print(f"  Any run with final_round or rounds_played > 20: "
          f"{[r['run_id'] for r in runs if r['final_round'] > 20 or r['rounds_played'] > 20]}")


if __name__ == "__main__":
    main()
