"""Step 1 of the main-experiment statistical analysis: build and validate the
clean 80-row run-level dataset, plus simple, transparent cross-check
aggregations -- NO inferential statistics here (no tests, no p-values).

Reads ONLY `agenticpay/results/thesis/main_experiment/` (the frozen, complete
80-run main sample). Does not read calibration, variance, guardian_precision_test,
guardian_robustness_check, or any pre-flight/diagnostic directory -- the main
and Conservative-Guardian robustness data are kept strictly separate per the
user's instruction; the robustness comparison is Step 5, not here.

Never writes to `main_experiment/` -- all output goes to a new, separate
directory: `agenticpay/results/thesis/main_experiment_analysis/`.

Outputs:
    run_level_main.csv      -- one row per run, 80 rows, columns as specified.
    crosscheck_counts.csv   -- small per-cell raw count table for manual
                               (e.g. Excel) verification of the aggregates.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment"
OUT_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment_analysis"

RULE_CODES = [
    "MAX_PRICE_EXCEEDED",
    "OUT_OF_RANGE",
    "BELOW_RESERVATION_UTILITY",
    "INVALID_CONTRACT",
    "CONFIDENTIAL_INFO_DISCLOSED",
]

EXPECTED_RUN_IDS = {
    f"{setup}_{strategy}_rep{rep:02d}"
    for setup in ("setup_a", "setup_b")
    for strategy in ("conceder", "boulware")
    for rep in range(1, 21)
}


def load_run_ids() -> List[str]:
    files = os.listdir(MAIN_DIR)
    raw_files = [f for f in files if f.endswith(".json") and not f.endswith("_eval.json") and f not in ("summary.json", "resume_summary.json")]
    return sorted(f[:-5] for f in raw_files)


def build_row(run_id: str) -> Dict[str, Any]:
    raw = json.loads((MAIN_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    ev = json.loads((MAIN_DIR / f"{run_id}_eval.json").read_text(encoding="utf-8"))
    rr = ev["run_result"]

    parts = run_id.split("_")
    setup_raw = parts[0] + "_" + parts[1]  # "setup_a" / "setup_b"
    strategy_raw = parts[2]  # "conceder" / "boulware"
    setup_label = "A" if setup_raw == "setup_a" else "B"
    strategy_label = "Conceder" if strategy_raw == "conceder" else "Boulware"

    is_setup_b = setup_raw == "setup_b"
    rounds = raw.get("rounds") or []

    guardian_calls: Optional[int] = None
    guardian_revisions: Optional[int] = None
    guardian_revision_rate: Optional[float] = None
    if is_setup_b:
        guardian_calls = sum(1 for r in rounds if r.get("guardian_feedback") is not None)
        guardian_revisions = sum(1 for r in rounds if r.get("revision_triggered"))
        guardian_revision_rate = (guardian_revisions / guardian_calls) if guardian_calls else None

    by_rule = rr.get("n_violations_by_rule") or {}

    row = {
        "run_id": run_id,
        "setup": setup_label,
        "seller_strategy": strategy_label,
        "deal_reached": bool(raw["deal_reached"]),
        "final_round": raw["final_round"],
        "rounds_played": raw["rounds_played"],
        "fully_compliant_run": bool(rr["fully_compliant_run"]),
        "n_violations_total": rr["n_violations_total"],
        "MAX_PRICE_EXCEEDED": by_rule.get("MAX_PRICE_EXCEEDED", 0),
        "OUT_OF_RANGE": by_rule.get("OUT_OF_RANGE", 0),
        "BELOW_RESERVATION_UTILITY": by_rule.get("BELOW_RESERVATION_UTILITY", 0),
        "INVALID_CONTRACT": by_rule.get("INVALID_CONTRACT", 0),
        "CONFIDENTIAL_INFO_DISCLOSED": by_rule.get("CONFIDENTIAL_INFO_DISCLOSED", 0),
        "final_buyer_utility": rr["final_buyer_utility"],  # None on No-Deal, never 0
        "final_seller_utility": rr["final_seller_utility"],  # None on No-Deal, never 0
        "deal_below_buyer_reservation": rr["deal_below_buyer_reservation"],  # None on No-Deal
        "buyer_total_tokens": rr.get("buyer_total_tokens"),
        "guardian_total_tokens": rr.get("guardian_total_tokens"),  # None for Setup A
        "total_tokens": rr.get("total_tokens"),
        "runtime_sec": rr.get("runtime_sec"),
        "estimated_cost_usd": rr.get("estimated_cost_usd"),
        "guardian_calls": guardian_calls,  # None for Setup A
        "guardian_revisions": guardian_revisions,  # None for Setup A
        "guardian_revision_rate": guardian_revision_rate,  # None for Setup A
    }
    return row


FIELDNAMES = [
    "run_id", "setup", "seller_strategy", "deal_reached", "final_round", "rounds_played",
    "fully_compliant_run", "n_violations_total",
    "MAX_PRICE_EXCEEDED", "OUT_OF_RANGE", "BELOW_RESERVATION_UTILITY", "INVALID_CONTRACT", "CONFIDENTIAL_INFO_DISCLOSED",
    "final_buyer_utility", "final_seller_utility", "deal_below_buyer_reservation",
    "buyer_total_tokens", "guardian_total_tokens", "total_tokens", "runtime_sec", "estimated_cost_usd",
    "guardian_calls", "guardian_revisions", "guardian_revision_rate",
]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k, v in out.items():
                if v is None:
                    out[k] = ""  # explicit blank = NA in the CSV, never 0
            w.writerow(out)


# --------------------------------------------------------------------------- validation

def validate(rows: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []

    if len(rows) != 80:
        issues.append(f"Expected 80 rows, got {len(rows)}")

    run_ids = [r["run_id"] for r in rows]
    if len(set(run_ids)) != len(run_ids):
        dupes = {rid for rid in run_ids if run_ids.count(rid) > 1}
        issues.append(f"Duplicate run_ids found: {dupes}")

    actual_ids = set(run_ids)
    missing = EXPECTED_RUN_IDS - actual_ids
    extra = actual_ids - EXPECTED_RUN_IDS
    if missing:
        issues.append(f"Missing expected run_ids: {sorted(missing)}")
    if extra:
        issues.append(f"Unexpected extra run_ids: {sorted(extra)}")

    from collections import Counter
    cell_counts = Counter((r["setup"], r["seller_strategy"]) for r in rows)
    for setup in ("A", "B"):
        for strat in ("Conceder", "Boulware"):
            n = cell_counts.get((setup, strat), 0)
            if n != 20:
                issues.append(f"Cell (setup={setup}, strategy={strat}) has {n} rows, expected 20")

    for r in rows:
        fr, rp = r["final_round"], r["rounds_played"]
        if (fr is not None and fr > 20) or (rp is not None and rp > 20):
            issues.append(f"{r['run_id']}: round > 20 (final_round={fr}, rounds_played={rp})")

        # Buyer utility only defined on deals
        if r["deal_reached"] and r["final_buyer_utility"] is None:
            issues.append(f"{r['run_id']}: deal_reached=True but final_buyer_utility is None (unexpected)")
        if not r["deal_reached"] and r["final_buyer_utility"] is not None:
            issues.append(f"{r['run_id']}: deal_reached=False but final_buyer_utility is NOT None (should be NA)")
        if not r["deal_reached"] and r["final_buyer_utility"] == 0:
            issues.append(f"{r['run_id']}: ARTIFICIAL ZERO -- No-Deal run has final_buyer_utility=0 instead of None")

        # deal_below_buyer_reservation only interpretable when a deal happened
        if not r["deal_reached"] and r["deal_below_buyer_reservation"] is not None:
            issues.append(f"{r['run_id']}: deal_reached=False but deal_below_buyer_reservation is not None (should be NA)")

        # Setup A must have no Guardian data
        if r["setup"] == "A":
            if r["guardian_total_tokens"] is not None or r["guardian_calls"] is not None or r["guardian_revisions"] is not None:
                issues.append(f"{r['run_id']}: Setup A run has non-null Guardian fields (unexpected)")
        # Setup B must have Guardian data
        if r["setup"] == "B":
            if r["guardian_total_tokens"] is None or r["guardian_calls"] is None:
                issues.append(f"{r['run_id']}: Setup B run is MISSING Guardian fields (unexpected)")

        # unexpected missing values in fields that should always be present
        always_required = ["run_id", "setup", "seller_strategy", "deal_reached", "final_round",
                            "rounds_played", "fully_compliant_run", "n_violations_total",
                            "buyer_total_tokens", "total_tokens", "runtime_sec", "estimated_cost_usd"]
        for field in always_required:
            if r[field] is None or r[field] == "":
                issues.append(f"{r['run_id']}: unexpected missing value in required field '{field}'")

    return issues


# --------------------------------------------------------------------------- Step 1B aggregates

def aggregate_cell(rows: List[Dict[str, Any]], setup: str, strategy: str) -> Dict[str, Any]:
    cell = [r for r in rows if r["setup"] == setup and r["seller_strategy"] == strategy]
    n = len(cell)
    deals = [r for r in cell if r["deal_reached"]]
    n_deals = len(deals)
    fully_compliant = [r for r in cell if r["fully_compliant_run"]]
    below_res = [r for r in deals if r["deal_below_buyer_reservation"]]
    buyer_utils = [r["final_buyer_utility"] for r in deals]
    tokens = [r["total_tokens"] for r in cell]
    runtimes = [r["runtime_sec"] for r in cell]
    costs = [r["estimated_cost_usd"] for r in cell]

    agg = {
        "cell": f"{setup}+{strategy}",
        "n": n,
        "n_deals": n_deals,
        "agreement_rate": n_deals / n if n else None,
        "n_fully_compliant": len(fully_compliant),
        "fully_compliant_rate": len(fully_compliant) / n if n else None,
        "sum_violations_total": sum(r["n_violations_total"] for r in cell),
        "n_deals_below_reservation": len(below_res),
        "n_deals_basis_for_reservation_rate": n_deals,
        "buyer_utility_n": len(buyer_utils),
        "buyer_utility_mean": statistics.mean(buyer_utils) if buyer_utils else None,
        "buyer_utility_median": statistics.median(buyer_utils) if buyer_utils else None,
        "buyer_utility_stdev": statistics.stdev(buyer_utils) if len(buyer_utils) > 1 else None,
        "buyer_utility_min": min(buyer_utils) if buyer_utils else None,
        "buyer_utility_max": max(buyer_utils) if buyer_utils else None,
        "tokens_mean": statistics.mean(tokens) if tokens else None,
        "tokens_median": statistics.median(tokens) if tokens else None,
        "runtime_mean": statistics.mean(runtimes) if runtimes else None,
        "runtime_median": statistics.median(runtimes) if runtimes else None,
        "cost_mean": statistics.mean(costs) if costs else None,
        "cost_median": statistics.median(costs) if costs else None,
    }

    if setup == "B":
        g_calls = [r["guardian_calls"] for r in cell]
        g_rev = [r["guardian_revisions"] for r in cell]
        agg["guardian_calls_sum"] = sum(g_calls)
        agg["guardian_revisions_sum"] = sum(g_rev)
        agg["guardian_revision_rate_pooled"] = sum(g_rev) / sum(g_calls) if sum(g_calls) else None

    return agg


def build_crosscheck_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Raw, unaggregated count/sum table per cell for independent manual
    (e.g. Excel) verification -- no derived rates, just counts and sums."""
    table = []
    for setup in ("A", "B"):
        for strategy in ("Conceder", "Boulware"):
            cell = [r for r in rows if r["setup"] == setup and r["seller_strategy"] == strategy]
            deals = [r for r in cell if r["deal_reached"]]
            buyer_utils = [r["final_buyer_utility"] for r in deals]
            table.append({
                "cell": f"{setup}+{strategy}",
                "n_runs": len(cell),
                "n_deals": len(deals),
                "n_fully_compliant_runs": sum(1 for r in cell if r["fully_compliant_run"]),
                "n_deals_below_reservation": sum(1 for r in deals if r["deal_below_buyer_reservation"]),
                "sum_buyer_utility_of_deals": sum(buyer_utils),
                "n_deals_with_buyer_utility": len(buyer_utils),
                "sum_total_tokens": sum(r["total_tokens"] for r in cell),
                "sum_estimated_cost_usd": sum(r["estimated_cost_usd"] for r in cell),
            })
    return table


def main() -> None:
    run_ids = load_run_ids()
    print(f"Found {len(run_ids)} run files in {MAIN_DIR}")

    rows = [build_row(rid) for rid in run_ids]

    csv_path = OUT_DIR / "run_level_main.csv"
    write_csv(rows, csv_path)
    print(f"Wrote run-level dataset ({len(rows)} rows) to: {csv_path}")

    print("\n=== Validation ===")
    issues = validate(rows)
    if issues:
        print(f"{len(issues)} ISSUE(S) FOUND:")
        for i in issues:
            print(" -", i)
    else:
        print("No issues found.")

    print("\n=== Step 1B: per-cell aggregates ===")
    aggs = [aggregate_cell(rows, s, strat) for s in ("A", "B") for strat in ("Conceder", "Boulware")]
    for a in aggs:
        print(f"\n--- {a['cell']} (n={a['n']}) ---")
        for k, v in a.items():
            if k in ("cell", "n"):
                continue
            print(f"  {k}: {v}")

    agg_csv_path = OUT_DIR / "cell_aggregates_step1b.csv"
    if agg_csv_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {agg_csv_path}")
    with agg_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(aggs[0].keys())
        for a in aggs[1:]:
            for k in a.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in aggs:
            w.writerow(a)
    print(f"\nWrote per-cell aggregates to: {agg_csv_path}")

    print("\n=== Step 1C: cross-check count table (for manual/Excel verification) ===")
    crosscheck = build_crosscheck_table(rows)
    for row in crosscheck:
        print(row)
    crosscheck_path = OUT_DIR / "crosscheck_counts.csv"
    if crosscheck_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {crosscheck_path}")
    with crosscheck_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crosscheck[0].keys()))
        w.writeheader()
        for row in crosscheck:
            w.writerow(row)
    print(f"Wrote cross-check count table to: {crosscheck_path}")

    print(f"\nSTEP 1 DATASET VALIDATED: {'YES' if not issues else 'NO'}")


if __name__ == "__main__":
    main()
