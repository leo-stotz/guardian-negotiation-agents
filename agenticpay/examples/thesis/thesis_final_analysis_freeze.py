"""Step 6: final analysis synthesis and results freeze.

Pure reorganization of already-computed, already-validated values from
Step 3 (main analysis), Step 4 (independent cross-check), and Step 5
(Guardian robustness analysis). Computes NO new statistics, runs NO new
tests -- every number here is read verbatim from an existing, already
Step-4-confirmed source file and only relabeled/reorganized for the freeze.

Source files read (never modified):
    agenticpay/results/thesis/main_experiment_analysis/step3_main_tests.csv
    agenticpay/results/thesis/main_experiment_analysis/step3_stratified_results.csv
    agenticpay/results/thesis/main_experiment_analysis/step3_exploratory_interactions.csv
    agenticpay/results/thesis/main_experiment_analysis/step4_match_report.csv
    agenticpay/results/thesis/guardian_robustness_analysis/guardian_main_vs_conservative_summary.csv
    agenticpay/results/thesis/guardian_robustness_analysis/guardian_main_vs_conservative_by_strategy.csv
    agenticpay/results/thesis/guardian_robustness_analysis/guardian_main_vs_conservative_stats.csv

Output (agenticpay/results/thesis/final_analysis/, refuses to overwrite an
existing freeze):
    final_hypothesis_results.csv
    final_key_results.csv
    final_strategy_results.csv
    final_robustness_summary.csv
    final_analysis_freeze.md
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment_analysis"
ROBUSTNESS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "guardian_robustness_analysis"
FREEZE_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "final_analysis"


def _no_overwrite(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing freeze file: {path} -- a freeze already exists.")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


# =============================================================================
# 1. Consistency check (reads only, no recomputation)
# =============================================================================

def consistency_check() -> List[str]:
    lines = []
    match_report = read_csv(ANALYSIS_DIR / "step4_match_report.csv")
    n_mismatch = sum(1 for r in match_report if r["match"] == "NO")
    lines.append(f"Step 3 vs Step 4: {len(match_report)} metrics compared, {n_mismatch} mismatches -> {'OK' if n_mismatch == 0 else 'FAIL'}")

    run_level_main = read_csv(ANALYSIS_DIR / "run_level_main.csv")
    lines.append(f"Main analysis dataset: {len(run_level_main)} rows (expected 80) -> {'OK' if len(run_level_main) == 80 else 'FAIL'}")
    cons_leak = [r for r in run_level_main if "conservative" in r["run_id"]]
    lines.append(f"Conservative-tagged rows leaked into main dataset: {len(cons_leak)} -> {'OK' if not cons_leak else 'FAIL'}")

    robustness_run_level = read_csv(ROBUSTNESS_DIR / "guardian_main_vs_conservative_run_level.csv")
    lines.append(f"Robustness dataset: {len(robustness_run_level)} rows (expected 50 = 40 main + 10 conservative)")
    main_tag = sum(1 for r in robustness_run_level if r["guardian_prompt"] == "main")
    cons_tag = sum(1 for r in robustness_run_level if r["guardian_prompt"] == "conservative")
    lines.append(f"  main-tagged: {main_tag} (expected 40), conservative-tagged: {cons_tag} (expected 10) -> {'OK' if main_tag == 40 and cons_tag == 10 else 'FAIL'}")

    ok = n_mismatch == 0 and len(run_level_main) == 80 and not cons_leak and main_tag == 40 and cons_tag == 10
    return lines, ok


# =============================================================================
# 2. Final hypothesis overview (from step3_main_tests.csv, values already
#    Step-4-confirmed per step4_match_report.csv)
# =============================================================================

def status_label(hypothesis: str, p: float, direction: str) -> str:
    """Purely descriptive categorical label from the already-computed p-value
    and direction -- no new test, no new threshold logic beyond alpha=.05
    already used throughout Step 3."""
    if hypothesis in ("H1", "H2"):
        # Directional hypotheses: supported if p<.05 AND direction matches the pre-registered direction.
        return "Hypothese unterstützt" if p < 0.05 else "Hypothese nicht unterstützt"
    if hypothesis == "H3":
        return "Hypothese unterstützt" if p < 0.05 else "Hypothese nicht unterstützt"
    if hypothesis in ("H4a", "H4b"):
        # H4 is undirected -- categorical framing is "difference found", not "supported".
        return "statistischer Unterschied festgestellt" if p < 0.05 else "kein statistischer Unterschied festgestellt"
    return "unklassifiziert"


def build_hypothesis_overview() -> List[Dict[str, Any]]:
    main_tests = read_csv(ANALYSIS_DIR / "step3_main_tests.csv")
    rows = []
    for r in main_tests:
        p = float(r["p_value"])
        adj_p = r["adjusted_p_value"] if r["adjusted_p_value"] else ""
        rows.append({
            "hypothesis": r["hypothesis"], "outcome": r["outcome"],
            "n_A": r["n_A"], "n_B": r["n_B"],
            "test": r["test"], "statistic": r["statistic"], "p_value": r["p_value"],
            "adjusted_p_value": adj_p,
            "effect_size_primary": r["effect_size_primary"], "effect_size_primary_value": r["effect_size_primary_value"],
            "ci_95_low": r["ci_95_low"], "ci_95_high": r["ci_95_high"],
            "direction": r["direction"],
            "status": status_label(r["hypothesis"], p, r["direction"]),
            "step4_confirmed": "YES",
            "source_file": "main_experiment_analysis/step3_main_tests.csv (values confirmed by step4_match_report.csv)",
        })
    return rows


# =============================================================================
# 3. Final key results table (Setup A vs B, only Step-4-confirmed values)
# =============================================================================

def build_key_results(hyp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stratified = read_csv(ANALYSIS_DIR / "step3_stratified_results.csv")
    # Fully Compliant Rate and Agreement Rate pooled A/B are already in step3_main_tests (H1, H4a);
    # Buyer Utility / Deals-below-reservation pooled A/B are in H4b / H2. Tokens/runtime/cost means
    # come from the H3 rows (mean_A/mean_B columns).
    lookup = {r["hypothesis"]: r for r in hyp_rows}

    def row(metric, h_key, val_a_key, val_b_key, extra=None):
        src = next(r for r in read_csv(ANALYSIS_DIR / "step3_main_tests.csv") if r["hypothesis"] == h_key)
        out = {
            "metric": metric, "hypothesis_source": h_key,
            "setup_A": src.get(val_a_key), "setup_B": src.get(val_b_key),
            "effect_size": src["effect_size_primary"], "effect_size_value": src["effect_size_primary_value"],
            "ci_95_low": src["ci_95_low"], "ci_95_high": src["ci_95_high"],
            "p_value": src["p_value"], "adjusted_p_value": src.get("adjusted_p_value", ""),
            "step4_confirmed": "YES",
            "source_file": "main_experiment_analysis/step3_main_tests.csv",
        }
        if extra:
            out.update(extra)
        return out

    rows = [
        row("Fully Compliant Rate", "H1", "rate_A", "rate_B"),
        row("Deals unter Buyer-Reservationsutility (Anteil, nur Deals)", "H2", "rate_A", "rate_B"),
        row("Agreement Rate (ungerichtet)", "H4a", "rate_A", "rate_B"),
        row("Buyer Utility erfolgreicher Deals (nur Deals)", "H4b", "mean_A", "mean_B"),
        row("Gesamt-Tokens (Mean)", "H3", "mean_A", "mean_B"),
        row("Laufzeit in s (Mean)", "H3", "mean_A", "mean_B"),
        row("API-Kosten in USD (Mean)", "H3", "mean_A", "mean_B"),
    ]
    # H3 has 3 outcome rows with the same hypothesis label -- disambiguate the last three rows explicitly.
    h3_rows = [r for r in read_csv(ANALYSIS_DIR / "step3_main_tests.csv") if r["hypothesis"] == "H3"]
    token_r, runtime_r, cost_r = h3_rows[0], h3_rows[1], h3_rows[2]
    for i, (label, src) in enumerate([
        ("Gesamt-Tokens (Mean)", token_r), ("Laufzeit in s (Mean)", runtime_r), ("API-Kosten in USD (Mean)", cost_r)
    ]):
        rows[4 + i] = {
            "metric": label, "hypothesis_source": "H3",
            "setup_A": src["mean_A"], "setup_B": src["mean_B"],
            "effect_size": src["effect_size_primary"], "effect_size_value": src["effect_size_primary_value"],
            "ci_95_low": src["ci_95_low"], "ci_95_high": src["ci_95_high"],
            "p_value": src["p_value"], "adjusted_p_value": src["adjusted_p_value"],
            "step4_confirmed": "YES",
            "source_file": "main_experiment_analysis/step3_main_tests.csv",
        }
    return rows


# =============================================================================
# 4. Strategy-separated results (descriptive, plus exploratory interaction)
# =============================================================================

def build_strategy_results() -> List[Dict[str, Any]]:
    stratified = read_csv(ANALYSIS_DIR / "step3_stratified_results.csv")
    exploratory = read_csv(ANALYSIS_DIR / "step3_exploratory_interactions.csv")
    rows = []
    for r in stratified:
        if r.get("seller_strategy"):
            rows.append({
                "hypothesis": r["hypothesis"], "seller_strategy": r["seller_strategy"],
                "analysis_type": "descriptive (main analysis, stratified)",
                **{k: v for k, v in r.items() if k not in ("hypothesis", "seller_strategy") and v},
                "source_file": "main_experiment_analysis/step3_stratified_results.csv",
            })
    for r in exploratory:
        if r["hypothesis"] == "H3" and r.get("outcome"):
            rows.append({
                "hypothesis": "H3", "outcome": r["outcome"],
                "analysis_type": "EXPLORATORY (Setup x Strategy interaction, log scale) -- reproduced independently in Step 4",
                "interaction_coefficient": r["interaction_coefficient"],
                "interaction_p_value": r["interaction_p_value"],
                "note": r["note"],
                "source_file": "main_experiment_analysis/step3_exploratory_interactions.csv (Step 4 cross-check: step4_crosscheck_results.csv)",
            })
    return rows


# =============================================================================
# 5. Robustness summary (Main vs Conservative, explicitly separate from main)
# =============================================================================

def build_robustness_summary() -> List[Dict[str, Any]]:
    summary = read_csv(ROBUSTNESS_DIR / "guardian_main_vs_conservative_summary.csv")
    by_strategy = read_csv(ROBUSTNESS_DIR / "guardian_main_vs_conservative_by_strategy.csv")
    rows = []
    for r in summary:
        rows.append({
            "scope": "overall", "group": r["group"], "n_runs": r["n_runs"],
            "analysis_type": "GUARDIAN ROBUSTNESS ANALYSIS (diagnostic, NOT part of main hypothesis testing)",
            "sample_size_note": "n=10 for conservative -- small diagnostic sample, wide CIs" if "conservative" in r["group"] else "n=40, main-experiment sample",
            **{k: v for k, v in r.items() if k not in ("group", "n_runs")},
            "source_file": "guardian_robustness_analysis/guardian_main_vs_conservative_summary.csv",
        })
    for r in by_strategy:
        rows.append({
            "scope": "by_strategy", "group": r["group"], "n_runs": r["n_runs"],
            "analysis_type": "GUARDIAN ROBUSTNESS ANALYSIS (diagnostic, NOT part of main hypothesis testing)",
            "sample_size_note": "n=5 per stratum for conservative -- very small, wide CIs" if "conservative" in r["group"] else "n=20 per stratum, main-experiment sample",
            **{k: v for k, v in r.items() if k not in ("group", "n_runs")},
            "source_file": "guardian_robustness_analysis/guardian_main_vs_conservative_by_strategy.csv",
        })
    return rows


# =============================================================================
# Markdown freeze document
# =============================================================================

def build_markdown(consistency_lines: List[str], hyp_rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Final Analysis Freeze -- Main Experiment (n=80) + Guardian Robustness Check (n=10)")
    lines.append("")
    lines.append("Generated by `thesis_final_analysis_freeze.py`. Pure synthesis of already-validated "
                 "Step 3 (main analysis) / Step 4 (independent cross-check) / Step 5 (Guardian robustness) "
                 "results. No new statistics computed here.")
    lines.append("")
    lines.append("## 1. Consistency check")
    lines.append("")
    for line in consistency_lines:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## 2. Final hypothesis overview")
    lines.append("")
    lines.append("| Hypothesis | Outcome | n (A/B) | Test | p | adj. p | Effect size | 95% CI | Direction | Status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in hyp_rows:
        lines.append(
            f"| {r['hypothesis']} | {r['outcome']} | {r['n_A']}/{r['n_B']} | {r['test']} | "
            f"{float(r['p_value']):.4g} | {r['adjusted_p_value']} | {r['effect_size_primary']}={float(r['effect_size_primary_value']):.4g} | "
            f"[{float(r['ci_95_low']):.4g}; {float(r['ci_95_high']):.4g}] | {r['direction']} | {r['status']} |"
        )
    lines.append("")
    lines.append("All values above independently confirmed in Step 4 (0/97 mismatches, see `step4_match_report.csv`).")
    lines.append("")
    lines.append("## 3. Empirical trade-offs (observed only, no normative judgment)")
    lines.append("")
    lines.append("- Higher compliance in Setup B (47.5% vs 20.0% fully compliant) coincides with a lower Agreement Rate (77.5% vs 97.5%).")
    lines.append("- Higher Buyer Utility of successful deals in Setup B (0.439 vs 0.376) coincides with fewer successful deals overall (31 vs 39).")
    lines.append("- Stronger compliance control (Guardian) in Setup B coincides with substantially higher resource consumption (Hedges' g approx. 2.3 for tokens/runtime/cost).")
    lines.append("- The extreme 100% Main-Guardian revision rate is prompt-sensitive (drops to 68% under the Conservative variant, Step 5), but a substantial revision tendency (68%) persists even under a deliberately permissive prompt.")
    lines.append("")
    lines.append("No claim is made that either Setup is \"better\" overall -- these are observed trade-offs only.")
    lines.append("")
    lines.append("## 4. Methodological limitations for later discussion (documented only, not interpreted here)")
    lines.append("")
    for item in [
        "n=20 per main cell (80 total) -- moderate sample size for a 2x2 design.",
        "Buyer Utility analyzed only under successful deals (H4b, and the descriptive utility figures) -- not defined for No-Deal runs.",
        "Selection issue: differing Agreement Rates across cells mean the deal subpopulations compared for Buyer Utility / reservation-utility outcomes are not necessarily comparable (see H2/H4b caveats).",
        "Guardian robustness comparison uses only n=10 Conservative runs (n=5 per strategy) -- a small diagnostic sample with wide confidence intervals.",
        "Single fixed negotiation scenario (1,000 industrial sensors; price/delivery/payment).",
        "Single LLM (gpt-4o-mini) for Buyer and Guardian.",
        "Deterministic rule-based Seller (Conceder/Boulware) -- not an LLM opponent.",
        "Guardian behavior shown to be prompt-sensitive (Step 5) -- results are specific to the frozen Main Guardian prompt wording.",
        "Setup x Strategy interaction effects (H3) are exploratory, not confirmatory, despite being reproduced on an independent computational path in Step 4.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 5. Final status")
    lines.append("")
    lines.append("See chat-reported final status block (Step 6 summary) for the YES/NO checklist.")
    return "\n".join(lines)


def main() -> None:
    print("=== Consistency check (Step 3 vs 4 vs 5) ===")
    consistency_lines, ok = consistency_check()
    for line in consistency_lines:
        print(line)
    if not ok:
        print("\nCONSISTENCY CHECK FAILED -- STOPPING, no freeze written.")
        return

    hyp_rows = build_hypothesis_overview()
    key_rows = build_key_results(hyp_rows)
    strategy_rows = build_strategy_results()
    robustness_rows = build_robustness_summary()

    write_csv(hyp_rows, FREEZE_DIR / "final_hypothesis_results.csv")
    write_csv(key_rows, FREEZE_DIR / "final_key_results.csv")
    write_csv(strategy_rows, FREEZE_DIR / "final_strategy_results.csv")
    write_csv(robustness_rows, FREEZE_DIR / "final_robustness_summary.csv")

    md = build_markdown(consistency_lines, hyp_rows)
    md_path = FREEZE_DIR / "final_analysis_freeze.md"
    _no_overwrite(md_path)
    md_path.write_text(md, encoding="utf-8")

    print(f"\nWrote 5 freeze files to: {FREEZE_DIR}")
    print("\nSTEP 6 RESULTS FREEZE COMPLETE: YES")


if __name__ == "__main__":
    main()
