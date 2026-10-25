"""Step 3 of the main-experiment statistical analysis: the confirmatory main
tests, exactly per the test plan frozen in Step 2 (no data-dependent test
switching -- the full rationale is documented in the thesis
and in the development log (not included in this repository)).

Reads ONLY the validated Step 1 dataset:
    agenticpay/results/thesis/main_experiment_analysis/run_level_main.csv
Never touches `main_experiment/` (raw data) or any calibration/robustness dir.

All established statistical tests use scipy.stats / statsmodels -- no
hand-rolled test implementations. Effect-size *formulas* that have no direct
one-line library equivalent (Hedges' g small-sample correction, Wald CIs for
risk difference/risk ratio, rank-biserial correlation from the Mann-Whitney U
statistic) are computed via their standard published formulas -- these are
descriptive-statistic formulas, not reimplementations of the tests themselves;
each is documented with its formula/source inline.

Pre-registered (Step 2), NOT changed here based on results:
    H1: CMH main test (fully_compliant_run ~ Setup, stratified by Strategy).
        Risk Difference primary effect size, common OR secondary. Breslow-Day
        heterogeneity test reported separately/exploratory -- never swaps out
        CMH as the main test.
    H2: Fisher Exact per Strategy stratum + CMH pooled, DEALS ONLY. Risk
        Difference/Risk Ratio. Descriptive-only "below reservation / all
        runs" ratio reported alongside, never substituted as the test.
    H3: Welch's t-test (fixed main test) for total_tokens, runtime_sec,
        estimated_cost_usd, all 80 runs. Hedges' g. Mann-Whitney-U +
        rank-biserial reported ONLY as a declared sensitivity analysis, never
        promoted to main test regardless of skew/outliers found. Holm-Bonferroni
        across the 3 unadjusted p-values.
    H4a: Fisher Exact pooled + per stratum for deal_reached. No logistic
        regression (A+Boulware = 20/20 -> quasi-complete separation). No OR as
        primary effect (undefined at a 100% cell) -- Risk Difference primary.
    H4b: Welch's t-test (fixed main test) for final_buyer_utility, DEALS ONLY.
        Hedges' g. Mann-Whitney-U + rank-biserial as sensitivity only.
    Setup x Strategy interactions for every outcome: EXPLORATORY ONLY, never
        used to alter H1-H4 conclusions.

Outputs (agenticpay/results/thesis/main_experiment_analysis/, no overwrite):
    step3_main_tests.csv
    step3_stratified_results.csv
    step3_sensitivity_results.csv
    step3_exploratory_interactions.csv
    step3_plausibility_checks.txt
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import CompareMeans

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment_analysis"
CSV_PATH = ANALYSIS_DIR / "run_level_main.csv"

Z95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975), used for Wald CIs


def _no_overwrite(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {path}")


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype={"run_id": str})
    # CSV stores blanks for None -- restore proper NaN for numeric/bool columns.
    bool_cols = ["deal_reached", "fully_compliant_run"]
    for c in bool_cols:
        df[c] = df[c].map({"True": True, "False": False, True: True, False: False})
    tri_bool_cols = ["deal_below_buyer_reservation"]
    for c in tri_bool_cols:
        # pandas may already parse True/False as native bool, or leave them as the
        # literal strings "True"/"False" -- map both forms; blank stays NaN (No-Deal).
        df[c] = df[c].map({"True": True, "False": False, True: True, False: False, "": np.nan})
    numeric_cols = [
        "final_round", "rounds_played", "n_violations_total",
        "MAX_PRICE_EXCEEDED", "OUT_OF_RANGE", "BELOW_RESERVATION_UTILITY",
        "INVALID_CONTRACT", "CONFIDENTIAL_INFO_DISCLOSED",
        "final_buyer_utility", "final_seller_utility",
        "buyer_total_tokens", "guardian_total_tokens", "total_tokens",
        "runtime_sec", "estimated_cost_usd", "guardian_calls", "guardian_revisions",
        "guardian_revision_rate",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =============================================================================
# Plausibility checks (must match Step 1 / Step 2 exactly)
# =============================================================================

def plausibility_checks(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    ok = True

    n_total = len(df)
    lines.append(f"Total rows: {n_total} (expected 80) -> {'OK' if n_total == 80 else 'MISMATCH'}")
    ok &= n_total == 80

    cell_counts = df.groupby(["setup", "seller_strategy"]).size()
    lines.append(f"Cell counts:\n{cell_counts.to_string()}")
    ok &= bool((cell_counts == 20).all())

    n_deals = int(df["deal_reached"].sum())
    lines.append(f"Total deals: {n_deals} (Step 1: 19+20+16+15=70) -> {'OK' if n_deals == 70 else 'MISMATCH'}")
    ok &= n_deals == 70

    deal_counts_by_cell = df[df["deal_reached"]].groupby(["setup", "seller_strategy"]).size()
    expected_deals = {("A", "Conceder"): 19, ("A", "Boulware"): 20, ("B", "Conceder"): 16, ("B", "Boulware"): 15}
    for k, v in expected_deals.items():
        actual = deal_counts_by_cell.get(k, 0)
        lines.append(f"Deals in {k}: {actual} (expected {v}) -> {'OK' if actual == v else 'MISMATCH'}")
        ok &= actual == v

    # No artificial zero utility on No-Deal
    no_deal = df[~df["deal_reached"]]
    bad_zero = no_deal[no_deal["final_buyer_utility"] == 0]
    lines.append(f"No-Deal runs with final_buyer_utility == 0 (must be 0): {len(bad_zero)} -> {'OK' if len(bad_zero) == 0 else 'MISMATCH'}")
    ok &= len(bad_zero) == 0
    no_deal_with_utility = no_deal[no_deal["final_buyer_utility"].notna()]
    lines.append(f"No-Deal runs with a non-null final_buyer_utility (must be 0): {len(no_deal_with_utility)} -> {'OK' if len(no_deal_with_utility) == 0 else 'MISMATCH'}")
    ok &= len(no_deal_with_utility) == 0

    # H3 uses all 80
    h3_n = df["total_tokens"].notna().sum()
    lines.append(f"Runs with non-null total_tokens (H3 must use all 80): {h3_n} -> {'OK' if h3_n == 80 else 'MISMATCH'}")
    ok &= h3_n == 80

    # H2/H4b deal-only counts
    lines.append(f"H2/H4b analysis n (deals only): {n_deals} (rows where deal_reached==True)")

    # Setup A has no guardian data, Setup B always does
    a_with_guardian = df[(df["setup"] == "A") & df["guardian_calls"].notna()]
    b_without_guardian = df[(df["setup"] == "B") & df["guardian_calls"].isna()]
    lines.append(f"Setup A runs with guardian_calls set (must be 0): {len(a_with_guardian)} -> {'OK' if len(a_with_guardian) == 0 else 'MISMATCH'}")
    lines.append(f"Setup B runs missing guardian_calls (must be 0): {len(b_without_guardian)} -> {'OK' if len(b_without_guardian) == 0 else 'MISMATCH'}")
    ok &= len(a_with_guardian) == 0 and len(b_without_guardian) == 0

    lines.append(f"\nOVERALL PLAUSIBILITY STATUS: {'PASS' if ok else 'FAIL -- STOP, see mismatches above'}")
    return lines, ok


# =============================================================================
# Effect-size helper formulas (standard published formulas, not test reimplementations)
# =============================================================================

def hedges_g(x1: np.ndarray, x0: np.ndarray) -> float:
    """Hedges' g: bias-corrected standardized mean difference (x1 - x0).
    Formula: d = (mean1-mean0)/pooled_sd; g = d * (1 - 3/(4*(n1+n0)-9))
    (Hedges & Olkin 1985 small-sample correction)."""
    n1, n0 = len(x1), len(x0)
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    d = (np.mean(x1) - np.mean(x0)) / pooled_sd
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    return d * correction


def rank_biserial_from_mannwhitney(u_stat: float, n1: int, n0: int) -> float:
    """Rank-biserial correlation from the Mann-Whitney U statistic for group 1
    vs group 0: r = 2*U/(n1*n0) - 1 (Wendt 1972 / Kerby 2014 formulation)."""
    return 2 * u_stat / (n1 * n0) - 1


def wald_ci_risk_difference(p1: float, n1: int, p0: float, n0: int) -> Tuple[float, float]:
    """95% Wald CI for a risk difference (p1 - p0)."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    diff = p1 - p0
    return diff - Z95 * se, diff + Z95 * se


def wald_ci_log_risk_ratio(p1: float, n1: int, x1: int, p0: float, n0: int, x0: int) -> Tuple[float, float]:
    """95% CI for a risk ratio (p1/p0) via the log-scale Wald method."""
    if x1 == 0 or x0 == 0 or p0 == 0:
        return (float("nan"), float("nan"))
    log_rr = math.log(p1 / p0)
    se_log = math.sqrt((1 - p1) / x1 + (1 - p0) / x0) if x1 > 0 and x0 > 0 else float("nan")
    lo, hi = log_rr - Z95 * se_log, log_rr + Z95 * se_log
    return math.exp(lo), math.exp(hi)


def skew_and_outliers(x: np.ndarray) -> Dict[str, Any]:
    """Descriptive-only: skewness (scipy) + IQR-rule outlier count."""
    skewness = float(sps.skew(x, bias=False))
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = x[(x < lo) | (x > hi)]
    return {"skewness": skewness, "n_outliers_iqr_rule": int(len(outliers)), "outlier_values": outliers.tolist()}


# =============================================================================
# H1: fully_compliant_run ~ Setup, stratified by Strategy (CMH main test)
# =============================================================================

def analyze_h1(df: pd.DataFrame) -> Dict[str, Any]:
    tables = []
    strat_rows = []
    for strat in ("Conceder", "Boulware"):
        sub = df[df["seller_strategy"] == strat]
        a = sub[sub["setup"] == "A"]["fully_compliant_run"]
        b = sub[sub["setup"] == "B"]["fully_compliant_run"]
        # Table2x2 expects [[a_yes, a_no],[b_yes, b_no]] as rows=exposure(Setup), cols=outcome
        tbl = np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])
        # rows: [B, A] so that odds ratio / risk diff are oriented as B vs A
        tables.append(tbl)
        strat_rows.append({
            "hypothesis": "H1", "seller_strategy": strat,
            "n_A": len(a), "n_B": len(b),
            "compliant_A": int(a.sum()), "compliant_B": int(b.sum()),
            "rate_A": a.mean(), "rate_B": b.mean(),
        })

    strat_table = StratifiedTable(tables)
    cmh_result = strat_table.test_null_odds()  # CMH chi-square test of common OR = 1
    common_or = strat_table.oddsratio_pooled
    or_ci = strat_table.oddsratio_pooled_confint()
    breslow_day = strat_table.test_equal_odds()  # heterogeneity test (exploratory)

    # Pooled (gepoolt) risk difference/risk ratio A vs B, ignoring stratification
    a_all = df[df["setup"] == "A"]["fully_compliant_run"]
    b_all = df[df["setup"] == "B"]["fully_compliant_run"]
    p_a, p_b = a_all.mean(), b_all.mean()
    n_a, n_b = len(a_all), len(b_all)
    rd = p_b - p_a
    rd_ci = wald_ci_risk_difference(p_b, n_b, p_a, n_a)
    rr = p_b / p_a if p_a > 0 else float("nan")
    rr_ci = wald_ci_log_risk_ratio(p_b, n_b, int(b_all.sum()), p_a, n_a, int(a_all.sum()))

    main_row = {
        "hypothesis": "H1", "outcome": "fully_compliant_run",
        "n_A": n_a, "n_B": n_b,
        "rate_A": p_a, "rate_B": p_b,
        "test": "Cochran-Mantel-Haenszel (stratified by seller_strategy)",
        "statistic": cmh_result.statistic, "p_value": cmh_result.pvalue,
        "adjusted_p_value": "",
        "effect_size_primary": "Risk Difference (B - A, pooled)", "effect_size_primary_value": rd,
        "ci_95_low": rd_ci[0], "ci_95_high": rd_ci[1],
        "effect_size_secondary": "Common Odds Ratio (CMH)", "effect_size_secondary_value": common_or,
        "secondary_ci_95_low": or_ci[0], "secondary_ci_95_high": or_ci[1],
        "risk_ratio_B_over_A": rr, "risk_ratio_ci_low": rr_ci[0], "risk_ratio_ci_high": rr_ci[1],
        "direction": "B higher than A" if rd > 0 else ("A higher than B" if rd < 0 else "no difference"),
    }

    exploratory_row = {
        "hypothesis": "H1", "analysis": "Breslow-Day heterogeneity test (Setup x Strategy interaction on OR scale)",
        "statistic": breslow_day.statistic, "p_value": breslow_day.pvalue,
        "note": "Exploratory only -- does not replace the pre-registered CMH main test.",
    }

    return {"main": main_row, "stratified": strat_rows, "exploratory": exploratory_row}


# =============================================================================
# H2: deal_below_buyer_reservation, DEALS ONLY, stratified by Strategy
# =============================================================================

def analyze_h2(df: pd.DataFrame) -> Dict[str, Any]:
    deals = df[df["deal_reached"]].copy()

    strat_rows = []
    fisher_tables = []
    for strat in ("Conceder", "Boulware"):
        sub = deals[deals["seller_strategy"] == strat]
        # .astype(bool) is safe here: deal_below_buyer_reservation is only NaN for
        # No-Deal runs, which are already excluded by the `deals` filter above.
        a = sub[sub["setup"] == "A"]["deal_below_buyer_reservation"].astype(bool)
        b = sub[sub["setup"] == "B"]["deal_below_buyer_reservation"].astype(bool)
        tbl_fisher = np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])
        odds, p_fisher = sps.fisher_exact(tbl_fisher)
        fisher_tables.append(tbl_fisher)
        strat_rows.append({
            "hypothesis": "H2", "seller_strategy": strat,
            "n_deals_A": len(a), "n_deals_B": len(b),
            "below_res_A": int(a.sum()), "below_res_B": int(b.sum()),
            "rate_A": a.mean(), "rate_B": b.mean(),
            "fisher_odds_ratio": odds, "fisher_p_value": p_fisher,
        })

    strat_table = StratifiedTable(fisher_tables)
    cmh_result = strat_table.test_null_odds()
    common_or = strat_table.oddsratio_pooled
    or_ci = strat_table.oddsratio_pooled_confint()

    a_all = deals[deals["setup"] == "A"]["deal_below_buyer_reservation"].astype(bool)
    b_all = deals[deals["setup"] == "B"]["deal_below_buyer_reservation"].astype(bool)
    p_a, p_b = a_all.mean(), b_all.mean()
    n_a, n_b = len(a_all), len(b_all)
    rd = p_b - p_a
    rd_ci = wald_ci_risk_difference(p_b, n_b, p_a, n_a)
    rr = p_b / p_a if p_a > 0 else float("nan")
    rr_ci = wald_ci_log_risk_ratio(p_b, n_b, int(b_all.sum()), p_a, n_a, int(a_all.sum()))

    main_row = {
        "hypothesis": "H2", "outcome": "deal_below_buyer_reservation (deals only)",
        "n_A": n_a, "n_B": n_b,
        "rate_A": p_a, "rate_B": p_b,
        "test": "Fisher Exact per stratum + CMH pooled (stratified by seller_strategy)",
        "statistic": cmh_result.statistic, "p_value": cmh_result.pvalue,
        "adjusted_p_value": "",
        "effect_size_primary": "Risk Difference (B - A, pooled among deals)", "effect_size_primary_value": rd,
        "ci_95_low": rd_ci[0], "ci_95_high": rd_ci[1],
        "effect_size_secondary": "Risk Ratio (B / A, pooled among deals)", "effect_size_secondary_value": rr,
        "secondary_ci_95_low": rr_ci[0], "secondary_ci_95_high": rr_ci[1],
        "direction": "B lower than A" if rd < 0 else ("A lower than B" if rd > 0 else "no difference"),
        "selection_caveat": "Agreement Rate differs by cell (A=39/40, B=31/40 deals) -- this compares potentially non-comparable deal subpopulations, see Step 2.",
    }

    # Descriptive-only: below-reservation / ALL runs per cell (not a test)
    desc_rows = []
    for setup in ("A", "B"):
        for strat in ("Conceder", "Boulware"):
            cell = df[(df["setup"] == setup) & (df["seller_strategy"] == strat)]
            below_all = int((cell["deal_below_buyer_reservation"] == True).sum())
            desc_rows.append({
                "cell": f"{setup}+{strat}", "n_all_runs": len(cell),
                "n_below_reservation": below_all,
                "share_below_reservation_of_all_runs": below_all / len(cell),
            })

    return {"main": main_row, "stratified": strat_rows, "descriptive_all_runs": desc_rows}


# =============================================================================
# H3: Tokens, runtime, cost -- Welch's t-test (fixed main test), all 80 runs
# =============================================================================

def analyze_continuous_welch(
    df: pd.DataFrame, outcome: str, hypothesis: str, subset_mask: Optional[pd.Series] = None, label: str = ""
) -> Dict[str, Any]:
    data = df if subset_mask is None else df[subset_mask]
    a = data[data["setup"] == "A"][outcome].dropna().to_numpy()
    b = data[data["setup"] == "B"][outcome].dropna().to_numpy()

    # Welch's t-test (scipy), main confirmatory test -- fixed regardless of diagnostics
    t_stat, p_val = sps.ttest_ind(b, a, equal_var=False)
    dof = sps.ttest_ind(b, a, equal_var=False).df if hasattr(sps.ttest_ind(b, a, equal_var=False), "df") else None
    cm = CompareMeans.from_data(b, a)
    ci_low, ci_high = cm.tconfint_diff(usevar="unequal")
    g = hedges_g(b, a)

    # Sensitivity: Mann-Whitney-U (never replaces Welch as main test)
    u_stat, p_mwu = sps.mannwhitneyu(b, a, alternative="two-sided")
    rbc = rank_biserial_from_mannwhitney(u_stat, len(b), len(a))

    dist_a = skew_and_outliers(a)
    dist_b = skew_and_outliers(b)

    main_row = {
        "hypothesis": hypothesis, "outcome": f"{outcome}{(' (' + label + ')') if label else ''}",
        "n_A": len(a), "n_B": len(b),
        "mean_A": np.mean(a), "median_A": np.median(a), "sd_A": np.std(a, ddof=1),
        "mean_B": np.mean(b), "median_B": np.median(b), "sd_B": np.std(b, ddof=1),
        "test": "Welch's t-test (equal_var=False)",
        "statistic": t_stat, "dof": dof, "p_value": p_val,
        "adjusted_p_value": "",
        "mean_diff_B_minus_A": np.mean(b) - np.mean(a),
        "ci_95_low": ci_low, "ci_95_high": ci_high,
        "effect_size_primary": "Hedges' g", "effect_size_primary_value": g,
        "direction": "B higher than A" if np.mean(b) > np.mean(a) else ("A higher than B" if np.mean(b) < np.mean(a) else "no difference"),
    }
    sens_row = {
        "hypothesis": hypothesis, "outcome": f"{outcome}{(' (' + label + ')') if label else ''}",
        "test": "Mann-Whitney-U (sensitivity analysis, not main test)",
        "statistic": u_stat, "p_value": p_mwu,
        "effect_size": "rank-biserial correlation", "effect_size_value": rbc,
        "n_A": len(a), "n_B": len(b),
    }
    dist_row = {
        "hypothesis": hypothesis, "outcome": f"{outcome}{(' (' + label + ')') if label else ''}",
        "skewness_A": dist_a["skewness"], "n_outliers_A_iqr_rule": dist_a["n_outliers_iqr_rule"],
        "skewness_B": dist_b["skewness"], "n_outliers_B_iqr_rule": dist_b["n_outliers_iqr_rule"],
    }
    return {"main": main_row, "sensitivity": sens_row, "distribution": dist_row, "raw": {"a": a, "b": b}}


def analyze_h3(df: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        results[outcome] = analyze_continuous_welch(df, outcome, "H3")

    # Holm-Bonferroni across the 3 H3 p-values
    pvals = [results[o]["main"]["p_value"] for o in ("total_tokens", "runtime_sec", "estimated_cost_usd")]
    reject, p_adj, _, _ = multipletests(pvals, method="holm")
    for o, p in zip(("total_tokens", "runtime_sec", "estimated_cost_usd"), p_adj):
        results[o]["main"]["adjusted_p_value"] = p

    # Descriptive breakdown by Seller Strategy (not a test)
    strategy_desc = []
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        for strat in ("Conceder", "Boulware"):
            for setup in ("A", "B"):
                cell = df[(df["setup"] == setup) & (df["seller_strategy"] == strat)][outcome].dropna()
                strategy_desc.append({
                    "hypothesis": "H3", "outcome": outcome, "setup": setup, "seller_strategy": strat,
                    "n": len(cell), "mean": cell.mean(), "median": cell.median(), "sd": cell.std(ddof=1),
                })

    # Exploratory factorial model: outcome ~ Setup * Strategy (OLS on log-transformed outcome, since
    # tokens/runtime/cost are right-skewed and non-negative)
    exploratory = []
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        d = df[["setup", "seller_strategy", outcome]].dropna().copy()
        d["log_outcome"] = np.log(d[outcome])
        d["setup_B"] = (d["setup"] == "B").astype(int)
        d["strategy_boulware"] = (d["seller_strategy"] == "Boulware").astype(int)
        import statsmodels.formula.api as smf
        model = smf.ols("log_outcome ~ setup_B * strategy_boulware", data=d).fit()
        interaction_coef = model.params.get("setup_B:strategy_boulware", float("nan"))
        interaction_p = model.pvalues.get("setup_B:strategy_boulware", float("nan"))
        exploratory.append({
            "hypothesis": "H3", "outcome": outcome,
            "analysis": "OLS on log(outcome) ~ Setup * Strategy (exploratory interaction test)",
            "interaction_coefficient": interaction_coef, "interaction_p_value": interaction_p,
            "note": "Exploratory only -- does not replace the Welch main test or alter H3.",
        })

    return {"results": results, "strategy_descriptive": strategy_desc, "exploratory": exploratory}


# =============================================================================
# H4a: deal_reached (undirected), Fisher Exact pooled + per stratum
# =============================================================================

def analyze_h4a(df: pd.DataFrame) -> Dict[str, Any]:
    strat_rows = []
    for strat in ("Conceder", "Boulware"):
        sub = df[df["seller_strategy"] == strat]
        a = sub[sub["setup"] == "A"]["deal_reached"]
        b = sub[sub["setup"] == "B"]["deal_reached"]
        tbl = np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])
        odds, p_fisher = sps.fisher_exact(tbl)
        rate_a, rate_b = a.mean(), b.mean()
        n_a, n_b = len(a), len(b)
        rd = rate_b - rate_a
        rd_ci = wald_ci_risk_difference(rate_b, n_b, rate_a, n_a)
        strat_rows.append({
            "hypothesis": "H4a", "seller_strategy": strat,
            "n_A": n_a, "n_B": n_b, "deals_A": int(a.sum()), "deals_B": int(b.sum()),
            "rate_A": rate_a, "rate_B": rate_b,
            "fisher_odds_ratio": odds if not math.isinf(odds) else "undefined (zero cell)",
            "fisher_p_value": p_fisher,
            "risk_difference": rd, "rd_ci_95_low": rd_ci[0], "rd_ci_95_high": rd_ci[1],
        })

    a_all = df[df["setup"] == "A"]["deal_reached"]
    b_all = df[df["setup"] == "B"]["deal_reached"]
    tbl_pooled = np.array([[int(b_all.sum()), int((~b_all).sum())], [int(a_all.sum()), int((~a_all).sum())]])
    odds_pooled, p_pooled = sps.fisher_exact(tbl_pooled)
    p_a, p_b = a_all.mean(), b_all.mean()
    n_a, n_b = len(a_all), len(b_all)
    rd = p_b - p_a
    rd_ci = wald_ci_risk_difference(p_b, n_b, p_a, n_a)
    rr = p_b / p_a if p_a > 0 else float("nan")
    rr_ci = wald_ci_log_risk_ratio(p_b, n_b, int(b_all.sum()), p_a, n_a, int(a_all.sum()))
    # Cramer's V (descriptive only) via chi2 on the pooled 2x2 (Yates-uncorrected, for descriptive V only)
    chi2, _, _, _ = sps.chi2_contingency(tbl_pooled, correction=False)
    cramers_v = math.sqrt(chi2 / (tbl_pooled.sum() * (min(tbl_pooled.shape) - 1)))

    main_row = {
        "hypothesis": "H4a", "outcome": "deal_reached (undirected)",
        "n_A": n_a, "n_B": n_b, "rate_A": p_a, "rate_B": p_b,
        "test": "Fisher Exact Test (pooled A vs B)",
        "statistic": "N/A (Fisher exact has no separate test statistic)", "p_value": p_pooled,
        "adjusted_p_value": "",
        "effect_size_primary": "Risk Difference (B - A, pooled)", "effect_size_primary_value": rd,
        "ci_95_low": rd_ci[0], "ci_95_high": rd_ci[1],
        "effect_size_secondary": "Risk Ratio (B / A, pooled)", "effect_size_secondary_value": rr,
        "secondary_ci_95_low": rr_ci[0], "secondary_ci_95_high": rr_ci[1],
        "cramers_v_descriptive": cramers_v,
        "odds_ratio_note": (
            f"Pooled OR = {odds_pooled:.4f}, but NOT used as primary effect size: "
            "the Boulware stratum (A=20/20=100%) yields an undefined/infinite stratum-specific OR "
            "(zero cell) -- Risk Difference used as primary effect size instead, per Step 2."
        ),
        "direction": "B higher than A" if rd > 0 else ("A higher than B" if rd < 0 else "no difference"),
    }
    return {"main": main_row, "stratified": strat_rows}


# =============================================================================
# H4b: final_buyer_utility, DEALS ONLY -- Welch's t-test (fixed main test)
# =============================================================================

def analyze_h4b(df: pd.DataFrame) -> Dict[str, Any]:
    deals_mask = df["deal_reached"]
    result = analyze_continuous_welch(df, "final_buyer_utility", "H4b", subset_mask=deals_mask, label="deals only")
    result["main"]["selection_caveat"] = (
        "Conditional on deal_reached=True; not independent of H4a (Agreement Rate differs by cell: "
        "n_deals A=39/40, B=31/40). See Step 2 discussion."
    )

    strategy_desc = []
    for strat in ("Conceder", "Boulware"):
        for setup in ("A", "B"):
            cell = df[(df["setup"] == setup) & (df["seller_strategy"] == strat) & (df["deal_reached"])]["final_buyer_utility"]
            strategy_desc.append({
                "hypothesis": "H4b", "setup": setup, "seller_strategy": strat,
                "n_deals": len(cell), "mean": cell.mean(), "median": cell.median(), "sd": cell.std(ddof=1),
            })

    import statsmodels.formula.api as smf
    d = df[deals_mask][["setup", "seller_strategy", "final_buyer_utility"]].dropna().copy()
    d["setup_B"] = (d["setup"] == "B").astype(int)
    d["strategy_boulware"] = (d["seller_strategy"] == "Boulware").astype(int)
    model = smf.ols("final_buyer_utility ~ setup_B * strategy_boulware", data=d).fit()
    interaction_coef = model.params.get("setup_B:strategy_boulware", float("nan"))
    interaction_p = model.pvalues.get("setup_B:strategy_boulware", float("nan"))
    exploratory = [{
        "hypothesis": "H4b", "outcome": "final_buyer_utility",
        "analysis": "OLS: final_buyer_utility ~ Setup * Strategy, deals only (exploratory interaction test)",
        "interaction_coefficient": interaction_coef, "interaction_p_value": interaction_p,
        "note": "Exploratory only -- does not replace the Welch main test or alter H4b.",
    }]

    return {"main": result["main"], "sensitivity": result["sensitivity"], "distribution": result["distribution"],
            "strategy_descriptive": strategy_desc, "exploratory": exploratory}


# =============================================================================
# Output writers
# =============================================================================

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
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    print("\n=== Plausibility checks ===")
    plaus_lines, plaus_ok = plausibility_checks(df)
    for line in plaus_lines:
        print(line)
    plaus_path = ANALYSIS_DIR / "step3_plausibility_checks.txt"
    _no_overwrite(plaus_path)
    plaus_path.write_text("\n".join(plaus_lines), encoding="utf-8")

    if not plaus_ok:
        print("\nPLAUSIBILITY CHECK FAILED -- STOPPING before running any test.")
        return

    h1 = analyze_h1(df)
    h2 = analyze_h2(df)
    h3 = analyze_h3(df)
    h4a = analyze_h4a(df)
    h4b = analyze_h4b(df)

    main_tests = [h1["main"], h2["main"]]
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        main_tests.append(h3["results"][outcome]["main"])
    main_tests.append(h4a["main"])
    main_tests.append(h4b["main"])
    write_csv(main_tests, ANALYSIS_DIR / "step3_main_tests.csv")

    stratified = list(h1["stratified"]) + list(h2["stratified"]) + list(h4a["stratified"]) \
        + h3["strategy_descriptive"] + h4b["strategy_descriptive"] + h2["descriptive_all_runs"]
    write_csv(stratified, ANALYSIS_DIR / "step3_stratified_results.csv")

    sensitivity = [h3["results"][o]["sensitivity"] for o in ("total_tokens", "runtime_sec", "estimated_cost_usd")]
    sensitivity.append(h4b["sensitivity"])
    dist_rows = [h3["results"][o]["distribution"] for o in ("total_tokens", "runtime_sec", "estimated_cost_usd")]
    dist_rows.append(h4b["distribution"])
    write_csv(sensitivity + dist_rows, ANALYSIS_DIR / "step3_sensitivity_results.csv")

    exploratory = [h1["exploratory"]] + h3["exploratory"] + h4b["exploratory"]
    write_csv(exploratory, ANALYSIS_DIR / "step3_exploratory_interactions.csv")

    print("\n=== H1 (CMH) ===")
    print(h1["main"])
    print("\n=== H2 (Fisher/CMH, deals only) ===")
    print(h2["main"])
    print("\n=== H3 (Welch, 3 outcomes) ===")
    for o in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        print(h3["results"][o]["main"])
    print("\n=== H4a (Fisher, undirected) ===")
    print(h4a["main"])
    print("\n=== H4b (Welch, deals only) ===")
    print(h4b["main"])

    print(f"\nWrote: step3_main_tests.csv, step3_stratified_results.csv, "
          f"step3_sensitivity_results.csv, step3_exploratory_interactions.csv, step3_plausibility_checks.txt")
    print(f"All in: {ANALYSIS_DIR}")

    print("\nSTEP 3 MAIN STATISTICAL ANALYSIS COMPLETE: YES")


if __name__ == "__main__":
    main()
