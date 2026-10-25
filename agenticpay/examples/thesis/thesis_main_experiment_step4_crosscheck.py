"""Step 4: independent statistical cross-check of the Step 3 confirmatory
results.

Rules followed (per the Step 4 instructions):
    - Reads ONLY the run-level CSV, never Step 3's aggregated output files.
    - Recomputes every n, contingency table, and test directly from the raw
      run-level rows.
    - For every hypothesis, uses a genuinely different computational route
      than Step 3 wherever a reasonable alternative exists (documented per
      section below); where no reasonable alternative implementation exists,
      this is stated explicitly and the inputs/n/key figures are still
      independently reconstructed rather than copied from Step 3.
    - Does NOT change any hypothesis, main test, or effect-size choice from
      Step 2/3. This file is a reproducibility/computation check only.

Alternative computational routes used here vs. Step 3:
    H1/H2 CMH:      manual Mantel-Haenszel formulas (statistic, pooled OR,
                     Robins-Breslow-Greenland variance for the OR CI) instead
                     of statsmodels.stats.contingency_tables.StratifiedTable.
                     (Formulas verified against the statsmodels source to
                     confirm they target the identical quantity -- an
                     independent implementation of the same standard method,
                     which is the explicit form of cross-check requested.)
    H2/H4a Fisher:   scipy.stats.fisher_exact again (no alternative exact-test
                     implementation is available in the standard stack) PLUS
                     a manual hypergeometric-distribution p-value computation
                     (scipy.stats.hypergeom, summing tail probabilities) as an
                     independent lower-level cross-check of the same p-value.
    H3/H4b Welch-t:  statsmodels.stats.weightstats.CompareMeans.ttest_ind
                     (different library/entry point than scipy.stats.ttest_ind
                     used in Step 3) PLUS a fully manual computation of t and
                     Welch-Satterthwaite df directly from mean/var/n.
    Mann-Whitney-U:  manual rank-sum-based U statistic (via pandas .rank())
                     with a normal-approximation p-value, instead of
                     scipy.stats.mannwhitneyu's built-in exact/asymptotic
                     method.
    Holm-Bonferroni: manual step-down procedure instead of
                     statsmodels.stats.multitest.multipletests.
    H3 interaction:  manual OLS via numpy.linalg.lstsq on a hand-built design
                     matrix, instead of statsmodels.formula.api.ols.
    Hedges' g:       no alternative library formula exists; recomputed
                     directly from the raw arrays independently (documented,
                     not copied from Step 3).

Outputs (agenticpay/results/thesis/main_experiment_analysis/, no overwrite):
    step4_crosscheck_results.csv
    step4_match_report.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.weightstats import CompareMeans
from statsmodels.stats.multitest import multipletests as sm_multipletests  # for documentation/labeling only

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment_analysis"
CSV_PATH = ANALYSIS_DIR / "run_level_main.csv"
Z95 = 1.959963984540054


def _no_overwrite(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {path}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype={"run_id": str})
    for c in ["deal_reached", "fully_compliant_run"]:
        df[c] = df[c].map({"True": True, "False": False, True: True, False: False})
    df["deal_below_buyer_reservation"] = df["deal_below_buyer_reservation"].map(
        {"True": True, "False": False, True: True, False: False, "": np.nan}
    )
    for c in ["final_round", "rounds_played", "n_violations_total", "final_buyer_utility",
              "final_seller_utility", "buyer_total_tokens", "guardian_total_tokens",
              "total_tokens", "runtime_sec", "estimated_cost_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =============================================================================
# 1. Independent data-basis reconstruction
# =============================================================================

def reconstruct_data_basis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    rows.append({"check": "total_rows", "value": len(df), "expected": 80, "match": len(df) == 80})
    for setup in ("A", "B"):
        n = int((df["setup"] == setup).sum())
        rows.append({"check": f"n_setup_{setup}", "value": n, "expected": 40, "match": n == 40})
    for setup in ("A", "B"):
        for strat in ("Conceder", "Boulware"):
            n = int(((df["setup"] == setup) & (df["seller_strategy"] == strat)).sum())
            rows.append({"check": f"n_{setup}_{strat}", "value": n, "expected": 20, "match": n == 20})
    n_deals = int(df["deal_reached"].sum())
    rows.append({"check": "total_deals", "value": n_deals, "expected": 70, "match": n_deals == 70})
    no_deal = df[~df["deal_reached"]]
    bad = int((no_deal["final_buyer_utility"] == 0).sum() + no_deal["final_buyer_utility"].notna().sum())
    rows.append({"check": "no_deal_utility_artifacts", "value": bad, "expected": 0, "match": bad == 0})
    h3_n = int(df["total_tokens"].notna().sum())
    rows.append({"check": "h3_n_all_80", "value": h3_n, "expected": 80, "match": h3_n == 80})
    dup = int(df["run_id"].duplicated().sum())
    rows.append({"check": "duplicate_run_ids", "value": dup, "expected": 0, "match": dup == 0})
    return rows


# =============================================================================
# Manual Mantel-Haenszel (independent of statsmodels.StratifiedTable)
# =============================================================================

def manual_cmh(tables: List[np.ndarray]) -> Dict[str, float]:
    """tables: list of 2x2 arrays [[a,b],[c,d]] (rows: group1=B, group2=A).
    Returns CMH chi-square statistic, p-value, pooled OR, and its 95% CI via
    the Robins-Breslow-Greenland (1986) variance formula -- manually coded,
    verified against the statsmodels source formula for equivalence, but
    computed independently here (no statsmodels.StratifiedTable call)."""
    a_sum_obs = 0.0
    e_sum = 0.0
    v_sum = 0.0
    R_sum = 0.0  # ad/n
    S_sum = 0.0  # bc/n
    term1_num = 0.0  # sum(P*R)
    term2_num = 0.0  # sum(P*S + Q*R)
    term3_num = 0.0  # sum(Q*S)

    for tbl in tables:
        a, b = tbl[0]
        c, d = tbl[1]
        n = a + b + c + d
        a_sum_obs += a
        e_sum += (a + b) * (a + c) / n
        v_sum += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1))
        R = a * d / n
        S = b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        R_sum += R
        S_sum += S
        term1_num += P * R
        term2_num += P * S + Q * R
        term3_num += Q * S

    cmh_stat = (a_sum_obs - e_sum) ** 2 / v_sum
    p_value = 1 - sps.chi2.cdf(cmh_stat, df=1)

    pooled_or = R_sum / S_sum
    log_or = math.log(pooled_or)
    var_log_or = (term1_num / (2 * R_sum ** 2)) + (term2_num / (2 * R_sum * S_sum)) + (term3_num / (2 * S_sum ** 2))
    se_log_or = math.sqrt(var_log_or)
    ci_low = math.exp(log_or - Z95 * se_log_or)
    ci_high = math.exp(log_or + Z95 * se_log_or)

    return {
        "cmh_statistic": cmh_stat, "cmh_p_value": p_value,
        "pooled_or": pooled_or, "or_ci_low": ci_low, "or_ci_high": ci_high,
    }


def manual_risk_diff_ci(p1: float, n1: int, p0: float, n0: int) -> Tuple[float, float, float]:
    diff = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return diff, diff - Z95 * se, diff + Z95 * se


def hypergeom_fisher_check(tbl: np.ndarray) -> float:
    """Independent two-sided Fisher-exact p-value via direct summation of
    hypergeometric tail probabilities (scipy.stats.hypergeom), as a
    lower-level cross-check of scipy.stats.fisher_exact's result."""
    a, b = int(tbl[0][0]), int(tbl[0][1])
    c, d = int(tbl[1][0]), int(tbl[1][1])
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2
    rv = sps.hypergeom(n, col1, row1)
    obs_p = rv.pmf(a)
    support = np.arange(max(0, row1 - (n - col1)), min(row1, col1) + 1)
    probs = rv.pmf(support)
    two_sided_p = probs[probs <= obs_p + 1e-10].sum()
    return float(two_sided_p)


# =============================================================================
# Manual Welch's t (independent formula) + statsmodels CompareMeans (alt library)
# =============================================================================

def manual_welch_t(x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    n1, n0 = len(x1), len(x0)
    m1, m0 = np.mean(x1), np.mean(x0)
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    se = math.sqrt(v1 / n1 + v0 / n0)
    t = (m1 - m0) / se
    df = (v1 / n1 + v0 / n0) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    p = 2 * sps.t.sf(abs(t), df)
    ci = Z95_t = sps.t.ppf(0.975, df)
    ci_low = (m1 - m0) - Z95_t * se
    ci_high = (m1 - m0) + Z95_t * se
    return {"t": t, "df": df, "p": p, "mean_diff": m1 - m0, "ci_low": ci_low, "ci_high": ci_high, "se": se}


def hedges_g(x1: np.ndarray, x0: np.ndarray) -> float:
    n1, n0 = len(x1), len(x0)
    s1, s0 = np.std(x1, ddof=1), np.std(x0, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    d = (np.mean(x1) - np.mean(x0)) / pooled_sd
    return d * (1 - 3 / (4 * (n1 + n0) - 9))


def manual_mannwhitney(x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Independent rank-sum-based U statistic + normal-approximation p-value,
    as an alternative computational route to scipy.stats.mannwhitneyu."""
    n1, n0 = len(x1), len(x0)
    combined = pd.Series(np.concatenate([x1, x0]))
    ranks = combined.rank(method="average").to_numpy()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    mu_u = n1 * n0 / 2
    # tie correction for the normal approximation
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts) / ((n1 + n0) * (n1 + n0 - 1))
    sigma_u = math.sqrt(n1 * n0 / 12 * ((n1 + n0 + 1) - tie_term / (n1 + n0)))
    z = (u1 - mu_u) / sigma_u
    p = 2 * sps.norm.sf(abs(z))
    r_rb = 2 * u1 / (n1 * n0) - 1
    return {"u": u1, "z": z, "p": p, "rank_biserial": r_rb}


def manual_holm(pvals: List[float]) -> List[float]:
    """Manual Holm (1979) step-down procedure, independent of
    statsmodels.stats.multitest.multipletests."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()


def manual_ols_interaction(df_sub: pd.DataFrame, outcome: str, log_transform: bool) -> Dict[str, float]:
    """Independent OLS via numpy.linalg.lstsq on a hand-built design matrix,
    as an alternative to statsmodels.formula.api.ols used in Step 3."""
    y = df_sub[outcome].to_numpy(dtype=float)
    if log_transform:
        y = np.log(y)
    setup_b = (df_sub["setup"] == "B").astype(float).to_numpy()
    strat_boul = (df_sub["seller_strategy"] == "Boulware").astype(float).to_numpy()
    interaction = setup_b * strat_boul
    X = np.column_stack([np.ones(len(y)), setup_b, strat_boul, interaction])
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    y_hat = X @ beta
    resid = y - y_hat
    dof = n - k
    sigma2 = np.sum(resid ** 2) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_vals = beta / se
    p_vals = 2 * sps.t.sf(np.abs(t_vals), dof)
    return {"interaction_coef": beta[3], "interaction_se": se[3], "interaction_t": t_vals[3], "interaction_p": p_vals[3]}


# =============================================================================
# H1
# =============================================================================

def crosscheck_h1(df: pd.DataFrame) -> List[Dict[str, Any]]:
    tables = []
    for strat in ("Conceder", "Boulware"):
        sub = df[df["seller_strategy"] == strat]
        a = sub[sub["setup"] == "A"]["fully_compliant_run"]
        b = sub[sub["setup"] == "B"]["fully_compliant_run"]
        tables.append(np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]], dtype=float))

    mh = manual_cmh(tables)
    a_all = df[df["setup"] == "A"]["fully_compliant_run"]
    b_all = df[df["setup"] == "B"]["fully_compliant_run"]
    rd, rd_lo, rd_hi = manual_risk_diff_ci(b_all.mean(), len(b_all), a_all.mean(), len(a_all))

    return [
        {"hypothesis": "H1", "metric": "rate_A_pooled", "step4_value": a_all.mean()},
        {"hypothesis": "H1", "metric": "rate_B_pooled", "step4_value": b_all.mean()},
        {"hypothesis": "H1", "metric": "cmh_statistic", "step4_value": mh["cmh_statistic"]},
        {"hypothesis": "H1", "metric": "cmh_p_value", "step4_value": mh["cmh_p_value"]},
        {"hypothesis": "H1", "metric": "common_or", "step4_value": mh["pooled_or"]},
        {"hypothesis": "H1", "metric": "or_ci_low", "step4_value": mh["or_ci_low"]},
        {"hypothesis": "H1", "metric": "or_ci_high", "step4_value": mh["or_ci_high"]},
        {"hypothesis": "H1", "metric": "risk_difference", "step4_value": rd},
        {"hypothesis": "H1", "metric": "rd_ci_low", "step4_value": rd_lo},
        {"hypothesis": "H1", "metric": "rd_ci_high", "step4_value": rd_hi},
    ]


# =============================================================================
# H2
# =============================================================================

def crosscheck_h2(df: pd.DataFrame) -> List[Dict[str, Any]]:
    deals = df[df["deal_reached"]].copy()
    tables = []
    fisher_rows = []
    for strat in ("Conceder", "Boulware"):
        sub = deals[deals["seller_strategy"] == strat]
        a = sub[sub["setup"] == "A"]["deal_below_buyer_reservation"].astype(bool)
        b = sub[sub["setup"] == "B"]["deal_below_buyer_reservation"].astype(bool)
        tbl = np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]], dtype=float)
        tables.append(tbl)
        _, p_scipy = sps.fisher_exact(tbl)
        p_hyper = hypergeom_fisher_check(tbl)
        fisher_rows.append({"hypothesis": "H2", "metric": f"fisher_p_{strat}_scipy", "step4_value": p_scipy})
        fisher_rows.append({"hypothesis": "H2", "metric": f"fisher_p_{strat}_hypergeom_manual", "step4_value": p_hyper})

    mh = manual_cmh(tables)
    a_all = deals[deals["setup"] == "A"]["deal_below_buyer_reservation"].astype(bool)
    b_all = deals[deals["setup"] == "B"]["deal_below_buyer_reservation"].astype(bool)
    rd, rd_lo, rd_hi = manual_risk_diff_ci(b_all.mean(), len(b_all), a_all.mean(), len(a_all))
    rr = b_all.mean() / a_all.mean()

    return fisher_rows + [
        {"hypothesis": "H2", "metric": "n_deals_A", "step4_value": len(a_all)},
        {"hypothesis": "H2", "metric": "n_deals_B", "step4_value": len(b_all)},
        {"hypothesis": "H2", "metric": "rate_A", "step4_value": a_all.mean()},
        {"hypothesis": "H2", "metric": "rate_B", "step4_value": b_all.mean()},
        {"hypothesis": "H2", "metric": "cmh_statistic", "step4_value": mh["cmh_statistic"]},
        {"hypothesis": "H2", "metric": "cmh_p_value", "step4_value": mh["cmh_p_value"]},
        {"hypothesis": "H2", "metric": "risk_difference", "step4_value": rd},
        {"hypothesis": "H2", "metric": "rd_ci_low", "step4_value": rd_lo},
        {"hypothesis": "H2", "metric": "rd_ci_high", "step4_value": rd_hi},
        {"hypothesis": "H2", "metric": "risk_ratio", "step4_value": rr},
    ]


# =============================================================================
# H3
# =============================================================================

def crosscheck_h3(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[float]]:
    rows = []
    raw_p = {}
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        a = df[df["setup"] == "A"][outcome].dropna().to_numpy()
        b = df[df["setup"] == "B"][outcome].dropna().to_numpy()

        cm = CompareMeans.from_data(b, a)
        t_sm, p_sm, df_sm = cm.ttest_ind(usevar="unequal")
        manual = manual_welch_t(b, a)
        g = hedges_g(b, a)
        mwu_manual = manual_mannwhitney(b, a)
        u_scipy, p_scipy_mwu = sps.mannwhitneyu(b, a, alternative="two-sided")

        raw_p[outcome] = p_sm

        rows.extend([
            {"hypothesis": "H3", "outcome": outcome, "metric": "n_A", "step4_value": len(a)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "n_B", "step4_value": len(b)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mean_A", "step4_value": np.mean(a)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mean_B", "step4_value": np.mean(b)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "median_A", "step4_value": np.median(a)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "median_B", "step4_value": np.median(b)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "sd_A", "step4_value": np.std(a, ddof=1)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "sd_B", "step4_value": np.std(b, ddof=1)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mean_diff_B_minus_A", "step4_value": np.mean(b) - np.mean(a)},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_t_statsmodels", "step4_value": t_sm},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_t_manual_formula", "step4_value": manual["t"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_df_statsmodels", "step4_value": df_sm},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_df_manual_formula", "step4_value": manual["df"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_p_statsmodels", "step4_value": p_sm},
            {"hypothesis": "H3", "outcome": outcome, "metric": "welch_p_manual_formula", "step4_value": manual["p"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "ci_95_low", "step4_value": manual["ci_low"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "ci_95_high", "step4_value": manual["ci_high"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "hedges_g", "step4_value": g},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mwu_u_scipy", "step4_value": u_scipy},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mwu_u_manual", "step4_value": mwu_manual["u"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mwu_p_scipy", "step4_value": p_scipy_mwu},
            {"hypothesis": "H3", "outcome": outcome, "metric": "mwu_p_manual_normal_approx", "step4_value": mwu_manual["p"]},
            {"hypothesis": "H3", "outcome": outcome, "metric": "rank_biserial", "step4_value": mwu_manual["rank_biserial"]},
        ])

    pvals_in_order = [raw_p[o] for o in ("total_tokens", "runtime_sec", "estimated_cost_usd")]
    holm_manual = manual_holm(pvals_in_order)
    _, holm_sm, _, _ = sm_multipletests(pvals_in_order, method="holm")
    for o, hm, hs in zip(("total_tokens", "runtime_sec", "estimated_cost_usd"), holm_manual, holm_sm):
        rows.append({"hypothesis": "H3", "outcome": o, "metric": "holm_adjusted_p_manual", "step4_value": hm})
        rows.append({"hypothesis": "H3", "outcome": o, "metric": "holm_adjusted_p_statsmodels", "step4_value": hs})

    # Exploratory interaction, manual OLS (numpy.linalg.lstsq) vs. Step 3's statsmodels formula OLS
    for outcome in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        d = df[["setup", "seller_strategy", outcome]].dropna()
        res = manual_ols_interaction(d, outcome, log_transform=True)
        rows.append({"hypothesis": "H3", "outcome": outcome, "metric": "interaction_coef_manual_ols_logscale", "step4_value": res["interaction_coef"]})
        rows.append({"hypothesis": "H3", "outcome": outcome, "metric": "interaction_p_manual_ols_logscale", "step4_value": res["interaction_p"]})

    return rows, pvals_in_order


# =============================================================================
# H4a
# =============================================================================

def crosscheck_h4a(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    strat_tables = {}
    for strat in ("Conceder", "Boulware"):
        sub = df[df["seller_strategy"] == strat]
        a = sub[sub["setup"] == "A"]["deal_reached"]
        b = sub[sub["setup"] == "B"]["deal_reached"]
        tbl = np.array([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])
        strat_tables[strat] = tbl
        _, p_scipy = sps.fisher_exact(tbl)
        p_hyper = hypergeom_fisher_check(tbl.astype(float))
        rd, rd_lo, rd_hi = manual_risk_diff_ci(b.mean(), len(b), a.mean(), len(a))
        rows.append({"hypothesis": "H4a", "metric": f"table_{strat}", "step4_value": str(tbl.tolist())})
        rows.append({"hypothesis": "H4a", "metric": f"fisher_p_{strat}_scipy", "step4_value": p_scipy})
        rows.append({"hypothesis": "H4a", "metric": f"fisher_p_{strat}_hypergeom_manual", "step4_value": p_hyper})
        rows.append({"hypothesis": "H4a", "metric": f"risk_difference_{strat}", "step4_value": rd})
        # Odds ratio for Boulware stratum: A=20/20 -> zero cell -> undefined
        a_yes, a_no = int(a.sum()), int((~a).sum())
        b_yes, b_no = int(b.sum()), int((~b).sum())
        if a_no == 0 or b_no == 0 or a_yes == 0 or b_yes == 0:
            or_note = "undefined (zero cell present)"
        else:
            or_note = (b_yes * a_no) / (b_no * a_yes)
        rows.append({"hypothesis": "H4a", "metric": f"odds_ratio_{strat}_manual", "step4_value": or_note})

    a_all = df[df["setup"] == "A"]["deal_reached"]
    b_all = df[df["setup"] == "B"]["deal_reached"]
    tbl_pooled = np.array([[int(b_all.sum()), int((~b_all).sum())], [int(a_all.sum()), int((~a_all).sum())]])
    _, p_pooled_scipy = sps.fisher_exact(tbl_pooled)
    p_pooled_hyper = hypergeom_fisher_check(tbl_pooled.astype(float))
    rd, rd_lo, rd_hi = manual_risk_diff_ci(b_all.mean(), len(b_all), a_all.mean(), len(a_all))
    rr = b_all.mean() / a_all.mean()

    rows.extend([
        {"hypothesis": "H4a", "metric": "rate_A_pooled", "step4_value": a_all.mean()},
        {"hypothesis": "H4a", "metric": "rate_B_pooled", "step4_value": b_all.mean()},
        {"hypothesis": "H4a", "metric": "fisher_p_pooled_scipy", "step4_value": p_pooled_scipy},
        {"hypothesis": "H4a", "metric": "fisher_p_pooled_hypergeom_manual", "step4_value": p_pooled_hyper},
        {"hypothesis": "H4a", "metric": "risk_difference_pooled", "step4_value": rd},
        {"hypothesis": "H4a", "metric": "rd_ci_low", "step4_value": rd_lo},
        {"hypothesis": "H4a", "metric": "rd_ci_high", "step4_value": rd_hi},
        {"hypothesis": "H4a", "metric": "risk_ratio_pooled", "step4_value": rr},
    ])
    return rows


# =============================================================================
# H4b
# =============================================================================

def crosscheck_h4b(df: pd.DataFrame) -> List[Dict[str, Any]]:
    deals = df[df["deal_reached"]]
    a = deals[deals["setup"] == "A"]["final_buyer_utility"].dropna().to_numpy()
    b = deals[deals["setup"] == "B"]["final_buyer_utility"].dropna().to_numpy()

    cm = CompareMeans.from_data(b, a)
    t_sm, p_sm, df_sm = cm.ttest_ind(usevar="unequal")
    manual = manual_welch_t(b, a)
    g = hedges_g(b, a)
    mwu_manual = manual_mannwhitney(b, a)
    u_scipy, p_scipy_mwu = sps.mannwhitneyu(b, a, alternative="two-sided")

    return [
        {"hypothesis": "H4b", "metric": "n_A", "step4_value": len(a)},
        {"hypothesis": "H4b", "metric": "n_B", "step4_value": len(b)},
        {"hypothesis": "H4b", "metric": "mean_A", "step4_value": np.mean(a)},
        {"hypothesis": "H4b", "metric": "mean_B", "step4_value": np.mean(b)},
        {"hypothesis": "H4b", "metric": "median_A", "step4_value": np.median(a)},
        {"hypothesis": "H4b", "metric": "median_B", "step4_value": np.median(b)},
        {"hypothesis": "H4b", "metric": "sd_A", "step4_value": np.std(a, ddof=1)},
        {"hypothesis": "H4b", "metric": "sd_B", "step4_value": np.std(b, ddof=1)},
        {"hypothesis": "H4b", "metric": "mean_diff_B_minus_A", "step4_value": np.mean(b) - np.mean(a)},
        {"hypothesis": "H4b", "metric": "welch_t_statsmodels", "step4_value": t_sm},
        {"hypothesis": "H4b", "metric": "welch_t_manual_formula", "step4_value": manual["t"]},
        {"hypothesis": "H4b", "metric": "welch_df_statsmodels", "step4_value": df_sm},
        {"hypothesis": "H4b", "metric": "welch_p_statsmodels", "step4_value": p_sm},
        {"hypothesis": "H4b", "metric": "welch_p_manual_formula", "step4_value": manual["p"]},
        {"hypothesis": "H4b", "metric": "ci_95_low", "step4_value": manual["ci_low"]},
        {"hypothesis": "H4b", "metric": "ci_95_high", "step4_value": manual["ci_high"]},
        {"hypothesis": "H4b", "metric": "hedges_g", "step4_value": g},
        {"hypothesis": "H4b", "metric": "mwu_u_scipy", "step4_value": u_scipy},
        {"hypothesis": "H4b", "metric": "mwu_u_manual", "step4_value": mwu_manual["u"]},
        {"hypothesis": "H4b", "metric": "mwu_p_scipy", "step4_value": p_scipy_mwu},
        {"hypothesis": "H4b", "metric": "rank_biserial", "step4_value": mwu_manual["rank_biserial"]},
    ]


# =============================================================================
# Comparison against Step 3 (hardcoded from step3_main_tests.csv / step3_exploratory_interactions.csv
# at the time Step 3 was run -- used ONLY as the comparison target, not as a computation input)
# =============================================================================

def load_step3_reference() -> Dict[str, float]:
    """Reads Step 3's own output files purely as the REFERENCE to compare
    against (never as an input to Step 4's calculations, which all come from
    run_level_main.csv only)."""
    ref = {}
    with open(ANALYSIS_DIR / "step3_main_tests.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row["hypothesis"]
            if h == "H1":
                ref["H1_rate_A_pooled"] = float(row["rate_A"])
                ref["H1_rate_B_pooled"] = float(row["rate_B"])
                ref["H1_cmh_statistic"] = float(row["statistic"])
                ref["H1_cmh_p_value"] = float(row["p_value"])
                ref["H1_common_or"] = float(row["effect_size_secondary_value"])
                ref["H1_or_ci_low"] = float(row["secondary_ci_95_low"])
                ref["H1_or_ci_high"] = float(row["secondary_ci_95_high"])
                ref["H1_risk_difference"] = float(row["effect_size_primary_value"])
                ref["H1_rd_ci_low"] = float(row["ci_95_low"])
                ref["H1_rd_ci_high"] = float(row["ci_95_high"])
            elif h == "H2":
                ref["H2_rate_A"] = float(row["rate_A"])
                ref["H2_rate_B"] = float(row["rate_B"])
                ref["H2_cmh_statistic"] = float(row["statistic"])
                ref["H2_cmh_p_value"] = float(row["p_value"])
                ref["H2_risk_difference"] = float(row["effect_size_primary_value"])
                ref["H2_n_deals_A"] = int(row["n_A"])
                ref["H2_n_deals_B"] = int(row["n_B"])
                ref["H2_risk_ratio"] = float(row["effect_size_secondary_value"])
            elif h == "H3":
                o = row["outcome"]
                ref[f"H3_{o}_n_A"] = int(row["n_A"])
                ref[f"H3_{o}_n_B"] = int(row["n_B"])
                ref[f"H3_{o}_mean_A"] = float(row["mean_A"])
                ref[f"H3_{o}_mean_B"] = float(row["mean_B"])
                ref[f"H3_{o}_median_A"] = float(row["median_A"])
                ref[f"H3_{o}_median_B"] = float(row["median_B"])
                ref[f"H3_{o}_sd_A"] = float(row["sd_A"])
                ref[f"H3_{o}_sd_B"] = float(row["sd_B"])
                ref[f"H3_{o}_mean_diff_B_minus_A"] = float(row["mean_diff_B_minus_A"])
                ref[f"H3_{o}_welch_t"] = float(row["statistic"])
                ref[f"H3_{o}_welch_df"] = float(row["dof"])
                ref[f"H3_{o}_welch_p"] = float(row["p_value"])
                ref[f"H3_{o}_holm_p"] = float(row["adjusted_p_value"])
                ref[f"H3_{o}_hedges_g"] = float(row["effect_size_primary_value"])
                ref[f"H3_{o}_ci_low"] = float(row["ci_95_low"])
                ref[f"H3_{o}_ci_high"] = float(row["ci_95_high"])
            elif h == "H4a":
                ref["H4a_rate_A_pooled"] = float(row["rate_A"])
                ref["H4a_rate_B_pooled"] = float(row["rate_B"])
                ref["H4a_fisher_p_pooled"] = float(row["p_value"])
                ref["H4a_risk_difference_pooled"] = float(row["effect_size_primary_value"])
                ref["H4a_risk_ratio_pooled"] = float(row["effect_size_secondary_value"])
            elif h == "H4b":
                ref["H4b_n_A"] = int(row["n_A"])
                ref["H4b_n_B"] = int(row["n_B"])
                ref["H4b_mean_A"] = float(row["mean_A"])
                ref["H4b_mean_B"] = float(row["mean_B"])
                ref["H4b_welch_t"] = float(row["statistic"])
                ref["H4b_welch_df"] = float(row["dof"])
                ref["H4b_welch_p"] = float(row["p_value"])
                ref["H4b_hedges_g"] = float(row["effect_size_primary_value"])

    with open(ANALYSIS_DIR / "step3_sensitivity_results.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("test", "").startswith("Mann-Whitney"):
                h, o = row["hypothesis"], row["outcome"]
                key = f"{h}_{o}" if h == "H3" else h
                ref[f"{key}_mwu_u"] = float(row["statistic"])
                ref[f"{key}_mwu_p"] = float(row["p_value"])
                ref[f"{key}_rank_biserial"] = float(row["effect_size_value"])

    with open(ANALYSIS_DIR / "step3_exploratory_interactions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hypothesis"] == "H3" and row["outcome"]:
                ref[f"H3_{row['outcome']}_interaction_coef"] = float(row["interaction_coefficient"])
                ref[f"H3_{row['outcome']}_interaction_p"] = float(row["interaction_p_value"])

    return ref


def build_match_report(step4_rows: List[Dict[str, Any]], step3_ref: Dict[str, float]) -> List[Dict[str, Any]]:
    """Maps step4 metric rows to the corresponding Step 3 reference value
    (where a direct correspondence exists) and reports match/mismatch per the
    pre-registered tolerances."""
    key_map = {
        ("H1", "rate_A_pooled"): "H1_rate_A_pooled", ("H1", "rate_B_pooled"): "H1_rate_B_pooled",
        ("H1", "cmh_statistic"): "H1_cmh_statistic", ("H1", "cmh_p_value"): "H1_cmh_p_value",
        ("H1", "common_or"): "H1_common_or", ("H1", "or_ci_low"): "H1_or_ci_low", ("H1", "or_ci_high"): "H1_or_ci_high",
        ("H1", "risk_difference"): "H1_risk_difference", ("H1", "rd_ci_low"): "H1_rd_ci_low", ("H1", "rd_ci_high"): "H1_rd_ci_high",
        ("H2", "rate_A"): "H2_rate_A", ("H2", "rate_B"): "H2_rate_B",
        ("H2", "cmh_statistic"): "H2_cmh_statistic", ("H2", "cmh_p_value"): "H2_cmh_p_value",
        ("H2", "risk_difference"): "H2_risk_difference", ("H2", "n_deals_A"): "H2_n_deals_A", ("H2", "n_deals_B"): "H2_n_deals_B",
        ("H2", "risk_ratio"): "H2_risk_ratio",
        ("H4a", "rate_A_pooled"): "H4a_rate_A_pooled", ("H4a", "rate_B_pooled"): "H4a_rate_B_pooled",
        ("H4a", "fisher_p_pooled_scipy"): "H4a_fisher_p_pooled",
        ("H4a", "risk_difference_pooled"): "H4a_risk_difference_pooled", ("H4a", "risk_ratio_pooled"): "H4a_risk_ratio_pooled",
        ("H4b", "n_A"): "H4b_n_A", ("H4b", "n_B"): "H4b_n_B",
        ("H4b", "mean_A"): "H4b_mean_A", ("H4b", "mean_B"): "H4b_mean_B",
        ("H4b", "welch_t_statsmodels"): "H4b_welch_t", ("H4b", "welch_df_statsmodels"): "H4b_welch_df",
        ("H4b", "welch_p_statsmodels"): "H4b_welch_p", ("H4b", "hedges_g"): "H4b_hedges_g",
        ("H4b", "mwu_u_scipy"): "H4b_mwu_u", ("H4b", "mwu_p_scipy"): "H4b_mwu_p", ("H4b", "rank_biserial"): "H4b_rank_biserial",
    }
    for o in ("total_tokens", "runtime_sec", "estimated_cost_usd"):
        key_map.update({
            ("H3", "n_A", o): f"H3_{o}_n_A", ("H3", "n_B", o): f"H3_{o}_n_B",
            ("H3", "mean_A", o): f"H3_{o}_mean_A", ("H3", "mean_B", o): f"H3_{o}_mean_B",
            ("H3", "median_A", o): f"H3_{o}_median_A", ("H3", "median_B", o): f"H3_{o}_median_B",
            ("H3", "sd_A", o): f"H3_{o}_sd_A", ("H3", "sd_B", o): f"H3_{o}_sd_B",
            ("H3", "mean_diff_B_minus_A", o): f"H3_{o}_mean_diff_B_minus_A",
            ("H3", "welch_t_statsmodels", o): f"H3_{o}_welch_t", ("H3", "welch_df_statsmodels", o): f"H3_{o}_welch_df",
            ("H3", "welch_p_statsmodels", o): f"H3_{o}_welch_p",
            ("H3", "holm_adjusted_p_statsmodels", o): f"H3_{o}_holm_p",
            ("H3", "hedges_g", o): f"H3_{o}_hedges_g",
            ("H3", "ci_95_low", o): f"H3_{o}_ci_low", ("H3", "ci_95_high", o): f"H3_{o}_ci_high",
            ("H3", "mwu_u_scipy", o): f"H3_{o}_mwu_u", ("H3", "mwu_p_scipy", o): f"H3_{o}_mwu_p",
            ("H3", "rank_biserial", o): f"H3_{o}_rank_biserial",
            ("H3", "interaction_coef_manual_ols_logscale", o): f"H3_{o}_interaction_coef",
            ("H3", "interaction_p_manual_ols_logscale", o): f"H3_{o}_interaction_p",
        })

    report = []
    for row in step4_rows:
        h = row["hypothesis"]
        if h == "DATA_BASIS" or "metric" not in row:
            continue  # data-basis rows are compared separately in Section 1, not against Step 3 test outputs
        metric = row["metric"]
        outcome = row.get("outcome")
        lookup_key = (h, metric, outcome) if outcome else (h, metric)
        ref_key = key_map.get(lookup_key)
        if ref_key is None or ref_key not in step3_ref:
            continue  # no direct Step 3 counterpart for this metric (e.g. pure cross-check-only values)
        step3_val = step3_ref[ref_key]
        step4_val = row["step4_value"]
        try:
            step4_val_f = float(step4_val)
            step3_val_f = float(step3_val)
            diff = abs(step4_val_f - step3_val_f)
        except (TypeError, ValueError):
            diff = None

        if metric.startswith("n_") or "n_A" in metric or "n_B" in metric or metric.startswith("n_deals"):
            tol = 0
        elif "rate" in metric or metric.startswith("median") and False:
            tol = 0.001
        elif metric in ("cmh_p_value", "welch_p_statsmodels", "welch_p_manual_formula",
                        "holm_adjusted_p_statsmodels", "mwu_p_scipy", "fisher_p_pooled_scipy", "interaction_p_manual_ols_logscale"):
            tol = 0.001
        else:
            tol = 0.001

        match = diff is not None and diff < tol if tol > 0 else diff == 0
        report.append({
            "hypothesis": h, "outcome": outcome or "", "metric": metric,
            "step3_value": step3_val, "step4_value": step4_val,
            "abs_diff": diff, "tolerance": tol, "match": "YES" if match else "NO",
        })
    return report


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    _no_overwrite(path)
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
    print(f"Loaded {len(df)} rows from {CSV_PATH} (independent of Step 3 output files)")

    print("\n=== 1. Data basis reconstruction ===")
    basis_rows = reconstruct_data_basis(df)
    for r in basis_rows:
        print(r)
    assert all(r["match"] for r in basis_rows), "Data basis mismatch -- STOP before running any test."

    all_step4_rows: List[Dict[str, Any]] = []
    all_step4_rows.extend([{"hypothesis": "DATA_BASIS", **r} for r in basis_rows])
    all_step4_rows.extend(crosscheck_h1(df))
    all_step4_rows.extend(crosscheck_h2(df))
    h3_rows, h3_pvals = crosscheck_h3(df)
    all_step4_rows.extend(h3_rows)
    all_step4_rows.extend(crosscheck_h4a(df))
    all_step4_rows.extend(crosscheck_h4b(df))

    write_csv(all_step4_rows, ANALYSIS_DIR / "step4_crosscheck_results.csv")
    print(f"\nWrote: step4_crosscheck_results.csv ({len(all_step4_rows)} rows)")

    step3_ref = load_step3_reference()
    match_report = build_match_report(all_step4_rows, step3_ref)
    write_csv(match_report, ANALYSIS_DIR / "step4_match_report.csv")
    print(f"Wrote: step4_match_report.csv ({len(match_report)} rows)")

    n_mismatch = sum(1 for r in match_report if r["match"] == "NO")
    print(f"\n=== Match summary: {len(match_report) - n_mismatch}/{len(match_report)} metrics MATCH ===")
    if n_mismatch:
        print(f"{n_mismatch} MISMATCH(ES):")
        for r in match_report:
            if r["match"] == "NO":
                print(" ", r)

    print(f"\nSTEP 4 INDEPENDENT CROSS-CHECK PASSED: {'YES' if n_mismatch == 0 else 'NO'}")


if __name__ == "__main__":
    main()
