"""Appendix M: detailed exploratory Setup x Strategy interaction results for the
three H3 resource metrics (total_tokens, runtime_sec, estimated_cost_usd).

Re-estimates EXACTLY the exploratory model already used in
thesis_main_experiment_step3_stats.py::analyze_h3 --
    OLS: log(outcome) ~ setup_B * strategy_boulware   (treatment coding,
    reference cell = Setup A / Conceder)
-- and reports the full set of statistics that step3 only summarised
(coefficient, SE, t, p, 95% CI, equivalent Type-III F, ratio scale) plus the
four cell means/medians on the raw scale. Robustness (reported, not
substituted): HC3 robust SEs on the log model, and OLS on the raw scale.

Exploratory only -- never used to alter the H3 conclusions.

Reads ONLY: agenticpay/results/thesis/main_experiment_analysis/run_level_main.csv
Writes (no overwrite): .../main_experiment_analysis/appendix_m_interactions.csv
                       .../main_experiment_analysis/appendix_m_cell_means.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_main_experiment_step3_stats import ANALYSIS_DIR, load_data, write_csv  # noqa: E402

OUTCOMES = ("total_tokens", "runtime_sec", "estimated_cost_usd")
TERM = "setup_B:strategy_boulware"


def main() -> None:
    df = load_data()
    assert len(df) == 80, f"expected n=80, got {len(df)}"

    rows, cells = [], []
    for outcome in OUTCOMES:
        d = df[["setup", "seller_strategy", outcome]].dropna().copy()
        d["log_outcome"] = np.log(d[outcome])
        d["setup_B"] = (d["setup"] == "B").astype(int)
        d["strategy_boulware"] = (d["seller_strategy"] == "Boulware").astype(int)

        m_log = smf.ols("log_outcome ~ setup_B * strategy_boulware", data=d).fit()
        m_hc3 = smf.ols("log_outcome ~ setup_B * strategy_boulware", data=d).fit(cov_type="HC3")
        m_raw = smf.ols(f"{outcome} ~ setup_B * strategy_boulware", data=d).fit()
        # Type-III ANOVA with sum-to-zero coding: F for the interaction equals t^2 in a 2x2 design
        m_anova = smf.ols("log_outcome ~ C(setup, Sum) * C(seller_strategy, Sum)", data=d).fit()
        aov = sm.stats.anova_lm(m_anova, typ=3)
        f_row = aov.loc["C(setup, Sum):C(seller_strategy, Sum)"]

        ci_lo, ci_hi = m_log.conf_int().loc[TERM]
        rows.append({
            "outcome": outcome, "n": int(m_log.nobs), "df_resid": int(m_log.df_resid),
            "model": "OLS log(outcome) ~ Setup * Strategy (ref: A/Conceder)",
            "b_setupB_log": m_log.params["setup_B"], "p_setupB_log": m_log.pvalues["setup_B"],
            "b_boulware_log": m_log.params["strategy_boulware"],
            "p_boulware_log": m_log.pvalues["strategy_boulware"],
            "b_interaction_log": m_log.params[TERM], "se_interaction_log": m_log.bse[TERM],
            "t_interaction": m_log.tvalues[TERM], "p_interaction": m_log.pvalues[TERM],
            "ci95_lo_log": ci_lo, "ci95_hi_log": ci_hi,
            "ratio_of_ratios_exp_b": float(np.exp(m_log.params[TERM])),
            "ratio_ci95_lo": float(np.exp(ci_lo)), "ratio_ci95_hi": float(np.exp(ci_hi)),
            "F_interaction_typeIII": f_row["F"], "p_F_interaction": f_row["PR(>F)"],
            "r2_log_model": m_log.rsquared,
            "hc3_z_interaction": m_hc3.tvalues[TERM], "hc3_p_interaction": m_hc3.pvalues[TERM],
            "raw_b_interaction": m_raw.params[TERM], "raw_t_interaction": m_raw.tvalues[TERM],
            "raw_p_interaction": m_raw.pvalues[TERM],
            "note": "Exploratory only -- does not replace the Welch main test or alter H3.",
        })

        for setup in ("A", "B"):
            for strat in ("Conceder", "Boulware"):
                c = d[(d["setup"] == setup) & (d["seller_strategy"] == strat)][outcome]
                cells.append({
                    "outcome": outcome, "setup": setup, "seller_strategy": strat, "n": len(c),
                    "mean": c.mean(), "sd": c.std(ddof=1), "median": c.median(),
                    "geometric_mean": float(np.exp(np.log(c).mean())),
                })

    write_csv(rows, ANALYSIS_DIR / "appendix_m_interactions.csv")
    write_csv(cells, ANALYSIS_DIR / "appendix_m_cell_means.csv")

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(pd.DataFrame(rows).drop(columns=["model", "note"]).T.to_string())
    print()
    print(pd.DataFrame(cells).to_string(index=False))


if __name__ == "__main__":
    main()
