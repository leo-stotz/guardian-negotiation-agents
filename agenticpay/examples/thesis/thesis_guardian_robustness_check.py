"""Guardian robustness check: conservative Guardian prompt variant, Setup B only.

Diagnostic-only robustness check (NOT a third main setup, NOT used to alter the
main experiment configuration -- see the thesis for the accompanying discussion).
Checks whether the very high strategic-revision rate observed with the frozen
main Guardian prompt is strongly dependent on that prompt's specific wording, by
re-running Setup B with everything else held identical except the Guardian's
system prompt (`ConservativeThesisGuardian` from
`agenticpay.agents.thesis_guardian_conservative`, restrained strategic-review
threshold, same compliance rules/codes/output format/technical structure).

Reuses `thesis_setup_b_test.run_negotiation()` completely unmodified -- that
file is never edited. The Guardian class it constructs internally
(`ThesisGuardian(model=guardian_model)`) is redirected to
`ConservativeThesisGuardian` only for the duration of each call here, via
`unittest.mock.patch.object` on the imported module's `ThesisGuardian` name.
Everything else `run_negotiation()` does -- scenario, buyer (incl. prompt),
seller strategies, utility functions, model parameters, compliance rules, deal
definition, the corrected exactly-20-real-rounds max_rounds logic (fixed in the
earlier, regression-tested in `agenticpay/test/test_thesis_max_rounds_fix.py`) -- is exercised exactly as in the main Setup B
runs.

Runs 5 reps x 2 strategies (conceder, boulware) = 10 runs total. All output
(raw run JSON + per-run `_eval.json` + `summary.json`) is written to a
dedicated, separate directory that never overlaps with any existing
calibration/verification/main-experiment path:
    agenticpay/results/thesis/guardian_robustness_check/

Usage: same as the other thesis setup scripts (OPENAI_API_KEY via local .env).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List
from unittest.mock import patch

from dotenv import load_dotenv

import agenticpay.examples.thesis.thesis_setup_b_test as setup_b
from agenticpay.agents.thesis_guardian_conservative import ConservativeThesisGuardian
from agenticpay.examples.thesis.thesis_scenario import REPO_ROOT, RESULTS_DIR

REPETITIONS = 5
STRATEGIES = ["conceder", "boulware"]
GUARDIAN_PROMPT_LABEL = "conservative"
ROBUSTNESS_DIR = RESULTS_DIR / "guardian_robustness_check"


def run_conservative_negotiation(strategy: str, run_id: str) -> Dict[str, Any]:
    """Run one Setup B negotiation with the conservative Guardian, via the real,
    unmodified `thesis_setup_b_test.run_negotiation()`, only redirecting the
    `ThesisGuardian` class it constructs internally."""
    with patch.object(setup_b, "ThesisGuardian", ConservativeThesisGuardian):
        return setup_b.run_negotiation(strategy=strategy, run_id=run_id, results_dir=ROBUSTNESS_DIR)


def summarize(strategy: str, rep: int, result: Dict[str, Any], duration_sec: float) -> Dict[str, Any]:
    plausibility = result["plausibility"]
    return {
        "guardian_prompt": GUARDIAN_PROMPT_LABEL,
        "setup": "setup_b",
        "strategy": strategy,
        "rep": rep,
        "deal_reached": result["deal_reached"],
        "final_round": result["final_round"],
        "rounds_played": result["rounds_played"],
        "max_rounds": result["max_rounds"],
        "final_buyer_utility": result["final_buyer_utility"],
        "final_seller_utility": result["final_seller_utility"],
        "agreed_contract": result["agreed_contract_thesis"],
        "guardian_revisions_triggered": plausibility.get("revisions_triggered"),
        "guardian_parse_failures": plausibility.get("guardian_parse_failures"),
        "no_reservation_violation_on_deal": plausibility.get("no_reservation_violation_on_deal"),
        "ran_without_error": plausibility.get("ran_without_error"),
        "error": result["error"],
        "duration_sec": round(duration_sec, 1),
    }


def main() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", flush=True)
        sys.exit(1)

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for strategy in STRATEGIES:
        for rep in range(1, REPETITIONS + 1):
            run_id = f"conservative_setup_b_{strategy}_rep{rep}"
            print(f"\n{'#' * 78}\n# guardian_prompt=conservative / setup_b / {strategy} / rep {rep} of {REPETITIONS}\n{'#' * 78}", flush=True)
            t0 = time.time()
            result = run_conservative_negotiation(strategy, run_id)
            duration = time.time() - t0

            out_path = ROBUSTNESS_DIR / f"{run_id}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

            summary = summarize(strategy, rep, result, duration)
            summaries.append(summary)
            print(json.dumps(summary, indent=2, default=str), flush=True)
            print(f"[{duration:.1f}s] Saved: {out_path}", flush=True)

            summary_path = ROBUSTNESS_DIR / "summary.json"
            summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    print(f"\n\n{'=' * 78}\nFULL SUMMARY ({len(summaries)} runs, guardian_prompt=conservative)\n{'=' * 78}", flush=True)
    for s in summaries:
        print(
            f"  {s['strategy']:10s} rep{s['rep']} "
            f"deal={s['deal_reached']!s:5} round={s['final_round']}/{s['max_rounds']} "
            f"rounds_played={s['rounds_played']} U_buyer={s['final_buyer_utility']} "
            f"guardian_revisions={s['guardian_revisions_triggered']}",
            flush=True,
        )
    print(f"\nSaved: {ROBUSTNESS_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
