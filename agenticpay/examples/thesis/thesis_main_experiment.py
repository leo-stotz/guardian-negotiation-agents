"""Main-experiment batch script: the final 80-run main sample.

Four cells (Setup A/B x Conceder/Boulware), 20 repetitions each, 80 runs total.
This is the frozen main sample for the thesis -- the sample size (20/cell, 80
total) is fixed before data collection starts and is not adjusted during it
(main-experiment freeze, documented in the development log (not included in this repository)).

Reuses `run_negotiation()` directly from `thesis_setup_a_test.py` /
`thesis_setup_b_test.py` completely unmodified (same pattern as
the earlier variance study, not included in this repository) -- no negotiation logic is duplicated here, and
Setup B's `run_negotiation()` constructs its own `ThesisGuardian` from
`agenticpay.agents.thesis_guardian` (the frozen main Guardian) internally;
nothing in this file patches, subclasses, or otherwise redirects that class.
`agenticpay.agents.thesis_guardian_conservative` is never imported here, not
even indirectly (verified: `thesis_setup_a_test.py`/`thesis_setup_b_test.py`
themselves never import it either -- pre-flight check documented in the development log).

No AgenticPay core file, Buyer/Guardian/Seller prompt, utility function,
evaluator, or scenario parameter is touched by this file -- it only
orchestrates calls to already-existing, frozen functions.

Output: `agenticpay/results/thesis/main_experiment/` -- a dedicated directory
used EXCLUSIVELY for these 80 main runs. Never used for calibration,
diagnostics, robustness checks, or pre-flight data (those each have their own
directory: `calibration/`, `guardian_precision_test/`, `variance/`,
`guardian_robustness_check/`, `_preflight_dry_run/`, `_max_rounds_fix_regression_test/`,
etc.).

Run-ID scheme (exact, zero-padded 2 digits): `{setup}_{strategy}_rep{01..20}`,
e.g. `setup_a_conceder_rep01` .. `setup_a_conceder_rep20`.

Collision safety: `build_plan()` and `check_collisions()` are pure,
side-effect-free (only *read* the filesystem, never write, never call an LLM)
so the planned run count/per-cell counts/run-id uniqueness/collision status
can be verified WITHOUT starting any negotiation -- exactly what the
pre-flight check needs. `main()` (the only side-effecting entry point) always
runs `check_collisions()` first and ABORTS BEFORE EXECUTING ANY RUN if a
single collision is found anywhere in the planned 80 -- it never silently
skips a colliding run and continues with the rest. `run_one()` additionally
re-checks immediately before executing, as a defense-in-depth measure against
a file appearing between the upfront check and that specific run.

Usage: same as the other thesis setup scripts (OPENAI_API_KEY via local .env).
This is expensive (80 negotiations, Setup B up to 3 LLM calls/round) -- run in
the background. NOT started by this file being created or imported; only
`main()` (i.e. `python agenticpay/examples/thesis/thesis_main_experiment.py`)
executes anything.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

from dotenv import load_dotenv

import agenticpay.examples.thesis.thesis_setup_a_test as setup_a
import agenticpay.examples.thesis.thesis_setup_b_test as setup_b
from agenticpay.examples.thesis.thesis_scenario import REPO_ROOT, RESULTS_DIR

REPETITIONS = 20
STRATEGIES = ["conceder", "boulware"]
CELLS: List[Tuple[str, Callable[..., Dict[str, Any]]]] = [
    ("setup_a", setup_a.run_negotiation),
    ("setup_b", setup_b.run_negotiation),
]
MAIN_EXPERIMENT_DIR = RESULTS_DIR / "main_experiment"


class PlannedRun(NamedTuple):
    setup: str
    strategy: str
    rep: int
    run_id: str
    run_fn: Callable[..., Dict[str, Any]]


def build_plan() -> List[PlannedRun]:
    """Pure: enumerate the exact 80 planned runs (4 cells x 20 reps). No I/O."""
    plan: List[PlannedRun] = []
    for setup, run_fn in CELLS:
        for strategy in STRATEGIES:
            for rep in range(1, REPETITIONS + 1):
                run_id = f"{setup}_{strategy}_rep{rep:02d}"
                plan.append(PlannedRun(setup=setup, strategy=strategy, rep=rep, run_id=run_id, run_fn=run_fn))
    return plan


def raw_path(run_id: str) -> Path:
    return MAIN_EXPERIMENT_DIR / f"{run_id}.json"


def eval_path(run_id: str) -> Path:
    return MAIN_EXPERIMENT_DIR / f"{run_id}_eval.json"


def check_collisions(plan: List[PlannedRun]) -> List[Tuple[str, List[str]]]:
    """Pure (read-only): return [(run_id, [existing filenames]), ...] for every
    planned run that already has a raw and/or eval output file on disk under
    MAIN_EXPERIMENT_DIR. Empty list means no collisions. Does not write
    anything, does not call any LLM."""
    collisions: List[Tuple[str, List[str]]] = []
    for p in plan:
        existing = []
        if raw_path(p.run_id).exists():
            existing.append(raw_path(p.run_id).name)
        if eval_path(p.run_id).exists():
            existing.append(eval_path(p.run_id).name)
        if existing:
            collisions.append((p.run_id, existing))
    return collisions


def validate_plan(plan: List[PlannedRun]) -> None:
    """Structural assertions: exactly 80 runs, exactly 20 per cell, all
    run_ids unique. Raises AssertionError on any violation. No I/O."""
    assert len(plan) == 80, f"Expected exactly 80 planned runs, got {len(plan)}"
    run_ids = [p.run_id for p in plan]
    assert len(run_ids) == len(set(run_ids)), "Duplicate run_id detected in plan"
    for setup, _ in CELLS:
        for strategy in STRATEGIES:
            n = sum(1 for p in plan if p.setup == setup and p.strategy == strategy)
            assert n == 20, f"Cell {setup}/{strategy} has {n} planned runs, expected exactly 20"


class RunFailure(RuntimeError):
    """Raised by run_one() for a genuine technical failure (in-run error,
    missing/corrupt output file) -- distinct from a Python exception escaping
    run_fn() itself, which already propagates and halts the batch on its own.
    main() catches this, reports it, and stops without proceeding further."""


def run_one(p: PlannedRun) -> Dict[str, Any]:
    """Execute exactly one planned run and persist its raw + eval output.
    Refuses (raises) if either output file already exists -- defense in depth
    on top of the upfront check_collisions() pass in main(). After the run,
    checks for an in-run technical error (`result["error"]`, set by
    run_negotiation()'s own try/except around the negotiation loop -- this
    does NOT propagate as a Python exception on its own, so it must be
    checked explicitly) and verifies both output files were actually written
    and are valid, non-empty JSON before returning -- a genuine problem here
    raises RunFailure so main() stops the batch rather than silently
    continuing to the next run."""
    if raw_path(p.run_id).exists() or eval_path(p.run_id).exists():
        existing = [
            name for name in (raw_path(p.run_id).name, eval_path(p.run_id).name)
            if (MAIN_EXPERIMENT_DIR / name).exists()
        ]
        raise RuntimeError(
            f"Refusing to run {p.run_id}: output file(s) already exist ({existing}). "
            "Main-experiment run_ids must never be overwritten -- resolve the collision "
            "(rename/move the existing file, or pick a different run_id scheme) before retrying."
        )
    result = p.run_fn(strategy=p.strategy, run_id=p.run_id, results_dir=MAIN_EXPERIMENT_DIR)

    if result.get("error"):
        raise RunFailure(
            f"{p.run_id}: run_negotiation() reported an in-run technical error "
            f"(negotiation loop caught an exception internally):\n{result['error']}"
        )

    raw_path(p.run_id).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Integrity check: both files must exist, be non-empty, and parse as JSON
    # with the run_id we expect -- catches a truncated/corrupt write.
    for path, label in ((raw_path(p.run_id), "raw"), (eval_path(p.run_id), "eval")):
        if not path.exists() or path.stat().st_size == 0:
            raise RunFailure(f"{p.run_id}: {label} output file missing or empty after run: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunFailure(f"{p.run_id}: {label} output file is not valid JSON ({path}): {exc}") from exc
        actual_run_id = loaded.get("run_id") if label == "eval" else None
        if label == "eval":
            actual_run_id = loaded.get("run_result", {}).get("run_id")
            if actual_run_id != p.run_id:
                raise RunFailure(
                    f"{p.run_id}: eval file run_id mismatch (found {actual_run_id!r} in {path})"
                )

    if result.get("rounds_played", 0) > 20 or result.get("final_round") and result["final_round"] > 20:
        raise RunFailure(
            f"{p.run_id}: rounds_played={result.get('rounds_played')} / "
            f"final_round={result.get('final_round')} exceeds 20 -- max_rounds fix violated."
        )

    return result


def main() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", flush=True)
        sys.exit(1)

    plan = build_plan()
    validate_plan(plan)
    print(f"Plan validated: {len(plan)} runs total, 20/cell across {len(CELLS)*len(STRATEGIES)} cells, all run_ids unique.")

    MAIN_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    collisions = check_collisions(plan)
    if collisions:
        print(f"\nCOLLISION DETECTED for {len(collisions)} of {len(plan)} planned run_id(s) -- ABORTING. "
              "No run has been executed.", flush=True)
        for run_id, files in collisions:
            print(f"  {run_id}: existing file(s) {files}", flush=True)
        print(
            "\nThe main experiment was NOT started. Resolve every collision above (move/rename the "
            "existing file(s), or confirm they belong to a prior, intentional partial main-experiment "
            "run) before re-running this script.",
            flush=True,
        )
        sys.exit(1)

    print(f"No collisions. Starting main experiment: {len(plan)} runs -> {MAIN_EXPERIMENT_DIR}", flush=True)

    summaries: List[Dict[str, Any]] = []
    for i, p in enumerate(plan, start=1):
        print(f"\n{'#' * 78}\n# [{i}/{len(plan)}] {p.run_id}\n{'#' * 78}", flush=True)
        t0 = time.time()
        try:
            result = run_one(p)
        except RunFailure as exc:
            print(f"\n{'!' * 78}\nTECHNICAL FAILURE on run {i}/{len(plan)} ({p.run_id}) -- STOPPING BATCH.\n{'!' * 78}", flush=True)
            print(str(exc), flush=True)
            print(
                f"\n{i - 1} of {len(plan)} runs completed successfully before this failure "
                f"(summary of those: {MAIN_EXPERIMENT_DIR / 'summary.json'}). "
                "No further runs were started. No experiment logic, prompts, or parameters were changed. "
                "No data was modified or repaired. Report this failure before deciding how to proceed.",
                flush=True,
            )
            sys.exit(1)
        duration = time.time() - t0
        summary = {
            "run_id": p.run_id,
            "setup": p.setup,
            "strategy": p.strategy,
            "rep": p.rep,
            "deal_reached": result["deal_reached"],
            "final_round": result["final_round"],
            "rounds_played": result["rounds_played"],
            "final_buyer_utility": result["final_buyer_utility"],
            "final_seller_utility": result["final_seller_utility"],
            "error": result["error"],
            "duration_sec": round(duration, 1),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2, default=str), flush=True)

        summary_path = MAIN_EXPERIMENT_DIR / "summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    print(f"\n\n{'=' * 78}\nMAIN EXPERIMENT COMPLETE ({len(summaries)} runs)\n{'=' * 78}", flush=True)
    print(f"Saved: {MAIN_EXPERIMENT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
