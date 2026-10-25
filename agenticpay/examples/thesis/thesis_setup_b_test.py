"""Setup B technical integration / plausibility test.

Setup B = Setup A (ThesisBuyerAgent vs. ThesisRuleBasedSeller) plus a
`ThesisGuardian` reviewing every buyer draft before it is sent to the seller,
per the exact Guardian role specification in `thesis_guardian.py`
(`GUARDIAN_SYSTEM_PROMPT`). Same fixed scenario, model and parameters
as Setup A (see `thesis_scenario.py`) -- only the buyer-side review step is
added, so the two runs are directly comparable.

Per round:
    1. Buyer drafts a proposal (draft 1) -- NOT yet sent to the seller.
    2. Rules 1-4 are computed deterministically (`thesis_guardian.run_deterministic_checks`,
       via `thesis_utility.py` only) against the correct candidate contract
       (the seller's last offer if the draft intends to accept it, otherwise
       the buyer's own draft), using the real env's own `_extract_contract` /
       `_contracts_compatible` (read-only) to detect "intends to accept".
    3. The Guardian LLM reviews the draft (compliance Rule 5 + strategic
       quality) and returns structured feedback.
    4. If `revision_required`, the SAME buyer gets exactly ONE revision
       attempt this round (draft 2, given the Guardian's feedback via
       `current_state["guardian_feedback"]`); no second Guardian call, no
       third attempt. Draft 2 (or draft 1, if no revision was required) is
       what actually gets sent to the seller.

This is a DIAGNOSTIC / technical-integration run, not calibration and not a
statistical result. No AgenticPay core file is modified.

Usage: same as `thesis_setup_a_test.py` (OPENAI_API_KEY via local `.env`).

Output: `agenticpay/results/thesis/{strategy}_test_setup_b.json` (does not
overwrite Setup A's `{strategy}_test.json`).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agenticpay.agents.Thesis_buyer_agent import ThesisBuyerAgent
from agenticpay.agents.thesis_guardian import (
    ThesisGuardian,
    extract_last_seller_contract,
    format_feedback_for_buyer,
    run_deterministic_checks,
)
from agenticpay.agents.thesis_rule_based_seller import ThesisRuleBasedSeller
from agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation import (
    Task1BasicPriceNegotiation,
)
from agenticpay.examples.thesis.thesis_scenario import (
    BUYER_MAX_PRICE,
    MAX_ROUNDS,
    MODEL_NAME,
    PRODUCT_INFO,
    REPO_ROOT,
    RESULTS_DIR,
    USER_REQUIREMENT,
    build_contract_config,
)
from agenticpay.models.openai_llm import OpenAILLM
from agenticpay.utils.thesis_evaluator import TokenTrackingLLM, evaluate_run, save_run_evaluation
from agenticpay.utils.thesis_utility import (
    BUYER_RESERVATION_UTILITY,
    SELLER_RESERVATION_UTILITY,
    agenticpay_contract_to_thesis,
    calculate_buyer_utility,
    calculate_seller_utility,
)


def run_negotiation(
    strategy: str, run_id: Optional[str] = None, results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run one full Setup B (Guardian-supervised) negotiation.

    As a side effect, also evaluates the completed negotiation via the
    Guardian-independent `thesis_evaluator.evaluate_run()` and saves the
    result (same RunResult schema as Setup A, plus Guardian-attributed
    token/cost breakdown) to `{results_dir}/{run_id}_eval.json`. `results_dir`
    defaults to the shared `RESULTS_DIR` (`agenticpay/results/thesis/`) for
    backward compatibility; callers running batches into their own
    subdirectory (e.g. calibration, variance studies) should pass their own
    directory here so the eval artifact is written alongside the raw run data
    instead of always landing in the shared top-level directory regardless of
    run_id (this fixes an earlier bug where eval files of different runs overwrote each other).
    """
    results_dir = results_dir or RESULTS_DIR
    contract_config = build_contract_config()
    buyer_preferences_text = contract_config["buyer_preferences"]["weight_descriptions"]

    # TokenTrackingLLM wraps OpenAILLM with identical call behavior (see
    # thesis_evaluator.TokenTrackingLLM) and additionally captures response.usage per call, kept
    # as SEPARATE instances for buyer vs. Guardian so their token/cost can be
    # attributed individually (chapter 5.3), not just as one combined number.
    buyer_openai_model = OpenAILLM(model=MODEL_NAME)
    guardian_openai_model = OpenAILLM(model=MODEL_NAME)
    buyer_model = TokenTrackingLLM(buyer_openai_model)
    guardian_model = TokenTrackingLLM(guardian_openai_model)
    buyer = ThesisBuyerAgent(model=buyer_model, buyer_max_price=BUYER_MAX_PRICE)
    guardian = ThesisGuardian(model=guardian_model)
    seller = ThesisRuleBasedSeller(strategy=strategy, max_rounds=MAX_ROUNDS)

    env = Task1BasicPriceNegotiation(
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=MAX_ROUNDS,
        buyer_max_price=BUYER_MAX_PRICE,
        environment_info={"contract_config": contract_config},
        price_tolerance=0.0,
    )

    observation, info = env.reset(user_requirement=USER_REQUIREMENT, product_info=PRODUCT_INFO)

    rounds_log: List[Dict[str, Any]] = []
    error: Optional[str] = None
    done = False
    negotiation_start = time.time()

    try:
        for _ in range(MAX_ROUNDS + 1):  # +1 kept as an inert outer safety net; the guard below is authoritative
            if done:
                break
            human_round = observation.get("current_round", 0) + 1  # env's round is 0-indexed pre-step
            if human_round > MAX_ROUNDS:
                # Off-by-one fix: env's own truncation check (`current_round >= max_rounds`)
                # only fires *after* processing a round, i.e. one round too late relative to
                # this human-numbered count. Without this guard, a real (LLM-costing) round
                # MAX_ROUNDS+1 -- including a Guardian call -- would be generated before
                # env.step() reports truncated=True.
                break
            history = observation["conversation_history"]
            seller_last_contract = extract_last_seller_contract(history)

            # --- Draft 1 ---------------------------------------------------
            draft1 = buyer.respond(conversation_history=history, current_state=observation)
            draft1_thesis_contract = dict(buyer.last_thesis_contract) if buyer.last_thesis_contract else None
            draft1_mental_model = buyer.last_mental_model
            draft1_raw_contract = env._extract_contract(draft1) if draft1 else None
            draft1_intends_to_accept = bool(
                seller_last_contract and draft1_raw_contract and env._contracts_compatible(draft1_raw_contract, seller_last_contract)
            )
            draft1_candidate_contract = (
                agenticpay_contract_to_thesis(seller_last_contract) if draft1_intends_to_accept and seller_last_contract else draft1_thesis_contract
            )
            draft1_deterministic_violations = run_deterministic_checks(draft1_candidate_contract, BUYER_MAX_PRICE)

            feedback = guardian.review(
                draft_message=draft1,
                candidate_thesis_contract=draft1_candidate_contract,
                intends_to_accept=draft1_intends_to_accept,
                deterministic_violations=draft1_deterministic_violations,
                buyer_preferences_text=buyer_preferences_text,
                conversation_history=history,
                current_round=human_round,
                max_rounds=MAX_ROUNDS,
                mental_model=draft1_mental_model,
            )

            # --- Draft 2 (at most one revision, per the Guardian's own rule) ---
            draft2 = None
            draft2_thesis_contract = None
            draft2_deterministic_violations = None
            if feedback["revision_required"]:
                feedback_text = format_feedback_for_buyer(feedback)
                observation_with_feedback = dict(observation)
                observation_with_feedback["guardian_feedback"] = feedback_text
                draft2 = buyer.respond(conversation_history=history, current_state=observation_with_feedback)
                draft2_thesis_contract = dict(buyer.last_thesis_contract) if buyer.last_thesis_contract else None
                # Log-only recheck: NOT a second Guardian call, NOT a third attempt.
                draft2_raw_contract = env._extract_contract(draft2) if draft2 else None
                draft2_intends_to_accept = bool(
                    seller_last_contract and draft2_raw_contract and env._contracts_compatible(draft2_raw_contract, seller_last_contract)
                )
                draft2_candidate_contract = (
                    agenticpay_contract_to_thesis(seller_last_contract) if draft2_intends_to_accept and seller_last_contract else draft2_thesis_contract
                )
                draft2_deterministic_violations = run_deterministic_checks(draft2_candidate_contract, BUYER_MAX_PRICE)
                final_buyer_message = draft2
            else:
                final_buyer_message = draft1

            # --- Seller responds to whatever the buyer actually sends ------
            updated_history = history.copy()
            if final_buyer_message:
                updated_history.append({"role": "buyer", "content": final_buyer_message, "round": observation.get("current_round", 0)})
            seller_action = seller.respond(conversation_history=updated_history, current_state=observation)

            # --- Diagnostics on what was actually sent (reuse env's own checks) ---
            final_raw_contract = env._extract_contract(final_buyer_message) if final_buyer_message else None
            final_contract_valid = bool(final_raw_contract) and env._validate_contract(final_raw_contract)
            seller_raw_contract = env._extract_contract(seller_action) if seller_action else None
            seller_contract_valid = bool(seller_raw_contract) and env._validate_contract(seller_raw_contract)

            seller_target_utility = seller.calculate_target_utility(human_round)
            final_thesis_contract = None
            buyer_utility_of_final_offer = None
            seller_utility_of_final_offer = None
            seller_decision = None
            if final_raw_contract:
                try:
                    final_thesis_contract = agenticpay_contract_to_thesis(final_raw_contract)
                    buyer_utility_of_final_offer = calculate_buyer_utility(final_thesis_contract)
                    seller_utility_of_final_offer = calculate_seller_utility(final_thesis_contract)
                    seller_decision = "accept" if seller.should_accept(final_thesis_contract, human_round) else "counter"
                except ValueError:
                    pass

            seller_thesis_contract = None
            buyer_utility_of_seller_offer = None
            seller_utility_of_seller_offer = None
            if seller_raw_contract:
                try:
                    seller_thesis_contract = agenticpay_contract_to_thesis(seller_raw_contract)
                    buyer_utility_of_seller_offer = calculate_buyer_utility(seller_thesis_contract)
                    seller_utility_of_seller_offer = calculate_seller_utility(seller_thesis_contract)
                except ValueError:
                    pass

            rounds_log.append(
                {
                    "round": human_round,
                    "draft1_message": draft1,
                    "draft1_contract": draft1_thesis_contract,
                    "draft1_intends_to_accept": draft1_intends_to_accept,
                    "draft1_deterministic_violations": draft1_deterministic_violations,
                    "guardian_feedback": {
                        "compliance_violations": feedback["compliance_violations"],
                        "strategic_concerns": feedback["strategic_concerns"],
                        "revision_required": feedback["revision_required"],
                        "llm_stated_revision_required": feedback["llm_stated_revision_required"],
                        "parse_ok": feedback["parse_ok"],
                    },
                    "revision_triggered": feedback["revision_required"],
                    "draft2_message": draft2,
                    "draft2_contract": draft2_thesis_contract,
                    "draft2_deterministic_violations": draft2_deterministic_violations,
                    "final_buyer_message": final_buyer_message,
                    "final_contract_raw": final_raw_contract,
                    "final_contract_valid": final_contract_valid,
                    "buyer_utility_of_final_offer": buyer_utility_of_final_offer,
                    "seller_utility_of_final_offer": seller_utility_of_final_offer,
                    "seller_target_utility": seller_target_utility,
                    "seller_decision": seller_decision,
                    "seller_message": seller_action,
                    "seller_contract_raw": seller_raw_contract,
                    "seller_contract_valid": seller_contract_valid,
                    "buyer_utility_of_seller_offer": buyer_utility_of_seller_offer,
                    "seller_utility_of_seller_offer": seller_utility_of_seller_offer,
                }
            )

            observation, reward, terminated, truncated, info = env.step(
                buyer_action=final_buyer_message, seller_action=seller_action
            )
            done = terminated or truncated
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    negotiation_runtime_sec = time.time() - negotiation_start

    agreed_contract_raw = info.get("agreed_contract") if not error else None
    deal_reached = bool(agreed_contract_raw)
    agreed_contract_thesis = None
    final_buyer_utility = None
    final_seller_utility = None
    if agreed_contract_raw:
        try:
            agreed_contract_thesis = agenticpay_contract_to_thesis(agreed_contract_raw)
            final_buyer_utility = calculate_buyer_utility(agreed_contract_thesis)
            final_seller_utility = calculate_seller_utility(agreed_contract_thesis)
        except ValueError:
            pass

    result: Dict[str, Any] = {
        "strategy": strategy,
        "model": MODEL_NAME,
        "max_rounds": MAX_ROUNDS,
        "rounds_played": len(rounds_log),
        "deal_reached": deal_reached,
        "final_round": rounds_log[-1]["round"] if rounds_log else None,
        "agreed_contract_raw": agreed_contract_raw,
        "agreed_contract_thesis": agreed_contract_thesis,
        "final_buyer_utility": final_buyer_utility,
        "final_seller_utility": final_seller_utility,
        "error": error,
        "rounds": rounds_log,
    }
    result["plausibility"] = compute_plausibility_checks(result)

    # --- Evaluator integration (post-hoc, Guardian-independent) ---------------
    # Same RunResult schema as Setup A; guardian_usage is populated here (Setup
    # A always passes None for it), giving a separate Guardian-attributed
    # token/cost breakdown alongside the buyer's and the combined total.
    resolved_run_id = run_id or f"setup_b_{strategy}"
    run_result, violations, semantic_findings = evaluate_run(
        conversation_history=observation["conversation_history"],
        agreed_contract_raw=agreed_contract_raw,
        deal_reached=deal_reached,
        final_round=result["final_round"],
        max_rounds=MAX_ROUNDS,
        buyer_max_price=BUYER_MAX_PRICE,
        run_id=resolved_run_id,
        setup="setup_b",
        strategy=strategy,
        model=MODEL_NAME,
        buyer_usage=buyer_model.usage_log,
        guardian_usage=guardian_model.usage_log,
        runtime_sec=negotiation_runtime_sec,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    save_run_evaluation(run_result, violations, semantic_findings, results_dir / f"{resolved_run_id}_eval.json")

    return result


def compute_plausibility_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    """Setup-B-specific plausibility summary, computed only from already-collected
    diagnostics (env's own contract validation, thesis_utility.py outputs,
    Guardian's own parsed output) -- no new evaluator/metric logic.
    """
    rounds = result["rounds"]

    contract_flow_ok = all(
        (r["final_contract_raw"] is None or r["final_contract_valid"])
        and (r["seller_contract_raw"] is None or r["seller_contract_valid"])
        for r in rounds
    )

    all_utilities = []
    for r in rounds:
        for key in ("buyer_utility_of_final_offer", "seller_utility_of_final_offer", "seller_target_utility", "buyer_utility_of_seller_offer", "seller_utility_of_seller_offer"):
            if r.get(key) is not None:
                all_utilities.append(r[key])
    if result.get("final_buyer_utility") is not None:
        all_utilities.append(result["final_buyer_utility"])
    if result.get("final_seller_utility") is not None:
        all_utilities.append(result["final_seller_utility"])
    utilities_in_range = all(0.0 <= u <= 1.0 for u in all_utilities)

    revisions_triggered = sum(1 for r in rounds if r["revision_triggered"])
    rounds_with_deterministic_violation_pre_revision = [
        r["round"] for r in rounds if r["draft1_deterministic_violations"]
    ]
    rounds_where_revision_fixed_deterministic_violation = [
        r["round"]
        for r in rounds
        if r["draft1_deterministic_violations"] and r["revision_triggered"] and not r["draft2_deterministic_violations"]
    ]
    rounds_with_deterministic_violation_in_final_offer = [
        r["round"]
        for r in rounds
        if r["final_contract_raw"]
        and r["buyer_utility_of_final_offer"] is not None
        and r["buyer_utility_of_final_offer"] < BUYER_RESERVATION_UTILITY - 1e-9
    ]
    guardian_parse_failures = [r["round"] for r in rounds if not r["guardian_feedback"]["parse_ok"]]

    no_reservation_violation_on_deal = True
    if result["deal_reached"]:
        no_reservation_violation_on_deal = (
            result["final_buyer_utility"] is not None
            and result["final_buyer_utility"] >= BUYER_RESERVATION_UTILITY - 1e-9
            and result["final_seller_utility"] is not None
            and result["final_seller_utility"] >= SELLER_RESERVATION_UTILITY - 1e-9
        )

    return {
        "contract_flow_ok": contract_flow_ok,
        "utilities_in_range": utilities_in_range,
        "revisions_triggered": revisions_triggered,
        "rounds_with_deterministic_violation_pre_revision": rounds_with_deterministic_violation_pre_revision,
        "rounds_where_revision_fixed_deterministic_violation": rounds_where_revision_fixed_deterministic_violation,
        "rounds_with_deterministic_violation_in_final_offer": rounds_with_deterministic_violation_in_final_offer,
        "guardian_parse_failures": guardian_parse_failures,
        "no_reservation_violation_on_deal": no_reservation_violation_on_deal,
        "ran_without_error": result["error"] is None,
    }


def print_transcript(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\nStrategy: {result['strategy']}  |  model: {result['model']}  |  max_rounds: {result['max_rounds']}\n{'=' * 78}")
    if result["error"]:
        print("ERROR DURING RUN:\n" + result["error"])
    for r in result["rounds"]:
        print(f"\n--- Round {r['round']} ---")
        print(f"  DRAFT 1: {r['draft1_message']!r}")
        print(f"  draft1_contract={r['draft1_contract']} intends_to_accept={r['draft1_intends_to_accept']} "
              f"deterministic_violations={r['draft1_deterministic_violations']}")
        gf = r["guardian_feedback"]
        print(f"  GUARDIAN: revision_required={gf['revision_required']} (llm_stated={gf['llm_stated_revision_required']}, parse_ok={gf['parse_ok']})")
        print(f"    compliance_violations={gf['compliance_violations']}")
        print(f"    strategic_concerns={gf['strategic_concerns']}")
        if r["revision_triggered"]:
            print(f"  DRAFT 2 (revised): {r['draft2_message']!r}")
            print(f"  draft2_contract={r['draft2_contract']} deterministic_violations_after_revision={r['draft2_deterministic_violations']}")
        print(f"  FINAL buyer message sent to seller: {r['final_buyer_message']!r}")
        print(f"  U_buyer(final)={_fmt(r['buyer_utility_of_final_offer'])}  U_seller(final)={_fmt(r['seller_utility_of_final_offer'])}")
        print(f"  Seller target_utility={_fmt(r['seller_target_utility'])}  decision={r['seller_decision']}")
        print(f"  SELLER msg: {r['seller_message']!r}")
        print(f"  SELLER contract (raw): {r['seller_contract_raw']}  valid={r['seller_contract_valid']}")

    print(f"\n--- Outcome ---")
    print(f"  Deal reached: {result['deal_reached']}   Final round: {result['final_round']}   Rounds played: {result['rounds_played']}")
    print(f"  Agreed contract (thesis): {result['agreed_contract_thesis']}")
    print(f"  Final U_buyer={_fmt(result['final_buyer_utility'])}  Final U_seller={_fmt(result['final_seller_utility'])}")
    print(f"\n  Plausibility checks: {json.dumps(result['plausibility'], indent=2)}")


def print_comparison(conceder: Dict[str, Any], boulware: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\nConceder vs. Boulware comparison (Setup B, diagnostic only -- n=1 per strategy)\n{'=' * 78}")
    for label, res in (("conceder", conceder), ("boulware", boulware)):
        print(
            f"  {label:10s} deal={res['deal_reached']!s:5} final_round={res['final_round']} "
            f"U_buyer={_fmt(res['final_buyer_utility'])} U_seller={_fmt(res['final_seller_utility'])} "
            f"revisions_triggered={res['plausibility']['revisions_triggered']} "
            f"contract={res['agreed_contract_thesis']} error={'yes' if res['error'] else 'no'}"
        )


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def main() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Either export it, or put it in a local .env file:")
        print('  echo \'OPENAI_API_KEY=sk-...\' >> .env   # repo root, gitignored')
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for strategy in ("conceder", "boulware"):
        print(f"\n{'#' * 78}\n# Setup B run (Guardian-supervised) -- strategy={strategy}\n{'#' * 78}")
        result = run_negotiation(strategy)
        results[strategy] = result
        out_path = RESULTS_DIR / f"{strategy}_test_setup_b.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print_transcript(result)
        print(f"\nSaved: {out_path}")

    print_comparison(results["conceder"], results["boulware"])


if __name__ == "__main__":
    main()
