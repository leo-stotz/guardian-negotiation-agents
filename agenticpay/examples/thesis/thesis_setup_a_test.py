"""Setup A technical integration / plausibility test.

Setup A = ThesisBuyerAgent (LLM) vs. ThesisRuleBasedSeller (rule-based Python).

Runs one full negotiation for strategy="conceder" and one for strategy="boulware"
over the real, unmodified AgenticPay contract-mode environment
(`Task1_basic_price_negotiation`), for the same fixed scenario (1,000 industrial
sensors; price/delivery_days/payment_days; parameters as listed in the README).

This is a DIAGNOSTIC / technical-integration run, not calibration and not a
statistical result -- a single run per strategy has no inferential power. See
the thesis for the accompanying discussion.

Utility computation uses ONLY `agenticpay.utils.thesis_utility` (no parallel
utility logic here). Contract validity checks reuse the real, unmodified env's
own `_extract_contract` / `_validate_contract` (called read-only for
diagnostics). No AgenticPay core file is modified.

Usage:
    Put OPENAI_API_KEY into a local `.env` file at the repo root (gitignored,
    never hardcode it here or in config.py):

        OPENAI_API_KEY=sk-...

    Then simply:

        python agenticpay/examples/thesis/thesis_setup_a_test.py

    `.env` is loaded via python-dotenv (`load_dotenv()` below); a real
    OPENAI_API_KEY already exported in the shell environment still works too
    and takes precedence (`load_dotenv()` does not override existing env vars).

Output: one JSON protocol per strategy under `agenticpay/results/thesis/`
(`conceder_test.json`, `boulware_test.json`; that directory is gitignored) plus
a human-readable transcript on stdout.
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
    AGENTICPAY_DELIVERY_TERM,
    AGENTICPAY_PAYMENT_TERM,
    BUYER_RESERVATION_UTILITY,
    SELLER_RESERVATION_UTILITY,
    agenticpay_contract_to_thesis,
    calculate_buyer_utility,
    calculate_seller_utility,
)


def run_negotiation(
    strategy: str, run_id: Optional[str] = None, results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run one full Setup A negotiation and return a structured protocol dict.

    As a side effect, also evaluates the completed negotiation via the
    Guardian-independent `thesis_evaluator.evaluate_run()` and saves the
    result (same RunResult schema as Setup B) to
    `{results_dir}/{run_id}_eval.json`. `results_dir` defaults to the shared
    `RESULTS_DIR` (`agenticpay/results/thesis/`) for backward compatibility;
    callers running batches into their own subdirectory (e.g. calibration,
    variance studies) should pass their own directory here so the eval
    artifact is written alongside the raw run data instead of always landing
    in the shared top-level directory regardless of run_id.
    """
    results_dir = results_dir or RESULTS_DIR
    contract_config = build_contract_config()

    # TokenTrackingLLM wraps OpenAILLM with identical call behavior (see
    # thesis_evaluator.TokenTrackingLLM) and additionally captures response.usage per call, so
    # token/cost data is available to the evaluator below.
    openai_model = OpenAILLM(model=MODEL_NAME)
    model = TokenTrackingLLM(openai_model)
    buyer = ThesisBuyerAgent(model=model, buyer_max_price=BUYER_MAX_PRICE)
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
                # MAX_ROUNDS+1 would be generated before env.step() reports truncated=True.
                break

            buyer_action = buyer.respond(
                conversation_history=observation["conversation_history"],
                current_state=observation,
            )
            buyer_compliance = dict(buyer.last_validation) if buyer.last_validation else None
            buyer_thesis_contract = dict(buyer.last_thesis_contract) if buyer.last_thesis_contract else None

            updated_history = observation["conversation_history"].copy()
            if buyer_action:
                updated_history.append(
                    {"role": "buyer", "content": buyer_action, "round": observation.get("current_round", 0)}
                )

            seller_action = seller.respond(conversation_history=updated_history, current_state=observation)

            # Diagnostics reuse only already-existing checks (no new validation logic):
            buyer_raw_contract = env._extract_contract(buyer_action) if buyer_action else None
            buyer_contract_valid = bool(buyer_raw_contract) and env._validate_contract(buyer_raw_contract)
            seller_raw_contract = env._extract_contract(seller_action) if seller_action else None
            seller_contract_valid = bool(seller_raw_contract) and env._validate_contract(seller_raw_contract)

            seller_target_utility = seller.calculate_target_utility(human_round)
            seller_decision = None
            seller_utility_of_buyer_offer = None
            buyer_utility_of_buyer_offer = None
            if buyer_thesis_contract is not None:
                seller_utility_of_buyer_offer = calculate_seller_utility(buyer_thesis_contract)
                buyer_utility_of_buyer_offer = calculate_buyer_utility(buyer_thesis_contract)
                seller_decision = "accept" if seller.should_accept(buyer_thesis_contract, human_round) else "counter"

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
                    "buyer_message": buyer_action,
                    "buyer_contract_raw": buyer_raw_contract,
                    "buyer_contract_valid": buyer_contract_valid,
                    "buyer_utility_of_own_offer": buyer_utility_of_buyer_offer,
                    "seller_utility_of_buyer_offer": seller_utility_of_buyer_offer,
                    "seller_target_utility": seller_target_utility,
                    "seller_decision": seller_decision,
                    "seller_message": seller_action,
                    "seller_contract_raw": seller_raw_contract,
                    "seller_contract_valid": seller_contract_valid,
                    "buyer_utility_of_seller_offer": buyer_utility_of_seller_offer,
                    "seller_utility_of_seller_offer": seller_utility_of_seller_offer,
                    "buyer_compliance": buyer_compliance,
                }
            )

            observation, reward, terminated, truncated, info = env.step(
                buyer_action=buyer_action, seller_action=seller_action
            )
            done = terminated or truncated
    except Exception as exc:  # capture and report -- do not silently swallow, do not crash the script
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
    # Setup A has no Guardian, so guardian_usage=None -- every guardian_* field
    # on RunResult stays None, but the schema is identical to Setup B's.
    resolved_run_id = run_id or f"setup_a_{strategy}"
    run_result, violations, semantic_findings = evaluate_run(
        conversation_history=observation["conversation_history"],
        agreed_contract_raw=agreed_contract_raw,
        deal_reached=deal_reached,
        final_round=result["final_round"],
        max_rounds=MAX_ROUNDS,
        buyer_max_price=BUYER_MAX_PRICE,
        run_id=resolved_run_id,
        setup="setup_a",
        strategy=strategy,
        model=MODEL_NAME,
        buyer_usage=model.usage_log,
        guardian_usage=None,
        runtime_sec=negotiation_runtime_sec,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    save_run_evaluation(run_result, violations, semantic_findings, results_dir / f"{resolved_run_id}_eval.json")

    return result


def compute_plausibility_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    """Section-7-style plausibility checks, computed only from already-collected
    diagnostics (env's own contract validation, thesis_utility.py outputs) --
    no new evaluator/metric logic.
    """
    rounds = result["rounds"]

    all_utilities = []
    for r in rounds:
        for key in (
            "buyer_utility_of_own_offer",
            "seller_utility_of_buyer_offer",
            "seller_target_utility",
            "buyer_utility_of_seller_offer",
            "seller_utility_of_seller_offer",
        ):
            if r.get(key) is not None:
                all_utilities.append(r[key])
    if result.get("final_buyer_utility") is not None:
        all_utilities.append(result["final_buyer_utility"])
    if result.get("final_seller_utility") is not None:
        all_utilities.append(result["final_seller_utility"])

    contract_flow_ok = all(
        (r["buyer_contract_raw"] is None or r["buyer_contract_valid"])
        and (r["seller_contract_raw"] is None or r["seller_contract_valid"])
        for r in rounds
    )
    utilities_in_range = all(0.0 <= u <= 1.0 for u in all_utilities)

    seller_prices = [r["seller_contract_raw"]["price"] for r in rounds if r.get("seller_contract_raw")]
    seller_deliveries = [
        r["seller_contract_raw"]["continuous_terms"].get(AGENTICPAY_DELIVERY_TERM)
        for r in rounds
        if r.get("seller_contract_raw")
    ]
    seller_payments = [
        r["seller_contract_raw"]["continuous_terms"].get(AGENTICPAY_PAYMENT_TERM)
        for r in rounds
        if r.get("seller_contract_raw")
    ]
    seller_moved_non_price_issue = len(set(seller_deliveries)) > 1 or len(set(seller_payments)) > 1

    buyer_prices = [r["buyer_contract_raw"]["price"] for r in rounds if r.get("buyer_contract_raw")]
    buyer_always_max_price = bool(buyer_prices) and all(p >= BUYER_MAX_PRICE - 1e-6 for p in buyer_prices)
    buyer_deliveries = [
        r["buyer_contract_raw"]["continuous_terms"].get(AGENTICPAY_DELIVERY_TERM)
        for r in rounds
        if r.get("buyer_contract_raw")
    ]
    buyer_payments = [
        r["buyer_contract_raw"]["continuous_terms"].get(AGENTICPAY_PAYMENT_TERM)
        for r in rounds
        if r.get("buyer_contract_raw")
    ]
    buyer_used_multiple_issues = len(set(buyer_deliveries)) > 1 or len(set(buyer_payments)) > 1 or len(set(buyer_prices)) > 1

    no_reservation_violation_on_deal = True
    if result["deal_reached"]:
        no_reservation_violation_on_deal = (
            result["final_buyer_utility"] is not None
            and result["final_buyer_utility"] >= BUYER_RESERVATION_UTILITY - 1e-9
            and result["final_seller_utility"] is not None
            and result["final_seller_utility"] >= SELLER_RESERVATION_UTILITY - 1e-9
        )

    buyer_compliance_violations = [
        {"round": r["round"], "compliance": r["buyer_compliance"]}
        for r in rounds
        if r["buyer_compliance"] is not None and not all(r["buyer_compliance"].values())
    ]

    return {
        "contract_flow_ok": contract_flow_ok,
        "utilities_in_range": utilities_in_range,
        "seller_moved_non_price_issue": seller_moved_non_price_issue,
        "buyer_always_max_price": buyer_always_max_price,
        "buyer_used_multiple_issues": buyer_used_multiple_issues,
        "no_reservation_violation_on_deal": no_reservation_violation_on_deal,
        "buyer_compliance_violations": buyer_compliance_violations,
        "ran_without_error": result["error"] is None,
    }


def print_transcript(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\nStrategy: {result['strategy']}  |  model: {result['model']}  |  max_rounds: {result['max_rounds']}\n{'=' * 78}")
    if result["error"]:
        print("ERROR DURING RUN:\n" + result["error"])
    for r in result["rounds"]:
        print(f"\n--- Round {r['round']} ---")
        print(f"  BUYER msg: {r['buyer_message']!r}")
        print(f"  BUYER contract (raw): {r['buyer_contract_raw']}  valid={r['buyer_contract_valid']}")
        print(f"  U_buyer(buyer offer)={_fmt(r['buyer_utility_of_own_offer'])}  U_seller(buyer offer)={_fmt(r['seller_utility_of_buyer_offer'])}")
        print(f"  Seller target_utility={_fmt(r['seller_target_utility'])}  decision={r['seller_decision']}")
        if r["buyer_compliance"] is not None:
            print(f"  Buyer compliance hook: {r['buyer_compliance']}")
        print(f"  SELLER msg: {r['seller_message']!r}")
        print(f"  SELLER contract (raw): {r['seller_contract_raw']}  valid={r['seller_contract_valid']}")
        print(f"  U_buyer(seller offer)={_fmt(r['buyer_utility_of_seller_offer'])}  U_seller(seller offer)={_fmt(r['seller_utility_of_seller_offer'])}")

    print(f"\n--- Outcome ---")
    print(f"  Deal reached: {result['deal_reached']}   Final round: {result['final_round']}   Rounds played: {result['rounds_played']}")
    print(f"  Agreed contract (raw): {result['agreed_contract_raw']}")
    print(f"  Agreed contract (thesis): {result['agreed_contract_thesis']}")
    print(f"  Final U_buyer={_fmt(result['final_buyer_utility'])}  Final U_seller={_fmt(result['final_seller_utility'])}")
    print(f"\n  Plausibility checks: {json.dumps(result['plausibility'], indent=2)}")


def print_comparison(conceder: Dict[str, Any], boulware: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\nConceder vs. Boulware comparison (diagnostic only -- n=1 per strategy)\n{'=' * 78}")
    for label, res in (("conceder", conceder), ("boulware", boulware)):
        print(
            f"  {label:10s} deal={res['deal_reached']!s:5} final_round={res['final_round']} "
            f"U_buyer={_fmt(res['final_buyer_utility'])} U_seller={_fmt(res['final_seller_utility'])} "
            f"contract={res['agreed_contract_thesis']} error={'yes' if res['error'] else 'no'}"
        )


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def main() -> None:
    # Loads OPENAI_API_KEY (and any other vars) from a local .env at the repo root,
    # if present. Never overrides a value already set in the shell environment.
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Either export it, or put it in a local .env file:")
        print('  echo \'OPENAI_API_KEY=sk-...\' >> .env   # repo root, gitignored')
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for strategy in ("conceder", "boulware"):
        print(f"\n{'#' * 78}\n# Setup A run -- strategy={strategy}\n{'#' * 78}")
        result = run_negotiation(strategy)
        results[strategy] = result
        out_path = RESULTS_DIR / f"{strategy}_test.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print_transcript(result)
        print(f"\nSaved: {out_path}")

    print_comparison(results["conceder"], results["boulware"])


if __name__ == "__main__":
    main()
