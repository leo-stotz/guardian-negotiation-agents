"""Regression tests for the max_rounds off-by-one fix in the thesis orchestration
scripts (`agenticpay/examples/thesis/thesis_setup_a_test.py` and
`thesis_setup_b_test.py`).

Standalone script, not pytest (this repo has no pytest suite --
`agenticpay/test/` holds standalone manual scripts). Run directly:

    python agenticpay/test/test_thesis_max_rounds_fix.py

No API key / network required: `ThesisBuyerAgent` and (for Setup B)
`ThesisGuardian` are replaced with deterministic stub objects via
`unittest.mock.patch`, so no real LLM call happens. Everything else --
`Task1BasicPriceNegotiation` (the real, unmodified AgenticPay core env) and
`ThesisRuleBasedSeller` (already deterministic, no LLM) -- runs for real. This
exercises exactly the orchestration-loop code path the fix touches, including
the real `_contracts_compatible()` / `_check_agreement()` env logic, without
needing to fake LLM prompt/response text formats.

Covers exactly the four cases requested:
    1. No-Deal ends after exactly MAX_ROUNDS (20) rounds -- no round 21.
    2. A deal in round MAX_ROUNDS (20) is still correctly detected.
    3. A deal before round MAX_ROUNDS (e.g. round 3) is unaffected.
    4. No round > MAX_ROUNDS is ever logged or executed, for both Setup A and
       Setup B (including Guardian calls in Setup B).

Writes its run artifacts under `agenticpay/results/thesis/_max_rounds_fix_regression_test/`
-- a dedicated, separate directory, never overlapping with any existing
pilot/calibration/verification run data.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "_max_rounds_fix_regression_test"

from agenticpay.examples.thesis import thesis_setup_a_test as setup_a
from agenticpay.examples.thesis import thesis_setup_b_test as setup_b


# --------------------------------------------------------------------------- helpers

def _extract_contract_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"<contract>\s*(.*?)\s*</contract>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _last_seller_contract(conversation_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in reversed(conversation_history):
        if msg.get("role") == "seller":
            c = _extract_contract_text(msg.get("content"))
            if c:
                return c
    return None


# A fixed offer that stays inside all configured bounds (price/delivery/payment)
# but uses off-grid delivery/payment values the rule-based seller's fixed grid
# search (DEFAULT_GRID_STEPS candidates, evenly spaced) never independently lands
# on -- verified empirically: an earlier choice using grid-endpoint values
# (delivery_days=40.0, payment_days=14.0, both literal grid points) caused an
# accidental real deal around round 8 in every case, since the seller's own
# deterministic trajectory passed through that exact point. continuous_terms
# must match buyer/seller *exactly* (see `_contracts_compatible()` in the
# environment), so off-grid fractional values reliably
# guarantee no accidental compatibility for the "no deal" test cases.
_NEVER_MATCHING_OFFER = {
    "price": 80000.0,
    "continuous_terms": {"delivery_days": 25.37, "payment_days": 33.71},
    "discrete_terms": {},
}


class StubBuyer:
    """Deterministic stand-in for ThesisBuyerAgent -- makes no LLM call.

    deal_from_round=None -> always sends `_NEVER_MATCHING_OFFER` (forces No-Deal
    all the way to the round cap).
    deal_from_round=N -> sends `_NEVER_MATCHING_OFFER` for round < N, then echoes
    the seller's exact last offer verbatim starting at round N (guarantees a
    compatible contract -- i.e. a deal -- triggers on round N, via the real,
    unmodified `_contracts_compatible()`/`_check_agreement()` env logic).
    """

    def __init__(self, model=None, buyer_max_price=None, name: str = "StubBuyer", deal_from_round: Optional[int] = None):
        self.buyer_max_price = buyer_max_price
        self.name = name
        self.deal_from_round = deal_from_round
        self.last_validation = {"range_ok": True, "max_price_ok": True, "reservation_ok": True}
        self.last_thesis_contract: Optional[Dict[str, float]] = None
        self.last_mental_model = "stub mental model"
        self._round_counter = 0

    def initialize(self, context: Dict[str, Any]) -> None:
        self._context = context

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        self._round_counter += 1
        round_now = self._round_counter
        seller_contract = _last_seller_contract(conversation_history)
        if self.deal_from_round is not None and round_now >= self.deal_from_round and seller_contract:
            contract = seller_contract
        else:
            contract = _NEVER_MATCHING_OFFER
        self.last_thesis_contract = {
            "price": float(contract["price"]),
            "delivery_days": float(contract["continuous_terms"]["delivery_days"]),
            "payment_days": float(contract["continuous_terms"]["payment_days"]),
        }
        contract_json = json.dumps(contract, sort_keys=True)
        return (
            f"<mental_model>[stub round {round_now}]</mental_model>"
            f"<action>COUNTER</action>"
            f"<message>Stub offer for round {round_now}.<contract>{contract_json}</contract></message>"
        )


class StubGuardian:
    """Deterministic stand-in for ThesisGuardian -- never requests a revision,
    so the draft2 path is never exercised (irrelevant to the round-cap fix)."""

    def __init__(self, model=None, name: str = "StubGuardian"):
        self.name = name

    def review(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "compliance_violations": [],
            "strategic_concerns": [],
            "revision_required": False,
            "llm_stated_revision_required": False,
            "parse_ok": True,
        }


class _DummyLLM:
    """Stand-in for OpenAILLM: constructs without an API key, .generate() must
    never actually be called by the stub buyer/guardian above."""

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("Dummy LLM .generate() was called -- stub buyer/guardian should never reach the model.")

    def __repr__(self) -> str:
        return "_DummyLLM()"


# --------------------------------------------------------------------------- test runners

def _run_setup_a(run_id: str, deal_from_round: Optional[int]) -> Dict[str, Any]:
    def buyer_factory(model=None, buyer_max_price=None):
        return StubBuyer(model=model, buyer_max_price=buyer_max_price, deal_from_round=deal_from_round)

    with patch.object(setup_a, "OpenAILLM", _DummyLLM), patch.object(setup_a, "ThesisBuyerAgent", buyer_factory):
        return setup_a.run_negotiation(strategy="conceder", run_id=run_id, results_dir=RESULTS_DIR)


def _run_setup_b(run_id: str, deal_from_round: Optional[int]) -> Dict[str, Any]:
    def buyer_factory(model=None, buyer_max_price=None):
        return StubBuyer(model=model, buyer_max_price=buyer_max_price, deal_from_round=deal_from_round)

    with patch.object(setup_b, "OpenAILLM", _DummyLLM), \
         patch.object(setup_b, "ThesisBuyerAgent", buyer_factory), \
         patch.object(setup_b, "ThesisGuardian", StubGuardian):
        return setup_b.run_negotiation(strategy="conceder", run_id=run_id, results_dir=RESULTS_DIR)


# --------------------------------------------------------------------------- assertions

_FAILURES: List[str] = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition:
        _FAILURES.append(description)


def assert_no_round_beyond_max(result: Dict[str, Any], max_rounds: int, label: str) -> None:
    rounds = [r["round"] for r in result["rounds"]]
    check(all(r <= max_rounds for r in rounds), f"{label}: no logged round exceeds {max_rounds} (rounds: {rounds[:3]}...{rounds[-3:] if len(rounds) > 3 else ''})")
    check(len(result["rounds"]) <= max_rounds, f"{label}: rounds_played ({len(result['rounds'])}) <= {max_rounds}")


def main() -> None:
    print(f"MAX_ROUNDS (Setup A) = {setup_a.MAX_ROUNDS}, MAX_ROUNDS (Setup B) = {setup_b.MAX_ROUNDS}")
    max_rounds_a = setup_a.MAX_ROUNDS
    max_rounds_b = setup_b.MAX_ROUNDS

    print("\n=== Setup A ===")

    print("\n-- Case A1: No-Deal must end after exactly MAX_ROUNDS rounds, no round 21 --")
    r = _run_setup_a("regtest_a_no_deal", deal_from_round=None)
    check(r["deal_reached"] is False, "A1: deal_reached == False")
    check(r["rounds_played"] == max_rounds_a, f"A1: rounds_played == {max_rounds_a} (got {r['rounds_played']})")
    check(r["rounds"][-1]["round"] == max_rounds_a, f"A1: last logged round == {max_rounds_a} (got {r['rounds'][-1]['round']})")
    check(not any(rr["round"] > max_rounds_a for rr in r["rounds"]), "A1: no round > MAX_ROUNDS was logged")
    assert_no_round_beyond_max(r, max_rounds_a, "A1")

    print("\n-- Case A2: Deal in round MAX_ROUNDS (20) must still be detected --")
    r = _run_setup_a("regtest_a_deal_at_20", deal_from_round=max_rounds_a)
    check(r["deal_reached"] is True, "A2: deal_reached == True")
    check(r["final_round"] == max_rounds_a, f"A2: final_round == {max_rounds_a} (got {r['final_round']})")
    check(r["rounds_played"] == max_rounds_a, f"A2: rounds_played == {max_rounds_a} (got {r['rounds_played']})")
    assert_no_round_beyond_max(r, max_rounds_a, "A2")

    print("\n-- Case A3: Deal before MAX_ROUNDS (round 3) unaffected by the fix --")
    r = _run_setup_a("regtest_a_deal_at_3", deal_from_round=3)
    check(r["deal_reached"] is True, "A3: deal_reached == True")
    check(r["final_round"] == 3, f"A3: final_round == 3 (got {r['final_round']})")
    check(r["rounds_played"] == 3, f"A3: rounds_played == 3 (got {r['rounds_played']})")

    print("\n=== Setup B (incl. Guardian) ===")

    print("\n-- Case B1: No-Deal must end after exactly MAX_ROUNDS rounds, no round 21, no extra Guardian call --")
    r = _run_setup_b("regtest_b_no_deal", deal_from_round=None)
    check(r["deal_reached"] is False, "B1: deal_reached == False")
    check(r["rounds_played"] == max_rounds_b, f"B1: rounds_played == {max_rounds_b} (got {r['rounds_played']})")
    check(r["rounds"][-1]["round"] == max_rounds_b, f"B1: last logged round == {max_rounds_b} (got {r['rounds'][-1]['round']})")
    n_guardian_calls = sum(1 for rr in r["rounds"] if rr.get("guardian_feedback") is not None)
    check(n_guardian_calls == max_rounds_b, f"B1: exactly {max_rounds_b} Guardian calls logged (got {n_guardian_calls}) -- no 21st Guardian call")
    assert_no_round_beyond_max(r, max_rounds_b, "B1")

    print("\n-- Case B2: Deal in round MAX_ROUNDS (20) must still be detected (Setup B) --")
    r = _run_setup_b("regtest_b_deal_at_20", deal_from_round=max_rounds_b)
    check(r["deal_reached"] is True, "B2: deal_reached == True")
    check(r["final_round"] == max_rounds_b, f"B2: final_round == {max_rounds_b} (got {r['final_round']})")
    assert_no_round_beyond_max(r, max_rounds_b, "B2")

    print("\n-- Case B3: Deal before MAX_ROUNDS (round 3) unaffected (Setup B) --")
    r = _run_setup_b("regtest_b_deal_at_3", deal_from_round=3)
    check(r["deal_reached"] is True, "B3: deal_reached == True")
    check(r["final_round"] == 3, f"B3: final_round == 3 (got {r['final_round']})")

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"RESULT: {len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: all checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("UNEXPECTED EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)
