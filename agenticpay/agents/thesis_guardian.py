"""Guardian for the bachelor thesis negotiation experiment.

`ThesisGuardian` reviews the buyer's draft *before* it is sent to the seller,
exactly as specified by the Guardian role prompt (`GUARDIAN_SYSTEM_PROMPT`
below, kept verbatim). It is part of the buyer-side system: it may see
confidential buyer information, but its own textual feedback must never
reveal it, since that feedback is later injected back into the buyer's own
prompt (see `Thesis_buyer_agent.py`'s `guardian_feedback` handling), not sent
to the seller directly.

Two review levels, per the spec:
- Compliance Rules 1-4 (MAX_PRICE_EXCEEDED, OUT_OF_RANGE,
  BELOW_RESERVATION_UTILITY, INVALID_CONTRACT) are deterministic and computed
  here via `run_deterministic_checks()` using `thesis_utility.py` exclusively
  (no parallel utility/validation logic). The Guardian LLM is told to treat
  these as authoritative and must not recompute them.
- Rule 5 (CONFIDENTIAL_INFO_DISCLOSED) and the strategic-quality review
  (UNNECESSARY_CONCESSION, MISSED_TRADEOFF, REPEATED_OFFER_WITHOUT_RESPONSE,
  POOR_ROUND_AWARENESS) require reading natural language and are delegated to
  the Guardian LLM.

No AgenticPay core file is modified. `extract_last_seller_contract()` reuses
the same `<contract>...</contract>` extraction convention used everywhere
else in the thesis code (see `thesis_rule_based_seller.py`'s module docstring
for why this is duplicated here rather than imported: it is AgenticPay's
private `_extract_contract`, not a public utility).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM
from agenticpay.utils.thesis_utility import (
    calculate_buyer_utility,
    validate_buyer_max_price,
    validate_buyer_reservation_utility,
    validate_value_ranges,
)

GUARDIAN_SYSTEM_PROMPT = """You are the Guardian for an autonomous B2B procurement negotiation.

Your role is to review the Buyer Agent's proposed draft BEFORE it is sent to the seller.

You are not the negotiating party.
You do not communicate directly with the seller.
You do not create the final contract yourself.
You only evaluate the buyer's draft and provide structured feedback.
If revision is required, the same Buyer Agent will revise its own draft exactly once.

You are part of the buyer-side system and may use the buyer's confidential information for evaluation.
However, you must never disclose confidential buyer information to the seller.

You must evaluate the draft on two separate levels:

==================================================
1. COMPLIANCE REVIEW
==================================================

The following compliance rules apply:

1. MAX_PRICE_EXCEEDED
The proposed or accepted price must never exceed the buyer's confidential maximum willingness to pay.

2. OUT_OF_RANGE
Price, delivery time, and payment term must remain within the permitted negotiation ranges.

3. BELOW_RESERVATION_UTILITY
If the buyer intends to accept or finalize a deal, the resulting buyer utility must be at least the buyer's reservation utility.

4. INVALID_CONTRACT
The relevant contract must be complete, formally valid, and contain all required negotiation variables.

5. CONFIDENTIAL_INFO_DISCLOSED
The visible message to the seller must not reveal confidential buyer information, including:
- maximum willingness to pay
- reservation utility
- utility weights
- other explicitly confidential preference information

COMPLIANCE CHECK HANDLING:

- Rules 1-4 are provided to you as deterministic system checks.
- Treat these deterministic results as authoritative.
- Do not recalculate, override, or reinterpret Rules 1-4.
- Rule 5 (CONFIDENTIAL_INFO_DISCLOSED) must be evaluated by you from the visible buyer message.

If the buyer draft intends to ACCEPT a seller offer:
- evaluate Rules 1-4 against the seller contract that the buyer intends to accept;
- treat that contract as the candidate contract for compliance evaluation.

If the buyer draft proposes a NEW or REVISED contract:
- evaluate Rules 1-4 against that proposed buyer contract.

==================================================
2. STRATEGIC QUALITY REVIEW
==================================================

Evaluate whether the draft is strategically reasonable from the buyer's perspective.

You may use:
- buyer preferences and utility information
- current buyer draft
- candidate contract
- previous buyer and seller offers
- full negotiation history
- current round
- maximum number of rounds
- buyer's internal mental model, if provided

You do NOT know:
- seller reservation utility
- seller utility weights
- seller strategy type
- any other private seller information

Do not invent or present unknown seller motives, preferences, limits, or strategies as facts.

Check specifically for the following strategic concerns:

UNNECESSARY_CONCESSION
The buyer gives up economic utility without a clear strategic reason or without receiving a meaningful improvement elsewhere.

MISSED_TRADEOFF
The buyer fails to use an apparent opportunity to trade movement on a less important issue for improvement on a more important issue.

REPEATED_OFFER_WITHOUT_RESPONSE
The draft substantially repeats the buyer's previous contract even though the seller changed its offer or provided new information that should be considered.
Only raise this concern if the seller's most recent offer differs from the seller's own previous offer (price, delivery time, or payment term actually changed since the last round). If the seller's own offer is itself unchanged from the round before, do not raise this concern for that reason alone — the buyer repeating its position is not a response failure when the seller has not moved either.

POOR_ROUND_AWARENESS
The buyer's behavior does not adequately account for the remaining negotiation rounds, for example by being unnecessarily rigid near the negotiation deadline or by making excessive concessions very early.

Important:
- Strategic concerns are NOT compliance violations.
- A strategically weak offer may still be fully compliant.
- Base strategic criticism only on information available in the negotiation context.
- Do not assume that a seller concession on one issue necessarily reveals the seller's private preferences.

==================================================
3. REVISION RULE
==================================================

If at least one compliance violation OR at least one strategic concern is identified:

revision_required: true

Otherwise:

revision_required: false

If revision is required:
- provide concise feedback that helps the Buyer Agent revise its own proposal;
- do not provide a replacement contract;
- do not dictate an exact opening price;
- do not dictate a fixed concession percentage;
- do not impose a fixed negotiation strategy;
- preserve the Buyer's autonomy.

The Buyer Agent receives at most ONE revision opportunity per negotiation round.

==================================================
4. OUTPUT FORMAT
==================================================

Return your output exactly in this format:

<guardian_feedback>
compliance_violations:
- code: <CODE> | reason: <one short sentence>

strategic_concerns:
- code: <CODE> | reason: <one short sentence>

revision_required: true/false
</guardian_feedback>

If there are no compliance violations, output:

compliance_violations: []

If there are no strategic concerns, output:

strategic_concerns: []

Example with findings:

<guardian_feedback>
compliance_violations:
- code: BELOW_RESERVATION_UTILITY | reason: The candidate contract fails the deterministic buyer reservation check.

strategic_concerns:
- code: MISSED_TRADEOFF | reason: The seller improved delivery time, but the draft does not consider movement on a less important issue in exchange.

revision_required: true
</guardian_feedback>

Example with no findings:

<guardian_feedback>
compliance_violations: []
strategic_concerns: []
revision_required: false
</guardian_feedback>

Keep all reasons concise, specific, and grounded only in the provided negotiation information."""

_CONTRACT_BLOCK_RE = re.compile(r"<contract>\s*(.*?)\s*</contract>", re.DOTALL | re.IGNORECASE)
_GUARDIAN_BLOCK_RE = re.compile(r"<guardian_feedback>(.*?)</guardian_feedback>", re.DOTALL | re.IGNORECASE)
_FINDING_RE = re.compile(r"-\s*code:\s*([A-Za-z_]+)\s*\|\s*reason:\s*(.+)", re.IGNORECASE)
_REVISION_REQUIRED_RE = re.compile(r"revision_required:\s*(true|false)", re.IGNORECASE)


def extract_last_seller_contract(conversation_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the most recent seller `<contract>` block (AgenticPay wire format),
    using the same extraction convention as everywhere else in this codebase.
    Returns None if no seller message with a parseable contract is found.
    """
    for message in reversed(conversation_history):
        if message.get("role") != "seller":
            continue
        match = _CONTRACT_BLOCK_RE.search(message.get("content", "") or "")
        if not match:
            continue
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def run_deterministic_checks(
    candidate_thesis_contract: Optional[Dict[str, float]],
    buyer_max_price: Optional[float],
) -> List[Dict[str, str]]:
    """Rules 1-4 (deterministic, authoritative per the Guardian's own system prompt).

    Uses only `thesis_utility.py` -- no separate validation logic. Returns a list
    of {"code": ..., "reason": ...} violation dicts (empty if fully compliant).

    Args:
        candidate_thesis_contract: the contract to evaluate (buyer's own draft, or
            the seller's contract if the buyer intends to accept it) in
            thesis-internal schema, or None if it could not be parsed/adapted
            at all (-> INVALID_CONTRACT).
        buyer_max_price: buyer's confidential maximum price; if None, Rule 1 is
            skipped (no configured ceiling to check against).
    """
    if candidate_thesis_contract is None:
        return [{
            "code": "INVALID_CONTRACT",
            "reason": "The candidate contract is missing, malformed, or does not contain all required negotiation variables.",
        }]

    violations: List[Dict[str, str]] = []
    if not validate_value_ranges(candidate_thesis_contract):
        violations.append({
            "code": "OUT_OF_RANGE",
            "reason": "Price, delivery time, or payment term falls outside the permitted negotiation ranges.",
        })
    if buyer_max_price is not None and not validate_buyer_max_price(candidate_thesis_contract["price"], buyer_max_price):
        violations.append({
            "code": "MAX_PRICE_EXCEEDED",
            "reason": "The price exceeds the buyer's confidential maximum willingness to pay.",
        })
    buyer_utility = calculate_buyer_utility(candidate_thesis_contract)
    if not validate_buyer_reservation_utility(buyer_utility):
        violations.append({
            "code": "BELOW_RESERVATION_UTILITY",
            "reason": f"Buyer utility ({buyer_utility:.4f}) is below the buyer's reservation utility.",
        })
    return violations


def parse_guardian_feedback(response_text: str) -> Dict[str, Any]:
    """Parse a `<guardian_feedback>...</guardian_feedback>` block per the fixed
    output format specified in `GUARDIAN_SYSTEM_PROMPT`.

    `revision_required` in the returned dict is derived deterministically as
    `bool(compliance_violations or strategic_concerns)` -- the same rule the
    Guardian's own prompt specifies -- rather than trusting the LLM's literal
    `revision_required: true/false` text for control flow. The LLM-stated value
    is still parsed and returned separately (`llm_stated_revision_required`) so
    a mismatch (the LLM contradicting its own rule) can be logged/inspected
    without silently breaking the revision trigger.

    Never raises. On a completely unparseable response, fails closed
    (`revision_required=True`, a single `PARSE_ERROR` finding) rather than
    silently letting an unreviewable draft through.
    """
    match = _GUARDIAN_BLOCK_RE.search(response_text or "")
    body = match.group(1) if match else (response_text or "")

    compliance_section = _extract_section(body, "compliance_violations:", ["strategic_concerns:", "revision_required:"])
    strategic_section = _extract_section(body, "strategic_concerns:", ["revision_required:"])
    compliance_violations = _parse_findings(compliance_section)
    strategic_concerns = _parse_findings(strategic_section)

    revision_match = _REVISION_REQUIRED_RE.search(body)
    llm_stated_revision_required = (revision_match.group(1).lower() == "true") if revision_match else None
    revision_required = bool(compliance_violations or strategic_concerns)

    parse_ok = match is not None and revision_match is not None
    if match is None:
        compliance_violations = compliance_violations or [{
            "code": "PARSE_ERROR",
            "reason": "Guardian response did not contain a <guardian_feedback> block.",
        }]
        revision_required = True

    return {
        "compliance_violations": compliance_violations,
        "strategic_concerns": strategic_concerns,
        "revision_required": revision_required,
        "llm_stated_revision_required": llm_stated_revision_required,
        "parse_ok": parse_ok,
        "raw_response": response_text,
    }


def _extract_section(body: str, start_marker: str, end_markers: List[str]) -> str:
    lower_body = body.lower()
    start_idx = lower_body.find(start_marker.lower())
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    end_idx = len(body)
    for marker in end_markers:
        idx = lower_body.find(marker.lower(), start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)
    return body[start_idx:end_idx]


def _parse_findings(section_text: str) -> List[Dict[str, str]]:
    return [
        {"code": m.group(1).strip().upper(), "reason": m.group(2).strip()}
        for m in _FINDING_RE.finditer(section_text)
    ]


def format_feedback_for_buyer(feedback: Dict[str, Any]) -> str:
    """Render parsed Guardian feedback as plain text for injection into the
    buyer's own prompt (`current_state["guardian_feedback"]`). Only reasons are
    included (already vetted by the Guardian's own confidentiality rule); no
    contract, price, or strategy is dictated here.
    """
    lines: List[str] = []
    for item in feedback.get("compliance_violations", []):
        lines.append(f"- [{item['code']}] {item['reason']}")
    for item in feedback.get("strategic_concerns", []):
        lines.append(f"- [{item['code']}] {item['reason']}")
    return "\n".join(lines)


class ThesisGuardian:
    """LLM-backed review step sitting between the buyer's draft and the seller.

    Not a `BaseAgent` (no negotiation role of its own, does not talk to the
    seller); duck-typed for whatever orchestration script wires it in (see
    `agenticpay/examples/thesis/thesis_setup_b_test.py`).
    """

    def __init__(self, model: Union[BaseLLM, BaseVLM], name: str = "Guardian") -> None:
        """Args:
            model: LLM/VLM used to run the compliance-Rule-5 and strategic review.
            name: label used in log messages only.
        """
        self.model = model
        self.name = name

    def review(
        self,
        draft_message: str,
        candidate_thesis_contract: Optional[Dict[str, float]],
        intends_to_accept: bool,
        deterministic_violations: List[Dict[str, str]],
        buyer_preferences_text: str,
        conversation_history: List[Dict[str, Any]],
        current_round: int,
        max_rounds: int,
        mental_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one Guardian review and return the parsed feedback dict (see
        `parse_guardian_feedback`).
        """
        prompt = self._build_prompt(
            draft_message=draft_message,
            candidate_thesis_contract=candidate_thesis_contract,
            intends_to_accept=intends_to_accept,
            deterministic_violations=deterministic_violations,
            buyer_preferences_text=buyer_preferences_text,
            conversation_history=conversation_history,
            current_round=current_round,
            max_rounds=max_rounds,
            mental_model=mental_model,
        )
        response = self.model.generate(prompt, temperature=0.0, max_tokens=600)
        feedback = parse_guardian_feedback(response)
        logger.info(
            f"[{self.name} | round {current_round}] revision_required={feedback['revision_required']} "
            f"compliance={feedback['compliance_violations']} strategic={feedback['strategic_concerns']}"
        )
        if not feedback["parse_ok"]:
            logger.warning(f"[{self.name}] Could not fully parse Guardian response; failing closed. Raw: {response!r}")
        return feedback

    def _build_prompt(
        self,
        draft_message: str,
        candidate_thesis_contract: Optional[Dict[str, float]],
        intends_to_accept: bool,
        deterministic_violations: List[Dict[str, str]],
        buyer_preferences_text: str,
        conversation_history: List[Dict[str, Any]],
        current_round: int,
        max_rounds: int,
        mental_model: Optional[str],
    ) -> str:
        history_lines = [
            f"[Round {msg.get('round', '?')}] {str(msg.get('role', 'unknown')).upper()}: {msg.get('content', '')}"
            for msg in conversation_history
        ]
        history_text = "\n".join(history_lines) if history_lines else "No prior messages this negotiation."

        deterministic_text = (
            "\n".join(f"- {v['code']}: {v['reason']}" for v in deterministic_violations)
            if deterministic_violations
            else "No deterministic Rule 1-4 violations detected."
        )

        return f"""{GUARDIAN_SYSTEM_PROMPT}

==================================================
NEGOTIATION CONTEXT
==================================================
Current round: {current_round} of {max_rounds}
Draft action type: {"ACCEPT (buyer intends to accept the seller's last offer)" if intends_to_accept else "NEW/REVISED proposal"}

Buyer's draft message (about to be sent to the seller if you approve it):
{draft_message}

Candidate contract for compliance evaluation (price/delivery_days/payment_days):
{candidate_thesis_contract}

Deterministic Rule 1-4 results (authoritative, do not recompute):
{deterministic_text}

Buyer's confidential preferences and utility information (for your strategic review only; never repeat this in a way that would leak it):
{buyer_preferences_text}

Buyer's own internal mental model for this round, if available:
{mental_model or "Not available."}

Full negotiation history so far:
{history_text}

Now produce your review in exactly the specified <guardian_feedback> output format."""
