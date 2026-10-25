"""Deterministic, rule-based seller for the bachelor thesis experiment.

`ThesisRuleBasedSeller` is the controlled counterparty in the negotiation: it has
no LLM dependency, is fully deterministic/reproducible, and uses **only**
`agenticpay.utils.thesis_utility` for every utility computation and value-range
check (no parallel utility logic is implemented here). AgenticPay itself remains
the infrastructure for negotiation flow, contract exchange, history and logging;
no AgenticPay core file is modified for this component.

Verified against the actual AgenticPay source before implementation (write-up in
the development log (not included in this repository)):

- Contract wire format and validation live in
  `agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation`
  (`_normalize_contract` / `_extract_contract` / `_validate_contract`), the env
  class underlying every `s1`-`s25` contract-mode task. Only `"price"` is a
  required field (float > 0); `"continuous_terms"` / `"discrete_terms"` default to
  `{}` and only need to be dicts. This is the *same* schema `thesis_utility.py`'s
  adapters already target, so no new assumption was introduced here.
- Agent interface: `BaseEnv.step(buyer_action: Optional[str], seller_action:
  Optional[str])` -- both are plain negotiation-message strings, stored verbatim
  into `ConversationMemory` and then scanned with
  `re.search(r"<contract>\\s*(.*?)\\s*</contract>", text, re.DOTALL | re.IGNORECASE)`.
  There is **no** separate dict-based return path for rule-based/non-LLM agents;
  buyer and seller use the *identical* interface and the *identical*
  `<contract>...</contract>` extraction convention. `respond()` below therefore
  returns a plain `str` with an embedded `<contract>{...}</contract>` block, using
  the same regex convention (duplicated here only because `_extract_contract` is a
  private method of the env class, not an importable utility -- this is text
  parsing, not utility logic, so it does not violate the "single utility
  implementation" rule).
- `BaseAgent` (the class `BuyerAgent`/`SellerAgent`/`ThesisBuyerAgent` inherit from)
  requires an LLM/VLM `model` in its constructor. Since this seller must have *no*
  LLM dependency, it intentionally does **not** subclass `BaseAgent` -- nothing in
  the env or the driver loop does an `isinstance(..., BaseAgent)` check, they only
  call `.initialize(context)` and `.respond(conversation_history, current_state)`
  duck-typed, so plain duck-type compatibility is sufficient and avoids forcing an
  unused `model` dependency onto a rule-based component.
- The env exposes `current_state["current_round"]` from `_get_observation()`, which
  reflects the env's internal `self.current_round` *before* the current round's
  `step()` call increments it (`self.current_round` starts at 0). `respond()`
  therefore converts it to this module's 1-indexed round convention via
  `current_round = int(current_state.get("current_round", 0)) + 1`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agenticpay.utils.thesis_utility import (
    DELIVERY_DAYS_MAX,
    DELIVERY_DAYS_MIN,
    PAYMENT_DAYS_MAX,
    PAYMENT_DAYS_MIN,
    PRICE_MAX,
    PRICE_MIN,
    SELLER_RESERVATION_UTILITY,
    agenticpay_contract_to_thesis,
    calculate_seller_utility,
    normalize_delivery,
    normalize_payment,
    normalize_price,
    thesis_contract_to_agenticpay,
    validate_value_ranges,
)

# ---------------------------------------------------------------------------
# Parameters for the concession tactic and the counter-offer candidate grid, as
# used in the main experiment. Chosen to produce clearly distinct concession
# shapes (conceder concedes early, boulware concedes late) and checked before
# the main experiment (see README, "Experiment parameters", and the thesis).
# ---------------------------------------------------------------------------

#: Default opening target utility (round 1) if the caller does not pass one.
#: 1.0 would anchor exactly on the seller's ideal-point contract; 0.95 is used
#: instead as a slightly less extreme, still clearly ambitious, opening anchor.
DEFAULT_START_UTILITY: float = 0.95

#: Faratin/Sierra/Jennings (1998) time-dependent concession exponent "beta" per
#: strategy. F(t) = k + (1 - k) * t**(1/beta): beta > 1 concedes quickly early
#: (conceder), beta < 1 stays close to the opening offer until late (boulware).
STRATEGY_BETA: Dict[str, float] = {
    "conceder": 5.0,
    "boulware": 0.2,
}

#: Initial concession fraction k at t=0 (Faratin's F(t) formula). 0.0 means the
#: negotiation opens exactly at start_utility.
DEFAULT_INITIAL_CONCESSION_K: float = 0.0

#: Number of evenly spaced grid points per issue (price/delivery/payment) used to
#: build the deterministic counter-offer candidate set. 11 -> steps of 10% of each
#: issue's configured range. Purely a resolution/runtime trade-off, not a
#: methodological assumption; safe to raise for finer-grained trade-offs later.
DEFAULT_GRID_STEPS: int = 11

#: Floating-point tolerance used when comparing candidate scores for ties.
_TIE_EPSILON: float = 1e-9

_CONTRACT_BLOCK_RE = re.compile(r"<contract>\s*(.*?)\s*</contract>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Standalone, testable helpers (no instance state).
# ---------------------------------------------------------------------------

def contract_distance(contract_a: Dict[str, float], contract_b: Dict[str, float]) -> float:
    """Normalized Euclidean distance between two thesis-internal contracts.

    Each issue is min-max normalized into [0, 1] via `thesis_utility.py`'s own
    `normalize_price` / `normalize_delivery` / `normalize_payment` (no separate
    normalization logic) before computing the distance, so price's larger
    absolute scale cannot dominate delivery/payment. All three issues are
    weighted equally (weight 1) -- no additional weighting is introduced here.

    Args:
        contract_a: {"price": float, "delivery_days": float, "payment_days": float}
        contract_b: same schema as contract_a.

    Returns:
        Euclidean distance in the normalized [0, 1]^3 space (range [0, sqrt(3)]).
    """
    pa = normalize_price(contract_a["price"])
    da = normalize_delivery(contract_a["delivery_days"])
    ya = normalize_payment(contract_a["payment_days"])
    pb = normalize_price(contract_b["price"])
    db = normalize_delivery(contract_b["delivery_days"])
    yb = normalize_payment(contract_b["payment_days"])
    return sqrt((pa - pb) ** 2 + (da - db) ** 2 + (ya - yb) ** 2)


@lru_cache(maxsize=None)
def _build_candidate_grid(steps: int) -> Tuple[Dict[str, float], ...]:
    """Build the deterministic counter-offer candidate grid (cached per `steps`).

    Evenly spaced fractions in [0, 1] (`steps` points, i.e. `steps - 1` intervals)
    are mapped onto each issue's configured range from `thesis_utility.py`, then
    combined into every price x delivery x payment triple (a full factorial grid).
    """
    if steps < 2:
        raise ValueError("steps must be >= 2 to include both range endpoints")
    fractions = [i / (steps - 1) for i in range(steps)]
    candidates = []
    for price_f in fractions:
        price = PRICE_MIN + price_f * (PRICE_MAX - PRICE_MIN)
        for delivery_f in fractions:
            delivery_days = DELIVERY_DAYS_MIN + delivery_f * (DELIVERY_DAYS_MAX - DELIVERY_DAYS_MIN)
            for payment_f in fractions:
                payment_days = PAYMENT_DAYS_MIN + payment_f * (PAYMENT_DAYS_MAX - PAYMENT_DAYS_MIN)
                candidates.append({"price": price, "delivery_days": delivery_days, "payment_days": payment_days})
    return tuple(candidates)


# ---------------------------------------------------------------------------
# The seller.
# ---------------------------------------------------------------------------

class ThesisRuleBasedSeller:
    """Deterministic, non-LLM seller for the bachelor thesis negotiation experiment.

    Negotiates over price, delivery time (`delivery_days`) and payment term
    (`payment_days`). Supports two time-dependent concession strategies
    (`"conceder"`, `"boulware"`) that share the same seller utility function,
    weights, value ranges and reservation utility from `thesis_utility.py`, and
    differ *only* in their concession curve (see `calculate_target_utility`).

    Does not subclass `BaseAgent` (no LLM dependency); it is duck-type
    compatible with the AgenticPay driver loop via `initialize()` / `respond()`.
    """

    def __init__(
        self,
        strategy: str = "conceder",
        reservation_utility: float = SELLER_RESERVATION_UTILITY,
        max_rounds: int = 20,
        start_utility: Optional[float] = None,
        beta: Optional[float] = None,
        k: float = DEFAULT_INITIAL_CONCESSION_K,
        grid_steps: int = DEFAULT_GRID_STEPS,
        name: str = "Seller",
    ) -> None:
        """Initialize the rule-based seller.

        Args:
            strategy: `"conceder"` (cooperative, concedes early) or `"boulware"`
                (competitive, concedes late). Only selects the default `beta`
                below if `beta` is not explicitly given.
            reservation_utility: Hard floor on `calculate_seller_utility()`;
                defaults to `thesis_utility.SELLER_RESERVATION_UTILITY` (0.40,
                see README).
            max_rounds: Total negotiation rounds this seller paces its concession
                curve against; should match the env's `max_rounds`.
            start_utility: Target utility at round 1. Defaults to
                `DEFAULT_START_UTILITY` (0.95) if not given. Must be
                `>= reservation_utility`.
            beta: Faratin-style concession exponent overriding `STRATEGY_BETA[strategy]`.
                Provided for later calibration; leave `None` to use the strategy default.
            k: Initial concession fraction at t=0 (Faratin's F(t) formula, see
                `calculate_target_utility`). Defaults to 0.0 (open at start_utility).
            grid_steps: Resolution of the deterministic counter-offer candidate
                grid (per issue). See `DEFAULT_GRID_STEPS`.
            name: Label used in log messages only.

        Raises:
            ValueError: on an unknown `strategy` or `start_utility < reservation_utility`.
        """
        if strategy not in STRATEGY_BETA:
            raise ValueError(f"strategy must be one of {sorted(STRATEGY_BETA)}, got {strategy!r}")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        self.strategy = strategy
        self.reservation_utility = reservation_utility
        self.max_rounds = max_rounds
        self.start_utility = DEFAULT_START_UTILITY if start_utility is None else start_utility
        if self.start_utility < self.reservation_utility:
            raise ValueError("start_utility must be >= reservation_utility")
        self.beta = STRATEGY_BETA[strategy] if beta is None else beta
        self.k = k
        self.grid_steps = grid_steps
        self.name = name

        self.context: Dict[str, Any] = {}
        self.initialized = False
        self._candidates = _build_candidate_grid(self.grid_steps)

    # -- AgenticPay-facing interface -----------------------------------------

    def initialize(self, context: Dict[str, Any]) -> None:
        """Store negotiation context.

        Kept only for interface parity with the AgenticPay driver loop pattern
        (`buyer.initialize(...)` / `seller.initialize(...)` before the first
        `respond()` call); the seller's own decisions do not depend on it, since
        all ranges/weights/reservation values come from `thesis_utility.py`.
        """
        self.context = context
        self.initialized = True

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        """Produce this round's seller message, AgenticPay-`<contract>`-compatible.

        Reads the buyer's most recent contract proposal from `conversation_history`
        (mirrors AgenticPay's own `<contract>...</contract>` extraction), decides
        accept/counter via `should_accept()`, and returns a short, fixed,
        deterministic text with the resulting contract embedded as
        `<contract>{...}</contract>` in AgenticPay wire format. Never reveals
        `reservation_utility` / target utility / strategy in the returned text.

        Args:
            conversation_history: Same structure the AgenticPay env stores in
                `ConversationMemory` / returns in `observation["conversation_history"]`.
            current_state: Same `observation` dict the env returns; only
                `current_state["current_round"]` is used (env's 0-indexed round
                counter, converted here to this module's 1-indexed convention).

        Returns:
            Negotiation message string containing exactly one
            `<contract>...</contract>` JSON block in AgenticPay wire format.

        Raises:
            ValueError: if `initialize()` was not called first.
        """
        if not self.initialized:
            raise ValueError("Seller not initialized. Call initialize() first.")

        current_round = int(current_state.get("current_round", 0)) + 1
        last_buyer_contract = self._extract_last_buyer_contract(conversation_history)

        if last_buyer_contract is not None and self.should_accept(last_buyer_contract, current_round):
            agreed = last_buyer_contract
            logger.info(
                f"[{self.name}] round={current_round} ACCEPT "
                f"contract={agreed} seller_utility={calculate_seller_utility(agreed):.4f}"
            )
            return self._format_message("accepts the proposed terms", agreed)

        counter = self.generate_counter_offer(last_buyer_contract, current_round)
        logger.info(f"[{self.name}] round={current_round} COUNTER contract={counter}")
        return self._format_message("proposes updated terms", counter)

    # -- Core decision logic (thesis_utility.py only) ------------------------

    def calculate_target_utility(self, current_round: int) -> float:
        """Current-round target seller utility (time-dependent concession).

        Faratin/Sierra/Jennings (1998)-style tactic:
            t = normalized_time(current_round) in [0, 1]
            F(t) = k + (1 - k) * t ** (1 / beta)
            target(t) = start_utility - F(t) * (start_utility - reservation_utility)

        `t = 0` at round 1 -> `target = start_utility` (most ambitious).
        `t = 1` at round `max_rounds` -> `target = reservation_utility` exactly.
        `beta > 1` (conceder): F(t) rises quickly for small t -> fast early
        concessions. `beta < 1` (boulware): F(t) stays near 0 until t is close to
        1 -> concessions arrive late. The result is always clamped to
        `>= reservation_utility`, so it can never imply accepting below the floor.

        Args:
            current_round: 1-indexed round number (1 = first round).

        Returns:
            Target seller utility in `[reservation_utility, start_utility]`.
        """
        t = self._normalized_time(current_round)
        concession_fraction = self.k + (1.0 - self.k) * (t ** (1.0 / self.beta))
        target = self.start_utility - concession_fraction * (self.start_utility - self.reservation_utility)
        return max(target, self.reservation_utility)

    def should_accept(self, contract: Dict[str, float], current_round: int) -> bool:
        """Decide whether to accept a (thesis-internal) buyer contract.

        1. Validate value ranges via `thesis_utility.validate_value_ranges()`.
        2. Compute `calculate_seller_utility()` from `thesis_utility.py` -- the
           only utility computation used, no parallel logic.
        3. Hard floor: reject anything below `reservation_utility`, in every
           round including the last one.
        4. Otherwise accept iff the utility is at least the current round's
           target utility (`calculate_target_utility`).

        Args:
            contract: {"price": float, "delivery_days": float, "payment_days": float}
            current_round: 1-indexed round number.

        Returns:
            True iff the contract should be accepted as-is.
        """
        if not validate_value_ranges(contract):
            return False
        seller_utility = calculate_seller_utility(contract)
        if seller_utility < self.reservation_utility:
            return False
        target_utility = self.calculate_target_utility(current_round)
        return seller_utility >= target_utility

    def generate_counter_offer(
        self,
        last_buyer_contract: Optional[Dict[str, float]],
        current_round: int,
    ) -> Dict[str, float]:
        """Deterministically pick a counter-offer contract for this round.

        Faratin et al. (2002)-style similarity/trade-off selection over a fixed
        candidate grid (see `_build_candidate_grid`), so the counter-offer can
        trade off price against delivery/payment instead of only moving price:

        1. Score every grid candidate by `calculate_seller_utility()` and keep
           only those whose utility is (tied-for-)closest to
           `calculate_target_utility(current_round)`.
        2. If a `last_buyer_contract` is given, narrow that set further to the
           candidate(s) with the smallest `contract_distance()` to it (keeps the
           offer as close as possible to what the buyer just proposed while still
           hitting the seller's target utility).
        3. Deterministic tie-break: sort the remaining candidates ascending by
           `(price, delivery_days, payment_days)` and take the first. This is a
           fixed rule purely for reproducibility, not an additional preference.

        Args:
            last_buyer_contract: Buyer's most recent thesis-internal contract, or
                `None` if no valid buyer contract has been seen yet (e.g. seller
                opens the negotiation) -- step 2 is then skipped.
            current_round: 1-indexed round number.

        Returns:
            {"price": float, "delivery_days": float, "payment_days": float}
        """
        target_utility = self.calculate_target_utility(current_round)

        utility_distances = [(c, abs(calculate_seller_utility(c) - target_utility)) for c in self._candidates]
        min_utility_distance = min(d for _, d in utility_distances)
        utility_best = [c for c, d in utility_distances if d <= min_utility_distance + _TIE_EPSILON]

        if last_buyer_contract is not None:
            similarity_distances = [(c, contract_distance(c, last_buyer_contract)) for c in utility_best]
            min_similarity_distance = min(d for _, d in similarity_distances)
            candidates = [c for c, d in similarity_distances if d <= min_similarity_distance + _TIE_EPSILON]
        else:
            candidates = utility_best

        candidates.sort(key=lambda c: (c["price"], c["delivery_days"], c["payment_days"]))
        return candidates[0]

    # -- Internal helpers ------------------------------------------------------

    def _normalized_time(self, current_round: int) -> float:
        """Map a 1-indexed round number to normalized negotiation time in [0, 1]."""
        if self.max_rounds <= 1:
            return 1.0
        t = (current_round - 1) / (self.max_rounds - 1)
        return min(max(t, 0.0), 1.0)

    def _extract_last_buyer_contract(self, conversation_history: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Extract and adapt the most recent buyer `<contract>` block, if any.

        Uses the same `<contract>...</contract>` extraction convention as
        AgenticPay's own (private) `_extract_contract`, then
        `agenticpay_contract_to_thesis()` for the schema adaptation -- no
        independent contract-parsing rules beyond the shared tag convention, and
        no independent utility/validation logic.
        """
        for message in reversed(conversation_history):
            if message.get("role") != "buyer":
                continue
            match = _CONTRACT_BLOCK_RE.search(message.get("content", "") or "")
            if not match:
                continue
            try:
                agenticpay_contract = json.loads(match.group(1).strip())
                return agenticpay_contract_to_thesis(agenticpay_contract)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(f"[{self.name}] Could not parse buyer contract: {exc}")
                return None
        return None

    def _format_message(self, action_phrase: str, contract: Dict[str, float]) -> str:
        """Build the minimal, deterministic, non-LLM negotiation message text.

        Never includes reservation_utility, target utility, or strategy -- only
        the action phrase and the AgenticPay wire-format contract JSON.
        """
        agenticpay_contract = thesis_contract_to_agenticpay(contract)
        contract_json = json.dumps(agenticpay_contract, sort_keys=True)
        return f"Seller {action_phrase}.\n<contract>{contract_json}</contract>"
