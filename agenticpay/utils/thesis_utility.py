"""Thesis utility module: independent MAUT-style evaluation logic for the bachelor
thesis experiment (Buyer, rule-based Seller, Guardian, Evaluator all share this).

This module is deliberately separate from AgenticPay's own contract-mode scoring in
`agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation`
(`_calculate_contract_utilities` / `_calculate_z_max`). That core logic computes
*unbounded*, money-denominated MAUT utilities from a `contract_config`'s
`v_base`/`c_base` plus per-term weights, and only runs bound/option validation when a
`contract_config` is supplied. The thesis instead needs a `contract_config`-independent,
min-max **normalized** utility in [0, 1] with fixed parameters, so it is
implemented independently here rather than reused or overridden.

AgenticPay wire-format contract (verified against `_normalize_contract` /
`_validate_contract` in Task1_basic_price_negotiation.py, which is the env class
underlying all `s1`-`s25` contract-mode tasks):
    {"price": <number>, "continuous_terms": {...}, "discrete_terms": {...}}
    - "price" is the only required field: must be present and parse to float > 0.
    - "continuous_terms" / "discrete_terms" both default to {} if absent (an empty
      dict is explicitly valid) but must be a dict if present (`isinstance(..., dict)`).
    - When a `contract_config` is active, `_validate_contract` additionally requires
      every key in `continuous_bounds`/`discrete_options` to be present and in-range/
      in-options; extra/unknown term keys are rejected.
    - Negotiation issue names live inside "continuous_terms"/"discrete_terms", e.g.
      the existing `s1` beauty-product task uses "delivery_days" as its
      `continuous_bounds` key (see envs/single_buyer_product_seller/Task4_s1_...).

Thesis-internal contract (this module's canonical representation, matches the
example given for the experiment):
    {"price": <float>, "delivery_days": <float>, "payment_days": <float>}
    (Lieferzeit = delivery_days, Zahlungsziel = payment_days)
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Scenario parameters, fixed for the main experiment after the calibration/
# sensitivity check (see README, "Experiment parameters").
# ---------------------------------------------------------------------------

PRICE_MIN: float = 80_000.0
PRICE_MAX: float = 120_000.0

DELIVERY_DAYS_MIN: float = 10.0
DELIVERY_DAYS_MAX: float = 40.0

PAYMENT_DAYS_MIN: float = 14.0
PAYMENT_DAYS_MAX: float = 60.0

BUYER_WEIGHTS: Dict[str, float] = {"price": 0.5, "delivery": 0.3, "payment": 0.2}
SELLER_WEIGHTS: Dict[str, float] = {"price": 0.5, "delivery": 0.2, "payment": 0.3}

BUYER_RESERVATION_UTILITY: float = 0.40
SELLER_RESERVATION_UTILITY: float = 0.40

# Keys used inside an AgenticPay contract's "continuous_terms" dict for the two
# thesis-specific negotiation issues (matches the naming convention already used by
# e.g. Task4_s1_beauty_product_negotiation.py's "delivery_days" continuous_bounds key).
AGENTICPAY_DELIVERY_TERM: str = "delivery_days"
AGENTICPAY_PAYMENT_TERM: str = "payment_days"


# ---------------------------------------------------------------------------
# Adapters: AgenticPay contract <-> thesis-internal contract
# ---------------------------------------------------------------------------

def agenticpay_contract_to_thesis(contract: Dict[str, Any]) -> Dict[str, float]:
    """Convert an AgenticPay wire-format contract to the thesis-internal contract.

    Args:
        contract: AgenticPay contract dict, i.e.
            {"price": <number>, "continuous_terms": {...}, "discrete_terms": {...}}
            as produced by `Task1_basic_price_negotiation._normalize_contract`.

    Returns:
        {"price": float, "delivery_days": float, "payment_days": float}

    Raises:
        ValueError: if "price" is missing/non-numeric, or "continuous_terms" does
            not contain both the delivery and payment term keys with numeric values.
    """
    if not isinstance(contract, dict):
        raise ValueError("contract must be a dict")

    try:
        price = float(contract["price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("contract['price'] is missing or not numeric") from exc

    continuous_terms = contract.get("continuous_terms") or {}
    if AGENTICPAY_DELIVERY_TERM not in continuous_terms:
        raise ValueError(f"continuous_terms missing '{AGENTICPAY_DELIVERY_TERM}'")
    if AGENTICPAY_PAYMENT_TERM not in continuous_terms:
        raise ValueError(f"continuous_terms missing '{AGENTICPAY_PAYMENT_TERM}'")

    try:
        delivery_days = float(continuous_terms[AGENTICPAY_DELIVERY_TERM])
        payment_days = float(continuous_terms[AGENTICPAY_PAYMENT_TERM])
    except (TypeError, ValueError) as exc:
        raise ValueError("delivery/payment terms must be numeric") from exc

    return {"price": price, "delivery_days": delivery_days, "payment_days": payment_days}


def thesis_contract_to_agenticpay(contract: Dict[str, float]) -> Dict[str, Any]:
    """Convert a thesis-internal contract to an AgenticPay wire-format contract.

    Args:
        contract: {"price": float, "delivery_days": float, "payment_days": float}

    Returns:
        AgenticPay contract dict with an empty "discrete_terms" dict (AgenticPay's
        `_normalize_contract` explicitly accepts {} here).
    """
    return {
        "price": float(contract["price"]),
        "continuous_terms": {
            AGENTICPAY_DELIVERY_TERM: float(contract["delivery_days"]),
            AGENTICPAY_PAYMENT_TERM: float(contract["payment_days"]),
        },
        "discrete_terms": {},
    }


# ---------------------------------------------------------------------------
# Objective, party-independent min-max normalization into [0, 1].
# 0 = low end of the configured range, 1 = high end. Preference direction (which
# end is "good") is applied only in calculate_buyer_utility / calculate_seller_utility.
# ---------------------------------------------------------------------------

def normalize_price(price: float) -> float:
    """Min-max normalize price into [0, 1] using [PRICE_MIN, PRICE_MAX]."""
    return _clamp((price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN))


def normalize_delivery(delivery_days: float) -> float:
    """Min-max normalize delivery time into [0, 1] using [DELIVERY_DAYS_MIN, DELIVERY_DAYS_MAX]."""
    return _clamp((delivery_days - DELIVERY_DAYS_MIN) / (DELIVERY_DAYS_MAX - DELIVERY_DAYS_MIN))


def normalize_payment(payment_days: float) -> float:
    """Min-max normalize payment term into [0, 1] using [PAYMENT_DAYS_MIN, PAYMENT_DAYS_MAX]."""
    return _clamp((payment_days - PAYMENT_DAYS_MIN) / (PAYMENT_DAYS_MAX - PAYMENT_DAYS_MIN))


def _clamp(value: float) -> float:
    """Clamp a value into [0, 1] (defends against out-of-range contract terms)."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Party utility functions -- preference direction is decided here only.
# ---------------------------------------------------------------------------

def calculate_buyer_utility(contract: Dict[str, float]) -> float:
    """Compute the buyer's normalized utility for a thesis-internal contract.

    Buyer preference direction: lower price is better, shorter delivery time is
    better, longer payment term is better. Weighted sum over BUYER_WEIGHTS (sums to
    1.0), so the result is in [0, 1].

    Args:
        contract: {"price": float, "delivery_days": float, "payment_days": float}
    """
    price_score = 1.0 - normalize_price(contract["price"])
    delivery_score = 1.0 - normalize_delivery(contract["delivery_days"])
    payment_score = normalize_payment(contract["payment_days"])

    return (
        BUYER_WEIGHTS["price"] * price_score
        + BUYER_WEIGHTS["delivery"] * delivery_score
        + BUYER_WEIGHTS["payment"] * payment_score
    )


def calculate_seller_utility(contract: Dict[str, float]) -> float:
    """Compute the seller's normalized utility for a thesis-internal contract.

    Seller preference direction: higher price is better, longer delivery time is
    better, shorter payment term is better. Weighted sum over SELLER_WEIGHTS (sums
    to 1.0), so the result is in [0, 1].

    Args:
        contract: {"price": float, "delivery_days": float, "payment_days": float}
    """
    price_score = normalize_price(contract["price"])
    delivery_score = normalize_delivery(contract["delivery_days"])
    payment_score = 1.0 - normalize_payment(contract["payment_days"])

    return (
        SELLER_WEIGHTS["price"] * price_score
        + SELLER_WEIGHTS["delivery"] * delivery_score
        + SELLER_WEIGHTS["payment"] * payment_score
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_value_ranges(contract: Dict[str, float]) -> bool:
    """Check that price/delivery_days/payment_days lie within their configured ranges.

    Args:
        contract: {"price": float, "delivery_days": float, "payment_days": float}
    """
    return (
        PRICE_MIN <= contract["price"] <= PRICE_MAX
        and DELIVERY_DAYS_MIN <= contract["delivery_days"] <= DELIVERY_DAYS_MAX
        and PAYMENT_DAYS_MIN <= contract["payment_days"] <= PAYMENT_DAYS_MAX
    )


def validate_buyer_max_price(price: float, buyer_max_price: float) -> bool:
    """Check that a price does not exceed the buyer's confidential maximum price."""
    return price <= buyer_max_price


def validate_buyer_reservation_utility(buyer_utility: float) -> bool:
    """Check that the buyer's utility meets the working reservation utility (0.40)."""
    return buyer_utility >= BUYER_RESERVATION_UTILITY


def validate_seller_reservation_utility(seller_utility: float) -> bool:
    """Check that the seller's utility meets the working reservation utility (0.40)."""
    return seller_utility >= SELLER_RESERVATION_UTILITY
