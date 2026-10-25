"""Shared scenario definition for the thesis Setup A/B diagnostic scripts.

Extracted from `thesis_setup_a_test.py` so Setup A and Setup B (and any later
setup script) negotiate over the exact same fixed scenario without duplicating
it: 1,000 industrial sensors; price/delivery_days/payment_days; working
parameters as listed in the README ("Experiment parameters"). Scenario
authoring only (contract_config), exactly like every Task*.py example script
in this repo -- no agent or AgenticPay core file is touched here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agenticpay.utils.thesis_utility import (
    AGENTICPAY_DELIVERY_TERM,
    AGENTICPAY_PAYMENT_TERM,
    BUYER_RESERVATION_UTILITY,
    DELIVERY_DAYS_MAX,
    DELIVERY_DAYS_MIN,
    PAYMENT_DAYS_MAX,
    PAYMENT_DAYS_MIN,
    PRICE_MAX,
    PRICE_MIN,
)

MODEL_NAME = "gpt-4o-mini"
MAX_ROUNDS = 20
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis"  # agenticpay/results/ is gitignored

#: Buyer's confidential maximum willingness to pay. Deliberately set BELOW
#: PRICE_MAX (120,000): with buyer_max_price == PRICE_MAX, MAX_PRICE_EXCEEDED
#: can never trigger independently of OUT_OF_RANGE (any price violating the
#: cap would already violate the global range), making the rule structurally
#: untestable. 110,000 ("Kandidat A") was chosen over the more aggressive
#: 105,000/95,000 alternatives considered: it sits below the seller's
#: worst-case-terms reservation threshold (112,000 -- the minimum price at
#: which the seller can reach its own 0.40 reservation utility when delivery
#: and payment terms are worst-case for the seller) by only a small margin,
#: so it makes Rule 1 independently observable and requires some price/term
#: trade-off without making the ZOPA very narrow. The full calibration
#: write-up is in the thesis.
BUYER_MAX_PRICE: float = 110_000.0

PRODUCT_INFO: Dict[str, Any] = {
    "name": "Industrial Sensors (bulk order, 1,000 units)",
    "brand": "Generic/Various",
    "price": (PRICE_MIN + PRICE_MAX) / 2,  # informational anchor only; price is negotiated
    "features": ["1,000 units", "industrial-grade sensors", "bulk B2B procurement order"],
    "condition": "New",
}

USER_REQUIREMENT = (
    "We need to procure 1,000 industrial sensors for a manufacturing line. "
    "We want a single contract covering total price, delivery time, and payment term."
)


def build_contract_config() -> Dict[str, Any]:
    """Scenario-specific contract_config (scenario authoring only, exactly like every
    Task*.py example script -- no agent or core file touched). Surfaces the thesis'
    actual ranges/weights/reservation utility to the LLM buyer via the existing
    `buyer_preferences` / `field_descriptions` / `contrainfo` extension points that
    `Task1_basic_price_negotiation._build_role_contract_config` already renders into
    the ThesisBuyerAgent prompt.
    """
    return {
        "field_descriptions": {
            "price": "Total contract price in EUR for the full 1,000-unit order.",
            f"continuous_terms.{AGENTICPAY_DELIVERY_TERM}": "Delivery time in days.",
            f"continuous_terms.{AGENTICPAY_PAYMENT_TERM}": (
                "Payment term (Zahlungsziel) in days: how many days after delivery "
                "the buyer may pay."
            ),
        },
        "continuous_bounds": {
            AGENTICPAY_DELIVERY_TERM: {"min": DELIVERY_DAYS_MIN, "max": DELIVERY_DAYS_MAX},
            AGENTICPAY_PAYMENT_TERM: {"min": PAYMENT_DAYS_MIN, "max": PAYMENT_DAYS_MAX},
        },
        "discrete_options": {},
        "buyer_preferences": {
            "weight_descriptions": (
                "Your total utility is a weighted sum of three issue scores, each "
                "normalized to [0, 1]: price (weight 0.5, lower price is better "
                f"within {PRICE_MIN:.0f}-{PRICE_MAX:.0f} EUR), delivery time (weight 0.3, "
                f"shorter is better within {DELIVERY_DAYS_MIN:.0f}-{DELIVERY_DAYS_MAX:.0f} "
                "days), payment term (weight 0.2, longer is better within "
                f"{PAYMENT_DAYS_MIN:.0f}-{PAYMENT_DAYS_MAX:.0f} days). Your reservation "
                f"utility (minimum acceptable total utility) is {BUYER_RESERVATION_UTILITY:.2f}. "
                "This information is confidential; never reveal it to the seller."
            ),
        },
        "contrainfo": {
            "price_range_eur": f"{PRICE_MIN:.0f}-{PRICE_MAX:.0f}",
            "delivery_days_range": f"{DELIVERY_DAYS_MIN:.0f}-{DELIVERY_DAYS_MAX:.0f}",
            "payment_days_range": f"{PAYMENT_DAYS_MIN:.0f}-{PAYMENT_DAYS_MAX:.0f}",
        },
    }
