"""Diagnostic-only plausibility/sensitivity check of the reservation utilities
used in the thesis experiment (BUYER_RESERVATION_UTILITY / SELLER_RESERVATION_UTILITY
in `agenticpay.utils.thesis_utility`, currently 0.40 / 0.40).

Purpose (pre-main-experiment diagnostic, see README, "Experiment parameters"):
enumerate the full grid of contract combinations over the final scenario parameters
(price / delivery_days / payment_days, ranges and weights taken verbatim from
`thesis_utility.py`) and measure how large the mutually-acceptable region is at the
final reservation utility (0.40 / 0.40), and how sensitive that region is to nearby
symmetric reservation values (0.35 / 0.35, 0.45 / 0.45).

This script does NOT modify any existing scenario parameters, prompts, utility
functions, seller strategy, or experiment logic. It imports and calls
`calculate_buyer_utility` / `calculate_seller_utility` / `validate_buyer_max_price`
from `thesis_utility.py` unchanged; the only thing varied locally is the reservation
utility threshold used for the >= comparison (0.35 / 0.40 / 0.45), which is never
written back into `thesis_utility.py` -- BUYER_RESERVATION_UTILITY /
SELLER_RESERVATION_UTILITY in that module remain 0.40 / 0.40 throughout, exactly as
configured for the main experiment. BUYER_MAX_PRICE (110,000) is imported unchanged
from `thesis_scenario.py`.

Outputs (under agenticpay/results/thesis/reservation_sensitivity/, gitignored):
    - grid_full.csv: all 8,064 contract combinations with utilities/flags.
    - summary.csv: aggregated counts/shares per reservation-utility level checked.
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

from agenticpay.utils.thesis_utility import (
    BUYER_WEIGHTS,
    DELIVERY_DAYS_MAX,
    DELIVERY_DAYS_MIN,
    PAYMENT_DAYS_MAX,
    PAYMENT_DAYS_MIN,
    PRICE_MAX,
    PRICE_MIN,
    SELLER_WEIGHTS,
    calculate_buyer_utility,
    calculate_seller_utility,
    validate_buyer_max_price,
)
from agenticpay.examples.thesis.thesis_scenario import BUYER_MAX_PRICE

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "reservation_sensitivity"

PRICE_STEP = 2_000.0
DELIVERY_STEP = 2.0
PAYMENT_STEP = 2.0

# Reservation utility levels to check: main value (0.40/0.40, unchanged in
# thesis_utility.py) plus the requested +/-0.05 sensitivity levels.
RESERVATION_LEVELS: List[float] = [0.35, 0.40, 0.45]
MAIN_LEVEL = 0.40


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Inclusive float range using integer step counting to avoid FP drift."""
    n_steps = round((stop - start) / step)
    return [start + i * step for i in range(n_steps + 1)]


def build_grid() -> List[Dict[str, float]]:
    prices = _frange(PRICE_MIN, PRICE_MAX, PRICE_STEP)
    deliveries = _frange(DELIVERY_DAYS_MIN, DELIVERY_DAYS_MAX, DELIVERY_STEP)
    payments = _frange(PAYMENT_DAYS_MIN, PAYMENT_DAYS_MAX, PAYMENT_STEP)

    grid = []
    for price, delivery_days, payment_days in product(prices, deliveries, payments):
        contract = {
            "price": price,
            "delivery_days": delivery_days,
            "payment_days": payment_days,
        }
        buyer_u = calculate_buyer_utility(contract)
        seller_u = calculate_seller_utility(contract)
        grid.append(
            {
                "price": price,
                "delivery_days": delivery_days,
                "payment_days": payment_days,
                "buyer_utility": buyer_u,
                "seller_utility": seller_u,
                "buyer_max_price_compliant": validate_buyer_max_price(price, BUYER_MAX_PRICE),
            }
        )
    return grid


def annotate_for_level(grid: List[Dict[str, float]], reservation: float) -> None:
    """Add per-level acceptability flags in place (buyer==seller==reservation)."""
    ba_key = f"buyer_acceptable_{reservation:.2f}"
    sa_key = f"seller_acceptable_{reservation:.2f}"
    ma_key = f"mutually_acceptable_{reservation:.2f}"
    for row in grid:
        buyer_ok = row["buyer_utility"] >= reservation
        seller_ok = row["seller_utility"] >= reservation
        row[ba_key] = buyer_ok
        row[sa_key] = seller_ok
        row[ma_key] = buyer_ok and seller_ok


def summarize_for_level(grid: List[Dict[str, float]], reservation: float) -> Dict[str, Any]:
    total = len(grid)
    buyer_ok = sum(1 for r in grid if r["buyer_utility"] >= reservation)
    seller_ok = sum(1 for r in grid if r["seller_utility"] >= reservation)
    mutual = [
        r
        for r in grid
        if r["buyer_utility"] >= reservation and r["seller_utility"] >= reservation
    ]
    mutual_n = len(mutual)
    mutual_compliant = [r for r in mutual if r["buyer_max_price_compliant"]]
    mutual_compliant_n = len(mutual_compliant)

    summary = {
        "reservation_utility_buyer": reservation,
        "reservation_utility_seller": reservation,
        "total_combinations": total,
        "buyer_acceptable_n": buyer_ok,
        "buyer_acceptable_share": buyer_ok / total,
        "seller_acceptable_n": seller_ok,
        "seller_acceptable_share": seller_ok / total,
        "mutually_acceptable_n": mutual_n,
        "mutually_acceptable_share": mutual_n / total,
        "mutually_acceptable_buyer_max_price_compliant_n": mutual_compliant_n,
        "mutually_acceptable_buyer_max_price_compliant_share": mutual_compliant_n / total,
        "mutually_acceptable_share_of_theoretical_zopa_that_is_buyer_compliant": (
            mutual_compliant_n / mutual_n if mutual_n else 0.0
        ),
    }

    if mutual_n:
        prices = sorted({r["price"] for r in mutual})
        summary["mutual_price_min"] = min(prices)
        summary["mutual_price_max"] = max(prices)
        summary["mutual_price_distinct_values"] = len(prices)
    else:
        summary["mutual_price_min"] = ""
        summary["mutual_price_max"] = ""
        summary["mutual_price_distinct_values"] = 0

    if mutual_compliant_n:
        prices_c = sorted({r["price"] for r in mutual_compliant})
        summary["mutual_compliant_price_min"] = min(prices_c)
        summary["mutual_compliant_price_max"] = max(prices_c)
        summary["mutual_compliant_price_distinct_values"] = len(prices_c)
    else:
        summary["mutual_compliant_price_min"] = ""
        summary["mutual_compliant_price_max"] = ""
        summary["mutual_compliant_price_distinct_values"] = 0

    return summary


def characterize_main_level(grid: List[Dict[str, float]], reservation: float) -> str:
    """Short textual characterization of the ZOPA location for one reservation level."""
    mutual = [
        r for r in grid if r["buyer_utility"] >= reservation and r["seller_utility"] >= reservation
    ]
    if not mutual:
        return f"No mutually acceptable combinations at reservation utility {reservation:.2f}."

    prices = sorted({r["price"] for r in mutual})
    compliant = [r for r in mutual if r["buyer_max_price_compliant"]]

    # price-bucket concentration
    price_counts: Dict[float, int] = {}
    for r in mutual:
        price_counts[r["price"]] = price_counts.get(r["price"], 0) + 1
    n_price_buckets_total = round((PRICE_MAX - PRICE_MIN) / PRICE_STEP) + 1

    # delivery/payment combos that are mutually acceptable across ALL tested prices
    # (i.e. term combos robust to price choice within the ZOPA)
    dp_all_prices: Dict[tuple, int] = {}
    for r in mutual:
        key = (r["delivery_days"], r["payment_days"])
        dp_all_prices[key] = dp_all_prices.get(key, 0) + 1
    dp_sorted = sorted(dp_all_prices.items(), key=lambda kv: -kv[1])[:5]

    lines = [
        f"Reservation utility {reservation:.2f}/{reservation:.2f}:",
        f"  Mutually acceptable price range: {min(prices):,.0f}-{max(prices):,.0f} EUR "
        f"({len(prices)}/{n_price_buckets_total} of tested price steps appear at least once).",
        f"  Buyer-max-price-compliant (<= {BUYER_MAX_PRICE:,.0f} EUR) subset: "
        f"{len(compliant)}/{len(mutual)} of ZOPA combinations "
        f"({(len(compliant) / len(mutual) * 100):.1f}%).",
        "  Top delivery_days/payment_days combos by frequency across the ZOPA "
        "(days, days -> count of price steps at which mutually acceptable):",
    ]
    for (delivery, payment), count in dp_sorted:
        lines.append(f"    delivery={delivery:.0f}, payment={payment:.0f} -> {count}")
    return "\n".join(lines)


def write_full_grid_csv(grid: List[Dict[str, float]], path: Path) -> None:
    fieldnames = [
        "price",
        "delivery_days",
        "payment_days",
        "buyer_utility",
        "seller_utility",
        "buyer_acceptable",
        "seller_acceptable",
        "mutually_acceptable",
        "buyer_max_price_compliant",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid:
            writer.writerow(
                {
                    "price": row["price"],
                    "delivery_days": row["delivery_days"],
                    "payment_days": row["payment_days"],
                    "buyer_utility": round(row["buyer_utility"], 6),
                    "seller_utility": round(row["seller_utility"], 6),
                    "buyer_acceptable": row[f"buyer_acceptable_{MAIN_LEVEL:.2f}"],
                    "seller_acceptable": row[f"seller_acceptable_{MAIN_LEVEL:.2f}"],
                    "mutually_acceptable": row[f"mutually_acceptable_{MAIN_LEVEL:.2f}"],
                    "buyer_max_price_compliant": row["buyer_max_price_compliant"],
                }
            )


def write_summary_csv(summaries: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = list(summaries[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def main() -> None:
    print("Building contract grid from thesis_utility.py ranges (unchanged):")
    print(f"  price:         {PRICE_MIN:,.0f}-{PRICE_MAX:,.0f} EUR, step {PRICE_STEP:,.0f}")
    print(f"  delivery_days: {DELIVERY_DAYS_MIN:.0f}-{DELIVERY_DAYS_MAX:.0f}, step {DELIVERY_STEP:.0f}")
    print(f"  payment_days:  {PAYMENT_DAYS_MIN:.0f}-{PAYMENT_DAYS_MAX:.0f}, step {PAYMENT_STEP:.0f}")
    print(f"  buyer_weights:  {BUYER_WEIGHTS}")
    print(f"  seller_weights: {SELLER_WEIGHTS}")
    print(f"  buyer_max_price (confidential, thesis_scenario.py): {BUYER_MAX_PRICE:,.0f} EUR")

    grid = build_grid()
    n_prices = len(_frange(PRICE_MIN, PRICE_MAX, PRICE_STEP))
    n_deliv = len(_frange(DELIVERY_DAYS_MIN, DELIVERY_DAYS_MAX, DELIVERY_STEP))
    n_pay = len(_frange(PAYMENT_DAYS_MIN, PAYMENT_DAYS_MAX, PAYMENT_STEP))
    expected = n_prices * n_deliv * n_pay
    print(
        f"\nGrid size: {n_prices} price steps x {n_deliv} delivery steps x "
        f"{n_pay} payment steps = {len(grid)} combinations "
        f"(expected {expected}; matches spec's 8,064: {expected == 8064})."
    )
    assert len(grid) == expected

    for level in RESERVATION_LEVELS:
        annotate_for_level(grid, level)

    summaries = [summarize_for_level(grid, level) for level in RESERVATION_LEVELS]

    print("\n=== Summary per reservation utility level ===")
    for s in summaries:
        print(
            f"\nReservation {s['reservation_utility_buyer']:.2f}/{s['reservation_utility_seller']:.2f}:"
        )
        print(f"  total combinations:                 {s['total_combinations']}")
        print(
            f"  buyer acceptable:                   {s['buyer_acceptable_n']:5d} "
            f"({s['buyer_acceptable_share'] * 100:5.1f}%)"
        )
        print(
            f"  seller acceptable:                  {s['seller_acceptable_n']:5d} "
            f"({s['seller_acceptable_share'] * 100:5.1f}%)"
        )
        print(
            f"  mutually acceptable (theoretical ZOPA): {s['mutually_acceptable_n']:5d} "
            f"({s['mutually_acceptable_share'] * 100:5.1f}%)"
        )
        print(
            f"  mutually acceptable & buyer-max-price-compliant: "
            f"{s['mutually_acceptable_buyer_max_price_compliant_n']:5d} "
            f"({s['mutually_acceptable_buyer_max_price_compliant_share'] * 100:5.1f}%)"
        )
        if s["mutually_acceptable_n"]:
            print(
                f"  mutual price range: {s['mutual_price_min']:,.0f}-{s['mutual_price_max']:,.0f} EUR"
                if isinstance(s["mutual_price_min"], (int, float)) and s["mutual_price_min"] != ""
                else "  mutual price range: n/a"
            )

    print("\n=== Characterization of main value (0.40/0.40) ZOPA ===")
    print(characterize_main_level(grid, MAIN_LEVEL))

    grid_csv_path = OUT_DIR / "grid_full.csv"
    summary_csv_path = OUT_DIR / "summary.csv"
    write_full_grid_csv(grid, grid_csv_path)
    write_summary_csv(summaries, summary_csv_path)
    print(f"\nWrote full grid ({len(grid)} rows) to: {grid_csv_path}")
    print(f"Wrote summary ({len(summaries)} rows) to: {summary_csv_path}")


if __name__ == "__main__":
    main()
