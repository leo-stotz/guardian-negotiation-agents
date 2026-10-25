"""Descriptive analysis: does the buyer actually change its offer after a
Guardian revision request? (Setup B, main experiment, 40 runs.)

Per round in Setup B (see thesis_setup_b_test.py): buyer writes draft 1 ->
Guardian reviews it -> if `revision_required`, the SAME buyer writes exactly one
draft 2 in the same round, which is what gets sent to the seller. The buyer's
"reaction" to a revision request is therefore draft1_contract vs.
draft2_contract of that round.

"Unchanged" = |d price| < PRICE_TOL AND |d delivery_days| < DAYS_TOL AND
|d payment_days| < DAYS_TOL. A sensitivity sweep over larger price tolerances
is reported alongside.

Reads ONLY: agenticpay/results/thesis/main_experiment/setup_b_*_repNN.json
Writes (no overwrite): agenticpay/results/thesis/main_experiment_analysis/
    guardian_revision_reaction_rounds.csv   (one row per revision request)
Descriptive only -- no hypothesis test, not part of H1-H4.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_main_experiment_step3_stats import ANALYSIS_DIR, write_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "agenticpay" / "results" / "thesis" / "main_experiment"

PRICE_TOL = 1.0  # EUR
DAYS_TOL = 0.5   # days
SENSITIVITY_PRICE_TOLS = (1.0, 100.0, 500.0, 1000.0)


def _codes(fb: dict) -> list:
    out = []
    for k in ("compliance_violations", "strategic_concerns"):
        for v in fb.get(k) or []:
            out.append(v.get("code"))
    return out


def _deltas(c1: dict | None, c2: dict | None):
    if not c1 or not c2:
        return None
    return (
        abs(c2["price"] - c1["price"]),
        abs(c2["delivery_days"] - c1["delivery_days"]),
        abs(c2["payment_days"] - c1["payment_days"]),
    )


def main() -> None:
    files = sorted(glob.glob(str(RAW_DIR / "setup_b_*_rep[0-9][0-9].json")))
    assert len(files) == 40, f"expected 40 Setup-B runs, got {len(files)}"

    rows = []
    total_rounds = 0
    for f in files:
        run_id = Path(f).stem  # e.g. setup_b_conceder_rep01
        d = json.load(open(f, encoding="utf-8"))
        strategy = d["strategy"].capitalize()
        total_rounds += len(d["rounds"])
        for r in d["rounds"]:
            if not r["revision_triggered"]:
                continue
            fb = r["guardian_feedback"] or {}
            codes = _codes(fb)
            c1, c2 = r["draft1_contract"], r["draft2_contract"]
            dl = _deltas(c1, c2)
            rows.append({
                "run_id": run_id, "seller_strategy": strategy, "round": r["round"],
                "rounds_played": d["rounds_played"],
                "codes": "|".join(codes),
                "n_codes": len(codes),
                "compliance_codes": "|".join(v["code"] for v in fb.get("compliance_violations") or []),
                "draft1_intends_to_accept": r["draft1_intends_to_accept"],
                "d1_price": c1["price"] if c1 else None,
                "d1_delivery": c1["delivery_days"] if c1 else None,
                "d1_payment": c1["payment_days"] if c1 else None,
                "d2_price": c2["price"] if c2 else None,
                "d2_delivery": c2["delivery_days"] if c2 else None,
                "d2_payment": c2["payment_days"] if c2 else None,
                "d_price": dl[0] if dl else None,
                "d_delivery": dl[1] if dl else None,
                "d_payment": dl[2] if dl else None,
                "contract_missing": dl is None,
                "unchanged": (dl is not None and dl[0] < PRICE_TOL and dl[1] < DAYS_TOL and dl[2] < DAYS_TOL),
                "message_identical": (r["draft1_message"] or "").strip() == (r["draft2_message"] or "").strip(),
                "d1_det_violations": "|".join(map(str, r["draft1_deterministic_violations"] or [])),
                "d2_det_violations": "|".join(map(str, r["draft2_deterministic_violations"] or [])),
                "guardian_reasons": " || ".join(
                    f'{v.get("code")}: {v.get("reason")}'
                    for k in ("compliance_violations", "strategic_concerns")
                    for v in fb.get(k) or []
                ),
            })

    write_csv(rows, ANALYSIS_DIR / "guardian_revision_reaction_rounds.csv")
    df = pd.DataFrame(rows)

    print(f"runs={len(files)} total_rounds={total_rounds} revision_requests={len(df)} "
          f"mean_per_run={len(df)/len(files):.3f}")
    per_run = df.groupby("run_id").size().reindex([Path(f).stem for f in files], fill_value=0)
    print("per-run: median", per_run.median(), "min", per_run.min(), "max", per_run.max(),
          "runs_with_0", int((per_run == 0).sum()))
    print("contract_missing:", int(df["contract_missing"].sum()))
    print("unchanged:", int(df["unchanged"].sum()), "changed:", int((~df["unchanged"] & ~df["contract_missing"]).sum()))
    print("message identical:", int(df["message_identical"].sum()))
    print("draft1_intends_to_accept among requests:", int(df["draft1_intends_to_accept"].sum()),
          "of which unchanged:", int((df["draft1_intends_to_accept"] & df["unchanged"]).sum()))

    print("\nchanged dimensions (changed rounds):")
    ch = df[~df["unchanged"] & ~df["contract_missing"]]
    print(" price", int((ch["d_price"] >= PRICE_TOL).sum()),
          " delivery", int((ch["d_delivery"] >= DAYS_TOL).sum()),
          " payment", int((ch["d_payment"] >= DAYS_TOL).sum()),
          " only_terms_no_price", int(((ch["d_price"] < PRICE_TOL)).sum()))
    print(" price change among changed: median", ch.loc[ch["d_price"] >= PRICE_TOL, "d_price"].median())

    print("\nsensitivity (price tol):")
    for t in SENSITIVITY_PRICE_TOLS:
        u = (df["d_price"] < t) & (df["d_delivery"] < DAYS_TOL) & (df["d_payment"] < DAYS_TOL)
        print(f"  tol<{t}: unchanged {int(u.sum())}/{len(df)}")

    print("\nby strategy:")
    for s, g in df.groupby("seller_strategy"):
        runs_s = sum(1 for f in files if s.lower() in Path(f).stem)
        rounds_s = sum(len(json.load(open(f, encoding='utf-8'))["rounds"]) for f in files if s.lower() in Path(f).stem)
        print(f"  {s}: runs={runs_s} rounds={rounds_s} requests={len(g)} per_run={len(g)/runs_s:.2f} "
              f"unchanged={int(g['unchanged'].sum())} changed={int((~g['unchanged']).sum())} "
              f"msg_identical={int(g['message_identical'].sum())}")

    print("\nby code (non-exclusive):")
    cc = Counter()
    for _, r in df.iterrows():
        for c in set(r["codes"].split("|")) if r["codes"] else {"<none>"}:
            cc[c] += 1
    for c in cc:
        m = df["codes"].apply(lambda x: c in (x.split("|") if x else ["<none>"]))
        g = df[m]
        print(f"  {c}: n={len(g)} unchanged={int(g['unchanged'].sum())} changed={int((~g['unchanged']).sum())}")
        for s, gs in g.groupby("seller_strategy"):
            print(f"      {s}: n={len(gs)} unchanged={int(gs['unchanged'].sum())}")

    print("\nby exclusive code combination:")
    for combo, g in df.groupby("codes"):
        print(f"  [{combo}] n={len(g)} unchanged={int(g['unchanged'].sum())}")

    print("\nBELOW_RESERVATION_UTILITY fixed by draft 2?")
    br = df[df["d1_det_violations"].str.contains("BELOW_RESERVATION_UTILITY")]
    print(f"  n={len(br)} still_violating_after={int(br['d2_det_violations'].str.contains('BELOW_RESERVATION_UTILITY').sum())} "
          f"unchanged={int(br['unchanged'].sum())}")

    print("\nlongest streaks of unchanged revisions per run:")
    streaks = []
    for run_id, g in df.sort_values("round").groupby("run_id"):
        best, cur, start, bstart = 0, 0, None, None
        prev = None
        for _, r in g.iterrows():
            if r["unchanged"] and prev is not None and r["round"] == prev + 1 and cur > 0:
                cur += 1
            elif r["unchanged"]:
                cur, start = 1, r["round"]
            else:
                cur = 0
            if cur > best:
                best, bstart = cur, start
            prev = r["round"]
        streaks.append((best, run_id, bstart, len(g), int(g["unchanged"].sum())))
    for s in sorted(streaks, reverse=True)[:8]:
        print("  streak", s)


if __name__ == "__main__":
    main()
