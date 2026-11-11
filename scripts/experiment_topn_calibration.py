#!/usr/bin/env python3
"""Top-N calibration: how often does the model's class-rank-K player actually
go in the top N?

The futures board used to derive P(pick <= N) from a conformal interval under
a log-uniform assumption. That capped every top-5 number near 66% and ordered
players by interval width rather than by rank (audit 2026-11-11). This script
measures the quantity directly instead:

  for each rolling-origin fold year Y (2015..2020), fit the pick heads on
  2000..Y-2 exactly as rolling_cv does, score year Y, rank the test class by
  blended model pick, and record (model_rank, actual_pick).

It then fits, per threshold N, a logistic curve  P(pick <= N) = sigmoid(a +
b*log(rank))  on five folds and reports Brier / log-loss on the sixth
(leave-one-year-out), against the base-rate baseline. The pooled fit is
written to models/topn_calibration.json for dv_edge to serve.

Usage: .venv/bin/python scripts/experiment_topn_calibration.py
"""
import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import train_models as tm
from dv_features import SUCCESS_FEATURES
from rolling_cv import FIRST_TRAIN_YEAR, TEST_FOLD_YEARS

THRESHOLDS = [1, 3, 5, 10, 16, 32, 64, 100]
OOF_PATH = os.path.join(REPO_ROOT, "models", "experiments", "topn_oof.json")
OUT_PATH = os.path.join(REPO_ROOT, "models", "topn_calibration.json")


def fold_rows(raw, seeds, year):
    np.random.seed(tm.SEED)
    train_years = set(range(FIRST_TRAIN_YEAR, year - 1))
    ref_years = set(range(FIRST_TRAIN_YEAR, year))
    stats = tm.stats_from_ref(raw, ref_years)
    df = tm.apply_z(raw, stats)
    train = pd.concat([df[df.draft_year.isin(train_years)], seeds], ignore_index=True)
    test = df[df.draft_year == year].reset_index(drop=True)
    g_members = tm.fit_grade_members(train)
    p_members = tm.fit_pick_members(train)
    X = test[list(SUCCESS_FEATURES)]
    G_raw = tm.ensemble_grade_probs(g_members, X, calibrated=False)
    reg_pick = tm.ensemble_pick_preds(p_members, X)
    cls_pick = tm.classifier_expected_pick(G_raw)
    blend = np.exp(0.5 * (np.log(reg_pick) + np.log(np.maximum(cls_pick, 1.0))))
    y_pick = np.exp(tm._pick_target(test))
    order = np.argsort(np.argsort(blend, kind="stable"), kind="stable") + 1
    out = []
    for r, b, y in zip(order, blend, y_pick):
        if not np.isfinite(y):
            continue
        out.append({"year": int(year), "rank": int(r), "model_pick": round(float(b), 3),
                    "actual_pick": round(float(y), 1)})
    return out


def fit_curve(ranks, y):
    """sigmoid(a + b*log(rank)); returns (a, b) or None when y is degenerate."""
    if y.sum() < 3 or y.sum() > len(y) - 3:
        return None
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(np.log(ranks).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def predict(ab, ranks):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a + b * np.log(ranks))))


def main() -> int:
    if os.path.exists(OOF_PATH) and "--refit" not in sys.argv:
        rows = json.load(open(OOF_PATH))["rows"]
        print(f"loaded {len(rows)} OOF rows from {OOF_PATH} (pass --refit to recompute)")
    else:
        raw, seeds = tm.load_raw_rows(), tm.seed_rows()
        rows = []
        for year in TEST_FOLD_YEARS:
            fr = fold_rows(raw, seeds, year)
            rows.extend(fr)
            print(f"fold {year}: {len(fr)} rows", flush=True)
        os.makedirs(os.path.dirname(OOF_PATH), exist_ok=True)
        json.dump({"rows": rows}, open(OOF_PATH, "w"))

    df = pd.DataFrame(rows)
    years = sorted(df.year.unique())
    report, curves = {}, {}
    for n in THRESHOLDS:
        y = (df.actual_pick <= n + 0.5).astype(int).to_numpy()
        ranks = df["rank"].to_numpy(dtype=float)
        # leave-one-year-out
        briers, base_briers, lls = [], [], []
        ok = True
        for yr in years:
            tr, te = (df.year != yr).to_numpy(), (df.year == yr).to_numpy()
            ab = fit_curve(ranks[tr], y[tr])
            if ab is None:
                ok = False
                break
            p = np.clip(predict(ab, ranks[te]), 1e-6, 1 - 1e-6)
            # score only the part of the class a market would ever list
            m = ranks[te] <= max(4 * n, 40)
            yt, pt = y[te][m], p[m]
            base = np.full(len(yt), y[tr][ranks[tr] <= max(4 * n, 40)].mean())
            briers.append(float(np.mean((pt - yt) ** 2)))
            base_briers.append(float(np.mean((base - yt) ** 2)))
            lls.append(float(-np.mean(yt * np.log(pt) + (1 - yt) * np.log(1 - pt))))
        if not ok:
            print(f"top-{n}: too few positives to fit; skipped")
            continue
        ab = fit_curve(ranks, y)
        curves[str(n)] = {"a": round(ab[0], 5), "b": round(ab[1], 5)}
        table = {str(r): round(float(predict(ab, np.array([float(r)]))[0]), 3)
                 for r in (1, 2, 3, 5, 8, 12, 20, 32, 50, 100)}
        emp = {}
        for lo, hi in ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40)):
            m = (ranks >= lo) & (ranks <= hi)
            emp[f"{lo}-{hi}"] = [int(y[m].sum()), int(m.sum())]
        report[str(n)] = {
            "loyo_brier": round(float(np.mean(briers)), 4),
            "loyo_brier_per_year": [round(b, 4) for b in briers],
            "base_rate_brier": round(float(np.mean(base_briers)), 4),
            "loyo_logloss": round(float(np.mean(lls)), 4),
            "curve_at_rank": table,
            "empirical_hits_by_rank_band": emp,
        }
        print(f"top-{n:<3} LOYO Brier {np.mean(briers):.4f} vs base-rate {np.mean(base_briers):.4f}"
              f" | curve {table} | empirical {emp}")

    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "logistic on log(model class rank), pooled rolling-origin folds",
        "folds": [int(y) for y in years],
        "n_rows": int(len(df)),
        "curves": curves,
        "validation": report,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2, sort_keys=True)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
