#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling-origin cross-validation for the v4 DraftVision models.

For each test year Y in TEST_FOLD_YEARS:
  train      = draft classes 2000..Y-2  (+ the 63 seed rows, weight 5)
  calibrate  = {Y-1}                    (success Platt calibrator)
  z-score ref= 2000..Y-1                (recomputed per fold; never includes Y)
  test       = {Y}                      (one year per fold, comparable folds)

Each fold refits all three heads exactly as train_models' EVAL phase does and
scores them via the serving-equivalent paths:
  success: calibrated XGB+CatBoost ensemble  -> AUC, Brier
  grade:   raw member-mean argmax            -> accuracy
  pick:    SERVED 50/50 log-space blend of the pick regressor and the
           classifier-implied expected pick  -> MAE (drafted), Spearman (all),
           Spearman top64 (where n>=20), R1-recall@45

Writes models/experiments/rolling_cv_<features>.json and prints a per-fold
table with mean/std per metric.

Usage:  .venv/bin/python scripts/rolling_cv.py [--features v4]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

# Reuse train_models' loading / fitting / scoring verbatim.
import train_models as tm
from dv_features import SUCCESS_FEATURES

# ── Config ────────────────────────────────────────────────────────────────────
# Feature sets: the fold's X columns. train_models' fit_* helpers train on
# tm.SUCCESS_FEATURES internally, so today only 'v4' (== SUCCESS_FEATURES) is
# runnable end-to-end; the hook exists so a future v5 list can be slotted in.
FEATURE_SETS = {
    "v4": list(SUCCESS_FEATURES),
}

TEST_FOLD_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
FIRST_TRAIN_YEAR = 2000

FOLD_METRICS = [
    "success_auc", "success_brier", "grade_acc",
    "pick_mae_drafted", "pick_spearman_all", "pick_spearman_top64",
    "pick_r1_recall_45",
]


def _assert_fold_no_leakage(train, cal, test, ref_years, test_years) -> None:
    """Per-fold version of tm._assert_no_leakage (which hardcodes tm.TEST_YEARS)."""
    forbidden = set(SUCCESS_FEATURES) & tm.FORBIDDEN_FEATURES
    assert not forbidden, f"forbidden feature in X: {forbidden}"
    assert not (set(train.draft_year.unique()) & test_years), "test year in train"
    assert not (set(cal.draft_year.unique()) & test_years), "test year in cal"
    assert set(test.draft_year.unique()) == test_years, "unexpected test years"
    assert not (ref_years & test_years), "test year in z-score reference"


def run_fold(raw: pd.DataFrame, seeds: pd.DataFrame, features: list, year: int) -> dict:
    np.random.seed(tm.SEED)
    train_years = set(range(FIRST_TRAIN_YEAR, year - 1))   # 2000..Y-2
    cal_years = {year - 1}
    ref_years = set(range(FIRST_TRAIN_YEAR, year))         # 2000..Y-1
    test_years = {year}

    stats = tm.stats_from_ref(raw, ref_years)
    df = tm.apply_z(raw, stats)
    train = pd.concat([df[df.draft_year.isin(train_years)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(cal_years)].reset_index(drop=True)
    test = df[df.draft_year.isin(test_years)].reset_index(drop=True)
    _assert_fold_no_leakage(train, cal, test, ref_years, test_years)

    # Fit all three heads exactly as the EVAL phase does.
    s_members = tm.fit_success_members(train)
    s_bundle = {**s_members,
                "calibrator": tm.fit_success_calibrator(s_members, cal, cal_years)}
    g_members = tm.fit_grade_members(train)
    p_members = tm.fit_pick_members(train)

    X = test[features]
    y_success = test.nfl_success.to_numpy()
    y_grade = test.draft_grade.to_numpy()

    # Serving-equivalent scoring paths.
    p = tm.ensemble_success_probs(s_bundle, X, calibrated=True)
    G_raw = tm.ensemble_grade_probs(g_members, X, calibrated=False)
    g_pred = G_raw.argmax(axis=1)
    reg_pick = tm.ensemble_pick_preds(p_members, X)
    cls_pick = tm.classifier_expected_pick(G_raw)
    blend_pick = np.exp(0.5 * (np.log(reg_pick) + np.log(np.maximum(cls_pick, 1.0))))

    y_pick = np.exp(tm._pick_target(test))
    pick_m = np.isfinite(y_pick)
    pm = tm.pick_metrics(y_pick[pick_m], blend_pick[pick_m])

    consensus_cov = (float(test["consensus_logrank"].notna().mean())
                     if "consensus_logrank" in test.columns else None)

    return {
        "test_year": year,
        "train_years": [FIRST_TRAIN_YEAR, year - 2],
        "cal_year": year - 1,
        "ref_years": [FIRST_TRAIN_YEAR, year - 1],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_pick_scored": int(pick_m.sum()),
        "test_success_rate": round(float(y_success.mean()), 4),
        "test_consensus_coverage": (round(consensus_cov, 4)
                                    if consensus_cov is not None else None),
        "success_auc": round(float(roc_auc_score(y_success, p)), 4),
        "success_brier": round(float(brier_score_loss(y_success, p)), 4),
        "grade_acc": round(float(accuracy_score(y_grade, g_pred)), 4),
        "pick_mae_drafted": pm["mae_picks_drafted"],
        "pick_spearman_all": pm["spearman_all"],
        "pick_spearman_top64": pm["spearman_top64"],
        "pick_r1_recall_45": pm["r1_recall_within_45"],
    }


def summarize(folds: list) -> dict:
    summary = {}
    for m in FOLD_METRICS:
        vals = [f[m] for f in folds if f[m] is not None]
        summary[m] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)), 4),
            "per_fold": vals,
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin CV for DraftVision")
    ap.add_argument("--features", default="v4", choices=sorted(FEATURE_SETS),
                    help="feature set to evaluate (default: v4)")
    args = ap.parse_args()
    features = FEATURE_SETS[args.features]
    if set(features) != set(SUCCESS_FEATURES):
        # tm.fit_* train on tm.SUCCESS_FEATURES internally; a divergent list
        # would silently score different columns than were fit.
        raise SystemExit(f"feature set '{args.features}' != train_models' "
                         "SUCCESS_FEATURES; extend train_models first")

    np.random.seed(tm.SEED)
    print(f"Loading rows via train_models (feature set: {args.features}, "
          f"{len(features)} features)")
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()

    folds = []
    for year in TEST_FOLD_YEARS:
        print(f"Fold {year}: train {FIRST_TRAIN_YEAR}-{year - 2} + seeds, "
              f"cal {year - 1}, ref {FIRST_TRAIN_YEAR}-{year - 1}, test {year} …",
              flush=True)
        fold = run_fold(raw, seeds, features, year)
        folds.append(fold)
        print(f"  AUC {fold['success_auc']:.4f}  Brier {fold['success_brier']:.4f}  "
              f"grade acc {fold['grade_acc']:.4f}  pick MAE {fold['pick_mae_drafted']:.1f}  "
              f"rho {fold['pick_spearman_all']:.4f}", flush=True)

    summary = summarize(folds)

    # Per-fold table.
    hdr = (f"{'year':>5} {'n':>4} {'AUC':>7} {'Brier':>7} {'gAcc':>7} "
           f"{'MAE':>6} {'rho':>7} {'rho64':>7} {'R1@45':>7} {'cons%':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for f in folds:
        r64 = f"{f['pick_spearman_top64']:.4f}" if f['pick_spearman_top64'] is not None else "   n/a"
        r1 = f"{f['pick_r1_recall_45']:.4f}" if f['pick_r1_recall_45'] is not None else "   n/a"
        cov = f"{f['test_consensus_coverage']:.2f}" if f['test_consensus_coverage'] is not None else "n/a"
        print(f"{f['test_year']:>5} {f['n_test']:>4} {f['success_auc']:>7.4f} "
              f"{f['success_brier']:>7.4f} {f['grade_acc']:>7.4f} "
              f"{f['pick_mae_drafted']:>6.1f} {f['pick_spearman_all']:>7.4f} "
              f"{r64:>7} {r1:>7} {cov:>6}")
    print("-" * len(hdr))
    for m in FOLD_METRICS:
        s = summary[m]
        print(f"{m:>22}: mean {s['mean']:.4f}  std {s['std']:.4f}  "
              f"(n={len(s['per_fold'])})")

    out = {
        "feature_set": args.features,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "seed": tm.SEED,
        "design": {
            "test_fold_years": TEST_FOLD_YEARS,
            "train": "2000..Y-2 draft classes + 63 seed rows (weight 5)",
            "calibrate": "Y-1 (success Platt)",
            "z_ref": "2000..Y-1 recomputed per fold (never includes test year)",
            "test": "Y only, one year per fold",
            "std": "sample std (ddof=1) across folds",
        },
        "folds": folds,
        "summary": summary,
    }
    out_path = os.path.join(REPO_ROOT, "models", "experiments",
                            f"rolling_cv_{args.features}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
