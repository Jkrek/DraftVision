#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment: quantile + conformal draft-pick intervals.

Fits q10/q90 quantile regressors (XGB reg:quantileerror + CatBoost Quantile),
averages the two families per quantile in log-pick space, conformalizes the
interval on the 2018 calibration fold (split conformal, per-side additive
offsets targeting 80% two-sided coverage), and evaluates on the frozen
2019-2020 holdout.

Reads nothing but training data via train_models; writes only
models/experiments/pick_intervals_results.json.

Usage:  .venv/bin/python scripts/experiment_pick_intervals.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

import train_models as tm
from dv_features import SUCCESS_FEATURES

OUTPUT_PATH = os.path.join(REPO_ROOT, "models", "experiments",
                           "pick_intervals_results.json")

TARGET_COVERAGE = 0.80  # two-sided; 10% miscoverage budget per side


# ── Quantile ensemble ─────────────────────────────────────────────────────────

def fit_quantile_members(train: pd.DataFrame, alpha: float) -> dict:
    """XGB + CatBoost quantile regressors on log(pick), same hyperparam family
    as tm.fit_pick_members; averaged per quantile in log space."""
    y = tm._pick_target(train)
    m = np.isfinite(y)
    X_tr, y_tr = train[SUCCESS_FEATURES][m], y[m]
    w_tr = train.sample_weight.to_numpy()[m]

    xgb_m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=tm.SEED,
        objective="reg:quantileerror", quantile_alpha=alpha,
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)

    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05,
        loss_function=f"Quantile:alpha={alpha}",
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "alpha": alpha, "n_rows": int(m.sum())}


def quantile_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Family-averaged quantile prediction in log-pick space (mirrors
    tm.ensemble_pick_preds: XGB gets the frame, CatBoost gets .values)."""
    return np.mean([bundle["xgb"].predict(X),
                    bundle["cb"].predict(X.values)], axis=0)


# ── Conformal offsets (split conformal, per side) ─────────────────────────────

def conformal_offsets(y_log: np.ndarray, q10: np.ndarray, q90: np.ndarray):
    """Additive offsets in log-pick space so that each side of
    [q10 - lo_off, q90 + hi_off] misses at most 10% of the cal fold
    (=> two-sided coverage 80%). Offsets may be negative (interval shrinks)
    if the raw quantiles over-cover. Uses the standard split-conformal
    finite-sample quantile ceil((n+1)*(1-a))/n on each side's residuals."""
    n = len(y_log)
    per_side = (1.0 - TARGET_COVERAGE) / 2.0          # 0.10 per side
    k = min(int(np.ceil((n + 1) * (1.0 - per_side))), n)
    r_lo = np.sort(q10 - y_log)   # >0 where truth fell below the q10 line
    r_hi = np.sort(y_log - q90)   # >0 where truth fell above the q90 line
    lo_off = float(r_lo[k - 1])
    hi_off = float(r_hi[k - 1])
    return lo_off, hi_off


# ── Metrics helpers ───────────────────────────────────────────────────────────

def interval_stats(y_pick, lo_pick, hi_pick, mask=None) -> dict:
    if mask is None:
        mask = np.ones_like(y_pick, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    y, lo, hi = y_pick[mask], lo_pick[mask], hi_pick[mask]
    width = hi - lo
    return {
        "n": n,
        "coverage": round(float(np.mean((y >= lo) & (y <= hi))), 4),
        "median_width_picks": round(float(np.median(width)), 1),
        "mean_width_picks": round(float(np.mean(width)), 1),
    }


def main() -> int:
    np.random.seed(tm.SEED)

    # ── Frozen eval split (identical to train_models Phase 1) ────────────────
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    df = tm.apply_z(raw, tm.stats_from_ref(raw, tm.EVAL_REF_YEARS))
    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    print(f"split: train {len(train)} / cal {len(cal)} / test {len(test)}")

    # ── 1. Quantile ensembles (q10, q90) ─────────────────────────────────────
    print("fitting q10 members (XGB quantileerror + CatBoost Quantile)…")
    q10_bundle = fit_quantile_members(train, 0.1)
    print("fitting q90 members…")
    q90_bundle = fit_quantile_members(train, 0.9)

    # ── 2. Conformalize on the 2018 cal fold ─────────────────────────────────
    y_cal = tm._pick_target(cal)
    cal_m = np.isfinite(y_cal)
    X_cal = cal[SUCCESS_FEATURES][cal_m]
    y_cal = y_cal[cal_m]
    q10_cal = quantile_preds(q10_bundle, X_cal)
    q90_cal = quantile_preds(q90_bundle, X_cal)
    raw_cal_cov = float(np.mean((y_cal >= q10_cal) & (y_cal <= q90_cal)))
    lo_off, hi_off = conformal_offsets(y_cal, q10_cal, q90_cal)
    cal_cov = float(np.mean((y_cal >= q10_cal - lo_off) & (y_cal <= q90_cal + hi_off)))
    print(f"cal (2018, n={len(y_cal)}): raw q10-q90 coverage {raw_cal_cov:.4f}; "
          f"offsets lo {lo_off:+.4f} / hi {hi_off:+.4f} (log-pick) "
          f"→ conformal cal coverage {cal_cov:.4f}")

    # ── Served point blend on test (regressor × classifier-expected) ─────────
    print("fitting pick regressor + grade members for the served blend…")
    p_members = tm.fit_pick_members(train)
    g_members = tm.fit_grade_members(train)

    y_test = tm._pick_target(test)
    te_m = np.isfinite(y_test)
    X_te = test[SUCCESS_FEATURES][te_m]
    y_pick = np.exp(y_test[te_m])

    reg_pick = tm.ensemble_pick_preds(p_members, X_te)
    P_raw = tm.ensemble_grade_probs(g_members, X_te, calibrated=False)
    cls_pick = tm.classifier_expected_pick(P_raw)
    blend = np.exp(0.5 * (np.log(reg_pick) + np.log(cls_pick)))

    # ── 3. Intervals on the 2019-2020 holdout ────────────────────────────────
    q10_te = quantile_preds(q10_bundle, X_te)
    q90_te = quantile_preds(q90_bundle, X_te)
    lo_log = q10_te - lo_off
    hi_log = q90_te + hi_off
    lo_pick = np.clip(np.exp(lo_log), 1.0, tm.UDFA_PICK)
    hi_pick = np.clip(np.exp(hi_log), 1.0, tm.UDFA_PICK)
    # guard: lo <= hi always (quantile crossing is possible in theory)
    crossed = int(np.sum(lo_pick > hi_pick))
    lo_pick, hi_pick = np.minimum(lo_pick, hi_pick), np.maximum(lo_pick, hi_pick)

    overall = interval_stats(y_pick, lo_pick, hi_pick)
    top64 = interval_stats(y_pick, lo_pick, hi_pick, mask=blend <= 64)

    buckets = {}
    for name, m in [("blend<=32", blend <= 32),
                    ("33-64", (blend > 32) & (blend <= 64)),
                    ("65-150", (blend > 64) & (blend <= 150)),
                    (">150", blend > 150)]:
        buckets[name] = interval_stats(y_pick, lo_pick, hi_pick, mask=m)

    # ── 5. Sanity: does the point blend live inside its own interval? ────────
    blend_outside = (blend < lo_pick) | (blend > hi_pick)
    violation_rate = float(np.mean(blend_outside))
    viol_top64 = float(np.mean(blend_outside[blend <= 64])) if (blend <= 64).any() else None

    print("\n══ HOLDOUT (2019-2020) — CONFORMAL 80% PICK INTERVALS ══")
    print(f"  overall     : {overall}")
    print(f"  blend top-64: {top64}")
    for k, v in buckets.items():
        print(f"  bucket {k:<9}: {v}")
    print(f"  raw (pre-conformal) holdout coverage: "
          f"{float(np.mean((np.log(y_pick) >= q10_te) & (np.log(y_pick) <= q90_te))):.4f}")
    print(f"  blend-outside-interval violation rate: {violation_rate:.4f} "
          f"(top-64: {viol_top64})")
    print(f"  crossed intervals (lo>hi before guard): {crossed}")

    results = {
        "experiment": "pick_intervals_quantile_conformal",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": tm.SEED,
        "target_coverage": TARGET_COVERAGE,
        "features": list(SUCCESS_FEATURES),
        "n_train_rows_pick": q10_bundle["n_rows"],
        "models": {
            "xgboost": {"objective": "reg:quantileerror",
                        "quantile_alphas": [0.1, 0.9],
                        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05},
            "catboost": {"loss_function": ["Quantile:alpha=0.1", "Quantile:alpha=0.9"],
                         "iterations": 400, "depth": 5, "learning_rate": 0.05},
            "combination": "per-quantile mean of the two families in log-pick space",
        },
        "conformal": {
            "method": "split conformal, per-side additive offsets in log-pick space",
            "cal_fold": sorted(tm.CAL_YEARS),
            "cal_n": int(len(y_cal)),
            "raw_cal_coverage_q10_q90": round(raw_cal_cov, 4),
            "lo_offset_log_pick": round(lo_off, 6),
            "hi_offset_log_pick": round(hi_off, 6),
            "cal_coverage_after_offsets": round(cal_cov, 4),
        },
        "holdout_2019_2020": {
            "n": int(te_m.sum()),
            "raw_coverage_pre_conformal": round(
                float(np.mean((np.log(y_pick) >= q10_te) & (np.log(y_pick) <= q90_te))), 4),
            "overall": overall,
            "blend_top64": top64,
            "by_blend_bucket": buckets,
            "crossed_intervals": crossed,
            "blend_outside_interval_rate": round(violation_rate, 4),
            "blend_outside_interval_rate_top64": (
                None if viol_top64 is None else round(viol_top64, 4)),
        },
        "baselines_v4_point_blend": {
            "mae_picks": 45.3, "spearman_all": 0.7962,
            "spearman_top64": 0.6898, "r1_recall_within_45": 0.8438,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
