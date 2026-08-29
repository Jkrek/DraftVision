#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment: continuous career-value regression head (log1p career_av).

Question: does an XGB+CatBoost regressor on y = log1p(max(career_av, 0))
add a user-visible signal ("projected career value") beyond the served
success head, or is it a worse duplicate?

Frozen eval split (identical to train_models.py Phase 1 / generate_backtest):
  z-ref 2000-2018, train 2000-2017 (+63 seeds for the SUCCESS head only),
  calibrate 2018, holdout 2019-2020. SEED=42, deterministic.

Seed rows carry nfl_success/draft_grade but NO career_av (verified below at
runtime) -> they are excluded from the career-AV fit.

Writes models/experiments/career_av_results.json. Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_career_av.py
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# Reuse train_models' loading / fitting / scoring verbatim.
import train_models as tm
from dv_features import SUCCESS_FEATURES

OUTPUT_PATH = os.path.join(REPO_ROOT, "models", "experiments",
                           "career_av_results.json")

# v4 holdout baselines (models/metadata.json / RESULTS.md)
BASELINE_SUCCESS_AUC = 0.8144


def fit_career_av_members(train: pd.DataFrame):
    """XGB+CatBoost regressors on log1p(max(career_av,0)) — mirrors
    fit_pick_members' hyperparameter style (400 trees, depth 4/5, lr 0.05,
    subsample/colsample 0.85, seed 42, sample_weight)."""
    y = np.log1p(np.clip(train["career_av"].to_numpy(dtype=float), 0.0, None))
    m = np.isfinite(y)
    X_tr, y_tr = train[SUCCESS_FEATURES][m], y[m]
    w_tr = train.sample_weight.to_numpy()[m]
    xgb_m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=tm.SEED, objective="reg:squarederror",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05, loss_function="RMSE",
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "n_rows": int(m.sum())}


def ensemble_av_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Mean of member predictions in log1p space (same ensembling shape as
    ensemble_pick_preds)."""
    return np.mean([bundle["xgb"].predict(X), bundle["cb"].predict(X.values)],
                   axis=0)


def decile_table(pred: np.ndarray, actual: np.ndarray) -> list:
    """Mean/median actual career_av by predicted-AV decile (1 = lowest pred,
    10 = highest)."""
    order = np.argsort(np.argsort(pred, kind="stable"))
    dec = np.minimum((order * 10) // len(pred), 9)
    rows = []
    for d in range(10):
        m = dec == d
        rows.append({
            "decile": d + 1,
            "count": int(m.sum()),
            "mean_pred_av": round(float(np.expm1(pred[m]).mean()), 2),
            "mean_actual_av": round(float(actual[m].mean()), 2),
            "median_actual_av": round(float(np.median(actual[m])), 2),
            "max_actual_av": round(float(actual[m].max()), 1),
            "frac_av_ge_30": round(float((actual[m] >= 30).mean()), 4),
        })
    return rows


def main() -> int:
    np.random.seed(tm.SEED)

    print(f"Loading {tm.TRAINING_DATA_PATH}")
    raw_csv = pd.read_csv(tm.TRAINING_DATA_PATH)
    raw = tm.load_raw_rows()
    assert len(raw_csv) == len(raw), "feature frame / raw CSV row mismatch"
    assert (raw_csv["draft_year"].to_numpy() == raw["draft_year"].to_numpy()).all()
    # load_raw_rows drops career_av (it's a FORBIDDEN feature, never in X);
    # attach it as a label column, same pattern as generate_backtest.py.
    raw["career_av"] = raw_csv["career_av"].to_numpy(dtype=float)
    csv_av_populated = float(raw["career_av"].notna().mean())
    n_negative_av = int((raw["career_av"] < 0).sum())

    seeds = tm.seed_rows()
    seeds_have_av = "career_av" in seeds.columns and bool(
        seeds["career_av"].notna().any())
    print(f"career_av populated on CSV rows: {csv_av_populated:.4f} "
          f"({n_negative_av} negative values, clipped to 0)")
    print(f"career_av populated on seed rows: {seeds_have_av} "
          f"-> seeds {'included in' if seeds_have_av else 'EXCLUDED from'} "
          f"the career-AV fit")

    # ── Frozen eval split ────────────────────────────────────────────────────
    eval_stats = tm.stats_from_ref(raw, tm.EVAL_REF_YEARS)
    df_eval = tm.apply_z(raw, eval_stats)
    train = pd.concat([df_eval[df_eval.draft_year.isin(tm.EVAL_TRAIN_YEARS)],
                       seeds], ignore_index=True)
    cal = df_eval[df_eval.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df_eval[df_eval.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    tm._assert_no_leakage(train, cal, test)
    print(f"train {len(train)} rows (incl. {len(seeds)} seeds), "
          f"cal {len(cal)}, holdout {len(test)} ({sorted(tm.TEST_YEARS)})")

    if "career_av" not in train.columns:
        train["career_av"] = np.nan  # would only happen if seeds frame lacked it
    # Seeds have NaN career_av -> the finite-target mask inside
    # fit_career_av_members drops them automatically (verified in output).

    # ── Fit heads ────────────────────────────────────────────────────────────
    print("Fitting career-AV regressors (XGB+CatBoost, log1p target)…")
    av_bundle = fit_career_av_members(train)
    n_seed_in_fit = av_bundle["n_rows"] - int(
        train.loc[train.draft_year != -1, "career_av"].notna().sum())
    print(f"  fit on {av_bundle['n_rows']} rows "
          f"(seed rows in fit: {max(n_seed_in_fit, 0)})")

    print("Fitting success head (frozen-split, for comparison)…")
    s_members = tm.fit_success_members(train)
    s_bundle = {**s_members,
                "calibrator": tm.fit_success_calibrator(s_members, cal,
                                                        tm.CAL_YEARS)}

    # ── Holdout scoring ──────────────────────────────────────────────────────
    X_te = test[SUCCESS_FEATURES]
    y_av = np.clip(test["career_av"].to_numpy(dtype=float), 0.0, None)
    y_success = test.nfl_success.to_numpy()
    drafted = np.isfinite(test["draft_pick"].to_numpy(dtype=float))

    av_pred = ensemble_av_preds(av_bundle, X_te)           # log1p space
    p_success = tm.ensemble_success_probs(s_bundle, X_te, calibrated=True)

    # 1. Spearman overall + drafted-only
    rho_all = float(spearmanr(av_pred, y_av).statistic)
    rho_drafted = float(spearmanr(av_pred[drafted], y_av[drafted]).statistic)

    # 2. Spearman within the model's own top-100 (does it separate stars from
    #    starters at the top of its board?)
    top100 = np.argsort(-av_pred)[:100]
    rho_top100 = float(spearmanr(av_pred[top100], y_av[top100]).statistic)

    # 3. AV-pred as a success ranker
    auc_av = float(roc_auc_score(y_success, av_pred))
    auc_success_head = float(roc_auc_score(y_success, p_success))

    # 4. Redundancy vs the success head
    pearson_r = float(pearsonr(av_pred, p_success).statistic)
    spearman_vs_success = float(spearmanr(av_pred, p_success).statistic)

    deciles = decile_table(av_pred, y_av)

    print("\n══ HOLDOUT (2019-2020) — CAREER-AV HEAD ══")
    print(f"  Spearman(pred, career_av) all rows:     {rho_all:.4f}")
    print(f"  Spearman(pred, career_av) drafted-only: {rho_drafted:.4f} "
          f"(n={int(drafted.sum())})")
    print(f"  Spearman within model top-100:          {rho_top100:.4f}")
    print(f"  AUC as success ranker:                  {auc_av:.4f} "
          f"(success head here: {auc_success_head:.4f}; "
          f"v4 baseline {BASELINE_SUCCESS_AUC})")
    print(f"  Pearson(av_pred, success_prob):         {pearson_r:.4f}")
    print(f"  Spearman(av_pred, success_prob):        {spearman_vs_success:.4f}")
    print("\n  Mean actual career_av by predicted decile (1=low, 10=high):")
    print(f"    {'dec':>3}{'n':>6}{'mean_pred':>11}{'mean_act':>10}"
          f"{'med_act':>9}{'max_act':>9}{'P(AV>=30)':>11}")
    for r in deciles:
        print(f"    {r['decile']:>3}{r['count']:>6}{r['mean_pred_av']:>11.2f}"
              f"{r['mean_actual_av']:>10.2f}{r['median_actual_av']:>9.2f}"
              f"{r['max_actual_av']:>9.1f}{r['frac_av_ge_30']:>11.4f}")

    redundant = pearson_r > 0.95
    complementary_ranker = auc_av >= auc_success_head - 0.01
    if redundant:
        rec = ("drop: Pearson r with the served success probability exceeds "
               "0.95 on holdout — the head is a re-scaled duplicate, not a "
               "new user-visible signal")
    elif rho_top100 >= 0.30 and rho_all >= 0.55:
        rec = ("serve as a display-only 4th head: it orders career value "
               "within the top of the board where the binary success head "
               "saturates, and is not a duplicate (r <= 0.95)")
    else:
        rec = ("drop: not redundant, but it fails to separate stars from "
               "starters at the top of its own board (top-100 Spearman "
               f"{rho_top100:.3f}), which is the only user-visible value a "
               "career-AV head would add over the existing heads")

    results = {
        "experiment": "career_av_regression_head",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target": "log1p(max(career_av, 0))",
        "features": list(SUCCESS_FEATURES),
        "split": {
            "train_years": f"{min(tm.EVAL_TRAIN_YEARS)}-{max(tm.EVAL_TRAIN_YEARS)}",
            "cal_years": sorted(tm.CAL_YEARS),
            "holdout_years": sorted(tm.TEST_YEARS),
            "holdout_rows": int(len(test)),
            "seed": tm.SEED,
        },
        "data_checks": {
            "csv_career_av_populated_frac": round(csv_av_populated, 4),
            "csv_negative_av_rows_clipped": n_negative_av,
            "seeds_have_career_av": seeds_have_av,
            "seeds_excluded_from_av_fit": not seeds_have_av,
            "av_fit_rows": av_bundle["n_rows"],
        },
        "holdout_metrics": {
            "spearman_all": round(rho_all, 4),
            "spearman_drafted_only": round(rho_drafted, 4),
            "n_drafted": int(drafted.sum()),
            "spearman_model_top100": round(rho_top100, 4),
            "auc_av_pred_as_success_ranker": round(auc_av, 4),
            "auc_success_head_refit": round(auc_success_head, 4),
            "auc_success_head_v4_baseline": BASELINE_SUCCESS_AUC,
            "pearson_av_pred_vs_success_prob": round(pearson_r, 4),
            "spearman_av_pred_vs_success_prob": round(spearman_vs_success, 4),
        },
        "decile_table_pred_vs_actual_av": deciles,
        "decision_inputs": {
            "redundant_r_gt_0.95": redundant,
            "auc_within_0.01_of_success_head": complementary_ranker,
        },
        "recommendation": rec,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nRecommendation: {rec}")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
