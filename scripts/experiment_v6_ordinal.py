#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 ordinal-grade experiment — rolling-origin A/B of the GRADE head.

The served grade head is a flat 4-class softmax (0=Top50, 1=Day2, 2=Late,
3=UDFA; XGB+CatBoost member-mean, argmax). The classes are ordered, so this
compares ordinal formulations against it on the six rolling folds used by
scripts/rolling_cv.py / rolling_ab_v5.py (test year Y in 2015..2020, train
2000..Y-2 + seeds, cal Y-1, z-ref 2000..Y-1):

  base       flat 4-class XGB+CB (production recipe, tm.fit_grade_members)
  base_cb    the CatBoost MultiClass member alone (for candidate (b)'s
             "CatBoost MultiClass vs regression-on-index" comparison)
  a_fh       Frank & Hall: three XGB+CB binary heads P(y > k), k=0,1,2,
             turned into class probabilities by differencing (monotonized),
             argmax; implied pick = P @ midpoints (as served)
  b_idx      XGB+CB regression on the class index (0..3), same balanced
             weights as the flat head; pred = round(yhat); implied pick =
             piecewise-linear map of yhat onto the class midpoints
  b_idx_cal  same regressors, cut-points tuned on the CAL year (Y-1) only
  c_reg50    grade = bucket(regressor pick) at 50/100/262 (task cutoffs);
             the classifier head is dropped, so the served pick is the
             regressor alone
  c_regfit   grade = bucket(regressor pick) with cut-points fitted to the
             label/pick relationship on pre-test rows (train+cal) only;
             served pick = regressor alone
  c_blend50  grade = bucket(SERVED blend) at 50/100/262 — the flat head is
             kept for the blend, so pick metrics equal base by construction
  c_blendfit as c_blend50 with pre-test-fitted cut-points

For every arm and fold: accuracy, adjacent accuracy (|pred-true|<=1),
macro-F1, and the downstream served pick blend (50/50 log-space blend of the
pick regressor and the arm's implied pick): MAE (drafted), Spearman (all),
Spearman top-64, R1 recall within 45.

The success head is NOT refit — it is untouched by any grade-head change, so
success AUC/Brier are identical to the baseline by construction.

Leakage guarantees: X = tm.SUCCESS_FEATURES (asserted disjoint from
FORBIDDEN_FEATURES); z-stats from 2000..Y-1 rows only; every fitted
cut-point uses train (2000..Y-2) and/or cal (Y-1) rows only; the test year
is asserted absent from train, cal and the z-reference on every fold.

Writes models/experiments/v6_ordinal_results.json.

Usage:  .venv/bin/python scripts/experiment_v6_ordinal.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score

import train_models as tm

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_ordinal_results.json")
FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]
MIDS = np.array([25.0, 85.0, 190.0, tm.UDFA_PICK])
TASK_CUTS = (50.0, 100.0, 262.0)

ARMS = ["base", "base_cb", "a_fh", "b_idx", "b_idx_cal",
        "c_reg50", "c_regfit", "c_blend50", "c_blendfit"]
GRADE_METRICS = ["grade_acc", "grade_adj_acc", "grade_macro_f1"]
PICK_METRICS = ["pick_blend_mae", "pick_blend_spearman", "pick_blend_top64",
                "pick_blend_r1_recall_45"]
ALL_METRICS = GRADE_METRICS + PICK_METRICS + ["cls_pick_mae", "cls_pick_spearman"]
LOWER_IS_BETTER = {"pick_blend_mae", "cls_pick_mae"}
# Served metrics that must be non-inferior (grade_acc is the target)
SERVED = ["grade_acc", "pick_blend_mae", "pick_blend_spearman",
          "pick_blend_top64", "pick_blend_r1_recall_45"]


# ── helpers ──────────────────────────────────────────────────────────────────

def balanced_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exactly the class-balancing used by tm.fit_grade_members."""
    total = w.sum()
    cls_w = {c: float(total / (4.0 * max(w[y == c].sum(), 1e-9))) for c in range(4)}
    return w * np.array([cls_w[c] for c in y])


def fit_fh_members(train: pd.DataFrame) -> list:
    """Frank & Hall: binary XGB+CB per threshold k, target 1[y > k]."""
    X_tr, y_tr, w_tr = tm._xy(train, "draft_grade")
    w_eff = balanced_weights(y_tr, w_tr)
    heads = []
    for k in range(3):
        yb = (y_tr > k).astype(int)
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
            random_state=tm.SEED, eval_metric="logloss",
        )
        xgb_m.fit(X_tr, yb, sample_weight=w_eff)
        cb_m = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05, loss_function="Logloss",
            random_seed=tm.SEED, verbose=0, allow_writing_files=False,
        )
        cb_m.fit(X_tr, yb, sample_weight=w_eff)
        heads.append({"xgb": xgb_m, "cb": cb_m})
    return heads


def fh_probs(heads: list, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities from P(y>k): monotonize cumulative tails then
    difference; clip and renormalize."""
    gt = np.stack([
        np.mean([h["xgb"].predict_proba(X)[:, 1], h["cb"].predict_proba(X)[:, 1]], axis=0)
        for h in heads], axis=1)                       # (n, 3) = P(y>0), P(y>1), P(y>2)
    gt = np.minimum.accumulate(gt, axis=1)             # enforce P(y>0) >= P(y>1) >= P(y>2)
    P = np.column_stack([1.0 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1] - gt[:, 2], gt[:, 2]])
    P = np.clip(P, 0.0, None)
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)


def fit_index_members(train: pd.DataFrame) -> dict:
    """XGB+CB regressors on the class index 0..3 (balanced weights so the
    decision rule sees the same class mass as the flat head)."""
    X_tr, y_tr, w_tr = tm._xy(train, "draft_grade")
    w_eff = balanced_weights(y_tr, w_tr)
    xgb_m = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        random_state=tm.SEED, objective="reg:squarederror",
    )
    xgb_m.fit(X_tr, y_tr.astype(float), sample_weight=w_eff)
    cb_m = CatBoostRegressor(
        iterations=300, depth=5, learning_rate=0.05, loss_function="RMSE",
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr.astype(float), sample_weight=w_eff)
    return {"xgb": xgb_m, "cb": cb_m}


def index_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.mean([bundle["xgb"].predict(X), bundle["cb"].predict(X.values)], axis=0),
                   0.0, 3.0)


def index_to_pick(yhat: np.ndarray) -> np.ndarray:
    """Piecewise-linear map of a continuous class index onto the class
    midpoints the served classifier_expected_pick uses."""
    return np.interp(yhat, [0.0, 1.0, 2.0, 3.0], MIDS)


def tune_index_cuts(yhat: np.ndarray, y: np.ndarray) -> tuple:
    """Grid-search three cut-points on a continuous score to maximize
    accuracy (called with CAL-year rows only)."""
    grid = np.round(np.arange(0.2, 2.9, 0.05), 3)
    best, best_c = -1.0, (0.5, 1.5, 2.5)
    for c0 in grid:
        for c1 in grid[grid > c0]:
            for c2 in grid[grid > c1]:
                pred = np.digitize(yhat, [c0, c1, c2])
                acc = float(np.mean(pred == y))
                if acc > best:
                    best, best_c = acc, (float(c0), float(c1), float(c2))
    return best_c


def bucket_pick(pick: np.ndarray, cuts: tuple) -> np.ndarray:
    c0, c1, c2 = cuts
    return np.where(pick <= c0, 0, np.where(pick <= c1, 1, np.where(pick <= c2, 2, 3)))


def fit_pick_cuts(df_pre: pd.DataFrame) -> tuple:
    """Cut-points that best reproduce draft_grade from the TRUE pick on
    pre-test rows (labels are round-based, not 50/100). Third cut stays 262:
    UDFA rows have no pick and are the pseudo-pick 300."""
    y_pick = np.exp(tm._pick_target(df_pre))
    y = df_pre.draft_grade.to_numpy()
    m = np.isfinite(y_pick)
    y_pick, y = y_pick[m], y[m]
    best, best_c = -1.0, (50.0, 100.0)
    for c0 in range(40, 90):
        for c1 in range(100, 170):
            pred = bucket_pick(y_pick, (c0, c1, 262.0))
            acc = float(np.mean(pred == y))
            if acc > best:
                best, best_c = acc, (float(c0), float(c1))
    return (best_c[0], best_c[1], 262.0)


def score(y_g, pred, y_pick, pick_m, reg, cls_pick, blend_override=None) -> dict:
    """Grade metrics + downstream served-blend metrics for one arm."""
    if blend_override is not None:
        blend = blend_override
    else:
        blend = np.exp(0.5 * (np.log(reg) + np.log(np.maximum(cls_pick, 1.0))))
    pk = tm.pick_metrics(y_pick[pick_m], blend[pick_m])
    out = {
        "grade_acc": round(float(accuracy_score(y_g, pred)), 4),
        "grade_adj_acc": round(float(np.mean(np.abs(pred - y_g) <= 1)), 4),
        "grade_macro_f1": round(float(f1_score(y_g, pred, average="macro")), 4),
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
        "pred_class_counts": [int((pred == c).sum()) for c in range(4)],
    }
    if cls_pick is not None:
        ck = tm.pick_metrics(y_pick[pick_m], np.maximum(cls_pick[pick_m], 1.0))
        out["cls_pick_mae"] = ck["mae_picks_drafted"]
        out["cls_pick_spearman"] = ck["spearman_all"]
    else:
        out["cls_pick_mae"] = None
        out["cls_pick_spearman"] = None
    return out


# ── one fold ─────────────────────────────────────────────────────────────────

def run_fold(raw: pd.DataFrame, seeds: pd.DataFrame, year: int) -> dict:
    np.random.seed(tm.SEED)
    tm.EVAL_TRAIN_YEARS = set(range(2000, year - 1))
    tm.CAL_YEARS = {year - 1}
    tm.TEST_YEARS = {year}
    ref_years = set(range(2000, year))

    df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))
    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds], ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    pre = df[df.draft_year < year].reset_index(drop=True)   # train+cal, no seeds

    # leakage guards
    assert not (set(tm.SUCCESS_FEATURES) & tm.FORBIDDEN_FEATURES)
    assert not (set(train.draft_year.unique()) & tm.TEST_YEARS)
    assert not (set(cal.draft_year.unique()) & tm.TEST_YEARS)
    assert not (ref_years & tm.TEST_YEARS)
    assert set(test.draft_year.unique()) == tm.TEST_YEARS
    assert pre.draft_year.max() == year - 1

    feats = list(tm.SUCCESS_FEATURES)
    X = test[feats]
    X_cal = cal[feats]
    y_g = test.draft_grade.to_numpy()
    y_cal = cal.draft_grade.to_numpy()
    y_pick = np.exp(tm._pick_target(test))
    pick_m = np.isfinite(y_pick)

    t0 = time.time()
    # shared pick regressor (unchanged across arms)
    p_members = tm.fit_pick_members(train)
    reg = tm.ensemble_pick_preds(p_members, X)

    # baseline flat head
    g_members = tm.fit_grade_members(train)
    P_base = tm.ensemble_grade_probs(g_members, X, calibrated=False)
    P_cb = g_members["cb"].predict_proba(X)
    # F&H
    fh = fit_fh_members(train)
    P_fh = fh_probs(fh, X)
    # index regression
    idx = fit_index_members(train)
    yhat = index_preds(idx, X)
    yhat_cal = index_preds(idx, X_cal)
    idx_cuts = tune_index_cuts(yhat_cal, y_cal)
    # pick-derived cut-points from pre-test rows
    fit_cuts = fit_pick_cuts(pre)
    fit_secs = round(time.time() - t0, 1)

    base_cls_pick = tm.classifier_expected_pick(P_base)
    base_blend = np.exp(0.5 * (np.log(reg) + np.log(np.maximum(base_cls_pick, 1.0))))

    arms = {}
    arms["base"] = score(y_g, P_base.argmax(1), y_pick, pick_m, reg, base_cls_pick)
    arms["base_cb"] = score(y_g, P_cb.argmax(1), y_pick, pick_m, reg,
                            tm.classifier_expected_pick(P_cb))
    arms["a_fh"] = score(y_g, P_fh.argmax(1), y_pick, pick_m, reg,
                         tm.classifier_expected_pick(P_fh))
    arms["b_idx"] = score(y_g, np.clip(np.rint(yhat), 0, 3).astype(int), y_pick, pick_m,
                          reg, index_to_pick(yhat))
    arms["b_idx_cal"] = score(y_g, np.digitize(yhat, list(idx_cuts)), y_pick, pick_m,
                              reg, index_to_pick(yhat))
    # (c): classifier dropped -> served pick is the regressor alone
    arms["c_reg50"] = score(y_g, bucket_pick(reg, TASK_CUTS), y_pick, pick_m, reg, None,
                            blend_override=reg)
    arms["c_regfit"] = score(y_g, bucket_pick(reg, fit_cuts), y_pick, pick_m, reg, None,
                             blend_override=reg)
    # (c'): grade from the served blend, flat head kept for the blend
    arms["c_blend50"] = score(y_g, bucket_pick(base_blend, TASK_CUTS), y_pick, pick_m, reg,
                              base_cls_pick, blend_override=base_blend)
    arms["c_blendfit"] = score(y_g, bucket_pick(base_blend, fit_cuts), y_pick, pick_m, reg,
                               base_cls_pick, blend_override=base_blend)

    return {
        "test_year": year,
        "train_years": [2000, year - 2], "cal_year": year - 1, "ref_years": [2000, year - 1],
        "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test)),
        "true_class_counts": [int((y_g == c).sum()) for c in range(4)],
        "idx_cuts_cal": list(idx_cuts),
        "pick_cuts_fit": list(fit_cuts),
        "fit_seconds": fit_secs,
        "arms": arms,
    }


# ── summary / verdict ────────────────────────────────────────────────────────

def summarize(folds: list) -> dict:
    base_std = {}
    for m in ALL_METRICS:
        vals = [f["arms"]["base"][m] for f in folds]
        base_std[m] = float(np.std(vals, ddof=1))

    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for m in ALL_METRICS:
            vals = [f["arms"][arm][m] for f in folds]
            if any(v is None for v in vals):
                summary[arm][m] = {"per_fold": vals}
                continue
            deltas = [round(f["arms"][arm][m] - f["arms"]["base"][m], 4) for f in folds]
            good = (lambda d: d < 0) if m in LOWER_IS_BETTER else (lambda d: d > 0)
            summary[arm][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4),
                "per_fold": vals,
                "delta_per_fold": deltas,
                "mean_delta": round(float(np.mean(deltas)), 4),
                "std_delta": round(float(np.std(deltas, ddof=1)), 4),
                "wins": int(sum(1 for d in deltas if good(d))),
                "ties": int(sum(1 for d in deltas if d == 0)),
            }
    return summary, base_std


def verdict(summary: dict, base_std: dict) -> dict:
    """Pre-registered bar: target metric (grade_acc) wins >=4/6 folds with
    mean delta beyond the fold std/2 (both the paired delta-std and the
    baseline fold-std are checked; the stricter must pass), and every other
    served metric non-inferior (mean delta in the bad direction no larger
    than the baseline fold std/2)."""
    out = {}
    for arm in ARMS:
        if arm == "base":
            continue
        t = summary[arm]["grade_acc"]
        bar = max(t["std_delta"], base_std["grade_acc"]) / 2.0
        target_ok = t["wins"] >= 4 and t["mean_delta"] > bar
        non_inf = {}
        for m in SERVED:
            if m == "grade_acc":
                continue
            s = summary[arm][m]
            if "mean_delta" not in s:
                non_inf[m] = None
                continue
            d = s["mean_delta"]
            bad = -d if m in LOWER_IS_BETTER else d
            non_inf[m] = bool(bad >= -base_std[m] / 2.0)
        ni_ok = all(v for v in non_inf.values() if v is not None)
        out[arm] = {
            "target_metric": "grade_acc",
            "target_wins": t["wins"], "target_mean_delta": t["mean_delta"],
            "target_bar_half_std": round(bar, 4),
            "target_passes": bool(target_ok),
            "non_inferior": non_inf,
            "non_inferior_everywhere": bool(ni_ok),
            "adopt": bool(target_ok and ni_ok),
        }
    return out


def main() -> int:
    np.random.seed(tm.SEED)
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    print(f"features ({len(tm.SUCCESS_FEATURES)}); folds {FOLDS}", flush=True)

    folds = []
    for y in FOLDS:
        t0 = time.time()
        f = run_fold(raw, seeds, y)
        folds.append(f)
        b = f["arms"]["base"]
        print(f"\nfold {y} ({time.time()-t0:.0f}s, fits {f['fit_seconds']}s) "
              f"idx_cuts {f['idx_cuts_cal']} pick_cuts {f['pick_cuts_fit']}", flush=True)
        print(f"  {'arm':<11}{'acc':>7}{'adj':>7}{'mF1':>7}{'MAE':>7}{'rho':>8}{'t64':>8}{'R1':>7}")
        for arm in ARMS:
            a = f["arms"][arm]
            print(f"  {arm:<11}{a['grade_acc']:>7.4f}{a['grade_adj_acc']:>7.4f}"
                  f"{a['grade_macro_f1']:>7.4f}{a['pick_blend_mae']:>7.1f}"
                  f"{a['pick_blend_spearman']:>8.4f}{a['pick_blend_top64']:>8.4f}"
                  f"{a['pick_blend_r1_recall_45']:>7.4f}", flush=True)

    summary, base_std = summarize(folds)
    gates = verdict(summary, base_std)

    out = {
        "experiment": "v6_ordinal (grade-head formulations, rolling-origin A/B)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "design": {
            "folds": FOLDS,
            "train": "2000..Y-2 + 63 seed rows (w=5)",
            "cal": "Y-1 (index cut-point tuning only)",
            "z_ref": "2000..Y-1 recomputed per fold",
            "test": "Y",
            "pick_cut_fit": "train+cal rows (2000..Y-1), no seeds",
            "success_head": "not refit — unchanged by any grade-head change",
            "class_midpoints": MIDS.tolist(),
            "task_cutoffs": list(TASK_CUTS),
        },
        "arms": {
            "base": "flat 4-class XGB+CB softmax, argmax (production)",
            "base_cb": "CatBoost MultiClass member alone",
            "a_fh": "Frank&Hall 3x binary P(y>k) XGB+CB, differenced, argmax",
            "b_idx": "XGB+CB regression on class index, round; pick via interp of midpoints",
            "b_idx_cal": "index regression with cut-points tuned on cal year",
            "c_reg50": "bucket(regressor pick) at 50/100/262; served pick = regressor only",
            "c_regfit": "bucket(regressor pick) at pre-test-fitted cuts; served pick = regressor only",
            "c_blend50": "bucket(served blend) at 50/100/262; pick metrics == base",
            "c_blendfit": "bucket(served blend) at pre-test-fitted cuts; pick metrics == base",
        },
        "baseline_fold_std": {m: round(v, 4) for m, v in base_std.items()},
        "folds": folds,
        "summary": summary,
        "gates": gates,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nWrote {OUT_PATH}\n")
    print(f"{'arm':<11}{'acc':>16}{'adj':>16}{'mF1':>16}{'MAE':>14}{'rho':>16}{'t64':>16}")
    for arm in ARMS:
        s = summary[arm]
        def cell(m, w=16, p=4):
            x = s[m]
            if "mean" not in x:
                return f"{'-':>{w}}"
            return f"{x['mean']:.{p}f}({x['mean_delta']:+.{p}f},{x['wins']}w)".rjust(w)
        print(f"{arm:<11}{cell('grade_acc')}{cell('grade_adj_acc')}{cell('grade_macro_f1')}"
              f"{cell('pick_blend_mae',14,1)}{cell('pick_blend_spearman')}{cell('pick_blend_top64')}")
    print("\nGATES:")
    for arm, g in gates.items():
        print(f"  {arm:<11} adopt={g['adopt']} target_pass={g['target_passes']} "
              f"(wins {g['target_wins']}, dAcc {g['target_mean_delta']:+.4f} vs bar {g['target_bar_half_std']}) "
              f"non_inf={g['non_inferior']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
