#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 experiment: learn the served pick-blend weight instead of the fixed 0.5.

The served pick estimator is
    blend = exp( w * log(regressor) + (1 - w) * log(classifier_expected) )
with w hard-coded to 0.5 (train_models.py, rolling_cv.py). This runs the
rolling-origin folds of scripts/rolling_ab_v5.py (test year Y in 2015..2020,
train 2000..Y-2 + seeds, cal Y-1, z-ref 2000..Y-1), fits the grade head and
the pick regressor exactly as the EVAL phase does, then — WITHOUT refitting —
evaluates w in {0.0, 0.1, ..., 1.0} on the fold's CAL year and reports the
cal-chosen w's TEST-year metrics against the fixed 0.5.

Variants (all selected on cal only, never on test):
  fixed_050          : the served estimator (baseline arm)
  cal_w_mae          : single w minimising cal MAE (drafted rows)
  cal_w_spearman     : single w maximising cal Spearman (all rows)
  cal_bucket_w_mae   : (w_top64, w_rest) — bucket = fixed-0.5 blend <= 64,
                       each w minimising cal MAE within its bucket
  oracle_*           : best w on TEST — NOT ADOPTABLE, headroom bound only
                       (per-fold oracle and a single pooled-over-folds w)

Leakage guarantees (asserted per fold):
  * X = tm.SUCCESS_FEATURES (31 cols); intersection with FORBIDDEN_FEATURES
    is asserted empty.
  * train rows are draft_year in 2000..Y-2 (+ seed rows, draft_year=-1);
    cal is {Y-1}; test is {Y}; test year asserted absent from train, cal and
    the z-score reference set 2000..Y-1.
  * w is chosen on cal metrics only; test metrics are read once per w for
    reporting / the clearly-labelled oracle.

Writes models/experiments/v6_blend_results.json.

Usage:  .venv/bin/python scripts/experiment_v6_blend.py
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
from sklearn.metrics import accuracy_score

import train_models as tm

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_blend_results.json")
FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]
W_GRID = [round(w, 1) for w in np.arange(0.0, 1.0 + 1e-9, 0.1)]
FIXED_W = 0.5
TOP64 = 64.0

PICK_METRICS = ["pick_blend_mae", "pick_blend_spearman",
                "pick_blend_top64", "pick_blend_r1_recall_45"]
LOWER_IS_BETTER = {"pick_blend_mae"}
TARGET_METRIC = "pick_blend_mae"


# ── helpers ──────────────────────────────────────────────────────────────────

def blend_pick(reg: np.ndarray, cls: np.ndarray, w) -> np.ndarray:
    """Served estimator with a free weight. w may be a scalar or per-row array."""
    return np.exp(w * np.log(reg) + (1.0 - w) * np.log(np.maximum(cls, 1.0)))


def score(y_pick: np.ndarray, pred: np.ndarray) -> dict:
    pk = tm.pick_metrics(y_pick, pred)
    return {
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
    }


def _better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def pick_best_w(grid_scores: dict, metric: str) -> float:
    """Best w on `metric`; ties broken toward the served 0.5."""
    best_w, best_v = None, None
    for w in W_GRID:
        v = grid_scores[w][metric]
        if v is None:
            continue
        if best_v is None or _better(metric, v, best_v) or (
                v == best_v and abs(w - FIXED_W) < abs(best_w - FIXED_W)):
            best_w, best_v = w, v
    return best_w


def bucket_best_w(y: np.ndarray, reg: np.ndarray, cls: np.ndarray,
                  in_bucket: np.ndarray) -> float:
    """w minimising drafted-row MAE inside one bucket (MAE is separable across
    buckets, so per-bucket argmin == joint argmin of overall MAE)."""
    drafted = (y < tm.UDFA_PICK) & in_bucket
    if drafted.sum() == 0:
        return FIXED_W
    best_w, best_v = FIXED_W, None
    for w in W_GRID:
        v = float(np.mean(np.abs(y[drafted] - blend_pick(reg[drafted], cls[drafted], w))))
        if best_v is None or v < best_v - 1e-12 or (
                abs(v - best_v) <= 1e-12 and abs(w - FIXED_W) < abs(best_w - FIXED_W)):
            best_w, best_v = w, v
    return best_w


def _assert_fold_no_leakage(features, train, cal, test, ref_years, test_years) -> None:
    forbidden = set(features) & tm.FORBIDDEN_FEATURES
    assert not forbidden, f"forbidden feature in X: {forbidden}"
    assert not (set(train.draft_year.unique()) & test_years), "test year in train"
    assert not (set(cal.draft_year.unique()) & test_years), "test year in cal"
    assert not (set(cal.draft_year.unique()) & set(train.draft_year.unique())), "cal in train"
    assert set(test.draft_year.unique()) == test_years, "unexpected test years"
    assert not (ref_years & test_years), "test year in z-score reference"


# ── one fold ─────────────────────────────────────────────────────────────────

def run_fold(raw: pd.DataFrame, seeds: pd.DataFrame, year: int) -> dict:
    np.random.seed(tm.SEED)
    features = list(tm.SUCCESS_FEATURES)
    tm.EVAL_TRAIN_YEARS = set(range(2000, year - 1))
    tm.CAL_YEARS = {year - 1}
    tm.TEST_YEARS = {year}
    ref_years = set(range(2000, year))

    df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))
    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    _assert_fold_no_leakage(features, train, cal, test, ref_years, tm.TEST_YEARS)

    t0 = time.time()
    g = tm.fit_grade_members(train)   # blend weight does not touch the grade head
    p = tm.fit_pick_members(train)    # ... nor the regressor: fits are shared by all arms
    fit_s = time.time() - t0

    def parents(frame):
        X = frame[features]
        y = np.exp(tm._pick_target(frame))
        m = np.isfinite(y)
        P_raw = tm.ensemble_grade_probs(g, X, calibrated=False)   # served label path
        reg = tm.ensemble_pick_preds(p, X)
        cls = tm.classifier_expected_pick(P_raw)
        return y[m], reg[m], cls[m], P_raw, m

    y_c, reg_c, cls_c, _, _ = parents(cal)
    y_t, reg_t, cls_t, P_t, _ = parents(test)
    grade_acc = round(float(accuracy_score(test.draft_grade.to_numpy(), P_t.argmax(axis=1))), 4)

    # Full grid on cal and on test (test grid is for reporting + oracle only).
    cal_grid = {w: score(y_c, blend_pick(reg_c, cls_c, w)) for w in W_GRID}
    test_grid = {w: score(y_t, blend_pick(reg_t, cls_t, w)) for w in W_GRID}

    # Bucketed variant: bucket by the SERVED (fixed 0.5) blend so the bucket
    # is defined before any weight is chosen — same rule applied on cal & test.
    served_c = blend_pick(reg_c, cls_c, FIXED_W)
    served_t = blend_pick(reg_t, cls_t, FIXED_W)
    top_c, top_t = served_c <= TOP64, served_t <= TOP64
    w_top = bucket_best_w(y_c, reg_c, cls_c, top_c)
    w_rest = bucket_best_w(y_c, reg_c, cls_c, ~top_c)
    w_vec_t = np.where(top_t, w_top, w_rest)
    bucket_test = score(y_t, blend_pick(reg_t, cls_t, w_vec_t))
    w_vec_c = np.where(top_c, w_top, w_rest)
    bucket_cal = score(y_c, blend_pick(reg_c, cls_c, w_vec_c))

    w_mae = pick_best_w(cal_grid, "pick_blend_mae")
    w_rho = pick_best_w(cal_grid, "pick_blend_spearman")
    w_oracle_mae = pick_best_w(test_grid, "pick_blend_mae")
    w_oracle_rho = pick_best_w(test_grid, "pick_blend_spearman")

    arms = {
        "fixed_050":        {"w": FIXED_W, "test": test_grid[FIXED_W], "cal": cal_grid[FIXED_W]},
        "cal_w_mae":        {"w": w_mae, "test": test_grid[w_mae], "cal": cal_grid[w_mae]},
        "cal_w_spearman":   {"w": w_rho, "test": test_grid[w_rho], "cal": cal_grid[w_rho]},
        "cal_bucket_w_mae": {"w": {"top64": w_top, "rest": w_rest},
                             "n_cal_top64": int(top_c.sum()), "n_test_top64": int(top_t.sum()),
                             "test": bucket_test, "cal": bucket_cal},
        "oracle_test_w_mae_NOT_ADOPTABLE":      {"w": w_oracle_mae, "test": test_grid[w_oracle_mae]},
        "oracle_test_w_spearman_NOT_ADOPTABLE": {"w": w_oracle_rho, "test": test_grid[w_oracle_rho]},
    }
    base = arms["fixed_050"]["test"]
    for name, a in arms.items():
        a["delta_vs_fixed"] = {
            m: (round(a["test"][m] - base[m], 4)
                if a["test"][m] is not None and base[m] is not None else None)
            for m in PICK_METRICS}

    return {
        "test_year": year,
        "train_years": [2000, year - 2], "cal_year": year - 1,
        "ref_years": [2000, year - 1],
        "n_train": int(len(train)), "n_cal_pick": int(len(y_c)), "n_test_pick": int(len(y_t)),
        "fit_seconds": round(fit_s, 1),
        "grade_acc_raw": grade_acc,
        "parents_test": {"regressor_only": test_grid[1.0], "classifier_only": test_grid[0.0]},
        "cal_grid": {str(w): cal_grid[w] for w in W_GRID},
        "test_grid": {str(w): test_grid[w] for w in W_GRID},
        "arms": arms,
    }


# ── summary / verdict ────────────────────────────────────────────────────────

def summarize(folds: list, arm: str) -> dict:
    out = {}
    for m in PICK_METRICS:
        ds = [f["arms"][arm]["delta_vs_fixed"][m] for f in folds]
        ds = [d for d in ds if d is not None]
        base_vals = [f["arms"]["fixed_050"]["test"][m] for f in folds]
        base_vals = [v for v in base_vals if v is not None]
        arm_vals = [f["arms"][arm]["test"][m] for f in folds]
        arm_vals = [v for v in arm_vals if v is not None]
        out[m] = {
            "mean_delta": round(float(np.mean(ds)), 4),
            "std_delta": round(float(np.std(ds, ddof=1)), 4) if len(ds) > 1 else None,
            "baseline_fold_std": round(float(np.std(base_vals, ddof=1)), 4),
            "baseline_mean": round(float(np.mean(base_vals)), 4),
            "arm_mean": round(float(np.mean(arm_vals)), 4),
            "wins": sum(1 for d in ds if (d < 0 if m in LOWER_IS_BETTER else d > 0)),
            "ties": sum(1 for d in ds if d == 0),
            "losses": sum(1 for d in ds if (d > 0 if m in LOWER_IS_BETTER else d < 0)),
            "per_fold_delta": ds,
        }
    return out


def verdict(summary: dict, n_folds: int) -> dict:
    """Pre-registered bar: >=4/6 wins on the target with |mean delta| beyond
    baseline fold std/2, and non-inferior (mean delta not worse by more than
    fold std/2) on every other served pick metric."""
    t = summary[TARGET_METRIC]
    sign = -1.0 if TARGET_METRIC in LOWER_IS_BETTER else 1.0
    target_gain = sign * t["mean_delta"]
    target_ok = t["wins"] >= 4 and target_gain > t["baseline_fold_std"] / 2.0
    others = {}
    for m in PICK_METRICS:
        if m == TARGET_METRIC:
            continue
        s = summary[m]
        sgn = -1.0 if m in LOWER_IS_BETTER else 1.0
        gain = sgn * s["mean_delta"]
        others[m] = {"mean_gain": round(gain, 4),
                     "tolerance": round(s["baseline_fold_std"] / 2.0, 4),
                     "non_inferior": bool(gain >= -s["baseline_fold_std"] / 2.0)}
    adopt = bool(target_ok and all(o["non_inferior"] for o in others.values()))
    return {"target_metric": TARGET_METRIC,
            "target_wins": f"{t['wins']}/{n_folds}",
            "target_mean_gain": round(target_gain, 4),
            "target_threshold_half_fold_std": round(t["baseline_fold_std"] / 2.0, 4),
            "target_passes": bool(target_ok),
            "other_metrics": others,
            "adopt": adopt}


def main() -> int:
    t_start = time.time()
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    print(f"features: {len(tm.SUCCESS_FEATURES)}  w grid: {W_GRID}", flush=True)

    folds = []
    for y in FOLDS:
        f = run_fold(raw, seeds, y)
        folds.append(f)
        a = f["arms"]
        b = a["fixed_050"]["test"]
        print(f"fold {y} (fit {f['fit_seconds']}s): fixed MAE {b['pick_blend_mae']} rho {b['pick_blend_spearman']} | "
              f"cal_w_mae w={a['cal_w_mae']['w']} dMAE {a['cal_w_mae']['delta_vs_fixed']['pick_blend_mae']:+.1f} "
              f"dRho {a['cal_w_mae']['delta_vs_fixed']['pick_blend_spearman']:+.4f} | "
              f"bucket w={a['cal_bucket_w_mae']['w']} dMAE {a['cal_bucket_w_mae']['delta_vs_fixed']['pick_blend_mae']:+.1f} | "
              f"oracle w={a['oracle_test_w_mae_NOT_ADOPTABLE']['w']} "
              f"dMAE {a['oracle_test_w_mae_NOT_ADOPTABLE']['delta_vs_fixed']['pick_blend_mae']:+.1f}",
              flush=True)

    arm_names = [k for k in folds[0]["arms"] if k != "fixed_050"]
    summary = {arm: summarize(folds, arm) for arm in arm_names}
    verdicts = {arm: verdict(summary[arm], len(folds)) for arm in arm_names
                if not arm.startswith("oracle")}

    # Pooled oracle: ONE w for all folds, chosen on pooled TEST metrics —
    # bounds the headroom of a fixed constant; not adoptable by construction.
    pooled = {}
    for w in W_GRID:
        pooled[str(w)] = {m: round(float(np.mean([f["test_grid"][str(w)][m] for f in folds
                                                  if f["test_grid"][str(w)][m] is not None])), 4)
                          for m in PICK_METRICS}
    pooled_best_mae = min(W_GRID, key=lambda w: (pooled[str(w)]["pick_blend_mae"], abs(w - FIXED_W)))
    pooled_best_rho = max(W_GRID, key=lambda w: (pooled[str(w)]["pick_blend_spearman"], -abs(w - FIXED_W)))
    pooled_cal = {}
    for w in W_GRID:
        pooled_cal[str(w)] = {m: round(float(np.mean([f["cal_grid"][str(w)][m] for f in folds
                                                      if f["cal_grid"][str(w)][m] is not None])), 4)
                              for m in PICK_METRICS}
    pooled_cal_best_mae = min(W_GRID, key=lambda w: (pooled_cal[str(w)]["pick_blend_mae"], abs(w - FIXED_W)))

    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "seed": tm.SEED,
        "n_features": len(tm.SUCCESS_FEATURES),
        "design": {
            "folds": FOLDS,
            "train": "2000..Y-2 draft classes + 63 seed rows (weight 5)",
            "cal": "Y-1 — the ONLY data used to choose w",
            "z_ref": "2000..Y-1 recomputed per fold (never includes test year)",
            "test": "Y only",
            "w_grid": W_GRID,
            "blend": "exp(w*log(regressor) + (1-w)*log(classifier_expected_pick))",
            "bucket_rule": "top64 bucket = served fixed-0.5 blend <= 64 (defined before w)",
            "selection_objective": {"cal_w_mae": "cal MAE (drafted rows)",
                                    "cal_w_spearman": "cal Spearman (all rows)",
                                    "cal_bucket_w_mae": "cal MAE within bucket"},
            "adoption_bar": "target wins >=4/6 AND mean gain > baseline fold std/2; "
                            "non-inferior (mean loss <= fold std/2) on the other pick metrics",
            "note": "success head untouched by construction (blend weight only affects the pick "
                    "estimator); grade head refit identically — grade_acc_raw reported per fold as a check",
        },
        "folds": folds,
        "summary": summary,
        "verdicts": verdicts,
        "pooled_test_grid_mean_over_folds": pooled,
        "pooled_cal_grid_mean_over_folds": pooled_cal,
        "pooled_oracle_NOT_ADOPTABLE": {
            "best_w_mae": pooled_best_mae, "metrics": pooled[str(pooled_best_mae)],
            "best_w_spearman": pooled_best_rho, "metrics_spearman_w": pooled[str(pooled_best_rho)],
            "fixed_050": pooled[str(FIXED_W)],
        },
        "pooled_cal_best_w_mae": pooled_cal_best_mae,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}  ({out['runtime_seconds']}s)")
    for arm in arm_names:
        print(f"\n== {arm} ==")
        for m in PICK_METRICS:
            s = summary[arm][m]
            print(f"  {m:>26}: mean d {s['mean_delta']:+.4f}  std d {s['std_delta']}  "
                  f"wins {s['wins']}/{len(folds)}  base fold std {s['baseline_fold_std']}")
        if arm in verdicts:
            print(f"  ADOPT: {verdicts[arm]['adopt']}  ({verdicts[arm]['target_wins']} wins, "
                  f"gain {verdicts[arm]['target_mean_gain']} vs thr {verdicts[arm]['target_threshold_half_fold_std']})")
    print("\npooled test grid (mean over folds):")
    for w in W_GRID:
        print(f"  w={w}: {pooled[str(w)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
