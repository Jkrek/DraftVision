#!/usr/bin/env python3
"""Experiment harness for the 4-class draft-grade model.

Replicates scripts/train_models.py's split + metric computation EXACTLY
(train 2010-2017 + 63 seed rows w=5, cal 2018, TEST 2019-2020) and layers on:
  A. extra combine measurables (raw or position-normalized)
  B. ordinal decompositions (cumulative-binary XGB; regression + thresholds)
  C. 2000-2009 history as TRAIN-ONLY rows (pooled or recency-weighted)

Writes NOTHING outside models/experiments/. Production artifacts untouched.

Run: .venv/bin/python scripts/experiments/harness.py
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from dv_features import (
    SUCCESS_FEATURES,
    SEED_TRAINING_PLAYERS,
    position_flags,
    _production_group,
    _raw_production_to_percentile,
)

V2_CSV = os.path.join(REPO_ROOT, "training_data", "experiments", "combine_outcomes_v2.csv")
OUT_JSON = os.path.join(REPO_ROOT, "models", "experiments", "results_runs.json")

SEED = 42
TRAIN_YEARS_BASE = set(range(2010, 2018))
HIST_YEARS = set(range(2000, 2010))
CAL_YEARS = {2018}
TEST_YEARS = {2019, 2020}
SEED_ROW_WEIGHT = 5.0

MEASURABLES = ["height_in", "weight_lb", "vertical", "bench", "broad_in", "cone", "shuttle"]
FORBIDDEN = {"draft_round", "pick", "experience", "experience_years", "pro_bowls",
             "probowls", "seasons_started", "w_av", "car_av", "career_av", "to",
             "nfl_success", "draft_grade", "draft_year", "name", "sample_weight"}


def _nan_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        if pd.isna(v):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float(v)


def load_rows() -> pd.DataFrame:
    """v2 CSV -> feature frame. Base 13 features built IDENTICALLY to
    scripts/train_models.load_csv_rows (incl. production percentile mapping);
    measurables/age carried through as NaN-preserving floats."""
    df = pd.read_csv(V2_CSV)
    rows = []
    for _, r in df.iterrows():
        pos = str(r.get("position") or "OTH").upper()
        raw_prod = _nan_float(r.get("production_score"))
        if math.isfinite(raw_prod):
            pct = _raw_production_to_percentile(_production_group(pos), raw_prod)
            prod = float(pct) if pct is not None else raw_prod
        else:
            prod = float("nan")
        row = {
            "name":                str(r.get("name") or ""),
            "draft_year":          int(r["draft_year"]),
            "sample_weight":       1.0,
            "production_score":    prod,
            "games_played":        _nan_float(r.get("games_college")),
            "combine_speed_score": _nan_float(r.get("combine_speed_score")),
            "conference_tier":     _nan_float(r.get("conference_tier")),
            **position_flags(pos),
            "nfl_success":         int(r["nfl_success"]),
            "draft_grade":         int(min(3, max(0, int(r["draft_grade"])))),
            "_pos_group":          _production_group(pos),
        }
        for m in MEASURABLES + ["age"]:
            row[m] = _nan_float(r.get(m))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_rows() -> pd.DataFrame:
    rows = []
    for sp in SEED_TRAINING_PLAYERS:
        row = {
            "name":                "(seed)",
            "draft_year":          -1,
            "sample_weight":       SEED_ROW_WEIGHT,
            "production_score":    float(sp["production_score"]),
            "games_played":        float(sp["games_played"]),
            "combine_speed_score": float(sp["combine_speed_score"]),
            "conference_tier":     float(sp["conference_tier"]),
            **position_flags(sp["position"]),
            "nfl_success":         int(sp["nfl_success"]),
            "draft_grade":         int(sp["draft_grade"]),
            "_pos_group":          _production_group(sp["position"]),
        }
        for m in MEASURABLES + ["age"]:
            row[m] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def add_posnorm(df: pd.DataFrame) -> pd.DataFrame:
    """Per-position-group z-scores for measurables, stats from NON-TEST years only."""
    ref = df[~df.draft_year.isin(TEST_YEARS)]
    for m in MEASURABLES:
        stats = ref.groupby("_pos_group")[m].agg(["mean", "std"])
        mu = df["_pos_group"].map(stats["mean"])
        sd = df["_pos_group"].map(stats["std"]).replace(0, np.nan)
        df[m + "_z"] = (df[m] - mu) / sd
    return df


# ── models ────────────────────────────────────────────────────────────────────

def class_weighted(w_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    cls_w = {}
    total_w = w_tr.sum()
    for c in range(4):
        cw = w_tr[y_tr == c].sum()
        cls_w[c] = float(total_w / (4.0 * max(cw, 1e-9)))
    return w_tr * np.array([cls_w[c] for c in y_tr])


def fit_flat(X_tr, y_tr, w_tr, feats):
    w_eff = class_weighted(w_tr, y_tr)
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
        objective="multi:softprob", num_class=4,
        random_state=SEED, eval_metric="mlogloss")
    xgb_m.fit(X_tr[feats], y_tr, sample_weight=w_eff)
    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="MultiClass", classes_count=4,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    cb_m.fit(X_tr[feats], y_tr, sample_weight=w_eff)
    return xgb_m, cb_m


def flat_probs(models, X, feats):
    xgb_m, cb_m = models
    return np.mean([xgb_m.predict_proba(X[feats]), cb_m.predict_proba(X[feats])], axis=0)


def fit_ordinal(X_tr, y_tr, w_tr, feats):
    """Three binary XGBs: P(y<=0), P(y<=1), P(y<=2)."""
    models = []
    for k in (0, 1, 2):
        yk = (y_tr <= k).astype(int)
        spw = float(w_tr[yk == 0].sum() / max(w_tr[yk == 1].sum(), 1e-9))
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
            scale_pos_weight=spw, random_state=SEED, eval_metric="logloss")
        m.fit(X_tr[feats], yk, sample_weight=w_tr)
        models.append(m)
    return models


def ordinal_probs(models, X, feats):
    cum = np.stack([m.predict_proba(X[feats])[:, 1] for m in models], axis=1)  # P<=0,1,2
    P = np.zeros((len(X), 4))
    P[:, 0] = cum[:, 0]
    P[:, 1] = cum[:, 1] - cum[:, 0]
    P[:, 2] = cum[:, 2] - cum[:, 1]
    P[:, 3] = 1.0 - cum[:, 2]
    P = np.clip(P, 0.0, None)
    P /= P.sum(axis=1, keepdims=True)
    return P


def fit_regression(X_tr, y_tr, w_tr, feats, X_cal, y_cal):
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
        random_state=SEED)
    m.fit(X_tr[feats], y_tr.astype(float), sample_weight=w_tr)
    # Tune thresholds on the calibration year (max accuracy, macro-F1 tiebreak).
    # correct(t0,t1,t2) = [A-B](t0) + [B-C](t1) + [C-D](t2) + n3, where
    # A/B/C/D(t) = #(y==c & s<=t) — separable, so evaluate on a 3-D broadcast.
    s_cal = m.predict(X_cal[feats])
    grid = np.arange(-0.5, 3.51, 0.05)
    le = s_cal[None, :] <= grid[:, None]                       # (G, n)
    cnt = [np.sum(le & (y_cal == c)[None, :], axis=1) for c in range(4)]
    f0, f1c, f2c = cnt[0] - cnt[1], cnt[1] - cnt[2], cnt[2] - cnt[3]
    total = (f0[:, None, None] + f1c[None, :, None] + f2c[None, None, :])
    i, j, k = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)),
                          np.arange(len(grid)), indexing="ij")
    total = np.where((i < j) & (j < k), total, -10**9)
    best_acc = total.max()
    cands = np.argwhere(total >= best_acc)
    best = (-1.0, (0.5, 1.5, 2.5))
    for ii, jj, kk in cands[:200]:
        th = (float(grid[ii]), float(grid[jj]), float(grid[kk]))
        pred = np.where(s_cal <= th[0], 0,
                        np.where(s_cal <= th[1], 1, np.where(s_cal <= th[2], 2, 3)))
        f1 = f1_score(y_cal, pred, average="macro")
        if f1 > best[0]:
            best = (f1, th)
    return m, best[1]


def regression_preds(model, thresholds, X, feats):
    s = model.predict(X[feats])
    t0, t1, t2 = thresholds
    return np.where(s <= t0, 0, np.where(s <= t1, 1, np.where(s <= t2, 2, 3)))


def calibrate_multinomial(P_cal, y_cal, P_te):
    logp = np.log(np.clip(P_cal, 1e-6, 1.0))
    mnl = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    mnl.fit(logp, y_cal)
    return mnl.predict_proba(np.log(np.clip(P_te, 1e-6, 1.0)))


def metrics(y, pred) -> dict:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class_recall": [round(float(r), 4) for r in rec],
        "confusion_matrix": cm.tolist(),
    }


# ── run driver ────────────────────────────────────────────────────────────────

def run_experiment(df, seeds, feats, method, hist, recency, label) -> dict:
    train_years = TRAIN_YEARS_BASE | (HIST_YEARS if hist else set())
    train_csv = df[df.draft_year.isin(train_years)].copy()
    if recency:
        train_csv["sample_weight"] = 0.97 ** (2020 - train_csv["draft_year"])
    train = pd.concat([train_csv, seeds], ignore_index=True)
    cal = df[df.draft_year.isin(CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(TEST_YEARS)].reset_index(drop=True)

    # ── E. sanity: no leakage ────────────────────────────────────────────────
    assert not (set(feats) & FORBIDDEN), f"forbidden feature in X: {set(feats) & FORBIDDEN}"
    assert not (set(train.draft_year.unique()) & TEST_YEARS), "test year leaked into train"
    assert not (set(cal.draft_year.unique()) & TEST_YEARS), "test year leaked into cal"
    assert set(test.draft_year.unique()) == TEST_YEARS

    y_tr = train.draft_grade.to_numpy()
    w_tr = train.sample_weight.to_numpy()
    y_cal = cal.draft_grade.to_numpy()
    y_te = test.draft_grade.to_numpy()

    out = {"label": label, "features": list(feats), "method": method,
           "history": hist, "recency": recency,
           "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test))}

    if method == "flat":
        models = fit_flat(train, y_tr, w_tr, feats)
        P_te = flat_probs(models, test, feats)
        out["raw"] = metrics(y_te, P_te.argmax(axis=1))
        P_cal = flat_probs(models, cal, feats)
        out["calibrated"] = metrics(y_te, calibrate_multinomial(P_cal, y_cal, P_te).argmax(axis=1))
    elif method == "ordinal":
        models = fit_ordinal(train, y_tr, w_tr, feats)
        P_te = ordinal_probs(models, test, feats)
        out["raw"] = metrics(y_te, P_te.argmax(axis=1))
    elif method == "regression":
        model, th = fit_regression(train, y_tr, w_tr, feats, cal, y_cal)
        out["thresholds"] = list(th)
        out["raw"] = metrics(y_te, regression_preds(model, th, test, feats))
    else:
        raise ValueError(method)

    m = out["raw"]
    print(f"  {label:<44} acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
          f"recall {m['per_class_recall']}"
          + (f"  | cal acc {out['calibrated']['accuracy']:.4f}" if "calibrated" in out else ""))
    return out


def main() -> int:
    np.random.seed(SEED)
    print(f"Loading {V2_CSV}")
    df = load_rows()
    df = add_posnorm(df)
    seeds = seed_rows()
    for m in MEASURABLES:
        seeds[m + "_z"] = float("nan")

    base = list(SUCCESS_FEATURES)
    raw_meas = base + MEASURABLES
    norm_meas = base + [m + "_z" for m in MEASURABLES]
    raw_meas_age = raw_meas + ["age"]

    feature_sets = {
        "base13": base,
        "+meas_raw": raw_meas,
        "+meas_norm": norm_meas,
        "+meas_raw+age": raw_meas_age,
    }

    runs = []
    print("\n== Baseline reproduction + experiment grid (test = 2019-2020, n=733) ==")
    grid = []
    # A/B on 2010-2017 train
    for fname in feature_sets:
        for method in ("flat", "ordinal", "regression"):
            grid.append((fname, method, False, False))
    # C: history for every feature set / method, pooled and recency
    for fname in feature_sets:
        for method in ("flat", "ordinal", "regression"):
            grid.append((fname, method, True, False))
            grid.append((fname, method, True, True))

    for fname, method, hist, recency in grid:
        label = f"{fname}|{method}|{'hist' + ('-w' if recency else '') if hist else '2010+'}"
        runs.append(run_experiment(df, seeds, feature_sets[fname], method, hist, recency, label))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(runs, fh, indent=2)
    print(f"\nWrote {OUT_JSON} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
