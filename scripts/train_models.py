#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train + evaluate the DraftVision success and draft-grade ensembles.

This is the ONLY place models are trained. The Flask app (XGBOost.py) is
load-only and will refuse to start if the artifacts written here are missing
or feature-mismatched (unless DV_ALLOW_MISSING_MODELS=1).

Data
----
training_data/combine_outcomes.csv is the only bulk source — real nflverse
draft picks (2010-2020) + undrafted combine invitees, with college production
from cfbfastR play-by-play where computable. NO synthetic rows are generated
anywhere. The 63 hand-curated SEED_TRAINING_PLAYERS (dv_features.py) are
appended ONCE each with sample_weight=5 (never duplicated); all CSV rows get
sample_weight=1.

Splits (by draft_year GROUPS, never random rows)
------------------------------------------------
  train:       2010-2017  (+ the 63 seed rows, which have no draft year and
                           carry curated labels — they never enter cal/test)
  calibration: 2018
  test:        2019-2020  (held out, touched only for the final metrics)

Why temporal groups: a player appears in exactly one draft class, so grouping
by year guarantees no player is in two sets, and evaluating on later years
mirrors the real task — predicting outcomes for classes the model has never
seen. Random row splits would leak era effects and same-class information.

Calibration (plan item 2.4)
---------------------------
The served quantity is the MEAN of the XGBoost and CatBoost probabilities, so
that is what gets calibrated: both members are fit on the train fold, their
mean probability on the 2018 calibration fold is computed, and the calibrator
is fit on that mean. Artifacts (existing filenames kept):
  success_xgboost_model.json / catboost_success_model.cbm      — members
  draft_grade_model.json     / catboost_draft_grade_model.cbm  — members
  success_calibrated_model.pkl     — dict {"kind": "platt_logit",  "model": LogisticRegression, ...}
  draft_grade_calibrated_model.pkl — dict {"kind": "multinomial_logprob", "model": LogisticRegression, ...}
The .pkl files now store the ENSEMBLE CALIBRATOR (not a CalibratedClassifierCV
wrapper); XGBOost.py's _apply_binary_calibrator/_apply_multiclass_calibrator
apply the identical transform at serving time.

Evaluation (plan item 2.2)
--------------------------
On the 2019-2020 holdout: AUC / Brier / log-loss / 10-bin reliability table
for success; accuracy / macro-F1 / confusion matrix for draft grade — for the
trained ensemble AND for the rule-based baseline (dv_heuristics.py, the same
functions the app uses as its emergency fallback). Everything is printed and
written to models/metadata.json along with provenance (git SHA, data SHA256,
row counts, class balance, feature list, seed, timestamp, library versions).

Usage
-----
  .venv/bin/python scripts/train_models.py            # full train + eval
  .venv/bin/python scripts/train_models.py --dry-run  # load data, build splits,
                                                      # print shapes/balances, exit
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
import catboost as catboost_pkg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    roc_auc_score,
)

from dv_features import (
    SUCCESS_FEATURES,
    DRAFT_GRADE_LABELS,
    SEED_TRAINING_PLAYERS,
    position_flags,
    _production_group,
    _raw_production_to_percentile,
)
from dv_heuristics import draft_grade_from_profile, success_prob_from_college_profile

TRAINING_DATA_PATH = os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv")
METADATA_PATH      = os.path.join(REPO_ROOT, "models", "metadata.json")

# Artifact filenames (repo root — identical to what XGBOost.py loads)
SUCCESS_MODEL_PATH          = os.path.join(REPO_ROOT, "success_xgboost_model.json")
CATBOOST_SUCCESS_PATH       = os.path.join(REPO_ROOT, "catboost_success_model.cbm")
SUCCESS_CALIBRATED_PATH     = os.path.join(REPO_ROOT, "success_calibrated_model.pkl")
DRAFT_GRADE_MODEL_PATH      = os.path.join(REPO_ROOT, "draft_grade_model.json")
CATBOOST_DRAFT_GRADE_PATH   = os.path.join(REPO_ROOT, "catboost_draft_grade_model.cbm")
DRAFT_GRADE_CALIBRATED_PATH = os.path.join(REPO_ROOT, "draft_grade_calibrated_model.pkl")

SEED = 42
TRAIN_YEARS = set(range(2010, 2018))   # 2010-2017
CAL_YEARS   = {2018}
TEST_YEARS  = {2019, 2020}
SEED_ROW_WEIGHT = 5.0

# Baseline-only neutral values: the heuristic is arithmetic and cannot take
# NaN, so unknown production/speed become the scale midpoint 50. This is a
# documented baseline convention, NOT a model imputation — the ML models see
# true NaN.
BASELINE_NEUTRAL = 50.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _nan_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        if pd.isna(v):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float(v)


def load_csv_rows() -> pd.DataFrame:
    """Load combine_outcomes.csv into the 13-feature frame + labels + year.

    The CSV's production_score is the RAW per-position composite (built by
    scripts/build_training_data.py with the pre-percentile formula). Serving
    (dv_features/compute_production_score) maps raw composites to empirical
    position-group percentiles, so we apply the identical mapping here —
    training and serving must see the same feature scale. NaN stays NaN.
    """
    df = pd.read_csv(TRAINING_DATA_PATH)
    rows = []
    for _, r in df.iterrows():
        pos = str(r.get("position") or "OTH").upper()
        raw_prod = _nan_float(r.get("production_score"))
        if math.isfinite(raw_prod):
            pct = _raw_production_to_percentile(_production_group(pos), raw_prod)
            prod = float(pct) if pct is not None else raw_prod
        else:
            prod = float("nan")
        rows.append({
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
        })
    return pd.DataFrame(rows)


def seed_rows() -> pd.DataFrame:
    """The 63 hand-curated seed players — added ONCE each, sample_weight=5.

    Seed production_score values were curated on the 0-100 percentile-like
    scale (elite ≈ 85-90), matching the served feature. Accolade fields are
    dropped — they are not model features.
    """
    rows = []
    for sp in SEED_TRAINING_PLAYERS:
        rows.append({
            "name":                "(seed)",
            "draft_year":          -1,  # curated exemplars — pinned to the train fold
            "sample_weight":       SEED_ROW_WEIGHT,
            "production_score":    float(sp["production_score"]),
            "games_played":        float(sp["games_played"]),
            "combine_speed_score": float(sp["combine_speed_score"]),
            "conference_tier":     float(sp["conference_tier"]),
            **position_flags(sp["position"]),
            "nfl_success":         int(sp["nfl_success"]),
            "draft_grade":         int(sp["draft_grade"]),
        })
    return pd.DataFrame(rows)


def build_splits():
    """Temporal, grouped-by-draft-year splits (see module docstring)."""
    csv_df = load_csv_rows()
    seeds  = seed_rows()

    train = pd.concat(
        [csv_df[csv_df.draft_year.isin(TRAIN_YEARS)], seeds],
        ignore_index=True)
    cal  = csv_df[csv_df.draft_year.isin(CAL_YEARS)].reset_index(drop=True)
    test = csv_df[csv_df.draft_year.isin(TEST_YEARS)].reset_index(drop=True)

    covered = TRAIN_YEARS | CAL_YEARS | TEST_YEARS
    dropped = sorted(set(csv_df.draft_year.unique()) - covered)
    if dropped:
        print(f"WARNING: CSV contains years outside all splits (unused): {dropped}")
    return train, cal, test


def frame_stats(name: str, df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "weighted_rows": float(df.sample_weight.sum()),
        "success_positive": int(df.nfl_success.sum()),
        "success_rate": round(float(df.nfl_success.mean()), 4),
        "draft_grade_counts": {int(k): int(v) for k, v in df.draft_grade.value_counts().sort_index().items()},
        "feature_missing_frac": {
            f: round(float(df[f].isna().mean()), 4)
            for f in ("production_score", "games_played", "combine_speed_score", "conference_tier")
        },
    }


# ── Baseline (rule-based) predictions ─────────────────────────────────────────

def baseline_success_probs(df: pd.DataFrame) -> np.ndarray:
    """dv_heuristics success probability per row, NaN→neutral 50, clipped."""
    probs = []
    for _, r in df.iterrows():
        prod  = r.production_score if math.isfinite(r.production_score) else BASELINE_NEUTRAL
        speed = r.combine_speed_score if math.isfinite(r.combine_speed_score) else BASELINE_NEUTRAL
        tier  = r.conference_tier if math.isfinite(r.conference_tier) else 10.0
        p = success_prob_from_college_profile(prod, tier, speed, 0, 0)
        probs.append(min(max(p, 0.01), 0.99))
    return np.asarray(probs)


def baseline_grade_preds(df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, r in df.iterrows():
        prod  = r.production_score if math.isfinite(r.production_score) else BASELINE_NEUTRAL
        speed = r.combine_speed_score if math.isfinite(r.combine_speed_score) else BASELINE_NEUTRAL
        tier  = r.conference_tier if math.isfinite(r.conference_tier) else 10.0
        preds.append(draft_grade_from_profile(prod, tier, speed, 0, 0))
    return np.asarray(preds)


# ── Metrics ───────────────────────────────────────────────────────────────────

def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc":       round(float(roc_auc_score(y, p)), 4),
        "brier":     round(float(brier_score_loss(y, p)), 4),
        "log_loss":  round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list:
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        table.append({
            "bin": f"[{lo:.1f},{hi:.1f})" if i < bins - 1 else f"[{lo:.1f},{hi:.1f}]",
            "count": n,
            "mean_predicted": round(float(p[mask].mean()), 4) if n else None,
            "fraction_positive": round(float(y[mask].mean()), 4) if n else None,
        })
    return table


def grade_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        "confusion_matrix_labels": DRAFT_GRADE_LABELS,
    }


def print_reliability(table: list) -> None:
    print(f"    {'bin':<12}{'count':>7}{'mean_pred':>11}{'frac_pos':>10}")
    for row in table:
        mp = f"{row['mean_predicted']:.3f}" if row["mean_predicted"] is not None else "-"
        fp = f"{row['fraction_positive']:.3f}" if row["fraction_positive"] is not None else "-"
        print(f"    {row['bin']:<12}{row['count']:>7}{mp:>11}{fp:>10}")


# ── Training ──────────────────────────────────────────────────────────────────

def _xy(df: pd.DataFrame, target: str):
    return df[SUCCESS_FEATURES], df[target].to_numpy(), df.sample_weight.to_numpy()


def train_success(train: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """Train XGB + CatBoost members, then the ensemble Platt calibrator."""
    X_tr, y_tr, w_tr = _xy(train, "nfl_success")
    X_cal, y_cal, _ = _xy(cal, "nfl_success")

    # Class imbalance from the TRAIN fold only (weighted counts)
    spw = float(w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-9))
    print(f"  scale_pos_weight (weighted, train fold): {spw:.3f}")

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
        scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    xgb_m.save_model(SUCCESS_MODEL_PATH)

    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: spw},
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    cb_m.save_model(CATBOOST_SUCCESS_PATH)

    # Ensemble output FIRST (mean of member probabilities on the cal fold),
    # THEN fit the calibrator on that mean — what is calibrated is what is served.
    p_cal = np.mean([
        xgb_m.predict_proba(X_cal)[:, 1],
        cb_m.predict_proba(X_cal)[:, 1],
    ], axis=0)
    z = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cal, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(z.reshape(-1, 1), y_cal)
    calibrator = {
        "kind": "platt_logit",
        "model": platt,
        "members": ["xgboost", "catboost"],
        "feature_names": list(SUCCESS_FEATURES),
        "calibration_years": sorted(CAL_YEARS),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    joblib.dump(calibrator, SUCCESS_CALIBRATED_PATH)
    print(f"  members → {SUCCESS_MODEL_PATH}, {CATBOOST_SUCCESS_PATH}")
    print(f"  ensemble calibrator (Platt on logit of member mean) → {SUCCESS_CALIBRATED_PATH}")
    return {"xgb": xgb_m, "cb": cb_m, "calibrator": calibrator,
            "scale_pos_weight": round(spw, 4)}


def train_draft_grade(train: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """Train the 4-class members, then the multinomial ensemble calibrator."""
    X_tr, y_tr, w_tr = _xy(train, "draft_grade")
    X_cal, y_cal, _ = _xy(cal, "draft_grade")

    # Balanced class weights from the TRAIN fold (weighted counts), folded
    # into the sample weights so XGB and CatBoost see identical weighting.
    cls_w = {}
    total_w = w_tr.sum()
    for c in range(4):
        cw = w_tr[y_tr == c].sum()
        cls_w[c] = float(total_w / (4.0 * max(cw, 1e-9)))
    print(f"  class weights (weighted, train fold): "
          + ", ".join(f"{c}:{cls_w[c]:.3f}" for c in range(4)))
    w_eff = w_tr * np.array([cls_w[c] for c in y_tr])

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
        objective="multi:softprob", num_class=4,
        random_state=SEED, eval_metric="mlogloss",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_eff)
    xgb_m.save_model(DRAFT_GRADE_MODEL_PATH)

    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="MultiClass", classes_count=4,
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_eff)
    cb_m.save_model(CATBOOST_DRAFT_GRADE_PATH)

    P_cal = np.mean([
        xgb_m.predict_proba(X_cal),
        cb_m.predict_proba(X_cal),
    ], axis=0)
    logp = np.log(np.clip(P_cal, 1e-6, 1.0))
    mnl = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    mnl.fit(logp, y_cal)
    calibrator = {
        "kind": "multinomial_logprob",
        "model": mnl,
        "members": ["xgboost", "catboost"],
        "feature_names": list(SUCCESS_FEATURES),
        "calibration_years": sorted(CAL_YEARS),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    joblib.dump(calibrator, DRAFT_GRADE_CALIBRATED_PATH)
    print(f"  members → {DRAFT_GRADE_MODEL_PATH}, {CATBOOST_DRAFT_GRADE_PATH}")
    print(f"  ensemble calibrator (multinomial on log member-mean probs) → {DRAFT_GRADE_CALIBRATED_PATH}")
    return {"xgb": xgb_m, "cb": cb_m, "calibrator": calibrator,
            "class_weights": {c: round(v, 4) for c, v in cls_w.items()}}


# ── Serving-equivalent inference (must mirror XGBOost.py) ─────────────────────

def ensemble_success_probs(bundle: dict, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    p = np.mean([
        bundle["xgb"].predict_proba(X)[:, 1],
        bundle["cb"].predict_proba(X)[:, 1],
    ], axis=0)
    if not calibrated:
        return p
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc))
    return bundle["calibrator"]["model"].predict_proba(z.reshape(-1, 1))[:, 1]


def ensemble_grade_probs(bundle: dict, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    P = np.mean([
        bundle["xgb"].predict_proba(X),
        bundle["cb"].predict_proba(X),
    ], axis=0)
    if not calibrated:
        return P
    logp = np.log(np.clip(P, 1e-6, 1.0))
    return bundle["calibrator"]["model"].predict_proba(logp)


# ── Provenance ────────────────────────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="load data, build splits, print shapes/balances, exit before fitting")
    args = ap.parse_args()

    np.random.seed(SEED)

    print(f"Loading {TRAINING_DATA_PATH}")
    train, cal, test = build_splits()
    split_stats = {
        "train_2010_2017_plus_seeds": frame_stats("train", train),
        "calibration_2018":           frame_stats("cal", cal),
        "test_2019_2020":             frame_stats("test", test),
    }

    print("\nSplit summary (temporal groups by draft_year — no player in two sets):")
    for name, st in split_stats.items():
        print(f"  {name}: {st['rows']} rows (weighted {st['weighted_rows']:.0f}), "
              f"success rate {st['success_rate']:.3f}, "
              f"draft grades {st['draft_grade_counts']}")
        print(f"    missing fractions: {st['feature_missing_frac']}")
    n_seed = int((train.draft_year == -1).sum())
    print(f"  seed rows in train: {n_seed} (each once, weight {SEED_ROW_WEIGHT:g}); "
          f"CSV rows weight 1; NO synthetic rows anywhere")
    assert n_seed == len(SEED_TRAINING_PLAYERS) == 63, "seed rows must appear exactly once"
    print(f"  features ({len(SUCCESS_FEATURES)}): {SUCCESS_FEATURES}")

    if args.dry_run:
        print("\n--dry-run: splits verified, exiting before any model fitting.")
        return 0

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining success ensemble (binary)…")
    s_bundle = train_success(train, cal)
    print("\nTraining draft-grade ensemble (4-class)…")
    g_bundle = train_draft_grade(train, cal)

    # ── Evaluate on the held-out 2019-2020 test years ────────────────────────
    X_te = test[SUCCESS_FEATURES]
    y_s  = test.nfl_success.to_numpy()
    y_g  = test.draft_grade.to_numpy()

    p_raw = ensemble_success_probs(s_bundle, X_te, calibrated=False)
    p_calib = ensemble_success_probs(s_bundle, X_te, calibrated=True)
    p_base = baseline_success_probs(test)

    success_eval = {
        "ensemble_calibrated": binary_metrics(y_s, p_calib),
        "ensemble_raw_mean":   binary_metrics(y_s, p_raw),
        "xgboost_member":      binary_metrics(y_s, s_bundle["xgb"].predict_proba(X_te)[:, 1]),
        "catboost_member":     binary_metrics(y_s, s_bundle["cb"].predict_proba(X_te)[:, 1]),
        "rule_based_baseline": binary_metrics(y_s, p_base),
        "reliability_10bin_calibrated": reliability_table(y_s, p_calib),
        "reliability_10bin_baseline":   reliability_table(y_s, p_base),
    }

    P_raw = ensemble_grade_probs(g_bundle, X_te, calibrated=False)
    P_calib = ensemble_grade_probs(g_bundle, X_te, calibrated=True)
    g_base = baseline_grade_preds(test)

    grade_eval = {
        "ensemble_calibrated": grade_metrics(y_g, P_calib.argmax(axis=1)),
        "ensemble_raw_mean":   grade_metrics(y_g, P_raw.argmax(axis=1)),
        "rule_based_baseline": grade_metrics(y_g, g_base),
    }

    # ── Print ────────────────────────────────────────────────────────────────
    print("\n══ HELD-OUT TEST (2019-2020) — SUCCESS ══")
    for k in ("ensemble_calibrated", "ensemble_raw_mean", "xgboost_member",
              "catboost_member", "rule_based_baseline"):
        m = success_eval[k]
        print(f"  {k:<22} AUC {m['auc']:.4f}  Brier {m['brier']:.4f}  logloss {m['log_loss']:.4f}")
    print("  Reliability (calibrated ensemble):")
    print_reliability(success_eval["reliability_10bin_calibrated"])
    print("  Reliability (rule-based baseline):")
    print_reliability(success_eval["reliability_10bin_baseline"])

    print("\n══ HELD-OUT TEST (2019-2020) — DRAFT GRADE ══")
    for k in ("ensemble_calibrated", "ensemble_raw_mean", "rule_based_baseline"):
        m = grade_eval[k]
        print(f"  {k:<22} accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
        print(f"    confusion (rows=true 0..3, cols=pred 0..3): {m['confusion_matrix']}")

    # ── Decision gate: serve whichever of {ensemble, heuristic} wins ─────────
    # The served success ensemble must beat the rule-based baseline on the
    # holdout on BOTH discrimination (AUC, higher) and calibration (Brier,
    # lower); otherwise XGBOost.py must serve the heuristic fallback.
    ens = success_eval["ensemble_calibrated"]
    base = success_eval["rule_based_baseline"]
    ensemble_wins = ens["auc"] > base["auc"] and ens["brier"] < base["brier"]
    serve = "ensemble" if ensemble_wins else "heuristic"
    print(f"\nDECISION GATE (success, holdout 2019-2020): "
          f"ensemble AUC {ens['auc']:.4f} vs baseline {base['auc']:.4f}; "
          f"ensemble Brier {ens['brier']:.4f} vs baseline {base['brier']:.4f} "
          f"→ serve = {serve}")
    if not ensemble_wins:
        print("  The trained ensemble did NOT beat the rule-based baseline on the "
              "holdout. XGBOost.py will honor serve=heuristic and prefer "
              "determine_success_fallback.")

    # ── Metadata ─────────────────────────────────────────────────────────────
    metadata = {
        "serve": serve,
        "serve_gate": {
            "rule": "serve ensemble iff holdout AUC(ensemble) > AUC(baseline) "
                    "AND Brier(ensemble) < Brier(baseline); else heuristic",
            "ensemble": {"auc": ens["auc"], "brier": ens["brier"]},
            "baseline": {"auc": base["auc"], "brier": base["brier"]},
        },
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "training_data": {
            "path": os.path.relpath(TRAINING_DATA_PATH, REPO_ROOT),
            "sha256": file_sha256(TRAINING_DATA_PATH),
        },
        "seed": SEED,
        "splits": {
            "strategy": "temporal groups by draft_year (train 2010-2017 + 63 curated seed rows w=5; "
                        "calibration 2018; test 2019-2020). No player appears in two sets; "
                        "evaluation is on draft classes after all training classes.",
            **split_stats,
        },
        "features": list(SUCCESS_FEATURES),
        "models": {
            "success": {
                "members": ["xgboost (success_xgboost_model.json)",
                            "catboost (catboost_success_model.cbm)"],
                "calibrator": "success_calibrated_model.pkl — Platt (LogisticRegression on "
                              "logit of member-mean prob), fit on 2018 ensemble outputs",
                "scale_pos_weight": s_bundle["scale_pos_weight"],
            },
            "draft_grade": {
                "members": ["xgboost (draft_grade_model.json)",
                            "catboost (catboost_draft_grade_model.cbm)"],
                "calibrator": "draft_grade_calibrated_model.pkl — multinomial LogisticRegression "
                              "on log member-mean probs, fit on 2018 ensemble outputs",
                "class_weights": g_bundle["class_weights"],
            },
        },
        "evaluation": {
            "holdout_years": sorted(TEST_YEARS),
            "success": success_eval,
            "draft_grade": grade_eval,
            "baseline_notes": "rule_based_baseline = dv_heuristics (the app's fallback), accolade "
                              f"flags 0, NaN production/speed → neutral {BASELINE_NEUTRAL:g}",
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "catboost": catboost_pkg.__version__,
        },
    }
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\nWrote {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
