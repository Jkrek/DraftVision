#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train + evaluate the DraftVision success and draft-grade ensembles (v3 features).

This is the ONLY place models are trained. The Flask app (XGBOost.py) is
load-only and will refuse to start if the artifacts written here are missing
or feature-mismatched (unless DV_ALLOW_MISSING_MODELS=1).

Data
----
training_data/combine_outcomes.csv (built by scripts/build_training_data.py):
real nflverse draft picks 2000-2026 + undrafted combine invitees, with
- legacy college production (cfbfastR) mapped to position-group percentiles,
- raw combine measurables (height/weight/vertical/bench/broad/cone/shuttle),
- CFBD recruiting pedigree (stars/rating/national rank) + the non-leaky age
  proxy (years_in_college = draft class - recruit class; early_declare),
- CFBD all-position final-season/career production composites + car_seasons,
- SP+ team rating of the final college season.
NO synthetic rows anywhere. The 63 hand-curated SEED_TRAINING_PLAYERS
(dv_features.py) are appended ONCE each with sample_weight=5 (NaN for every
v3 feature); all CSV rows get sample_weight=1.

Feature standardization (models/feature_stats.json)
---------------------------------------------------
Measurables are z-scored per position group and production composites per
composite group. The reference mean/std NEVER include years after the split's
test years; the FINAL production fit's reference stats are persisted to
models/feature_stats.json so serving (XGBOost.py / dv_features.py)
standardizes with exactly the numbers training used.

Two phases (both run every time)
--------------------------------
1. EVAL — the measured v3 winner configuration ("+A+B+C+D | flat | hist",
   models/experiments/RESULTS.md): train 2000-2017 (+63 seed rows w=5),
   calibration 2018, frozen test 2019-2020, z-reference 2000-2018. All holdout
   metrics in models/metadata.json come from THIS phase (the final fit has no
   honest holdout left, by design). The success serve-gate vs the rule-based
   baseline is decided here, exactly as before.
2. FINAL — production artifacts: draft-grade members on ALL classes 2000-2026
   (+seeds); success members on all label-mature classes 2000-2021 (+seeds;
   later classes have right-censored success labels). Calibrators are refit on
   OUT-OF-FOLD predictions from shadow members: grade shadow trains 2000-2024
   and calibrates on 2025-2026; success shadow trains 2000-2020 and calibrates
   on 2021. z-reference = all trained years (persisted, see above).

Artifacts (existing filenames kept)
-----------------------------------
  success_xgboost_model.json / catboost_success_model.cbm      — members
  draft_grade_model.json     / catboost_draft_grade_model.cbm  — members
  success_calibrated_model.pkl     — {"kind": "platt_logit", ...}
  draft_grade_calibrated_model.pkl — {"kind": "multinomial_logprob", ...}
  models/feature_stats.json        — frozen z-score reference stats
  models/metadata.json             — eval metrics + final-fit provenance + gate

Usage
-----
  .venv/bin/python scripts/train_models.py            # eval + final fit
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
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
import catboost as catboost_pkg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    roc_auc_score,
)

from dv_features import (
    SUCCESS_FEATURES,
    MEASURABLE_COLS,
    DRAFT_GRADE_LABELS,
    SEED_TRAINING_PLAYERS,
    FEATURE_STATS_PATH,
    OFF_FLOOR_CLASS,
    DEF_FLOOR_CLASS,
    position_flags,
    _production_group,
    _composite_group,
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
DRAFT_PICK_MODEL_PATH       = os.path.join(REPO_ROOT, "draft_pick_model.json")
CATBOOST_DRAFT_PICK_PATH    = os.path.join(REPO_ROOT, "catboost_draft_pick_model.cbm")
CAREER_AV_MODEL_PATH        = os.path.join(REPO_ROOT, "career_av_model.json")
CATBOOST_CAREER_AV_PATH     = os.path.join(REPO_ROOT, "catboost_career_av_model.cbm")
CATBOOST_PICK_Q10_PATH      = os.path.join(REPO_ROOT, "catboost_draft_pick_q10_model.cbm")
CATBOOST_PICK_Q90_PATH      = os.path.join(REPO_ROOT, "catboost_draft_pick_q90_model.cbm")

# Pick-interval nominal coverage (conformalized on a held-out fold)
PICK_INTERVAL_COVERAGE = 0.80

# Pick-regression target: overall pick 1-262 for drafted rows; undrafted rows
# train at a pseudo-pick past the end of the draft so the model learns the
# full ordering. Regression happens in log space (pick value ratios matter:
# pick 5 vs 15 is a chasm, 155 vs 165 is noise).
UDFA_PICK = 300.0

SEED = 42
SEED_ROW_WEIGHT = 5.0

FIRST_CLASS, LAST_CLASS = 2000, 2026
SUCCESS_LABEL_MATURE_THROUGH = 2021   # >= 5 completed NFL seasons (2021-2025)

# EVAL phase — the measured v3 winner split (frozen holdout, RESULTS.md)
EVAL_TRAIN_YEARS = set(range(2000, 2018))
CAL_YEARS   = {2018}
TEST_YEARS  = {2019, 2020}
EVAL_REF_YEARS = set(range(2000, 2019))      # z-score reference: train + cal only

# FINAL phase — production fit
FINAL_GRADE_YEARS       = set(range(FIRST_CLASS, LAST_CLASS + 1))          # 2000-2026
FINAL_SUCCESS_YEARS     = set(range(FIRST_CLASS, SUCCESS_LABEL_MATURE_THROUGH + 1))
FINAL_REF_YEARS         = FINAL_GRADE_YEARS                                 # persisted
GRADE_SHADOW_TRAIN      = set(range(2000, 2025))   # OOF calibration folds
GRADE_CAL_FOLD          = {2025, 2026}
SUCCESS_SHADOW_TRAIN    = set(range(2000, 2021))
SUCCESS_CAL_FOLD        = {2021}

# Leakage guard: none of these may ever appear in X (v2 list + the disqualified
# draft-file `age` + raw identifiers of the new CFBD columns).
FORBIDDEN_FEATURES = {
    "draft_round", "pick", "experience", "experience_years", "pro_bowls",
    "probowls", "seasons_started", "w_av", "car_av", "career_av", "to",
    "nfl_success", "draft_grade", "draft_year", "name", "sample_weight",
    "age", "rec_year", "rec_match_type", "draft_pick",
}

# Baseline-only neutral values: the heuristic is arithmetic and cannot take
# NaN, so unknown production/speed become the scale midpoint 50. This is a
# documented baseline convention, NOT a model imputation — the ML models see
# true NaN.
BASELINE_NEUTRAL = 50.0

FS_STATS = ["pass_yds", "pass_td", "pass_int", "pass_att",
            "rush_yds", "rush_td", "rush_car",
            "rec", "rec_yds", "rec_td",
            "tackles", "tfl", "sacks", "pd", "def_int"]

REC_FEATS = ["rec_stars", "rec_rating", "rec_ranking"]
AGE_FEATS = ["years_in_college", "early_declare"]
SP_FEATS = ["sp_rating"]


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


def load_raw_rows() -> pd.DataFrame:
    """CSV -> frame with base-13 features, raw measurables, CFBD features and
    raw production composites (identical recipe to harness_v3.load_rows_v3 +
    add_composites, which produced the measured 0.5102 holdout accuracy).
    z-features are added per phase by apply_z()."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    fs_cols = [c for c in df.columns if c.startswith(("fs_", "car_"))]
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
            "draft_pick":          _nan_float(r.get("draft_pick")),
            # v4 scout signals (scripts/add_consensus_columns.py): unranked in
            # a covered year = rank 400 (boards passed); uncovered = NaN
            "consensus_logrank":   (
                (math.log(_nan_float(r.get("consensus_rank")))
                 if math.isfinite(_nan_float(r.get("consensus_rank")))
                 else math.log(400.0))
                if _nan_float(r.get("consensus_covered")) == 1.0 else float("nan")
            ),
            "allstar_invite":      _nan_float(r.get("allstar_invite")),
            # label passthrough for the career-value head — in
            # FORBIDDEN_FEATURES, so it can never enter X
            "career_av":           _nan_float(r.get("career_av")),
            "_pos_group":          _production_group(pos),
            "_grp":                _composite_group(pos),
        }
        for m in MEASURABLE_COLS + REC_FEATS + AGE_FEATS + SP_FEATS + fs_cols:
            row[m] = _nan_float(r.get(m))
        rows.append(row)
    out = pd.DataFrame(rows)
    return add_composites(out)


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Raw final-season / career production composites per composite group.
    NaN below the source coverage floor for the group (offense 2010+ classes,
    defense 2017+) and for OL/OTHER — identical to harness_v3.add_composites
    (formulas mirrored in dv_features.production_composite for serving)."""
    for pre in ("fs", "car"):
        g = lambda k: df[f"{pre}_{k}"]
        qb = g("pass_yds") + 20 * g("pass_td") - 45 * g("pass_int") \
            + 0.5 * g("rush_yds") + 10 * g("rush_td")
        rb = g("rush_yds") + 20 * g("rush_td") + g("rec_yds") + 20 * g("rec_td")
        wr = g("rec_yds") + 20 * g("rec_td") + 0.5 * g("rush_yds")
        dfe = g("tackles") + 3 * g("tfl") + 8 * g("sacks") + 4 * g("pd") \
            + 10 * g("def_int")
        comp = pd.Series(np.nan, index=df.index)
        grp = df["_grp"]
        yr = df["draft_year"]
        comp[(grp == "QB") & (yr >= OFF_FLOOR_CLASS)] = qb
        comp[(grp == "RB") & (yr >= OFF_FLOOR_CLASS)] = rb
        comp[grp.isin(["WR", "TE"]) & (yr >= OFF_FLOOR_CLASS)] = wr
        comp[grp.isin(["DL", "LB", "DB"]) & (yr >= DEF_FLOOR_CLASS)] = dfe
        df[f"prod_{pre}_raw"] = comp
    df.loc[df["draft_year"] < OFF_FLOOR_CLASS, "car_seasons"] = np.nan
    return df


def stats_from_ref(df: pd.DataFrame, ref_years: set) -> dict:
    """JSON-able frozen z-score reference stats from ref_years rows only."""
    ref = df[df.draft_year.isin(ref_years)]
    meas = {}
    for grp, sub in ref.groupby("_pos_group"):
        meas[grp] = {}
        for m in MEASURABLE_COLS:
            mu, sd = float(sub[m].mean()), float(sub[m].std())
            meas[grp][m] = {"mean": mu, "std": sd}
    comps = {}
    for grp, sub in ref.groupby("_grp"):
        comps[grp] = {}
        for pre in ("fs", "car"):
            col = f"prod_{pre}_raw"
            mu, sd = float(sub[col].mean()), float(sub[col].std())
            comps[grp][pre] = {"mean": mu, "std": sd}
    return {"reference_years": sorted(int(y) for y in ref_years),
            "measurables": meas, "composites": comps}


def apply_z(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Add measurable *_z and prod_fs_z/prod_car_z from frozen stats.
    Identical arithmetic to harness_v3.prepare (std 0/NaN -> feature NaN)."""
    df = df.copy()

    def _map(groups, key, sub):
        mu = df[key].map(lambda gname: (groups.get(gname) or {}).get(sub, {}).get("mean", np.nan))
        sd = df[key].map(lambda gname: (groups.get(gname) or {}).get(sub, {}).get("std", np.nan))
        sd = sd.replace(0, np.nan)
        return mu.astype(float), sd.astype(float)

    for m in MEASURABLE_COLS:
        mu, sd = _map(stats["measurables"], "_pos_group", m)
        df[m + "_z"] = (df[m] - mu) / sd
    for pre in ("fs", "car"):
        mu, sd = _map(stats["composites"], "_grp", pre)
        df[f"prod_{pre}_z"] = (df[f"prod_{pre}_raw"] - mu) / sd
    return df


def load_csv_rows() -> pd.DataFrame:
    """Full feature frame standardized with the PERSISTED production stats
    (models/feature_stats.json) — the serving-identical view. Kept for
    scripts/generate_backtest.py; training itself uses phase-specific refs."""
    with open(FEATURE_STATS_PATH) as fh:
        stats = json.load(fh)
    return apply_z(load_raw_rows(), stats)


def seed_rows() -> pd.DataFrame:
    """The 63 hand-curated seed players — added ONCE each, sample_weight=5.
    Every v3 feature (measurables, recruiting, age proxy, composites, SP+)
    is NaN on seeds, exactly as in the measured experiments."""
    rows = []
    for sp in SEED_TRAINING_PLAYERS:
        row = {
            "name":                "(seed)",
            "draft_year":          -1,  # curated exemplars — pinned to the train fold
            "sample_weight":       SEED_ROW_WEIGHT,
            "production_score":    float(sp["production_score"]),
            # games_played MUST stay NaN: the CSV's games_college column is
            # empty on all 9,965 rows, so a value here would make this
            # feature a seed-row indicator — and CatBoost's quantile heads
            # measurably learned "games_played present → seed-bust → UDFA",
            # sending every LIVE player (who always has a games count) to a
            # pick-300 upper bound. All-NaN in training = unsplittable =
            # harmless at serve.
            "games_played":        float("nan"),
            "combine_speed_score": float(sp["combine_speed_score"]),
            "conference_tier":     float(sp["conference_tier"]),
            **position_flags(sp["position"]),
            "nfl_success":         int(sp["nfl_success"]),
            "draft_grade":         int(sp["draft_grade"]),
            "_pos_group":          _production_group(sp["position"]),
            "_grp":                _composite_group(sp["position"]),
        }
        for f in SUCCESS_FEATURES:
            row.setdefault(f, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def frame_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "weighted_rows": float(df.sample_weight.sum()),
        "success_positive": int(df.nfl_success.sum()),
        "success_rate": round(float(df.nfl_success.mean()), 4),
        "draft_grade_counts": {int(k): int(v) for k, v in df.draft_grade.value_counts().sort_index().items()},
        "feature_missing_frac": {
            f: round(float(df[f].isna().mean()), 4)
            for f in ("production_score", "combine_speed_score", "height_in_z",
                      "rec_stars", "years_in_college", "prod_fs_z", "sp_rating")
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
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class_recall": [round(float(r), 4) for r in rec],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": DRAFT_GRADE_LABELS,
    }


def print_reliability(table: list) -> None:
    print(f"    {'bin':<12}{'count':>7}{'mean_pred':>11}{'frac_pos':>10}")
    for row in table:
        mp = f"{row['mean_predicted']:.3f}" if row["mean_predicted"] is not None else "-"
        fp = f"{row['fraction_positive']:.3f}" if row["fraction_positive"] is not None else "-"
        print(f"    {row['bin']:<12}{row['count']:>7}{mp:>11}{fp:>10}")


# ── Model fitting (member fits shared by eval and final phases) ───────────────

def _xy(df: pd.DataFrame, target: str):
    return df[SUCCESS_FEATURES], df[target].to_numpy(), df.sample_weight.to_numpy()


def fit_success_members(train: pd.DataFrame):
    X_tr, y_tr, w_tr = _xy(train, "nfl_success")
    spw = float(w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-9))
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
        scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: spw},
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "scale_pos_weight": round(spw, 4)}


def fit_success_calibrator(members: dict, cal: pd.DataFrame, cal_years) -> dict:
    """Platt on the logit of the member-mean probability (what is served)."""
    X_cal, y_cal, _ = _xy(cal, "nfl_success")
    p_cal = np.mean([
        members["xgb"].predict_proba(X_cal)[:, 1],
        members["cb"].predict_proba(X_cal)[:, 1],
    ], axis=0)
    z = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cal, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(z.reshape(-1, 1), y_cal)
    return {
        "kind": "platt_logit",
        "model": platt,
        "members": ["xgboost", "catboost"],
        "feature_names": list(SUCCESS_FEATURES),
        "calibration_years": sorted(cal_years),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def fit_grade_members(train: pd.DataFrame):
    """Flat 4-class XGB+CatBoost members — the measured v3 winner method."""
    X_tr, y_tr, w_tr = _xy(train, "draft_grade")
    cls_w = {}
    total_w = w_tr.sum()
    for c in range(4):
        cw = w_tr[y_tr == c].sum()
        cls_w[c] = float(total_w / (4.0 * max(cw, 1e-9)))
    w_eff = w_tr * np.array([cls_w[c] for c in y_tr])
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
        objective="multi:softprob", num_class=4,
        random_state=SEED, eval_metric="mlogloss",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_eff)
    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="MultiClass", classes_count=4,
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_eff)
    return {"xgb": xgb_m, "cb": cb_m,
            "class_weights": {c: round(v, 4) for c, v in cls_w.items()}}


def fit_grade_calibrator(members: dict, cal: pd.DataFrame, cal_years) -> dict:
    """Multinomial LR on log member-mean probabilities."""
    X_cal, y_cal, _ = _xy(cal, "draft_grade")
    P_cal = np.mean([
        members["xgb"].predict_proba(X_cal),
        members["cb"].predict_proba(X_cal),
    ], axis=0)
    logp = np.log(np.clip(P_cal, 1e-6, 1.0))
    mnl = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    mnl.fit(logp, y_cal)
    return {
        "kind": "multinomial_logprob",
        "model": mnl,
        "members": ["xgboost", "catboost"],
        "feature_names": list(SUCCESS_FEATURES),
        "calibration_years": sorted(cal_years),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _pick_target(df: pd.DataFrame) -> np.ndarray:
    """log(expected pick): real pick for drafted rows, UDFA_PICK for round-8
    rows, NaN (excluded) for rows with neither (seed rows)."""
    pick = df["draft_pick"].to_numpy(dtype=float)
    y = np.where(np.isfinite(pick), pick, np.nan)
    y[(df["draft_grade"].to_numpy() == 3) & ~np.isfinite(pick)] = UDFA_PICK
    return np.log(y)


def fit_pick_members(train: pd.DataFrame):
    """XGB+CatBoost regressors on log(overall pick) — the fine-grained head
    behind round-level projections (the 4-class head can't tell pick 3
    from pick 45)."""
    y = _pick_target(train)
    m = np.isfinite(y)
    X_tr, y_tr = train[SUCCESS_FEATURES][m], y[m]
    w_tr = train.sample_weight.to_numpy()[m]
    xgb_m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=SEED, objective="reg:squarederror",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05, loss_function="RMSE",
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "n_rows": int(m.sum())}


def ensemble_pick_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Serving-equivalent: exp(mean of member log-pick predictions)."""
    logp = np.mean([bundle["xgb"].predict(X), bundle["cb"].predict(X.values)], axis=0)
    return np.clip(np.exp(logp), 1.0, UDFA_PICK)


def fit_av_members(train: pd.DataFrame):
    """XGB+CatBoost regressors on log1p(career_av clipped at 0) — the
    continuous career-value head. Display-only: it ranks likely hits by
    career ceiling (stars vs starters), which the binary success head cannot
    express. Seed rows carry no career_av and drop out via the NaN mask."""
    y_raw = pd.to_numeric(train.get("career_av"), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(y_raw)
    y = np.log1p(np.clip(y_raw[m], 0.0, None))
    X_tr, w_tr = train[SUCCESS_FEATURES][m], train.sample_weight.to_numpy()[m]
    xgb_m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=SEED, objective="reg:squarederror",
    )
    xgb_m.fit(X_tr, y, sample_weight=w_tr)
    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05, loss_function="RMSE",
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "n_rows": int(m.sum())}


def ensemble_av_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Serving-equivalent: expm1(mean of member log1p-AV predictions), >= 0."""
    logv = np.mean([bundle["xgb"].predict(X), bundle["cb"].predict(X.values)], axis=0)
    return np.maximum(np.expm1(logv), 0.0)


def fit_pick_quantile_members(train: pd.DataFrame, alpha: float):
    """CatBoost quantile regressor on log(pick) — one bundle per side of the
    pick interval (alpha 0.1 = optimistic bound, 0.9 = pessimistic).

    CatBoost-ONLY by measurement, not preference: ~45% of the target mass
    sits exactly at the UDFA ceiling (log 300), and XGBoost's
    reg:quantileerror at alpha=0.9 collapses onto that point mass — it
    predicts the global quantile (300) for every row, including actual
    top-10 picks, across all tested configs. CatBoost's Quantile loss
    handles the ceiling correctly (top-10-pick q90 median ~51). Coverage is
    guaranteed by the conformal offsets either way."""
    y = _pick_target(train)
    m = np.isfinite(y)
    X_tr, y_tr, w_tr = train[SUCCESS_FEATURES][m], y[m], train.sample_weight.to_numpy()[m]
    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05,
        loss_function=f"Quantile:alpha={alpha}",
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"cb": cb_m}


def pick_quantile_log_preds(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Quantile prediction in LOG-pick space (no exp)."""
    return np.asarray(bundle["cb"].predict(X.values), dtype=float)


def conformal_pick_offsets(q10_log, q90_log, y_log,
                           coverage: float = PICK_INTERVAL_COVERAGE):
    """Split-conformal per-side additive offsets in log-pick space.

    Widens each bound so the calibration fold misses ~(1-coverage)/2 per
    side. Offsets are clipped at 0 — conformal only widens, never narrows
    (a negative offset would trade honesty for cosmetics)."""
    miss = (1.0 - coverage) / 2.0
    lo = float(np.quantile(q10_log - y_log, 1.0 - miss))  # how far q10 overshoots
    hi = float(np.quantile(y_log - q90_log, 1.0 - miss))  # how far q90 undershoots
    return max(lo, 0.0), max(hi, 0.0)


def interval_metrics(y_pick, lo_pick, hi_pick, blend=None) -> dict:
    """Coverage + width diagnostics for conformalized pick intervals."""
    cov = float(np.mean((y_pick >= lo_pick) & (y_pick <= hi_pick)))
    width = hi_pick - lo_pick
    out = {
        "coverage": round(cov, 4),
        "median_width_picks": round(float(np.median(width)), 1),
        "mean_width_picks": round(float(np.mean(width)), 1),
    }
    if blend is not None:
        top = blend <= 64
        if top.any():
            out["coverage_top64"] = round(
                float(np.mean((y_pick[top] >= lo_pick[top]) & (y_pick[top] <= hi_pick[top]))), 4)
            out["median_width_picks_top64"] = round(float(np.median(width[top])), 1)
    return out


def classifier_expected_pick(P: np.ndarray) -> np.ndarray:
    """Baseline the regressor must beat: expected pick implied by the 4-class
    head's probabilities (class midpoints 25/85/190/300)."""
    mids = np.array([25.0, 85.0, 190.0, UDFA_PICK])
    return P @ mids


def pick_metrics(y_pick: np.ndarray, pred: np.ndarray) -> dict:
    drafted = y_pick < UDFA_PICK
    rho = spearmanr(y_pick, pred).statistic
    rho_d = spearmanr(y_pick[drafted], pred[drafted]).statistic
    mae_d = float(np.mean(np.abs(y_pick[drafted] - pred[drafted])))
    r1_hit = float(np.mean((pred[y_pick <= 32] <= 45))) if (y_pick <= 32).any() else None
    t64 = y_pick <= 64
    rho_64 = spearmanr(y_pick[t64], pred[t64]).statistic if t64.sum() >= 20 else None
    return {
        "spearman_all": round(float(rho), 4),
        "spearman_drafted": round(float(rho_d), 4),
        "spearman_top64": round(float(rho_64), 4) if rho_64 is not None else None,
        "mae_picks_drafted": round(mae_d, 1),
        "r1_recall_within_45": round(r1_hit, 4) if r1_hit is not None else None,
    }


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


def _assert_no_leakage(train, cal, test) -> None:
    forbidden = set(SUCCESS_FEATURES) & FORBIDDEN_FEATURES
    assert not forbidden, f"forbidden feature in X: {forbidden}"
    assert not (set(train.draft_year.unique()) & TEST_YEARS), "test year in train"
    assert not (set(cal.draft_year.unique()) & TEST_YEARS), "test year in cal"
    assert set(test.draft_year.unique()) == TEST_YEARS
    assert not (EVAL_REF_YEARS & TEST_YEARS), "test year in z-score reference"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="load data, build splits, print shapes/balances, exit before fitting")
    args = ap.parse_args()

    np.random.seed(SEED)

    print(f"Loading {TRAINING_DATA_PATH}")
    raw = load_raw_rows()
    seeds = seed_rows()

    # ════ Phase 1: EVAL (frozen v3-winner split — all reported metrics) ════
    eval_stats = stats_from_ref(raw, EVAL_REF_YEARS)
    df_eval = apply_z(raw, eval_stats)

    train = pd.concat([df_eval[df_eval.draft_year.isin(EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal  = df_eval[df_eval.draft_year.isin(CAL_YEARS)].reset_index(drop=True)
    test = df_eval[df_eval.draft_year.isin(TEST_YEARS)].reset_index(drop=True)
    _assert_no_leakage(train, cal, test)

    split_stats = {
        "train_2000_2017_plus_seeds": frame_stats(train),
        "calibration_2018":           frame_stats(cal),
        "test_2019_2020":             frame_stats(test),
    }
    print("\nEVAL split (temporal groups by draft_year — no player in two sets):")
    for name, st in split_stats.items():
        print(f"  {name}: {st['rows']} rows (weighted {st['weighted_rows']:.0f}), "
              f"success rate {st['success_rate']:.3f}, "
              f"draft grades {st['draft_grade_counts']}")
    n_seed = int((train.draft_year == -1).sum())
    assert n_seed == len(SEED_TRAINING_PLAYERS) == 63, "seed rows must appear exactly once"
    print(f"  seed rows in train: {n_seed} (each once, weight {SEED_ROW_WEIGHT:g}); "
          f"CSV rows weight 1; NO synthetic rows anywhere")
    print(f"  features ({len(SUCCESS_FEATURES)}): {SUCCESS_FEATURES}")

    if args.dry_run:
        print("\n--dry-run: splits verified, exiting before any model fitting.")
        return 0

    print("\n[EVAL] Training success ensemble (binary, 2000-2017 + seeds)…")
    s_members = fit_success_members(train)
    s_bundle = {**s_members, "calibrator": fit_success_calibrator(s_members, cal, CAL_YEARS)}
    print("[EVAL] Training draft-grade ensemble (4-class flat, 2000-2017 + seeds)…")
    g_members = fit_grade_members(train)
    g_bundle = {**g_members, "calibrator": fit_grade_calibrator(g_members, cal, CAL_YEARS)}

    X_te = test[SUCCESS_FEATURES]
    y_s  = test.nfl_success.to_numpy()
    y_g  = test.draft_grade.to_numpy()

    p_raw   = ensemble_success_probs(s_bundle, X_te, calibrated=False)
    p_calib = ensemble_success_probs(s_bundle, X_te, calibrated=True)
    p_base  = baseline_success_probs(test)

    success_eval = {
        "ensemble_calibrated": binary_metrics(y_s, p_calib),
        "ensemble_raw_mean":   binary_metrics(y_s, p_raw),
        "xgboost_member":      binary_metrics(y_s, s_bundle["xgb"].predict_proba(X_te)[:, 1]),
        "catboost_member":     binary_metrics(y_s, s_bundle["cb"].predict_proba(X_te)[:, 1]),
        "rule_based_baseline": binary_metrics(y_s, p_base),
        "reliability_10bin_calibrated": reliability_table(y_s, p_calib),
        "reliability_10bin_baseline":   reliability_table(y_s, p_base),
    }

    P_raw   = ensemble_grade_probs(g_bundle, X_te, calibrated=False)
    P_calib = ensemble_grade_probs(g_bundle, X_te, calibrated=True)
    g_base  = baseline_grade_preds(test)

    grade_eval = {
        "ensemble_raw_mean":   grade_metrics(y_g, P_raw.argmax(axis=1)),   # served label
        "ensemble_calibrated": grade_metrics(y_g, P_calib.argmax(axis=1)),
        "rule_based_baseline": grade_metrics(y_g, g_base),
    }

    print("\n══ EVAL HOLDOUT (2019-2020) — SUCCESS ══")
    for k in ("ensemble_calibrated", "ensemble_raw_mean", "xgboost_member",
              "catboost_member", "rule_based_baseline"):
        m = success_eval[k]
        print(f"  {k:<22} AUC {m['auc']:.4f}  Brier {m['brier']:.4f}  logloss {m['log_loss']:.4f}")
    print("  Reliability (calibrated ensemble):")
    print_reliability(success_eval["reliability_10bin_calibrated"])

    print("\n══ EVAL HOLDOUT (2019-2020) — DRAFT GRADE ══")
    for k in ("ensemble_raw_mean", "ensemble_calibrated", "rule_based_baseline"):
        m = grade_eval[k]
        print(f"  {k:<22} accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
              f"recall {m['per_class_recall']}")
    print(f"    confusion (raw, rows=true 0..3): {grade_eval['ensemble_raw_mean']['confusion_matrix']}")

    # ── Draft-pick regressor (fine-grained projections) ──────────────────────
    print("\n[EVAL] Training draft-pick regressor (log-pick, 2000-2017 + seeds)…")
    p_members = fit_pick_members(train)
    y_pick_te = np.exp(_pick_target(test))
    pick_m = np.isfinite(y_pick_te)
    pred_pick = ensemble_pick_preds(p_members, X_te[pick_m])
    base_pick = classifier_expected_pick(P_raw[pick_m])
    # SERVED estimator: 50/50 log-space blend of regressor and the 4-class
    # head's implied pick — beats both parents on holdout (regressor alone
    # is noisy mid-draft; classifier alone can't order the top).
    blend_pick = np.exp(0.5 * (np.log(pred_pick) + np.log(base_pick)))
    pick_eval = {
        "blend_50_50_SERVED":   pick_metrics(y_pick_te[pick_m], blend_pick),
        "regressor":            pick_metrics(y_pick_te[pick_m], pred_pick),
        "classifier_expected":  pick_metrics(y_pick_te[pick_m], base_pick),
        "n_train_rows":         p_members["n_rows"],
    }
    print("\n══ EVAL HOLDOUT (2019-2020) — DRAFT PICK (served blend vs parents) ══")
    for k in ("blend_50_50_SERVED", "regressor", "classifier_expected"):
        m = pick_eval[k]
        print(f"  {k:<22} Spearman(all) {m['spearman_all']:.4f}  "
              f"top64 {m['spearman_top64']}  "
              f"MAE {m['mae_picks_drafted']:.1f} picks  "
              f"R1-recall@45 {m['r1_recall_within_45']}")
    # Gate on the SERVED estimator. The blend exists for TOP-OF-DRAFT
    # granularity (the 4-class head can't tell pick 3 from pick 45), so it
    # must strictly win top-64 ordering, and be non-inferior on global
    # ordering (within 0.01 rho) and first-round recall.
    r, c = pick_eval["blend_50_50_SERVED"], pick_eval["classifier_expected"]
    pick_wins = (
        (r["spearman_top64"] or 0) > (c["spearman_top64"] or 0)
        and r["spearman_all"] >= c["spearman_all"] - 0.01
        and (r["r1_recall_within_45"] or 0) >= (c["r1_recall_within_45"] or 0)
    )
    print(f"  DECISION GATE (pick, blend): all {r['spearman_all']} vs {c['spearman_all']}, "
          f"top64 {r['spearman_top64']} vs {c['spearman_top64']}, "
          f"R1-recall {r['r1_recall_within_45']} vs {c['r1_recall_within_45']} "
          f"→ serve_pick = {pick_wins}")

    # ── Career-value head (display-only; measured in experiment_career_av) ───
    print("\n[EVAL] Training career-AV regressor (log1p AV, 2000-2017 + seeds)…")
    av_members = fit_av_members(train)
    av_pred = ensemble_av_preds(av_members, X_te)
    av_true = pd.to_numeric(test["career_av"], errors="coerce").to_numpy(dtype=float)
    drafted_m = np.isfinite(test["draft_pick"].to_numpy(dtype=float))
    top100 = np.argsort(-p_calib)[:100]
    av_eval = {
        "spearman_all": round(float(spearmanr(av_true, av_pred).statistic), 4),
        "spearman_drafted": round(float(spearmanr(av_true[drafted_m], av_pred[drafted_m]).statistic), 4),
        "spearman_success_top100": round(float(spearmanr(av_true[top100], av_pred[top100]).statistic), 4),
        "pearson_vs_success_prob": round(float(np.corrcoef(av_pred, p_calib)[0, 1]), 4),
        "n_fit_rows": av_members["n_rows"],
    }
    # Gate (pre-registered in models/experiments/career_av_results.json):
    # useful ordering overall AND not a duplicate of the success head.
    serve_av = (av_eval["spearman_all"] >= 0.55
                and av_eval["pearson_vs_success_prob"] < 0.95)
    print(f"  Spearman(all) {av_eval['spearman_all']}  "
          f"drafted {av_eval['spearman_drafted']}  "
          f"success-top100 {av_eval['spearman_success_top100']}  "
          f"pearson-vs-success {av_eval['pearson_vs_success_prob']}")
    print(f"  DECISION GATE (career AV): spearman >= 0.55 and pearson < 0.95 "
          f"→ serve_av = {serve_av}")

    # ── Pick intervals: quantile heads + split conformal (report-only here;
    #    production offsets come from the FINAL phase's OOF fold) ─────────────
    print("\n[EVAL] Training pick quantile heads (alpha 0.1 / 0.9)…")
    q10_members = fit_pick_quantile_members(train, 0.1)
    q90_members = fit_pick_quantile_members(train, 0.9)
    y_cal_pick = _pick_target(cal)
    cal_pm = np.isfinite(y_cal_pick)
    X_cal_p = cal[SUCCESS_FEATURES][cal_pm]
    eval_lo_off, eval_hi_off = conformal_pick_offsets(
        pick_quantile_log_preds(q10_members, X_cal_p),
        pick_quantile_log_preds(q90_members, X_cal_p),
        y_cal_pick[cal_pm])
    lo_pick = np.clip(np.exp(
        pick_quantile_log_preds(q10_members, X_te[pick_m]) - eval_lo_off), 1.0, UDFA_PICK)
    hi_pick = np.clip(np.exp(
        pick_quantile_log_preds(q90_members, X_te[pick_m]) + eval_hi_off), 1.0, UDFA_PICK)
    interval_eval = interval_metrics(y_pick_te[pick_m], lo_pick, hi_pick,
                                     blend=blend_pick)
    interval_eval["lo_offset_log"] = round(eval_lo_off, 4)
    interval_eval["hi_offset_log"] = round(eval_hi_off, 4)
    print(f"  holdout coverage {interval_eval['coverage']:.3f} "
          f"(target {PICK_INTERVAL_COVERAGE}), median width "
          f"{interval_eval['median_width_picks']} picks "
          f"(top64: {interval_eval.get('median_width_picks_top64')})")

    # ── Decision gate: serve whichever of {ensemble, heuristic} wins ─────────
    ens = success_eval["ensemble_calibrated"]
    base = success_eval["rule_based_baseline"]
    ensemble_wins = ens["auc"] > base["auc"] and ens["brier"] < base["brier"]
    serve = "ensemble" if ensemble_wins else "heuristic"
    print(f"\nDECISION GATE (success, eval holdout 2019-2020): "
          f"ensemble AUC {ens['auc']:.4f} vs baseline {base['auc']:.4f}; "
          f"ensemble Brier {ens['brier']:.4f} vs baseline {base['brier']:.4f} "
          f"→ serve = {serve}")
    if not ensemble_wins:
        print("  The trained ensemble did NOT beat the rule-based baseline on the "
              "holdout. XGBOost.py will honor serve=heuristic and prefer "
              "determine_success_fallback.")

    # ════ Phase 2: FINAL production fit (all data; artifacts written here) ════
    print("\n[FINAL] Recomputing frozen z-reference stats from classes "
          f"{min(FINAL_REF_YEARS)}-{max(FINAL_REF_YEARS)} → {FEATURE_STATS_PATH}")
    final_stats = stats_from_ref(raw, FINAL_REF_YEARS)
    df_final = apply_z(raw, final_stats)
    os.makedirs(os.path.dirname(FEATURE_STATS_PATH), exist_ok=True)
    with open(FEATURE_STATS_PATH, "w") as fh:
        json.dump(final_stats, fh, indent=2)

    # Success: members on all label-mature classes; calibrator on OOF 2021 preds
    print(f"[FINAL] Success: shadow fit {min(SUCCESS_SHADOW_TRAIN)}-{max(SUCCESS_SHADOW_TRAIN)} "
          f"→ calibrator on {sorted(SUCCESS_CAL_FOLD)}; members on "
          f"{min(FINAL_SUCCESS_YEARS)}-{max(FINAL_SUCCESS_YEARS)} (+seeds)")
    s_shadow = fit_success_members(pd.concat(
        [df_final[df_final.draft_year.isin(SUCCESS_SHADOW_TRAIN)], seeds], ignore_index=True))
    s_calibrator = fit_success_calibrator(
        s_shadow, df_final[df_final.draft_year.isin(SUCCESS_CAL_FOLD)].reset_index(drop=True),
        SUCCESS_CAL_FOLD)
    s_final = fit_success_members(pd.concat(
        [df_final[df_final.draft_year.isin(FINAL_SUCCESS_YEARS)], seeds], ignore_index=True))
    s_final["xgb"].save_model(SUCCESS_MODEL_PATH)
    s_final["cb"].save_model(CATBOOST_SUCCESS_PATH)
    joblib.dump(s_calibrator, SUCCESS_CALIBRATED_PATH)

    # Grade: members on ALL classes; calibrator on OOF 2025-2026 preds
    print(f"[FINAL] Grade: shadow fit {min(GRADE_SHADOW_TRAIN)}-{max(GRADE_SHADOW_TRAIN)} "
          f"→ calibrator on {sorted(GRADE_CAL_FOLD)}; members on "
          f"{min(FINAL_GRADE_YEARS)}-{max(FINAL_GRADE_YEARS)} (+seeds)")
    g_shadow = fit_grade_members(pd.concat(
        [df_final[df_final.draft_year.isin(GRADE_SHADOW_TRAIN)], seeds], ignore_index=True))
    g_calibrator = fit_grade_calibrator(
        g_shadow, df_final[df_final.draft_year.isin(GRADE_CAL_FOLD)].reset_index(drop=True),
        GRADE_CAL_FOLD)
    g_final = fit_grade_members(pd.concat(
        [df_final[df_final.draft_year.isin(FINAL_GRADE_YEARS)], seeds], ignore_index=True))
    g_final["xgb"].save_model(DRAFT_GRADE_MODEL_PATH)
    g_final["cb"].save_model(CATBOOST_DRAFT_GRADE_PATH)
    joblib.dump(g_calibrator, DRAFT_GRADE_CALIBRATED_PATH)

    # Pick regressor: members on ALL classes (pick labels are immediate —
    # known on draft night, no NFL-outcome maturity needed)
    print(f"[FINAL] Pick: members on {min(FINAL_GRADE_YEARS)}-{max(FINAL_GRADE_YEARS)} (+seeds)")
    p_final = fit_pick_members(pd.concat(
        [df_final[df_final.draft_year.isin(FINAL_GRADE_YEARS)], seeds], ignore_index=True))
    p_final["xgb"].save_model(DRAFT_PICK_MODEL_PATH)
    p_final["cb"].save_model(CATBOOST_DRAFT_PICK_PATH)
    print(f"  pick members    → {DRAFT_PICK_MODEL_PATH}, {CATBOOST_DRAFT_PICK_PATH}")

    # Career-AV head: label-mature classes only (same window as success)
    print(f"[FINAL] Career-AV: members on {min(FINAL_SUCCESS_YEARS)}-{max(FINAL_SUCCESS_YEARS)}")
    av_final = fit_av_members(
        df_final[df_final.draft_year.isin(FINAL_SUCCESS_YEARS)].reset_index(drop=True))
    av_final["xgb"].save_model(CAREER_AV_MODEL_PATH)
    av_final["cb"].save_model(CATBOOST_CAREER_AV_PATH)
    print(f"  AV members      → {CAREER_AV_MODEL_PATH}, {CATBOOST_CAREER_AV_PATH}")

    # Pick quantile heads + production conformal offsets. House shadow
    # pattern (mirrors the grade calibrator): shadow quantiles fit WITHOUT
    # the conformal fold, offsets computed out-of-fold on 2025-2026 (their
    # picks are known — pick labels mature on draft night), final members
    # refit on all classes with those offsets carried over.
    q_shadow_years = FINAL_GRADE_YEARS - GRADE_CAL_FOLD
    print(f"[FINAL] Pick quantiles: shadow {min(q_shadow_years)}-{max(q_shadow_years)} "
          f"→ conformal offsets on {sorted(GRADE_CAL_FOLD)}; members on all classes")
    q_shadow_frame = pd.concat(
        [df_final[df_final.draft_year.isin(q_shadow_years)], seeds], ignore_index=True)
    q10_shadow = fit_pick_quantile_members(q_shadow_frame, 0.1)
    q90_shadow = fit_pick_quantile_members(q_shadow_frame, 0.9)
    conf = df_final[df_final.draft_year.isin(GRADE_CAL_FOLD)].reset_index(drop=True)
    y_conf = _pick_target(conf)
    conf_m = np.isfinite(y_conf)
    X_conf = conf[SUCCESS_FEATURES][conf_m]
    final_lo_off, final_hi_off = conformal_pick_offsets(
        pick_quantile_log_preds(q10_shadow, X_conf),
        pick_quantile_log_preds(q90_shadow, X_conf),
        y_conf[conf_m])
    print(f"  conformal offsets (log-pick): lo {final_lo_off:.4f}, hi {final_hi_off:.4f} "
          f"on {int(conf_m.sum())} OOF rows")
    q_final_frame = pd.concat(
        [df_final[df_final.draft_year.isin(FINAL_GRADE_YEARS)], seeds], ignore_index=True)
    q10_final = fit_pick_quantile_members(q_final_frame, 0.1)
    q90_final = fit_pick_quantile_members(q_final_frame, 0.9)
    q10_final["cb"].save_model(CATBOOST_PICK_Q10_PATH)
    q90_final["cb"].save_model(CATBOOST_PICK_Q90_PATH)
    print(f"  quantile members → {CATBOOST_PICK_Q10_PATH}, {CATBOOST_PICK_Q90_PATH}")

    print(f"  success members → {SUCCESS_MODEL_PATH}, {CATBOOST_SUCCESS_PATH}")
    print(f"  grade members   → {DRAFT_GRADE_MODEL_PATH}, {CATBOOST_DRAFT_GRADE_PATH}")
    print(f"  calibrators     → {SUCCESS_CALIBRATED_PATH}, {DRAFT_GRADE_CALIBRATED_PATH}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    metadata = {
        "serve": serve,
        "serve_pick": bool(pick_wins),
        "pick_eval": pick_eval,
        "serve_av": bool(serve_av),
        "av_eval": av_eval,
        "pick_interval": {
            "coverage_target": PICK_INTERVAL_COVERAGE,
            "lo_offset_log": round(final_lo_off, 4),
            "hi_offset_log": round(final_hi_off, 4),
            "conformal_fold": sorted(GRADE_CAL_FOLD),
            "eval": interval_eval,
        },
        "serve_gate": {
            "rule": "serve ensemble iff eval-holdout AUC(ensemble) > AUC(baseline) "
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
        "features": list(SUCCESS_FEATURES),
        "feature_config": "v3 winner '+A+B+C+D | flat | hist' (models/experiments/RESULTS.md): "
                          "base13 + position-normalized measurables + recruiting pedigree + "
                          "years_in_college/early_declare + all-position fs/car production "
                          "composites + SP+; z-scores frozen in models/feature_stats.json",
        "evaluation": {
            "note": "ALL metrics below are from the EVAL phase (train 2000-2017 + 63 seed "
                    "rows w=5, calibration 2018, holdout 2019-2020, z-reference 2000-2018) — "
                    "the frozen split every experiment in RESULTS.md is measured on. The "
                    "final production fit trains on all classes and has no untouched "
                    "holdout; its expected forward accuracy is the RESULTS.md forward-split "
                    "number (~0.44-0.47), not the frozen one.",
            "holdout_years": sorted(TEST_YEARS),
            "splits": split_stats,
            "success": success_eval,
            "draft_grade": grade_eval,
            "previous_production_baseline": {
                "draft_grade": {"accuracy": 0.4093, "macro_f1": 0.4022},
                "success": {"auc": 0.6387, "brier": 0.1626},
                "note": "base13 feature set, train 2010-2017 (models/metadata.json "
                        "prior to the v3 feature promotion)",
            },
            "baseline_notes": "rule_based_baseline = dv_heuristics (the app's fallback), accolade "
                              f"flags 0, NaN production/speed → neutral {BASELINE_NEUTRAL:g}",
        },
        "final_fit": {
            "draft_grade": {
                "member_train_years": [min(FINAL_GRADE_YEARS), max(FINAL_GRADE_YEARS)],
                "calibrator": {"shadow_train_years": [min(GRADE_SHADOW_TRAIN), max(GRADE_SHADOW_TRAIN)],
                               "calibration_fold": sorted(GRADE_CAL_FOLD),
                               "kind": "multinomial_logprob on out-of-fold member-mean probs"},
                "class_weights": g_final["class_weights"],
            },
            "success": {
                "member_train_years": [min(FINAL_SUCCESS_YEARS), max(FINAL_SUCCESS_YEARS)],
                "label_maturity": f"classes ≤ {SUCCESS_LABEL_MATURE_THROUGH} only "
                                  "(≥5 completed NFL seasons; later labels right-censored)",
                "calibrator": {"shadow_train_years": [min(SUCCESS_SHADOW_TRAIN), max(SUCCESS_SHADOW_TRAIN)],
                               "calibration_fold": sorted(SUCCESS_CAL_FOLD),
                               "kind": "platt_logit on out-of-fold member-mean probs"},
                "scale_pos_weight": s_final["scale_pos_weight"],
            },
            "feature_stats": {
                "path": os.path.relpath(FEATURE_STATS_PATH, REPO_ROOT),
                "reference_years": [min(FINAL_REF_YEARS), max(FINAL_REF_YEARS)],
            },
            "seed_rows": {"count": 63, "weight": SEED_ROW_WEIGHT},
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
