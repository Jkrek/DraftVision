#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 stacking experiment — a logistic meta-learner over the served heads,
evaluated on the six rolling-origin folds (test year Y in 2015..2020).

Arms (per fold, same base members, only the success COMBINER differs):

  baseline   : served recipe — arithmetic mean of the XGB+CatBoost success
               members, Platt-calibrated on the cal year (Y-1).
  stack_full : level-1 LogisticRegression on level-0 signals
                 [xgb_logit, cb_logit, grade_logp_0..3 (member-mean 4-class
                  log-probs), pick_logpick (member-mean log-pick),
                  consensus_logrank (median-imputed) + consensus_missing]
               fit on 5-fold OUT-OF-FOLD level-0 predictions of the training
               rows (2000..Y-2 + seeds), C chosen on the cal year (Y-1) by
               log-loss, scored on the test year (Y).
  stack_members_only : same, but only [xgb_logit, cb_logit] -> answers
               "is the stacker just re-weighting the two members?"
  stack_no_consensus : all signals except the consensus pair.
  stack_full_plus_cal: stack_full refit on train-OOF + cal-year level-0 rows
               with the tuned C (uses cal for fitting, like Platt does).

Leakage guarantees (asserted in code):
  * per fold: train years <= Y-2, cal = Y-1, test = Y; z-stats from 2000..Y-1
  * inner 5-fold OOF splits only the training rows — cal/test rows are never
    used to fit any level-0 model that produces their own level-0 features
    (cal/test features come from level-0 models fit on train rows only)
  * C (and the consensus imputation median / scaler) come from train/cal only
  * level-1 inputs are model outputs + consensus_logrank; no FORBIDDEN column

Compute: ~36 three-head fit-sets (6 folds x [1 full + 5 inner]) ~ 4 min.

Writes models/experiments/v6_stacking_results.json. Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_v6_stacking.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import train_models as tm

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_stacking_results.json")
BASELINE_REF_PATH = os.path.join(REPO_ROOT, "models", "experiments", "rolling_cv_v4.json")

FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]
INNER_K = 5
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]

L0_NAMES = ["xgb_logit", "cb_logit",
            "grade_logp_0", "grade_logp_1", "grade_logp_2", "grade_logp_3",
            "pick_logpick", "consensus_logrank", "consensus_missing"]
VARIANT_COLS = {
    "stack_full": L0_NAMES,
    "stack_members_only": ["xgb_logit", "cb_logit"],
    "stack_no_consensus": [c for c in L0_NAMES if not c.startswith("consensus")],
}
PRIMARY = "stack_full"
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_level0(train: pd.DataFrame) -> dict:
    """The three served heads' members, fit exactly as train_models does."""
    np.random.seed(tm.SEED)
    return {"s": tm.fit_success_members(train),
            "g": tm.fit_grade_members(train),
            "p": tm.fit_pick_members(train)}


def level0_frame(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-imputed) level-0 signals for `rows` from fitted level-0 models."""
    X = rows[tm.SUCCESS_FEATURES]
    assert not (set(X.columns) & tm.FORBIDDEN_FEATURES)
    s, g, p = models["s"], models["g"], models["p"]
    out = pd.DataFrame(index=rows.index)
    out["xgb_logit"] = _logit(s["xgb"].predict_proba(X)[:, 1])
    out["cb_logit"] = _logit(s["cb"].predict_proba(X)[:, 1])
    P = tm.ensemble_grade_probs(g, X, calibrated=False)  # member-mean 4-class
    logP = np.log(np.clip(P, EPS, 1.0))
    for c in range(4):
        out[f"grade_logp_{c}"] = logP[:, c]
    out["pick_logpick"] = np.log(tm.ensemble_pick_preds(p, X))  # member-mean log-pick
    cons = rows["consensus_logrank"].to_numpy(dtype=float)
    out["consensus_logrank"] = cons  # imputed later with a TRAIN-only median
    out["consensus_missing"] = (~np.isfinite(cons)).astype(float)
    return out


def oof_level0(train: pd.DataFrame) -> pd.DataFrame:
    """5-fold out-of-fold level-0 signals for every training row."""
    skf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=tm.SEED)
    parts = []
    y = train.nfl_success.to_numpy()
    for k, (i_tr, i_ho) in enumerate(skf.split(train, y)):
        m = fit_level0(train.iloc[i_tr])
        parts.append(level0_frame(m, train.iloc[i_ho]))
    oof = pd.concat(parts).loc[train.index]
    assert len(oof) == len(train) and oof.index.equals(train.index)
    return oof


class Level1:
    """Median-impute consensus (train-only median) -> StandardScaler ->
    LogisticRegression(C). Coefficients reported on the standardized scale."""

    def __init__(self, cols: list, C: float, impute_median: float):
        self.cols, self.C, self.med = cols, C, impute_median
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, solver="lbfgs", max_iter=5000)

    def _mat(self, F: pd.DataFrame) -> np.ndarray:
        M = F[self.cols].copy()
        if "consensus_logrank" in M.columns:
            M["consensus_logrank"] = M["consensus_logrank"].fillna(self.med)
        M = M.to_numpy(dtype=float)
        assert np.isfinite(M).all()
        return M

    def fit(self, F, y, w=None):
        Z = self.scaler.fit_transform(self._mat(F))
        self.lr.fit(Z, y, sample_weight=w)
        return self

    def predict_proba(self, F) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(self._mat(F)))[:, 1]

    def coefs(self) -> dict:
        d = {c: round(float(b), 4) for c, b in zip(self.cols, self.lr.coef_[0])}
        d["intercept"] = round(float(self.lr.intercept_[0]), 4)
        return d


def score(y, p) -> dict:
    return {"success_auc": round(float(roc_auc_score(y, p)), 4),
            "success_brier": round(float(brier_score_loss(y, p)), 4),
            "success_logloss": round(float(log_loss(y, np.clip(p, EPS, 1 - EPS))), 4)}


def run_fold(y_test: int, raw: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    t0 = time.time()
    tm.EVAL_TRAIN_YEARS = set(range(2000, y_test - 1))
    tm.CAL_YEARS = {y_test - 1}
    tm.TEST_YEARS = {y_test}
    ref_years = set(range(2000, y_test))
    assert max(ref_years) < y_test and max(tm.EVAL_TRAIN_YEARS) == y_test - 2

    df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))
    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    for part in (train, cal):
        assert not (set(part.draft_year.unique()) & tm.TEST_YEARS)
    assert not (set(train.draft_year.unique()) & tm.CAL_YEARS)

    y_tr, w_tr = train.nfl_success.to_numpy(), train.sample_weight.to_numpy()
    y_cal, y_te = cal.nfl_success.to_numpy(), test.nfl_success.to_numpy()

    # ── level-0: full-train models (shared by every arm) + inner OOF ──────────
    full = fit_level0(train)
    F_cal, F_te = level0_frame(full, cal), level0_frame(full, test)
    F_oof = oof_level0(train)
    cons_med = float(np.nanmedian(F_oof["consensus_logrank"]))  # train-only
    if not np.isfinite(cons_med):
        cons_med = float(np.log(400.0))

    # ── baseline arm: member-mean + Platt on cal (the served path) ───────────
    s = full["s"]
    s["calibrator"] = tm.fit_success_calibrator(s, cal, tm.CAL_YEARS)
    p_base = tm.ensemble_success_probs(s, test[tm.SUCCESS_FEATURES], calibrated=True)
    p_base_raw = tm.ensemble_success_probs(s, test[tm.SUCCESS_FEATURES], calibrated=False)
    arms = {"baseline": score(y_te, p_base),
            "baseline_uncalibrated_mean": score(y_te, p_base_raw)}

    # ── stacking arms: C tuned on cal ONLY, fit on train OOF, scored on test ─
    coefs, chosen_C, cal_curves = {}, {}, {}
    for name, cols in VARIANT_COLS.items():
        curve = {}
        for C in C_GRID:
            l1 = Level1(cols, C, cons_med).fit(F_oof, y_tr, w_tr)
            curve[C] = float(log_loss(y_cal, np.clip(l1.predict_proba(F_cal), EPS, 1 - EPS)))
        C_best = min(curve, key=curve.get)
        l1 = Level1(cols, C_best, cons_med).fit(F_oof, y_tr, w_tr)
        arms[name] = score(y_te, l1.predict_proba(F_te))
        coefs[name], chosen_C[name] = l1.coefs(), C_best
        cal_curves[name] = {str(k): round(v, 4) for k, v in curve.items()}

    # secondary: full stack refit on train-OOF + cal rows (cal rows' level-0
    # features come from full-train models, so they are out-of-sample too)
    C_best = chosen_C[PRIMARY]
    F_plus = pd.concat([F_oof, F_cal], ignore_index=True)
    y_plus = np.concatenate([y_tr, y_cal])
    w_plus = np.concatenate([w_tr, cal.sample_weight.to_numpy()])
    l1 = Level1(VARIANT_COLS[PRIMARY], C_best, cons_med).fit(F_plus, y_plus, w_plus)
    arms["stack_full_plus_cal"] = score(y_te, l1.predict_proba(F_te))
    coefs["stack_full_plus_cal"] = l1.coefs()
    chosen_C["stack_full_plus_cal"] = C_best

    # OOF-vs-full distribution shift diagnostic (the classic stacking caveat)
    shift = {c: {"oof_train_mean": round(float(F_oof[c].mean()), 3),
                 "test_mean": round(float(F_te[c].mean()), 3)}
             for c in ("xgb_logit", "cb_logit", "pick_logpick")}

    deltas = {name: {m: round(arms[name][m] - arms["baseline"][m], 4)
                     for m in ("success_auc", "success_brier", "success_logloss")}
              for name in arms if name != "baseline"}
    return {
        "test_year": y_test,
        "train_years": [2000, y_test - 2], "cal_year": y_test - 1,
        "ref_years": [2000, y_test - 1],
        "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test)),
        "train_consensus_coverage": round(float(np.isfinite(
            train.consensus_logrank.to_numpy(dtype=float)).mean()), 4),
        "consensus_impute_median": round(cons_med, 4),
        "arms": arms, "delta_vs_baseline": deltas,
        "chosen_C": chosen_C, "cal_logloss_curve": cal_curves,
        "level1_coefs_standardized": coefs,
        "level0_shift": shift,
        "seconds": round(time.time() - t0, 1),
    }


def summarize(folds: list, arm: str, metric: str) -> dict:
    base = np.array([f["arms"]["baseline"][metric] for f in folds])
    cand = np.array([f["arms"][arm][metric] for f in folds])
    ds = cand - base
    lower_better = metric != "success_auc"
    wins = int(sum((d < 0) if lower_better else (d > 0) for d in ds))
    return {"baseline_per_fold": [round(float(v), 4) for v in base],
            "candidate_per_fold": [round(float(v), 4) for v in cand],
            "delta_per_fold": [round(float(v), 4) for v in ds],
            "baseline_mean": round(float(base.mean()), 4),
            "baseline_std": round(float(base.std(ddof=1)), 4),
            "candidate_mean": round(float(cand.mean()), 4),
            "candidate_std": round(float(cand.std(ddof=1)), 4),
            "mean_delta": round(float(ds.mean()), 4),
            "std_delta": round(float(ds.std(ddof=1)), 4),
            "wins": wins, "n_folds": len(folds)}


def verdict(summary: dict) -> dict:
    """Pre-registered bar: >=4/6 wins on the target metric with mean delta
    beyond std/2 (checked against BOTH the delta std and the baseline fold
    std — both must hold), AND non-inferior on the other metric (mean delta
    inside one delta-std of zero, on the harmful side)."""
    def clears(s, lower_better):
        md = -s["mean_delta"] if lower_better else s["mean_delta"]
        return bool(s["wins"] >= 4 and md > s["std_delta"] / 2
                    and md > s["baseline_std"] / 2)

    def non_inf(s, lower_better):
        md = -s["mean_delta"] if lower_better else s["mean_delta"]
        return bool(md >= -s["std_delta"])

    auc, brier = summary["success_auc"], summary["success_brier"]
    auc_ok, brier_ok = clears(auc, False), clears(brier, True)
    return {
        "auc_clears_bar": auc_ok, "brier_clears_bar": brier_ok,
        "auc_non_inferior": non_inf(auc, False),
        "brier_non_inferior": non_inf(brier, True),
        # grade / pick heads are untouched by a success-only stacker, so the
        # non-inferiority requirement on those served metrics holds trivially.
        "other_heads_unchanged": True,
        "adopt": bool((auc_ok and non_inf(brier, True)) or
                      (brier_ok and non_inf(auc, False))),
    }


def main() -> int:
    t0 = time.time()
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    ref = None
    if os.path.exists(BASELINE_REF_PATH):
        with open(BASELINE_REF_PATH) as fh:
            ref = json.load(fh)

    folds = []
    for y in FOLDS:
        f = run_fold(y, raw, seeds)
        folds.append(f)
        a = f["arms"]
        print(f"fold {y}: base AUC {a['baseline']['success_auc']:.4f} "
              f"Brier {a['baseline']['success_brier']:.4f} | "
              f"stack_full AUC {a['stack_full']['success_auc']:.4f} "
              f"(d{f['delta_vs_baseline']['stack_full']['success_auc']:+.4f}) "
              f"Brier {a['stack_full']['success_brier']:.4f} "
              f"(d{f['delta_vs_baseline']['stack_full']['success_brier']:+.4f}) | "
              f"members_only dAUC {f['delta_vs_baseline']['stack_members_only']['success_auc']:+.4f} | "
              f"C={f['chosen_C']['stack_full']}  {f['seconds']}s", flush=True)
        print("   coefs:", json.dumps(f["level1_coefs_standardized"]["stack_full"]), flush=True)

    arm_names = [n for n in folds[0]["arms"] if n != "baseline"]
    summary = {arm: {m: summarize(folds, arm, m)
                     for m in ("success_auc", "success_brier", "success_logloss")}
               for arm in arm_names}
    verdicts = {arm: verdict(summary[arm]) for arm in arm_names}

    # mean standardized coefficients across folds (sign-stability too)
    coef_summary = {}
    for arm in ("stack_full", "stack_members_only", "stack_no_consensus", "stack_full_plus_cal"):
        keys = folds[0]["level1_coefs_standardized"][arm].keys()
        coef_summary[arm] = {}
        for k in keys:
            vals = [f["level1_coefs_standardized"][arm][k] for f in folds]
            coef_summary[arm][k] = {"mean": round(float(np.mean(vals)), 4),
                                    "std": round(float(np.std(vals, ddof=1)), 4),
                                    "per_fold": vals,
                                    "sign_stable": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals))}

    baseline_ref = None
    if ref:
        baseline_ref = {"path": os.path.relpath(BASELINE_REF_PATH, REPO_ROOT),
                        "success_auc_per_fold": ref["summary"]["success_auc"]["per_fold"],
                        "success_brier_per_fold": ref["summary"]["success_brier"]["per_fold"],
                        "reproduced_here_auc": summary["stack_full"]["success_auc"]["baseline_per_fold"],
                        "reproduced_here_brier": summary["stack_full"]["success_brier"]["baseline_per_fold"]}

    out = {
        "experiment": "v6_stacking (logistic meta-learner over served heads, success target)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(), "seed": tm.SEED,
        "design": {
            "test_fold_years": FOLDS,
            "train": "2000..Y-2 draft classes + 63 seed rows (weight 5)",
            "calibrate": "Y-1: Platt for baseline; C selection (log-loss) for stack arms",
            "z_ref": "2000..Y-1 recomputed per fold (never includes test year)",
            "test": "Y only",
            "level0": L0_NAMES,
            "level0_oof": f"StratifiedKFold(k={INNER_K}, shuffle, seed={tm.SEED}) within train rows only",
            "level1": "StandardScaler + LogisticRegression(lbfgs, L2), sample_weight = row weights",
            "C_grid": C_GRID,
            "consensus_impute": "train-OOF median of finite consensus_logrank + missing indicator",
            "leakage_guarantees": [
                "asserts: no test-year rows in train or cal; no cal-year rows in train",
                "z stats from 2000..Y-1 rows only",
                "inner OOF folds partition train rows only; cal/test level-0 signals "
                "come from models fit on train rows only",
                "C, scaler, and consensus median derived from train/cal only",
                "level-1 inputs are model outputs + consensus_logrank (a served feature); "
                "asserted disjoint from FORBIDDEN_FEATURES",
            ],
            "adoption_bar": "target metric wins >=4/6 folds AND mean delta > std/2 "
                            "(both delta-std and baseline fold-std); non-inferior on the other",
        },
        "primary_arm": PRIMARY,
        "folds": folds, "summary": summary, "verdicts": verdicts,
        "level1_coef_summary": coef_summary,
        "baseline_reference": baseline_ref,
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}  ({out['total_seconds']}s)")
    for arm in arm_names:
        s = summary[arm]
        print(f"{arm:28s} dAUC {s['success_auc']['mean_delta']:+.4f}±{s['success_auc']['std_delta']:.4f} "
              f"wins {s['success_auc']['wins']}/6 | dBrier {s['success_brier']['mean_delta']:+.4f}"
              f"±{s['success_brier']['std_delta']:.4f} wins {s['success_brier']['wins']}/6 | "
              f"adopt={verdicts[arm]['adopt']}")
    print("mean standardized coefs (stack_full):",
          json.dumps({k: v["mean"] for k, v in coef_summary["stack_full"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
