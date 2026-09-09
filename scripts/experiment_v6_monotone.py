#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 experiment — monotonic constraints as a bias/variance lever.

Rolling-origin A/B (pattern: scripts/rolling_ab_v5.py) of the production
recipe vs the same recipe with XGB + CatBoost `monotone_constraints` on the
SUCCESS and PICK heads. The 4-class grade head is left unconstrained and is
fit ONCE per fold and shared by every arm (it feeds the served 50/50 pick
blend, so it must be identical across arms for the pick comparison to be
about the pick head).

Constrained features (direction stated for the SUCCESS head, where a higher
output is better):
    consensus_logrank    -   lower rank = better
    rec_rating           +
    rec_stars            +
    rec_ranking          -
    production_score     +
    prod_fs_z            +
    prod_car_z           +
    sp_rating            +
    combine_speed_score  +
The PICK head regresses log(overall pick) where LOWER is better, so every
sign is FLIPPED for that head. The constraint vector is built in
dv_features.SUCCESS_FEATURES order (0 = unconstrained) so it lines up with
the X columns train_models._xy / fit_pick_members hand the learners.

NaN handling (verified on xgboost 3.2.0 / catboost 1.2.7):
  * XGBoost — the constraint binds only the ordering of non-missing values;
    a NaN goes to the learned default branch and is NOT ordered against the
    rest (a NaN row can land anywhere between the min and max response).
  * CatBoost — a NaN is quantised as an extreme border (global default
    nan_mode="Min", i.e. treated as the SMALLEST value). Under a decreasing
    constraint that means a missing consensus_logrank / rec_ranking is
    treated as the BEST rank — the opposite of what "unranked" means. Arm
    "mono_nanworst" therefore pins NaN to the WORST end per feature via
    per_float_feature_quantization (nan_mode=Max for decreasing features,
    Min for increasing ones); arm "mono" keeps the library default so the
    two effects can be separated.

Arms per fold:
    base            unconstrained (production recipe, re-fit here)
    mono            constrained, library-default NaN placement
    mono_nanworst   constrained, CatBoost NaN pinned to the worst end

Leakage guarantees (same as rolling_ab_v5.py): per fold Y the z-stats are
computed from draft years 2000..Y-1 only (stats_from_ref), train is
2000..Y-2 + the 63 seed rows, Platt calibration on Y-1, and the test frame
is Y only — asserted per fold. No FORBIDDEN_FEATURES enter X (asserted).
Constraint directions are fixed a priori (not tuned on any fold).

Writes models/experiments/v6_monotone_results.json. Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_v6_monotone.py
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
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import train_models as tm
from dv_features import SUCCESS_FEATURES as DV_FEATURES

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_monotone_results.json")
FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]

# direction for the SUCCESS head (higher output = better prospect)
SUCCESS_DIRECTIONS = {
    "consensus_logrank": -1,
    "rec_rating": +1,
    "rec_stars": +1,
    "rec_ranking": -1,
    "production_score": +1,
    "prod_fs_z": +1,
    "prod_car_z": +1,
    "sp_rating": +1,
    "combine_speed_score": +1,
}

# served metrics (rolling_ab_v5 list) + regressor-only diagnostics
SERVED_METRICS = ["success_auc", "success_brier", "grade_acc_raw", "pick_blend_mae",
                  "pick_blend_spearman", "pick_blend_top64", "pick_blend_r1_recall_45"]
DIAG_METRICS = ["pick_reg_mae", "pick_reg_spearman", "pick_reg_top64",
                "success_top64_spearman"]
LOWER_IS_BETTER = {"success_brier", "pick_blend_mae", "pick_reg_mae"}
# where the constraints are expected to bite (pre-registered targets)
TARGET_METRICS = ["success_auc", "pick_blend_mae", "pick_blend_top64"]

# Baseline rolling-CV fold stds from models/experiments/rolling_cv_v4.json
# (post seed-fix, production recipe) — the "fold std" of the adoption bar.
BASELINE_FOLD_STD = {"success_auc": 0.0265, "grade_acc_raw": 0.0145,
                     "pick_blend_mae": 1.46, "pick_blend_spearman": 0.0181}


def constraint_vector(features: list, sign: int) -> list:
    """Monotone vector in `features` order; sign=+1 for the success head,
    -1 for the pick head (log-pick, lower is better)."""
    return [sign * SUCCESS_DIRECTIONS.get(f, 0) for f in features]


def cb_nan_worst(features: list, vec: list) -> list:
    """per_float_feature_quantization entries pinning NaN to the WORST end of
    each constrained feature: increasing feature -> NaN=Min, decreasing ->
    NaN=Max. Unconstrained features keep the global default (Min)."""
    out = []
    for i, d in enumerate(vec):
        if d > 0:
            out.append(f"{i}:nan_mode=Min")
        elif d < 0:
            out.append(f"{i}:nan_mode=Max")
    return out


# ── constrained fits (hyper-parameters copied verbatim from train_models) ─────

def fit_success_members_mono(train, vec, nan_worst):
    X_tr, y_tr, w_tr = tm._xy(train, "nfl_success")
    spw = float(w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-9))
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
        scale_pos_weight=spw, random_state=tm.SEED, eval_metric="logloss",
        monotone_constraints=tuple(vec),
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    extra = {"per_float_feature_quantization": cb_nan_worst(list(X_tr.columns), vec)} if nan_worst else {}
    cb_m = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: spw},
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
        monotone_constraints=list(vec), **extra,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "scale_pos_weight": round(spw, 4)}


def fit_pick_members_mono(train, vec, nan_worst):
    y = tm._pick_target(train)
    m = np.isfinite(y)
    X_tr, y_tr = train[tm.SUCCESS_FEATURES][m], y[m]
    w_tr = train.sample_weight.to_numpy()[m]
    xgb_m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=tm.SEED, objective="reg:squarederror",
        monotone_constraints=tuple(vec),
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    extra = {"per_float_feature_quantization": cb_nan_worst(list(X_tr.columns), vec)} if nan_worst else {}
    cb_m = CatBoostRegressor(
        iterations=400, depth=5, learning_rate=0.05, loss_function="RMSE",
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
        monotone_constraints=list(vec), **extra,
    )
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "n_rows": int(m.sum())}


# ── scoring (mirrors experiment_v5_traj.run_variant, grade head shared) ───────

def score_arm(features, s, p, P_raw, test):
    X = test[features]
    y_s = test.nfl_success.to_numpy()
    y_g = test.draft_grade.to_numpy()
    prob = tm.ensemble_success_probs(s, X, calibrated=True)

    y_pick = np.exp(tm._pick_target(test))
    m = np.isfinite(y_pick)
    reg = tm.ensemble_pick_preds(p, X[m])
    cls_pick = tm.classifier_expected_pick(P_raw[m])
    blend = np.exp(0.5 * (np.log(reg) + np.log(cls_pick)))  # SERVED estimator

    pk = tm.pick_metrics(y_pick[m], blend)
    pr = tm.pick_metrics(y_pick[m], reg)
    # success head at the top of the board: does the calibrated success prob
    # order the eventual top-64 picks the way the draft did?
    yp = y_pick[m]
    t64 = yp <= 64
    from scipy.stats import spearmanr
    s_top64 = spearmanr(yp[t64], -prob[m][t64]).statistic if t64.sum() >= 20 else np.nan
    return {
        "n_features": len(features),
        "success_auc": round(float(roc_auc_score(y_s, prob)), 4),
        "success_brier": round(float(brier_score_loss(y_s, prob)), 4),
        "grade_acc_raw": round(float(accuracy_score(y_g, P_raw.argmax(axis=1))), 4),
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
        "pick_reg_mae": pr["mae_picks_drafted"],
        "pick_reg_spearman": pr["spearman_all"],
        "pick_reg_top64": pr["spearman_top64"],
        "success_top64_spearman": round(float(s_top64), 4),
    }


def monotonicity_violations(s, p, test, features):
    """Sanity probe: sweep consensus_logrank across its observed range on the
    test rows (others held fixed) and count rows where the success prob is
    not non-increasing / the regressor pick not non-decreasing."""
    X = test[features].copy()
    grid = np.linspace(np.log(1), np.log(400), 12)
    ps, pk = [], []
    for g in grid:
        Xg = X.copy()
        Xg["consensus_logrank"] = g
        ps.append(tm.ensemble_success_probs(s, Xg, calibrated=False))
        pk.append(np.log(tm.ensemble_pick_preds(p, Xg)))
    ps, pk = np.array(ps), np.array(pk)
    viol_s = float(np.mean((np.diff(ps, axis=0) > 1e-6).any(axis=0)))
    viol_p = float(np.mean((np.diff(pk, axis=0) < -1e-6).any(axis=0)))
    return {"success_frac_rows_nonmonotone": round(viol_s, 4),
            "pick_reg_frac_rows_nonmonotone": round(viol_p, 4)}


def main() -> int:
    t0 = time.time()
    feats = list(DV_FEATURES)
    assert feats == list(tm.SUCCESS_FEATURES), "feature order drift vs train_models"
    assert not (set(feats) & tm.FORBIDDEN_FEATURES), "forbidden feature in X"
    missing = [f for f in SUCCESS_DIRECTIONS if f not in feats]
    assert not missing, f"constrained features not in SUCCESS_FEATURES: {missing}"
    vec_s = constraint_vector(feats, +1)
    vec_p = constraint_vector(feats, -1)
    print("success constraint vector:", dict(zip(feats, vec_s)))
    print("pick    constraint vector:", dict(zip(feats, vec_p)))
    print("catboost NaN-worst quantization (success head):", cb_nan_worst(feats, vec_s))

    raw0 = tm.load_raw_rows()
    seeds = tm.seed_rows()
    arms = {"base": None, "mono": False, "mono_nanworst": True}

    folds = []
    for y in FOLDS:
        tf = time.time()
        tm.EVAL_TRAIN_YEARS = set(range(2000, y - 1))
        tm.CAL_YEARS = {y - 1}
        tm.TEST_YEARS = {y}
        ref_years = set(range(2000, y))
        assert not (ref_years & tm.TEST_YEARS)
        df = tm.apply_z(raw0, tm.stats_from_ref(raw0, ref_years))

        train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                          ignore_index=True)
        cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
        test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
        assert not (set(train.draft_year.unique()) & tm.TEST_YEARS)
        assert not (set(cal.draft_year.unique()) & tm.TEST_YEARS)
        assert set(test.draft_year.unique()) == tm.TEST_YEARS

        np.random.seed(tm.SEED)
        g = tm.fit_grade_members(train)          # shared, unconstrained
        P_raw = tm.ensemble_grade_probs(g, test[feats], calibrated=False)

        res, probes = {}, {}
        for arm, nan_worst in arms.items():
            np.random.seed(tm.SEED)
            if arm == "base":
                s = tm.fit_success_members(train)
                p = tm.fit_pick_members(train)
            else:
                s = fit_success_members_mono(train, vec_s, nan_worst)
                p = fit_pick_members_mono(train, vec_p, nan_worst)
            s["calibrator"] = tm.fit_success_calibrator(s, cal, tm.CAL_YEARS)
            res[arm] = score_arm(feats, s, p, P_raw, test)
            probes[arm] = monotonicity_violations(s, p, test, feats)

        deltas = {arm: {m: round(res[arm][m] - res["base"][m], 4)
                        for m in SERVED_METRICS + DIAG_METRICS}
                  for arm in arms if arm != "base"}
        folds.append({"test_year": y, "n_train": int(len(train)), "n_test": int(len(test)),
                      "results": res, "delta": deltas, "monotonicity_probe": probes,
                      "seconds": round(time.time() - tf, 1)})
        for arm, d in deltas.items():
            print(f"fold {y} [{arm:>13}]: dAUC {d['success_auc']:+.4f} "
                  f"dBrier {d['success_brier']:+.4f} dMAE {d['pick_blend_mae']:+.1f} "
                  f"dRho {d['pick_blend_spearman']:+.4f} dTop64 {d['pick_blend_top64']:+.4f} "
                  f"dRegTop64 {d['pick_reg_top64']:+.4f} dR1 {d['pick_blend_r1_recall_45']:+.4f}",
                  flush=True)
        print(f"  probe base={probes['base']} mono={probes['mono']}  "
              f"({folds[-1]['seconds']}s)", flush=True)

    # ── summary + adoption verdict ────────────────────────────────────────
    summary, verdicts = {}, {}
    for arm in [a for a in arms if a != "base"]:
        summary[arm] = {}
        for m in SERVED_METRICS + DIAG_METRICS:
            ds = [f["delta"][arm][m] for f in folds]
            base_vals = [f["results"]["base"][m] for f in folds]
            arm_vals = [f["results"][arm][m] for f in folds]
            good = (lambda d: d < 0) if m in LOWER_IS_BETTER else (lambda d: d > 0)
            summary[arm][m] = {
                "base_mean": round(float(np.mean(base_vals)), 4),
                "base_std": round(float(np.std(base_vals, ddof=1)), 4),
                "arm_mean": round(float(np.mean(arm_vals)), 4),
                "arm_std": round(float(np.std(arm_vals, ddof=1)), 4),
                "mean_delta": round(float(np.mean(ds)), 4),
                "std_delta": round(float(np.std(ds, ddof=1)), 4),
                "wins": int(sum(1 for d in ds if good(d))),
                "per_fold_delta": ds,
                "per_fold_base": base_vals,
                "per_fold_arm": arm_vals,
            }
        # Adoption bar: target metric wins >=4/6 AND |mean delta| > fold std/2
        # (fold std = std of the baseline metric across folds, as quoted in
        # the brief; the paired delta-std/2 check is reported alongside),
        # AND non-inferior (mean delta within paired noise or favourable) on
        # every other served metric.
        target_pass = {}
        for m in TARGET_METRICS:
            st = summary[arm][m]
            sign = -1 if m in LOWER_IS_BETTER else 1
            favorable = sign * st["mean_delta"]
            fold_std = BASELINE_FOLD_STD.get(m, st["base_std"])
            target_pass[m] = {
                "wins_ge_4": st["wins"] >= 4,
                "beyond_fold_std_half": bool(favorable > fold_std / 2),
                "beyond_delta_std_half": bool(favorable > st["std_delta"] / 2),
                "pass": bool(st["wins"] >= 4 and favorable > fold_std / 2),
            }
        non_inf = {}
        for m in SERVED_METRICS:
            st = summary[arm][m]
            sign = -1 if m in LOWER_IS_BETTER else 1
            favorable = sign * st["mean_delta"]
            non_inf[m] = bool(favorable >= -st["std_delta"] / 2)
        any_target = any(v["pass"] for v in target_pass.values())
        verdicts[arm] = {
            "target_metrics": target_pass,
            "non_inferior_all_served": non_inf,
            "adopt": bool(any_target and all(non_inf.values())),
        }

    out = {
        "experiment": "v6_monotone (monotone constraints on success + pick heads)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "versions": {"xgboost": xgb.__version__,
                     "catboost": __import__("catboost").__version__},
        "design": {
            "folds": FOLDS,
            "train": "2000..Y-2 + 63 seed rows (w=5)", "cal": "Y-1 (success Platt)",
            "z_ref": "2000..Y-1 per fold (never includes test year)", "test": "Y only",
            "grade_head": "unconstrained, fit once per fold, shared by all arms",
            "arms": {"base": "production recipe re-fit",
                     "mono": "monotone constraints, library-default NaN placement "
                             "(XGB: learned default branch; CatBoost: NaN=Min)",
                     "mono_nanworst": "monotone constraints + CatBoost NaN pinned to "
                                      "the worst end per feature "
                                      "(per_float_feature_quantization nan_mode)"},
            "constraint_directions_success_head": SUCCESS_DIRECTIONS,
            "pick_head": "signs flipped (target log pick, lower is better)",
            "success_vector": vec_s, "pick_vector": vec_p, "feature_order": feats,
            "adoption_bar": "target metric wins >=4/6 and mean delta beyond baseline "
                            "fold std/2; non-inferior (mean delta >= -delta_std/2) on "
                            "every served metric",
            "target_metrics": TARGET_METRICS,
            "baseline_fold_std_used": BASELINE_FOLD_STD,
        },
        "folds": folds,
        "summary": summary,
        "verdicts": verdicts,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}  ({out['runtime_seconds']}s)")
    for arm in summary:
        print(f"\n== {arm} ==")
        for m in SERVED_METRICS + DIAG_METRICS:
            st = summary[arm][m]
            print(f"  {m:<26} base {st['base_mean']:.4f}±{st['base_std']:.4f}  "
                  f"arm {st['arm_mean']:.4f}  d {st['mean_delta']:+.4f}±{st['std_delta']:.4f}  "
                  f"wins {st['wins']}/6")
        print("  verdict:", json.dumps(verdicts[arm]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
