#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 experiment — position-group-specific SUCCESS and PICK heads (rolling CV).

Question: does a separate XGB+CatBoost ensemble per position group beat the
single global model that currently serves? Six groups, defined from the
dv_features position flags (which are themselves derived from
_production_group / the position string):

    QB, RB, WRTE (WR+TE), OL, DLLB (DL+EDGE+LB), DB

Rows whose position maps to none of these ("OTHER": K/P/FB/LS/unknown) are
always scored by the global model in every arm.

Arms (all share the SAME global 4-class grade head — this experiment touches
only the success and pick heads, so grade accuracy is identical across arms
and the classifier-implied half of the served pick blend is unchanged):

  baseline      global success members + global Platt (Y-1 cal) ;
                global pick regressors.  == production recipe.
  group_platt   global success members, but a SEPARATE Platt calibrator per
                group fitted on that group's Y-1 cal rows (falls back to the
                global Platt when the group has < MIN_CAL_ROWS cal rows or a
                single class).  Pick head unchanged (== baseline), so only the
                success metrics can move.
  group_heads   per-group success members and per-group pick regressors,
                identical hyperparameters, trained on the group's train rows
                (+ that group's seed exemplars).  A group with < MIN_TRAIN_ROWS
                train rows falls back to the global members.  One pooled Platt
                is fitted on all Y-1 cal rows scored by their group model
                (a per-group Platt on 15-20 QB cal rows would be noise).

Fold design (mirrors scripts/rolling_ab_v5.py exactly): for test year Y in
2015..2020, train = 2000..Y-2 (+63 seeds, w=5), cal = Y-1, z-ref = 2000..Y-1.
All z-statistics, group thresholds, and calibrators use only pre-Y rows;
asserted per fold.  Leakage guard: X is tm.SUCCESS_FEATURES, which is
asserted disjoint from tm.FORBIDDEN_FEATURES.

Adoption bar (pre-registered): candidate must beat baseline on the target
metric in >= 4 of 6 folds with mean delta beyond fold-std/2, and be
non-inferior (|mean delta| within noise) on every other served metric.

Writes models/experiments/v6_position_heads_results.json.  Nothing else is
modified.

Usage:  .venv/bin/python scripts/experiment_v6_position_heads.py
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
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import train_models as tm

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_position_heads_results.json")
BASELINE_REF_PATH = os.path.join(REPO_ROOT, "models", "experiments", "rolling_cv_v4.json")

FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]
GROUPS = ["QB", "RB", "WRTE", "OL", "DLLB", "DB"]
MIN_TRAIN_ROWS = 150   # per task brief: smaller group -> global model
MIN_CAL_ROWS = 40      # per-group Platt needs at least this many Y-1 rows (+ both classes)

METRICS = ["success_auc", "success_brier", "grade_acc_raw", "pick_blend_mae",
           "pick_blend_spearman", "pick_blend_top64", "pick_blend_r1_recall_45"]
LOWER_IS_BETTER = {"success_brier", "pick_blend_mae"}
# Fixed noise bands for the non-inferiority check on the OTHER metrics
# (same bands scripts/experiment_v5_traj.py used).
NOISE = {"success_auc": 0.005, "success_brier": 0.002, "grade_acc_raw": 0.01,
         "pick_blend_mae": 1.5, "pick_blend_spearman": 0.01,
         "pick_blend_top64": 0.01, "pick_blend_r1_recall_45": 0.02}


# ── group assignment ──────────────────────────────────────────────────────────

def assign_group(df: pd.DataFrame) -> np.ndarray:
    """Six position groups from the dv_features position flags; 'OTHER' for
    rows outside them (always scored by the global model)."""
    g = np.full(len(df), "OTHER", dtype=object)
    g[df["position_db"].to_numpy() == 1] = "DB"
    g[(df["position_dl"].to_numpy() == 1) | (df["position_lb"].to_numpy() == 1)] = "DLLB"
    g[df["position_ol"].to_numpy() == 1] = "OL"
    g[(df["position_wr"].to_numpy() == 1) | (df["position_te"].to_numpy() == 1)] = "WRTE"
    g[df["position_rb"].to_numpy() == 1] = "RB"
    g[df["position_qb"].to_numpy() == 1] = "QB"
    return g


# ── small helpers ─────────────────────────────────────────────────────────────

def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


def fit_platt(p_uncal: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Same estimator train_models.fit_success_calibrator uses (Platt on the
    logit of the member-mean probability)."""
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(_logit(p_uncal).reshape(-1, 1), y)
    return platt


def apply_platt(platt: LogisticRegression, p_uncal: np.ndarray) -> np.ndarray:
    return platt.predict_proba(_logit(p_uncal).reshape(-1, 1))[:, 1]


def uncal_success(members: dict, X: pd.DataFrame) -> np.ndarray:
    return tm.ensemble_success_probs(members, X, calibrated=False)


def score_arm(test: pd.DataFrame, prob: np.ndarray, P_raw: np.ndarray,
              reg_pick: np.ndarray) -> dict:
    """Serving-equivalent metrics given calibrated success probs, raw grade
    probs, and regressor pick preds for ALL test rows (in test order)."""
    y_s = test.nfl_success.to_numpy()
    y_g = test.draft_grade.to_numpy()
    y_pick = np.exp(tm._pick_target(test))
    m = np.isfinite(y_pick)
    cls_pick = tm.classifier_expected_pick(P_raw[m])
    blend = np.exp(0.5 * (np.log(reg_pick[m]) + np.log(cls_pick)))  # SERVED estimator
    pk = tm.pick_metrics(y_pick[m], blend)
    return {
        "success_auc": round(float(roc_auc_score(y_s, prob)), 4),
        "success_brier": round(float(brier_score_loss(y_s, prob)), 4),
        "grade_acc_raw": round(float(accuracy_score(y_g, P_raw.argmax(axis=1))), 4),
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
    }


def per_group_diag(test: pd.DataFrame, groups: np.ndarray, prob: np.ndarray,
                   reg_pick: np.ndarray, P_raw: np.ndarray) -> dict:
    """Within-group AUC / Brier / pick MAE for one arm (diagnostic only)."""
    out = {}
    y_s = test.nfl_success.to_numpy()
    y_pick = np.exp(tm._pick_target(test))
    for g in GROUPS + ["OTHER"]:
        mk = groups == g
        if mk.sum() < 5:
            continue
        d = {"n": int(mk.sum())}
        if len(np.unique(y_s[mk])) == 2:
            d["auc"] = round(float(roc_auc_score(y_s[mk], prob[mk])), 4)
        d["brier"] = round(float(brier_score_loss(y_s[mk], prob[mk])), 4)
        mm = mk & np.isfinite(y_pick)
        cls_pick = tm.classifier_expected_pick(P_raw[mm])
        blend = np.exp(0.5 * (np.log(reg_pick[mm]) + np.log(cls_pick)))
        drafted = y_pick[mm] < tm.UDFA_PICK
        if drafted.sum() >= 5:
            d["mae_drafted"] = round(float(np.mean(np.abs(y_pick[mm][drafted] - blend[drafted]))), 1)
        out[g] = d
    return out


# ── one fold ──────────────────────────────────────────────────────────────────

def run_fold(y: int, raw: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    tm.EVAL_TRAIN_YEARS = set(range(2000, y - 1))
    tm.CAL_YEARS = {y - 1}
    tm.TEST_YEARS = {y}
    ref_years = set(range(2000, y))
    assert not (ref_years & tm.TEST_YEARS)

    df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))
    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    assert not (set(train.draft_year.unique()) & tm.TEST_YEARS)
    assert not (set(cal.draft_year.unique()) & tm.TEST_YEARS)
    assert train.draft_year.max() < y - 1 and cal.draft_year.unique().tolist() == [y - 1]

    feats = list(tm.SUCCESS_FEATURES)
    assert not (set(feats) & tm.FORBIDDEN_FEATURES), "forbidden feature in X"
    X_cal, X_te = cal[feats], test[feats]
    g_tr, g_cal, g_te = assign_group(train), assign_group(cal), assign_group(test)

    # ── global heads (baseline) ───────────────────────────────────────────
    np.random.seed(tm.SEED)
    s_glob = tm.fit_success_members(train)
    s_glob["calibrator"] = tm.fit_success_calibrator(s_glob, cal, tm.CAL_YEARS)
    platt_glob = s_glob["calibrator"]["model"]
    g_glob = tm.fit_grade_members(train)      # shared by every arm
    p_glob = tm.fit_pick_members(train)

    P_raw = tm.ensemble_grade_probs(g_glob, X_te, calibrated=False)
    p_unc_te_glob = uncal_success(s_glob, X_te)
    p_unc_cal_glob = uncal_success(s_glob, X_cal)
    prob_base = apply_platt(platt_glob, p_unc_te_glob)
    reg_base = tm.ensemble_pick_preds(p_glob, X_te)
    res_base = score_arm(test, prob_base, P_raw, reg_base)

    # ── arm: global members + per-group Platt ─────────────────────────────
    y_cal = cal.nfl_success.to_numpy()
    prob_gp = prob_base.copy()
    platt_used = {}
    for g in GROUPS:
        mc, mt = g_cal == g, g_te == g
        if mc.sum() >= MIN_CAL_ROWS and len(np.unique(y_cal[mc])) == 2:
            pl = fit_platt(p_unc_cal_glob[mc], y_cal[mc])
            prob_gp[mt] = apply_platt(pl, p_unc_te_glob[mt])
            platt_used[g] = {"cal_rows": int(mc.sum()), "used": "group"}
        else:
            platt_used[g] = {"cal_rows": int(mc.sum()), "used": "global_fallback"}
    res_gp = score_arm(test, prob_gp, P_raw, reg_base)

    # ── arm: per-group success + pick members ─────────────────────────────
    p_unc_te_grp = p_unc_te_glob.copy()
    p_unc_cal_grp = p_unc_cal_glob.copy()
    reg_grp = reg_base.copy()
    heads_used = {}
    for g in GROUPS:
        mtr, mc, mt = g_tr == g, g_cal == g, g_te == g
        n_tr = int(mtr.sum())
        if n_tr < MIN_TRAIN_ROWS:
            heads_used[g] = {"train_rows": n_tr, "used": "global_fallback"}
            continue
        sub = train[mtr].reset_index(drop=True)
        np.random.seed(tm.SEED)
        s_g = tm.fit_success_members(sub)
        p_g = tm.fit_pick_members(sub)
        p_unc_te_grp[mt] = uncal_success(s_g, X_te[mt])
        p_unc_cal_grp[mc] = uncal_success(s_g, X_cal[mc])
        reg_grp[mt] = tm.ensemble_pick_preds(p_g, X_te[mt])
        heads_used[g] = {"train_rows": n_tr, "pick_rows": p_g["n_rows"],
                         "test_rows": int(mt.sum()), "used": "group",
                         "scale_pos_weight": s_g["scale_pos_weight"]}
    platt_pooled = fit_platt(p_unc_cal_grp, y_cal)   # Y-1 rows only
    prob_gh = apply_platt(platt_pooled, p_unc_te_grp)
    res_gh = score_arm(test, prob_gh, P_raw, reg_grp)

    # mixed views come for free (heads are independent): success-only / pick-only
    res_gh_success_only = score_arm(test, prob_gh, P_raw, reg_base)
    res_gh_pick_only = score_arm(test, prob_base, P_raw, reg_grp)

    diag = {
        "baseline": per_group_diag(test, g_te, prob_base, reg_base, P_raw),
        "group_platt": per_group_diag(test, g_te, prob_gp, reg_base, P_raw),
        "group_heads": per_group_diag(test, g_te, prob_gh, reg_grp, P_raw),
    }

    return {
        "test_year": y,
        "train_years": [2000, y - 2], "cal_year": y - 1, "ref_years": [2000, y - 1],
        "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test)),
        "test_group_sizes": {g: int((g_te == g).sum()) for g in GROUPS + ["OTHER"]},
        "group_platt_calibrators": platt_used,
        "group_heads_fits": heads_used,
        "arms": {"baseline": res_base, "group_platt": res_gp, "group_heads": res_gh,
                 "group_heads_success_only": res_gh_success_only,
                 "group_heads_pick_only": res_gh_pick_only},
        "per_group_diag": diag,
    }


# ── summary / verdict ─────────────────────────────────────────────────────────

def summarize(folds: list, arm: str) -> dict:
    out = {}
    for m in METRICS:
        base = [f["arms"]["baseline"][m] for f in folds]
        cand = [f["arms"][arm][m] for f in folds]
        ds = [round(c - b, 4) for c, b in zip(cand, base)]
        sign = -1.0 if m in LOWER_IS_BETTER else 1.0
        out[m] = {
            "baseline_per_fold": base, "candidate_per_fold": cand, "delta_per_fold": ds,
            "baseline_mean": round(float(np.mean(base)), 4),
            "candidate_mean": round(float(np.mean(cand)), 4),
            "mean_delta": round(float(np.mean(ds)), 4),
            "std_delta": round(float(np.std(ds, ddof=1)), 4),
            "wins": int(sum(1 for d in ds if sign * d > 0)),
            "ties": int(sum(1 for d in ds if d == 0)),
            "fold_std_baseline": round(float(np.std(base, ddof=1)), 4),
        }
    return out


def verdict(summary: dict, targets: list) -> dict:
    """Pre-registered bar: on each TARGET metric win >= 4/6 folds AND mean
    delta beyond fold-std(baseline)/2; on every other served metric the mean
    delta must be within the noise band (non-inferior)."""
    target_checks = {}
    for m in targets:
        s = summary[m]
        sign = -1.0 if m in LOWER_IS_BETTER else 1.0
        improved = sign * s["mean_delta"]
        bar = s["fold_std_baseline"] / 2.0
        target_checks[m] = {
            "wins": s["wins"], "wins_ok": s["wins"] >= 4,
            "mean_delta": s["mean_delta"], "half_fold_std": round(bar, 4),
            "mean_delta_ok": improved > bar,
            "pass": bool(s["wins"] >= 4 and improved > bar),
        }
    others = {}
    for m in METRICS:
        if m in targets:
            continue
        s = summary[m]
        sign = -1.0 if m in LOWER_IS_BETTER else 1.0
        worse_by = -sign * s["mean_delta"]
        others[m] = {"mean_delta": s["mean_delta"], "noise": NOISE[m],
                     "non_inferior": bool(worse_by <= NOISE[m])}
    any_target_pass = any(v["pass"] for v in target_checks.values())
    all_targets_pass = all(v["pass"] for v in target_checks.values())
    non_inf = all(v["non_inferior"] for v in others.values())
    return {"targets": target_checks, "others": others,
            "any_target_passes_bar": bool(any_target_pass),
            "all_targets_pass_bar": bool(all_targets_pass),
            "non_inferior_on_others": bool(non_inf),
            "adopt": bool(any_target_pass and non_inf)}


def main() -> int:
    t0 = time.time()
    np.random.seed(tm.SEED)
    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    print(f"rows={len(raw)}  seeds={len(seeds)}  features={len(tm.SUCCESS_FEATURES)}", flush=True)

    folds = []
    for y in FOLDS:
        tf = time.time()
        f = run_fold(y, raw, seeds)
        folds.append(f)
        b, gp, gh = f["arms"]["baseline"], f["arms"]["group_platt"], f["arms"]["group_heads"]
        print(f"fold {y} ({time.time()-tf:.0f}s): "
              f"AUC base {b['success_auc']:.4f} | gPlatt {gp['success_auc']:.4f} | gHeads {gh['success_auc']:.4f}  "
              f"Brier {b['success_brier']:.4f} | {gp['success_brier']:.4f} | {gh['success_brier']:.4f}  "
              f"MAE {b['pick_blend_mae']:.1f} | {gh['pick_blend_mae']:.1f}  "
              f"rho {b['pick_blend_spearman']:.4f} | {gh['pick_blend_spearman']:.4f}",
              flush=True)

    arms = ["group_platt", "group_heads", "group_heads_success_only", "group_heads_pick_only"]
    summary = {a: summarize(folds, a) for a in arms}
    verdicts = {
        # group_platt only touches success calibration -> success metrics are the target
        "group_platt": verdict(summary["group_platt"], ["success_auc", "success_brier"]),
        "group_heads": verdict(summary["group_heads"],
                               ["success_auc", "success_brier", "pick_blend_mae", "pick_blend_spearman"]),
        "group_heads_success_only": verdict(summary["group_heads_success_only"],
                                            ["success_auc", "success_brier"]),
        "group_heads_pick_only": verdict(summary["group_heads_pick_only"],
                                         ["pick_blend_mae", "pick_blend_spearman"]),
    }

    # pooled per-group diagnostics (mean over folds of within-group metrics)
    pooled = {}
    for arm in ("baseline", "group_platt", "group_heads"):
        pooled[arm] = {}
        for g in GROUPS + ["OTHER"]:
            aucs = [f["per_group_diag"][arm][g]["auc"] for f in folds
                    if g in f["per_group_diag"][arm] and "auc" in f["per_group_diag"][arm][g]]
            maes = [f["per_group_diag"][arm][g]["mae_drafted"] for f in folds
                    if g in f["per_group_diag"][arm] and "mae_drafted" in f["per_group_diag"][arm][g]]
            briers = [f["per_group_diag"][arm][g]["brier"] for f in folds
                      if g in f["per_group_diag"][arm]]
            pooled[arm][g] = {
                "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
                "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
                "mean_mae_drafted": round(float(np.mean(maes)), 1) if maes else None,
                "n_folds": len(briers),
            }

    baseline_ref = None
    if os.path.exists(BASELINE_REF_PATH):
        with open(BASELINE_REF_PATH) as fh:
            baseline_ref = json.load(fh).get("summary")

    out = {
        "experiment": "v6_position_heads (per-position-group success + pick heads)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "seed": tm.SEED,
        "design": {
            "folds": FOLDS,
            "train": "2000..Y-2 draft classes + 63 seed rows (w=5)",
            "calibrate": "Y-1 only",
            "z_ref": "2000..Y-1 recomputed per fold (never includes test year)",
            "groups": GROUPS + ["OTHER -> always global"],
            "min_train_rows_for_group_head": MIN_TRAIN_ROWS,
            "min_cal_rows_for_group_platt": MIN_CAL_ROWS,
            "grade_head": "global in every arm (only success/pick heads vary)",
            "arms": {
                "baseline": "global members + global Platt; global pick regressors (production recipe)",
                "group_platt": "global members; per-group Platt on Y-1 cal rows (fallback global)",
                "group_heads": "per-group success + pick members; pooled Platt over Y-1 cal rows scored by group models",
                "group_heads_success_only": "group success head + baseline pick head (derived, no extra fit)",
                "group_heads_pick_only": "baseline success head + group pick head (derived, no extra fit)",
            },
            "leakage_guarantee": (
                "per fold: train years < Y-1, cal == Y-1, z-ref < Y (asserted); group "
                "membership is a position flag (an X feature, never an outcome); "
                "MIN_TRAIN_ROWS / MIN_CAL_ROWS fixed a priori; X == tm.SUCCESS_FEATURES "
                "asserted disjoint from tm.FORBIDDEN_FEATURES; no tuning performed."
            ),
            "adoption_bar": "target metric: >=4/6 fold wins AND mean delta > baseline fold-std/2; "
                            "other served metrics: mean delta within noise band",
            "noise_bands": NOISE,
        },
        "reference_rolling_cv_v4_summary": baseline_ref,
        "folds": folds,
        "summary": summary,
        "per_group_pooled": pooled,
        "verdicts": verdicts,
        "runtime_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}  ({out['runtime_sec']}s)")
    for a in arms:
        print(f"\n== {a} ==")
        for m in METRICS:
            s = summary[a][m]
            print(f"  {m:26s} mean d {s['mean_delta']:+.4f} ± {s['std_delta']:.4f}  wins {s['wins']}/6  "
                  f"(base {s['baseline_mean']:.4f} -> {s['candidate_mean']:.4f})")
        print("  verdict:", json.dumps({k: verdicts[a][k] for k in
                                       ("any_target_passes_bar", "non_inferior_on_others", "adopt")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
