#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 hyperparameter search for the SUCCESS head (XGB + CatBoost members),
rolling-origin CV, with a pre-registered untouched honesty fold.

Design (mirrors scripts/rolling_ab_v5.py fold-wise A/B):
  fold Y in 2015..2020:
    train = 2000..Y-2 draft classes + 63 seed rows (w=5)
    cal   = Y-1 (Platt on member-mean logit — the served calibrator)
    test  = Y
    z-ref = 2000..Y-1 recomputed per fold (never includes the test year)

  SEARCH folds  = 2015..2019   (selection score = mean calibrated AUC)
  HONESTY fold  = 2020         (never used for selection; scored once for
                                the production config and once for the winner)

Search space (sklearn ParameterSampler, seed 42, 24 draws + production config
as trial 0 = 25 trials), sampled INDEPENDENTLY for each member:
  XGB:      max_depth {3,4,5,6}, learning_rate {0.02,0.03,0.05,0.08},
            n_estimators {300,400,600,800}, subsample {0.7,0.85,1.0},
            colsample_bytree {0.7,0.85,1.0}, min_child_weight {1,3,5,10}
  CatBoost: depth {3,4,5,6}, learning_rate {...}, iterations {...},
            subsample {...} (Bernoulli bootstrap), rsm {...}, l2_leaf_reg {1,3,10}

Only the success head is refit per trial. The grade and pick heads do not
consume success-head output (the served pick blend = regressor x 4-class
classifier-implied pick), so grade acc / pick MAE / Spearman are fit ONCE per
fold and reported identically for both arms — they are unaffected by
construction.

Leakage guarantees:
  * FORBIDDEN_FEATURES asserted disjoint from the X columns.
  * z-stats computed from rows with draft_year < test year only.
  * train / cal rows asserted to have draft_year < test year.
  * Hyperparameter selection reads only folds 2015-2019; 2020 is scored
    after the winner is frozen.

Writes models/experiments/v6_hparams_results.json (all trials, per-fold).
Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_v6_hparams.py [--trials 25]
"""

from __future__ import annotations

import argparse
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
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler

import train_models as tm

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v6_hparams_results.json")
SEARCH_FOLDS = [2015, 2016, 2017, 2018, 2019]
HONESTY_FOLD = 2020
ALL_FOLDS = SEARCH_FOLDS + [HONESTY_FOLD]

# The production recipe as it exists in train_models.fit_success_members today.
PROD_CONFIG = {
    "xgb_max_depth": 4, "xgb_learning_rate": 0.05, "xgb_n_estimators": 300,
    "xgb_subsample": 0.85, "xgb_colsample_bytree": 0.85, "xgb_min_child_weight": 3,
    "cb_depth": 5, "cb_learning_rate": 0.05, "cb_iterations": 300,
    "cb_subsample": None, "cb_rsm": None, "cb_l2_leaf_reg": 3,   # CatBoost defaults
}

SPACE = {
    "xgb_max_depth": [3, 4, 5, 6],
    "xgb_learning_rate": [0.02, 0.03, 0.05, 0.08],
    "xgb_n_estimators": [300, 400, 600, 800],
    "xgb_subsample": [0.7, 0.85, 1.0],
    "xgb_colsample_bytree": [0.7, 0.85, 1.0],
    "xgb_min_child_weight": [1, 3, 5, 10],
    "cb_depth": [3, 4, 5, 6],
    "cb_learning_rate": [0.02, 0.03, 0.05, 0.08],
    "cb_iterations": [300, 400, 600, 800],
    "cb_subsample": [0.7, 0.85, 1.0],
    "cb_rsm": [0.7, 0.85, 1.0],
    "cb_l2_leaf_reg": [1, 3, 10],
}


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit_success_members_cfg(train: pd.DataFrame, cfg: dict, prod: bool) -> dict:
    """Same recipe as tm.fit_success_members (spw, gamma, seed, weights) with
    the searched knobs swapped in. prod=True delegates to the real function so
    the baseline arm is byte-for-byte the production fit."""
    if prod:
        return tm.fit_success_members(train)
    X_tr, y_tr, w_tr = tm._xy(train, "nfl_success")
    spw = float(w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-9))
    xgb_m = xgb.XGBClassifier(
        n_estimators=int(cfg["xgb_n_estimators"]), max_depth=int(cfg["xgb_max_depth"]),
        learning_rate=float(cfg["xgb_learning_rate"]),
        subsample=float(cfg["xgb_subsample"]), colsample_bytree=float(cfg["xgb_colsample_bytree"]),
        min_child_weight=int(cfg["xgb_min_child_weight"]), gamma=0.1,
        scale_pos_weight=spw, random_state=tm.SEED, eval_metric="logloss",
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    cb_kwargs = dict(
        iterations=int(cfg["cb_iterations"]), depth=int(cfg["cb_depth"]),
        learning_rate=float(cfg["cb_learning_rate"]),
        l2_leaf_reg=float(cfg["cb_l2_leaf_reg"]),
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: spw},
        random_seed=tm.SEED, verbose=0, allow_writing_files=False,
    )
    if cfg.get("cb_subsample") is not None:
        cb_kwargs["bootstrap_type"] = "Bernoulli"
        cb_kwargs["subsample"] = float(cfg["cb_subsample"])
    if cfg.get("cb_rsm") is not None:
        cb_kwargs["rsm"] = float(cfg["cb_rsm"])
    cb_m = CatBoostClassifier(**cb_kwargs)
    cb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    return {"xgb": xgb_m, "cb": cb_m, "scale_pos_weight": round(spw, 4)}


def score_success(cfg: dict, prod: bool, fold: dict) -> dict:
    np.random.seed(tm.SEED)
    tm.CAL_YEARS = {fold["year"] - 1}
    s = fit_success_members_cfg(fold["train"], cfg, prod)
    s["calibrator"] = tm.fit_success_calibrator(s, fold["cal"], tm.CAL_YEARS)
    X = fold["test"][tm.SUCCESS_FEATURES]
    y = fold["test"].nfl_success.to_numpy()
    p_cal = tm.ensemble_success_probs(s, X, calibrated=True)
    p_raw = tm.ensemble_success_probs(s, X, calibrated=False)
    return {
        "success_auc": round(float(roc_auc_score(y, p_cal)), 4),
        "success_brier": round(float(brier_score_loss(y, p_cal)), 4),
        "success_auc_raw": round(float(roc_auc_score(y, p_raw)), 4),
    }


def score_other_heads(fold: dict) -> dict:
    """Grade + pick heads — independent of the success config; fit once per fold."""
    np.random.seed(tm.SEED)
    train, test = fold["train"], fold["test"]
    g = tm.fit_grade_members(train)
    p = tm.fit_pick_members(train)
    X = test[tm.SUCCESS_FEATURES]
    y_g = test.draft_grade.to_numpy()
    P_raw = tm.ensemble_grade_probs(g, X, calibrated=False)
    y_pick = np.exp(tm._pick_target(test))
    m = np.isfinite(y_pick)
    reg = tm.ensemble_pick_preds(p, X[m])
    cls_pick = tm.classifier_expected_pick(P_raw[m])
    blend = np.exp(0.5 * (np.log(reg) + np.log(cls_pick)))
    pk = tm.pick_metrics(y_pick[m], blend)
    return {
        "grade_acc_raw": round(float(accuracy_score(y_g, P_raw.argmax(axis=1))), 4),
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
    }


# ── Folds ────────────────────────────────────────────────────────────────────

def build_folds(raw: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    folds = {}
    for y in ALL_FOLDS:
        train_years = set(range(2000, y - 1))
        cal_years = {y - 1}
        ref_years = set(range(2000, y))
        assert max(ref_years) < y and max(train_years) < y and max(cal_years) < y
        df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))
        train = pd.concat([df[df.draft_year.isin(train_years)], seeds], ignore_index=True)
        cal = df[df.draft_year.isin(cal_years)].reset_index(drop=True)
        test = df[df.draft_year == y].reset_index(drop=True)
        assert (train.draft_year < y).all() and (cal.draft_year < y).all()
        assert (test.draft_year == y).all() and len(test) > 0
        folds[y] = {"year": y, "train": train, "cal": cal, "test": test,
                    "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test))}
    return folds


def summarize(deltas: list, higher_better: bool) -> dict:
    ds = np.asarray(deltas, dtype=float)
    return {
        "mean_delta": round(float(ds.mean()), 4),
        "std_delta": round(float(ds.std(ddof=1)), 4) if len(ds) > 1 else None,
        "wins": int(sum((d > 0) if higher_better else (d < 0) for d in ds)),
        "per_fold": [round(float(d), 4) for d in ds],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25, help="total trials incl. production")
    args = ap.parse_args()

    t0 = time.time()
    feats = list(tm.SUCCESS_FEATURES)
    leaked = set(feats) & tm.FORBIDDEN_FEATURES
    assert not leaked, f"forbidden features in X: {leaked}"

    raw = tm.load_raw_rows()
    seeds = tm.seed_rows()
    folds = build_folds(raw, seeds)
    print("folds:", {y: (f["n_train"], f["n_cal"], f["n_test"]) for y, f in folds.items()}, flush=True)

    sampler = ParameterSampler(SPACE, n_iter=args.trials - 1, random_state=tm.SEED)
    trials = [{"trial": 0, "name": "production", "prod": True, "config": dict(PROD_CONFIG)}]
    for i, cfg in enumerate(sampler, start=1):
        trials.append({"trial": i, "name": f"trial_{i:02d}", "prod": False,
                       "config": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in cfg.items()}})

    # ── other heads: once per fold (unaffected by success hparams) ──
    other = {}
    for y in ALL_FOLDS:
        other[y] = score_other_heads(folds[y])
        print(f"[other heads] fold {y}: {other[y]}  ({time.time()-t0:.0f}s)", flush=True)

    # ── search: folds 2015-2019 only ──
    def _dump(final: bool = False, extra: dict | None = None):
        out = {
            "experiment": "v6_hparams (success-head XGB+CatBoost hyperparameter search)",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": tm.git_sha(),
            "seed": tm.SEED,
            "n_features": len(feats),
            "design": {
                "search_folds": SEARCH_FOLDS, "honesty_fold": HONESTY_FOLD,
                "train": "2000..Y-2 + 63 seeds (w=5)", "cal": "Y-1 Platt", "test": "Y",
                "z_ref": "2000..Y-1 per fold", "selection_score": "mean calibrated AUC over search folds",
                "space": SPACE, "n_trials": len(trials),
                "other_heads": "grade + pick fit once per fold; independent of success hparams",
            },
            "other_heads_per_fold": {str(y): v for y, v in other.items()},
            "trials": trials,
            "complete": final,
        }
        if extra:
            out.update(extra)
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as fh:
            json.dump(out, fh, indent=2)

    for t in trials:
        t["per_fold"] = {}
        for y in SEARCH_FOLDS:
            t["per_fold"][str(y)] = score_success(t["config"], t["prod"], folds[y])
        aucs = [t["per_fold"][str(y)]["success_auc"] for y in SEARCH_FOLDS]
        briers = [t["per_fold"][str(y)]["success_brier"] for y in SEARCH_FOLDS]
        t["search_mean_auc"] = round(float(np.mean(aucs)), 4)
        t["search_mean_brier"] = round(float(np.mean(briers)), 4)
        print(f"trial {t['trial']:2d} {t['name']:<11} searchAUC {t['search_mean_auc']:.4f} "
              f"searchBrier {t['search_mean_brier']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        _dump()

    prod = trials[0]
    ranked = sorted(trials, key=lambda t: (-t["search_mean_auc"], t["search_mean_brier"]))
    winner = ranked[0]
    print(f"\nWINNER on search folds: {winner['name']} searchAUC {winner['search_mean_auc']:.4f} "
          f"vs production {prod['search_mean_auc']:.4f}", flush=True)
    print("winner config:", json.dumps(winner["config"]), flush=True)

    # Best NON-production trial: if production wins the search outright the
    # winner table is trivially zero, so the best challenger is also carried
    # through the honesty fold to give a real per-fold delta profile.
    challenger = next(t for t in ranked if not t["prod"])
    print(f"best challenger: {challenger['name']} searchAUC {challenger['search_mean_auc']:.4f}", flush=True)

    # ── honesty fold 2020: scored ONLY now, for production + frozen winner (+ challenger) ──
    for t in {id(t): t for t in (prod, winner, challenger)}.values():
        t["per_fold"][str(HONESTY_FOLD)] = score_success(t["config"], t["prod"], folds[HONESTY_FOLD])
    print(f"fold {HONESTY_FOLD} (untouched): production AUC {prod['per_fold'][str(HONESTY_FOLD)]['success_auc']:.4f} "
          f"winner AUC {winner['per_fold'][str(HONESTY_FOLD)]['success_auc']:.4f} "
          f"challenger AUC {challenger['per_fold'][str(HONESTY_FOLD)]['success_auc']:.4f}", flush=True)

    # ── comparison tables, all 6 folds ──
    def compare(cand: dict, label: str) -> list:
        rows = []
        for y in ALL_FOLDS:
            b, w = prod["per_fold"][str(y)], cand["per_fold"][str(y)]
            row = {"test_year": y, "in_search": y in SEARCH_FOLDS,
                   "production": {**b, **other[y]}, label: {**w, **other[y]},
                   "delta": {"success_auc": round(w["success_auc"] - b["success_auc"], 4),
                             "success_brier": round(w["success_brier"] - b["success_brier"], 4),
                             "success_auc_raw": round(w["success_auc_raw"] - b["success_auc_raw"], 4),
                             "grade_acc_raw": 0.0, "pick_blend_mae": 0.0, "pick_blend_spearman": 0.0}}
            rows.append(row)
            print(f"[{label}] fold {y}{'*' if y == HONESTY_FOLD else ' '}: AUC {b['success_auc']:.4f} -> {w['success_auc']:.4f} "
                  f"({row['delta']['success_auc']:+.4f})  Brier {b['success_brier']:.4f} -> {w['success_brier']:.4f} "
                  f"({row['delta']['success_brier']:+.4f})", flush=True)
        return rows

    table = compare(winner, "winner")
    ch_table = compare(challenger, "challenger")
    ch_d_auc = [r["delta"]["success_auc"] for r in ch_table]
    ch_d_brier = [r["delta"]["success_brier"] for r in ch_table]
    challenger_summary = {
        "name": challenger["name"], "config": challenger["config"],
        "success_auc": summarize(ch_d_auc, True),
        "success_brier": summarize(ch_d_brier, False),
        "honesty_fold_2020": {"production": prod["per_fold"][str(HONESTY_FOLD)]["success_auc"],
                              "challenger": challenger["per_fold"][str(HONESTY_FOLD)]["success_auc"],
                              "delta": ch_table[-1]["delta"]["success_auc"]},
        "mean_auc_6fold": round(float(np.mean([r["challenger"]["success_auc"] for r in ch_table])), 4),
        "mean_brier_6fold": round(float(np.mean([r["challenger"]["success_brier"] for r in ch_table])), 4),
    }

    d_auc = [r["delta"]["success_auc"] for r in table]
    d_brier = [r["delta"]["success_brier"] for r in table]
    d_auc_search = [r["delta"]["success_auc"] for r in table if r["in_search"]]
    summary = {
        "success_auc": summarize(d_auc, True),
        "success_brier": summarize(d_brier, False),
        "success_auc_search_folds_only": summarize(d_auc_search, True),
        "success_auc_honesty_fold": {
            "production": prod["per_fold"][str(HONESTY_FOLD)]["success_auc"],
            "winner": winner["per_fold"][str(HONESTY_FOLD)]["success_auc"],
            "delta": table[-1]["delta"]["success_auc"],
        },
        "production_mean": {
            "success_auc": round(float(np.mean([r["production"]["success_auc"] for r in table])), 4),
            "success_auc_std": round(float(np.std([r["production"]["success_auc"] for r in table], ddof=1)), 4),
            "success_brier": round(float(np.mean([r["production"]["success_brier"] for r in table])), 4),
        },
        "winner_mean": {
            "success_auc": round(float(np.mean([r["winner"]["success_auc"] for r in table])), 4),
            "success_auc_std": round(float(np.std([r["winner"]["success_auc"] for r in table], ddof=1)), 4),
            "success_brier": round(float(np.mean([r["winner"]["success_brier"] for r in table])), 4),
        },
        "unaffected_heads_mean": {
            "grade_acc_raw": round(float(np.mean([other[y]["grade_acc_raw"] for y in ALL_FOLDS])), 4),
            "pick_blend_mae": round(float(np.mean([other[y]["pick_blend_mae"] for y in ALL_FOLDS])), 2),
            "pick_blend_spearman": round(float(np.mean([other[y]["pick_blend_spearman"] for y in ALL_FOLDS])), 4),
        },
    }

    # ── pre-registered adoption bar ──
    sa = summary["success_auc"]
    prod_fold_std = summary["production_mean"]["success_auc_std"]
    wins_ok = sa["wins"] >= 4
    beyond_delta_std = sa["std_delta"] is not None and sa["mean_delta"] > sa["std_delta"] / 2
    beyond_prod_std = sa["mean_delta"] > prod_fold_std / 2
    sb = summary["success_brier"]
    brier_noninferior = sb["std_delta"] is None or sb["mean_delta"] <= max(0.002, sb["std_delta"] / 2)
    honesty_ok = summary["success_auc_honesty_fold"]["delta"] >= -0.005
    adopt = bool(wins_ok and beyond_delta_std and brier_noninferior and honesty_ok)
    verdict = {
        "auc_wins_ge_4_of_6": bool(wins_ok),
        "mean_dAUC_beyond_half_delta_std": bool(beyond_delta_std),
        "mean_dAUC_beyond_half_baseline_fold_std": bool(beyond_prod_std),
        "brier_non_inferior": bool(brier_noninferior),
        "honesty_fold_2020_non_inferior": bool(honesty_ok),
        "grade_pick_unaffected": True,
        "adopt": adopt,
        "recommendation": ("ADOPT winner config for the success head" if adopt
                           else "KEEP current production config (depth4/lr0.05/300 XGB, depth5/lr0.05/300 CB)"),
        "note": "selection used folds 2015-2019 across 25 trials; search-fold deltas carry optimistic "
                "selection bias — fold 2020 is the only unbiased point estimate",
    }
    print("\nSUMMARY:", json.dumps(summary, indent=1))
    print("VERDICT:", json.dumps(verdict, indent=1))
    print("BEST CHALLENGER:", json.dumps(challenger_summary, indent=1))
    _dump(final=True, extra={
        "winner": {"name": winner["name"], "trial": winner["trial"], "config": winner["config"]},
        "ranking": [{"trial": t["trial"], "name": t["name"], "search_mean_auc": t["search_mean_auc"],
                     "search_mean_brier": t["search_mean_brier"]} for t in ranked],
        "comparison": table, "summary": summary, "verdict": verdict,
        "best_challenger": {"comparison": ch_table, "summary": challenger_summary},
        "runtime_s": round(time.time() - t0, 1),
    })
    print(f"\nWrote {OUT_PATH}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
