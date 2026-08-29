#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 trajectory-shape experiment — EVAL ONLY, frozen 2019-20 holdout.

Candidate NEW features on top of the 31 v4 features, computed purely from
columns already in the CSV (no API calls):

  A. prod_ascension      = prod_fs_raw / prod_car_raw
       career composite includes the final season, so this is ~(0, 1]:
       ~1.0 = one-year wonder / late breakout, low = early long-term producer.
       NaN when either raw composite is missing or the career raw is <= 0
       (ratio is meaningless against a non-positive denominator).
       Uses ONLY college-career stat columns — no draft-outcome inputs.

  B. prod_per_season_z   = z(prod_car_raw / car_seasons)
       volume per college season, z-scored within the SAME composite position
       group ("_grp", dv_features._composite_group) the other prod features
       use. FOLD-SAFE: the z mean/std are computed from EVAL_REF_YEARS
       (2000-2018) rows only — mirroring train_models.stats_from_ref — so the
       2019-2020 test classes never touch the z statistics.

Harness mirrors scripts/experiment_v4.py: monkeypatch train_models'
SUCCESS_FEATURES and refit all three heads on the frozen eval split for each
variant (v4 baseline, +A, +B, +A+B), then score the serving-equivalent paths:
  success: calibrated member-mean prob  -> AUC / Brier
  grade:   raw member-mean argmax       -> accuracy (the served label)
  pick:    50/50 log-space blend of regressor and classifier-implied pick
           (the SERVED estimator)      -> MAE / Spearman / top64 / R1-recall

Writes models/experiments/v5_traj_results.json. Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_v5_traj.py
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
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import train_models as tm

OUTPUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v5_traj_results.json")

FEAT_A = "prod_ascension"
FEAT_B = "prod_per_season_z"

# v4 served holdout baselines (models/metadata.json / task brief) for reference
V4_RECORDED = {
    "success_auc": 0.8144, "success_brier": 0.1286, "grade_acc_raw": 0.6535,
    "pick_blend_mae": 45.3, "pick_blend_spearman": 0.7962,
    "pick_blend_top64": 0.6898, "pick_blend_r1_recall_45": 0.8438,
}


def add_trajectory_features(raw: pd.DataFrame, ref_years: set) -> pd.DataFrame:
    """Attach prod_ascension and prod_per_season_z to the raw feature frame.

    ref_years: the fold's z-score reference years. The per-season z stats are
    computed from these rows ONLY (mirrors train_models.stats_from_ref's
    per-"_grp" grouping) so there is no test-year leakage.
    """
    df = raw.copy()
    fs = df["prod_fs_raw"].to_numpy(dtype=float)
    car = df["prod_car_raw"].to_numpy(dtype=float)
    seasons = df["car_seasons"].to_numpy(dtype=float)

    # A. ascension ratio — NaN unless both composites exist and career > 0
    ok = np.isfinite(fs) & np.isfinite(car) & (car > 0)
    asc = np.full(len(df), np.nan)
    asc[ok] = fs[ok] / car[ok]
    df[FEAT_A] = asc

    # B. per-season volume, z-scored per composite group on ref-year rows only
    okb = np.isfinite(car) & np.isfinite(seasons) & (seasons > 0)
    pps = np.full(len(df), np.nan)
    pps[okb] = car[okb] / seasons[okb]
    df["_pps_raw"] = pps

    ref = df[df.draft_year.isin(ref_years)]
    assert not (set(ref.draft_year.unique()) & tm.TEST_YEARS), \
        "test year leaked into per-season z reference"
    zstats = {}
    for grp, sub in ref.groupby("_grp"):
        mu, sd = float(sub["_pps_raw"].mean()), float(sub["_pps_raw"].std())
        zstats[grp] = (mu, sd if (np.isfinite(sd) and sd > 0) else np.nan)
    mu = df["_grp"].map(lambda g: zstats.get(g, (np.nan, np.nan))[0]).astype(float)
    sd = df["_grp"].map(lambda g: zstats.get(g, (np.nan, np.nan))[1]).astype(float)
    df[FEAT_B] = (df["_pps_raw"] - mu) / sd
    df = df.drop(columns=["_pps_raw"])
    return df


def run_variant(features: list, df: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    """Refit all three heads on the frozen eval split with the given feature
    list (experiment_v4.py monkeypatch pattern) and score the holdout."""
    np.random.seed(tm.SEED)
    tm.SUCCESS_FEATURES = features  # monkeypatch — _xy / fit_pick_members read this

    train = pd.concat([df[df.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df[df.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df[df.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    assert not (set(train.draft_year.unique()) & tm.TEST_YEARS)
    assert not (set(cal.draft_year.unique()) & tm.TEST_YEARS)

    s = tm.fit_success_members(train)
    s["calibrator"] = tm.fit_success_calibrator(s, cal, tm.CAL_YEARS)
    g = tm.fit_grade_members(train)
    p = tm.fit_pick_members(train)

    X = test[features]
    y_s = test.nfl_success.to_numpy()
    y_g = test.draft_grade.to_numpy()

    prob = tm.ensemble_success_probs(s, X, calibrated=True)
    P_raw = tm.ensemble_grade_probs(g, X, calibrated=False)  # served label path

    y_pick = np.exp(tm._pick_target(test))
    m = np.isfinite(y_pick)
    reg = tm.ensemble_pick_preds(p, X[m])
    cls_pick = tm.classifier_expected_pick(P_raw[m])
    blend = np.exp(0.5 * (np.log(reg) + np.log(cls_pick)))  # SERVED estimator

    pk = tm.pick_metrics(y_pick[m], blend)
    return {
        "n_features": len(features),
        "success_auc": round(float(roc_auc_score(y_s, prob)), 4),
        "success_brier": round(float(brier_score_loss(y_s, prob)), 4),
        "grade_acc_raw": round(float(accuracy_score(y_g, P_raw.argmax(axis=1))), 4),
        "pick_blend_mae": pk["mae_picks_drafted"],
        "pick_blend_spearman": pk["spearman_all"],
        "pick_blend_top64": pk["spearman_top64"],
        "pick_blend_r1_recall_45": pk["r1_recall_within_45"],
    }


def main() -> int:
    np.random.seed(tm.SEED)
    print(f"Loading {tm.TRAINING_DATA_PATH}")
    raw = tm.load_raw_rows()
    raw = add_trajectory_features(raw, tm.EVAL_REF_YEARS)

    seeds = tm.seed_rows()
    for c in (FEAT_A, FEAT_B):
        seeds[c] = np.nan  # seed exemplars have no college stat lines

    # Frozen eval split z-features (z-ref 2000-2018, train+cal only)
    df = tm.apply_z(raw, tm.stats_from_ref(raw, tm.EVAL_REF_YEARS))

    # Feature sanity / distribution report
    fin_a = df[FEAT_A].dropna()
    fin_b = df[FEAT_B].dropna()
    test_mask = df.draft_year.isin(tm.TEST_YEARS)
    coverage = {
        FEAT_A: {"non_nan_frac": round(float(df[FEAT_A].notna().mean()), 4),
                 "min": round(float(fin_a.min()), 4),
                 "median": round(float(fin_a.median()), 4),
                 "max": round(float(fin_a.max()), 4),
                 "frac_above_1": round(float((fin_a > 1.0).mean()), 4)},
        FEAT_B: {"non_nan_frac": round(float(df[FEAT_B].notna().mean()), 4),
                 "min": round(float(fin_b.min()), 4),
                 "median": round(float(fin_b.median()), 4),
                 "max": round(float(fin_b.max()), 4),
                 "test_rows_non_nan": int(df.loc[test_mask, FEAT_B].notna().sum())},
    }
    print("Feature coverage/distribution:", json.dumps(coverage, indent=2))

    V4 = list(tm.SUCCESS_FEATURES)  # dv_features list — the served v4 set
    assert FEAT_A not in V4 and FEAT_B not in V4
    variants = {
        "v4_baseline": V4,
        "v4_plus_A_ascension": V4 + [FEAT_A],
        "v4_plus_B_per_season": V4 + [FEAT_B],
        "v4_plus_A_plus_B": V4 + [FEAT_A, FEAT_B],
    }

    results = {}
    for name, feats in variants.items():
        print(f"\n[{name}] refitting all heads ({len(feats)} features)…")
        results[name] = run_variant(feats, df, seeds)
        print(f"  {results[name]}")

    base = results["v4_baseline"]

    def verdict(r):
        # Adoption bar (v4 spirit): strictly better somewhere that matters,
        # non-inferior (within noise) everywhere else. Noise bands: 0.005 AUC,
        # 0.002 Brier, 0.01 acc, 0.01 rho, 1.5 picks MAE, 0.02 recall.
        better = (
            r["success_auc"] > base["success_auc"] + 0.005
            or r["success_brier"] < base["success_brier"] - 0.002
            or r["grade_acc_raw"] > base["grade_acc_raw"] + 0.01
            or r["pick_blend_spearman"] > base["pick_blend_spearman"] + 0.01
            or r["pick_blend_top64"] > base["pick_blend_top64"] + 0.01
            or r["pick_blend_mae"] < base["pick_blend_mae"] - 1.5
            or r["pick_blend_r1_recall_45"] > base["pick_blend_r1_recall_45"] + 0.02
        )
        non_inferior = (
            r["success_auc"] >= base["success_auc"] - 0.005
            and r["success_brier"] <= base["success_brier"] + 0.002
            and r["grade_acc_raw"] >= base["grade_acc_raw"] - 0.01
            and r["pick_blend_spearman"] >= base["pick_blend_spearman"] - 0.01
            and r["pick_blend_top64"] >= base["pick_blend_top64"] - 0.01
            and r["pick_blend_mae"] <= base["pick_blend_mae"] + 1.5
            and r["pick_blend_r1_recall_45"] >= base["pick_blend_r1_recall_45"] - 0.02
        )
        return {"strictly_better_somewhere": bool(better),
                "non_inferior_everywhere": bool(non_inferior),
                "adopt": bool(better and non_inferior)}

    gates = {k: verdict(v) for k, v in results.items() if k != "v4_baseline"}

    out = {
        "experiment": "v5_traj (trajectory-shape features)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": tm.git_sha(),
        "split": {
            "train_years": f"{min(tm.EVAL_TRAIN_YEARS)}-{max(tm.EVAL_TRAIN_YEARS)} + 63 seeds (w=5)",
            "cal_years": sorted(tm.CAL_YEARS),
            "test_years": sorted(tm.TEST_YEARS),
            "z_ref_years": f"{min(tm.EVAL_REF_YEARS)}-{max(tm.EVAL_REF_YEARS)}",
        },
        "features": {
            FEAT_A: "prod_fs_raw / prod_car_raw; NaN if either missing or car<=0",
            FEAT_B: "z(prod_car_raw / car_seasons) per _grp, stats from z-ref years only",
        },
        "coverage": coverage,
        "v4_recorded_baselines": V4_RECORDED,
        "results": results,
        "gates": gates,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")
    print("GATES:", json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
