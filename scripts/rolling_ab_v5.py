#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling-origin A/B: v4 feature set vs the v5 candidate (+sos +trajectory).

The single 2019-20 holdout showed the 34-feature candidate improving 5 of 7
metrics, all inside noise bands. This runs the same comparison across the six
rolling folds (test year Y in 2015..2020, train 2000..Y-2 + seeds, cal Y-1,
z-ref 2000..Y-1) and reports per-fold deltas — adoption requires the candidate
to win on average across folds, not once.

Writes models/experiments/rolling_ab_v5.json.

Usage:  .venv/bin/python scripts/rolling_ab_v5.py
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

import train_models as tm
from experiment_v5_traj import FEAT_A, FEAT_B, add_trajectory_features, run_variant
from experiment_v5_sos import attach_sos

OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "rolling_ab_v5.json")
FOLDS = [2015, 2016, 2017, 2018, 2019, 2020]
METRICS = ["success_auc", "success_brier", "grade_acc_raw", "pick_blend_mae",
           "pick_blend_spearman", "pick_blend_top64", "pick_blend_r1_recall_45"]


def main() -> int:
    raw0 = tm.load_raw_rows()
    if "college" not in raw0.columns:
        csv = pd.read_csv(tm.TRAINING_DATA_PATH)
        assert len(csv) == len(raw0)
        raw0["college"] = csv["college"].to_numpy()
    raw0 = attach_sos(raw0)

    V4 = list(tm.SUCCESS_FEATURES)
    V5 = V4 + ["sos_sp", FEAT_A, FEAT_B]
    seeds = tm.seed_rows()
    for f in ("sos_sp", FEAT_A, FEAT_B):
        seeds[f] = np.nan

    folds = []
    for y in FOLDS:
        # per-fold split constants (run_variant reads these module globals)
        tm.EVAL_TRAIN_YEARS = set(range(2000, y - 1))
        tm.CAL_YEARS = {y - 1}
        tm.TEST_YEARS = {y}
        ref_years = set(range(2000, y))

        raw = add_trajectory_features(raw0, ref_years)
        df = tm.apply_z(raw, tm.stats_from_ref(raw, ref_years))

        r4 = run_variant(V4, df, seeds)
        r5 = run_variant(V5, df, seeds)
        delta = {m: round(r5[m] - r4[m], 4) for m in METRICS}
        folds.append({"test_year": y, "v4": r4, "v5": r5, "delta": delta})
        print(f"fold {y}: dAUC {delta['success_auc']:+.4f}  "
              f"dBrier {delta['success_brier']:+.4f}  "
              f"dAcc {delta['grade_acc_raw']:+.4f}  "
              f"dMAE {delta['pick_blend_mae']:+.1f}  "
              f"dRho {delta['pick_blend_spearman']:+.4f}  "
              f"dR1 {delta['pick_blend_r1_recall_45']:+.4f}", flush=True)

    summary = {}
    for m in METRICS:
        ds = [f["delta"][m] for f in folds]
        summary[m] = {"mean_delta": round(float(np.mean(ds)), 4),
                      "std_delta": round(float(np.std(ds, ddof=1)), 4),
                      "wins": sum(1 for d in ds
                                  if (d < 0 if m in ("success_brier", "pick_blend_mae") else d > 0)),
                      "per_fold": ds}

    out = {"folds": folds, "summary": summary,
           "v4_features": len(V4), "v5_features": len(V5),
           "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
