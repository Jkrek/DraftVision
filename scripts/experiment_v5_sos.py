#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 strength-of-schedule experiment — EVAL ONLY, frozen 2019-20 holdout.

Joins training_data/enrich_sos.csv (per team-season mean opponent SP+, built
by scripts/add_sos_column.py from year-level CFBD pulls) onto the training
frame as `sos_sp` — the schedule-strength signal sp_rating (own-team SP+)
cannot express — and scores it with the exact harness experiment_v5_traj.py
used, so all numbers are comparable:

  variants: v4 baseline | +sos | +sos +ascension +per-season (the full wave)

Join: (college with aliases, draft_year - 1) == (team, season) — the final
college season. Unmatched rows (FCS schools etc.) stay NaN.

Writes models/experiments/v5_sos_results.json. Nothing else is modified.

Usage:  .venv/bin/python scripts/experiment_v5_sos.py
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
from experiment_v5_traj import (FEAT_A, FEAT_B, add_trajectory_features,
                                run_variant)

SOS_PATH = os.path.join(REPO_ROOT, "training_data", "enrich_sos.csv")
OUT_PATH = os.path.join(REPO_ROOT, "models", "experiments", "v5_sos_results.json")

# CSV college name -> CFBD school name (from add_sos_column.py's unmatched report)
COLLEGE_ALIASES = {
    "miami (fl)": "miami",
    "connecticut": "uconn",
    "hawaii": "hawai'i",
    "san jose state": "san josé state",
    "la-monroe": "louisiana monroe",
    "middle tenn. state": "middle tennessee",
}


def attach_sos(raw: pd.DataFrame) -> pd.DataFrame:
    """Join sos_sp on (college aliased+casefolded, draft_year-1)."""
    sos = pd.read_csv(SOS_PATH)
    lookup = {(str(t).strip().casefold(), int(s)): float(v)
              for t, s, v in zip(sos["team"], sos["season"], sos["sos_sp"])}
    df = raw.copy()

    def key(college, year):
        c = str(college).strip().casefold()
        c = COLLEGE_ALIASES.get(c, c)
        return (c, int(year) - 1)

    vals = [lookup.get(key(c, y), np.nan) if pd.notna(c) and pd.notna(y) else np.nan
            for c, y in zip(df.get("college", df.get("name")), df["draft_year"])]
    df["sos_sp"] = vals
    return df


def main() -> int:
    np.random.seed(tm.SEED)
    print(f"Loading {tm.TRAINING_DATA_PATH}")
    raw = tm.load_raw_rows()

    # college column: load_raw_rows may not carry it — pull from the CSV by row order
    if "college" not in raw.columns:
        csv = pd.read_csv(tm.TRAINING_DATA_PATH)
        assert len(csv) == len(raw)
        raw["college"] = csv["college"].to_numpy()

    raw = attach_sos(raw)
    cov = float(np.isfinite(raw["sos_sp"]).mean())
    cov_2006 = float(np.isfinite(raw.loc[raw.draft_year >= 2006, "sos_sp"]).mean())
    print(f"sos_sp coverage: {cov:.1%} all rows, {cov_2006:.1%} draft_year>=2006")

    # trajectory features for the combined variant (fold-safe z on eval ref)
    raw = add_trajectory_features(raw, tm.EVAL_REF_YEARS)

    eval_stats = tm.stats_from_ref(raw, tm.EVAL_REF_YEARS)
    df = tm.apply_z(raw, eval_stats)
    seeds = tm.seed_rows()
    for f in ("sos_sp", FEAT_A, FEAT_B):
        if f not in seeds.columns:
            seeds[f] = np.nan

    V4 = list(tm.SUCCESS_FEATURES)
    variants = {
        "v4_baseline": V4,
        "plus_sos": V4 + ["sos_sp"],
        "plus_sos_traj": V4 + ["sos_sp", FEAT_A, FEAT_B],
    }

    results = {}
    for name, feats in variants.items():
        print(f"\n── {name} ({len(feats)} features) ──")
        results[name] = run_variant(feats, df, seeds)
        for k, v in results[name].items():
            print(f"  {k}: {v}")

    results["sos_coverage"] = {"all_rows": round(cov, 4),
                               "draft_year_ge_2006": round(cov_2006, 4)}
    results["generated_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
