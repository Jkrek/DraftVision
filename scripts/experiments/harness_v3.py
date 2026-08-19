#!/usr/bin/env python3
"""v3 experiment harness — recruiting pedigree (A), non-leaky age proxy (B),
all-position production (C), SP+ team rating (D), modern classes (E), combos
and ablations (F), sanity/missingness audit (G).

Builds on scripts/experiments/harness.py (v2): model fitting, probs, metrics
and seed rows are IMPORTED unchanged. Frozen test = 2019-2020 (identical rows
to v2); v2-winner base config = +meas_norm | flat | hist (2000-2017 train,
cal 2018). Forward split = train 2000-2023, cal 2024, test 2025-2026.

Writes NOTHING outside models/experiments/. Production artifacts untouched.

Run: .venv/bin/python scripts/experiments/harness_v3.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import harness as h  # v2 harness: models, metrics, seeds, calibration
from dv_features import (
    _BASE_FEATURES,  # base-13 pinned: SUCCESS_FEATURES now includes the promoted v3 blocks
    position_flags,
    _production_group,
    _raw_production_to_percentile,
)

V3_CSV = os.path.join(REPO_ROOT, "training_data", "experiments", "combine_outcomes_v3.csv")
OUT_JSON = os.path.join(REPO_ROOT, "models", "experiments", "results_runs_v3.json")

MEASURABLES = h.MEASURABLES
SEED_ROW_WEIGHT = 5.0

# splits
FROZEN = {"train": set(range(2000, 2018)), "cal": {2018}, "test": {2019, 2020},
          "ref": set(range(2000, 2019))}          # z-score reference == v2 exactly
FROZEN_E = {**FROZEN, "train": set(range(2000, 2018)) | set(range(2021, 2027))}
FORWARD = {"train": set(range(2000, 2024)), "cal": {2024}, "test": {2025, 2026},
           "ref": set(range(2000, 2025))}

REC_FEATS = ["rec_stars", "rec_rating", "rec_ranking"]
AGE_FEATS = ["years_in_college", "early_declare"]
PROD_FEATS = ["prod_fs_z", "prod_car_z", "car_seasons"]
SP_FEATS = ["sp_rating"]
NEW_FEATS = REC_FEATS + AGE_FEATS + PROD_FEATS + SP_FEATS

NEUTRAL_FILL = {"rec_stars": 2.0, "rec_ranking": 4000.0,
                "years_in_college": 4.0, "early_declare": 0.0}  # rec_rating: min obs

_GRP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
        "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "G": "OL", "LS": "OL",
        "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL", "EDGE": "DL",
        "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
        "CB": "DB", "S": "DB", "DB": "DB", "FS": "DB", "SS": "DB"}

OFF_FLOOR_CLASS = 2010   # season stats thin before 2009
DEF_FLOOR_CLASS = 2017   # defensive category starts season 2016


def load_rows_v3() -> pd.DataFrame:
    """v3 CSV -> feature frame. Base-13 identical to v2 harness; carries the
    new CFBD columns and raw fs_/car_ stats for composite building."""
    df = pd.read_csv(V3_CSV)
    fs_cols = [c for c in df.columns if c.startswith(("fs_", "car_"))]
    rows = []
    for _, r in df.iterrows():
        pos = str(r.get("position") or "OTH").upper()
        raw_prod = h._nan_float(r.get("production_score"))
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
            "games_played":        h._nan_float(r.get("games_college")),
            "combine_speed_score": h._nan_float(r.get("combine_speed_score")),
            "conference_tier":     h._nan_float(r.get("conference_tier")),
            **position_flags(pos),
            "nfl_success":         int(r["nfl_success"]),
            "draft_grade":         int(min(3, max(0, int(r["draft_grade"])))),
            "_pos_group":          _production_group(pos),
            "_grp":                _GRP.get(pos, "OTHER"),
            "_match":              str(r.get("rec_match_type") or "none"),
        }
        for m in MEASURABLES + ["age"] + REC_FEATS + AGE_FEATS + SP_FEATS + fs_cols:
            row[m] = h._nan_float(r.get(m))
        rows.append(row)
    return pd.DataFrame(rows)


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Raw final-season / career production composites per position group.
    NaN below the source coverage floor for the group (offense 2010+ classes,
    defense 2017+) and for OL/OTHER (no counting stats)."""
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


def prepare(df: pd.DataFrame, ref_years: set) -> pd.DataFrame:
    """Split-specific z-scores (stats from ref_years only, never test years):
    measurable *_z within _pos_group (identical recipe to v2) and production
    composites within _grp x {fs,car}."""
    df = df.copy()
    ref = df[df.draft_year.isin(ref_years)]
    for m in MEASURABLES:
        stats = ref.groupby("_pos_group")[m].agg(["mean", "std"])
        mu = df["_pos_group"].map(stats["mean"])
        sd = df["_pos_group"].map(stats["std"]).replace(0, np.nan)
        df[m + "_z"] = (df[m] - mu) / sd
    for pre in ("fs", "car"):
        col = f"prod_{pre}_raw"
        stats = ref.groupby("_grp")[col].agg(["mean", "std"])
        mu = df["_grp"].map(stats["mean"])
        sd = df["_grp"].map(stats["std"]).replace(0, np.nan)
        df[f"prod_{pre}_z"] = (df[col] - mu) / sd
    # neutral-filled copies of recruiting features (missing != outcome here,
    # but report both per experiment A)
    rating_fill = float(ref["rec_rating"].min())
    for c, fill in {**NEUTRAL_FILL, "rec_rating": rating_fill}.items():
        df[c + "_nf"] = df[c].fillna(fill)
    return df


def seed_rows_v3(extra_cols) -> pd.DataFrame:
    seeds = h.seed_rows()
    for c in extra_cols:
        seeds[c] = float("nan")
    seeds["_grp"] = "OTHER"
    seeds["_match"] = "none"
    return seeds


def run(df, seeds, feats, method, split, label, calibrate=True) -> dict:
    train_csv = df[df.draft_year.isin(split["train"])].copy()
    train = pd.concat([train_csv, seeds], ignore_index=True)
    cal = df[df.draft_year.isin(split["cal"])].reset_index(drop=True)
    test = df[df.draft_year.isin(split["test"])].reset_index(drop=True)

    # G. sanity: no leakage
    forbidden = h.FORBIDDEN | {"age"}
    assert not (set(feats) & forbidden), f"forbidden feature in X: {set(feats) & forbidden}"
    assert not (set(train.draft_year.unique()) & split["test"]), "test year in train"
    assert not (set(cal.draft_year.unique()) & split["test"]), "test year in cal"
    assert set(test.draft_year.unique()) == split["test"]
    assert not (split["ref"] & split["test"]), "test year in z-score reference"

    y_tr = train.draft_grade.to_numpy()
    w_tr = train.sample_weight.to_numpy()
    y_cal = cal.draft_grade.to_numpy()
    y_te = test.draft_grade.to_numpy()

    out = {"label": label, "features": list(feats), "method": method,
           "n_train": int(len(train)), "n_cal": int(len(cal)), "n_test": int(len(test))}

    if method == "flat":
        models = h.fit_flat(train, y_tr, w_tr, feats)
        P_te = h.flat_probs(models, test, feats)
        out["raw"] = h.metrics(y_te, P_te.argmax(axis=1))
        if calibrate:
            P_cal = h.flat_probs(models, cal, feats)
            out["calibrated"] = h.metrics(
                y_te, h.calibrate_multinomial(P_cal, y_cal, P_te).argmax(axis=1))
    elif method == "ordinal":
        models = h.fit_ordinal(train, y_tr, w_tr, feats)
        out["raw"] = h.metrics(y_te, h.ordinal_probs(models, test, feats).argmax(axis=1))
    else:
        raise ValueError(method)

    m = out["raw"]
    print(f"  {label:<40} acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
          f"recall {m['per_class_recall']}"
          + (f"  | cal acc {out['calibrated']['accuracy']:.4f}" if "calibrated" in out else ""))
    return out


def missingness_report(df: pd.DataFrame) -> None:
    """G. drafted-vs-UDFA coverage of every new feature; flag v2-age-style
    traps (missingness that alone separates classes)."""
    print("\n== G. missingness audit (non-NaN %, drafted vs UDFA) ==")
    print(f"{'feature':<18}{'set':<12}{'drafted%':>9}{'udfa%':>8}{'ratio':>7}")
    for c in NEW_FEATS:
        for tag, sub in (("2010-17 tr", df[df.draft_year.between(2010, 2017)]),
                         ("test 19-20", df[df.draft_year.isin([2019, 2020])]),
                         ("21-26", df[df.draft_year.between(2021, 2026)])):
            dr = sub[sub.draft_grade < 3][c].notna().mean() * 100
            ud = sub[sub.draft_grade == 3][c].notna().mean() * 100
            ratio = dr / ud if ud else float("inf")
            flag = "  <-- CHECK" if (max(dr, ud) > 20 and (ratio > 2 or ratio < 0.5)) else ""
            print(f"{c:<18}{tag:<12}{dr:>8.1f}{ud:>8.1f}{ratio:>7.2f}{flag}")


def main() -> int:
    np.random.seed(h.SEED)
    print(f"Loading {V3_CSV}")
    df = load_rows_v3()
    df = add_composites(df)

    df_frozen = prepare(df, FROZEN["ref"])
    df_forward = prepare(df, FORWARD["ref"])

    z_cols = [m + "_z" for m in MEASURABLES] + ["prod_fs_z", "prod_car_z"]
    nf_cols = [c + "_nf" for c in NEUTRAL_FILL] + ["rec_rating_nf"]
    extra = (MEASURABLES + ["age"] + NEW_FEATS + z_cols + nf_cols
             + ["prod_fs_raw", "prod_car_raw"])
    seeds = seed_rows_v3(extra)

    missingness_report(df_frozen)

    base = list(_BASE_FEATURES)
    v2w = base + [m + "_z" for m in MEASURABLES]          # v2 winner features
    fs = {
        "v2winner":        v2w,
        "+A":              v2w + REC_FEATS,
        "+A(neutral)":     v2w + ["rec_stars_nf", "rec_rating_nf", "rec_ranking_nf"],
        "+A+B":            v2w + REC_FEATS + AGE_FEATS,
        "+A+B(neutral)":   v2w + ["rec_stars_nf", "rec_rating_nf", "rec_ranking_nf",
                                  "years_in_college_nf", "early_declare_nf"],
        "+A+B+C":          v2w + REC_FEATS + AGE_FEATS + PROD_FEATS,
        "+A+B+C+D":        v2w + REC_FEATS + AGE_FEATS + PROD_FEATS + SP_FEATS,
    }

    runs = []
    print("\n== F. frozen test 2019-2020 (n=733), flat ensemble, hist train ==")
    for name, feats in fs.items():
        runs.append(run(df_frozen, seeds, feats, "flat", FROZEN, f"{name}|flat|hist"))

    print("\n== E. + modern classes 2021-2026 as train-only rows ==")
    for name in ("+A+B", "+A+B+C", "+A+B+C+D"):
        runs.append(run(df_frozen, seeds, fs[name], "flat", FROZEN_E,
                        f"{name}+E|flat|hist+21-26"))

    # best combo on frozen test -> ordinal rerun + forward split
    frozen_runs = [r for r in runs]
    best = max(frozen_runs, key=lambda r: (r["raw"]["accuracy"], r["raw"]["macro_f1"]))
    best_name = best["label"].split("|")[0].replace("+E", "")
    print(f"\n== best combo: {best['label']} — ordinal rerun ==")
    runs.append(run(df_frozen, seeds, fs[best_name], "ordinal",
                    FROZEN_E if "+E" in best["label"] else FROZEN,
                    f"{best_name}|ordinal|{'hist+21-26' if '+E' in best['label'] else 'hist'}"))

    print("\n== E(ii). forward split: train 2000-2023, cal 2024, test 2025-2026 ==")
    fwd_v2 = run(df_forward, seeds, fs["v2winner"], "flat", FORWARD, "v2winner|flat|fwd")
    fwd_best = run(df_forward, seeds, fs[best_name], "flat", FORWARD, f"{best_name}|flat|fwd")
    runs += [fwd_v2, fwd_best]

    print(f"\nBest frozen run: {best['label']}  acc {best['raw']['accuracy']:.4f} "
          f"macroF1 {best['raw']['macro_f1']:.4f}")
    print("Confusion matrix (rows true 0..3):")
    for row in best["raw"]["confusion_matrix"]:
        print("   ", row)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(runs, fh, indent=2)
    print(f"\nWrote {OUT_JSON} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
