#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate public/data/backtest.json — the "receipts" page data.

Loads the PRODUCTION artifacts from disk (read-only) exactly as
scripts/train_models.py's eval path scores them:

  success:     success_xgboost_model.json + catboost_success_model.cbm,
               mean of member probabilities -> Platt calibrator from
               success_calibrated_model.pkl (joblib)
  draft grade: draft_grade_model.json + catboost_draft_grade_model.cbm,
               mean of member probabilities -> multinomial calibrator from
               draft_grade_calibrated_model.pkl (joblib)

Feature assembly is REUSED from train_models.load_csv_rows (which itself uses
dv_features: position_flags + raw-production -> percentile mapping), so the
holdout rows here are byte-identical to the frame the training eval saw.

Scored rows: draft classes 2019-2020 from training_data/combine_outcomes.csv —
the temporal HOLDOUT (train 2010-2017 + seeds, calibration 2018). These
classes were never used to fit or calibrate anything.

Nothing is trained or written except public/data/backtest.json.

Usage:  .venv/bin/python scripts/generate_backtest.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

# Reuse train_models' loading / scoring / feature assembly verbatim.
import train_models as tm
from dv_features import SUCCESS_FEATURES, DRAFT_GRADE_LABELS

OUTPUT_PATH = os.path.join(REPO_ROOT, "public", "data", "backtest.json")
METADATA_PATH = tm.METADATA_PATH


# ── Artifact loading (mirrors XGBOost.py / train_models eval, read-only) ──────

def load_bundle(xgb_path: str, cb_path: str, cal_path: str, num_class=None) -> dict:
    xgb_m = xgb.XGBClassifier()
    xgb_m.load_model(xgb_path)
    cb_m = CatBoostClassifier()
    cb_m.load_model(cb_path)
    cal = joblib.load(cal_path)  # dict {"kind", "model", "feature_names", ...}
    assert list(cal.get("feature_names", SUCCESS_FEATURES)) == list(SUCCESS_FEATURES), (
        f"feature mismatch in {os.path.basename(cal_path)}")
    return {"xgb": xgb_m, "cb": cb_m, "calibrator": cal}


# ── Holdout frame: train_models features + display metadata columns ───────────

def load_holdout() -> pd.DataFrame:
    raw = pd.read_csv(tm.TRAINING_DATA_PATH)
    feat = tm.load_csv_rows()  # same order as raw (row-for-row transform)
    assert len(raw) == len(feat), "feature frame / raw CSV row mismatch"
    assert (raw["draft_year"].to_numpy() == feat["draft_year"].to_numpy()).all()
    for col in ("college", "position", "draft_round", "pro_bowls",
                "seasons_started", "career_av"):
        feat[col] = raw[col].to_numpy()
    holdout = feat[feat.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    assert len(holdout) > 0, "no holdout rows found"
    return holdout


def career_note(row) -> str:
    pb = int(row.pro_bowls) if pd.notna(row.pro_bowls) else 0
    ss = int(row.seasons_started) if pd.notna(row.seasons_started) else 0
    parts = []
    if pb:
        parts.append(f"{pb}x Pro Bowl")
    if ss:
        parts.append(f"{ss} season{'s' if ss != 1 else ''} as a primary starter")
    if not parts:
        return "Never a primary NFL starter, no Pro Bowls"
    return ", ".join(parts)


def player_record(row, prob: float, grade_idx: int) -> dict:
    rnd = row.draft_round
    return {
        "name": str(row["name"]),
        "college": None if pd.isna(row.college) else str(row.college),
        "position": str(row.position),
        "draft_year": int(row.draft_year),
        "pred_success_prob": round(float(prob), 4),
        "pred_grade_bucket": DRAFT_GRADE_LABELS[int(grade_idx)],
        "actual_round_bucket": DRAFT_GRADE_LABELS[int(row.draft_grade)],
        "actual_round": None if (pd.isna(rnd) or int(rnd) > 7) else int(rnd),
        "actual_success": int(row.nfl_success),
        "career_note": career_note(row),
        "categories": [],
    }


def main() -> int:
    print("Loading production artifacts (read-only)…")
    s_bundle = load_bundle(tm.SUCCESS_MODEL_PATH, tm.CATBOOST_SUCCESS_PATH,
                           tm.SUCCESS_CALIBRATED_PATH)
    g_bundle = load_bundle(tm.DRAFT_GRADE_MODEL_PATH, tm.CATBOOST_DRAFT_GRADE_PATH,
                           tm.DRAFT_GRADE_CALIBRATED_PATH)

    holdout = load_holdout()
    years = sorted(tm.TEST_YEARS)
    print(f"Holdout: {len(holdout)} rows, draft classes {years} "
          f"(never used in training or calibration)")

    X = holdout[SUCCESS_FEATURES]
    y_success = holdout.nfl_success.to_numpy()
    y_grade = holdout.draft_grade.to_numpy()

    # Identical scoring path to train_models' eval (and XGBOost.py serving):
    # member-mean probability, then the ensemble calibrator.
    p = tm.ensemble_success_probs(s_bundle, X, calibrated=True)
    G = tm.ensemble_grade_probs(g_bundle, X, calibrated=True)
    g_pred = G.argmax(axis=1)

    metrics = {
        "auc": round(float(roc_auc_score(y_success, p)), 4),
        "brier": round(float(brier_score_loss(y_success, p)), 4),
        "accuracy": round(float(accuracy_score(y_grade, g_pred)), 4),
        "holdout_rows": int(len(holdout)),
        "holdout_years": years,
        "holdout_success_rate": round(float(y_success.mean()), 4),
    }

    # Baseline (rule-based heuristic) metrics come from models/metadata.json —
    # they were computed on this same holdout by train_models.py.
    with open(METADATA_PATH) as fh:
        meta = json.load(fh)
    base_s = meta["evaluation"]["success"]["rule_based_baseline"]
    base_g = meta["evaluation"]["draft_grade"]["rule_based_baseline"]
    metrics["baseline"] = {
        "auc": base_s["auc"],
        "brier": base_s["brier"],
        "accuracy": base_g["accuracy"],
        "source": "models/metadata.json (rule_based_baseline, same holdout)",
    }
    # Sanity: our recomputed numbers should match the metadata's holdout eval.
    meta_ens = meta["evaluation"]["success"]["ensemble_calibrated"]
    for k in ("auc", "brier"):
        if abs(metrics[k] - meta_ens[k]) > 0.005:
            print(f"WARNING: recomputed {k}={metrics[k]} differs from "
                  f"metadata {meta_ens[k]} — artifacts may be newer than metadata")

    print(f"Success:  AUC {metrics['auc']:.4f}  Brier {metrics['brier']:.4f}  "
          f"(baseline AUC {base_s['auc']:.4f}, Brier {base_s['brier']:.4f})")
    print(f"Grade:    accuracy {metrics['accuracy']:.4f}  "
          f"(baseline {base_g['accuracy']:.4f})")

    # ── Notable rows ─────────────────────────────────────────────────────────
    df = holdout.copy()
    df["prob"] = p
    df["g_pred"] = g_pred

    order = df.sort_values("prob", ascending=False)
    top20 = order.head(20)

    succ = order[order.nfl_success == 1]
    bust = order[order.nfl_success == 0]

    hits = succ.head(10)                      # high prob, succeeded
    misses = bust.head(10)                    # high prob, busted
    steals = succ[(succ.g_pred == 0) & (succ.draft_grade >= 1)].head(10)
    fades = bust.sort_values("prob").head(10)  # lowest prob, correctly busted

    players: dict = {}

    def add(rows, category):
        for _, row in rows.iterrows():
            key = (row["name"], int(row.draft_year))
            if key not in players:
                players[key] = player_record(row, row.prob, row.g_pred)
            players[key]["categories"].append(category)

    add(top20, "top20")
    add(hits, "hit")
    add(misses, "miss")
    add(steals, "steal")
    add(fades, "fade")

    player_list = sorted(players.values(),
                         key=lambda r: r["pred_success_prob"], reverse=True)
    counts = {c: sum(1 for pl in player_list if c in pl["categories"])
              for c in ("top20", "hit", "miss", "steal", "fade")}
    print(f"Notable rows: {len(player_list)} unique "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())})")

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "holdout_note": ("Draft classes 2019-2020 are a temporal holdout: the "
                         "models were trained on 2010-2017 (plus 63 curated seed "
                         "rows) and calibrated on 2018. No 2019-2020 player was "
                         "used to fit or calibrate anything."),
        "metrics": metrics,
        "players": player_list,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
