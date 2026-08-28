#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate public/data/backtest.json — the "receipts" page data.

Refits the EVAL-phase models exactly as scripts/train_models.py Phase 1 does —
train on 2000-2017 draft classes (+63 curated seed rows, weight 5), calibrate
on 2018, z-score reference 2000-2018 — then scores the 2019-2020 temporal
holdout. Deterministic (SEED=42), so the numbers reproduce the holdout
evaluation recorded in models/metadata.json.

Why refit instead of loading the production artifacts: the FINAL-phase
production fit trains on ALL draft classes (grade/pick through 2026, success
through 2021), which include 2019-2020. Scoring those artifacts on 2019-2020
would be in-sample — inflated, not a backtest.

Heads scored, each via its serving-equivalent path:
  success:  XGB+CatBoost member-mean probability -> Platt calibrator
  grade:    member-mean 4-class probabilities, raw argmax (the served label)
  pick:     50/50 log-space blend of the pick regressor and the pick implied
            by the 4-class probabilities (the served estimator)

Nothing is written except public/data/backtest.json.

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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

# Reuse train_models' loading / fitting / scoring verbatim.
import train_models as tm
from dv_features import SUCCESS_FEATURES, DRAFT_GRADE_LABELS

OUTPUT_PATH = os.path.join(REPO_ROOT, "public", "data", "backtest.json")

DISPLAY_COLS = ("name", "college", "position", "draft_year", "draft_round",
                "draft_pick", "pro_bowls", "seasons_started", "career_av")


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


def player_record(row, prob: float, grade_idx: int, pick: float) -> dict:
    rnd = row.draft_round
    actual_pick = row.draft_pick
    return {
        "name": str(row["name"]),
        "college": None if pd.isna(row.college) else str(row.college),
        "position": str(row.position),
        "draft_year": int(row.draft_year),
        "pred_success_prob": round(float(prob), 4),
        "pred_grade_bucket": DRAFT_GRADE_LABELS[int(grade_idx)],
        "pred_pick": int(round(float(pick))),
        "actual_round_bucket": DRAFT_GRADE_LABELS[int(row.draft_grade)],
        "actual_round": None if (pd.isna(rnd) or int(rnd) > 7) else int(rnd),
        "actual_pick": None if pd.isna(actual_pick) else int(actual_pick),
        "actual_success": int(row.nfl_success),
        "career_note": career_note(row),
        "categories": [],
    }


def main() -> int:
    np.random.seed(tm.SEED)

    print(f"Loading {tm.TRAINING_DATA_PATH}")
    raw_csv = pd.read_csv(tm.TRAINING_DATA_PATH)
    raw = tm.load_raw_rows()
    assert len(raw_csv) == len(raw), "feature frame / raw CSV row mismatch"
    assert (raw_csv["draft_year"].to_numpy() == raw["draft_year"].to_numpy()).all()
    for col in DISPLAY_COLS:
        raw[col] = raw_csv[col].to_numpy()

    seeds = tm.seed_rows()

    # Identical to train_models.py Phase 1 (EVAL): z-ref 2000-2018,
    # train 2000-2017 + seeds, calibrate 2018, test 2019-2020.
    eval_stats = tm.stats_from_ref(raw, tm.EVAL_REF_YEARS)
    df_eval = tm.apply_z(raw, eval_stats)
    train = pd.concat([df_eval[df_eval.draft_year.isin(tm.EVAL_TRAIN_YEARS)], seeds],
                      ignore_index=True)
    cal = df_eval[df_eval.draft_year.isin(tm.CAL_YEARS)].reset_index(drop=True)
    test = df_eval[df_eval.draft_year.isin(tm.TEST_YEARS)].reset_index(drop=True)
    tm._assert_no_leakage(train, cal, test)
    years = sorted(tm.TEST_YEARS)
    print(f"Holdout: {len(test)} rows, draft classes {years} "
          f"(never used to fit or calibrate the eval models scored here)")

    print("Refitting eval-phase models (train "
          f"{min(tm.EVAL_TRAIN_YEARS)}-{max(tm.EVAL_TRAIN_YEARS)} + seeds, "
          f"calibrate {sorted(tm.CAL_YEARS)})…")
    s_members = tm.fit_success_members(train)
    s_bundle = {**s_members,
                "calibrator": tm.fit_success_calibrator(s_members, cal, tm.CAL_YEARS)}
    g_members = tm.fit_grade_members(train)
    p_members = tm.fit_pick_members(train)

    X = test[SUCCESS_FEATURES]
    y_success = test.nfl_success.to_numpy()
    y_grade = test.draft_grade.to_numpy()

    # Serving-equivalent scoring paths.
    p = tm.ensemble_success_probs(s_bundle, X, calibrated=True)
    G_raw = tm.ensemble_grade_probs(g_members, X, calibrated=False)
    g_pred = G_raw.argmax(axis=1)
    reg_pick = tm.ensemble_pick_preds(p_members, X)
    cls_pick = tm.classifier_expected_pick(G_raw)
    blend_pick = np.exp(0.5 * (np.log(reg_pick) + np.log(np.maximum(cls_pick, 1.0))))

    metrics = {
        "auc": round(float(roc_auc_score(y_success, p)), 4),
        "brier": round(float(brier_score_loss(y_success, p)), 4),
        "accuracy": round(float(accuracy_score(y_grade, g_pred)), 4),
        "holdout_rows": int(len(test)),
        "holdout_years": years,
        "holdout_success_rate": round(float(y_success.mean()), 4),
    }

    # Rule-based heuristic baseline, recomputed on the same holdout.
    p_base = tm.baseline_success_probs(test)
    g_base = tm.baseline_grade_preds(test)
    metrics["baseline"] = {
        "auc": round(float(roc_auc_score(y_success, p_base)), 4),
        "brier": round(float(brier_score_loss(y_success, p_base)), 4),
        "accuracy": round(float(accuracy_score(y_grade, g_base)), 4),
        "source": "rule-based heuristic fallback, same holdout",
    }

    # Pick metrics: served blend vs the classifier-implied baseline, over rows
    # with a known pick target (drafted, or confirmed round-8/UDFA).
    y_pick = np.exp(tm._pick_target(test))
    pick_m = np.isfinite(y_pick)
    metrics["pick"] = {
        "served_blend": tm.pick_metrics(y_pick[pick_m], blend_pick[pick_m]),
        "classifier_baseline": tm.pick_metrics(y_pick[pick_m], cls_pick[pick_m]),
        "n_scored": int(pick_m.sum()),
    }

    print(f"Success:  AUC {metrics['auc']:.4f}  Brier {metrics['brier']:.4f}  "
          f"(baseline AUC {metrics['baseline']['auc']:.4f}, "
          f"Brier {metrics['baseline']['brier']:.4f})")
    print(f"Grade:    accuracy {metrics['accuracy']:.4f}  "
          f"(baseline {metrics['baseline']['accuracy']:.4f})")
    pk, pb = metrics["pick"]["served_blend"], metrics["pick"]["classifier_baseline"]
    print(f"Pick:     MAE {pk['mae_picks_drafted']:.1f}  "
          f"Spearman {pk['spearman_all']:.4f}  "
          f"R1-recall@45 {pk['r1_recall_within_45']}  "
          f"(classifier baseline MAE {pb['mae_picks_drafted']:.1f})")

    # Sanity: the refit must reproduce the holdout eval in models/metadata.json.
    with open(tm.METADATA_PATH) as fh:
        meta = json.load(fh)
    checks = [
        ("success AUC", metrics["auc"],
         meta["evaluation"]["success"]["ensemble_calibrated"]["auc"], 0.005),
        ("success Brier", metrics["brier"],
         meta["evaluation"]["success"]["ensemble_calibrated"]["brier"], 0.005),
        ("grade accuracy", metrics["accuracy"],
         meta["evaluation"]["draft_grade"]["ensemble_raw_mean"]["accuracy"], 0.005),
        ("pick Spearman", pk["spearman_all"],
         meta["pick_eval"]["blend_50_50_SERVED"]["spearman_all"], 0.005),
        ("pick MAE", pk["mae_picks_drafted"],
         meta["pick_eval"]["blend_50_50_SERVED"]["mae_picks_drafted"], 0.5),
    ]
    for label, ours, theirs, tol in checks:
        if abs(ours - theirs) > tol:
            print(f"WARNING: recomputed {label}={ours} differs from "
                  f"metadata {theirs} — training recipe may have drifted")

    # ── Notable rows ─────────────────────────────────────────────────────────
    df = test.copy()
    df["prob"] = p
    df["g_pred"] = g_pred
    df["pick_pred"] = blend_pick

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
                players[key] = player_record(row, row.prob, row.g_pred, row.pick_pred)
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
                         "models scored here were trained on the 2000-2017 "
                         "classes (plus 63 curated seed rows) and calibrated "
                         "on 2018 — the exact evaluation recipe behind the "
                         "production pipeline's reported metrics. No "
                         "2019-2020 player was used to fit or calibrate them."),
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
