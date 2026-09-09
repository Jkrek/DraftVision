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

Also computed on the SAME holdout rows, so the page can be checked
independently:
  * the consensus big board (WideLeft/ESPN staged boards, consensus_rank in
    the training CSV) scored as a success ranker and as a pick predictor —
    the comparator that matters, not the app's rule-based fallback;
  * 80% conformal pick intervals (CatBoost q10/q90 on the eval train,
    split-conformal offsets on the 2018 calibration fold — the same recipe
    train_models.py reports under pick_interval.eval);
  * a 10-bin reliability table for the calibrated success probability;
  * the rolling-origin CV folds (models/experiments/rolling_cv_v4.json) and
    the v3 forward-split disclosure (models/experiments/results_runs_v3.json)
    are embedded verbatim so the page has one fetch.

Writes public/data/backtest.json and public/data/backtest_predictions.csv
(the full per-player ledger for every holdout row).

Usage:  .venv/bin/python scripts/generate_backtest.py
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

# Reuse train_models' loading / fitting / scoring verbatim.
import train_models as tm
from dv_features import SUCCESS_FEATURES, DRAFT_GRADE_LABELS

OUTPUT_PATH = os.path.join(REPO_ROOT, "public", "data", "backtest.json")
LEDGER_PATH = os.path.join(REPO_ROOT, "public", "data", "backtest_predictions.csv")
ROLLING_CV_PATH = os.path.join(REPO_ROOT, "models", "experiments", "rolling_cv_v4.json")
V3_RUNS_PATH = os.path.join(REPO_ROOT, "models", "experiments", "results_runs_v3.json")

DISPLAY_COLS = ("name", "college", "position", "draft_year", "draft_round",
                "draft_pick", "pro_bowls", "seasons_started", "career_av",
                "consensus_rank", "consensus_covered")

# Unranked-in-a-covered-year on the consensus board: "the boards passed" —
# treated as worse than every ranked player (same convention as the
# consensus_logrank feature, which pins these at rank 400).
CONSENSUS_UNRANKED = 400.0
# Only picks 1-262 exist; a board rank past that is not a pick prediction.
LAST_DRAFT_PICK = 262

LEDGER_COLS = ("name", "college", "position", "draft_year",
               "pred_success_prob", "pred_grade_bucket", "pred_pick",
               "pick_lo", "pick_hi", "consensus_rank",
               "actual_pick", "actual_round", "actual_success", "career_av")


def _opt_int(v):
    return None if pd.isna(v) else int(v)


def drafted_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows with a real overall pick (1-262). The single definition of
    "drafted" used in this script; keep it off the float pick target."""
    return np.isfinite(df["draft_pick"].to_numpy(dtype=float))


def consensus_metrics(test: pd.DataFrame, y_success, y_pick, pick_m,
                      blend_pick) -> dict:
    """Score the consensus big board on the holdout as (a) a success ranker
    and (b) a pick predictor, on exactly the rows the model is scored on."""
    covered = test["consensus_covered"].to_numpy(dtype=float) == 1.0
    rank = test["consensus_rank"].to_numpy(dtype=float)
    ranked = np.isfinite(rank) & covered
    # Board rank as a pick predictor: rank, or the "passed" sentinel.
    board_pick = np.where(ranked, rank, CONSENSUS_UNRANKED)
    board_pick = np.where(covered, board_pick, np.nan)
    cov_m = covered & np.isfinite(board_pick)

    # (a) success ranker — lower rank is better, so score = -rank.
    auc = float(roc_auc_score(y_success[cov_m], -board_pick[cov_m]))
    auc_ranked_only = float(roc_auc_score(y_success[ranked], -rank[ranked]))
    model_auc_same_rows = None  # filled by caller (needs the model probs)

    # (b) pick predictor — Spearman / top-64 / R1 recall via the shared
    # metric function; MAE only where the board actually names a pick.
    m = pick_m & cov_m
    pm = tm.pick_metrics(y_pick[m], board_pick[m])
    # Drafted = has a real pick. NOT `y_pick < UDFA_PICK`: the target is
    # exp(log(300)) = 299.99999999999994, so that strict test let every UDFA
    # row through (2026-09-09 review).
    drafted = drafted_mask(test)
    in_draft = pick_m & ranked & (rank <= LAST_DRAFT_PICK) & drafted
    mae_board = float(np.mean(np.abs(y_pick[in_draft] - rank[in_draft])))
    mae_model_same = float(np.mean(np.abs(y_pick[in_draft] - blend_pick[in_draft])))
    return {
        "source": ("consensus big board (WideLeft/ESPN staged boards, "
                   "consensus_rank in training_data/combine_outcomes.csv), "
                   "scored on the same holdout rows"),
        "n_covered": int(covered.sum()),
        "n_ranked": int(ranked.sum()),
        "unranked_rule": (f"unranked in a covered year = rank {int(CONSENSUS_UNRANKED)} "
                          "(worse than every ranked player)"),
        "auc": round(auc, 4),
        "auc_ranked_only": round(auc_ranked_only, 4),
        "n_auc": int(cov_m.sum()),
        "model_auc_same_rows": model_auc_same_rows,
        "pick": {
            "spearman_all": pm["spearman_all"],
            "spearman_top64": pm["spearman_top64"],
            "r1_recall_within_45": pm["r1_recall_within_45"],
            "n_scored": int(m.sum()),
            "mae_picks_drafted_ranked": round(mae_board, 1),
            "model_mae_same_rows": round(mae_model_same, 1),
            "n_mae": int(in_draft.sum()),
            "mae_rule": (f"drafted players the board ranked 1-{LAST_DRAFT_PICK}; "
                         "the model's MAE on those same rows is alongside"),
        },
    }


def forward_split_disclosure() -> dict | None:
    """The v3 experiments' deployment-realistic forward split (train
    2000-2023, cal 2024, test 2025-2026), read from the experiment log — the
    only forward-split numbers on record. None if the runs are missing."""
    try:
        with open(V3_RUNS_PATH) as fh:
            runs = {r["label"]: r for r in json.load(fh)}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # A shape change in the experiment log must omit the box, not abort the
    # whole generation.
    try:
        fwd = runs.get("+A+B+C+D|flat|fwd")
        frozen = runs.get("+A+B+C+D|flat|hist")
        fwd_v2 = runs.get("v2winner|flat|fwd")
        if not (fwd and frozen):
            return None
        return {
            "split": "train 2000-2023, calibrate 2024, test 2025-2026",
            "feature_stack": ("v3 winner (+A+B+C+D: base13 + position-normalized "
                              "measurables + recruiting + years-in-college + "
                              "all-position production + SP+) — BEFORE the v4 "
                              "consensus-board / all-star features and the v5 heads"),
            "n_test": int(fwd["n_test"]),
            "grade_accuracy_raw": float(fwd["raw"]["accuracy"]),
            "grade_accuracy_calibrated": float(fwd["calibrated"]["accuracy"]),
            "macro_f1_raw": float(fwd["raw"]["macro_f1"]),
            "frozen_grade_accuracy_same_features": float(frozen["raw"]["accuracy"]),
            "frozen_n_test": int(frozen["n_test"]),
            "v2_features_forward_raw": (float(fwd_v2["raw"]["accuracy"])
                                        if fwd_v2 else None),
            "source": os.path.relpath(V3_RUNS_PATH, REPO_ROOT),
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        print(f"WARNING: forward-split runs have an unexpected shape ({exc!r}); "
              "disclosure box will be omitted")
        return None


def verify_ledger(path: str, metrics: dict) -> None:
    """Re-derive interval coverage and pick Spearman from the CSV as written
    and check them against the published metrics — the ledger is the audit
    trail, so it must reproduce the headline numbers to 4 dp."""
    led = pd.read_csv(path)
    actual = led["actual_pick"].to_numpy(dtype=float)
    actual = np.where(np.isfinite(actual), actual, tm.UDFA_PICK)
    lo = led["pick_lo"].to_numpy(dtype=float)
    hi = led["pick_hi"].to_numpy(dtype=float)
    cov = round(float(np.mean((actual >= lo) & (actual <= hi))), 4)
    rho = round(float(spearmanr(actual, led["pred_pick"].to_numpy(dtype=float)).statistic), 4)
    pub_cov = metrics["pick_interval"]["coverage"]
    pub_rho = metrics["pick"]["served_blend"]["spearman_all"]
    ok = abs(cov - pub_cov) < 5e-5 and abs(rho - pub_rho) < 5e-5
    print(f"Ledger check: CSV coverage {cov:.4f} vs published {pub_cov:.4f}; "
          f"CSV Spearman {rho:.4f} vs published {pub_rho:.4f} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("ledger CSV does not reproduce the published metrics")


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

    # ── Consensus big board, same rows ───────────────────────────────────────
    cons = consensus_metrics(test, y_success, y_pick, pick_m, blend_pick)
    cov_m = test["consensus_covered"].to_numpy(dtype=float) == 1.0
    cons["model_auc_same_rows"] = round(float(roc_auc_score(y_success[cov_m], p[cov_m])), 4)
    # Apples-to-apples on only the players the board actually ranked (no
    # "passed" sentinel involved on either side).
    ranked_m = cov_m & np.isfinite(test["consensus_rank"].to_numpy(dtype=float))
    cons["model_auc_ranked_only"] = round(float(roc_auc_score(y_success[ranked_m], p[ranked_m])), 4)
    metrics["consensus"] = cons
    print(f"Consensus board: AUC {cons['auc']:.4f} on {cons['n_auc']} covered rows "
          f"({cons['n_ranked']} ranked; model AUC on the same rows "
          f"{cons['model_auc_same_rows']:.4f}); ranked-only AUC board "
          f"{cons['auc_ranked_only']:.4f} vs model {cons['model_auc_ranked_only']:.4f}")
    cp = cons["pick"]
    print(f"                 Spearman {cp['spearman_all']:.4f}  top64 {cp['spearman_top64']}  "
          f"R1-recall@45 {cp['r1_recall_within_45']}  "
          f"MAE(ranked<=262, drafted) {cp['mae_picks_drafted_ranked']:.1f} "
          f"vs model {cp['model_mae_same_rows']:.1f} on n={cp['n_mae']}")

    # ── 80% pick intervals: eval-phase recipe from train_models.py ──────────
    q10 = tm.fit_pick_quantile_members(train, 0.1)
    q90 = tm.fit_pick_quantile_members(train, 0.9)
    y_cal_pick = tm._pick_target(cal)
    cal_pm = np.isfinite(y_cal_pick)
    X_cal_p = cal[SUCCESS_FEATURES][cal_pm]
    lo_off, hi_off = tm.conformal_pick_offsets(
        tm.pick_quantile_log_preds(q10, X_cal_p),
        tm.pick_quantile_log_preds(q90, X_cal_p),
        y_cal_pick[cal_pm])
    pick_lo = np.clip(np.exp(tm.pick_quantile_log_preds(q10, X) - lo_off), 1.0, tm.UDFA_PICK)
    pick_hi = np.clip(np.exp(tm.pick_quantile_log_preds(q90, X) + hi_off), 1.0, tm.UDFA_PICK)
    interval = tm.interval_metrics(y_pick[pick_m], pick_lo[pick_m], pick_hi[pick_m],
                                   blend=blend_pick[pick_m])
    interval.update({"coverage_target": tm.PICK_INTERVAL_COVERAGE,
                     "lo_offset_log": round(lo_off, 4), "hi_offset_log": round(hi_off, 4),
                     "n_scored": int(pick_m.sum())})
    metrics["pick_interval"] = interval
    print(f"Intervals: coverage {interval['coverage']:.3f} (target "
          f"{tm.PICK_INTERVAL_COVERAGE}), median width {interval['median_width_picks']} picks")

    # ── Reliability (calibrated success probability, 10 bins) ───────────────
    metrics["reliability"] = tm.reliability_table(y_success, p)

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
        ("interval coverage", interval["coverage"],
         meta["pick_interval"]["eval"]["coverage"], 0.01),
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

    # ── Full ledger: every scored holdout row ────────────────────────────────
    df["pick_lo"] = pick_lo
    df["pick_hi"] = pick_hi
    ledger = []
    for _, row in df.sort_values("prob", ascending=False).iterrows():
        rnd = row.draft_round
        ledger.append({
            "name": str(row["name"]),
            "college": None if pd.isna(row.college) else str(row.college),
            "position": str(row.position),
            "draft_year": int(row.draft_year),
            "pred_success_prob": round(float(row.prob), 4),
            "pred_grade_bucket": DRAFT_GRADE_LABELS[int(row.g_pred)],
            # Two decimals, not integers: coverage / Spearman recomputed from
            # the CSV must reproduce metrics.pick_interval / metrics.pick.
            # The UI rounds for display.
            "pred_pick": round(float(row.pick_pred), 2),
            "pick_lo": round(float(row.pick_lo), 2),
            "pick_hi": round(float(row.pick_hi), 2),
            "consensus_rank": _opt_int(row.consensus_rank),
            "actual_pick": _opt_int(row.draft_pick),
            "actual_round": None if (pd.isna(rnd) or int(rnd) > 7) else int(rnd),
            "actual_success": int(row.nfl_success),
            "career_av": None if pd.isna(row.career_av) else round(float(row.career_av), 1),
        })
    assert len(ledger) == len(test)
    with open(LEDGER_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        w.writeheader()
        for rec in ledger:
            w.writerow({k: ("" if rec[k] is None else rec[k]) for k in LEDGER_COLS})
    print(f"Wrote {LEDGER_PATH} ({len(ledger)} rows)")
    verify_ledger(LEDGER_PATH, metrics)

    # ── Rolling-origin CV (verbatim from the experiment log) ────────────────
    rolling = None
    try:
        with open(ROLLING_CV_PATH) as fh:
            rc = json.load(fh)
        rolling = {
            "source": os.path.relpath(ROLLING_CV_PATH, REPO_ROOT),
            "feature_set": rc.get("feature_set"),
            "design": rc.get("design"),
            "folds": [{k: f.get(k) for k in (
                "test_year", "train_years", "cal_year", "n_train", "n_test",
                "test_success_rate", "success_auc", "success_brier", "grade_acc",
                "pick_mae_drafted", "pick_spearman_all", "pick_spearman_top64",
                "pick_r1_recall_45")} for f in rc["folds"]],
            "summary": {k: {"mean": v["mean"], "std": v["std"]}
                        for k, v in rc["summary"].items()},
        }
        print(f"Rolling CV: {len(rolling['folds'])} folds from {ROLLING_CV_PATH}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"WARNING: rolling CV not embedded ({exc})")

    forward = forward_split_disclosure()
    if forward:
        print(f"Forward split: grade acc {forward['grade_accuracy_raw']} raw / "
              f"{forward['grade_accuracy_calibrated']} calibrated on n={forward['n_test']} "
              f"(frozen {forward['frozen_grade_accuracy_same_features']}, same features)")
    else:
        print("WARNING: no forward-split runs found; disclosure box will be omitted")

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
        "ledger": ledger,
        "ledger_csv": "/data/backtest_predictions.csv",
        "rolling_cv": rolling,
        "forward_split": forward,
        "selection_note": (f"The {years[0]}-{years[-1]} holdout is also the split "
                           "every v2-v5 feature and architecture experiment was "
                           "selected on (models/experiments/RESULTS.md, "
                           "models/metadata.json evaluation.note). It was never "
                           "trained or calibrated on, but it was looked at "
                           "repeatedly, so its numbers carry selection optimism."),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
