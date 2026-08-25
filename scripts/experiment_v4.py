#!/usr/bin/env python3
"""v4 feature-wave experiment — EVAL ONLY, frozen 2019-20 holdout.

Candidate features on top of the 29 v3 features:
  consensus_logrank — log(scout consensus rank); unranked-in-covered-year =
                      log(400) (signal: boards passed on him); uncovered = NaN
  allstar_invite    — senior-bowl-circuit invite flag (coverage-gated)

Gates (ALL must pass to promote v4):
  success: AUC(v4) > AUC(v3) on the calibrated ensemble
  success: AUC(v4) > AUC(consensus-rank-only baseline)
  pick:    Spearman(v4 blend) >= Spearman(v3 blend)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

import train_models as T


def add_v4_features(df: pd.DataFrame, raw_csv: pd.DataFrame) -> pd.DataFrame:
    """Attach consensus/allstar features by (name, draft_year) from the CSV."""
    key = raw_csv.apply(lambda r: (str(r["name"]).lower().strip(), int(r["draft_year"])), axis=1)
    cov = dict(zip(key, raw_csv["consensus_covered"]))
    rnk = dict(zip(key, raw_csv["consensus_rank"]))
    als = dict(zip(key, raw_csv["allstar_invite"]))

    def logrank(k):
        c = cov.get(k)
        try:
            if int(float(c)) != 1:
                return np.nan
        except (TypeError, ValueError):
            return np.nan
        try:
            return np.log(float(rnk[k]))
        except (TypeError, ValueError, KeyError):
            return np.log(400.0)

    def invite(k):
        try:
            return float(als[k])
        except (TypeError, ValueError, KeyError):
            return np.nan

    keys = df.apply(lambda r: (str(r["name"]).lower().strip(), int(r["draft_year"])), axis=1)
    df = df.copy()
    df["consensus_logrank"] = [logrank(k) for k in keys]
    df["allstar_invite"] = [invite(k) for k in keys]
    return df


def run(features):
    T.SUCCESS_FEATURES = features
    train = pd.concat([DF[DF.draft_year.isin(T.EVAL_TRAIN_YEARS)], SEEDS], ignore_index=True)
    cal = DF[DF.draft_year.isin(T.CAL_YEARS)].reset_index(drop=True)
    test = DF[DF.draft_year.isin(T.TEST_YEARS)].reset_index(drop=True)
    s = T.fit_success_members(train)
    s["calibrator"] = T.fit_success_calibrator(s, cal, T.CAL_YEARS)
    g = T.fit_grade_members(train)
    p = T.fit_pick_members(train)
    X = test[features]
    y = test.nfl_success.to_numpy()
    prob = T.ensemble_success_probs(s, X, calibrated=True)
    y_pick = np.exp(T._pick_target(test)); m = np.isfinite(y_pick)
    reg = T.ensemble_pick_preds(p, X[m])
    P = np.mean([g["xgb"].predict_proba(X[m]), g["cb"].predict_proba(X[m])], axis=0)
    blend = np.exp(0.5 * (np.log(reg) + np.log(T.classifier_expected_pick(P))))
    return {
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
        "pick_rho": round(float(spearmanr(y_pick[m], blend).statistic), 4),
        "pick_rho_top64": round(float(spearmanr(y_pick[m][y_pick[m] <= 64], blend[y_pick[m] <= 64]).statistic), 4),
    }


np.random.seed(T.SEED)
RAW_CSV = pd.read_csv(T.TRAINING_DATA_PATH)
raw = T.load_raw_rows()
SEEDS = T.seed_rows()
for c in ("consensus_logrank", "allstar_invite"):
    SEEDS[c] = np.nan
base_stats = T.stats_from_ref(raw, T.EVAL_REF_YEARS)
DF = add_v4_features(T.apply_z(raw, base_stats), RAW_CSV)

V3 = list(T.SUCCESS_FEATURES)
V4 = V3 + ["consensus_logrank", "allstar_invite"]

r3 = run(V3)
r4 = run(V4)

# consensus-only baseline on the holdout (covered rows all ranked or 400)
test = DF[DF.draft_year.isin(T.TEST_YEARS)]
score = -test["consensus_logrank"].to_numpy()
y = test.nfl_success.to_numpy()
mm = np.isfinite(score)
consensus_auc = round(float(roc_auc_score(y[mm], score[mm])), 4)

print(f"v3 (29 feats):        {r3}")
print(f"v4 (+consensus+star): {r4}")
print(f"consensus-only AUC:   {consensus_auc} (n={int(mm.sum())})")
gates = {
    "auc_beats_v3": r4["auc"] > r3["auc"],
    "auc_beats_consensus": r4["auc"] > consensus_auc,
    "pick_rho_noninferior": r4["pick_rho"] >= r3["pick_rho"],
}
print("GATES:", gates, "→ PROMOTE" if all(gates.values()) else "→ REJECT")
