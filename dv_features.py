#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared feature definitions for the DraftVision models.

Single source of truth imported by BOTH:
  - XGBOost.py            (Flask serving path — builds the same 13 features)
  - scripts/train_models.py (offline training + evaluation)

Everything here must stay import-side-effect free apart from reading the
static production-percentile JSON (no Flask, no DB, no network, no models).
Paths are anchored at the repo root so imports work from any cwd.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Feature list ──────────────────────────────────────────────────────────────
# Features shared by both models — purely college-performance based, NO draft_round.
# NOTE: accolade flags (is_award_winner / is_all_american) were removed — there is
# no CFBD access, so real training rows can never carry them; they existed only on
# hand-curated seeds and acted as a laundering channel. detect_accolades remains
# (in XGBOost.py) for display purposes only. 13 features: 4 numeric + 9 position flags.
_BASE_FEATURES = [
    "production_score",      # 0–100 empirical position-group percentile
    "games_played",
    "combine_speed_score",   # 0–100 (position-normalized 40-yard dash)
    "conference_tier",       # 1 (SEC elite) → 10 (FCS lower)
    # Position flags
    "position_qb", "position_rb", "position_wr", "position_te",
    "position_db", "position_lb", "position_dl", "position_ol", "position_other",
]

# Draft-grade model: predict which bracket a player will be drafted in
# Output classes: 0=Top50(R1-2), 1=Day2(R3-4), 2=LateRound(R5-7), 3=Undrafted
DRAFT_GRADE_FEATURES = _BASE_FEATURES

# Success model: predict NFL career success from college profile ONLY
# No draft_round — that's what we're trying to predict
SUCCESS_FEATURES = _BASE_FEATURES

DRAFT_GRADE_LABELS = ["Top 50 Pick", "Day 2 Pick", "Late Round Pick", "Undrafted Prospect"]

# ── Position flags ────────────────────────────────────────────────────────────
_DB_POSITIONS = {"CB", "S", "DB", "FS", "SS"}
_LB_POSITIONS = {"LB", "ILB", "OLB", "MLB"}
_DL_POSITIONS = {"DL", "DE", "DT", "NT", "EDGE"}
_OL_POSITIONS = {"OL", "OT", "OG", "G", "C", "LS"}
_KNOWN_POSITIONS = {"QB", "RB", "WR", "TE"} | _DB_POSITIONS | _LB_POSITIONS | _DL_POSITIONS | _OL_POSITIONS


def position_flags(position: str) -> Dict[str, int]:
    p = (position or "Unknown").upper()
    return {
        "position_qb":    int(p == "QB"),
        "position_rb":    int(p == "RB"),
        "position_wr":    int(p == "WR"),
        "position_te":    int(p == "TE"),
        "position_db":    int(p in _DB_POSITIONS),
        "position_lb":    int(p in _LB_POSITIONS),
        "position_dl":    int(p in _DL_POSITIONS),
        "position_ol":    int(p in _OL_POSITIONS),
        "position_other": int(p not in _KNOWN_POSITIONS),
    }


# ── Production percentile table (built by scripts/compute_production_percentiles.py) ──
PRODUCTION_PERCENTILES_PATH = os.path.join(
    _REPO_ROOT, "training_data", "production_percentiles.json")

# Position → percentile group. Keep in sync with scripts/compute_production_percentiles.py.
_PRODUCTION_GROUP_BY_POS = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL", "OT": "OL", "OG": "OL", "G": "OL", "C": "OL", "LS": "OL",
    "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
}


def _production_group(position: str) -> str:
    return _PRODUCTION_GROUP_BY_POS.get((position or "").strip().upper(), "OTHER")


def _load_production_percentiles() -> Dict[str, np.ndarray]:
    """Load sorted raw-composite arrays per position group. Empty dict on failure."""
    try:
        with open(PRODUCTION_PERCENTILES_PATH) as fh:
            data = json.load(fh)
        groups = data.get("groups", {}) or {}
        return {
            g: np.sort(np.asarray(vals, dtype=float))
            for g, vals in groups.items()
            if vals
        }
    except Exception as exc:
        print(f"Production percentile table unavailable ({exc}) — raw composites will be used")
        return {}


_PRODUCTION_PERCENTILES = _load_production_percentiles()


def _raw_production_to_percentile(group: str, raw: float) -> Optional[float]:
    """Mean-rank empirical percentile of `raw` within its position group, or None."""
    arr = _PRODUCTION_PERCENTILES.get(group)
    if arr is None or arr.size == 0:
        return None
    lo = int(np.searchsorted(arr, raw, side="left"))
    hi = int(np.searchsorted(arr, raw, side="right"))
    pct = (lo + hi) / 2.0 / arr.size * 100.0
    return float(min(100.0, max(0.0, pct)))


# ── Real historical players — hand-curated ground truth seed rows ─────────────
# Used ONLY by scripts/train_models.py (added once each, sample_weight=5).
# Fields: position, conference_tier (1-10), production_score (0-100),
#         combine_speed_score (0-100), games_played, is_award_winner, is_all_american,
#         draft_grade (0=Top50/R1-2, 1=Day2/R3-4, 2=LateRound/R5-7, 3=UDFA),
#         nfl_success (1=Pro Bowl or 5+ yr starter, 0=bust/journeyman)
SEED_TRAINING_PLAYERS = [
    # ── QBs ──────────────────────────────────────────────────────────────────
    {"position":"QB","conference_tier":1,"production_score":90,"combine_speed_score":70,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Burrow
    {"position":"QB","conference_tier":1,"production_score":85,"combine_speed_score":74,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Trevor Lawrence
    {"position":"QB","conference_tier":2,"production_score":88,"combine_speed_score":85,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Lamar Jackson (Louisville T2)
    {"position":"QB","conference_tier":3,"production_score":76,"combine_speed_score":74,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":0,"nfl_success":1},  # Josh Allen (Wyoming T3)
    {"position":"QB","conference_tier":2,"production_score":82,"combine_speed_score":78,"games_played":14,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Kyler Murray
    {"position":"QB","conference_tier":2,"production_score":80,"combine_speed_score":67,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Jalen Hurts
    {"position":"QB","conference_tier":1,"production_score":82,"combine_speed_score":65,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Caleb Williams
    {"position":"QB","conference_tier":4,"production_score":72,"combine_speed_score":62,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":1},  # Brock Purdy (Iowa St, R7!)
    {"position":"QB","conference_tier":2,"production_score":78,"combine_speed_score":72,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":0,"nfl_success":0},  # R1 QB bust
    {"position":"QB","conference_tier":3,"production_score":68,"combine_speed_score":60,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":0},  # Day2 bust
    {"position":"QB","conference_tier":4,"production_score":60,"combine_speed_score":55,"games_played":10,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"QB","conference_tier":5,"production_score":58,"combine_speed_score":58,"games_played":10,"is_award_winner":0,"is_all_american":0,"draft_grade":3,"nfl_success":0},
    # ── WRs ──────────────────────────────────────────────────────────────────
    {"position":"WR","conference_tier":2,"production_score":88,"combine_speed_score":93,"games_played":14,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Justin Jefferson (LSU)
    {"position":"WR","conference_tier":1,"production_score":85,"combine_speed_score":90,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Ja'Marr Chase
    {"position":"WR","conference_tier":1,"production_score":86,"combine_speed_score":79,"games_played":14,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Devonta Smith
    {"position":"WR","conference_tier":1,"production_score":80,"combine_speed_score":83,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # CeeDee Lamb
    {"position":"WR","conference_tier":2,"production_score":78,"combine_speed_score":80,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":1},  # Amon-Ra St. Brown (USC T2)
    {"position":"WR","conference_tier":8,"production_score":82,"combine_speed_score":96,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":1},  # Tyreek Hill (West Alabama T8)
    {"position":"WR","conference_tier":1,"production_score":82,"combine_speed_score":87,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # A.J. Brown
    {"position":"WR","conference_tier":2,"production_score":65,"combine_speed_score":72,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"WR","conference_tier":3,"production_score":70,"combine_speed_score":68,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"WR","conference_tier":4,"production_score":62,"combine_speed_score":75,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":3,"nfl_success":0},
    # ── RBs ──────────────────────────────────────────────────────────────────
    {"position":"RB","conference_tier":3,"production_score":90,"combine_speed_score":82,"games_played":14,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Saquon Barkley (Penn St T3)
    {"position":"RB","conference_tier":3,"production_score":88,"combine_speed_score":88,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # McCaffrey (Stanford T3)
    {"position":"RB","conference_tier":2,"production_score":85,"combine_speed_score":80,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Bijan Robinson
    {"position":"RB","conference_tier":1,"production_score":80,"combine_speed_score":78,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":1},  # Jahmyr Gibbs
    {"position":"RB","conference_tier":7,"production_score":88,"combine_speed_score":85,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Ashton Jeanty (Boise St T7)
    {"position":"RB","conference_tier":1,"production_score":75,"combine_speed_score":75,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":0},
    {"position":"RB","conference_tier":4,"production_score":78,"combine_speed_score":77,"games_played":13,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"RB","conference_tier":5,"production_score":72,"combine_speed_score":80,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    # ── TEs ──────────────────────────────────────────────────────────────────
    {"position":"TE","conference_tier":4,"production_score":80,"combine_speed_score":68,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":1,"nfl_success":1},  # Travis Kelce (Cincy T4)
    {"position":"TE","conference_tier":2,"production_score":82,"combine_speed_score":72,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Kyle Pitts (Florida T2)
    {"position":"TE","conference_tier":3,"production_score":78,"combine_speed_score":65,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # George Kittle (Iowa T3)
    {"position":"TE","conference_tier":3,"production_score":75,"combine_speed_score":62,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Tyler Warren (Penn St)
    {"position":"TE","conference_tier":2,"production_score":70,"combine_speed_score":60,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":0},
    {"position":"TE","conference_tier":4,"production_score":60,"combine_speed_score":55,"games_played":10,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    # ── CBs ──────────────────────────────────────────────────────────────────
    {"position":"CB","conference_tier":1,"production_score":78,"combine_speed_score":91,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Patrick Surtain II
    {"position":"CB","conference_tier":4,"production_score":75,"combine_speed_score":89,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Sauce Gardner (Cincinnati T4)
    {"position":"CB","conference_tier":1,"production_score":72,"combine_speed_score":87,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Devon Witherspoon
    {"position":"CB","conference_tier":1,"production_score":68,"combine_speed_score":85,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":1},  # Day2 CB success
    {"position":"CB","conference_tier":3,"production_score":65,"combine_speed_score":80,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":0},
    {"position":"CB","conference_tier":5,"production_score":58,"combine_speed_score":78,"games_played":10,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    # ── Safeties ─────────────────────────────────────────────────────────────
    {"position":"S","conference_tier":2,"production_score":80,"combine_speed_score":82,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Kyle Hamilton (ND T2)
    {"position":"S","conference_tier":1,"production_score":75,"combine_speed_score":80,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Caleb Downs profile (Bama T1)
    {"position":"S","conference_tier":1,"production_score":72,"combine_speed_score":78,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},
    {"position":"S","conference_tier":2,"production_score":65,"combine_speed_score":75,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":0},
    {"position":"S","conference_tier":4,"production_score":60,"combine_speed_score":72,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    # ── LBs ──────────────────────────────────────────────────────────────────
    {"position":"LB","conference_tier":3,"production_score":82,"combine_speed_score":78,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Micah Parsons (PSU T3)
    {"position":"LB","conference_tier":1,"production_score":85,"combine_speed_score":80,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Will Anderson (Bama T1)
    {"position":"LB","conference_tier":1,"production_score":78,"combine_speed_score":74,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Roquan Smith (Georgia)
    {"position":"LB","conference_tier":3,"production_score":75,"combine_speed_score":70,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":1,"nfl_success":1},
    {"position":"LB","conference_tier":4,"production_score":65,"combine_speed_score":62,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"LB","conference_tier":6,"production_score":60,"combine_speed_score":58,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":3,"nfl_success":0},
    # ── DLs ──────────────────────────────────────────────────────────────────
    {"position":"DL","conference_tier":1,"production_score":88,"combine_speed_score":82,"games_played":13,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Chase Young (OSU T1)
    {"position":"DL","conference_tier":2,"production_score":85,"combine_speed_score":78,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Myles Garrett (TAMU T2)
    {"position":"DL","conference_tier":1,"production_score":82,"combine_speed_score":80,"games_played":12,"is_award_winner":1,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # Jalen Carter (Georgia T1)
    {"position":"DL","conference_tier":1,"production_score":80,"combine_speed_score":77,"games_played":12,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},
    {"position":"DL","conference_tier":3,"production_score":65,"combine_speed_score":65,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"DL","conference_tier":5,"production_score":58,"combine_speed_score":60,"games_played":10,"is_award_winner":0,"is_all_american":0,"draft_grade":3,"nfl_success":0},
    # ── OLs ──────────────────────────────────────────────────────────────────
    {"position":"OL","conference_tier":1,"production_score":55,"combine_speed_score":52,"games_played":14,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},  # elite T1 OT
    {"position":"OL","conference_tier":2,"production_score":52,"combine_speed_score":50,"games_played":13,"is_award_winner":0,"is_all_american":1,"draft_grade":0,"nfl_success":1},
    {"position":"OL","conference_tier":3,"production_score":45,"combine_speed_score":44,"games_played":12,"is_award_winner":0,"is_all_american":0,"draft_grade":2,"nfl_success":0},
    {"position":"OL","conference_tier":5,"production_score":38,"combine_speed_score":40,"games_played":11,"is_award_winner":0,"is_all_american":0,"draft_grade":3,"nfl_success":0},
]
