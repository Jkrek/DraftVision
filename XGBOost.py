#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Future Star Predictor backend.

- Provides health and prediction endpoints.
- Uses a direct ML success classifier for Success/No Success output.
- Falls back to rule scoring only if model inference is unavailable.
"""

import functools
import json
import math
import os
import sqlite3
import time
import hashlib
import threading
import unicodedata
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from flask import Flask, has_request_context, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

# Shared, side-effect-free modules (also imported by scripts/train_models.py so
# the served features/heuristics can never drift from the trained/evaluated ones)
from dv_features import (
    _BASE_FEATURES,
    DRAFT_GRADE_FEATURES,
    SUCCESS_FEATURES,
    DRAFT_GRADE_LABELS,
    MEASURABLE_COLS,
    position_flags,
    _production_group,
    _composite_group,
    _raw_production_to_percentile,
    measurable_z,
    composite_z,
)
from dv_heuristics import (  # noqa: F401 — rule-based fallback + baseline source of truth
    success_prob_from_college_profile,
    draft_grade_from_profile,
)

# CatBoost is a required ensemble member — imported at module level so startup
# clearly shows status
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed (pip install catboost) — XGBoost-only mode")

POSITION_MODEL_PATH        = "nfl_xgboost_model.json"
ENCODER_PATH               = "label_encoders.pkl"
SUCCESS_MODEL_PATH         = "success_xgboost_model.json"        # raw XGBoost ensemble member
SUCCESS_CALIBRATED_PATH    = "success_calibrated_model.pkl"      # ENSEMBLE calibrator dict (see scripts/train_models.py)
CATBOOST_SUCCESS_PATH      = "catboost_success_model.cbm"        # CatBoost ensemble member
DRAFT_GRADE_MODEL_PATH     = "draft_grade_model.json"            # raw XGBoost ensemble member
DRAFT_GRADE_CALIBRATED_PATH = "draft_grade_calibrated_model.pkl" # ENSEMBLE calibrator dict
CATBOOST_DRAFT_GRADE_PATH  = "catboost_draft_grade_model.cbm"    # CatBoost ensemble member
TRAINING_DATA_PATH         = "training_data/combine_outcomes.csv"
PROSPECT_CACHE_PATH        = "training_data/prospect_cache.json"
MOCK_DRAFT_PATH            = "mock_draft.json"
HS_PROSPECT_CACHE_PATH     = "training_data/hs_prospect_cache.json"
ENRICHMENT_PATH            = "training_data/enrichment.json"    # scripts/build_enrichment.py
BOARD_MOVERS_PATH          = "training_data/board_movers.json"
BIG_BOARDS_PATH            = "training_data/big_boards.json"     # hand-ordered per-class boards
CFBD_API_KEY               = os.environ.get("CFBD_API_KEY", "")
CFBD_BASE_URL              = "https://api.collegefootballdata.com"
PLAYER_DB_PATH       = "players.db"
ESPN_CFB_TEAMS_URL        = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
ESPN_CFB_TEAM_ROSTER_URL  = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"
ESPN_CFB_ATHLETE_OVERVIEW_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{espn_id}/overview"
ESPN_NFL_CORE_ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{espn_id}"
HTTP_TIMEOUT_SECONDS = 12
STATS_CACHE_TTL = 3600  # cache real stats for 1 hour

# Feature lists (_BASE_FEATURES / SUCCESS_FEATURES / DRAFT_GRADE_FEATURES) and
# DRAFT_GRADE_LABELS now live in dv_features.py — shared with scripts/train_models.py.

# Power 5 conference schools (partial list for tier classification)
POWER5_SCHOOLS = {
    "alabama", "georgia", "ohio state", "michigan", "clemson", "lsu", "oklahoma",
    "texas", "notre dame", "penn state", "florida", "auburn", "tennessee", "oregon",
    "washington", "usc", "ucla", "stanford", "miami", "florida state", "nebraska",
    "iowa", "wisconsin", "minnesota", "purdue", "illinois", "indiana", "northwestern",
    "rutgers", "maryland", "michigan state", "kansas state", "iowa state", "baylor",
    "tcu", "west virginia", "texas tech", "kansas", "oklahoma state", "cincinnati",
    "pittsburgh", "virginia", "virginia tech", "north carolina", "nc state", "duke",
    "wake forest", "syracuse", "boston college", "louisville", "kentucky", "vanderbilt",
    "south carolina", "mississippi", "ole miss", "mississippi state", "arkansas",
    "texas a&m", "missouri", "colorado", "utah", "arizona state", "arizona", "cal",
    "oregon state", "washington state", "cal poly",
}

# Tier classification lives in dv_tiers.py (shared with the cache builder).
from dv_tiers import (  # noqa: F401
    NFL_FRANCHISE_NAMES, _normalize_team, is_nfl_franchise,
    _TIER_SCHOOLS, _SCHOOL_TIERS, _SCHOOL_KEYS, classify_college_tier,
)


def forty_to_speed_score(position: str, forty: float) -> float:
    """Convert raw 40-yard dash time to 0–100 position-normalized score. 100=elite."""
    if not forty or forty <= 0:
        return 0.0  # unknown — caller should fall back to estimate
    p = (position or "").upper()
    # (elite_time, poor_time) → maps to (100, 0)
    benchmarks = {
        "QB":  (4.30, 5.10), "RB":  (4.20, 4.80), "WR":  (4.20, 4.70),
        "TE":  (4.40, 5.00), "CB":  (4.20, 4.65), "S":   (4.30, 4.75),
        "DB":  (4.25, 4.70), "LB":  (4.35, 4.85), "DL":  (4.50, 5.30),
        "DE":  (4.45, 5.10), "DT":  (4.55, 5.35), "EDGE":(4.45, 5.10),
        "OL":  (4.70, 5.55), "OT":  (4.75, 5.60), "OG":  (4.80, 5.55), "C": (4.85, 5.60),
    }
    elite_t, poor_t = benchmarks.get(p, (4.35, 5.00))
    score = (poor_t - forty) / (poor_t - elite_t) * 100.0
    return float(max(0.0, min(100.0, score)))


# ── Known award winners & All-Americans (drives is_award_winner / is_all_american) ──
_AWARD_WINNERS = {
    # Heisman Trophy winners
    "caleb williams", "bryce young", "joe burrow", "jalen hurts", "devonta smith",
    "kyler murray", "baker mayfield", "lamar jackson", "marcus mariota",
    "jameis winston", "johnny manziel", "robert griffin", "cam newton",
    "mark ingram", "sam bradford", "tim tebow", "troy smith", "matt leinart",
    "jason white", "eric crouch", "chris weinke", "ron dayne", "ricky williams",
    "charles woodson", "danny wuerffel", "travis hunter", "ashton jeanty",
    # Nagurski / Bednarik / defensive awards
    "chase young", "myles garrett", "khalil mack", "micah parsons",
    "nick bosa", "joey bosa", "will anderson", "jalen carter",
    # Outland / Rimington
    "mason graham", "will campbell",
}

_ALL_AMERICANS = {
    "travis hunter", "ashton jeanty", "tetairoa mcmillan", "emeka egbuka",
    "will campbell", "mason graham", "shedeur sanders", "cam ward",
    "tyler warren", "kelvin banks", "darius robinson", "laiatu latu",
    "caleb downs", "malaki starks", "nick herbig",
    "will anderson", "bralen trice", "jalen carter", "devonta smith",
    "justyn ross", "rashee rice", "quentin johnston",
    "patrick surtain", "sauce gardner", "kyle hamilton",
    "ja'marr chase", "justin jefferson", "ceedee lamb",
    "saquon barkley", "bijan robinson", "christian mccaffrey",
    "george kittle", "travis kelce", "kyle pitts",
    "trevor lawrence", "joe burrow", "kyler murray",
}


def detect_accolades(name: str) -> Dict[str, int]:
    """Return is_award_winner / is_all_american flags — DISPLAY ONLY, not ML features.

    Exact normalized-name match only. Bidirectional substring matching wrongly
    flagged any player sharing a famous name (e.g. a Notre Dame "DeVonta Smith"
    matched the Alabama Heisman winner) and empty strings matched everything.
    """
    n = " ".join((name or "").lower().split())
    if not n:
        return {"is_award_winner": 0, "is_all_american": 0}
    return {
        "is_award_winner": int(n in _AWARD_WINNERS),
        "is_all_american": int(n in _ALL_AMERICANS),
    }


def combine_speed_for_position(position: str, seed: int) -> float:
    """Estimate 0–100 combine speed score deterministically when no real 40 time exists."""
    p = (position or "").upper()
    ranges = {
        "QB": (55, 20), "RB": (60, 20), "WR": (62, 18), "TE": (50, 20),
        "CB": (62, 18), "S":  (58, 18), "LB": (52, 18), "DL": (48, 18),
        "DE": (50, 18), "OL": (45, 15),
    }
    mean, _ = ranges.get(p, (50, 20))
    offset = sum(ord(c) for c in p) if p else 0
    raw = mean + ((seed + offset) % 41) - 20
    return float(max(0.0, min(100.0, raw)))


# Production percentile mapping (_production_group / _raw_production_to_percentile)
# now lives in dv_features.py — shared with scripts/train_models.py.


def compute_production_score(position: str, stats: dict) -> float:
    """Production score: raw per-position composite mapped to its empirical
    position-group percentile (0–100).

    The percentile mapping (via training_data/production_percentiles.json,
    built from 9k+ graded FBS players) removes the raw composite's saturation
    wall at 100.0 and ranks OL against other OL rather than against skill
    positions. Falls back to the raw composite if the table is missing.
    Returns NaN when games_played is unknown (missing ≠ average).
    """
    p = (position or "").upper()
    raw = _raw_production_composite(p, stats)
    if not math.isfinite(raw):
        return float("nan")
    pct = _raw_production_to_percentile(_production_group(p), raw)
    if pct is None:
        return raw
    return pct


def _raw_production_composite(p: str, stats: dict) -> float:
    """Raw 0–100 composite production score (internal math, pre-percentile)."""
    def _stat(key: str) -> float:
        v = stats.get(key)
        return float(v) if v is not None else 0.0

    games_raw = stats.get("games_played")
    games = float(games_raw) if games_raw is not None else float("nan")
    if not math.isfinite(games) or games <= 0:
        return float("nan")
    games = max(games, 1.0)
    pass_td  = _stat("passing_touchdowns")
    pass_yds = _stat("passing_yards")
    rush_td  = _stat("rushing_touchdowns")
    rush_yds = _stat("rushing_yards")
    tackles  = _stat("tackles")
    sacks    = _stat("sacks")
    ints     = _stat("interceptions")
    pds      = _stat("pass_deflections")

    if p == "QB":
        td_rate = (pass_td / games) / (35 / 15)
        yd_rate = (pass_yds / games) / (4500 / 15)
        score = (td_rate * 0.5 + yd_rate * 0.5) * 100
    elif p == "RB":
        td_rate = (rush_td / games) / (14 / 15)
        yd_rate = (rush_yds / games) / (1500 / 15)
        score = (td_rate * 0.45 + yd_rate * 0.55) * 100
    elif p in {"WR", "TE"}:
        combined_td  = (pass_td + rush_td) / games / (12 / 15)
        combined_yds = (pass_yds + rush_yds) / games / (1200 / 15)
        score = (combined_td * 0.4 + combined_yds * 0.6) * 100
    elif p in {"LB", "ILB", "OLB", "MLB"}:
        # ~100 tackles, ~8 sacks, ~3 INTs per elite LB season
        t_rate = (tackles / games) / (100 / 13)
        s_rate = (sacks / games) / (8 / 13)
        i_rate = (ints / games) / (3 / 13)
        score = (t_rate * 0.55 + s_rate * 0.30 + i_rate * 0.15) * 100
    elif p in {"CB", "S", "DB", "FS", "SS"}:
        # ~55 tackles, ~4 INTs, ~12 PDs per elite DB season
        t_rate = (tackles / games) / (55 / 13)
        i_rate = (ints / games) / (4 / 13)
        p_rate = (pds / games) / (12 / 13)
        score = (t_rate * 0.25 + i_rate * 0.40 + p_rate * 0.35) * 100
    elif p in {"DL", "DE", "DT", "NT", "EDGE"}:
        # ~45 tackles, ~10 sacks per elite DL season
        t_rate = (tackles / games) / (45 / 13)
        s_rate = (sacks / games) / (10 / 13)
        score = (t_rate * 0.35 + s_rate * 0.65) * 100
    elif p in {"OL", "OT", "OG", "G", "C", "LS"}:
        # OL: no counting stats — games played is the only durability proxy.
        # The raw cap at 60 is fine: percentile mapping ranks OL against OL.
        score = min(games / 13.0, 1.0) * 60.0
    else:
        combined_td  = (pass_td + rush_td) / games / (12 / 15)
        combined_yds = (pass_yds + rush_yds) / games / (1200 / 15)
        score = (combined_td * 0.4 + combined_yds * 0.6) * 100

    return float(max(0.0, min(100.0, score)))

# ── Database abstraction (SQLite locally, Postgres in production) ─────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Railway injects postgres:// — psycopg2 needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

    def _get_conn():
        return psycopg2.connect(DATABASE_URL)

    def _placeholder():
        return "%s"  # Postgres uses %s

    print(f"Using Postgres: {DATABASE_URL[:40]}…")
else:
    def _get_conn():
        return sqlite3.connect(PLAYER_DB_PATH)

    def _placeholder():
        return "?"  # SQLite uses ?

    print("Using SQLite (local dev)")


def _rows_as_dicts(cursor) -> list:
    """Convert fetchall() results to list of dicts for both SQLite and Postgres."""
    cols = [d[0] for d in (cursor.description or [])]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _row_as_dict(cursor, row) -> Optional[dict]:
    """Convert a single fetchone() row to a dict, or None."""
    if row is None:
        return None
    cols = [d[0] for d in (cursor.description or [])]
    return dict(zip(cols, row))


app = Flask(__name__, static_folder=None)  # catch-all serves build/


def _parse_allowed_origins(raw_value: str) -> list[str] | str:
    cleaned = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    if not cleaned:
        return "*"
    if len(cleaned) == 1 and cleaned[0] == "*":
        return "*"
    return cleaned


allowed_origins = _parse_allowed_origins(os.getenv("FRONTEND_ORIGIN", "*"))
CANONICAL_HOST = os.getenv("CANONICAL_HOST", "").strip().lower()
FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").strip().lower() in {"1", "true", "yes", "on"}
LOCAL_HOSTS = {"localhost", "127.0.0.1"}

CORS(app, resources={r"/*": {"origins": allowed_origins}})

# Response compression (gzip/brotli) — the prospect board JSON is ~1-2MB raw
# and compresses ~10x. Optional dependency: the app must boot without it.
try:
    from flask_compress import Compress
    Compress(app)
    print("flask-compress enabled (gzip/br responses)")
except ImportError:
    print("flask-compress not installed — responses served uncompressed")


@app.before_request
def enforce_canonical_origin():
    """Redirect production traffic to the configured canonical host and HTTPS.

    Only active when CANONICAL_HOST is configured, and only ever redirects TO
    that host. X-Forwarded-Host is attacker-controlled — echoing it into a
    redirect target is an open redirect.
    """
    if request.method == "OPTIONS" or not CANONICAL_HOST:
        return None

    forwarded_host = request.headers.get("X-Forwarded-Host", request.host or "")
    host = forwarded_host.split(",")[0].strip().split(":")[0].lower()
    if not host or host in LOCAL_HOSTS:
        return None

    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme or "http")
    scheme = forwarded_proto.split(",")[0].strip().lower()

    target_scheme = "https" if FORCE_HTTPS else scheme

    if host == CANONICAL_HOST and scheme == target_scheme:
        return None

    query = request.query_string.decode("utf-8") if request.query_string else ""
    destination = f"{target_scheme}://{CANONICAL_HOST}{request.path}"
    if query:
        destination = f"{destination}?{query}"
    return redirect(destination, code=308)


position_model = None
label_encoders = None
success_model = None            # raw XGBoost ensemble member
catboost_success_model = None   # CatBoost ensemble member
success_calibrator = None       # ensemble calibrator dict (applied to the member mean)
draft_grade_model = None        # raw XGBoost ensemble member (multiclass)
catboost_draft_grade_model = None
draft_grade_calibrator = None   # ensemble calibrator dict (applied to the member mean)

SEED_PLAYERS = [
    # ── Active NFL stars (for demo / search) ────────────────────────────────
    {"name": "Patrick Mahomes",    "position": "QB", "team": "Kansas City Chiefs",       "jersey": 15},
    {"name": "Josh Allen",         "position": "QB", "team": "Buffalo Bills",             "jersey": 17},
    {"name": "Joe Burrow",         "position": "QB", "team": "Cincinnati Bengals",        "jersey": 9},
    {"name": "Lamar Jackson",      "position": "QB", "team": "Baltimore Ravens",          "jersey": 8},
    {"name": "C.J. Stroud",        "position": "QB", "team": "Houston Texans",            "jersey": 7},
    {"name": "Jalen Hurts",        "position": "QB", "team": "Philadelphia Eagles",       "jersey": 1},
    {"name": "Brock Purdy",        "position": "QB", "team": "San Francisco 49ers",       "jersey": 13},
    {"name": "Saquon Barkley",     "position": "RB", "team": "Philadelphia Eagles",       "jersey": 26},
    {"name": "Christian McCaffrey","position": "RB", "team": "San Francisco 49ers",       "jersey": 23},
    {"name": "Bijan Robinson",     "position": "RB", "team": "Atlanta Falcons",           "jersey": 7},
    {"name": "Jahmyr Gibbs",       "position": "RB", "team": "Detroit Lions",             "jersey": 26},
    {"name": "Derrick Henry",      "position": "RB", "team": "Baltimore Ravens",          "jersey": 22},
    {"name": "Justin Jefferson",   "position": "WR", "team": "Minnesota Vikings",         "jersey": 18},
    {"name": "Tyreek Hill",        "position": "WR", "team": "Miami Dolphins",            "jersey": 10},
    {"name": "CeeDee Lamb",        "position": "WR", "team": "Dallas Cowboys",            "jersey": 88},
    {"name": "Amon-Ra St. Brown",  "position": "WR", "team": "Detroit Lions",             "jersey": 14},
    {"name": "Puka Nacua",         "position": "WR", "team": "Los Angeles Rams",          "jersey": 17},
    {"name": "Ja'Marr Chase",      "position": "WR", "team": "Cincinnati Bengals",        "jersey": 1},
    {"name": "A.J. Brown",         "position": "WR", "team": "Philadelphia Eagles",       "jersey": 11},
    {"name": "Davante Adams",      "position": "WR", "team": "Las Vegas Raiders",         "jersey": 17},
    {"name": "Travis Kelce",       "position": "TE", "team": "Kansas City Chiefs",        "jersey": 87},
    {"name": "Sam LaPorta",        "position": "TE", "team": "Detroit Lions",             "jersey": 87},
    {"name": "George Kittle",      "position": "TE", "team": "San Francisco 49ers",       "jersey": 85},
    {"name": "Mark Andrews",       "position": "TE", "team": "Baltimore Ravens",          "jersey": 89},
    # ── 2025 NFL Draft top prospects ────────────────────────────────────────
    {"name": "Cam Ward",           "position": "QB", "team": "Miami Hurricanes",          "jersey": 1,  "espn_id": "4432865"},
    {"name": "Shedeur Sanders",    "position": "QB", "team": "Colorado Buffaloes",        "jersey": 2},
    {"name": "Dillon Gabriel",     "position": "QB", "team": "Oregon Ducks",              "jersey": 8,  "espn_id": "4360939"},
    {"name": "Travis Hunter",      "position": "WR", "team": "Colorado Buffaloes",        "jersey": 12},
    {"name": "Tetairoa McMillan",  "position": "WR", "team": "Arizona Wildcats",          "jersey": 4,  "espn_id": "4685751"},
    {"name": "Emeka Egbuka",       "position": "WR", "team": "Ohio State Buckeyes",       "jersey": 2,  "espn_id": "4567048"},
    {"name": "Luther Burden",      "position": "WR", "team": "Missouri Tigers",           "jersey": 3,  "espn_id": "4685299"},
    {"name": "Ashton Jeanty",      "position": "RB", "team": "Boise State Broncos",       "jersey": 2,  "espn_id": "4685865"},
    {"name": "Omarion Hampton",    "position": "RB", "team": "North Carolina Tar Heels",  "jersey": 8,  "espn_id": "4432751"},
    {"name": "RJ Harvey",          "position": "RB", "team": "UCF Knights",               "jersey": 20, "espn_id": "4685928"},
    {"name": "Kaleb Johnson",      "position": "RB", "team": "Iowa Hawkeyes",             "jersey": 2,  "espn_id": "4685898"},
    {"name": "Tyler Warren",       "position": "TE", "team": "Penn State Nittany Lions",  "jersey": 44, "espn_id": "4686066"},
    {"name": "Grey Zinter",        "position": "TE", "team": "Michigan Wolverines",       "jersey": 86, "espn_id": "4686100"},
    {"name": "Arch Manning",       "position": "QB", "team": "Texas Longhorns",           "jersey": 16, "espn_id": "4870906"},
    {"name": "Dante Moore",        "position": "QB", "team": "Oregon Ducks",              "jersey": 2,  "espn_id": "4870921"},
    # ── 2026 Freshmen (2025 recruiting class) ────────────────────────────────
    {"name": "Bryce Underwood",    "position": "QB", "team": "Michigan Wolverines",       "jersey": 10},
    {"name": "Jared Curtis",       "position": "QB", "team": "Ohio State Buckeyes",       "jersey": 10},
    {"name": "Husan Longstreet",   "position": "QB", "team": "Georgia Bulldogs",          "jersey": 10},
    {"name": "Jake Merklinger",    "position": "QB", "team": "Alabama Crimson Tide",      "jersey": 10},
    {"name": "Julian Lewis",       "position": "QB", "team": "USC Trojans",               "jersey": 10},
    {"name": "Ty Haywood",         "position": "RB", "team": "Alabama Crimson Tide",      "jersey": 2},
    {"name": "Jordon Davison",     "position": "RB", "team": "Georgia Bulldogs",          "jersey": 4},
    {"name": "Harlem Berry",       "position": "RB", "team": "Tennessee Volunteers",      "jersey": 3},
    {"name": "Elijah Rushing",     "position": "RB", "team": "Oregon Ducks",              "jersey": 5},
    {"name": "Savion Hiter",       "position": "RB", "team": "Texas Longhorns",           "jersey": 3},
    {"name": "Dakorien Moore",     "position": "WR", "team": "Oregon Ducks",              "jersey": 1},
    {"name": "Nate Marshall",      "position": "WR", "team": "Georgia Bulldogs",          "jersey": 5},
    {"name": "Vernell Brown",      "position": "WR", "team": "Florida State Seminoles",   "jersey": 1},
    {"name": "KD Traylor",         "position": "WR", "team": "LSU Tigers",                "jersey": 5},
    {"name": "Jordan Smith",       "position": "WR", "team": "Alabama Crimson Tide",      "jersey": 4},
    {"name": "Micah Tease",        "position": "WR", "team": "Oklahoma Sooners",          "jersey": 2},
    {"name": "Eli Raridon",        "position": "TE", "team": "Notre Dame Fighting Irish", "jersey": 82},
    {"name": "George MacIntyre",   "position": "TE", "team": "Vanderbilt Commodores",     "jersey": 80},
    {"name": "Landen Thomas",      "position": "TE", "team": "Georgia Bulldogs",          "jersey": 88},
]


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def load_position_model_artifacts() -> None:
    global position_model, label_encoders

    if os.path.exists(POSITION_MODEL_PATH) and os.path.exists(ENCODER_PATH):
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(POSITION_MODEL_PATH)
        position_model = loaded_model
        label_encoders = joblib.load(ENCODER_PATH)
        print("Position model artifacts loaded.")
    else:
        print("Position model artifacts not found.")


def initialize_player_database() -> None:
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS players (
            name TEXT PRIMARY KEY,
            position TEXT NOT NULL,
            team TEXT NOT NULL,
            jersey INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'seed',
            espn_id TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    if USE_POSTGRES:
        # Postgres: check information_schema for missing columns and add them.
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'players' AND table_schema = 'public'"
        )
        existing_columns = {row[0] for row in cursor.fetchall()}
        if "source" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'")
        if "updated_at" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN updated_at TEXT")
        if "espn_id" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN espn_id TEXT")
    else:
        # SQLite: backward-compatible migration for older local DBs.
        cursor.execute("PRAGMA table_info(players)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "source" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'")
        cursor.execute("UPDATE players SET source = 'legacy' WHERE source IS NULL OR source = ''")
        if "updated_at" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN updated_at TEXT")
        cursor.execute("UPDATE players SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL OR updated_at = ''")
        if "espn_id" not in existing_columns:
            cursor.execute("ALTER TABLE players ADD COLUMN espn_id TEXT")

    cursor.execute("SELECT COUNT(*) FROM players")
    existing_count = cursor.fetchone()[0]
    if existing_count == 0:
        staged = [
            (p["name"], p["position"], p["team"], p["jersey"],
             "nfl_draft_2025" if p.get("espn_id") else "nfl_seed",
             p.get("espn_id"))
            for p in SEED_PLAYERS
        ]

        if USE_POSTGRES:
            cursor.executemany(
                "INSERT INTO players (name, position, team, jersey, source, espn_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                staged,
            )
        else:
            cursor.executemany(
                "INSERT OR IGNORE INTO players (name, position, team, jersey, source, espn_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                staged,
            )
        print(f"Initialized player database with {len(staged)} records.")

    conn.commit()
    conn.close()


def upsert_player_record(name: str, position: str, team: str, jersey: int,
                         source: str = "runtime", espn_id: Optional[str] = None) -> None:
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO players (name, position, team, jersey, source, espn_id, updated_at)
        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, CURRENT_TIMESTAMP)
        ON CONFLICT(name) DO UPDATE SET
            position = EXCLUDED.position,
            team = EXCLUDED.team,
            jersey = EXCLUDED.jersey,
            source = EXCLUDED.source,
            espn_id = COALESCE(EXCLUDED.espn_id, players.espn_id),
            updated_at = CURRENT_TIMESTAMP
        """,
        (name, position, team, jersey, source, espn_id),
    )
    conn.commit()
    conn.close()


def search_players(query: str = "", limit: int = 30, source: Optional[str] = None):
    safe_limit = max(1, min(limit, 200))
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()

    if query and source:
        pattern = f"%{query.strip()}%"
        cursor.execute(
            f"SELECT name, position, team, jersey, source FROM players "
            f"WHERE name LIKE {ph} AND source = {ph} ORDER BY name ASC LIMIT {ph}",
            (pattern, source, safe_limit),
        )
    elif query:
        pattern = f"%{query.strip()}%"
        cursor.execute(
            f"SELECT name, position, team, jersey, source FROM players "
            f"WHERE name LIKE {ph} ORDER BY name ASC LIMIT {ph}",
            (pattern, safe_limit),
        )
    elif source:
        cursor.execute(
            f"SELECT name, position, team, jersey, source FROM players "
            f"WHERE source = {ph} ORDER BY name ASC LIMIT {ph}",
            (source, safe_limit),
        )
    else:
        cursor.execute(
            f"SELECT name, position, team, jersey, source FROM players "
            f"ORDER BY name ASC LIMIT {ph}",
            (safe_limit,),
        )

    rows = _rows_as_dicts(cursor)
    conn.close()
    return rows


def get_player_by_exact_name(name: str) -> Optional[Dict[str, object]]:
    if not name.strip():
        return None

    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT name, position, team, jersey, source, espn_id FROM players "
        f"WHERE lower(name) = lower({ph}) LIMIT 1",
        (name.strip(),),
    )
    row = cursor.fetchone()
    result = _row_as_dict(cursor, row)
    conn.close()
    return result


def player_database_count() -> int:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM players")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def player_database_count_by_source(source: str) -> int:
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM players WHERE source = {ph}", (source,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def http_get_json(url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def iter_athlete_like_nodes(node):
    if isinstance(node, dict):
        has_name = any(key in node for key in ("displayName", "fullName", "shortName"))
        has_identity = "id" in node or "$ref" in node
        if has_name and has_identity:
            yield node
        for value in node.values():
            yield from iter_athlete_like_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_athlete_like_nodes(item)


def normalize_prospect_position(position_value: str) -> str:
    value = (position_value or "").upper()
    if value in {"QB", "RB", "WR", "TE"}:
        return value
    return value or "UNK"


def is_likely_prospect(athlete: Dict[str, object]) -> bool:
    experience = athlete.get("experience")
    raw_experience = ""
    if isinstance(experience, dict):
        raw_experience = str(
            experience.get("abbreviation")
            or experience.get("displayValue")
            or experience.get("name")
            or ""
        )
    else:
        raw_experience = str(experience or "")

    exp = raw_experience.upper().replace(" ", "").replace("_", "-")
    non_prospect_markers = {"SR", "GR", "RS-SR", "6TH", "SUPER-SR"}
    return exp not in non_prospect_markers


def extract_team_entries(teams_payload: Dict[str, object]):
    entries = []
    sports = teams_payload.get("sports", [])
    for sport in sports if isinstance(sports, list) else []:
        leagues = sport.get("leagues", [])
        for league in leagues if isinstance(leagues, list) else []:
            teams = league.get("teams", [])
            for row in teams if isinstance(teams, list) else []:
                team = row.get("team", {}) if isinstance(row, dict) else {}
                team_id = str(team.get("id") or "").strip()
                team_name = str(team.get("displayName") or team.get("name") or "").strip()
                if team_id and team_name:
                    entries.append((team_id, team_name))
    return entries


def sync_college_prospects(max_teams: int = 250, max_players: int = 4000) -> Dict[str, int]:
    teams_payload = http_get_json(ESPN_CFB_TEAMS_URL, params={"limit": max_teams, "groups": 80})
    team_entries = extract_team_entries(teams_payload)[:max_teams]

    inserted = 0
    scanned = 0
    for team_id, fallback_team_name in team_entries:
        try:
            roster_payload = http_get_json(ESPN_CFB_TEAM_ROSTER_URL.format(team_id=team_id))
        except Exception:
            continue

        team_name = str(
            (roster_payload.get("team", {}) if isinstance(roster_payload.get("team"), dict) else {}).get("displayName")
            or fallback_team_name
        )

        seen_names = set()
        for athlete in iter_athlete_like_nodes(roster_payload):
            name = str(athlete.get("displayName") or athlete.get("fullName") or athlete.get("shortName") or "").strip()
            if not name:
                continue
            normalized_name = normalize_name(name)
            if normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            scanned += 1

            if not is_likely_prospect(athlete):
                continue

            position_obj = athlete.get("position")
            if isinstance(position_obj, dict):
                raw_position = str(position_obj.get("abbreviation") or position_obj.get("name") or "UNK")
            else:
                raw_position = str(position_obj or "UNK")
            position = normalize_prospect_position(raw_position)

            jersey_raw = athlete.get("jersey", 0)
            try:
                jersey = int(jersey_raw or 0)
            except ValueError:
                jersey = 0

            espn_id = str(athlete.get("id") or "").strip() or None
            upsert_player_record(name=name, position=position, team=team_name, jersey=jersey,
                                 source="college_prospect", espn_id=espn_id)
            inserted += 1

            if inserted >= max_players:
                return {"teams": len(team_entries), "scanned": scanned, "inserted": inserted}

        # Avoid hammering upstream endpoint.
        time.sleep(0.02)

    return {"teams": len(team_entries), "scanned": scanned, "inserted": inserted}


def baseline_stats(name: str, position: str, team: str, jersey: int = 0) -> Dict[str, object]:
    p = (position or "").upper()
    defaults = {
        "games_played": 12,
        "passing_touchdowns": 0, "passing_yards": 0,
        "rushing_touchdowns": 0, "rushing_yards": 0,
        "tackles": 0, "sacks": 0.0, "interceptions": 0, "pass_deflections": 0,
    }

    if p == "QB":
        defaults.update({"passing_touchdowns": 24, "passing_yards": 3400})
    elif p == "RB":
        defaults.update({"rushing_touchdowns": 9, "rushing_yards": 980})
    elif p in {"WR", "TE"}:
        defaults.update({"rushing_touchdowns": 2, "rushing_yards": 240})
    elif p in {"LB", "ILB", "OLB", "MLB"}:
        defaults.update({"tackles": 80, "sacks": 4.0, "interceptions": 1, "pass_deflections": 4})
    elif p in {"CB", "S", "DB", "FS", "SS"}:
        defaults.update({"tackles": 55, "sacks": 0.5, "interceptions": 3, "pass_deflections": 9})
    elif p in {"DL", "DE", "DT", "NT", "EDGE"}:
        defaults.update({"tackles": 45, "sacks": 6.5, "pass_deflections": 2})

    return {"name": name, "position": position, "team": team, "jersey": jersey, **defaults}


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def generate_estimated_profile(name: str, position: str, team: str, jersey: int = 0) -> Dict[str, object]:
    seed = stable_int(f"{name}|{position}|{team}|{jersey}")
    p = (position or "UNK").upper()

    def scaled(min_v: int, max_v: int, offset: int = 0) -> int:
        span = max_v - min_v + 1
        return min_v + ((seed + offset) % max(span, 1))

    games = scaled(8, 17, 11)

    if p == "QB":
        passing_touchdowns = scaled(8, 38, 23)
        passing_yards = scaled(1700, 4800, 41)
        rushing_touchdowns = scaled(0, 8, 61)
        rushing_yards = scaled(20, 620, 83)
    elif p == "RB":
        passing_touchdowns = scaled(0, 2, 7)
        passing_yards = scaled(0, 90, 19)
        rushing_touchdowns = scaled(2, 18, 29)
        rushing_yards = scaled(300, 1900, 37)
    elif p in {"WR", "TE"}:
        passing_touchdowns = scaled(0, 3, 5)
        passing_yards = scaled(0, 240, 13)
        rushing_touchdowns = scaled(2, 14, 17)
        rushing_yards = scaled(350, 1650, 31)
    # Defensive stat defaults
    tackles = 0
    sacks = 0.0
    interceptions = 0
    pass_deflections = 0

    if p in {"LB", "ILB", "OLB", "MLB"}:
        passing_touchdowns = 0; passing_yards = 0
        rushing_touchdowns = 0; rushing_yards = 0
        tackles = scaled(30, 130, 3)
        sacks = round(scaled(0, 12, 7) * 0.5, 1)
        interceptions = scaled(0, 5, 11)
        pass_deflections = scaled(0, 12, 17)
    elif p in {"CB", "S", "DB", "FS", "SS"}:
        passing_touchdowns = 0; passing_yards = 0
        rushing_touchdowns = 0; rushing_yards = 0
        tackles = scaled(25, 90, 5)
        sacks = round(scaled(0, 2, 9) * 0.5, 1)
        interceptions = scaled(0, 7, 13)
        pass_deflections = scaled(1, 20, 19)
    elif p in {"DL", "DE", "DT", "NT", "EDGE"}:
        passing_touchdowns = 0; passing_yards = 0
        rushing_touchdowns = 0; rushing_yards = 0
        tackles = scaled(20, 80, 5)
        sacks = round(scaled(0, 16, 11) * 0.5, 1)
        pass_deflections = scaled(0, 6, 17)
    elif p in {"OL", "OT", "OG", "G", "C", "LS"}:
        passing_touchdowns = 0; passing_yards = 0
        rushing_touchdowns = 0; rushing_yards = 0
    else:
        passing_touchdowns = scaled(0, 4, 3)
        passing_yards = scaled(0, 260, 13)
        rushing_touchdowns = scaled(0, 7, 23)
        rushing_yards = scaled(60, 720, 47)

    # Derive features deterministically from the same seed
    conference_tier = classify_college_tier(team)
    combine_speed = combine_speed_for_position(p, seed)
    production_raw = compute_production_score(p, {
        "games_played": games, "passing_touchdowns": passing_touchdowns,
        "passing_yards": passing_yards, "rushing_touchdowns": rushing_touchdowns,
        "rushing_yards": rushing_yards, "tackles": tackles, "sacks": sacks,
        "interceptions": interceptions, "pass_deflections": pass_deflections,
    })
    accolades = detect_accolades(name)
    # Draft round kept for display only (NOT a model feature)
    composite = (production_raw * 0.6 + combine_speed * 0.4) / 100.0
    raw_round = 8 - int(composite * 7)
    draft_round = max(1, min(8, raw_round))

    # Estimated physical profile (position averages — replaced by real data when available)
    pos_key = p if p in _POS_AVG_PHYSICAL else "QB"
    est_h, est_w = _POS_AVG_PHYSICAL.get(pos_key, (73, 220))

    return {
        "name": name,
        "position": position,
        "team": team,
        "jersey": jersey,
        "games_played": games,
        "passing_touchdowns": passing_touchdowns,
        "passing_yards": passing_yards,
        "rushing_touchdowns": rushing_touchdowns,
        "rushing_yards": rushing_yards,
        "tackles": tackles,
        "sacks": sacks,
        "interceptions": interceptions,
        "pass_deflections": pass_deflections,
        "draft_round": draft_round,
        "combine_speed_score": round(combine_speed, 1),
        "conference_tier": conference_tier,
        "production_score": round(production_raw, 1),
        "is_award_winner": accolades["is_award_winner"],
        "is_all_american": accolades["is_all_american"],
        "height_inches":   est_h,
        "weight_lbs":      est_w,
        "display_height":  f"{est_h // 12}'{est_h % 12}\"",
        "display_weight":  f"{est_w} lbs",
        "height_score":    round(height_to_score(p, est_h), 1),
        "weight_score":    round(weight_to_score(p, est_w), 1),
        "vert_score":      50.0,
        "physical_is_real": False,
    }


def _parse_int(val: str) -> int:
    """Parse ESPN stat string like '3,163' → 3163."""
    try:
        return int(str(val).replace(",", "").replace("--", "0").strip() or 0)
    except ValueError:
        return 0


ESPN_CORE_ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes/{espn_id}"


def _espn_resolve_athlete_info(espn_id: str) -> Dict[str, str]:
    """Resolve team name, position, height, weight from the ESPN core athlete endpoint."""
    cache_key = f"espn_athlete:{espn_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    result = {"team": "", "position": "", "height_inches": 0, "weight_lbs": 0,
              "display_height": "", "display_weight": ""}
    try:
        r = requests.get(ESPN_CORE_ATHLETE_URL.format(espn_id=espn_id), timeout=5)
        if not r.ok:
            return result
        ath = r.json()

        # Position
        pos_ref = (ath.get("position") or {}).get("$ref", "")
        if pos_ref:
            rp = requests.get(pos_ref, timeout=5)
            if rp.ok:
                result["position"] = rp.json().get("abbreviation", "")
        if not result["position"]:
            pos_inline = ath.get("position") or {}
            result["position"] = str(pos_inline.get("abbreviation") or pos_inline.get("name") or "")

        # Team
        team_ref = (ath.get("team") or {}).get("$ref", "")
        if team_ref:
            rt = requests.get(team_ref, timeout=5)
            if rt.ok:
                result["team"] = rt.json().get("displayName", "")

        # Physical measurements
        result["height_inches"]  = int(ath.get("height") or 0)
        result["weight_lbs"]     = int(ath.get("weight") or 0)
        result["display_height"] = str(ath.get("displayHeight") or "")
        result["display_weight"] = str(ath.get("displayWeight") or "")

        cache_set(cache_key, result, ttl=STATS_CACHE_TTL)
    except Exception:
        pass
    return result


def fetch_real_espn_stats(espn_id: str, position: str, player_name: str) -> Optional[Dict]:
    """Fetch real season stats from ESPN's athlete overview endpoint.

    Returns a stats dict compatible with generate_estimated_profile output,
    or None if the fetch fails or no stats are found.
    """
    if not espn_id:
        return None

    cache_key = f"espn_stats:{espn_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached if cached else None

    try:
        url = ESPN_CFB_ATHLETE_OVERVIEW_URL.format(espn_id=espn_id)
        resp = requests.get(url, timeout=6)
        if not resp.ok:
            cache_set(cache_key, {}, ttl=STATS_CACHE_TTL)
            return None

        data = resp.json()
        stats_section = data.get("statistics", {})
        names  = stats_section.get("names", [])
        splits = stats_section.get("splits", [])

        if not names or not splits:
            cache_set(cache_key, {}, ttl=STATS_CACHE_TTL)
            return None

        # Use the most recent season split that has real data
        best_split = None
        for split in splits:
            vals = split.get("stats", [])
            if any(v and v not in ("--", "0", "0.0") for v in vals):
                best_split = split
                break

        if not best_split:
            cache_set(cache_key, {}, ttl=STATS_CACHE_TTL)
            return None

        sv  = dict(zip(names, best_split.get("stats", [])))
        pos = (position or "").upper()

        passing_yards = _parse_int(sv.get("passingYards", 0))
        passing_tds   = _parse_int(sv.get("passingTouchdowns", 0))
        rushing_yards = _parse_int(sv.get("rushingYards", 0))
        rushing_tds   = _parse_int(sv.get("rushingTouchdowns", 0))

        # WR/TE: map receiving stats into the rushing slots (model compatibility)
        if pos in {"WR", "TE"}:
            rec_yards = _parse_int(sv.get("receivingYards", 0))
            rec_tds   = _parse_int(sv.get("receivingTouchdowns", 0))
            if rec_yards > rushing_yards:
                rushing_yards = rec_yards
                rushing_tds   = rec_tds

        # Defensive stats
        tackles        = _parse_int(sv.get("totalTackles", 0)) or _parse_int(sv.get("tackles", 0))
        sacks          = float(sv.get("sacks", 0) or 0)
        interceptions  = _parse_int(sv.get("interceptions", 0))
        pass_deflections = _parse_int(sv.get("passesDefended", 0)) or _parse_int(sv.get("passDeflections", 0))

        # Estimate games from available counting stats
        attempts = _parse_int(sv.get("passingAttempts", 0)) or _parse_int(sv.get("rushingAttempts", 0))
        if attempts:
            games = max(1, min(17, round(attempts / 28)))
        elif tackles:
            games = max(1, min(17, round(tackles / 6)))
        else:
            games = 13

        # Resolve real team + position from ESPN athlete info
        ath_info  = _espn_resolve_athlete_info(espn_id)
        real_team = ath_info.get("team", "")

        result = {
            "games_played":       games,
            "passing_touchdowns": passing_tds,
            "passing_yards":      passing_yards,
            "rushing_touchdowns": rushing_tds,
            "rushing_yards":      rushing_yards,
            "tackles":            tackles,
            "sacks":              sacks,
            "interceptions":      interceptions,
            "pass_deflections":   pass_deflections,
            "_team":              real_team,
            "_season":            best_split.get("displayName", ""),
            "_completion_pct":    sv.get("completionPct", ""),
            "_qb_rating":         sv.get("QBRating", ""),
        }
        cache_set(cache_key, result, ttl=STATS_CACHE_TTL)
        return result

    except Exception as exc:
        print(f"ESPN stats fetch failed for {player_name} (id={espn_id}): {exc}")
        cache_set(cache_key, {}, ttl=300)
        return None


def height_to_score(position: str, height_inches: float) -> float:
    """0-100 position-normalized height score. 100=elite prototypical height."""
    if not height_inches or height_inches < 60:
        return 50.0
    p = (position or "").upper()
    # (poor_threshold, elite_threshold) in inches
    bm = {
        "QB": (71, 76), "RB": (68, 72), "WR": (70, 75),
        "TE": (75, 79), "CB": (70, 74), "S":  (71, 75),
        "DB": (70, 74), "LB": (73, 77), "DL": (74, 78),
        "DE": (74, 78), "OL": (76, 80), "OT": (77, 81),
    }
    poor_h, elite_h = bm.get(p, (71, 75))
    return float(max(0.0, min(100.0, (height_inches - poor_h) / (elite_h - poor_h) * 100)))


def weight_to_score(position: str, weight_lbs: float) -> float:
    """0-100 position-normalized weight score (too light OR too heavy penalized)."""
    if not weight_lbs or weight_lbs < 150:
        return 50.0
    p = (position or "").upper()
    # (ideal_low, ideal_high) — staying in range = 100, outside = scaled down
    bm = {
        "QB": (210, 235), "RB": (195, 225), "WR": (185, 215),
        "TE": (245, 270), "CB": (185, 210), "S":  (200, 220),
        "DB": (190, 215), "LB": (230, 255), "DL": (270, 310),
        "DE": (255, 290), "OL": (295, 325), "OT": (300, 330),
    }
    lo, hi = bm.get(p, (200, 240))
    if lo <= weight_lbs <= hi:
        return 100.0
    if weight_lbs < lo:
        return float(max(0.0, 100.0 - (lo - weight_lbs) * 3))
    return float(max(0.0, 100.0 - (weight_lbs - hi) * 2))


def vertical_to_score(position: str, vertical_inches: float) -> float:
    """0-100 position-normalized vertical jump score."""
    if not vertical_inches or vertical_inches < 20:
        return 50.0
    p = (position or "").upper()
    bm = {
        "QB": (29, 38), "RB": (32, 42), "WR": (34, 44),
        "TE": (30, 40), "CB": (34, 44), "S":  (33, 43),
        "LB": (31, 40), "DL": (29, 38), "OL": (26, 34),
    }
    poor_v, elite_v = bm.get(p, (30, 40))
    return float(max(0.0, min(100.0, (vertical_inches - poor_v) / (elite_v - poor_v) * 100)))


# Position-average height/weight used when no ESPN data is available
_POS_AVG_PHYSICAL = {
    "QB": (75, 218), "RB": (71, 212), "WR": (73, 200), "TE": (77, 255),
    "CB": (71, 196), "S":  (73, 208), "DB": (72, 202), "LB": (75, 242),
    "DL": (76, 288), "DE": (75, 265), "OL": (78, 312), "OT": (79, 315),
}


def fetch_combine_measurables(espn_id: str, position: str) -> dict:
    """Fetch full NFL combine profile from ESPN core athlete.

    Returns dict with combine_speed_score, forty, vertical, bench, broad,
    shuttle, 3cone, height_inches, weight_lbs, and derived scores.
    Returns empty dict if no ESPN entry or no 40-time found.
    """
    if not espn_id:
        return {}
    cache_key = f"combine:{espn_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached if cached else {}
    try:
        url  = ESPN_NFL_CORE_ATHLETE_URL.format(espn_id=espn_id)
        resp = requests.get(url, timeout=6)
        if not resp.ok:
            cache_set(cache_key, {}, ttl=STATS_CACHE_TTL)
            return {}
        data  = resp.json()
        draft = data.get("draft") or {}
        forty    = float(draft.get("combined40yd") or 0)
        vertical = float(draft.get("combineVert")  or 0)
        bench    = int(float(draft.get("combineBench")   or 0))
        broad    = float(draft.get("combineBroad")  or 0)
        shuttle  = float(draft.get("combineShuttle") or 0)
        cone3    = float(draft.get("combine3Cone")   or 0)
        height_in = int(data.get("height") or 0)
        weight_lb = int(data.get("weight") or 0)

        if not forty and not height_in:
            cache_set(cache_key, {}, ttl=STATS_CACHE_TTL)
            return {}

        # No 40 time → no speed score. None is skipped by the profile-update
        # loops (`if combine_data.get(key) is not None`), so the profile keeps
        # its existing value instead of a fake 50.0 average.
        speed = forty_to_speed_score(position, forty) if forty else None
        vert_sc  = vertical_to_score(position, vertical)
        height_sc = height_to_score(position, height_in)
        weight_sc = weight_to_score(position, weight_lb)

        # Build display strings
        def fmt_height(h):
            if not h: return ""
            return f"{h // 12}'{h % 12}\""

        result = {
            "combine_speed_score":  round(speed, 1) if speed is not None else None,
            "combine_forty":        forty,
            "combine_vertical":     vertical,
            "combine_bench":        bench,
            "combine_broad":        broad,
            "combine_shuttle":      shuttle,
            "combine_3cone":        cone3,
            "height_inches":        height_in,
            "weight_lbs":           weight_lb,
            "display_height":       str(data.get("displayHeight") or fmt_height(height_in)),
            "display_weight":       str(data.get("displayWeight") or (f"{weight_lb} lbs" if weight_lb else "")),
            "vert_score":           round(vert_sc, 1),
            "height_score":         round(height_sc, 1),
            "weight_score":         round(weight_sc, 1),
            "physical_is_real":     True,
        }
        cache_set(cache_key, result, ttl=STATS_CACHE_TTL * 24)
        return result
    except Exception:
        cache_set(cache_key, {}, ttl=300)
        return {}


def fetch_player_data(player_name: str, fallback_position: str = "Unknown", fallback_team: str = "Unknown",
                      espn_id_hint: str = "") -> Tuple[Optional[Dict[str, object]], str]:
    normalized = normalize_name(player_name)
    espn_id_hint = str(espn_id_hint or "").strip()

    # 1) Resolve from local database first — or from a caller-supplied ESPN
    #    athlete id (the cache builder knows every roster player's id, which
    #    unlocks real season stats for players the DB has never seen).
    db_player = get_player_by_exact_name(player_name)
    # Same-name collision guard: when the caller supplies an athlete id and
    # the DB row is a DIFFERENT athlete, trust the caller's identity — the
    # DB once resolved South Carolina's Dylan Stewart to Delaware's.
    if db_player and espn_id_hint:
        _db_id = str(db_player.get("espn_id") or "").strip()
        _db_team = _normalize_team(str(db_player.get("team") or ""))
        _hint_team = _normalize_team(fallback_team)
        _id_conflict = _db_id and _db_id != espn_id_hint
        # id-less DB row at a different school = same-name different athlete
        _team_conflict = (not _db_id and _hint_team not in ("", "unknown")
                          and _db_team not in ("", "unknown")
                          and _db_team != _hint_team)
        if _id_conflict or _team_conflict:
            db_player = None
    if db_player or espn_id_hint:
        if db_player:
            source   = str(db_player.get("source", "db_lookup") or "db_lookup")
            name     = str(db_player.get("name", player_name)).strip()
            position = str(db_player.get("position", "Unknown") or "Unknown")
            team     = str(db_player.get("team", "Unknown") or "Unknown")
            jersey   = int(db_player.get("jersey", 0) or 0)
            espn_id  = str(db_player.get("espn_id") or "").strip() or espn_id_hint
        else:
            source   = "roster_hint"
            name     = player_name
            position = fallback_position
            team     = fallback_team
            jersey   = 0
            espn_id  = espn_id_hint
        # A stale DB row must not beat what the caller told us: an Unknown
        # position one-hots to position_other and tanks the prediction.
        _unk = {"unknown", "unk", ""}
        if position.lower() in _unk and fallback_position.lower() not in _unk:
            position = fallback_position
        if team.lower() in _unk and fallback_team.lower() not in _unk:
            team = fallback_team

        # Try to get real season stats from ESPN when we have an athlete ID
        real_stats = fetch_real_espn_stats(espn_id, position, name)
        if real_stats:
            ath_info   = _espn_resolve_athlete_info(espn_id)  # cached, near-zero cost on repeat
            real_team  = str(real_stats.get("_team") or "").strip()
            real_pos   = str(ath_info.get("position") or "").strip().upper()
            effective_team = real_team if real_team else team
            effective_pos  = real_pos  if real_pos  else position

            # Persist real team + position to DB if they were Unknown
            if (real_team and team in ("Unknown", "")) or (real_pos and position in ("Unknown", "")):
                upsert_player_record(name, effective_pos or position, effective_team or team,
                                     jersey, source, espn_id)

            profile = generate_estimated_profile(name=name, position=effective_pos or position,
                                                  team=effective_team, jersey=jersey)
            profile.update({
                "games_played":       real_stats["games_played"],
                "passing_touchdowns": real_stats["passing_touchdowns"],
                "passing_yards":      real_stats["passing_yards"],
                "rushing_touchdowns": real_stats["rushing_touchdowns"],
                "rushing_yards":      real_stats["rushing_yards"],
                "tackles":            real_stats.get("tackles", 0),
                "sacks":              real_stats.get("sacks", 0.0),
                "interceptions":      real_stats.get("interceptions", 0),
                "pass_deflections":   real_stats.get("pass_deflections", 0),
            })
            if real_team:
                profile["team"] = real_team
            if effective_pos:
                profile["position"] = effective_pos
            for k in ("_team", "_season", "_completion_pct", "_interceptions", "_qb_rating"):
                if real_stats.get(k) is not None:
                    profile[k] = real_stats[k]
            # Enrich with real combine + physical measurables
            combine_data = fetch_combine_measurables(espn_id, effective_pos or position)
            if combine_data:
                for key in ("combine_speed_score","combine_forty","combine_vertical",
                            "combine_bench","combine_broad","combine_shuttle","combine_3cone",
                            "height_inches","weight_lbs","display_height","display_weight",
                            "height_score","weight_score","vert_score","physical_is_real"):
                    if combine_data.get(key) is not None:
                        profile[key] = combine_data[key]
            # Fill height/weight from athlete info if combine didn't have it
            if not profile.get("height_inches") and ath_info.get("height_inches"):
                profile["height_inches"]  = ath_info["height_inches"]
                profile["weight_lbs"]     = ath_info.get("weight_lbs", 0)
                profile["display_height"] = ath_info.get("display_height", "")
                profile["display_weight"] = ath_info.get("display_weight", "")
                profile["height_score"]   = round(height_to_score(effective_pos or position, ath_info["height_inches"]), 1)
                profile["weight_score"]   = round(weight_to_score(effective_pos or position, ath_info.get("weight_lbs", 0)), 1)
                profile["physical_is_real"] = True
            # Recompute production_score with updated real stats
            profile["production_score"] = round(compute_production_score(
                profile["position"], profile), 1)
            # Speed-score guard (same rule as _neutralize_fabricated): the base
            # profile inherits a HASH-FABRICATED combine_speed_score from
            # generate_estimated_profile. Unless real combine data was merged
            # above (physical_is_real / a real forty), the model must see NaN,
            # not an invented number. Real production/games from ESPN stay.
            if not (profile.get("physical_is_real") or profile.get("combine_forty")):
                profile["combine_speed_score"] = float("nan")
            return profile, "espn_live"

        # No real ESPN stats — try combine measurables anyway
        result = generate_estimated_profile(name=name, position=position, team=team, jersey=jersey)
        if espn_id:
            ath_info = _espn_resolve_athlete_info(espn_id)
            combine_data = fetch_combine_measurables(espn_id, position)
            if combine_data:
                for key in ("combine_speed_score","combine_forty","combine_vertical",
                            "combine_bench","combine_broad","combine_shuttle","combine_3cone",
                            "height_inches","weight_lbs","display_height","display_weight",
                            "height_score","weight_score","vert_score","physical_is_real"):
                    if combine_data.get(key) is not None:
                        result[key] = combine_data[key]
            if not result.get("height_inches") and ath_info.get("height_inches"):
                result["height_inches"]  = ath_info["height_inches"]
                result["weight_lbs"]     = ath_info.get("weight_lbs", 0)
                result["display_height"] = ath_info.get("display_height", "")
                result["display_weight"] = ath_info.get("display_weight", "")
                result["height_score"]   = round(height_to_score(position, ath_info["height_inches"]), 1)
                result["weight_score"]   = round(weight_to_score(position, ath_info.get("weight_lbs", 0)), 1)
                result["physical_is_real"] = True
        return _neutralize_fabricated(result), source

    # 2) Last fallback: generic baseline, using caller-supplied position/team if known.
    result = generate_estimated_profile(name=player_name.strip(), position=fallback_position, team=fallback_team)
    upsert_player_record(
        result["name"],
        result["position"],
        result["team"],
        int(result["jersey"]),
        source="default_baseline",
    )
    return _neutralize_fabricated(result), "default_baseline"


def _neutralize_fabricated(profile: Dict[str, object]) -> Dict[str, object]:
    """No verified season stats -> the model must see MISSING, not invented.

    generate_estimated_profile fabricates a full stat line from a name hash;
    fed to the ensemble it produced inflated 90%+ grades for unknown players
    at tier-1 schools, out-ranking verified stars. NaN routes them through the
    NaN-native model path so they grade on position/tier priors instead.
    Real combine data (physical_is_real / a real forty) is kept.
    """
    profile["production_score"] = float("nan")
    profile["games_played"]     = float("nan")
    if not (profile.get("physical_is_real") or profile.get("combine_forty")):
        profile["combine_speed_score"] = float("nan")
    return profile


# position_flags and the position-group sets now live in dv_features.py.


# ── CFBD enrichment for current players (training_data/enrichment.json) ──────
# Built by scripts/build_enrichment.py; maps "Name|Team" → recruiting pedigree,
# recruit_year (for the years-in-college proxy), SP+ and raw CFBD production
# composites. Hot-reloaded by file mtime like the prospect cache, so an
# enrichment rebuild is picked up without restarting workers.

_ENRICHMENT_INDEX: Dict[str, list] = {}   # norm name -> [(norm team, entry), ...]
_ENRICHMENT_MTIME: float = 0.0
_enrichment_lock = threading.Lock()

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _enrich_norm_name(name: str) -> str:
    """Lowercase, de-accent, strip punctuation + generational suffixes
    (identical to scripts/build_training_data.norm_name)."""
    n = unicodedata.normalize("NFKD", str(name or ""))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    for ch in ".'’,-":
        n = n.replace(ch, " ")
    return " ".join(w for w in n.split() if w not in _NAME_SUFFIXES)


def _enrich_teams_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return (a == b or a.startswith(b + " ") or b.startswith(a + " ")
            or a in b or b in a)


def load_enrichment() -> None:
    global _ENRICHMENT_INDEX, _ENRICHMENT_MTIME
    try:
        mtime = os.path.getmtime(ENRICHMENT_PATH)  # capture BEFORE reading
        with open(ENRICHMENT_PATH) as fh:
            players = json.load(fh).get("players", {}) or {}
        index: Dict[str, list] = {}
        for key, entry in players.items():
            name, _, team = key.partition("|")
            index.setdefault(_enrich_norm_name(name), []).append(
                (_normalize_team(team), entry))
        _ENRICHMENT_INDEX = index
        _ENRICHMENT_MTIME = mtime
        print(f"Enrichment loaded: {len(players)} players ({ENRICHMENT_PATH})")
    except Exception as exc:
        # mtime NOT updated on failure (e.g. file missing/mid-rewrite) → retried
        print(f"Enrichment unavailable ({exc}) — new features degrade to NaN")


def _maybe_reload_enrichment() -> None:
    global _ENRICHMENT_MTIME
    try:
        mtime = os.path.getmtime(ENRICHMENT_PATH)
    except OSError:
        return
    if mtime == _ENRICHMENT_MTIME:
        return
    with _enrichment_lock:
        if mtime == _ENRICHMENT_MTIME:  # another thread already reloaded
            return
        load_enrichment()


def enrichment_for(name: str, team: str) -> Optional[dict]:
    """Entry for this player, or None. Team disambiguates same-named players;
    a unique name match is accepted even when team spellings disagree."""
    _maybe_reload_enrichment()
    cands = _ENRICHMENT_INDEX.get(_enrich_norm_name(name))
    if not cands:
        return None
    t = _normalize_team(team)
    for ct, entry in cands:
        if ct == t:
            return entry
    for ct, entry in cands:
        if _enrich_teams_compatible(ct, t):
            return entry
    if len(cands) == 1:
        return cands[0][1]
    return None


def _upcoming_draft_year(now_struct=None) -> int:
    """The next NFL draft's class year (draft is late April: May+ → next year)."""
    t = now_struct or time.gmtime()
    return t.tm_year + 1 if t.tm_mon >= 5 else t.tm_year


def build_success_features(player_stats: Dict[str, object]) -> pd.DataFrame:
    """Build the SUCCESS_FEATURES vector (base 13 + the v3 feature blocks).
    No draft_round — that is an output, not an input.

    Missing values become NaN (XGBoost/CatBoost handle them natively) instead of
    fake defaults — an unknown 40 time is not a 50.0, an unknown season is not
    13 games. Explicit `is None` checks so legitimate zeros are never swallowed.

    v3 blocks:
    - measurable *_z: only REAL physical data (physical_is_real) is z-scored,
      with the FROZEN training stats in models/feature_stats.json; estimated
      heights/weights and ESPN zero-placeholders stay NaN.
    - recruiting / age proxy / production composites / SP+: merged from
      training_data/enrichment.json by name+team. years_in_college is anchored
      to the next draft the player is eligible for (recruit class + 3), NaN for
      players not yet draft-eligible — mirroring the training match window.
    Players absent from the enrichment file (or any missing piece) degrade
    gracefully: the column is NaN, exactly how unmatched training rows look.
    """
    position = str(player_stats.get("position", "Unknown"))
    team     = str(player_stats.get("team", "") or "")
    name     = str(player_stats.get("name", "") or "")
    flags    = position_flags(position)

    def _value(key: str):
        v = player_stats.get(key)
        return float(v) if v is not None else float("nan")

    def _counting_stat(key: str) -> float:
        # Counting stats: absent means the category doesn't apply (0), not unknown.
        v = player_stats.get(key)
        return float(v) if v is not None else 0.0

    def _pos_float(v) -> float:
        # Enrichment/physical values: None/0 placeholders → NaN.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return f if math.isfinite(f) and f > 0 else float("nan")

    games_played = _value("games_played")

    stats_dict = {
        "games_played":       player_stats.get("games_played"),
        "passing_touchdowns": _counting_stat("passing_touchdowns"),
        "passing_yards":      _counting_stat("passing_yards"),
        "rushing_touchdowns": _counting_stat("rushing_touchdowns"),
        "rushing_yards":      _counting_stat("rushing_yards"),
        "tackles":            _counting_stat("tackles"),
        "sacks":              _counting_stat("sacks"),
        "interceptions":      _counting_stat("interceptions"),
        "pass_deflections":   _counting_stat("pass_deflections"),
    }

    ps = player_stats.get("production_score")
    production_score = (
        float(ps) if ps is not None
        else compute_production_score(position, stats_dict)  # NaN if games unknown
    )

    combine_speed = _value("combine_speed_score")

    ct = player_stats.get("conference_tier")
    if ct is not None:
        conference_tier = float(ct)
    elif team and team.strip().lower() not in {"", "unknown"}:
        conference_tier = float(classify_college_tier(team))
    else:
        conference_tier = float("nan")

    row = {
        "production_score":    production_score,
        "games_played":        games_played,
        "combine_speed_score": combine_speed,
        "conference_tier":     conference_tier,
        **flags,
    }

    # ── Measurable z-scores (real physical data only, frozen training stats) ──
    pos_group = _production_group(position)
    raw_meas = {c: float("nan") for c in MEASURABLE_COLS}
    if player_stats.get("physical_is_real"):
        raw_meas["height_in"] = _pos_float(player_stats.get("height_inches"))
        raw_meas["weight_lb"] = _pos_float(player_stats.get("weight_lbs"))
        raw_meas["vertical"]  = _pos_float(player_stats.get("combine_vertical"))
        raw_meas["bench"]     = _pos_float(player_stats.get("combine_bench"))
        raw_meas["broad_in"]  = _pos_float(player_stats.get("combine_broad"))
        raw_meas["cone"]      = _pos_float(player_stats.get("combine_3cone"))
        raw_meas["shuttle"]   = _pos_float(player_stats.get("combine_shuttle"))
    for c in MEASURABLE_COLS:
        row[c + "_z"] = measurable_z(pos_group, c, raw_meas[c])

    # ── CFBD enrichment (recruiting / age proxy / production / SP+) ──────────
    e = enrichment_for(name, team) or {}
    row["rec_stars"]   = _pos_float(e.get("stars"))
    row["rec_rating"]  = _pos_float(e.get("rating"))
    row["rec_ranking"] = _pos_float(e.get("national_rank"))

    years_in_college = float("nan")
    early_declare = float("nan")
    ry = e.get("recruit_year")
    if ry:
        dy = _upcoming_draft_year()
        if dy - int(ry) >= 3:  # draft-eligible: same 3+ year floor as training
            years_in_college = float(min(dy - int(ry), 7))
            early_declare = 1.0 if years_in_college <= 3 else 0.0
    row["years_in_college"] = years_in_college
    row["early_declare"]    = early_declare

    comp_group = _composite_group(position)
    row["prod_fs_z"]  = composite_z(comp_group, "fs", e.get("prod_fs_raw"))
    row["prod_car_z"] = composite_z(comp_group, "car", e.get("prod_car_raw"))
    row["car_seasons"] = _pos_float(e.get("car_seasons"))
    sp = e.get("sp_plus")  # SP+ is a real-valued rating — negative is meaningful
    try:
        row["sp_rating"] = float(sp) if sp is not None and math.isfinite(float(sp)) else float("nan")
    except (TypeError, ValueError):
        row["sp_rating"] = float("nan")

    return pd.DataFrame([row], columns=SUCCESS_FEATURES)


def proxy_success_score(position: str, stats: Dict[str, float]) -> float:
    p = (position or "Unknown").upper()
    games    = stats.get("games_played", 0)
    pass_td  = stats.get("passing_touchdowns", 0)
    pass_yds = stats.get("passing_yards", 0)
    rush_td  = stats.get("rushing_touchdowns", 0)
    rush_yds = stats.get("rushing_yards", 0)
    tackles  = stats.get("tackles", 0)
    sacks    = stats.get("sacks", 0)
    ints     = stats.get("interceptions", 0)
    pds      = stats.get("pass_deflections", 0)

    if p == "QB":
        return min(pass_td / 35.0, 1.0) * 0.45 + min(pass_yds / 4200.0, 1.0) * 0.45 + min(games / 17.0, 1.0) * 0.10
    if p == "RB":
        return min(rush_td / 14.0, 1.0) * 0.45 + min(rush_yds / 1300.0, 1.0) * 0.45 + min(games / 17.0, 1.0) * 0.10
    if p in {"WR", "TE"}:
        return (min((pass_td + rush_td) / 14.0, 1.0) * 0.35
                + min((pass_yds + rush_yds) / 2000.0, 1.0) * 0.50
                + min(games / 17.0, 1.0) * 0.15)
    if p in {"LB", "ILB", "OLB", "MLB"}:
        return (min(tackles / 100.0, 1.0) * 0.55 + min(sacks / 8.0, 1.0) * 0.30
                + min(ints / 3.0, 1.0) * 0.15)
    if p in {"CB", "S", "DB", "FS", "SS"}:
        return (min(ints / 4.0, 1.0) * 0.40 + min(pds / 12.0, 1.0) * 0.35
                + min(tackles / 55.0, 1.0) * 0.25)
    if p in {"DL", "DE", "DT", "NT", "EDGE"}:
        return min(sacks / 10.0, 1.0) * 0.65 + min(tackles / 45.0, 1.0) * 0.35
    # OL and other: durability proxy
    return min(games / 17.0, 1.0) * 0.60 + 0.20


# ── Training removed from the serving app ────────────────────────────────────
# All model training (real data only, temporal splits, ensemble calibration)
# lives in scripts/train_models.py. The synthetic-row generator, heuristic
# label sampler and in-app trainers were deleted; SEED_TRAINING_PLAYERS moved
# to dv_features.py and the rule heuristics to dv_heuristics.py.


# ── Model loading (LOAD-ONLY — training lives in scripts/train_models.py) ─────
# Artifacts per target: raw XGBoost member (.json) + CatBoost member (.cbm) +
# ensemble calibrator (.pkl, a dict {"kind", "model", "feature_names", ...}
# fitted on the MEAN of the two members' calibration-fold probabilities).
# Serving must therefore apply: member probas → mean → calibrator.

DV_ALLOW_MISSING_MODELS = os.environ.get("DV_ALLOW_MISSING_MODELS", "") == "1"


def _load_serve_mode() -> str:
    """Honest decision gate written by scripts/train_models.py.

    models/metadata.json carries "serve": "ensemble" | "heuristic" — whichever
    beat the other on the 2019-2020 holdout (AUC + Brier). When it says
    "heuristic", /predict prefers determine_success_fallback over the ML
    ensemble. Missing/unreadable metadata defaults to "ensemble".
    """
    try:
        with open("models/metadata.json") as fh:
            mode = str(json.load(fh).get("serve", "ensemble")).lower()
        return mode if mode in {"ensemble", "heuristic"} else "ensemble"
    except Exception:
        return "ensemble"


SERVE_MODE = _load_serve_mode()


def _model_load_failure(kind: str, reason: str) -> None:
    msg = (f"{kind} model artifacts unusable ({reason}) — run scripts/train_models.py")
    if DV_ALLOW_MISSING_MODELS:
        print(f"WARNING: {msg}. DV_ALLOW_MISSING_MODELS=1 — continuing with the "
              "rule-based fallback only.")
        return
    raise RuntimeError(msg)


def _load_calibrator(path: str, kind: str) -> Optional[dict]:
    """Load an ensemble-calibrator dict written by scripts/train_models.py."""
    cal = joblib.load(path)
    if not (isinstance(cal, dict) and "model" in cal and "kind" in cal):
        raise ValueError(
            f"{path} is not an ensemble-calibrator dict (found {type(cal).__name__}); "
            "it is probably a stale pre-retrain artifact")
    feats = cal.get("feature_names")
    if feats is not None and list(feats) != list(SUCCESS_FEATURES):
        raise ValueError(f"{path} was trained on different features: {feats}")
    return cal


def _load_xgb_member(path: str) -> "xgb.XGBClassifier":
    m = xgb.XGBClassifier()
    m.load_model(path)
    n = int(m.get_booster().num_features())
    if n != len(SUCCESS_FEATURES):
        raise ValueError(f"{path} expects {n} features, app builds {len(SUCCESS_FEATURES)}")
    return m


def load_draft_grade_models() -> None:
    """Load the draft-grade ensemble (XGB + CatBoost members + calibrator). LOAD-ONLY."""
    global draft_grade_model, catboost_draft_grade_model, draft_grade_calibrator

    if not CATBOOST_AVAILABLE:
        _model_load_failure("draft grade", "CatBoost library not installed")
        return
    missing = [p for p in (DRAFT_GRADE_MODEL_PATH, CATBOOST_DRAFT_GRADE_PATH,
                           DRAFT_GRADE_CALIBRATED_PATH) if not os.path.exists(p)]
    if missing:
        _model_load_failure("draft grade", f"missing artifacts: {missing}")
        return
    try:
        xgb_m = _load_xgb_member(DRAFT_GRADE_MODEL_PATH)
        cb = CatBoostClassifier()
        cb.load_model(CATBOOST_DRAFT_GRADE_PATH)
        cal = _load_calibrator(DRAFT_GRADE_CALIBRATED_PATH, "draft grade")
    except Exception as exc:
        _model_load_failure("draft grade", str(exc))
        return
    draft_grade_model = xgb_m
    catboost_draft_grade_model = cb
    draft_grade_calibrator = cal
    print("Draft grade ensemble loaded (XGBoost + CatBoost + calibrator).")


def _apply_binary_calibrator(cal: dict, p: float) -> float:
    """Map a raw ensemble-mean success probability through the fitted calibrator."""
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    if cal["kind"] == "platt_logit":
        z = math.log(p / (1.0 - p))
        return float(cal["model"].predict_proba(np.array([[z]]))[0][1])
    if cal["kind"] == "isotonic":
        return float(cal["model"].predict(np.array([p]))[0])
    raise ValueError(f"Unknown binary calibrator kind: {cal['kind']}")


def _apply_multiclass_calibrator(cal: dict, proba: np.ndarray) -> np.ndarray:
    """Map a raw ensemble-mean class-probability vector through the calibrator."""
    if cal["kind"] == "multinomial_logprob":
        logp = np.log(np.clip(proba, 1e-6, 1.0)).reshape(1, -1)
        return cal["model"].predict_proba(logp)[0]
    raise ValueError(f"Unknown multiclass calibrator kind: {cal['kind']}")


def predict_draft_grade(player_stats: Dict[str, object]) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """Ensemble draft grade: member probas → mean → calibrator (matches training)."""
    model_input  = build_success_features(player_stats)
    X_arr        = model_input.values
    proba_arrays = []

    if draft_grade_model is not None:
        try:
            proba_arrays.append(np.array(draft_grade_model.predict_proba(model_input)[0], dtype=float))
        except Exception as exc:
            print(f"XGBoost draft grade inference failed: {exc}")

    if catboost_draft_grade_model is not None:
        try:
            proba_arrays.append(np.array(catboost_draft_grade_model.predict_proba(X_arr)[0], dtype=float))
        except Exception as exc:
            print(f"CatBoost draft grade inference failed: {exc}")

    if not proba_arrays:
        return None, None, None

    avg_proba = np.mean(proba_arrays, axis=0)
    # Label from the RAW ensemble mean: the multinomial calibrator sharpens
    # probabilities but collapses the argmax away from class 1 (Day 2) —
    # holdout macro-F1 0.40 raw vs 0.34 calibrated. Calibrated proba is still
    # reported as the confidence number.
    grade_class = int(np.argmax(avg_proba))
    conf_proba  = avg_proba
    if draft_grade_calibrator is not None:
        try:
            conf_proba = _apply_multiclass_calibrator(draft_grade_calibrator, avg_proba)
        except Exception as exc:
            print(f"Draft grade calibration failed (serving raw ensemble mean): {exc}")
    label = DRAFT_GRADE_LABELS[grade_class]
    return label, grade_class, round(float(conf_proba[grade_class]) * 100.0, 1)


def load_success_models() -> None:
    """Load the success ensemble (XGB + CatBoost members + calibrator). LOAD-ONLY."""
    global success_model, catboost_success_model, success_calibrator

    if not CATBOOST_AVAILABLE:
        _model_load_failure("success", "CatBoost library not installed")
        return
    missing = [p for p in (SUCCESS_MODEL_PATH, CATBOOST_SUCCESS_PATH,
                           SUCCESS_CALIBRATED_PATH) if not os.path.exists(p)]
    if missing:
        _model_load_failure("success", f"missing artifacts: {missing}")
        return
    try:
        xgb_m = _load_xgb_member(SUCCESS_MODEL_PATH)
        cb = CatBoostClassifier()
        cb.load_model(CATBOOST_SUCCESS_PATH)
        cal = _load_calibrator(SUCCESS_CALIBRATED_PATH, "success")
    except Exception as exc:
        _model_load_failure("success", str(exc))
        return
    success_model = xgb_m
    catboost_success_model = cb
    success_calibrator = cal
    print("Success ensemble loaded (XGBoost + CatBoost + calibrator).")


def predict_position_with_model(player_stats: Dict[str, object]) -> Optional[str]:
    if position_model is None or label_encoders is None:
        return None

    try:
        encoder_position = label_encoders.get("position")
        encoder_team = label_encoders.get("team")
        if not encoder_position or not encoder_team:
            return None

        def encode(encoder, value: str) -> int:
            classes = list(encoder.classes_)
            if value in classes:
                return int(encoder.transform([value])[0])
            if "Unknown" in classes:
                return int(encoder.transform(["Unknown"])[0])
            return int(encoder.transform([classes[0]])[0])

        model_input = pd.DataFrame(
            [
                {
                    "jersey": int(player_stats.get("jersey", 0) or 0),
                    "position": encode(encoder_position, str(player_stats.get("position", "Unknown"))),
                    "team": encode(encoder_team, str(player_stats.get("team", "Unknown"))),
                }
            ],
            columns=["jersey", "position", "team"],
        )

        pred_label = int(position_model.predict(model_input)[0])
        return str(encoder_position.inverse_transform([pred_label])[0])
    except Exception as exc:
        print(f"Position inference failed: {exc}")
        return None


FEATURE_DISPLAY_NAMES = {
    "production_score":     "Production Score",
    "combine_speed_score":  "Combine Athleticism",
    "conference_tier":      "College Competition Level",
    "games_played":         "Games Played",
    "position_qb":          "Position: QB",
    "position_rb":          "Position: RB",
    "position_wr":          "Position: WR",
    "position_te":          "Position: TE",
    "position_db":          "Position: DB",
    "position_lb":          "Position: LB",
    "position_dl":          "Position: DL",
    "position_ol":          "Position: OL",
    "position_other":       "Position: Other",
    # v3 feature blocks
    "height_in_z":          "Height (vs position)",
    "weight_lb_z":          "Weight (vs position)",
    "vertical_z":           "Vertical Jump (vs position)",
    "bench_z":              "Bench Press (vs position)",
    "broad_in_z":           "Broad Jump (vs position)",
    "cone_z":               "3-Cone Drill (vs position)",
    "shuttle_z":            "Shuttle Drill (vs position)",
    "rec_stars":            "Recruiting Stars",
    "rec_rating":           "Recruiting Rating",
    "rec_ranking":          "Recruiting National Rank",
    "years_in_college":     "Years in College",
    "early_declare":        "Early Declare",
    "prod_fs_z":            "Final-Season Production (all positions)",
    "prod_car_z":           "Career Production (all positions)",
    "car_seasons":          "College Seasons Played",
    "sp_rating":            "Team SP+ Rating",
}


def top_feature_importances(n: int = 4) -> list:
    """Return the top-n features by XGBoost gain importance."""
    if success_model is None:
        return []
    try:
        model = success_model
        if hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV has no get_booster(); unwrap the fitted XGBoost
            model = model.calibrated_classifiers_[0].estimator
        importances = model.get_booster().get_score(importance_type="gain")
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_feats) or 1.0
        return [
            {
                "feature": FEATURE_DISPLAY_NAMES.get(f, f),
                "importance": round(v / total * 100, 1),
            }
            for f, v in sorted_feats[:n]
        ]
    except Exception:
        return []


def per_player_top_factors(player_stats: Dict[str, object], n: int = 5) -> list:
    """Per-prediction feature contributions (SHAP values) for THIS player.

    Uses the raw XGBoost success member's built-in TreeSHAP
    (booster.predict(pred_contribs=True) — no extra dependencies). Keeps the
    frontend contract of top_feature_importances (list of {feature, importance}
    where importance is a 0-100 bar width) and adds fields additively:
      contribution — signed SHAP value in log-odds space (+ pushes toward
                     Success, − pushes toward No Success)
      direction    — "positive" | "negative"
    The bias term (last column) is skipped; features are ranked by |SHAP|.
    Falls back to the global gain importances if SHAP fails.
    """
    if success_model is None:
        return top_feature_importances(n)
    try:
        model_input = build_success_features(player_stats)
        booster = success_model.get_booster()
        dm = xgb.DMatrix(model_input, feature_names=list(model_input.columns))
        contribs = booster.predict(dm, pred_contribs=True)[0]
        pairs = [(f, float(c)) for f, c in zip(model_input.columns, contribs[:-1])]  # drop bias
        total_abs = sum(abs(c) for _, c in pairs) or 1.0
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return [
            {
                "feature":      FEATURE_DISPLAY_NAMES.get(f, f),
                "importance":   round(abs(c) / total_abs * 100, 1),
                "contribution": round(c, 4),
                "direction":    "positive" if c >= 0 else "negative",
            }
            for f, c in pairs[:n]
        ]
    except Exception as exc:
        print(f"Per-player SHAP factors failed ({exc}) — falling back to global importances")
        return top_feature_importances(n)


def predict_success_with_model(player_stats: Dict[str, object]) -> Tuple[Optional[str], Optional[float], Optional[float], bool]:
    """Ensemble prediction: member probas → mean → calibrator (matches training)."""
    model_input = build_success_features(player_stats)
    X_arr = model_input.values  # numpy array for CatBoost

    probas = []

    # 1. XGBoost member (raw, uncalibrated — calibration happens on the mean)
    if success_model is not None:
        try:
            probas.append(float(success_model.predict_proba(model_input)[0][1]))
        except Exception as exc:
            print(f"XGBoost success inference failed: {exc}")

    # 2. CatBoost member
    if catboost_success_model is not None:
        try:
            probas.append(float(catboost_success_model.predict_proba(X_arr)[0][1]))
        except Exception as exc:
            print(f"CatBoost success inference failed: {exc}")

    if not probas:
        return None, None, None, False

    # 3. Mean of member probabilities, then the ensemble calibrator — the exact
    #    pipeline scripts/train_models.py calibrated and evaluated.
    probability = sum(probas) / len(probas)
    if success_calibrator is not None:
        try:
            probability = _apply_binary_calibrator(success_calibrator, probability)
        except Exception as exc:
            print(f"Success calibration failed (serving raw ensemble mean): {exc}")
    label = "Success" if probability >= 0.5 else "No Success"
    confidence = round((probability if label == "Success" else 1.0 - probability) * 100.0, 1)
    return label, confidence, round(probability * 100.0, 1), True


def determine_success_fallback(player_stats: Dict[str, object], projected_position: str) -> Tuple[str, float, str]:
    stats = {
        "games_played": float(player_stats.get("games_played", 0) or 0),
        "passing_touchdowns": float(player_stats.get("passing_touchdowns", 0) or 0),
        "passing_yards": float(player_stats.get("passing_yards", 0) or 0),
        "rushing_touchdowns": float(player_stats.get("rushing_touchdowns", 0) or 0),
        "rushing_yards": float(player_stats.get("rushing_yards", 0) or 0),
    }
    score = proxy_success_score(projected_position, stats)
    label = "Success" if score >= 0.60 else "No Success"
    confidence = round((score if label == "Success" else 1.0 - score) * 100.0, 1)
    return label, confidence, "Fallback scoring was used because ML inference was unavailable."


# ── In-memory cache ──────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict] = {}
CACHE_TTL = 120  # seconds


def cache_set(key: str, data, ttl: int = CACHE_TTL) -> None:
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time(), "ttl": ttl}


def cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < entry.get("ttl", CACHE_TTL):
            return entry["data"]
    return None


def cache_invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


# ── Prospect leaderboard cache ────────────────────────────────────────────────
_PROSPECT_CACHE: list = []
_PROSPECT_CACHE_META: dict = {}
_PROSPECT_CACHE_MTIME: float = 0.0   # mtime of the JSON file at last successful load
_prospect_reload_lock = threading.Lock()

# ── Board movers / trend cache (written by build_prospect_cache.py) ───────────
_BOARD_MOVERS: dict = {}          # raw board_movers.json contents
_BOARD_TRENDS: dict = {}          # (name_lower, team_lower) -> {"delta_prob","delta_rank"}
_BOARD_MOVERS_MTIME: float = 0.0  # mtime of the JSON file at last successful load
_board_movers_reload_lock = threading.Lock()

# ── Mock draft storage ─────────────────────────────────────────────────────────
_MOCK_DRAFT_DATA: dict = {"picks": [], "title": "", "generated_at": None, "total": 0}

# ── High school prospect cache ─────────────────────────────────────────────────
_HS_PROSPECT_CACHE: list = []
_HS_PROSPECT_CACHE_META: dict = {}
_HS_PROSPECT_CACHE_MTIME: float = 0.0
_hs_prospect_reload_lock = threading.Lock()

_POS_GROUPS = {
    "DB": {"CB", "S", "DB", "FS", "SS"},
    "LB": {"LB", "ILB", "OLB", "MLB"},
    "DL": {"DL", "DE", "DT", "EDGE", "NT"},
    "OL": {"OL", "OT", "OG", "C", "LS"},
}
_GRADE_ORDER = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C+": 6, "C": 7, "C-": 8, "D": 9}


# ── HTTP caching (Cache-Control + weak ETag) ──────────────────────────────────
# The board only changes when a cache file is rebuilt (weekly), so clients can
# revalidate cheaply: identical file mtime + query string → identical ETag →
# 304, instead of redownloading identical megabytes. Applied to the big list
# endpoints only — /search is tiny and /api/analytics/* must never be cached.
_HTTP_CACHE_CONTROL = "public, max-age=300, stale-while-revalidate=600"


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _weak_etag(mtimes) -> str:
    raw = "|".join(f"{m:.6f}" for m in mtimes) \
          + "|" + request.query_string.decode("utf-8", "replace")
    return 'W/"' + hashlib.md5(raw.encode("utf-8")).hexdigest() + '"'


def http_cached(mtime_getter):
    """Decorator: weak ETag derived from the underlying cache file mtime(s) +
    query string; honors If-None-Match with 304. `mtime_getter` runs
    per-request (triggering the endpoint's hot-reload first, so the ETag always
    reflects the file that would be served)."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not has_request_context():  # e.g. /init warm-up at boot
                return fn(*args, **kwargs)
            etag = _weak_etag(mtime_getter())
            tokens = {t.strip() for t in
                      request.headers.get("If-None-Match", "").split(",") if t.strip()}
            if "*" in tokens or etag in tokens or etag[2:] in tokens:
                resp = app.response_class(status=304)
            else:
                resp = app.make_response(fn(*args, **kwargs))
            resp.headers["ETag"] = etag
            resp.headers["Cache-Control"] = _HTTP_CACHE_CONTROL
            return resp
        return wrapper
    return decorator


def _prospects_mtimes():
    _maybe_reload_prospect_cache()
    _maybe_reload_board_movers()  # per-row "trend" comes from the movers file
    return (_PROSPECT_CACHE_MTIME, _BOARD_MOVERS_MTIME)


def _hs_prospects_mtimes():
    _maybe_reload_hs_prospect_cache()
    return (_HS_PROSPECT_CACHE_MTIME,)


def _movers_mtimes():
    _maybe_reload_board_movers()
    return (_BOARD_MOVERS_MTIME,)


def _init_mtimes():
    _maybe_reload_prospect_cache()
    return (_PROSPECT_CACHE_MTIME,)


def load_prospect_cache() -> None:
    global _PROSPECT_CACHE, _PROSPECT_CACHE_META, _PROSPECT_CACHE_MTIME
    if not os.path.exists(PROSPECT_CACHE_PATH):
        print("Prospect cache not found (run build_prospect_cache.py to create it).")
        return
    try:
        mtime = os.path.getmtime(PROSPECT_CACHE_PATH)  # capture BEFORE reading
        with open(PROSPECT_CACHE_PATH) as f:
            data = json.load(f)
        # Atomic reference swaps — readers see either the old list or the new one
        _PROSPECT_CACHE = data.get("prospects", [])
        _PROSPECT_CACHE_META = {
            "generated_at": data.get("generated_at"),
            "total":        data.get("total", len(_PROSPECT_CACHE)),
        }
        _PROSPECT_CACHE_MTIME = mtime
        print(f"Loaded {len(_PROSPECT_CACHE)} prospects from cache.")
    except Exception as exc:
        # mtime NOT updated on failure (e.g. file mid-rewrite) → retried next request
        print(f"Failed to load prospect cache: {exc}")


def _maybe_reload_prospect_cache() -> None:
    """Hot-reload the prospect cache if the JSON file changed on disk.

    Each gunicorn worker checks the file's mtime itself, so a cache rebuild
    (or a new deploy) is picked up without restarting anything. getmtime per
    request is a cheap stat(2).
    """
    try:
        mtime = os.path.getmtime(PROSPECT_CACHE_PATH)
    except OSError:
        return
    if mtime == _PROSPECT_CACHE_MTIME:
        return
    with _prospect_reload_lock:
        if mtime == _PROSPECT_CACHE_MTIME:  # another thread already reloaded
            return
        load_prospect_cache()


@app.get("/api/prospects")
@http_cached(_prospects_mtimes)
def api_prospects():
    _maybe_reload_prospect_cache()
    _maybe_reload_board_movers()  # keeps per-row "trend" in sync with movers file
    position     = (request.args.get("position") or "").strip().upper()
    grade_filter = (request.args.get("grade") or "").strip().upper()
    query        = (request.args.get("q") or "").strip().lower()
    team_filter  = (request.args.get("team") or "").strip().lower()
    class_filter = (request.args.get("draft_class") or "").strip()
    sort_by      = (request.args.get("sort") or "grade").strip()
    try:
        limit  = min(int(request.args.get("limit") or 500), 2000)
        offset = int(request.args.get("offset") or 0)
    except (TypeError, ValueError):
        limit, offset = 500, 0

    results = _PROSPECT_CACHE

    # Position filter — support group aliases (DB, LB, DL, OL)
    if position and position not in ("", "ALL"):
        group_set = _POS_GROUPS.get(position)
        if group_set:
            results = [p for p in results if (p.get("position") or "").upper() in group_set]
        else:
            results = [p for p in results if (p.get("position") or "").upper() == position]

    # Grade filter — "A" matches A+, A, A-
    if grade_filter and grade_filter not in ("", "ALL"):
        results = [p for p in results if (p.get("grade") or "").upper().startswith(grade_filter)]

    # Text search (name or team)
    if query:
        results = [p for p in results
                   if query in (p.get("name") or "").lower()
                   or query in (p.get("team") or "").lower()]

    # Team filter
    if team_filter and team_filter not in ("", "all"):
        results = [p for p in results if team_filter in (p.get("team") or "").lower()]

    # Draft-class filter (e.g. ?draft_class=2027) — combinable with the rest
    if class_filter and class_filter.upper() != "ALL":
        try:
            wanted_class = int(class_filter)
            results = [p for p in results
                       if _safe_int(p.get("draft_class")) == wanted_class]
        except ValueError:
            pass  # non-numeric → ignore the filter, same as other bad params

    # Sort
    if sort_by == "name":
        results = sorted(results, key=lambda p: (p.get("name") or "").lower())
    elif sort_by == "success":
        results = sorted(results, key=lambda p: -(p.get("success_probability") or 0))
    elif sort_by == "team":
        results = sorted(results, key=lambda p: (p.get("team") or "").lower())
    else:  # grade (default, already sorted in cache)
        results = sorted(results, key=lambda p: (
            _GRADE_ORDER.get(p.get("grade"), 9),
            -(p.get("success_probability") or 0),
        ))

    total     = len(results)
    paginated = results[offset: offset + limit]

    # Merge per-player trend (delta vs prior board snapshot) into the rows we
    # actually return. Rows are shallow-copied so the cached dicts stay pristine
    # when board_movers.json changes or disappears.
    if _BOARD_TRENDS:
        merged = []
        for p in paginated:
            trend = _BOARD_TRENDS.get((
                (p.get("name") or "").strip().lower(),
                (p.get("team") or "").strip().lower(),
            ))
            merged.append({**p, "trend": trend} if trend else p)
        paginated = merged

    return jsonify({
        "total":     total,
        "offset":    offset,
        "limit":     limit,
        "meta":      _PROSPECT_CACHE_META,
        "prospects": paginated,
    })


# ── Board movers (weekly risers/fallers vs the prior board snapshot) ──────────

def load_board_movers() -> None:
    global _BOARD_MOVERS, _BOARD_TRENDS, _BOARD_MOVERS_MTIME
    if not os.path.exists(BOARD_MOVERS_PATH):
        return
    try:
        mtime = os.path.getmtime(BOARD_MOVERS_PATH)  # capture BEFORE reading
        with open(BOARD_MOVERS_PATH) as f:
            data = json.load(f)
        # "all_deltas" is the builder's superset of the risers/fallers lists:
        # every player matched to the prior snapshot, keyed here by name+team
        # so /api/prospects can merge a "trend" object into each row.
        trends = {}
        for row in data.get("all_deltas") or []:
            key = ((row.get("name") or "").strip().lower(),
                   (row.get("team") or "").strip().lower())
            trends[key] = {"delta_prob": row.get("delta_prob", 0),
                           "delta_rank": row.get("delta_rank", 0)}
        # Atomic reference swaps — readers see either the old dicts or the new
        _BOARD_MOVERS = data
        _BOARD_TRENDS = trends
        _BOARD_MOVERS_MTIME = mtime
        print(f"Loaded board movers ({data.get('count', 0)} movers, "
              f"{len(trends)} player trends).")
    except Exception as exc:
        # mtime NOT updated on failure (e.g. file mid-rewrite) → retried next request
        print(f"Failed to load board movers: {exc}")


def _maybe_reload_board_movers() -> None:
    """Hot-reload movers/trends if board_movers.json changed on disk (same
    mtime pattern as the prospect cache)."""
    try:
        mtime = os.path.getmtime(BOARD_MOVERS_PATH)
    except OSError:
        return
    if mtime == _BOARD_MOVERS_MTIME:
        return
    with _board_movers_reload_lock:
        if mtime == _BOARD_MOVERS_MTIME:  # another thread already reloaded
            return
        load_board_movers()


@app.get("/api/movers")
@http_cached(_movers_mtimes)
def api_movers():
    _maybe_reload_board_movers()
    data = _BOARD_MOVERS
    # Missing/empty file → valid empty shape (200), per the API contract
    return jsonify({
        "generated_at": data.get("generated_at")
                        or __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "since":        data.get("since"),
        "count":        data.get("count", 0),
        "risers":       data.get("risers", []),
        "fallers":      data.get("fallers", []),
    })


# ── Mock draft ─────────────────────────────────────────────────────────────────

def load_mock_draft() -> None:
    global _MOCK_DRAFT_DATA
    if os.path.exists(MOCK_DRAFT_PATH):
        try:
            with open(MOCK_DRAFT_PATH) as f:
                _MOCK_DRAFT_DATA = json.load(f)
            print(f"Loaded mock draft: {_MOCK_DRAFT_DATA.get('total', 0)} picks.")
        except Exception as exc:
            print(f"Failed to load mock draft: {exc}")


_NFL_TEAM_COLORS = {
    "49ers": "#AA0000", "bears": "#0B162A", "bengals": "#FB4F14", "bills": "#00338D",
    "broncos": "#FB4F14", "browns": "#311D00", "buccaneers": "#D50A0A", "cardinals": "#97233F",
    "chargers": "#0080C6", "chiefs": "#E31837", "colts": "#003A70", "commanders": "#5A1414",
    "cowboys": "#003594", "dolphins": "#008E97", "eagles": "#004C54", "falcons": "#A71930",
    "giants": "#0B2265", "jaguars": "#006778", "jets": "#125740", "lions": "#0076B6",
    "packers": "#203731", "panthers": "#0085CA", "patriots": "#002244", "raiders": "#000000",
    "rams": "#003594", "ravens": "#241773", "saints": "#D3BC8D", "seahawks": "#002244",
    "steelers": "#101820", "texans": "#03202F", "titans": "#0C2340", "vikings": "#4F2683",
}

def _team_color(nfl_team: str) -> str:
    t = (nfl_team or "").lower()
    for key, color in _NFL_TEAM_COLORS.items():
        if key in t:
            return color
    return "#334155"


@app.post("/api/mock-draft/upload")
def upload_mock_draft():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    image_b64  = str(payload.get("image_b64") or "").strip()
    media_type = str(payload.get("media_type") or "image/png").strip()
    title      = str(payload.get("title") or "JKrek's Mock Draft").strip()

    if not image_b64:
        return jsonify({"error": "image_b64 is required"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured on server"}), 503

    # ── Call Claude vision to extract picks ────────────────────────────────────
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a screenshot of an NFL mock draft. "
                            "Extract every pick that is visible.\n\n"
                            "Return ONLY a valid JSON array — no markdown, no explanation. "
                            "Each element must have these exact fields:\n"
                            "  pick       (integer: overall pick number)\n"
                            "  round      (integer: round number)\n"
                            "  nfl_team   (string: full NFL team name)\n"
                            "  player     (string: player full name)\n"
                            "  position   (string: position abbreviation, e.g. QB, WR, OT)\n"
                            "  school     (string: college/university, empty string if unknown)\n"
                            "  pff_grade  (string: PFF grade if shown, empty string if not)\n\n"
                            "If a field is not visible, use an empty string or 0 for integers. "
                            "Return the array sorted by pick number ascending."
                        ),
                    },
                ],
            }],
        )

        raw = msg.content[0].text.strip()
        # Strip markdown code fences if Claude wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        picks_raw = json.loads(raw)

    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Claude returned unparseable JSON: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Vision extraction failed: {exc}"}), 500

    picks = []
    for p in (picks_raw if isinstance(picks_raw, list) else []):
        player = str(p.get("player") or "").strip()
        if not player:
            continue
        nfl_team = str(p.get("nfl_team") or "").strip()
        picks.append({
            "pick":      int(p.get("pick") or len(picks) + 1),
            "round":     int(p.get("round") or 1),
            "nfl_team":  nfl_team,
            "player":    player,
            "position":  str(p.get("position") or "").upper().strip(),
            "school":    str(p.get("school") or "").strip(),
            "pff_grade": str(p.get("pff_grade") or "").strip(),
            "color":     _team_color(nfl_team),
        })

    if not picks:
        return jsonify({"error": "No picks found in image"}), 400

    global _MOCK_DRAFT_DATA
    _MOCK_DRAFT_DATA = {
        "picks":        picks,
        "title":        title,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "total":        len(picks),
    }
    try:
        with open(MOCK_DRAFT_PATH, "w") as f:
            json.dump(_MOCK_DRAFT_DATA, f, separators=(",", ":"))
    except Exception:
        pass

    return jsonify({"success": True, "total": len(picks)})


@app.get("/api/mock-draft")
def get_mock_draft():
    return jsonify(_MOCK_DRAFT_DATA)


# ── High school prospects ──────────────────────────────────────────────────────

def load_hs_prospect_cache() -> None:
    global _HS_PROSPECT_CACHE, _HS_PROSPECT_CACHE_META, _HS_PROSPECT_CACHE_MTIME
    if not os.path.exists(HS_PROSPECT_CACHE_PATH):
        print("HS prospect cache not found (run build_hs_prospect_cache.py to create it).")
        return
    try:
        mtime = os.path.getmtime(HS_PROSPECT_CACHE_PATH)  # capture BEFORE reading
        with open(HS_PROSPECT_CACHE_PATH) as f:
            data = json.load(f)
        _HS_PROSPECT_CACHE = data.get("prospects", [])
        _HS_PROSPECT_CACHE_META = {
            "generated_at": data.get("generated_at"),
            "total":        data.get("total", len(_HS_PROSPECT_CACHE)),
            "years":        data.get("years", []),
        }
        _HS_PROSPECT_CACHE_MTIME = mtime
        print(f"Loaded {len(_HS_PROSPECT_CACHE)} HS prospects from cache.")
    except Exception as exc:
        print(f"Failed to load HS prospect cache: {exc}")


def _maybe_reload_hs_prospect_cache() -> None:
    """Hot-reload the HS prospect cache if the JSON file changed on disk."""
    try:
        mtime = os.path.getmtime(HS_PROSPECT_CACHE_PATH)
    except OSError:
        return
    if mtime == _HS_PROSPECT_CACHE_MTIME:
        return
    with _hs_prospect_reload_lock:
        if mtime == _HS_PROSPECT_CACHE_MTIME:
            return
        load_hs_prospect_cache()


@app.get("/api/hs-prospects")
@http_cached(_hs_prospects_mtimes)
def api_hs_prospects():
    _maybe_reload_hs_prospect_cache()
    pos_filter   = (request.args.get("position") or "").strip().upper()
    stars_filter = request.args.get("stars", "")
    year_filter  = request.args.get("year", "")
    search       = (request.args.get("search") or "").strip().lower()
    sort_by      = (request.args.get("sort") or "rank").strip()
    try:
        page  = max(0, int(request.args.get("page", 0)))
        limit = min(500, max(10, int(request.args.get("limit", 100))))
    except ValueError:
        page, limit = 0, 100
    offset = page * limit

    results = _HS_PROSPECT_CACHE

    _HS_POS_GROUPS = {
        "DB": {"CB", "S", "DB", "FS", "SS"},
        "LB": {"LB", "ILB", "OLB", "MLB"},
        "DL": {"DL", "DE", "DT", "EDGE", "NT"},
        "OL": {"OL", "OT", "OG", "C"},
    }

    if pos_filter and pos_filter != "ALL":
        group = _HS_POS_GROUPS.get(pos_filter)
        if group:
            results = [p for p in results if (p.get("position") or "").upper() in group]
        else:
            results = [p for p in results if (p.get("position") or "").upper() == pos_filter]

    if stars_filter and stars_filter != "ALL":
        try:
            s = int(stars_filter)
            results = [p for p in results if p.get("stars") == s]
        except ValueError:
            pass

    if year_filter and year_filter != "ALL":
        results = [p for p in results if str(p.get("year", "")) == year_filter]

    if search:
        results = [p for p in results if
                   search in (p.get("name") or "").lower() or
                   search in (p.get("school") or "").lower() or
                   search in (p.get("committed_to") or "").lower() or
                   search in (p.get("state") or "").lower()]

    if sort_by == "stars":
        results = sorted(results, key=lambda p: -(p.get("stars") or 0))
    elif sort_by == "name":
        results = sorted(results, key=lambda p: (p.get("name") or ""))
    elif sort_by == "rating":
        results = sorted(results, key=lambda p: -(float(p.get("rating") or 0)))
    else:  # default: rank
        results = sorted(results, key=lambda p: (p.get("ranking") or 9999))

    # Recruiting classes graduate: the class signed in February ENROLLS by
    # summer. From July on, "current HS" means next year's class. Rows from
    # enrolled classes are flagged so the UI never calls a college freshman
    # a high schooler again (Bryce Underwood was listed here in his second
    # college season).
    _lt = time.localtime()
    current_hs_class = _lt.tm_year + 1 if _lt.tm_mon >= 7 else _lt.tm_year
    # Roster cross-validation: appearing on an FBS roster (the college cache)
    # is PROOF of enrollment, stronger than class-year bookkeeping.
    _maybe_reload_prospect_cache()
    college_names = {(r.get("name") or "").lower().strip() for r in _PROSPECT_CACHE}
    out_rows = []
    for p in results[offset: offset + limit]:
        q = dict(p)
        try:
            stale_class = int(q.get("year") or 0) < current_hs_class
        except (TypeError, ValueError):
            stale_class = False
        on_roster = (q.get("name") or "").lower().strip() in college_names
        q["enrolled"] = bool(stale_class or on_roster)
        out_rows.append(q)

    meta = dict(_HS_PROSPECT_CACHE_META)
    meta["current_hs_class"] = current_hs_class

    return jsonify({
        "total":         len(results),
        "offset":        offset,
        "limit":         limit,
        "meta":          meta,
        "api_key_set":   bool(CFBD_API_KEY),
        "prospects":     out_rows,
    })


# ── Class big boards (hand-ordered rankings per draft class) ──────────────────
# training_data/big_boards.json maps draft-class → ordered list of player keys
# ("Name|Team"). The file lives on the app machine's disk: it survives restarts
# and hot-reloads on mtime like the other caches, but a redeploy resets it to
# the committed copy — export + commit to make rankings permanent (see DEPLOY.md).

_BIG_BOARD_SEED = {"2027": [], "2028": [], "2029": [], "2030": []}
_BIG_BOARDS: dict = {k: [] for k in _BIG_BOARD_SEED}
_BIG_BOARDS_MTIME: float = 0.0
_big_boards_reload_lock = threading.Lock()


def load_big_boards() -> None:
    global _BIG_BOARDS, _BIG_BOARDS_MTIME
    if not os.path.exists(BIG_BOARDS_PATH):
        print("Big boards file not found — serving empty seed boards.")
        return
    try:
        mtime = os.path.getmtime(BIG_BOARDS_PATH)  # capture BEFORE reading
        with open(BIG_BOARDS_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("big_boards.json root must be an object")
        _BIG_BOARDS = {str(k): [str(x) for x in v]
                       for k, v in data.items() if isinstance(v, list)}
        _BIG_BOARDS_MTIME = mtime
        counts = {k: len(v) for k, v in sorted(_BIG_BOARDS.items())}
        print(f"Loaded big boards: {counts}")
    except Exception as exc:
        # mtime NOT updated on failure (e.g. file mid-rewrite) → retried next request
        print(f"Failed to load big boards: {exc}")


def _maybe_reload_big_boards() -> None:
    """Hot-reload on mtime (same pattern as the prospect cache)."""
    try:
        mtime = os.path.getmtime(BIG_BOARDS_PATH)
    except OSError:
        return
    if mtime == _BIG_BOARDS_MTIME:
        return
    with _big_boards_reload_lock:
        if mtime == _BIG_BOARDS_MTIME:  # another thread already reloaded
            return
        load_big_boards()


def _big_board_guard_ok() -> bool:
    """EXACTLY the /api/analytics/summary guard: invisible until ANALYTICS_KEY
    is configured; wrong key is also treated as 404 so the endpoint's
    existence can't be probed."""
    configured = os.getenv("ANALYTICS_KEY", "").strip()
    if not configured:
        return False
    supplied = request.args.get("key") or request.headers.get("X-Analytics-Key") or ""
    return supplied == configured


def _prospect_key(p: dict) -> str:
    return f"{(p.get('name') or '').strip()}|{(p.get('team') or '').strip()}"


def _prospect_key_index() -> dict:
    """lowercase 'name|team' → cached row (first occurrence wins)."""
    idx: dict = {}
    for p in _PROSPECT_CACHE:
        idx.setdefault(_prospect_key(p).lower(), p)
    return idx


def _big_boards_updated_at() -> Optional[str]:
    if not _BIG_BOARDS_MTIME:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_BIG_BOARDS_MTIME))


@app.get("/api/big-board")
def api_big_board():
    _maybe_reload_prospect_cache()
    _maybe_reload_big_boards()
    cls = _safe_int(request.args.get("class") or 2027)
    if cls is None:
        return jsonify({"error": "class must be an integer"}), 400
    if str(cls) not in _BIG_BOARDS:
        return jsonify({"error": f"unknown class {cls}",
                        "classes": sorted(_BIG_BOARDS)}), 400

    idx = _prospect_key_index()
    board_rows, missing, on_board = [], [], set()
    for key in _BIG_BOARDS[str(cls)]:
        row = idx.get(key.strip().lower())
        if row is None:
            missing.append(key)  # skipped, but reported
            continue
        on_board.add(_prospect_key(row).lower())
        board_rows.append(row)

    # Remaining prospects of that class, model-ranked (grade, then success prob)
    rest = [p for p in _PROSPECT_CACHE
            if _safe_int(p.get("draft_class")) == cls
            and _prospect_key(p).lower() not in on_board]
    rest.sort(key=lambda p: (_GRADE_ORDER.get(p.get("grade"), 9),
                             -(p.get("success_probability") or 0)))

    return jsonify({
        "class":      cls,
        "board":      board_rows,
        "missing":    missing,
        "rest":       rest,
        "updated_at": _big_boards_updated_at(),
    })


@app.post("/api/big-board")
def api_big_board_save():
    if not _big_board_guard_ok():
        return jsonify({"error": "not found"}), 404
    _maybe_reload_prospect_cache()
    _maybe_reload_big_boards()

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400
    cls = _safe_int(payload.get("class"))
    if cls is None:
        return jsonify({"error": "class must be an integer"}), 400
    if str(cls) not in _BIG_BOARDS:
        return jsonify({"error": f"unknown class {cls}",
                        "classes": sorted(_BIG_BOARDS)}), 400
    board = payload.get("board")
    if not isinstance(board, list) or not all(isinstance(k, str) for k in board):
        return jsonify({"error": "board must be a list of 'Name|Team' strings"}), 400

    # Validate every key against the live prospect cache; canonicalize to the
    # cache's exact "Name|Team" spelling and drop duplicate keys (first wins).
    idx = _prospect_key_index()
    canonical, invalid, seen = [], [], set()
    for key in board:
        row = idx.get(key.strip().lower())
        if row is None:
            invalid.append(key)
            continue
        canon = _prospect_key(row)
        if canon.lower() in seen:
            continue
        seen.add(canon.lower())
        canonical.append(canon)
    if invalid:
        return jsonify({"error": "unknown player keys", "invalid_keys": invalid}), 400

    data = {k: list(v) for k, v in _BIG_BOARDS.items()}
    data[str(cls)] = canonical

    # Atomic write: temp file + os.replace, so a reader (or the mtime
    # hot-reloader in another worker) never sees a half-written file.
    tmp_path = BIG_BOARDS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, BIG_BOARDS_PATH)
    load_big_boards()  # this worker updates now; others hot-reload on mtime

    return jsonify({"ok": True, "class": cls, "count": len(canonical),
                    "updated_at": _big_boards_updated_at()})


@app.get("/api/big-board/export")
def api_big_board_export():
    if not _big_board_guard_ok():
        return jsonify({"error": "not found"}), 404
    _maybe_reload_big_boards()
    if not os.path.exists(BIG_BOARDS_PATH):
        # Never written on this machine — export the in-memory (seed) boards
        resp = jsonify(_BIG_BOARDS)
        resp.headers["Content-Disposition"] = 'attachment; filename="big_boards.json"'
        return resp
    return send_from_directory(
        os.path.abspath(os.path.dirname(BIG_BOARDS_PATH)),
        os.path.basename(BIG_BOARDS_PATH),
        as_attachment=True, max_age=0,
    )


# ── Prospect grade ────────────────────────────────────────────────────────────

def compute_prospect_grade(success_prob: Optional[float], draft_grade_class: Optional[int]) -> str:
    """Map calibrated success probability (%) → letter grade A+…D.

    Thresholds are EMPIRICAL PERCENTILES of the calibrated ensemble's success
    probability over the LIVE 8,150-player FBS board (2026-08-17: ESPN
    athlete-id passthrough gives 42% verified-stat coverage; players WITHOUT
    verified stats feed the model NaN and grade on position/tier priors,
    landing 11-25% — they no longer outrank verified stars on fabricated
    numbers). Recompute these cutoffs whenever the board is rebuilt after a
    model or feature change: sorted success_probability, value at each
    percentile below.
        A+ ≥ 98th (top 2%)   → p ≥ 41.9 (v3 model, 2026-08-19 board)
        A  ≥ 95th            → p ≥ 30.6
        A- ≥ 90th            → p ≥ 24.4
        B+ ≥ 80th            → p ≥ 20.7
        B  ≥ 70th            → p ≥ 19.8
        B- ≥ 55th            → p ≥ 18.4
        C+ ≥ 45th            → p ≥ 17.0   (C+/C straddle the median)
        C  ≥ 35th            → p ≥ 14.8
        C- ≥ 10th            → p ≥ 9.8
        D  < 10th (bottom 10%)
    draft_grade_class is intentionally NOT a gate anymore: the old
    (p, class) AND-gates structurally locked whole position groups (0 of
    1,556 OL could reach A-range). Grade caller passes class for API
    stability; it feeds the separate draft_grade display field instead.
    """
    if success_prob is None:
        return "D"  # fallback path with no calibrated probability
    p = float(success_prob)
    if p >= 41.9: return "A+"
    if p >= 33.5: return "A"
    if p >= 24.0: return "A-"
    if p >= 17.5: return "B+"
    if p >= 15.5: return "B"
    if p >= 13.7: return "B-"
    if p >= 12.3: return "C+"
    if p >= 9.8:  return "C"
    if p >= 4.4:  return "C-"
    return "D"


# ── Named historical players for similarity comps ─────────────────────────────
# Stored by position group so matching only compares same-position players
_POS_GROUP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "CB": "DB", "S": "DB", "DB": "DB", "FS": "DB", "SS": "DB",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DL": "DL", "DE": "DL", "DT": "DL", "EDGE": "DL",
    "OL": "OL", "OT": "OL", "OG": "OL", "G": "OL", "C": "OL",
}

NAMED_HISTORICAL_COMPS = [
    # QBs
    {"name":"Patrick Mahomes","position":"QB","conference_tier":5,"production_score":85,"combine_speed_score":72,"is_award_winner":0,"is_all_american":0,"nfl_success":1,"outcome":"3x Super Bowl MVP"},
    {"name":"Josh Allen","position":"QB","conference_tier":3,"production_score":76,"combine_speed_score":74,"is_award_winner":0,"is_all_american":0,"nfl_success":1,"outcome":"Elite starter, 4x Pro Bowl"},
    {"name":"Joe Burrow","position":"QB","conference_tier":1,"production_score":90,"combine_speed_score":70,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#1 Pick, Heisman, Super Bowl"},
    {"name":"Lamar Jackson","position":"QB","conference_tier":2,"production_score":88,"combine_speed_score":85,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"2x NFL MVP"},
    {"name":"Jalen Hurts","position":"QB","conference_tier":2,"production_score":80,"combine_speed_score":67,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"Super Bowl, 3x Pro Bowl"},
    {"name":"Brock Purdy","position":"QB","conference_tier":4,"production_score":72,"combine_speed_score":62,"is_award_winner":0,"is_all_american":0,"nfl_success":1,"outcome":"49ers starter (Mr. Irrelevant, R7)"},
    {"name":"Trevor Lawrence","position":"QB","conference_tier":1,"production_score":85,"combine_speed_score":74,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#1 Overall Pick"},
    # WRs
    {"name":"Justin Jefferson","position":"WR","conference_tier":2,"production_score":88,"combine_speed_score":93,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#22 Pick, 4x Pro Bowl, All-Pro"},
    {"name":"Ja'Marr Chase","position":"WR","conference_tier":1,"production_score":85,"combine_speed_score":90,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#5 Pick, 2x All-Pro"},
    {"name":"Devonta Smith","position":"WR","conference_tier":1,"production_score":86,"combine_speed_score":79,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"Heisman, #10 Pick"},
    {"name":"Tyreek Hill","position":"WR","conference_tier":8,"production_score":82,"combine_speed_score":96,"is_award_winner":0,"is_all_american":0,"nfl_success":1,"outcome":"7x Pro Bowl (small-school speed)"},
    {"name":"CeeDee Lamb","position":"WR","conference_tier":2,"production_score":80,"combine_speed_score":83,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#17 Pick, 2x All-Pro"},
    # RBs
    {"name":"Saquon Barkley","position":"RB","conference_tier":3,"production_score":90,"combine_speed_score":82,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#2 Pick, 3x Pro Bowl"},
    {"name":"Christian McCaffrey","position":"RB","conference_tier":3,"production_score":88,"combine_speed_score":88,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#8 Pick, 4x Pro Bowl"},
    {"name":"Bijan Robinson","position":"RB","conference_tier":2,"production_score":85,"combine_speed_score":80,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#8 Pick, Pro Bowl"},
    {"name":"Ashton Jeanty","position":"RB","conference_tier":7,"production_score":88,"combine_speed_score":85,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"2025 Draft — projected Top 10"},
    # TEs
    {"name":"Travis Kelce","position":"TE","conference_tier":4,"production_score":80,"combine_speed_score":68,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"9x Pro Bowl (Day 2 pick!)"},
    {"name":"Kyle Pitts","position":"TE","conference_tier":2,"production_score":82,"combine_speed_score":72,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#4 Overall Pick"},
    {"name":"George Kittle","position":"TE","conference_tier":3,"production_score":78,"combine_speed_score":65,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"5x Pro Bowl"},
    # DBs (CB + S)
    {"name":"Patrick Surtain II","position":"CB","conference_tier":1,"production_score":78,"combine_speed_score":91,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#9 Pick, 2x All-Pro"},
    {"name":"Sauce Gardner","position":"CB","conference_tier":4,"production_score":75,"combine_speed_score":89,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#4 Pick, DROY, All-Pro"},
    {"name":"Kyle Hamilton","position":"S","conference_tier":2,"production_score":80,"combine_speed_score":82,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#14 Pick, 2x Pro Bowl"},
    {"name":"Derwin James","position":"S","conference_tier":1,"production_score":78,"combine_speed_score":84,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#17 Pick, 3x Pro Bowl"},
    # LBs
    {"name":"Micah Parsons","position":"LB","conference_tier":3,"production_score":82,"combine_speed_score":78,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#12 Pick, 3x All-Pro"},
    {"name":"Will Anderson Jr.","position":"LB","conference_tier":1,"production_score":85,"combine_speed_score":80,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#3 Pick, Nagurski, Pro Bowl"},
    {"name":"Roquan Smith","position":"LB","conference_tier":1,"production_score":78,"combine_speed_score":74,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#8 Pick, 2x All-Pro"},
    # DLs
    {"name":"Chase Young","position":"DL","conference_tier":1,"production_score":88,"combine_speed_score":82,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#2 Pick, Nagurski, DROY"},
    {"name":"Myles Garrett","position":"DL","conference_tier":2,"production_score":85,"combine_speed_score":78,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#1 Pick, 4x All-Pro"},
    {"name":"Jalen Carter","position":"DL","conference_tier":1,"production_score":82,"combine_speed_score":80,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#9 Pick, Nagurski"},
    # OLs
    {"name":"Penei Sewell","position":"OL","conference_tier":5,"production_score":55,"combine_speed_score":52,"is_award_winner":1,"is_all_american":1,"nfl_success":1,"outcome":"#7 Pick, Outland Trophy, All-Pro"},
    {"name":"Tristan Wirfs","position":"OL","conference_tier":3,"production_score":52,"combine_speed_score":60,"is_award_winner":0,"is_all_american":1,"nfl_success":1,"outcome":"#13 Pick, 2x All-Pro"},

    # ── Busts / journeymen (nfl_success=0) — honest comps for low-probability
    #    profiles. Real, well-documented outcomes; feature values are the same
    #    hand-curated 0-100 scales as the success entries above. ──────────────
    # QBs
    {"name":"JaMarcus Russell","position":"QB","conference_tier":1,"production_score":78,"combine_speed_score":55,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"#1 Pick 2007 — out of the NFL by 2010"},
    {"name":"Johnny Manziel","position":"QB","conference_tier":1,"production_score":85,"combine_speed_score":75,"is_award_winner":1,"is_all_american":1,"nfl_success":0,"outcome":"Heisman; R1 2014 — out of the NFL by 2016"},
    {"name":"Zach Wilson","position":"QB","conference_tier":5,"production_score":80,"combine_speed_score":70,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"#2 Pick 2021 — benched, journeyman backup"},
    # WRs
    {"name":"John Ross","position":"WR","conference_tier":2,"production_score":78,"combine_speed_score":99,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"#9 Pick 2017 (4.22s forty) — injuries, out by 2022"},
    {"name":"Corey Coleman","position":"WR","conference_tier":2,"production_score":82,"combine_speed_score":90,"is_award_winner":1,"is_all_american":1,"nfl_success":0,"outcome":"Biletnikoff; #15 Pick 2016 — out of the NFL by 2020"},
    # RBs
    {"name":"Trent Richardson","position":"RB","conference_tier":1,"production_score":90,"combine_speed_score":78,"is_award_winner":1,"is_all_american":1,"nfl_success":0,"outcome":"Doak Walker; #3 Pick 2012 — out of the NFL by 2015"},
    {"name":"Rashaad Penny","position":"RB","conference_tier":7,"production_score":90,"combine_speed_score":80,"is_award_winner":0,"is_all_american":1,"nfl_success":0,"outcome":"#27 Pick 2018 — injury-plagued journeyman"},
    # TEs
    {"name":"O.J. Howard","position":"TE","conference_tier":1,"production_score":65,"combine_speed_score":80,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"#19 Pick 2017 — journeyman, never a full-time starter"},
    # DBs
    {"name":"Justin Gilbert","position":"CB","conference_tier":2,"production_score":70,"combine_speed_score":90,"is_award_winner":0,"is_all_american":1,"nfl_success":0,"outcome":"#8 Pick 2014 — out of the NFL by 2017"},
    {"name":"Jeff Okudah","position":"CB","conference_tier":1,"production_score":72,"combine_speed_score":85,"is_award_winner":0,"is_all_american":1,"nfl_success":0,"outcome":"#3 Pick 2020 — injuries, journeyman"},
    # LBs
    {"name":"Reuben Foster","position":"LB","conference_tier":1,"production_score":80,"combine_speed_score":72,"is_award_winner":1,"is_all_american":1,"nfl_success":0,"outcome":"Butkus; #31 Pick 2017 — out of the NFL by 2020"},
    # DLs
    {"name":"Dion Jordan","position":"DL","conference_tier":2,"production_score":68,"combine_speed_score":85,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"#3 Pick 2013 — suspensions, journeyman"},
    {"name":"Solomon Thomas","position":"DL","conference_tier":3,"production_score":72,"combine_speed_score":75,"is_award_winner":0,"is_all_american":1,"nfl_success":0,"outcome":"#3 Pick 2017 — rotational journeyman"},
    # OLs
    {"name":"Greg Robinson","position":"OL","conference_tier":1,"production_score":50,"combine_speed_score":55,"is_award_winner":0,"is_all_american":1,"nfl_success":0,"outcome":"#2 Pick 2014 — bust, out of the NFL by 2020"},
    {"name":"Isaiah Wilson","position":"OL","conference_tier":1,"production_score":45,"combine_speed_score":45,"is_award_winner":0,"is_all_american":0,"nfl_success":0,"outcome":"R1 2020 — played 4 NFL snaps, out of the league"},
]


_COMP_BANK = None          # list of dicts built from combine_outcomes.csv
_COMP_BANK_STATS = None    # per-feature (mean, std) over the bank

# Similarity features: (key, weight). Values are percentile/z-comparable at
# serve time; bank production is percentile-ized through the same table the
# serving path uses. Distance is masked — only features BOTH sides have.
_COMP_FEATURES = [
    ("production", 2.0), ("speed", 1.2), ("tier", 1.0), ("stars", 1.5),
    ("height", 0.7), ("weight", 0.7), ("years", 0.8),
]


def _load_comp_bank() -> None:
    """Historical comp bank: real prospects 2000-2023 with known outcomes."""
    global _COMP_BANK, _COMP_BANK_STATS
    import csv as _csv
    from dv_features import _raw_production_to_percentile
    bank = []
    try:
        with open(TRAINING_DATA_PATH) as fh:
            for r in _csv.DictReader(fh):
                try:
                    year = int(float(r.get("draft_year") or 0))
                except ValueError:
                    continue
                if not (2000 <= year <= 2023):
                    continue  # outcome must be settled
                pos = (r.get("position") or "").upper()
                group = _POS_GROUP.get(pos, pos)
                if not group:
                    continue

                def fv(key):
                    v = r.get(key)
                    try:
                        f = float(v)
                        return None if math.isnan(f) else f
                    except (TypeError, ValueError):
                        return None

                raw_prod = fv("production_score")
                feats = {
                    "production": (_raw_production_to_percentile(group, raw_prod)
                                   if raw_prod is not None else None),
                    "speed":  fv("combine_speed_score"),
                    "tier":   fv("conference_tier"),
                    "stars":  fv("rec_stars"),
                    "height": fv("height_in"),
                    "weight": fv("weight_lb"),
                    "years":  fv("years_in_college"),
                }
                rd = fv("draft_round")
                rd = int(rd) if rd is not None else 8
                bits = [f"R{rd} · {year}" if rd <= 7 else f"UDFA · {year}"]
                pb = int(fv("pro_bowls") or 0)
                ss = int(fv("seasons_started") or 0)
                if pb:
                    bits.append(f"{pb}x Pro Bowl")
                elif ss >= 3:
                    bits.append(f"{ss} seasons as a starter")
                elif int(fv("nfl_success") or 0) == 0 and rd <= 3:
                    bits.append("did not stick")
                bank.append({
                    "name": (r.get("name") or "").strip(),
                    "position": pos, "group": group, "feats": feats,
                    "outcome": " · ".join(bits),
                    "nfl_success": int(fv("nfl_success") or 0),
                })
    except Exception as exc:
        print(f"Comp bank unavailable ({exc}) — comps disabled")
        _COMP_BANK, _COMP_BANK_STATS = [], {}
        return
    stats = {}
    for key, _w in _COMP_FEATURES:
        vals = [b["feats"][key] for b in bank if b["feats"][key] is not None]
        if len(vals) >= 30:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            stats[key] = (mean, max(var ** 0.5, 1e-6))
    _COMP_BANK, _COMP_BANK_STATS = bank, stats
    print(f"Comp bank loaded: {len(bank)} historical prospects (2000-2023).")


def find_historical_comps(player_stats: Dict[str, object], n: int = 3) -> list:
    """Top-n statistically closest REAL historical prospects (2000-2023),
    nearest-neighbor over a masked z-scored feature distance within the
    player's position group. Replaces the old 46-name hand-curated list,
    whose tiny per-position pools gave different players identical comps."""
    if _COMP_BANK is None:
        _load_comp_bank()
    if not _COMP_BANK:
        return []

    pos = (str(player_stats.get("position", "") or "")).upper()
    group = _POS_GROUP.get(pos, pos)

    def _num(v):
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    name = str(player_stats.get("name") or "")
    team = str(player_stats.get("team") or "")
    enr = enrichment_for(name, team) or {}
    phys_real = bool(player_stats.get("physical_is_real"))
    me = {
        "production": _num(player_stats.get("production_score")),
        "speed":  _num(player_stats.get("combine_speed_score")),
        "tier":   _num(player_stats.get("conference_tier")),
        "stars":  _num(enr.get("stars")),
        # hash-estimated height/weight must not drive similarity
        "height": _num(player_stats.get("height_inches")) if phys_real else None,
        "weight": _num(player_stats.get("weight_lbs")) if phys_real else None,
        # years in college at the next draft, from the recruiting class
        "years": (
            (time.localtime().tm_year + 1) - int(enr["recruit_year"])
            if _num(enr.get("recruit_year")) else None
        ),
    }

    scored = []
    for b in _COMP_BANK:
        if b["group"] != group:
            continue
        acc, wsum, used = 0.0, 0.0, 0
        for key, w in _COMP_FEATURES:
            a, c = me.get(key), b["feats"][key]
            st = _COMP_BANK_STATS.get(key)
            if a is None or c is None or st is None:
                continue
            z = (a - c) / st[1]
            acc += w * z * z
            wsum += w
            used += 1
        if used < 2:
            continue  # not enough shared evidence for an honest comp
        dist = (acc / wsum) ** 0.5
        scored.append((dist, used, b))
    scored.sort(key=lambda x: x[0])

    out = []
    for dist, used, b in scored[:n]:
        # exp decay reads naturally (dist 0 -> ~99, 1σ avg -> ~37); capped
        # below 100 because "100% match" overclaims, and thin-evidence comps
        # (few shared features) are further discounted.
        sim = min(99.0, 100.0 * math.exp(-dist))
        if used < 4:
            sim = min(sim, 60.0 + 5.0 * used)
        out.append({
            "name": b["name"],
            "position": b["position"],
            "similarity": max(1, round(sim)),
            "outcome": b["outcome"],
            "nfl_success": b["nfl_success"],
        })
    return out


# ── Kalshi market-edge board (dv_edge.py) ─────────────────────────────────────
# Read-only analysis of public Kalshi prices vs model output; all discovery,
# matching, honest-mapping policy, and the paper ledger live in dv_edge.py.
import dv_edge


@app.route("/api/edge")
def api_edge():
    _maybe_reload_prospect_cache()
    return jsonify(dv_edge.edge_payload(_PROSPECT_CACHE))


@app.route("/api/edge/ledger")
def api_edge_ledger():
    return jsonify(dv_edge.ledger_payload())


# ── Serve React production build ──────────────────────────────────────────────
BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path=""):
    # Serve real files (JS chunks, CSS, images, favicon, etc.) directly.
    # Everything else falls through to index.html so React Router handles it.
    target = os.path.join(BUILD_DIR, path)
    if path and os.path.isfile(target):
        return send_from_directory(BUILD_DIR, path)
    return send_from_directory(BUILD_DIR, "index.html")


# ── Combined init endpoint (replaces 3 separate fetches on page load) ─────────
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Sources that appear in the browseable prospect list
PROSPECT_SOURCES = ("college_prospect", "nfl_draft_2025", "freshman_2026")

@app.get("/init")
@http_cached(_init_mtimes)
def init_data():
    cached = cache_get("init")
    if cached:
        return jsonify(cached)

    # Primary source: the in-memory prospect cache (grade-sorted, no SQL) —
    # in prod the players DB only holds seed rows, so seeding the predict
    # page from it surfaced ~13 names.
    _maybe_reload_prospect_cache()
    if _PROSPECT_CACHE:
        players = [
            {
                "name":     p.get("name") or "",
                "position": p.get("position") or "",
                "team":     p.get("team") or "",
                "source":   "college_prospect",
            }
            for p in _PROSPECT_CACHE
        ]
        teams = sorted({p.get("team") or "" for p in _PROSPECT_CACHE} - {"", "Unknown"})
        positions = sorted({(p.get("position") or "").upper() for p in _PROSPECT_CACHE}
                           - {"", "UNK", "UNKNOWN"})
        payload = {"players": players, "teams": teams, "positions": positions}
        cache_set("init", payload)
        return jsonify(payload)

    # Fallback: legacy DB path (dev environments without the cache file)
    conn = _get_conn()
    cursor = conn.cursor()
    ph = _placeholder()

    # All prospect sources, skill positions first, then alphabetical
    source_placeholders = ", ".join([ph] * len(PROSPECT_SOURCES))
    cursor.execute(
        f"""
        SELECT name, position, team, jersey, source FROM players
        WHERE source IN ({source_placeholders})
          AND upper(position) NOT IN ('UNK', 'UNKNOWN', '')
        ORDER BY
          CASE WHEN upper(position) IN ('QB','RB','WR','TE') THEN 0 ELSE 1 END ASC,
          source ASC,
          name ASC
        LIMIT 6000
        """,
        PROSPECT_SOURCES,
    )
    players = _rows_as_dicts(cursor)

    # Teams from all prospect sources
    cursor.execute(
        f"""
        SELECT DISTINCT team FROM players
        WHERE source IN ({source_placeholders})
          AND team != 'Unknown'
        ORDER BY team ASC
        """,
        PROSPECT_SOURCES,
    )
    teams = [r[0] for r in cursor.fetchall()]

    # All positions present in DB
    cursor.execute(
        f"""
        SELECT DISTINCT upper(position) FROM players
        WHERE source IN ({source_placeholders})
          AND upper(position) NOT IN ('UNK', 'UNKNOWN', '')
        ORDER BY upper(position)
        """,
        PROSPECT_SOURCES,
    )
    positions = [r[0] for r in cursor.fetchall()]

    conn.close()

    payload = {"players": players, "teams": teams, "positions": positions}
    cache_set("init", payload)
    return jsonify(payload)


@app.get("/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "position_model_loaded": position_model is not None,
            "success_model_loaded": success_model is not None,
            "encoder_loaded": label_encoders is not None,
            "player_db_size": player_database_count(),
            "college_prospect_count": player_database_count_by_source("college_prospect"),
        }
    )


def search_players_filtered(query: str = "", limit: int = 200,
                             source: Optional[str] = None,
                             position: Optional[str] = None,
                             team: Optional[str] = None):
    safe_limit = max(1, min(limit, 1000))
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()

    conditions = []
    params = []

    if source:
        conditions.append(f"source = {ph}")
        params.append(source)
    if query:
        conditions.append(f"name LIKE {ph}")
        params.append(f"%{query.strip()}%")
    if position and position.upper() != "ALL":
        conditions.append(f"upper(position) = {ph}")
        params.append(position.upper())
    if team and team != "ALL":
        conditions.append(f"team = {ph}")
        params.append(team)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(safe_limit)
    cursor.execute(f"SELECT name, position, team, jersey, source FROM players {where} ORDER BY name ASC LIMIT {ph}", params)
    rows = _rows_as_dicts(cursor)
    conn.close()
    return rows


@app.get("/search")
def search_all():
    """Fast autocomplete search across ALL player sources.

    Order: (a) in-memory college prospect cache (kind:"college"),
    (b) HS prospect cache (kind:"hs"), (c) legacy players DB (kind:"db").
    (a) and (b) never touch SQL, so search works in prod even when the
    players DB only holds the handful of seed rows (the old "13 players"
    bug came from this endpoint reading the DB exclusively).
    """
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"players": []})
    cached_key = f"search:{q.lower()}"
    cached = cache_get(cached_key)
    if cached:
        return jsonify(cached)

    _maybe_reload_prospect_cache()
    _maybe_reload_hs_prospect_cache()
    ql = q.lower()

    # (a) College prospects — full in-memory scan (8k rows, cheap).
    college = []
    for p in _PROSPECT_CACHE:
        name = p.get("name") or ""
        team = p.get("team") or ""
        if ql in name.lower() or ql in team.lower():
            row = {
                "name":                name,
                "position":            p.get("position") or "",
                "team":                team,
                "espn_team_id":        p.get("espn_team_id") or "",
                "grade":               p.get("grade"),
                "success_probability": p.get("success_probability"),
                "kind":                "college",
                "source":              "college_prospect",  # legacy row shape
            }
            # Older cache builds don't carry the player's own espn_id —
            # omit it and let /predict resolve by name+team.
            if p.get("espn_id"):
                row["espn_id"] = p["espn_id"]
            college.append(row)
    # Prefix name matches first; ties keep the cache's grade ordering.
    college.sort(key=lambda r: 0 if r["name"].lower().startswith(ql) else 1)
    college = college[:14]

    # (b) High-school prospects — name/school match.
    hs = []
    for p in _HS_PROSPECT_CACHE:
        name = p.get("name") or ""
        school = p.get("school") or ""
        if ql in name.lower() or ql in school.lower():
            hs.append({
                "name":     name,
                "position": p.get("position") or "",
                "school":   school,
                "stars":    p.get("stars"),
                "year":     p.get("year"),
                "kind":     "hs",
                "source":   "hs_prospect",
            })
    hs.sort(key=lambda r: 0 if r["name"].lower().startswith(ql) else 1)
    hs = hs[:5]

    # (c) Legacy players DB (NFL seed names). Best-effort — an empty or
    # missing DB must never break search.
    db_rows = []
    try:
        seen = {r["name"].lower() for r in college}
        ph = _placeholder()
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name, position, team, source, espn_id FROM players "
            f"WHERE name LIKE {ph} ORDER BY "
            "CASE source WHEN 'college_prospect' THEN 0 WHEN 'nfl_seed' THEN 1 WHEN 'legacy' THEN 2 ELSE 3 END, "
            "name ASC LIMIT 10",
            (f"%{q}%",),
        )
        for r in _rows_as_dicts(cursor):
            if (r.get("name") or "").lower() in seen:
                continue
            row = {
                "name":     r.get("name") or "",
                "position": r.get("position") or "",
                "team":     r.get("team") or "",
                "kind":     "db",
                "source":   r.get("source") or "legacy",
            }
            if r.get("espn_id"):
                row["espn_id"] = r["espn_id"]
            db_rows.append(row)
        conn.close()
    except Exception as exc:
        print(f"/search DB leg skipped: {exc}")

    payload = {"players": (college + hs + db_rows)[:20]}
    cache_set(cached_key, payload)
    return jsonify(payload)


@app.get("/teams")
def teams():
    source = (request.args.get("source") or "college_prospect").strip()
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    if source and source.lower() != "all":
        cursor.execute(f"SELECT DISTINCT team FROM players WHERE source = {ph} AND team != 'Unknown' ORDER BY team ASC", (source,))
    else:
        cursor.execute("SELECT DISTINCT team FROM players WHERE team != 'Unknown' ORDER BY team ASC")
    teams_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"teams": teams_list})


@app.get("/positions")
def positions():
    source = (request.args.get("source") or "college_prospect").strip()
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    if source and source.lower() != "all":
        cursor.execute(f"SELECT DISTINCT upper(position) FROM players WHERE source = {ph} AND position != 'UNK' ORDER BY position ASC", (source,))
    else:
        cursor.execute("SELECT DISTINCT upper(position) FROM players WHERE position != 'UNK' ORDER BY position ASC")
    pos_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"positions": pos_list})


@app.get("/players")
def players():
    query = (request.args.get("q") or "").strip()
    limit_raw = request.args.get("limit", "200")
    source = (request.args.get("source") or "college_prospect").strip()
    position = (request.args.get("position") or "").strip()
    team = (request.args.get("team") or "").strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 200

    source_filter = source if source.lower() != "all" else None
    pos_filter = position if position and position.upper() != "ALL" else None
    team_filter = team if team and team != "ALL" else None
    players_list = search_players_filtered(query, limit, source_filter, pos_filter, team_filter)
    return jsonify({"players": players_list})


@app.post("/sync/college-prospects")
def sync_college_prospects_endpoint():
    payload = request.get_json(silent=True) or {}

    try:
        max_teams = int(payload.get("max_teams", 250))
    except (TypeError, ValueError):
        max_teams = 250
    try:
        max_players = int(payload.get("max_players", 4000))
    except (TypeError, ValueError):
        max_players = 4000

    try:
        result = sync_college_prospects(max_teams=max_teams, max_players=max_players)
        cache_invalidate("init")
        return jsonify(
            {
                "status": "ok",
                "synced": result,
                "player_db_size": player_database_count(),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prospect sync failed: {exc}"}), 500


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    player_name_raw = payload.get("name", "")
    if not isinstance(player_name_raw, str):
        return jsonify({"error": "Player name must be a string."}), 400

    player_name = player_name_raw.strip()
    if len(player_name) < 2:
        return jsonify({"error": "Player name must be at least 2 characters."}), 400
    if len(player_name) > 80:
        return jsonify({"error": "Player name is too long."}), 400

    # Optional roster hints — used as fallback when ESPN can't resolve the player
    fallback_position = str(payload.get("position") or "Unknown").strip().upper() or "Unknown"
    fallback_team = str(payload.get("team") or "Unknown").strip() or "Unknown"
    espn_id_hint = str(payload.get("espn_id") or "").strip()[:20]

    player_data, data_source = fetch_player_data(player_name, fallback_position, fallback_team,
                                                 espn_id_hint=espn_id_hint)
    if not player_data:
        return jsonify({"error": "Unable to resolve player data."}), 404

    # Use the position already resolved from ESPN/DB; the legacy position model
    # was trained on old NFL data and reliably returns None for current players.
    _raw_pos = str(player_data.get("position") or "").strip().upper()
    predicted_position = _raw_pos if _raw_pos and _raw_pos not in {"UNKNOWN", "UNK", ""} else (
        predict_position_with_model(player_data) or "Unknown"
    )

    if SERVE_MODE == "heuristic":
        # Decision gate (models/metadata.json): the rule-based baseline beat the
        # ensemble on the holdout, so the fallback IS the served predictor.
        success_label, confidence, success_probability, model_used = None, None, None, False
    else:
        success_label, confidence, success_probability, model_used = predict_success_with_model(player_data)

    if not success_label:
        success_label, confidence, reasoning = determine_success_fallback(player_data, predicted_position)
    else:
        reasoning = "Calibrated XGBoost+CatBoost ensemble prediction from college production, athleticism, and conference tier."

    # Draft grade + prospect grade
    draft_grade_label_str, draft_grade_class, draft_grade_prob = predict_draft_grade(player_data)
    prospect_grade = compute_prospect_grade(success_probability, draft_grade_class)

    position = str(player_data.get("position", "Unknown"))
    draft_round = int(player_data.get("draft_round") or 8)
    combine_speed = float(player_data.get("combine_speed_score") or 50.0)
    if not math.isfinite(combine_speed):
        combine_speed = 0.0  # display only — the model saw NaN
    conference_tier = int(player_data.get("conference_tier") or classify_college_tier(
        str(player_data.get("team", "") or "")))
    _prod_val = player_data.get("production_score")
    production = float(
        _prod_val if _prod_val is not None
        else compute_production_score(position, player_data)
    )
    if not math.isfinite(production):
        production = 0.0  # display only — the model saw NaN, the UI shows 0

    round_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
                    5: "5th", 6: "6th", 7: "7th", 8: "Undrafted"}

    stored_team = str(player_data.get("team", "") or "")
    is_nfl_player = is_nfl_franchise(stored_team)
    if is_nfl_player:
        tier_label = "Power 5 (NFL Pro)"
    elif conference_tier == 1:
        tier_label = "Tier 1 (Elite P5)"
    elif conference_tier <= 2:
        tier_label = "Tier 2 (Major P5)"
    elif conference_tier <= 4:
        tier_label = "Tier 3-4 (P5/G5)"
    elif conference_tier <= 6:
        tier_label = "Tier 5-6 (G5)"
    elif conference_tier <= 8:
        tier_label = "Tier 7-8 (Mid-Major)"
    else:
        tier_label = "Tier 9-10 (FCS)"

    season_label = str(player_data.get("_season") or "")
    completion_pct = str(player_data.get("_completion_pct") or "")
    interceptions = player_data.get("_interceptions")
    qb_rating = str(player_data.get("_qb_rating") or "")

    summary = {
        "draft_grade":         draft_grade_label_str or round_labels.get(draft_round, "Undrafted"),
        "combine_athleticism": f"{combine_speed:.0f} / 100",
        "college_level":       tier_label,
        "production_score":    f"{production:.0f} / 100",
    }
    if season_label:
        summary["season"] = season_label
    if completion_pct and position.upper() == "QB":
        summary["completion_pct"] = f"{completion_pct}%"
    if qb_rating and position.upper() == "QB":
        summary["passer_rating"] = qb_rating
    if interceptions is not None and position.upper() == "QB":
        summary["interceptions"] = str(interceptions)

    # Historical player comps
    historical_comps = find_historical_comps(player_data)

    # Full physical profile dict for frontend display
    physical = {
        "height_inches":   player_data.get("height_inches", 0),
        "weight_lbs":      player_data.get("weight_lbs", 0),
        "display_height":  player_data.get("display_height", ""),
        "display_weight":  player_data.get("display_weight", ""),
        "combine_forty":   player_data.get("combine_forty", 0),
        "vertical_inches": player_data.get("combine_vertical", 0),
        "combine_bench":   player_data.get("combine_bench", 0),
        "combine_broad":   player_data.get("combine_broad", 0),
        "combine_shuttle": player_data.get("combine_shuttle", 0),
        "combine_3cone":   player_data.get("combine_3cone", 0),
        "height_score":    player_data.get("height_score", 50),
        "weight_score":    player_data.get("weight_score", 50),
        "vert_score":      player_data.get("vert_score", 50),
        "is_real":         bool(player_data.get("physical_is_real", False)),
    }

    return jsonify(_json_safe(
        {
            "requested_name":      player_name,
            "resolved_name":       str(player_data.get("name", player_name)),
            "success":             success_label,
            "confidence":          confidence,
            "reasoning":           reasoning,
            "predicted_position":  predicted_position,
            "success_probability": success_probability,
            "model_confidence":    confidence,
            "model_used":          model_used,
            "model_type":          "success_classifier",
            "data_source":         data_source,
            "prospect_grade":      prospect_grade,
            "draft_grade":         draft_grade_label_str,
            "draft_grade_class":   draft_grade_class,
            "draft_grade_prob":    draft_grade_prob,
            "historical_comps":    historical_comps,
            "physical":            physical,
            "stats":               player_data,
            "summary":             summary,
            # Per-player SHAP contributions when the ensemble served the
            # prediction; global gain importances otherwise (heuristic mode /
            # fallback — per-player attributions of an unserved model would lie).
            "top_factors":         (per_player_top_factors(player_data, 5)
                                    if model_used else top_feature_importances(4)),
        }
    ))


def _json_safe(obj):
    """NaN/Inf are invalid in strict JSON (browsers reject them) — map to None.

    Needed because the response embeds the raw profile dict, whose numeric
    fields are NaN for players with no verified stats."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


initialize_player_database()
load_position_model_artifacts()
load_success_models()
load_draft_grade_models()
load_prospect_cache()
load_enrichment()
load_board_movers()
load_mock_draft()
load_hs_prospect_cache()
load_big_boards()

# ── Usage analytics + optional Auth0 login (dv_analytics.py) ──────────────────
# Registers before/after_request logging hooks, the /api/analytics/summary
# endpoint (guarded by ANALYTICS_KEY), and — when AUTH0_DOMAIN/AUTH0_AUDIENCE
# are set — Bearer-token verification that attaches g.user_sub.
import dv_analytics

dv_analytics.init_app(
    app,
    get_conn=_get_conn,
    placeholder=_placeholder,
    use_postgres=USE_POSTGRES,
)

AUTO_SYNC_COLLEGE_PROSPECTS = os.getenv("AUTO_SYNC_COLLEGE_PROSPECTS", "true").lower() == "true"
if AUTO_SYNC_COLLEGE_PROSPECTS and player_database_count_by_source("college_prospect") == 0:
    try:
        synced = sync_college_prospects(max_teams=220, max_players=3000)
        print(f"Auto-synced college prospects: {synced}")
    except Exception as exc:
        print(f"Auto-sync skipped due to error: {exc}")

# Pre-warm the /init cache so the first page load is instant
try:
    with app.app_context():
        init_data()
    print("Init cache warmed.")
except Exception as exc:
    print(f"Cache warm-up skipped: {exc}")


if __name__ == "__main__":
    # DV_PORT/PORT let a throwaway/smoke-test instance run beside the main :5001 server
    app.run(debug=False, use_reloader=False,
            port=int(os.getenv("DV_PORT") or os.getenv("PORT") or "5001"))
