#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Future Star Predictor backend.

- Provides health and prediction endpoints.
- Uses a direct ML success classifier for Success/No Success output.
- Falls back to rule scoring only if model inference is unavailable.
"""

import json
import os
import random
import sqlite3
import time
import hashlib
import threading
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Optional heavy models — imported at module level so startup clearly shows status
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed (pip install catboost) — XGBoost-only mode")

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not installed (pip install tabpfn) — skipping TabPFN model")

POSITION_MODEL_PATH        = "nfl_xgboost_model.json"
ENCODER_PATH               = "label_encoders.pkl"
SUCCESS_MODEL_PATH         = "success_xgboost_model.json"        # raw XGBoost (legacy load)
SUCCESS_CALIBRATED_PATH    = "success_calibrated_model.pkl"      # CalibratedClassifierCV (primary)
CATBOOST_SUCCESS_PATH      = "catboost_success_model.cbm"
TABPFN_SUCCESS_PATH        = "tabpfn_success_model.pkl"
DRAFT_GRADE_MODEL_PATH     = "draft_grade_model.json"            # raw XGBoost
DRAFT_GRADE_CALIBRATED_PATH = "draft_grade_calibrated_model.pkl" # calibrated primary
CATBOOST_DRAFT_GRADE_PATH  = "catboost_draft_grade_model.cbm"
TRAINING_DATA_PATH         = "training_data/combine_outcomes.csv"
PROSPECT_CACHE_PATH        = "training_data/prospect_cache.json"
MOCK_DRAFT_PATH            = "mock_draft.json"
HS_PROSPECT_CACHE_PATH     = "training_data/hs_prospect_cache.json"
CFBD_API_KEY               = os.environ.get("CFBD_API_KEY", "")
CFBD_BASE_URL              = "https://api.collegefootballdata.com"
PLAYER_DATA_PATH     = "nfl_players.csv"
PLAYER_DB_PATH       = "players.db"
ESPN_CFB_TEAMS_URL        = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
ESPN_CFB_TEAM_ROSTER_URL  = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"
ESPN_CFB_ATHLETE_OVERVIEW_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{espn_id}/overview"
ESPN_NFL_CORE_ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{espn_id}"
HTTP_TIMEOUT_SECONDS = 12
STATS_CACHE_TTL = 3600  # cache real stats for 1 hour

# Features shared by both models — purely college-performance based, NO draft_round
_BASE_FEATURES = [
    "production_score",      # 0–100 composite (position-normalized)
    "games_played",
    "combine_speed_score",   # 0–100 (position-normalized 40-yard dash)
    "conference_tier",       # 1 (SEC elite) → 10 (FCS lower)
    "is_award_winner",       # 1 if Nagurski/Heisman/Bednarik/etc.
    "is_all_american",       # 1 if first-team All-American
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

# NFL franchise keywords — players stored with these team names are active pros
# who came from college; default their tier to P5 (1) since reaching the NFL
# almost always implies a major-program background.
NFL_FRANCHISE_KEYWORDS = {
    "chiefs", "bills", "bengals", "ravens", "browns", "steelers", "texans",
    "colts", "jaguars", "titans", "broncos", "raiders", "chargers", "dolphins",
    "patriots", "jets", "giants", "eagles", "cowboys", "commanders", "bears",
    "lions", "packers", "vikings", "falcons", "panthers", "saints", "buccaneers",
    "cardinals", "rams", "seahawks", "49ers",
}


def classify_college_tier(team: str) -> int:
    """10-tier conference classification. 1=SEC/OSU elite, 10=FCS lower.

    NFL franchise names → tier 1 (they came from somewhere major).
    """
    t = (team or "").lower().strip()

    for kw in NFL_FRANCHISE_KEYWORDS:
        if kw in t: return 1

    _T1 = {"alabama","ohio state","georgia","clemson","lsu","michigan"}
    _T2 = {"texas","oklahoma","florida","penn state","notre dame","florida state",
           "tennessee","texas a&m","usc","oregon","miami","auburn","washington"}
    _T3 = {"north carolina","virginia tech","pittsburgh","wisconsin","iowa",
           "michigan state","nebraska","oklahoma state","baylor","tcu","arkansas",
           "ole miss","mississippi state","south carolina","stanford","utah",
           "arizona state","colorado","georgia tech"}
    _T4 = {"west virginia","kansas state","iowa state","texas tech","kentucky",
           "vanderbilt","missouri","arizona","cal","oregon state","washington state",
           "indiana","purdue","illinois","minnesota","maryland","rutgers",
           "louisville","virginia","nc state","duke","wake forest","syracuse",
           "boston college","cincinnati","ucf"}
    _T5 = {"ucla","northwestern","navy","army","air force","liberty","byu",
           "western kentucky","louisiana tech"}
    _T6 = {"memphis","houston","smu","tulane","east carolina","south florida",
           "temple","connecticut","tulsa","rice","utep","uab"}
    _T7 = {"boise state","fresno state","hawaii","san diego state","wyoming",
           "utah state","nevada","colorado state","new mexico","san jose state"}
    _T8 = {"appalachian state","coastal carolina","marshall","utsa","troy",
           "louisiana","james madison","buffalo","kent state","ohio",
           "western michigan","central michigan","eastern michigan",
           "northern illinois","ball state","toledo"}
    _T9 = {"north dakota state","montana","south dakota state","furman",
           "villanova","richmond","delaware","sacramento state","central arkansas"}

    for kw in _T1:
        if kw in t: return 1
    for kw in _T2:
        if kw in t: return 2
    for kw in _T3:
        if kw in t: return 3
    for kw in _T4:
        if kw in t: return 4
    for kw in _T5:
        if kw in t: return 5
    for kw in _T6:
        if kw in t: return 6
    for kw in _T7:
        if kw in t: return 7
    for kw in _T8:
        if kw in t: return 8
    for kw in _T9:
        if kw in t: return 9
    return 10


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
    """Return is_award_winner and is_all_american flags from known lists."""
    n = (name or "").lower().strip()
    return {
        "is_award_winner": int(any(aw in n or n in aw for aw in _AWARD_WINNERS)),
        "is_all_american": int(any(aa in n or n in aa for aa in _ALL_AMERICANS)),
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


def compute_production_score(position: str, stats: dict) -> float:
    """Compute a 0–100 composite production score normalized per position."""
    p = (position or "").upper()
    games = max(float(stats.get("games_played", 1) or 1), 1)
    pass_td  = float(stats.get("passing_touchdowns", 0) or 0)
    pass_yds = float(stats.get("passing_yards", 0) or 0)
    rush_td  = float(stats.get("rushing_touchdowns", 0) or 0)
    rush_yds = float(stats.get("rushing_yards", 0) or 0)
    tackles  = float(stats.get("tackles", 0) or 0)
    sacks    = float(stats.get("sacks", 0) or 0)
    ints     = float(stats.get("interceptions", 0) or 0)
    pds      = float(stats.get("pass_deflections", 0) or 0)

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
    elif p in {"OL", "OT", "OG", "C"}:
        # OL: no counting stats — use games played as proxy for durability
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


@app.before_request
def enforce_canonical_origin():
    """Redirect production traffic to the configured canonical host and HTTPS."""
    if request.method == "OPTIONS":
        return None

    forwarded_host = request.headers.get("X-Forwarded-Host", request.host or "")
    host = forwarded_host.split(",")[0].strip().split(":")[0].lower()
    if not host or host in LOCAL_HOSTS:
        return None

    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme or "http")
    scheme = forwarded_proto.split(",")[0].strip().lower()

    target_host = CANONICAL_HOST or host
    target_scheme = "https" if FORCE_HTTPS else scheme

    if host == target_host and scheme == target_scheme:
        return None

    query = request.query_string.decode("utf-8") if request.query_string else ""
    destination = f"{target_scheme}://{target_host}{request.path}"
    if query:
        destination = f"{destination}?{query}"
    return redirect(destination, code=308)


position_model = None
label_encoders = None
success_model = None           # primary: CalibratedClassifierCV(XGBoost) or raw XGBoost
catboost_success_model = None  # secondary ensemble member
tabpfn_success_model = None    # tertiary ensemble member (optional)
draft_grade_model = None       # primary: CalibratedClassifierCV(XGBoost) or raw XGBoost
catboost_draft_grade_model = None

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


def load_player_lookup(csv_path: str = PLAYER_DATA_PATH) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Failed to read {csv_path}: {exc}")
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    possible_name_columns = ["displayName", "fullName", "shortName"]

    for _, row in df.iterrows():
        position = str(row.get("position") or "Unknown")
        team = str(row.get("team") or "Unknown")
        jersey = row.get("jersey")

        for col in possible_name_columns:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                key = normalize_name(value)
                if key and key not in lookup:
                    lookup[key] = {
                        "name": value.strip(),
                        "position": position,
                        "team": team,
                        "jersey": str(jersey) if jersey is not None else "0",
                    }

    print(f"Loaded {len(lookup)} player lookup entries from {csv_path}.")
    return lookup


PLAYER_LOOKUP = load_player_lookup()


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
        staged = []
        if PLAYER_LOOKUP:
            seen = set()
            for item in PLAYER_LOOKUP.values():
                name = item.get("name", "").strip()
                if not name or normalize_name(name) in seen:
                    continue
                seen.add(normalize_name(name))
                jersey_str = item.get("jersey", "0") if isinstance(item, dict) else "0"
                try:
                    jersey = int(jersey_str or 0)
                except ValueError:
                    jersey = 0
                staged.append((name, item.get("position", "Unknown"), item.get("team", "Unknown"), jersey, "csv_seed", None))
        else:
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
    elif p in {"OL", "OT", "OG", "C"}:
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

        speed = forty_to_speed_score(position, forty) if forty else 50.0
        vert_sc  = vertical_to_score(position, vertical)
        height_sc = height_to_score(position, height_in)
        weight_sc = weight_to_score(position, weight_lb)

        # Build display strings
        def fmt_height(h):
            if not h: return ""
            return f"{h // 12}'{h % 12}\""

        result = {
            "combine_speed_score":  round(speed, 1),
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


def fetch_player_data(player_name: str, fallback_position: str = "Unknown", fallback_team: str = "Unknown") -> Tuple[Optional[Dict[str, object]], str]:
    normalized = normalize_name(player_name)

    # 1) Resolve from local database first.
    db_player = get_player_by_exact_name(player_name)
    if db_player:
        source   = str(db_player.get("source", "db_lookup") or "db_lookup")
        name     = str(db_player.get("name", player_name)).strip()
        position = str(db_player.get("position", "Unknown") or "Unknown")
        team     = str(db_player.get("team", "Unknown") or "Unknown")
        jersey   = int(db_player.get("jersey", 0) or 0)
        espn_id  = str(db_player.get("espn_id") or "").strip()

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
            return profile, "espn_live"

        # No real ESPN stats — try combine measurables anyway
        result = generate_estimated_profile(name=name, position=position, team=team, jersey=jersey)
        if espn_id:
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
        return result, source

    # 2) Fallback to local CSV metadata.
    player_meta = PLAYER_LOOKUP.get(normalized)
    if player_meta:
        try:
            jersey = int(player_meta.get("jersey", "0") or 0)
        except ValueError:
            jersey = 0
        result = generate_estimated_profile(
            name=player_meta["name"],
            position=player_meta["position"],
            team=player_meta["team"],
            jersey=jersey,
        )
        upsert_player_record(
            result["name"],
            result["position"],
            result["team"],
            int(result["jersey"]),
            source="csv_fallback",
        )
        return result, "csv_fallback"

    # 3) Last fallback: generic baseline, using caller-supplied position/team if known.
    result = generate_estimated_profile(name=player_name.strip(), position=fallback_position, team=fallback_team)
    upsert_player_record(
        result["name"],
        result["position"],
        result["team"],
        int(result["jersey"]),
        source="default_baseline",
    )
    return result, "default_baseline"


_DB_POSITIONS = {"CB", "S", "DB", "FS", "SS"}
_LB_POSITIONS = {"LB", "ILB", "OLB", "MLB"}
_DL_POSITIONS = {"DL", "DE", "DT", "NT", "EDGE"}
_OL_POSITIONS = {"OL", "OT", "OG", "C", "LS"}
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


def build_success_features(player_stats: Dict[str, object]) -> pd.DataFrame:
    """Build feature vector aligned with SUCCESS_FEATURES / _BASE_FEATURES.
    No draft_round — that is an output, not an input."""
    name     = str(player_stats.get("name", "") or "")
    position = str(player_stats.get("position", "Unknown"))
    team     = str(player_stats.get("team", "") or "")
    flags    = position_flags(position)
    accolades = detect_accolades(name)

    stats_dict = {
        "games_played":       float(player_stats.get("games_played", 0) or 0),
        "passing_touchdowns": float(player_stats.get("passing_touchdowns", 0) or 0),
        "passing_yards":      float(player_stats.get("passing_yards", 0) or 0),
        "rushing_touchdowns": float(player_stats.get("rushing_touchdowns", 0) or 0),
        "rushing_yards":      float(player_stats.get("rushing_yards", 0) or 0),
        "tackles":            float(player_stats.get("tackles", 0) or 0),
        "sacks":              float(player_stats.get("sacks", 0) or 0),
        "interceptions":      float(player_stats.get("interceptions", 0) or 0),
        "pass_deflections":   float(player_stats.get("pass_deflections", 0) or 0),
    }

    production_score = float(
        player_stats.get("production_score")
        or compute_production_score(position, stats_dict)
    )
    combine_speed = float(player_stats.get("combine_speed_score") or 50.0)
    conference_tier = float(
        player_stats.get("conference_tier")
        or classify_college_tier(team)
    )
    is_award_winner = int(player_stats.get("is_award_winner") or accolades["is_award_winner"])
    is_all_american = int(player_stats.get("is_all_american") or accolades["is_all_american"])

    row = {
        "production_score":    production_score,
        "games_played":        stats_dict["games_played"],
        "combine_speed_score": combine_speed,
        "conference_tier":     conference_tier,
        "is_award_winner":     is_award_winner,
        "is_all_american":     is_all_american,
        **flags,
    }
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


def synthetic_player_sample(position: str) -> Dict[str, float]:
    base = {
        "games_played": random.randint(4, 17),
        "passing_touchdowns": 0, "passing_yards": 0,
        "rushing_touchdowns": 0, "rushing_yards": 0,
        "tackles": 0, "sacks": 0.0, "interceptions": 0, "pass_deflections": 0,
    }
    if position == "QB":
        base.update({
            "passing_touchdowns": max(0, int(random.gauss(18, 10))),
            "passing_yards":      max(0, int(random.gauss(2800, 1200))),
            "rushing_touchdowns": max(0, int(random.gauss(2, 2))),
            "rushing_yards":      max(0, int(random.gauss(180, 150))),
        })
    elif position == "RB":
        base.update({
            "rushing_touchdowns": max(0, int(random.gauss(7, 4))),
            "rushing_yards":      max(0, int(random.gauss(760, 380))),
        })
    elif position in {"WR", "TE"}:
        base.update({
            "passing_touchdowns": max(0, int(random.gauss(1, 2))),
            "passing_yards":      max(0, int(random.gauss(120, 260))),
            "rushing_touchdowns": max(0, int(random.gauss(5, 4))),
            "rushing_yards":      max(0, int(random.gauss(820, 420))),
        })
    elif position == "LB":
        base.update({
            "tackles":          max(0, int(random.gauss(75, 28))),
            "sacks":            max(0.0, round(random.gauss(4.0, 3.0), 1)),
            "interceptions":    max(0, int(random.gauss(1, 1))),
            "pass_deflections": max(0, int(random.gauss(4, 3))),
        })
    elif position in {"CB", "S"}:
        base.update({
            "tackles":          max(0, int(random.gauss(55, 20))),
            "sacks":            max(0.0, round(random.gauss(0.5, 0.8), 1)),
            "interceptions":    max(0, int(random.gauss(3, 2))),
            "pass_deflections": max(0, int(random.gauss(9, 5))),
        })
    elif position == "DL":
        base.update({
            "tackles": max(0, int(random.gauss(45, 18))),
            "sacks":   max(0.0, round(random.gauss(6.5, 4.0), 1)),
            "pass_deflections": max(0, int(random.gauss(2, 2))),
        })
    elif position == "OL":
        base.update({"games_played": random.randint(6, 17)})
    else:  # OTHER / K / P
        base.update({
            "passing_touchdowns": max(0, int(random.gauss(1, 1))),
            "passing_yards":      max(0, int(random.gauss(90, 120))),
            "rushing_touchdowns": max(0, int(random.gauss(3, 3))),
            "rushing_yards":      max(0, int(random.gauss(420, 290))),
        })
    return base


def realistic_nfl_success_probability(
    draft_round: int,
    combine_speed: float,
    college_tier: int,
    production: float,
) -> float:
    """
    Estimate NFL success probability using factors that actually predict outcomes.

    Draft round is the dominant signal (scouts already aggregate all information
    into where a player is picked).  Combine athleticism, college competition
    level, and production adjust the probability modestly.

    Approximate empirical hit-rates by round (career starter or better):
      Round 1 → ~68%   Round 2 → ~52%   Round 3 → ~38%
      Round 4 → ~26%   Round 5 → ~18%   Round 6 → ~12%
      Round 7 → ~8%    Undrafted → ~4%
    """
    ROUND_BASE = {1: 0.68, 2: 0.52, 3: 0.38, 4: 0.26,
                  5: 0.18, 6: 0.12, 7: 0.08, 8: 0.04}
    base = ROUND_BASE.get(int(draft_round), 0.04)

    # Combine speed: 0–100 scale; 50 = average. Adjust ±8 pp.
    speed_adj = (combine_speed - 50.0) / 50.0 * 0.08

    # College tier: P5 (1) > G5 (2) > FCS (3). Adjust ±5 pp.
    tier_adj = (2.0 - college_tier) / 2.0 * 0.05  # +5 for P5, 0 for G5, −5 for FCS

    # Production: 0–100 composite. Adjust ±7 pp.
    prod_adj = (production - 50.0) / 100.0 * 0.14

    prob = base + speed_adj + tier_adj + prod_adj
    return float(min(max(prob, 0.02), 0.97))


# ── Real historical players — ground truth for both models ────────────────────
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


def _success_prob_from_college_profile(
    production_score: float,
    conference_tier: int,
    combine_speed: float,
    is_award_winner: int,
    is_all_american: int,
) -> float:
    """Estimate P(NFL success) purely from college profile — NO draft round."""
    prod_factor  = production_score / 100.0
    tier_factor  = max(0.0, (11.0 - conference_tier) / 10.0)  # 1.0=T1 elite, 0.1=T10 FCS
    speed_factor = combine_speed / 100.0
    award_boost  = 0.08 if is_award_winner else 0.0
    aa_boost     = 0.05 if is_all_american else 0.0
    base = prod_factor * 0.45 + tier_factor * 0.30 + speed_factor * 0.25
    return float(min(0.92, max(0.03, base + award_boost + aa_boost)))


def _draft_grade_from_profile(
    production_score: float,
    conference_tier: int,
    combine_speed: float,
    is_award_winner: int,
    is_all_american: int,
) -> int:
    """Deterministic draft grade class from college profile (used for synthetic labels)."""
    score = (
        production_score * 0.40
        + max(0.0, (11.0 - conference_tier) / 10.0) * 100 * 0.30
        + combine_speed * 0.20
        + (is_award_winner * 8 + is_all_american * 5)
    )
    if score >= 78: return 0  # Top 50 (R1-2)
    if score >= 60: return 1  # Day 2 (R3-4)
    if score >= 42: return 2  # Late Round (R5-7)
    return 3                   # UDFA


def _build_training_rows(samples: int = 4000) -> Tuple[pd.DataFrame, "pd.Series"]:
    """
    Build training data for both models.

    Priority:
      1. Real ESPN data from collect_training_data.py (combine_outcomes.csv) if available
      2. SEED_TRAINING_PLAYERS (real ground-truth, weighted 5x)
      3. Synthetic samples to pad to `samples` total

    Returns (X, y_success, y_draft_grade) but callers pick which y they need.
    """
    random.seed(42)
    positions = ["QB", "RB", "WR", "TE", "LB", "CB", "S", "DL", "OL", "OTHER"]
    pos_w     = [0.15, 0.14, 0.20, 0.08, 0.10, 0.10, 0.08, 0.08, 0.04, 0.03]

    rows_s, rows_d = [], []   # success rows, draft-grade rows (same features, different labels)
    labels_s, labels_d = [], []

    # ── 1. Real ESPN data ───────────────────────────────────────────────────
    real_count = 0
    if os.path.exists(TRAINING_DATA_PATH):
        try:
            df_real = pd.read_csv(TRAINING_DATA_PATH)
            for _, r in df_real.iterrows():
                pos   = str(r.get("position") or "OTH").upper()
                flags = position_flags(pos)
                speed = float(r.get("combine_speed_score") or 50)
                tier  = float(r.get("conference_tier") or 5)
                # production_score not in ESPN CSV → estimate from draft_grade
                dg    = int(r.get("draft_grade", 2) or 2)
                prod  = max(0.0, min(100.0, 80.0 - dg * 12.0 + random.gauss(0, 8)))
                games = float(r.get("games_played", 13) or 13)
                ns    = int(r.get("nfl_success", 0) or 0)
                row = {
                    "production_score":    round(prod, 1),
                    "games_played":        games,
                    "combine_speed_score": round(speed, 1),
                    "conference_tier":     tier,
                    "is_award_winner":     0,
                    "is_all_american":     0,
                    **flags,
                }
                rows_s.append(row); labels_s.append(ns)
                rows_d.append(row); labels_d.append(min(3, max(0, dg)))
                real_count += 1
            print(f"  Loaded {real_count} real ESPN training rows from {TRAINING_DATA_PATH}")
        except Exception as exc:
            print(f"  Real data load failed ({exc}) — using synthetic only")

    # ── 2. Seed players (always included, 5× weight) ────────────────────────
    for sp in SEED_TRAINING_PLAYERS * 5:
        flags = position_flags(sp["position"])
        row = {
            "production_score":    float(sp["production_score"]),
            "games_played":        float(sp["games_played"]),
            "combine_speed_score": float(sp["combine_speed_score"]),
            "conference_tier":     float(sp["conference_tier"]),
            "is_award_winner":     int(sp["is_award_winner"]),
            "is_all_american":     int(sp["is_all_american"]),
            **flags,
        }
        rows_s.append(row); labels_s.append(sp["nfl_success"])
        rows_d.append(row); labels_d.append(sp["draft_grade"])

    # ── 3. Synthetic samples ─────────────────────────────────────────────────
    # Higher label noise (0.12) so the model learns robust boundaries, not
    # a thin approximation of our heuristic function.
    needed = max(0, samples - len(rows_s))
    for _ in range(needed):
        position    = random.choices(positions, weights=pos_w, k=1)[0]
        tier        = random.choices(range(1, 11), weights=[14,12,11,10,9,9,8,8,7,12], k=1)[0]
        speed       = float(max(0, min(100, random.gauss(62, 18))))
        production  = float(max(0, min(100, random.gauss(65, 22))))
        is_award    = int(random.random() < 0.04)
        is_aa       = int(random.random() < 0.08)
        games       = random.randint(8, 16)

        prob_s = _success_prob_from_college_profile(production, tier, speed, is_award, is_aa)
        noisy_s = min(max(prob_s + random.gauss(0, 0.12), 0.01), 0.99)
        success = 1 if random.random() < noisy_s else 0

        grade_d = _draft_grade_from_profile(production, tier, speed, is_award, is_aa)
        noisy_d = min(3, max(0, grade_d + random.choices([-1, 0, 0, 0, 1], k=1)[0]))

        flags = position_flags(position)
        row = {
            "production_score":    round(production, 1),
            "games_played":        float(games),
            "combine_speed_score": round(speed, 1),
            "conference_tier":     float(tier),
            "is_award_winner":     is_award,
            "is_all_american":     is_aa,
            **flags,
        }
        rows_s.append(row); labels_s.append(success)
        rows_d.append(row); labels_d.append(noisy_d)

    X_s = pd.DataFrame(rows_s, columns=SUCCESS_FEATURES)
    X_d = pd.DataFrame(rows_d, columns=DRAFT_GRADE_FEATURES)
    return X_s, pd.Series(labels_s), X_d, pd.Series(labels_d)


def train_success_model_from_synthetic(samples: int = 4000):
    """
    Train the full success model ensemble:
      1. XGBoost  — calibrated with Platt scaling (saved as .pkl)
      2. CatBoost — if available (saved as .cbm)
      3. TabPFN   — if available (saved as .pkl)

    Returns the primary calibrated XGBoost model.
    """
    global catboost_success_model, tabpfn_success_model

    print("Building training data…")
    X, y, _, _ = _build_training_rows(samples)

    # Class balance ratio for scale_pos_weight
    n_neg = int((y == 0).sum()); n_pos = int((y == 1).sum())
    spw = round(n_neg / max(n_pos, 1), 2)
    print(f"  Success labels: {n_pos} positive / {n_neg} negative  (scale_pos_weight={spw})")

    # Train / calibration split — stratified so calibration set has both classes
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # ── 1. XGBoost + Platt calibration ──────────────────────────────────────
    xgb_base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,              # reduced from 5 to limit overfitting on small data
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=spw,     # fix class imbalance
        random_state=42,
        eval_metric="logloss",
    )
    xgb_base.fit(X_train, y_train)
    xgb_base.save_model(SUCCESS_MODEL_PATH)  # keep raw for legacy loading

    calibrated = CalibratedClassifierCV(xgb_base, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    joblib.dump(calibrated, SUCCESS_CALIBRATED_PATH)
    print(f"  Saved calibrated XGBoost → {SUCCESS_CALIBRATED_PATH}")

    # ── 2. CatBoost ─────────────────────────────────────────────────────────
    if CATBOOST_AVAILABLE:
        try:
            cb = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                eval_metric="AUC", random_seed=42, verbose=0,
                class_weights={0: 1.0, 1: float(spw)},
                loss_function="Logloss",
            )
            cb.fit(X_train, y_train)
            cb.save_model(CATBOOST_SUCCESS_PATH)
            catboost_success_model = cb
            print(f"  Saved CatBoost success model → {CATBOOST_SUCCESS_PATH}")
        except Exception as exc:
            print(f"  CatBoost training failed: {exc}")

    # ── 3. TabPFN ────────────────────────────────────────────────────────────
    if TABPFN_AVAILABLE:
        try:
            # TabPFN works best with ≤10k samples; subsample if larger
            max_tabpfn = 3000
            if len(X_train) > max_tabpfn:
                idx = np.random.choice(len(X_train), max_tabpfn, replace=False)
                X_tf = X_train.iloc[idx]; y_tf = y_train.iloc[idx]
            else:
                X_tf = X_train; y_tf = y_train
            tfpn = TabPFNClassifier(device="cpu", n_estimators=16)
            tfpn.fit(X_tf.values, y_tf.values)
            joblib.dump(tfpn, TABPFN_SUCCESS_PATH)
            tabpfn_success_model = tfpn
            print(f"  Saved TabPFN success model → {TABPFN_SUCCESS_PATH}")
        except Exception as exc:
            print(f"  TabPFN training skipped: {exc}")

    return calibrated


def train_draft_grade_model():
    """
    Train the draft grade ensemble:
      1. XGBoost multiclass — calibrated (saved as .pkl)
      2. CatBoost multiclass — if available (saved as .cbm)

    Returns the primary calibrated XGBoost model.
    """
    global catboost_draft_grade_model

    print("Building draft grade training data…")
    _, _, X, y = _build_training_rows(4000)

    print(f"  Draft grade distribution: {dict(y.value_counts().sort_index())}")

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # ── 1. XGBoost multiclass + calibration ─────────────────────────────────
    xgb_base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        objective="multi:softprob",
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_base.fit(X_train, y_train)
    xgb_base.save_model(DRAFT_GRADE_MODEL_PATH)

    calibrated = CalibratedClassifierCV(xgb_base, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    joblib.dump(calibrated, DRAFT_GRADE_CALIBRATED_PATH)
    print(f"  Saved calibrated XGBoost draft grade → {DRAFT_GRADE_CALIBRATED_PATH}")

    # ── 2. CatBoost multiclass ───────────────────────────────────────────────
    if CATBOOST_AVAILABLE:
        try:
            cb = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                random_seed=42, verbose=0,
                loss_function="MultiClass",
                classes_count=4,
            )
            cb.fit(X_train, y_train)
            cb.save_model(CATBOOST_DRAFT_GRADE_PATH)
            catboost_draft_grade_model = cb
            print(f"  Saved CatBoost draft grade model → {CATBOOST_DRAFT_GRADE_PATH}")
        except Exception as exc:
            print(f"  CatBoost draft grade training failed: {exc}")

    return calibrated


def load_or_train_draft_grade_model() -> None:
    global draft_grade_model, catboost_draft_grade_model

    # Load calibrated pkl (preferred) or fall back to raw JSON
    if os.path.exists(DRAFT_GRADE_CALIBRATED_PATH):
        draft_grade_model = joblib.load(DRAFT_GRADE_CALIBRATED_PATH)
        print("Draft grade model loaded (calibrated).")
    elif os.path.exists(DRAFT_GRADE_MODEL_PATH):
        m = xgb.XGBClassifier()
        m.load_model(DRAFT_GRADE_MODEL_PATH)
        draft_grade_model = m
        print("Draft grade model loaded (raw XGBoost).")
    else:
        print("Draft grade model not found. Training ensemble…")
        draft_grade_model = train_draft_grade_model()
        return

    # Load CatBoost companion if available
    if CATBOOST_AVAILABLE and os.path.exists(CATBOOST_DRAFT_GRADE_PATH):
        try:
            cb = CatBoostClassifier()
            cb.load_model(CATBOOST_DRAFT_GRADE_PATH)
            catboost_draft_grade_model = cb
            print("CatBoost draft grade model loaded.")
        except Exception as exc:
            print(f"CatBoost draft grade load failed: {exc}")


def predict_draft_grade(player_stats: Dict[str, object]) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """Ensemble draft grade: average softmax probabilities across all available models."""
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

    avg_proba   = np.mean(proba_arrays, axis=0)
    grade_class = int(avg_proba.argmax())
    label       = DRAFT_GRADE_LABELS[grade_class]
    return label, grade_class, round(float(avg_proba[grade_class]) * 100.0, 1)



def load_or_train_success_model() -> None:
    global success_model, catboost_success_model, tabpfn_success_model

    # Load calibrated pkl (preferred) or fall back to raw XGBoost JSON
    if os.path.exists(SUCCESS_CALIBRATED_PATH):
        success_model = joblib.load(SUCCESS_CALIBRATED_PATH)
        print("Success model loaded (calibrated XGBoost).")
    elif os.path.exists(SUCCESS_MODEL_PATH):
        m = xgb.XGBClassifier()
        m.load_model(SUCCESS_MODEL_PATH)
        success_model = m
        print("Success model loaded (raw XGBoost — re-train recommended).")
    else:
        print("Success model not found. Training ensemble…")
        success_model = train_success_model_from_synthetic()
        return

    # Load CatBoost companion
    if CATBOOST_AVAILABLE and os.path.exists(CATBOOST_SUCCESS_PATH):
        try:
            cb = CatBoostClassifier()
            cb.load_model(CATBOOST_SUCCESS_PATH)
            catboost_success_model = cb
            print("CatBoost success model loaded.")
        except Exception as exc:
            print(f"CatBoost success load failed: {exc}")

    # Load TabPFN companion
    if TABPFN_AVAILABLE and os.path.exists(TABPFN_SUCCESS_PATH):
        try:
            tabpfn_success_model = joblib.load(TABPFN_SUCCESS_PATH)
            print("TabPFN success model loaded.")
        except Exception as exc:
            print(f"TabPFN success load failed: {exc}")


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
    "is_award_winner":      "Award Winner",
    "is_all_american":      "All-American",
    "position_qb":          "Position: QB",
    "position_rb":          "Position: RB",
    "position_wr":          "Position: WR",
    "position_te":          "Position: TE",
    "position_db":          "Position: DB",
    "position_lb":          "Position: LB",
    "position_dl":          "Position: DL",
    "position_ol":          "Position: OL",
    "position_other":       "Position: Other",
}


def top_feature_importances(n: int = 4) -> list:
    """Return the top-n features by XGBoost gain importance."""
    if success_model is None:
        return []
    try:
        importances = success_model.get_booster().get_score(importance_type="gain")
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


def predict_success_with_model(player_stats: Dict[str, object]) -> Tuple[Optional[str], Optional[float], Optional[float], bool]:
    """Ensemble prediction: average predict_proba from all available models."""
    model_input = build_success_features(player_stats)
    X_arr = model_input.values  # numpy array for CatBoost / TabPFN

    probas = []

    # 1. Primary: calibrated XGBoost (or raw fallback)
    if success_model is not None:
        try:
            probas.append(float(success_model.predict_proba(model_input)[0][1]))
        except Exception as exc:
            print(f"XGBoost success inference failed: {exc}")

    # 2. CatBoost
    if catboost_success_model is not None:
        try:
            probas.append(float(catboost_success_model.predict_proba(X_arr)[0][1]))
        except Exception as exc:
            print(f"CatBoost success inference failed: {exc}")

    # 3. TabPFN
    if tabpfn_success_model is not None:
        try:
            probas.append(float(tabpfn_success_model.predict_proba(X_arr)[0][1]))
        except Exception as exc:
            print(f"TabPFN success inference failed: {exc}")

    if not probas:
        return None, None, None, False

    probability = sum(probas) / len(probas)
    label = "Success" if probability >= 0.5 else "No Success"
    confidence = round((probability if label == "Success" else 1.0 - probability) * 100.0, 1)
    models_used = len(probas)
    _ = models_used  # available for logging if needed
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

# ── Mock draft storage ─────────────────────────────────────────────────────────
_MOCK_DRAFT_DATA: dict = {"picks": [], "title": "", "generated_at": None, "total": 0}

# ── High school prospect cache ─────────────────────────────────────────────────
_HS_PROSPECT_CACHE: list = []
_HS_PROSPECT_CACHE_META: dict = {}

_POS_GROUPS = {
    "DB": {"CB", "S", "DB", "FS", "SS"},
    "LB": {"LB", "ILB", "OLB", "MLB"},
    "DL": {"DL", "DE", "DT", "EDGE", "NT"},
    "OL": {"OL", "OT", "OG", "C", "LS"},
}
_GRADE_ORDER = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C+": 6, "C": 7, "C-": 8, "D": 9}


def load_prospect_cache() -> None:
    global _PROSPECT_CACHE, _PROSPECT_CACHE_META
    if not os.path.exists(PROSPECT_CACHE_PATH):
        print("Prospect cache not found (run build_prospect_cache.py to create it).")
        return
    try:
        with open(PROSPECT_CACHE_PATH) as f:
            data = json.load(f)
        _PROSPECT_CACHE = data.get("prospects", [])
        _PROSPECT_CACHE_META = {
            "generated_at": data.get("generated_at"),
            "total":        data.get("total", len(_PROSPECT_CACHE)),
        }
        print(f"Loaded {len(_PROSPECT_CACHE)} prospects from cache.")
    except Exception as exc:
        print(f"Failed to load prospect cache: {exc}")


@app.get("/api/prospects")
def api_prospects():
    position     = (request.args.get("position") or "").strip().upper()
    grade_filter = (request.args.get("grade") or "").strip().upper()
    query        = (request.args.get("q") or "").strip().lower()
    team_filter  = (request.args.get("team") or "").strip().lower()
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

    return jsonify({
        "total":     total,
        "offset":    offset,
        "limit":     limit,
        "meta":      _PROSPECT_CACHE_META,
        "prospects": paginated,
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
    import io, csv as csv_mod
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    csv_content = str(payload.get("csv_content") or "").strip()
    title       = str(payload.get("title") or "JKrek's Mock Draft").strip()
    if not csv_content:
        return jsonify({"error": "csv_content is required"}), 400

    picks = []
    try:
        reader = csv_mod.DictReader(io.StringIO(csv_content))
        for row in reader:
            # Normalize keys to lowercase for flexible column matching
            keys = {k.lower().strip().replace(" ", "_"): str(v).strip() for k, v in (row or {}).items()}

            def _find(*candidates):
                for c in candidates:
                    v = keys.get(c, "")
                    if v and v.lower() not in ("", "none", "n/a", "-"):
                        return v
                return ""

            player = _find("player", "name", "player_name", "athlete")
            if not player:
                continue
            pick_raw = _find("pick", "overall", "overall_pick", "#", "pick_#")
            try:
                pick_num = int(float(pick_raw)) if pick_raw else len(picks) + 1
            except ValueError:
                pick_num = len(picks) + 1

            nfl_team = _find("team", "nfl_team", "franchise", "club")
            position = _find("position", "pos").upper()
            school   = _find("school", "college", "university")
            grade    = _find("pff_grade", "pff", "grade", "rating", "score")
            round_n  = _find("round", "rd", "rnd")

            picks.append({
                "pick":     pick_num,
                "round":    round_n,
                "nfl_team": nfl_team,
                "player":   player,
                "position": position,
                "school":   school,
                "pff_grade": grade,
                "color":    _team_color(nfl_team),
            })
    except Exception as exc:
        return jsonify({"error": f"CSV parse error: {exc}"}), 400

    if not picks:
        return jsonify({"error": "No valid picks found in CSV"}), 400

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
    global _HS_PROSPECT_CACHE, _HS_PROSPECT_CACHE_META
    if not os.path.exists(HS_PROSPECT_CACHE_PATH):
        print("HS prospect cache not found (run build_hs_prospect_cache.py to create it).")
        return
    try:
        with open(HS_PROSPECT_CACHE_PATH) as f:
            data = json.load(f)
        _HS_PROSPECT_CACHE = data.get("prospects", [])
        _HS_PROSPECT_CACHE_META = {
            "generated_at": data.get("generated_at"),
            "total":        data.get("total", len(_HS_PROSPECT_CACHE)),
            "years":        data.get("years", []),
        }
        print(f"Loaded {len(_HS_PROSPECT_CACHE)} HS prospects from cache.")
    except Exception as exc:
        print(f"Failed to load HS prospect cache: {exc}")


@app.get("/api/hs-prospects")
def api_hs_prospects():
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

    return jsonify({
        "total":         len(results),
        "offset":        offset,
        "limit":         limit,
        "meta":          _HS_PROSPECT_CACHE_META,
        "api_key_set":   bool(CFBD_API_KEY),
        "prospects":     results[offset: offset + limit],
    })


# ── Prospect grade ────────────────────────────────────────────────────────────

def compute_prospect_grade(success_prob: Optional[float], draft_grade_class: Optional[int]) -> str:
    """Map (success_probability %, draft_grade_class) → letter grade A+…D."""
    p = float(success_prob or 0)
    d = int(draft_grade_class) if draft_grade_class is not None else 3
    if p >= 88 and d == 0: return "A+"
    if p >= 80 and d <= 1: return "A"
    if p >= 72 and d <= 1: return "A-"
    if p >= 64 and d <= 2: return "B+"
    if p >= 54 and d <= 2: return "B"
    if p >= 44:            return "B-"
    if p >= 34:            return "C+"
    if p >= 24:            return "C"
    if p >= 15:            return "C-"
    return "D"


# ── Named historical players for similarity comps ─────────────────────────────
# Stored by position group so matching only compares same-position players
_POS_GROUP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "CB": "DB", "S": "DB", "DB": "DB", "FS": "DB", "SS": "DB",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DL": "DL", "DE": "DL", "DT": "DL", "EDGE": "DL",
    "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL",
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
]


def find_historical_comps(player_stats: Dict[str, object], n: int = 3) -> list:
    """Return top-n most similar historical players by feature distance."""
    pos = (str(player_stats.get("position", "") or "")).upper()
    player_group = _POS_GROUP.get(pos, pos)

    prod  = float(player_stats.get("production_score") or 0)
    speed = float(player_stats.get("combine_speed_score") or 50)
    tier  = float(player_stats.get("conference_tier") or 5)
    award = int(player_stats.get("is_award_winner") or 0)
    aa    = int(player_stats.get("is_all_american") or 0)
    tier_norm = (11.0 - tier) / 10.0 * 100  # invert so higher = better

    scored = []
    for comp in NAMED_HISTORICAL_COMPS:
        comp_group = _POS_GROUP.get(comp["position"].upper(), comp["position"].upper())
        if comp_group != player_group:
            continue
        cp = float(comp["production_score"])
        cs = float(comp["combine_speed_score"])
        ct = (11.0 - float(comp["conference_tier"])) / 10.0 * 100
        ca = int(comp["is_award_winner"])
        caa = int(comp["is_all_american"])
        dist = (
            ((prod - cp) / 100) ** 2 * 2.0
            + ((speed - cs) / 100) ** 2 * 1.5
            + ((tier_norm - ct) / 100) ** 2 * 1.0
            + (award - ca) ** 2 * 0.3
            + (aa - caa) ** 2 * 0.2
        ) ** 0.5
        similarity = max(0, round(100 - dist * 100))
        scored.append({
            "name":       comp["name"],
            "position":   comp["position"],
            "similarity": similarity,
            "outcome":    comp["outcome"],
            "nfl_success": comp["nfl_success"],
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:n]


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
def init_data():
    cached = cache_get("init")
    if cached:
        return jsonify(cached)

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
            "player_lookup_size": len(PLAYER_LOOKUP),
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
    """Fast autocomplete search across ALL player sources."""
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"players": []})
    cached_key = f"search:{q.lower()}"
    cached = cache_get(cached_key)
    if cached:
        return jsonify(cached)
    ph = _placeholder()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT name, position, team, source FROM players "
        f"WHERE name LIKE {ph} ORDER BY "
        "CASE source WHEN 'college_prospect' THEN 0 WHEN 'nfl_seed' THEN 1 WHEN 'legacy' THEN 2 ELSE 3 END, "
        "name ASC LIMIT 10",
        (f"%{q}%",),
    )
    players = _rows_as_dicts(cursor)
    conn.close()
    payload = {"players": players}
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

    player_data, data_source = fetch_player_data(player_name, fallback_position, fallback_team)
    if not player_data:
        return jsonify({"error": "Unable to resolve player data."}), 404

    # Use the position already resolved from ESPN/DB; the legacy position model
    # was trained on old NFL data and reliably returns None for current players.
    _raw_pos = str(player_data.get("position") or "").strip().upper()
    predicted_position = _raw_pos if _raw_pos and _raw_pos not in {"UNKNOWN", "UNK", ""} else (
        predict_position_with_model(player_data) or "Unknown"
    )

    success_label, confidence, success_probability, model_used = predict_success_with_model(player_data)

    if not success_label:
        success_label, confidence, reasoning = determine_success_fallback(player_data, predicted_position)
    else:
        reasoning = "XGBoost prediction from college production, athleticism, conference tier, and accolades."

    # Draft grade + prospect grade
    draft_grade_label_str, draft_grade_class, draft_grade_prob = predict_draft_grade(player_data)
    prospect_grade = compute_prospect_grade(success_probability, draft_grade_class)

    position = str(player_data.get("position", "Unknown"))
    draft_round = int(player_data.get("draft_round") or 8)
    combine_speed = float(player_data.get("combine_speed_score") or 50.0)
    conference_tier = int(player_data.get("conference_tier") or classify_college_tier(
        str(player_data.get("team", "") or "")))
    production = float(
        player_data.get("production_score")
        or compute_production_score(position, player_data)
    )

    round_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
                    5: "5th", 6: "6th", 7: "7th", 8: "Undrafted"}

    stored_team = str(player_data.get("team", "") or "")
    is_nfl_player = any(kw in stored_team.lower() for kw in NFL_FRANCHISE_KEYWORDS)
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

    return jsonify(
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
            "top_factors":         top_feature_importances(4),
        }
    )


initialize_player_database()
load_position_model_artifacts()
load_or_train_success_model()
load_or_train_draft_grade_model()
load_prospect_cache()
load_mock_draft()
load_hs_prospect_cache()

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
    app.run(debug=False, use_reloader=False, port=5001)
