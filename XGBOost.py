#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Future Star Predictor backend.

- Provides health and prediction endpoints.
- Uses a direct ML success classifier for Success/No Success output.
- Falls back to rule scoring only if model inference is unavailable.
"""

import os
import random
import sqlite3
import time
import hashlib
import threading
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
import xgboost as xgb
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

POSITION_MODEL_PATH = "nfl_xgboost_model.json"
ENCODER_PATH = "label_encoders.pkl"
SUCCESS_MODEL_PATH = "success_xgboost_model.json"
PLAYER_DATA_PATH = "nfl_players.csv"
PLAYER_DB_PATH = "players.db"
ESPN_CFB_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
ESPN_CFB_TEAM_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"
ESPN_CFB_ATHLETE_OVERVIEW_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{espn_id}/overview"
HTTP_TIMEOUT_SECONDS = 12
STATS_CACHE_TTL = 3600  # cache real stats for 1 hour

SUCCESS_FEATURES = [
    "games_played",
    # Offensive stats
    "passing_touchdowns",
    "passing_yards",
    "rushing_touchdowns",
    "rushing_yards",
    # Defensive stats
    "tackles",
    "sacks",
    "interceptions",
    "pass_deflections",
    # Meta features
    "draft_round",          # 1–7 = draft rounds; 8 = undrafted (biggest real-world predictor)
    "combine_speed_score",  # 0–100 position-normalized athleticism proxy
    "college_tier",         # 1 = Power 5, 2 = Group of 5, 3 = FCS/other
    "production_score",     # composite normalized production (0–100)
    # Position flags
    "position_qb",
    "position_rb",
    "position_wr",
    "position_te",
    "position_db",    # CB, S, FS, SS
    "position_lb",    # LB, ILB, OLB, MLB
    "position_dl",    # DL, DE, DT, EDGE
    "position_ol",    # OL, OT, OG, C
    "position_other",
]

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
    """Return 1 (P5), 2 (G5), or 3 (FCS) based on school name.

    If the team string is an NFL franchise, return 1 (P5) by default —
    active pros almost certainly played at major-program level.
    """
    normalized = (team or "").lower().strip()

    # Active NFL player stored with franchise name → treat as P5
    for franchise in NFL_FRANCHISE_KEYWORDS:
        if franchise in normalized:
            return 1

    for school in POWER5_SCHOOLS:
        if school in normalized:
            return 1

    # G5 conference keywords
    g5_keywords = ["app state", "boise state", "memphis", "uab", "marshall",
                   "army", "navy", "air force", "hawaii", "san jose", "fresno",
                   "utsa", "middle tennessee", "troy", "louisiana", "western",
                   "central", "eastern", "northern", "southern", "sam houston"]
    for kw in g5_keywords:
        if kw in normalized:
            return 2
    return 3


def combine_speed_for_position(position: str, seed: int) -> float:
    """Generate a 0–100 combine speed score (100 = elite) seeded deterministically."""
    p = (position or "").upper()
    # Mean and std of 40-yard dash by position; lower time = faster = higher score
    # We invert so higher number = better athlete
    ranges = {
        "QB":  (55, 20),
        "RB":  (60, 20),
        "WR":  (62, 18),
        "TE":  (50, 20),
    }
    mean, std_dev = ranges.get(p, (50, 20))
    # Use seed for determinism; add position-based offset
    offset = sum(ord(c) for c in p) if p else 0
    raw = mean + ((seed + offset) % 41) - 20  # [-20, +20] around mean
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
success_model = None

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

    # Derive the new features deterministically from the same seed
    college_tier = classify_college_tier(team)
    combine_speed = combine_speed_for_position(p, seed)
    production_raw = compute_production_score(p, {
        "games_played": games, "passing_touchdowns": passing_touchdowns,
        "passing_yards": passing_yards, "rushing_touchdowns": rushing_touchdowns,
        "rushing_yards": rushing_yards, "tackles": tackles, "sacks": sacks,
        "interceptions": interceptions, "pass_deflections": pass_deflections,
    })
    composite = (production_raw * 0.6 + combine_speed * 0.4) / 100.0
    raw_round = 8 - int(composite * 7)
    draft_round = max(1, min(8, raw_round))

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
        "college_tier": college_tier,
        "production_score": round(production_raw, 1),
    }


def _parse_int(val: str) -> int:
    """Parse ESPN stat string like '3,163' → 3163."""
    try:
        return int(str(val).replace(",", "").replace("--", "0").strip() or 0)
    except ValueError:
        return 0


ESPN_CORE_ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes/{espn_id}"


def _espn_resolve_athlete_info(espn_id: str) -> Dict[str, str]:
    """Resolve team name and position from the ESPN core athlete endpoint.
    Returns {"team": "...", "position": "..."} or empty strings on failure.
    """
    cache_key = f"espn_athlete:{espn_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    result = {"team": "", "position": ""}
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
        # Also try inline
        if not result["position"]:
            pos_inline = ath.get("position") or {}
            result["position"] = str(pos_inline.get("abbreviation") or pos_inline.get("name") or "")

        # Team
        team_ref = (ath.get("team") or {}).get("$ref", "")
        if team_ref:
            rt = requests.get(team_ref, timeout=5)
            if rt.ok:
                result["team"] = rt.json().get("displayName", "")

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


def fetch_player_data(player_name: str) -> Tuple[Optional[Dict[str, object]], str]:
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
            return profile, "espn_live"

        return generate_estimated_profile(name=name, position=position, team=team, jersey=jersey), source

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

    # 3) Last fallback: generic baseline.
    result = generate_estimated_profile(name=player_name.strip(), position="Unknown", team="Unknown")
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
    position = str(player_stats.get("position", "Unknown"))
    flags = position_flags(position)

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

    draft_round = float(player_stats.get("draft_round") or 8)
    combine_speed = float(player_stats.get("combine_speed_score") or 50.0)
    college_tier = float(player_stats.get("college_tier") or 3)
    production_score = float(
        player_stats.get("production_score")
        or compute_production_score(position, stats_dict)
    )

    row = {
        **stats_dict,
        "draft_round": draft_round,
        "combine_speed_score": combine_speed,
        "college_tier": college_tier,
        "production_score": production_score,
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


def train_success_model_from_synthetic(samples: int = 4000) -> xgb.XGBClassifier:
    """
    Train using realistic draft-round-based success probabilities rather than
    the proxy scoring rule, so the model learns features that actually matter.
    """
    random.seed(42)

    positions = ["QB", "RB", "WR", "TE", "LB", "CB", "S", "DL", "OL", "OTHER"]
    weights   = [0.15, 0.14, 0.20, 0.08, 0.10, 0.10, 0.08, 0.08, 0.04, 0.03]

    # Empirical draft-round distribution (roughly mirrors real NFL drafts)
    draft_rounds = [1, 2, 3, 4, 5, 6, 7, 8]
    draft_weights = [0.04, 0.06, 0.09, 0.11, 0.12, 0.14, 0.16, 0.28]  # more undrafted

    rows = []
    labels = []

    for _ in range(samples):
        position = random.choices(positions, weights=weights, k=1)[0]
        stats = synthetic_player_sample(position)
        draft_round = random.choices(draft_rounds, weights=draft_weights, k=1)[0]

        # College tier: better prospects more likely to come from P5 schools
        tier_weights = [0.55, 0.28, 0.17] if draft_round <= 3 else [0.35, 0.38, 0.27]
        college_tier = random.choices([1, 2, 3], weights=tier_weights, k=1)[0]

        # Combine speed correlated with draft round: earlier picks tend to be faster
        speed_mean = 70 - (draft_round - 1) * 5
        combine_speed = float(max(0, min(100, random.gauss(speed_mean, 15))))

        production = compute_production_score(position, stats)

        prob = realistic_nfl_success_probability(draft_round, combine_speed, college_tier, production)
        noisy_prob = min(max(prob + random.gauss(0, 0.04), 0.01), 0.99)
        success = 1 if random.random() < noisy_prob else 0

        flags = position_flags(position)
        rows.append(
            {
                "games_played":       stats["games_played"],
                "passing_touchdowns": stats["passing_touchdowns"],
                "passing_yards":      stats["passing_yards"],
                "rushing_touchdowns": stats["rushing_touchdowns"],
                "rushing_yards":      stats["rushing_yards"],
                "tackles":            stats["tackles"],
                "sacks":              stats["sacks"],
                "interceptions":      stats["interceptions"],
                "pass_deflections":   stats["pass_deflections"],
                "draft_round":        draft_round,
                "combine_speed_score": round(combine_speed, 1),
                "college_tier":       college_tier,
                "production_score":   round(production, 1),
                **flags,
            }
        )
        labels.append(success)

    X = pd.DataFrame(rows, columns=SUCCESS_FEATURES)
    y = pd.Series(labels)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    model.save_model(SUCCESS_MODEL_PATH)
    print(f"Trained and saved improved success model: {SUCCESS_MODEL_PATH}")
    return model


def load_or_train_success_model() -> None:
    global success_model

    if os.path.exists(SUCCESS_MODEL_PATH):
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(SUCCESS_MODEL_PATH)
        success_model = loaded_model
        print("Success model loaded.")
        return

    print("Success model not found. Training a new one from synthetic samples...")
    success_model = train_success_model_from_synthetic()


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
    "draft_round":          "Draft Round",
    "production_score":     "Production Score",
    "combine_speed_score":  "Combine Athleticism",
    "college_tier":         "College Competition Level",
    "games_played":         "Games Played",
    "passing_touchdowns":   "Passing TDs",
    "passing_yards":        "Passing Yards",
    "rushing_touchdowns":   "Rushing TDs",
    "rushing_yards":        "Rushing Yards",
    "position_qb":          "Position: QB",
    "position_rb":          "Position: RB",
    "position_wr":          "Position: WR",
    "position_te":          "Position: TE",
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
    if success_model is None:
        return None, None, None, False

    try:
        model_input = build_success_features(player_stats)
        probability = float(success_model.predict_proba(model_input)[0][1])
        label = "Success" if probability >= 0.5 else "No Success"
        confidence = round((probability if label == "Success" else 1.0 - probability) * 100.0, 1)
        return label, confidence, round(probability * 100.0, 1), True
    except Exception as exc:
        print(f"Success model inference failed: {exc}")
        return None, None, None, False


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

    player_data, data_source = fetch_player_data(player_name)
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
        reasoning = "Direct ML success classifier prediction from player usage and production features."

    position = str(player_data.get("position", "Unknown"))
    draft_round = int(player_data.get("draft_round") or 8)
    combine_speed = float(player_data.get("combine_speed_score") or 50.0)
    college_tier = int(player_data.get("college_tier") or 3)
    production = float(
        player_data.get("production_score")
        or compute_production_score(position, player_data)
    )

    round_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
                    5: "5th", 6: "6th", 7: "7th", 8: "Undrafted"}

    # Detect NFL team stored in DB → show clearer label
    stored_team = str(player_data.get("team", "") or "")
    is_nfl_player = any(kw in stored_team.lower() for kw in NFL_FRANCHISE_KEYWORDS)
    if is_nfl_player:
        tier_label = "Power 5 (NFL Pro)"
    else:
        tier_label = {1: "Power 5", 2: "Group of 5", 3: "FCS / Other"}.get(college_tier, "Unknown")

    season_label = str(player_data.get("_season") or "")
    completion_pct = str(player_data.get("_completion_pct") or "")
    interceptions = player_data.get("_interceptions")
    qb_rating = str(player_data.get("_qb_rating") or "")

    summary = {
        "draft_projection": round_labels.get(draft_round, "Undrafted"),
        "combine_athleticism": f"{combine_speed:.0f} / 100",
        "college_tier": tier_label,
        "production_score": f"{production:.0f} / 100",
    }
    if season_label:
        summary["season"] = season_label
    if completion_pct and position.upper() == "QB":
        summary["completion_pct"] = f"{completion_pct}%"
    if qb_rating and position.upper() == "QB":
        summary["passer_rating"] = qb_rating
    if interceptions is not None and position.upper() == "QB":
        summary["interceptions"] = str(interceptions)

    return jsonify(
        {
            "requested_name": player_name,
            "resolved_name": str(player_data.get("name", player_name)),
            "success": success_label,
            "confidence": confidence,
            "reasoning": reasoning,
            "predicted_position": predicted_position,
            "success_probability": success_probability,
            "model_confidence": confidence,
            "model_used": model_used,
            "model_type": "success_classifier",
            "data_source": data_source,
            "stats": player_data,
            "summary": summary,
            "top_factors": top_feature_importances(4),
        }
    )


initialize_player_database()
load_position_model_artifacts()
load_or_train_success_model()

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
