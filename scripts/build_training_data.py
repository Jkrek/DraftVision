#!/usr/bin/env python
"""Build training_data/combine_outcomes.csv from REAL public data (no API keys).

Sources
-------
1. nflverse draft_picks.csv  — every draft pick w/ career outcomes
   (probowls, seasons_started, w_av, games, `to` = last season played).
2. nflverse combine.csv      — every combine invitee 2000+ w/ measured forty,
   pfr_id linkage, and draft round/pick when drafted.
3. sportsdataverse cfbfastR-data player_stats_{season}.csv — PLAY-level CFB
   participation data (2014+ only). Aggregated to season totals per player to
   compute the same position-normalized production composite the app uses.

Scope: draft classes 2010-2020 (>=5 completed NFL seasons through 2025, so
outcome labels are not right-censored) PLUS combine invitees from those years
who went undrafted — the structurally-missing negative/UDFA examples that the
old ESPN-scraped dataset (collect_training_data.py) could never see.

Honest limitations (reported, not papered over):
- cfbfastR player stats only exist from the 2014 season, so college production
  is only computable for draft classes 2015-2020 (final season = class - 1).
- The play-level data has NO tackle events, so a faithful production composite
  is only computable for QB/RB/WR/TE (the offensive branches of
  compute_production_score). Defensive/OL rows get a blank production_score
  rather than a fabricated one. OL never appear in play-participation data at
  all, so their games-played proxy is also unobtainable.
- games_college is left blank for every row: the play-level source only counts
  games where a player recorded a stat, which systematically undercounts and
  would corrupt a games-played feature. Blank is honest; the play counts still
  back the per-game rates inside production_score for skill positions.
- Undrafted combine invitees with no NFL record are labeled nfl_success=0.
  A handful of successful UDFAs (e.g. Malcolm Butler) are therefore false
  negatives; these files carry no NFL outcome data for undrafted players.

Run with:  .venv/bin/python scripts/build_training_data.py
"""

import os
import re
import ssl
import unicodedata
import urllib.request

try:  # macOS system Python ships without usable CA roots; prefer certifi
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CTX = ssl.create_default_context()
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv")
CACHE_DIR = os.environ.get("DV_DATA_CACHE", "/tmp/dv_training_cache")

DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
CFB_URL_TMPL = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/"
    "player_stats/csv/player_stats_{season}.csv"
)

FIRST_CLASS, LAST_CLASS = 2010, 2020
CFB_FIRST_SEASON = 2014  # earliest season cfbfastR-data publishes player stats

# ---------------------------------------------------------------------------
# Tier / speed-score / production logic replicated VERBATIM from XGBOost.py
# (importing XGBOost would boot the Flask app, so the small pure functions and
# their tables are copied here; values must stay identical).
# ---------------------------------------------------------------------------

NFL_FRANCHISE_NAMES = {
    "arizona cardinals", "atlanta falcons", "baltimore ravens", "buffalo bills",
    "carolina panthers", "chicago bears", "cincinnati bengals", "cleveland browns",
    "dallas cowboys", "denver broncos", "detroit lions", "green bay packers",
    "houston texans", "indianapolis colts", "jacksonville jaguars",
    "kansas city chiefs", "las vegas raiders", "los angeles chargers",
    "los angeles rams", "miami dolphins", "minnesota vikings",
    "new england patriots", "new orleans saints", "new york giants",
    "new york jets", "philadelphia eagles", "pittsburgh steelers",
    "san francisco 49ers", "seattle seahawks", "tampa bay buccaneers",
    "tennessee titans", "washington commanders",
}

_TIER_SCHOOLS = {
    1: {"alabama", "ohio state", "georgia", "clemson", "lsu", "michigan"},
    2: {"texas", "oklahoma", "florida", "penn state", "notre dame", "florida state",
        "tennessee", "texas a&m", "usc", "oregon", "miami", "auburn", "washington"},
    3: {"north carolina", "virginia tech", "pittsburgh", "wisconsin", "iowa",
        "michigan state", "nebraska", "oklahoma state", "baylor", "tcu", "arkansas",
        "ole miss", "mississippi state", "south carolina", "stanford", "utah",
        "arizona state", "colorado", "georgia tech"},
    4: {"west virginia", "kansas state", "iowa state", "texas tech", "kentucky",
        "vanderbilt", "missouri", "arizona", "cal", "california", "oregon state",
        "washington state", "indiana", "purdue", "illinois", "minnesota",
        "maryland", "rutgers", "louisville", "virginia", "nc state", "duke",
        "wake forest", "syracuse", "boston college", "cincinnati", "ucf"},
    5: {"ucla", "northwestern", "navy", "army", "air force", "liberty", "byu",
        "western kentucky", "louisiana tech"},
    6: {"memphis", "houston", "smu", "tulane", "east carolina", "south florida",
        "temple", "connecticut", "uconn", "tulsa", "rice", "utep", "uab"},
    7: {"boise state", "fresno state", "hawaii", "san diego state", "wyoming",
        "utah state", "nevada", "colorado state", "new mexico", "san jose state"},
    8: {"appalachian state", "app state", "coastal carolina", "marshall", "utsa",
        "troy", "louisiana", "james madison", "buffalo", "kent state", "ohio",
        "miami (oh)", "western michigan", "central michigan", "eastern michigan",
        "northern illinois", "ball state", "toledo"},
    9: {"north dakota state", "montana", "south dakota state", "furman",
        "villanova", "richmond", "delaware", "sacramento state",
        "central arkansas", "cal poly"},
}

_SCHOOL_TIERS: Dict[str, int] = {
    school: tier for tier, schools in _TIER_SCHOOLS.items() for school in schools
}
_SCHOOL_KEYS = sorted(_SCHOOL_TIERS, key=len, reverse=True)


def _normalize_team(team: str) -> str:
    t = (team or "").lower().strip()
    for ch in ("’", "'", "."):
        t = t.replace(ch, "")
    return t.replace("é", "e")  # San José State


def classify_college_tier(team: str) -> int:
    """Identical to XGBOost.classify_college_tier (10-tier, longest-prefix)."""
    t = _normalize_team(team)
    if not t:
        return 10
    if t in NFL_FRANCHISE_NAMES:
        return 1
    for key in _SCHOOL_KEYS:
        if t == key or t.startswith(key + " "):
            return _SCHOOL_TIERS[key]
    return 10


def forty_to_speed_score(position: str, forty: float) -> float:
    """Identical benchmarks to XGBOost.forty_to_speed_score."""
    if not forty or forty <= 0:
        return 0.0
    p = (position or "").upper()
    benchmarks = {
        "QB":  (4.30, 5.10), "RB":  (4.20, 4.80), "WR":  (4.20, 4.70),
        "TE":  (4.40, 5.00), "CB":  (4.20, 4.65), "S":   (4.30, 4.75),
        "DB":  (4.25, 4.70), "LB":  (4.35, 4.85), "DL":  (4.50, 5.30),
        "DE":  (4.45, 5.10), "DT":  (4.55, 5.35), "EDGE": (4.45, 5.10),
        "OL":  (4.70, 5.55), "OT":  (4.75, 5.60), "OG":  (4.80, 5.55), "C": (4.85, 5.60),
    }
    elite_t, poor_t = benchmarks.get(p, (4.35, 5.00))
    score = (poor_t - forty) / (poor_t - elite_t) * 100.0
    return float(max(0.0, min(100.0, score)))


def compute_production_score(position: str, stats: dict) -> float:
    """Identical composite to XGBOost.compute_production_score."""
    p = (position or "").upper()
    games = max(float(stats.get("games_played", 1) or 1), 1)
    pass_td = float(stats.get("passing_touchdowns", 0) or 0)
    pass_yds = float(stats.get("passing_yards", 0) or 0)
    rush_td = float(stats.get("rushing_touchdowns", 0) or 0)
    rush_yds = float(stats.get("rushing_yards", 0) or 0)
    tackles = float(stats.get("tackles", 0) or 0)
    sacks = float(stats.get("sacks", 0) or 0)
    ints = float(stats.get("interceptions", 0) or 0)
    pds = float(stats.get("pass_deflections", 0) or 0)

    if p == "QB":
        td_rate = (pass_td / games) / (35 / 15)
        yd_rate = (pass_yds / games) / (4500 / 15)
        score = (td_rate * 0.5 + yd_rate * 0.5) * 100
    elif p == "RB":
        td_rate = (rush_td / games) / (14 / 15)
        yd_rate = (rush_yds / games) / (1500 / 15)
        score = (td_rate * 0.45 + yd_rate * 0.55) * 100
    elif p in {"WR", "TE"}:
        combined_td = (pass_td + rush_td) / games / (12 / 15)
        combined_yds = (pass_yds + rush_yds) / games / (1200 / 15)
        score = (combined_td * 0.4 + combined_yds * 0.6) * 100
    elif p in {"LB", "ILB", "OLB", "MLB"}:
        t_rate = (tackles / games) / (100 / 13)
        s_rate = (sacks / games) / (8 / 13)
        i_rate = (ints / games) / (3 / 13)
        score = (t_rate * 0.55 + s_rate * 0.30 + i_rate * 0.15) * 100
    elif p in {"CB", "S", "DB", "FS", "SS"}:
        t_rate = (tackles / games) / (55 / 13)
        i_rate = (ints / games) / (4 / 13)
        p_rate = (pds / games) / (12 / 13)
        score = (t_rate * 0.25 + i_rate * 0.40 + p_rate * 0.35) * 100
    elif p in {"DL", "DE", "DT", "NT", "EDGE"}:
        t_rate = (tackles / games) / (45 / 13)
        s_rate = (sacks / games) / (10 / 13)
        score = (t_rate * 0.35 + s_rate * 0.65) * 100
    elif p in {"OL", "OT", "OG", "G", "C", "LS"}:
        score = min(games / 13.0, 1.0) * 60.0
    else:
        combined_td = (pass_td + rush_td) / games / (12 / 15)
        combined_yds = (pass_yds + rush_yds) / games / (1200 / 15)
        score = (combined_td * 0.4 + combined_yds * 0.6) * 100

    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# School-name cleanup: PFR/nflverse spellings -> the app's tier-table spellings.
# This is input normalization only; tier VALUES come from the identical tables
# above. Schools absent from the app's tables still fall to tier 10, exactly
# as the app would score them.
# ---------------------------------------------------------------------------

_SCHOOL_ALIASES = {
    "mississippi": "Ole Miss",
    "boston col": "Boston College",
    "central florida": "UCF",
    "ala-birmingham": "UAB",
    "texas-el paso": "UTEP",
    "texas-san antonio": "UTSA",
    "louisiana-lafayette": "Louisiana",
    "north carolina state": "NC State",
    "north carolina st": "NC State",
    "southern california": "USC",
    "cal poly-san luis obispo": "Cal Poly",
    "brigham young": "BYU",
    "texas christian": "TCU",
    "louisiana state": "LSU",
    "southern methodist": "SMU",
}


def clean_school(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\bSt\.($|\s)", r"State\1", s)          # "Penn St." -> "Penn State"
    s = re.sub(r"\bWest\.\s", "Western ", s)
    s = re.sub(r"\bEast\.\s", "Eastern ", s)
    key = _normalize_team(s)
    return _SCHOOL_ALIASES.get(key, s)


# Draft-file position spellings -> combine/app spellings
_POS_MAP = {"T": "OT", "G": "OG"}


def norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    return _POS_MAP.get(p, p)


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """Lowercase, de-accent, strip punctuation and generational suffixes."""
    n = unicodedata.normalize("NFKD", str(name or ""))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"[.'’,-]", " ", n)
    parts = [w for w in n.split() if w not in _SUFFIXES]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch(url: str, fname: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "DraftVision-training-builder"})
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp, \
            open(path, "wb") as f:
        f.write(resp.read())
    return path


# ---------------------------------------------------------------------------
# College production from cfbfastR play-level data (2014+ seasons only)
# ---------------------------------------------------------------------------

# Play-participation columns used to accumulate season stats
_STAT_COLS = [
    "game_id", "season",
    "completion_player", "completion_yds",
    "reception_player", "reception_yds",
    "rush_player", "rush_yds",
    "touchdown_player",
    "incompletion_player", "target_player", "sack_taken_player",
]


def load_cfb_season(season: int) -> Dict[str, dict]:
    """Aggregate one season of play-level data into per-player offense totals.

    Returns {norm_name: {passing_yards, passing_touchdowns, rushing_yards,
    rushing_touchdowns, receiving_yards, receiving_touchdowns, games_played,
    teams:set}} — only offensive events exist in this source (no tackles), so
    only QB/RB/WR/TE production is computable downstream.
    """
    path = fetch(CFB_URL_TMPL.format(season=season), f"player_stats_{season}.csv")
    df = pd.read_csv(path, usecols=lambda c: c in set(_STAT_COLS + ["team"]),
                     low_memory=False)
    for col in _STAT_COLS[2:]:
        if col not in df.columns:
            df[col] = np.nan

    players: Dict[str, dict] = defaultdict(lambda: {
        "passing_yards": 0.0, "passing_touchdowns": 0.0,
        "rushing_yards": 0.0, "rushing_touchdowns": 0.0,
        "receiving_yards": 0.0, "receiving_touchdowns": 0.0,
        "games": set(), "teams": set(),
    })

    def _num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _nm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip()
        return "" if s in ("", "NA", "nan", "None") else norm_name(s)

    td_col = df["touchdown_player"].map(_nm)
    team_col = df["team"].astype(str)
    gid = df["game_id"]

    for role, yds_col in (("completion_player", "completion_yds"),
                          ("reception_player", "reception_yds"),
                          ("rush_player", "rush_yds")):
        names = df[role].map(_nm)
        yds = df[yds_col].map(_num)
        mask = names != ""
        for name, y, td, team, g in zip(names[mask], yds[mask], td_col[mask],
                                        team_col[mask], gid[mask]):
            rec = players[name]
            rec["games"].add(g)
            rec["teams"].add(team)
            if role == "completion_player":
                rec["passing_yards"] += y
                if td:  # a scorer on a completed pass = passing TD for the passer
                    rec["passing_touchdowns"] += 1
            elif role == "reception_player":
                rec["receiving_yards"] += y
                # NOTE: parser eras disagree on touchdown_player for pass TDs
                # (sometimes the passer, sometimes the receiver). On a completed-
                # reception row, any TD flag means the receiver scored.
                if td:
                    rec["receiving_touchdowns"] += 1
            else:
                rec["rushing_yards"] += y
                if td:
                    rec["rushing_touchdowns"] += 1

    # Participation-only columns still mark a game appearance
    for role in ("incompletion_player", "target_player", "sack_taken_player"):
        names = df[role].map(_nm)
        mask = names != ""
        for name, team, g in zip(names[mask], team_col[mask], gid[mask]):
            if name in players:
                players[name]["games"].add(g)
                players[name]["teams"].add(team)

    out = {}
    for name, rec in players.items():
        rec["games_played"] = len(rec["games"])
        del rec["games"]
        out[name] = rec
    return out


_PRODUCTION_POSITIONS = {"QB", "RB", "WR", "TE"}  # offense only — see module docstring


def production_for(name: str, school: str, position: str, draft_year: int,
                   cfb_by_season: Dict[int, Dict[str, dict]]) -> Optional[float]:
    """Position-normalized production for the final college season before the
    draft (falls back one more year for opt-outs/injuries). None if the source
    lacks the season, the player, or the stats needed for this position."""
    pos = norm_pos(position)
    if pos not in _PRODUCTION_POSITIONS:
        return None
    key = norm_name(name)
    school_key = _normalize_team(clean_school(school))
    for season in (draft_year - 1, draft_year - 2):
        table = cfb_by_season.get(season)
        if not table or key not in table:
            continue
        rec = table[key]
        # School sanity check: accept if any of the player's teams share a
        # word-prefix with the draft-file school (team spellings differ).
        teams = {_normalize_team(t) for t in rec["teams"]}
        if school_key and teams:
            ok = any(t == school_key or t.startswith(school_key + " ")
                     or school_key.startswith(t + " ") or school_key in t or t in school_key
                     for t in teams)
            if not ok:
                continue  # same name, different program — do not use
        stats = {"games_played": rec["games_played"]}
        if pos == "QB":
            stats["passing_yards"] = rec["passing_yards"]
            stats["passing_touchdowns"] = rec["passing_touchdowns"]
        elif pos == "RB":
            stats["rushing_yards"] = rec["rushing_yards"]
            stats["rushing_touchdowns"] = rec["rushing_touchdowns"]
        else:  # WR/TE — composite sums the pass+rush slots, so receiving
            # production is fed through the passing slot (same effective math
            # as the app's scrimmage-combining WR/TE branch).
            stats["passing_yards"] = rec["receiving_yards"]
            stats["passing_touchdowns"] = rec["receiving_touchdowns"]
            stats["rushing_yards"] = rec["rushing_yards"]
            stats["rushing_touchdowns"] = rec["rushing_touchdowns"]
        if rec["games_played"] < 3:
            continue  # too little participation to trust a per-game rate
        return round(compute_production_score(pos, stats), 2)
    return None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# nfl_success = 1 if any of:
#   probowls >= 1            (peak recognition)
#   seasons_started >= 3     (sustained starter)
#   career weighted AV >= 30 (PFR's w_av; ~5 seasons of average-starter value
#                             at roughly 6 AV/season — catches high-value
#                             rotational players who never "started" 3 years)
AV_SUCCESS_THRESHOLD = 30


def success_label(probowls, seasons_started, av) -> int:
    if (probowls or 0) >= 1:
        return 1
    if (seasons_started or 0) >= 3:
        return 1
    if (av or 0) >= AV_SUCCESS_THRESHOLD:
        return 1
    return 0


def draft_grade_for(rnd: int) -> int:
    if rnd <= 2:
        return 0
    if rnd <= 4:
        return 1
    if rnd <= 7:
        return 2
    return 3  # undrafted


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching nflverse data…")
    draft = pd.read_csv(fetch(DRAFT_URL, "draft_picks.csv"))
    combine = pd.read_csv(fetch(COMBINE_URL, "combine.csv"))

    draft = draft[(draft.season >= FIRST_CLASS) & (draft.season <= LAST_CLASS)].copy()
    combine = combine[(combine.season >= FIRST_CLASS) & (combine.season <= LAST_CLASS)].copy()

    # Career AV: car_av is null for these classes in the current release; use w_av.
    draft["av"] = draft["car_av"].fillna(draft["w_av"])

    # Combine lookup by pfr_id and by (name, school) for forty times
    comb_by_pfr = {}
    comb_by_name = {}
    for _, r in combine.iterrows():
        if pd.notna(r.get("pfr_id")):
            comb_by_pfr[(r["pfr_id"], int(r["season"]))] = r
        comb_by_name[(norm_name(r["player_name"]), int(r["season"]))] = r

    print("Fetching cfbfastR play-level stats (2014+; earlier seasons not published)…")
    cfb_by_season: Dict[int, Dict[str, dict]] = {}
    for season in range(CFB_FIRST_SEASON, LAST_CLASS):  # 2014..2019 final seasons
        try:
            cfb_by_season[season] = load_cfb_season(season)
            print(f"  {season}: {len(cfb_by_season[season])} players with offense events")
        except Exception as exc:  # noqa: BLE001 — source outage must not kill the build
            print(f"  WARNING: could not load {season} ({exc}); production blank for that season")

    rows = []
    drafted_pfr_ids = set(draft["pfr_player_id"].dropna())
    drafted_names = {(norm_name(n), int(s)) for n, s in
                     zip(draft["pfr_player_name"], draft["season"])}

    prod_eligible = 0  # QB/RB/WR/TE rows in classes with CFB data — the honest denominator
    prod_matched = 0

    print("Building drafted rows…")
    for _, r in draft.iterrows():
        year = int(r["season"])
        pos = norm_pos(r["position"])
        name = r["pfr_player_name"]
        college = clean_school(r["college"] if pd.notna(r["college"]) else "")

        crow = None
        if pd.notna(r.get("pfr_player_id")):
            crow = comb_by_pfr.get((r["pfr_player_id"], year))
        if crow is None:
            crow = comb_by_name.get((norm_name(name), year))
        forty = None
        if crow is not None and pd.notna(crow.get("forty")):
            forty = float(crow["forty"])

        prod = production_for(name, college, pos, year, cfb_by_season)
        if pos in _PRODUCTION_POSITIONS and (year - 1) in cfb_by_season:
            prod_eligible += 1
            if prod is not None:
                prod_matched += 1

        to_season = r["to"] if pd.notna(r["to"]) else None
        exp = int(max(0, to_season - year + 1)) if to_season else 0
        probowls = int(r["probowls"]) if pd.notna(r["probowls"]) else 0
        sstart = int(r["seasons_started"]) if pd.notna(r["seasons_started"]) else 0
        av = float(r["av"]) if pd.notna(r["av"]) else 0.0
        rnd = int(r["round"])

        rows.append({
            "name": name,
            "draft_year": year,
            "position": pos,
            "college": college,
            "conference_tier": classify_college_tier(college),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": rnd,
            "draft_grade": draft_grade_for(rnd),
            "nfl_success": success_label(probowls, sstart, av),
            "experience_years": exp,
            "pro_bowls": probowls,
            "seasons_started": sstart,
            "career_av": av,
        })

    print("Building undrafted combine-invitee rows…")
    for _, r in combine.iterrows():
        year = int(r["season"])
        # Drafted according to the combine file itself?
        if pd.notna(r.get("draft_round")):
            continue
        # …or matched to an actual draft pick (belt and suspenders)?
        if pd.notna(r.get("pfr_id")) and r["pfr_id"] in drafted_pfr_ids:
            continue
        if (norm_name(r["player_name"]), year) in drafted_names:
            continue

        pos = norm_pos(r["pos"])
        name = r["player_name"]
        school = clean_school(r["school"] if pd.notna(r["school"]) else "")
        forty = float(r["forty"]) if pd.notna(r["forty"]) else None

        prod = production_for(name, school, pos, year, cfb_by_season)
        if pos in _PRODUCTION_POSITIONS and (year - 1) in cfb_by_season:
            prod_eligible += 1
            if prod is not None:
                prod_matched += 1

        rows.append({
            "name": name,
            "draft_year": year,
            "position": pos,
            "college": school,
            "conference_tier": classify_college_tier(school),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": 8,
            "draft_grade": 3,
            # These sources carry no NFL outcome data for undrafted players;
            # label 0 (a few successful UDFAs become false negatives).
            "nfl_success": 0,
            "experience_years": 0,
            "pro_bowls": 0,
            "seasons_started": 0,
            "career_av": 0.0,
        })

    df = pd.DataFrame(rows, columns=[
        "name", "draft_year", "position", "college", "conference_tier",
        "combine_forty", "combine_speed_score", "games_college",
        "production_score", "draft_round", "draft_grade", "nfl_success",
        "experience_years", "pro_bowls", "seasons_started", "career_av",
    ])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # ── Report ──────────────────────────────────────────────────────────────
    n = len(df)
    drafted = int((df.draft_round <= 7).sum())
    undrafted = n - drafted
    print(f"\nWrote {OUT_PATH}")
    print(f"Total rows: {n}  (drafted {drafted}, undrafted {undrafted})")
    print(f"nfl_success overall: {int(df.nfl_success.sum())} / {n} "
          f"({df.nfl_success.mean() * 100:.1f}%)")
    for grade, label in ((0, "R1-2"), (1, "R3-4"), (2, "R5-7"), (3, "UDFA")):
        sub = df[df.draft_grade == grade]
        print(f"  {label}: {int(sub.nfl_success.sum())}/{len(sub)} success "
              f"({sub.nfl_success.mean() * 100:.1f}%)")
    prod_cov = (df.production_score != "").sum()
    forty_cov = (df.combine_forty != "").sum()
    print(f"production_score coverage: {prod_cov}/{n} ({prod_cov / n * 100:.1f}%) "
          f"— eligible pool (QB/RB/WR/TE, classes {CFB_FIRST_SEASON + 1}-{LAST_CLASS}): "
          f"{prod_matched}/{prod_eligible} matched "
          f"({(prod_matched / prod_eligible * 100) if prod_eligible else 0:.1f}%)")
    print(f"combine_forty coverage: {forty_cov}/{n} ({forty_cov / n * 100:.1f}%)")
    print("Rows per draft year:")
    for year, cnt in df.draft_year.value_counts().sort_index().items():
        print(f"  {year}: {cnt}")


if __name__ == "__main__":
    main()
