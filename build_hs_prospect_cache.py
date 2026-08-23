#!/usr/bin/env python3
"""
Fetch high school football recruiting rankings and save them to
training_data/hs_prospect_cache.json.

Sources, in preference order per class year:
  1. CFBD (/recruiting/players) — carries the 247Sports composite rating, but
     only publishes a class around its signing cycle.
  2. ESPN public core API (recruiting/{year}/athletes) — has future classes
     (e.g. 2027-2030) years early: national/position/state rank, ESPN 0-100
     grade, HS + hometown, commitment. Used when CFBD returns zero rows.

Setup:
    1. Get a free API key at https://collegefootballdata.com/key
    2. Export it: export CFBD_API_KEY=your_key_here
    3. Run: python build_hs_prospect_cache.py

Options:
    --years         Comma-separated class years (default: 2027,2028 = current HS)
    --min-stars     Minimum star rating to include (default: 3)
    --output        Output file path (default: training_data/hs_prospect_cache.json)
"""

import argparse
import json
import os
import time
import requests
from datetime import datetime, timezone

CFBD_BASE_URL   = "https://api.collegefootballdata.com"
OUTPUT_FILE     = "training_data/hs_prospect_cache.json"

# Partial mapping of college teams → conference tier (mirrors XGBOost.py logic)
POWER5_SCHOOLS = {
    "alabama", "georgia", "ohio state", "michigan", "clemson", "lsu", "oklahoma",
    "texas", "notre dame", "penn state", "florida", "auburn", "tennessee", "oregon",
    "washington", "usc", "ucla", "stanford", "miami", "florida state", "nebraska",
    "iowa", "wisconsin", "minnesota", "purdue", "illinois", "indiana", "northwestern",
    "rutgers", "maryland", "michigan state", "kansas state", "iowa state", "baylor",
    "tcu", "west virginia", "texas tech", "kansas", "oklahoma state", "cincinnati",
    "pittsburgh", "virginia", "virginia tech", "north carolina", "nc state", "duke",
    "wake forest", "syracuse", "boston college", "louisville", "kentucky", "vanderbilt",
    "south carolina", "mississippi state", "ole miss", "arkansas", "texas a&m",
    "missouri", "colorado", "utah", "arizona state", "arizona", "cal", "oregon state",
    "washington state",
}

G5_SCHOOLS = {
    "boise state", "houston", "ucf", "memphis", "appalachian state", "coastal carolina",
    "liberty", "james madison", "army", "navy", "air force", "miami (oh)",
    "western kentucky", "marshall", "troy", "georgia southern", "louisiana",
    "toledo", "ohio", "bowling green", "northern illinois", "ball state",
    "fresno state", "san diego state", "hawaii", "utah state", "nevada",
    "unlv", "wyoming", "air force",
}


def classify_tier(committed_to: str) -> int:
    """Return 1–10 conference tier for a committed-to college name."""
    t = (committed_to or "").lower().strip()
    if not t or t in ("uncommitted", "undecided", ""):
        return 5  # neutral
    for school in POWER5_SCHOOLS:
        if school in t:
            return 2
    for school in G5_SCHOOLS:
        if school in t:
            return 4
    return 5  # unknown mid-major


def compute_hs_grade(stars: int, rating: float, ranking: int, committed_to: str) -> str:
    """Assign a letter grade based on recruiting profile."""
    s = int(stars or 0)
    r = float(rating or 0)
    rk = int(ranking or 9999)

    if s == 5 and rk <= 10:
        return "A+"
    if s == 5 and rk <= 50:
        return "A"
    if s == 5 or rk <= 100:
        return "A-"
    if s == 4 and rk <= 200:
        return "B+"
    if s == 4 and rk <= 500:
        return "B"
    if s == 4:
        return "B-"
    if s == 3 and r >= 0.88:
        return "C+"
    if s == 3:
        return "C"
    if s == 2:
        return "D"
    return "D"


def cfbd_get(endpoint: str, params: dict, api_key: str, retries: int = 3) -> list:
    """Make a CFBD API request and return the JSON list."""
    url     = f"{CFBD_BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.ok:
                return r.json() or []
            if r.status_code == 401:
                print("ERROR: Invalid CFBD API key. Get one free at https://collegefootballdata.com/key")
                return []
            if r.status_code == 429:
                print(f"Rate limited — waiting 5s...")
                time.sleep(5)
                continue
            print(f"  CFBD {r.status_code}: {endpoint}")
        except Exception as e:
            if attempt == retries - 1:
                print(f"  CFBD request failed: {e}")
            time.sleep(1 * (attempt + 1))
    return []


def fetch_recruits_for_year(year: int, api_key: str, min_stars: int = 3) -> list:
    """Fetch all HS recruits for a given class year from CFBD."""
    raw = cfbd_get(
        "/recruiting/players",
        {"year": year, "classification": "HighSchool"},
        api_key,
    )
    if not raw:
        return []

    prospects = []
    for p in raw:
        stars   = int(p.get("stars") or 0)
        if stars < min_stars:
            continue
        rating  = float(p.get("rating") or 0)
        ranking = int(p.get("ranking") or 9999)
        committed = str(p.get("committedTo") or "").strip()
        position  = str(p.get("position") or "").upper().strip()

        # Normalize some CFBD position names to match our model
        pos_map = {
            "APB": "RB", "ATH": "WR", "DUAL": "QB", "ILB": "LB", "OLB": "LB",
            "PRO": "QB", "SDE": "DE", "WDE": "DE", "OT": "OL", "OG": "OL",
            "FS": "S", "SS": "S",
        }
        position = pos_map.get(position, position)

        grade = compute_hs_grade(stars, rating, ranking, committed)

        prospects.append({
            "name":         str(p.get("name") or "").strip(),
            "position":     position,
            "school":       str(p.get("school") or "").strip(),
            "city":         str(p.get("city") or "").strip(),
            "state":        str(p.get("stateProvince") or "").strip(),
            "country":      str(p.get("country") or "USA").strip(),
            "year":         year,
            "stars":        stars,
            "rating":       round(rating, 4),
            "ranking":      ranking,
            "committed_to": committed,
            "conference_tier": classify_tier(committed),
            "grade":        grade,
            "cfbd_id":      str(p.get("id") or ""),
        })

    print(f"  Year {year}: {len(prospects)} prospects (≥{min_stars}★)")
    return prospects


ESPN_RECRUITING_URL = ("https://sports.core.api.espn.com/v2/sports/football/"
                       "leagues/college-football/recruiting/{year}/athletes")
ESPN_TEAMS_URL      = ("https://site.api.espn.com/apis/site/v2/sports/football/"
                       "college-football/teams?limit=1000")

# CFBD/ESPN position spellings → our model's buckets
POS_MAP = {
    "APB": "RB", "ATH": "WR", "DUAL": "QB", "ILB": "LB", "OLB": "LB",
    "PRO": "QB", "SDE": "DE", "WDE": "DE", "OT": "OL", "OG": "OL",
    "FS": "S", "SS": "S",
}


def espn_stars(grade) -> int:
    """ESPN's published grade bands → star equivalents."""
    g = float(grade or 0)
    if g >= 90: return 5
    if g >= 80: return 4
    if g >= 70: return 3
    if g >= 60: return 2
    return 1 if g > 0 else 0


def fetch_espn_team_names() -> dict:
    """ESPN team id → school name (e.g. '194' → 'Ohio State')."""
    try:
        r = requests.get(ESPN_TEAMS_URL, timeout=20)
        r.raise_for_status()
        teams = r.json()["sports"][0]["leagues"][0]["teams"]
        return {str(t["team"]["id"]): t["team"].get("location") or t["team"].get("displayName", "")
                for t in teams}
    except Exception as e:
        print(f"  WARN: ESPN team list fetch failed ({e}) — commitments will be blank")
        return {}


def fetch_recruits_espn(year: int, min_stars: int, team_names: dict) -> list:
    """Fetch a class year from ESPN's public recruiting API."""
    items, page = [], 1
    while True:
        try:
            r = requests.get(ESPN_RECRUITING_URL.format(year=year),
                             params={"limit": 200, "page": page}, timeout=20)
            if not r.ok:
                print(f"  ESPN {r.status_code} for class {year} page {page}")
                break
            d = r.json()
        except Exception as e:
            print(f"  ESPN request failed for class {year}: {e}")
            break
        items.extend(d.get("items") or [])
        if page >= int(d.get("pageCount") or 1):
            break
        page += 1
        time.sleep(0.3)

    prospects = []
    for it in items:
        ath   = it.get("athlete") or {}
        name  = str(ath.get("fullName") or "").strip()
        if not name:
            continue
        stars = espn_stars(it.get("grade"))
        if stars < min_stars:
            continue

        rank = 9999
        for a in it.get("attributes") or []:
            if a.get("name") == "rank":
                try: rank = int(a.get("value") or 9999)
                except (TypeError, ValueError): pass
                break

        committed = ""
        for s in it.get("schools") or []:
            desc = ((s.get("status") or {}).get("description") or "")
            if desc.lower() in ("verbal", "commit", "committed", "signed", "enrolled"):
                ref = ((s.get("team") or {}).get("$ref") or "")
                tid = ref.rstrip("/").split("teams/")[-1].split("?")[0] if "teams/" in ref else ""
                committed = team_names.get(tid, "")
                break

        hs       = ath.get("highSchool") or {}
        addr     = hs.get("address") or {}
        hometown = ath.get("hometown") or {}
        # ESPN two-way kids come as "QB-DT" etc. — keep the primary side
        position = str((ath.get("position") or {}).get("abbreviation") or "").upper().strip()
        position = POS_MAP.get(position.split("-")[0], position.split("-")[0])
        rating   = round(float(it.get("grade") or 0) / 100.0, 4)

        prospects.append({
            "name":         name,
            "position":     position,
            "school":       str(hs.get("name") or "").strip(),
            "city":         str(addr.get("city") or hometown.get("city") or "").strip(),
            "state":        str(addr.get("stateAbbreviation") or hometown.get("stateAbbreviation") or "").strip(),
            "country":      "USA",
            "year":         year,
            "stars":        stars,
            "rating":       rating,
            "ranking":      rank,
            "committed_to": committed,
            "conference_tier": classify_tier(committed),
            "grade":        compute_hs_grade(stars, rating, rank, committed),
            "espn_id":      str(ath.get("id") or ""),
            "source":       "espn",
        })

    print(f"  Year {year}: {len(prospects)} prospects (≥{min_stars}★) [ESPN]")
    return prospects


def main():
    parser = argparse.ArgumentParser(description="Build HS prospect cache from CFBD API.")
    parser.add_argument("--years",      default="2026,2027,2028,2029,2030",
                        help="Comma-separated class years (default: current HS classes)")
    parser.add_argument("--min-stars",  type=int, default=3,
                        help="Minimum star rating to include (default: 3)")
    parser.add_argument("--output",     default=OUTPUT_FILE,
                        help=f"Output path (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    api_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not api_key:
        print("ERROR: CFBD_API_KEY environment variable not set.")
        print("Get a free key at https://collegefootballdata.com/key")
        print("Then: export CFBD_API_KEY=your_key_here")
        return

    years = [int(y.strip()) for y in args.years.split(",") if y.strip().isdigit()]
    if not years:
        print("No valid years specified.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Fetching HS prospects for classes: {years}")
    print(f"Minimum stars: {args.min_stars}\n")

    all_prospects   = []
    seen_names      = set()
    espn_team_names = None

    for year in years:
        batch = fetch_recruits_for_year(year, api_key, args.min_stars)
        if not batch:
            # CFBD hasn't published this class yet — ESPN carries future
            # classes years earlier via its public core API.
            if espn_team_names is None:
                espn_team_names = fetch_espn_team_names()
            batch = fetch_recruits_espn(year, args.min_stars, espn_team_names)
        for p in batch:
            key = (p["name"].lower().strip(), p["year"])
            if key not in seen_names:
                seen_names.add(key)
                all_prospects.append(p)
        time.sleep(0.5)

    # Sort by national ranking within year, then year descending
    all_prospects.sort(key=lambda p: (-(p.get("year") or 0), p.get("ranking") or 9999))

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total":        len(all_prospects),
        "years":        years,
        "min_stars":    args.min_stars,
        "prospects":    all_prospects,
    }

    with open(args.output, "w") as f:
        json.dump(cache, f, separators=(",", ":"))

    print(f"\n✓ Saved {len(all_prospects)} HS prospects to {args.output}")

    star_counts = {}
    for p in all_prospects:
        s = p.get("stars", 0)
        star_counts[s] = star_counts.get(s, 0) + 1
    for s in sorted(star_counts, reverse=True):
        print(f"  {'★' * s}: {star_counts[s]}")

    grade_counts = {}
    for p in all_prospects:
        g = (p.get("grade") or "?")[0]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print()
    for g in sorted(grade_counts):
        print(f"  Grade {g}: {grade_counts[g]}")


if __name__ == "__main__":
    main()
