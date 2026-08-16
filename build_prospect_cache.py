#!/usr/bin/env python3
"""
Build a leaderboard-ready prediction cache for every FBS college prospect.

Fetches rosters from ESPN, calls the local (or remote) /predict API for each player,
and saves results to training_data/prospect_cache.json.

Usage:
    # Against local dev server (start XGBOost.py first):
    python build_prospect_cache.py

    # Against production:
    python build_prospect_cache.py --api-url https://jkrek.com

    # Limit to fewer teams (faster, for testing):
    python build_prospect_cache.py --max-teams 20
"""

import argparse
import json
import os
import time
import requests
from datetime import datetime, timezone


def current_season_year(now=None):
    """College football season year: Aug-Dec belong to that year's season,
    Jan-Jul to the previous year's."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def projected_draft_class(class_abbr, season_year):
    """Earliest realistic draft class from roster class standing.

    A season-N player's final possible college season is season_year + (4 - N),
    and the draft follows the season by one calendar year. FR->+4, SO->+3,
    JR->+2, SR/GR->+1. Redshirts and early declares shift this; it is a
    projection, not a certainty.
    """
    years = {"FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 4}.get((class_abbr or "").upper())
    if years is None:
        return None
    return season_year + (4 - years) + 1

OUTPUT_FILE = "training_data/prospect_cache.json"

ESPN_CFB_TEAMS_URL      = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
ESPN_CFB_TEAM_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"

GRADE_ORDER = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C+": 6, "C": 7, "C-": 8, "D": 9}

# A roster player whose /predict resolution comes back with an NFL team collided
# with a same-named pro in the lookup DB — the stats/grade belong to the pro,
# so the row is dropped rather than cached under the wrong identity.
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

# Only include FBS-level conferences (groups=80 gets all FBS)
# Conference tier <= 6 means P4/G5 up to mid-majors; skip FCS (tiers 9-10)
MAX_CONFERENCE_TIER = 8


def espn_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.ok:
                return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ESPN GET failed: {url} — {e}")
        time.sleep(0.5 * (attempt + 1))
    return {}


def fetch_teams(max_teams=250):
    """Fetch all FBS teams from ESPN."""
    data = espn_get(ESPN_CFB_TEAMS_URL, params={"limit": max_teams, "groups": 80})
    teams = []
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for row in league.get("teams", []):
                team = row.get("team", {}) if isinstance(row, dict) else {}
                team_id   = str(team.get("id") or "").strip()
                team_name = str(team.get("displayName") or team.get("name") or "").strip()
                if team_id and team_name:
                    teams.append({"id": team_id, "name": team_name})
    return teams


def iter_athlete_nodes(node):
    """Recursively find athlete-like nodes (same logic as XGBOost.py)."""
    if isinstance(node, dict):
        has_name = any(k in node for k in ("displayName", "fullName", "shortName"))
        has_id   = "id" in node or "$ref" in node
        if has_name and has_id:
            yield node
        for value in node.values():
            yield from iter_athlete_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_athlete_nodes(item)


def fetch_roster(team_id, team_name):
    """Return list of player dicts for a team."""
    data    = espn_get(ESPN_CFB_TEAM_ROSTER_URL.format(team_id=team_id))
    players = []
    seen    = set()

    for athlete in iter_athlete_nodes(data):
        name = str(
            athlete.get("displayName") or athlete.get("fullName") or athlete.get("shortName") or ""
        ).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())

        pos_obj  = athlete.get("position") or {}
        position = str(
            pos_obj.get("abbreviation") or pos_obj.get("name") or "UNK"
        ).upper()

        # Skip non-skill positions that clutter the leaderboard
        if position in {"K", "P", "LS", "UNK", ""}:
            continue

        exp = athlete.get("experience") or {}
        class_abbr = str(exp.get("abbreviation") or "").upper() if isinstance(exp, dict) else ""

        players.append({
            "name":       name,
            "position":   position,
            "espn_id":    str(athlete.get("id") or "").strip(),
            "team":       team_name,
            "team_id":    str(team_id).strip(),
            "class_year": class_abbr,
        })

    return players


def call_predict(player, api_url, timeout=15):
    """POST to /predict and return a lean result dict, or None on failure."""
    try:
        r = requests.post(
            f"{api_url}/predict",
            json={
                "name":     player["name"],
                "position": player.get("position", "Unknown"),
                "team":     player.get("team", "Unknown"),
            },
            timeout=timeout,
        )
        if not r.ok:
            return None
        d = r.json()
        stats = d.get("stats") or {}
        team = str(stats.get("team") or player["team"])
        if team.lower().strip() in NFL_FRANCHISE_NAMES:
            return None
        return {
            "name":               d.get("resolved_name") or player["name"],
            "position":           d.get("predicted_position") or player["position"],
            "team":               team,
            "grade":              d.get("prospect_grade") or "C",
            "success_probability": d.get("success_probability") or 0,
            "draft_grade":        d.get("draft_grade") or "",
            "draft_grade_class":  d.get("draft_grade_class"),
            "conference_tier":    stats.get("conference_tier") or 5,
            "production_score":   round(float(stats.get("production_score") or 0), 1),
            "combine_speed_score": round(float(stats.get("combine_speed_score") or 50), 1),
            "games_played":       int(float(stats.get("games_played") or 0)),
            # ESPN NCAA team id -> logo at a.espncdn.com/i/teamlogos/ncaa/500/{id}.png
            "espn_team_id":       player.get("team_id") or "",
            "class_year":         player.get("class_year") or "",
            "draft_class":        projected_draft_class(
                                      player.get("class_year"),
                                      current_season_year()),
            "is_award_winner":    int(stats.get("is_award_winner") or 0),
            "is_all_american":    int(stats.get("is_all_american") or 0),
            # "espn_live" = graded from real ESPN season stats; anything else
            # means the stat line was estimated — surfaced as a UI badge.
            "data_source":        d.get("data_source") or "unknown",
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Build prospect prediction cache.")
    parser.add_argument("--api-url",   default="http://localhost:5001",
                        help="Base URL of the running prediction API")
    parser.add_argument("--max-teams", type=int, default=250,
                        help="Max number of FBS teams to process (default: 250 = all)")
    parser.add_argument("--delay",     type=float, default=0.25,
                        help="Seconds between /predict calls (default: 0.25)")
    args = parser.parse_args()

    os.makedirs("training_data", exist_ok=True)

    print(f"Using API: {args.api_url}")
    print(f"Output:    {OUTPUT_FILE}\n")

    # Verify API is reachable
    try:
        r = requests.get(f"{args.api_url}/health", timeout=5)
        if not r.ok:
            raise ConnectionError(f"API returned {r.status_code}")
        print(f"API health: {r.json().get('status')}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.api_url} — {e}")
        print("Start the Flask app with: python XGBOost.py")
        return

    print("Fetching FBS teams from ESPN…")
    teams = fetch_teams(args.max_teams)
    print(f"Found {len(teams)} teams\n")

    all_prospects = []
    seen_names    = set()
    errors        = 0

    for idx, team in enumerate(teams):
        roster = fetch_roster(team["id"], team["name"])
        team_prospects = 0
        print(f"[{idx+1:3}/{len(teams)}] {team['name']:<35} {len(roster)} players", end="")

        for player in roster:
            norm = player["name"].lower().strip()
            if norm in seen_names:
                continue
            seen_names.add(norm)

            result = call_predict(player, args.api_url)
            if result:
                # Skip FCS-level tier players (tier 9-10) to keep list P5/G5 focused
                if int(result.get("conference_tier") or 10) <= MAX_CONFERENCE_TIER:
                    all_prospects.append(result)
                    team_prospects += 1
            else:
                errors += 1

            time.sleep(args.delay)

        print(f"  → {team_prospects} cached")
        time.sleep(0.05)

    # Sort by grade then success probability
    all_prospects.sort(key=lambda p: (
        GRADE_ORDER.get(p.get("grade"), 9),
        -(p.get("success_probability") or 0),
    ))

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total":        len(all_prospects),
        "api_url":      args.api_url,
        "prospects":    all_prospects,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(cache, f, separators=(",", ":"))

    print(f"\n✓ Saved {len(all_prospects)} prospects to {OUTPUT_FILE}")
    print(f"  Errors: {errors}")
    grade_counts = {}
    for p in all_prospects:
        g = (p.get("grade") or "?")[0]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    for g in sorted(grade_counts):
        print(f"  Grade {g}: {grade_counts[g]}")


if __name__ == "__main__":
    main()
