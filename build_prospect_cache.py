#!/usr/bin/env python3
"""
Build a leaderboard-ready prediction cache for every FBS college prospect.

Fetches rosters from ESPN, calls the local (or remote) /predict API for each player,
and saves results to training_data/prospect_cache.json.

Usage:
    # Against local dev server (start XGBOost.py first):
    python build_prospect_cache.py

    # Against production:
    python build_prospect_cache.py --api-url https://draft.jkrek.com

    # Limit to fewer teams (faster, for testing):
    python build_prospect_cache.py --max-teams 20
"""

import argparse
import json
import os
import re
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

OUTPUT_FILE  = "training_data/prospect_cache.json"
HISTORY_DIR  = "training_data/board_history"
MOVERS_FILE  = "training_data/board_movers.json"
MOVERS_TOP_N = 15

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


def fetch_teams(max_teams=1000):
    """Fetch all college teams from ESPN.

    ESPN ignores groups=80 on this endpoint and interleaves every division —
    a low limit silently drops real FBS programs (a 250 cap once lost Oregon,
    Miami, South Carolina…). Fetch everything; the conference-tier filter at
    predict time discards sub-FBS rosters."""
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
                "espn_id":  player.get("espn_id", ""),
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
            # None (not 0/50) when the server graded on missing data — the UI
            # skips the bar instead of drawing a fake league-average one.
            "production_score":   (round(float(stats["production_score"]), 1)
                                   if stats.get("production_score") is not None else None),
            "combine_speed_score": (round(float(stats["combine_speed_score"]), 1)
                                    if stats.get("combine_speed_score") is not None else None),
            "games_played":       (int(float(stats["games_played"]))
                                   if stats.get("games_played") is not None else None),
            # Player's own ESPN athlete id — passed to /predict to unlock
            # verified stats (espn_live) without a name+team lookup.
            "espn_id":            player.get("espn_id") or "",
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


# ── Board history snapshots + movers ──────────────────────────────────────────

def _snapshot_date_from_iso(iso_str, fallback_path=None):
    """YYYY-MM-DD from an ISO timestamp; falls back to file mtime, then now."""
    try:
        return datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).date().isoformat()
    except (ValueError, TypeError):
        if fallback_path and os.path.exists(fallback_path):
            return datetime.fromtimestamp(
                os.path.getmtime(fallback_path), tz=timezone.utc).date().isoformat()
        return datetime.now(timezone.utc).date().isoformat()


def slim_board(prospects):
    """Slim per-player rows; rank = 1-based index in the grade-sorted list."""
    return [
        {
            "name":                p.get("name"),
            "team":                p.get("team"),
            "position":            p.get("position"),
            "grade":               p.get("grade"),
            "success_probability": p.get("success_probability") or 0,
            "rank":                rank,
        }
        for rank, p in enumerate(prospects, start=1)
    ]


def write_snapshot(prospects, generated_at, date_str):
    """Write training_data/board_history/board_<YYYY-MM-DD>.json."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"board_{date_str}.json")
    with open(path, "w") as f:
        json.dump({"generated_at": generated_at, "players": slim_board(prospects)},
                  f, separators=(",", ":"))
    return path


def load_latest_snapshot():
    """Return the most recent snapshot dict (by date in filename), or None."""
    if not os.path.isdir(HISTORY_DIR):
        return None
    dated = []
    for fn in os.listdir(HISTORY_DIR):
        m = re.fullmatch(r"board_(\d{4}-\d{2}-\d{2})\.json", fn)
        if m:
            dated.append((m.group(1), fn))
    if not dated:
        return None
    dated.sort()
    latest = dated[-1][1]
    try:
        with open(os.path.join(HISTORY_DIR, latest)) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not read prior snapshot {latest}: {e}")
        return None


def bootstrap_history_from_existing_cache():
    """One-time migration: history starts empty, so if an OLD cache already
    exists at OUTPUT_FILE, snapshot it (dated from its generated_at) BEFORE it
    gets overwritten — the very first refreshed board then has deltas."""
    if os.path.isdir(HISTORY_DIR) and any(
            re.fullmatch(r"board_\d{4}-\d{2}-\d{2}\.json", fn)
            for fn in os.listdir(HISTORY_DIR)):
        return  # history already seeded
    if not os.path.exists(OUTPUT_FILE):
        return
    try:
        with open(OUTPUT_FILE) as f:
            old = json.load(f)
    except Exception as e:
        print(f"Bootstrap skipped: existing cache unreadable ({e})")
        return
    prospects = old.get("prospects") or []
    if not prospects:
        return
    generated_at = old.get("generated_at") or ""
    date_str = _snapshot_date_from_iso(generated_at, OUTPUT_FILE)
    path = write_snapshot(prospects, generated_at, date_str)
    print(f"Bootstrapped board history from existing cache → {path}")


def compute_movers(prospects, prior, generated_at):
    """Deltas vs the prior snapshot, shaped for the /api/movers contract.

    Players are matched by name+team (case-insensitive); players absent from
    the prior snapshot get no delta. delta_rank = prior_rank - new_rank, so
    positive means the player climbed the board.
    """
    movers = {"generated_at": generated_at, "since": None,
              "count": 0, "risers": [], "fallers": []}
    prior_players = (prior or {}).get("players") or []
    if not prior_players:
        return movers

    prev = {
        ((q.get("name") or "").strip().lower(), (q.get("team") or "").strip().lower()): q
        for q in prior_players
    }
    all_deltas = []  # every matched player — merged into /api/prospects rows
    scored     = []  # rows shaped for risers/fallers
    for rank, p in enumerate(prospects, start=1):
        key = ((p.get("name") or "").strip().lower(), (p.get("team") or "").strip().lower())
        q = prev.get(key)
        if not q:
            continue
        delta_prob = round(float(p.get("success_probability") or 0)
                           - float(q.get("success_probability") or 0), 2)
        delta_rank = int(q.get("rank") or rank) - rank
        all_deltas.append({"name": p.get("name"), "team": p.get("team"),
                           "delta_prob": delta_prob, "delta_rank": delta_rank})
        if delta_prob != 0:
            scored.append({
                "name":                p.get("name"),
                "team":                p.get("team"),
                "position":            p.get("position"),
                "grade":               p.get("grade"),
                "success_probability": p.get("success_probability") or 0,
                "delta_prob":          delta_prob,
                "delta_rank":          delta_rank,
                "espn_team_id":        p.get("espn_team_id") or "",
            })

    risers  = sorted((m for m in scored if m["delta_prob"] > 0),
                     key=lambda m: -m["delta_prob"])[:MOVERS_TOP_N]
    fallers = sorted((m for m in scored if m["delta_prob"] < 0),
                     key=lambda m: m["delta_prob"])[:MOVERS_TOP_N]
    movers.update({
        "since":   (prior or {}).get("generated_at") or None,
        "count":   len(risers) + len(fallers),
        "risers":  risers,
        "fallers": fallers,
        # Superset of the /api/movers contract: full per-player deltas, used
        # server-side to merge "trend" into /api/prospects rows. The endpoint
        # itself serves only the contract keys above.
        "all_deltas": all_deltas,
    })
    return movers


def write_board_history(all_prospects, generated_at):
    """Snapshot today's board and write board_movers.json vs the prior one."""
    prior = load_latest_snapshot()  # read BEFORE writing today's snapshot —
    # a same-day rerun then computes deltas against the previous run instead
    # of diffing the file against itself.
    date_str  = _snapshot_date_from_iso(generated_at)
    snap_path = write_snapshot(all_prospects, generated_at, date_str)
    movers    = compute_movers(all_prospects, prior, generated_at)
    with open(MOVERS_FILE, "w") as f:
        json.dump(movers, f, separators=(",", ":"))
    print(f"  Snapshot: {snap_path}")
    print(f"  Movers:   {MOVERS_FILE} "
          f"({movers['count']} movers since {movers['since'] or 'n/a — first snapshot'})")


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

    # One-time migration: seed board history from the pre-existing cache BEFORE
    # this run overwrites it, so the first refreshed board already has deltas.
    bootstrap_history_from_existing_cache()

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
    # Local pre-filter: ESPN's list interleaves every division; grading a
    # sub-FBS roster costs ~90 pointless /predict round-trips per team.
    from dv_tiers import classify_college_tier
    before = len(teams)
    teams = [t for t in teams if classify_college_tier(t["name"]) <= MAX_CONFERENCE_TIER]
    print(f"Found {before} teams; {len(teams)} classify as FBS (tier ≤ {MAX_CONFERENCE_TIER})\n")

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

    # Snapshot today's board and compute movers vs the most recent prior snapshot
    write_board_history(all_prospects, cache["generated_at"])

    print(f"  Errors: {errors}")
    grade_counts = {}
    for p in all_prospects:
        g = (p.get("grade") or "?")[0]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    for g in sorted(grade_counts):
        print(f"  Grade {g}: {grade_counts[g]}")


if __name__ == "__main__":
    main()
