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

    # Concurrent grading (CI / local server only — per-call HTTP is I/O bound
    # on the server's own ESPN fetches, so threads give near-linear speedup):
    python build_prospect_cache.py --workers 6 --delay 0

    # Hermetic smoke run — writes cache/history/movers under /tmp, never
    # touching training_data/:
    python build_prospect_cache.py --max-teams 3 --workers 4 \
        --output /tmp/dv_smoke/prospect_cache.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor
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
# Resolved against this file, not the cwd — scripts/ callers and cron once
# silently produced model_git_sha='unknown' because the relative path missed.
MODEL_METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "metadata.json")

# Ledger row schema (see slim_board). REQUIRED keys are always present;
# OPTIONAL keys are OMITTED from a row when their value is null so a season
# of weekly snapshots stays small — readers must treat absence as null.
SNAPSHOT_ROW_REQUIRED = ("name", "team", "position", "grade",
                         "success_probability", "rank", "data_source")
SNAPSHOT_ROW_OPTIONAL = ("espn_id", "draft_class", "projected_pick", "model_pick",
                         "pick_lo", "pick_hi", "draft_grade",
                         "projected_career_av", "games_played")

# Mover gate: a week-over-week delta is only scorable when the CURRENT row was
# graded from verified ESPN season stats with enough games for the stat line
# to have moved — roster-hint / carried-over lines produce noise, not news.
MOVER_MIN_GAMES     = 2
MOVER_DATA_SOURCE   = "espn_live"

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
            "projected_pick":     d.get("projected_pick"),
            "model_pick":         d.get("model_pick"),
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
            # Conformal pick interval {lo, hi} and career-value head — carried
            # onto the row so the weekly ledger can score them later.
            "pick_range":         (d.get("pick_range")
                                   if isinstance(d.get("pick_range"), dict) else None),
            # Same interval in RAW model-pick space (floats, pre nominal
            # ranking). The served pick_range lo/hi were ranked against the
            # board the server had loaded — i.e. the PREVIOUS week's — so
            # scripts/postprocess_board.py re-ranks these against the rebuilt
            # board and writes pick_lo / pick_hi from them.
            "pick_range_raw":     (d.get("pick_range_raw")
                                   if isinstance(d.get("pick_range_raw"), dict) else None),
            "projected_career_av": d.get("projected_career_av"),
            # "espn_live" = graded from real ESPN season stats; anything else
            # means the stat line was estimated — surfaced as a UI badge.
            "data_source":        d.get("data_source") or "unknown",
        }
    except Exception as e:
        return None


# ── Concurrency ───────────────────────────────────────────────────────────────

class RateLimiter:
    """Global requests/sec ceiling shared by every worker thread.

    Evenly spaces request STARTS 1/rate seconds apart (a one-token bucket).
    The /predict server fans each call out to ESPN, so this — not per-thread
    sleeps — is what keeps ESPN traffic polite when --workers > 1.
    """

    def __init__(self, rate_per_sec):
        self.interval = (1.0 / rate_per_sec) if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_start = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.interval
        wait = start - now
        if wait > 0:
            time.sleep(wait)


def grade_players(players, api_url, workers, delay, limiter):
    """Grade an already-deduped list of roster players → (results, errors).

    workers <= 1 reproduces the original serial loop exactly (per-call
    --delay sleep, no rate limiter). workers > 1 fans the /predict calls out
    over a thread pool with the shared RateLimiter enforcing the global
    req/s ceiling. Result order is irrelevant — the board is fully re-sorted
    before writing — but pool.map keeps it stable anyway.
    """
    results, errors = [], 0

    if workers <= 1:
        for player in players:
            result = call_predict(player, api_url)
            if result:
                results.append(result)
            else:
                errors += 1
            time.sleep(delay)
        return results, errors

    def task(player):
        if limiter:
            limiter.acquire()
        return call_predict(player, api_url)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(task, players):
            if result:
                results.append(result)
            else:
                errors += 1
    return results, errors


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


def row_pick_bounds(p):
    """(pick_lo, pick_hi) for a ledger row.

    Prefers the bounds scripts/postprocess_board.py writes onto the row
    (pick_range_raw re-ranked against the REBUILT board); falls back to the
    served nominal pick_range, whose lo/hi were ranked against whatever board
    the server had loaded at grading time — the previous week's."""
    lo, hi = p.get("pick_lo"), p.get("pick_hi")
    if lo is not None and hi is not None:
        return lo, hi
    pr = p.get("pick_range")
    pr = pr if isinstance(pr, dict) else {}
    return pr.get("lo"), pr.get("hi")


def slim_board(prospects):
    """Slim per-player ledger rows; rank = 1-based index in the grade-sorted
    list. Carries the scorable projections (pick, interval, career AV, ladder
    label) plus the evidence basis (games_played / data_source) — no stat
    blobs, and null-valued OPTIONAL keys are omitted (SNAPSHOT_ROW_OPTIONAL),
    so a season of weekly snapshots stays a few MB."""
    rows = []
    for rank, p in enumerate(prospects, start=1):
        lo, hi = row_pick_bounds(p)
        row = {
            "name":                p.get("name"),
            "team":                p.get("team"),
            "position":            p.get("position"),
            "grade":               p.get("grade"),
            "success_probability": p.get("success_probability") or 0,
            "rank":                rank,
            "data_source":         p.get("data_source") or "unknown",
            "espn_id":             p.get("espn_id") or None,
            "draft_class":         p.get("draft_class"),
            "projected_pick":      p.get("projected_pick"),
            "model_pick":          p.get("model_pick"),
            "pick_lo":             lo,
            "pick_hi":             hi,
            "draft_grade":         p.get("draft_grade") or None,
            "projected_career_av": p.get("projected_career_av"),
            "games_played":        p.get("games_played"),
        }
        rows.append({k: v for k, v in row.items()
                     if v is not None or k in SNAPSHOT_ROW_REQUIRED})
    return rows


def _model_provenance():
    """{model_git_sha, features_n, metadata_sha256} from models/metadata.json.

    metadata.json's git_sha is the HEAD scripts/train_models.py saw at fit
    time — the commit the models were trained FROM, which is normally one
    commit BEFORE the one that ships the model files. metadata_sha256 (digest
    of the file itself) is therefore the exact model-identity key; treat
    model_git_sha as approximate provenance. Values are 'unknown' / None when
    the file is missing or unreadable (resolved via MODEL_METADATA_FILE, which
    is anchored to this file's directory, not the cwd)."""
    out = {"model_git_sha": "unknown", "features_n": None,
           "metadata_sha256": _sha256_file(MODEL_METADATA_FILE)}
    try:
        with open(MODEL_METADATA_FILE) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  Warning: models/metadata.json unreadable ({e}) — "
              f"snapshot provenance will be 'unknown'")
        return out
    feats = meta.get("features")
    out["model_git_sha"] = meta.get("git_sha") or "unknown"
    out["features_n"] = len(feats) if isinstance(feats, list) else None
    return out


def _sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def snapshot_sidecar_path(json_path):
    """board_<date>.json -> board_<date>.sha256 (sha256sum-style line)."""
    return re.sub(r"\.json$", ".sha256", json_path)


def write_snapshot(prospects, generated_at, date_str, cache_path=None):
    """Write training_data/board_history/board_<YYYY-MM-DD>.json plus a
    sibling board_<date>.sha256 holding the json's digest.

    Top-level provenance: metadata_sha256 (digest of models/metadata.json —
    the exact model identity), model_git_sha / features_n from that file
    (approximate: the commit the models were trained FROM, usually one before
    the commit that shipped them — see _model_provenance), and cache_sha256
    (digest of the prospect cache the rows came from), so a later scoring
    pass can tie every ledger row to a specific model + board.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"board_{date_str}.json")
    cache_path = cache_path or OUTPUT_FILE
    prov = _model_provenance()
    doc = {
        "generated_at":    generated_at,
        "snapshot_date":   date_str,
        "model_git_sha":   prov["model_git_sha"],
        "metadata_sha256": prov["metadata_sha256"],
        "features_n":      prov["features_n"],
        "cache_sha256":    _sha256_file(cache_path),
        "players":         slim_board(prospects),
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    digest = _sha256_file(path)
    with open(snapshot_sidecar_path(path), "w") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")
    return path


def load_latest_snapshot(before=None):
    """Return the most recent snapshot dict (by date in filename), or None.

    before: optional 'YYYY-MM-DD' — only snapshots dated strictly earlier are
    considered, so a post-build rewrite of today's file can still diff
    against last week's board instead of itself.
    """
    if not os.path.isdir(HISTORY_DIR):
        return None
    dated = []
    for fn in os.listdir(HISTORY_DIR):
        m = re.fullmatch(r"board_(\d{4}-\d{2}-\d{2})\.json", fn)
        if m and (before is None or m.group(1) < before):
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


def mover_games(p):
    """games_played as an int, or None when missing/unparseable."""
    try:
        return int(float(p.get("games_played")))
    except (TypeError, ValueError):
        return None


def first_verified_line(p, prior=None):
    """True when this week's row is the player's FIRST verified line: the
    current row is espn_live but the prior-snapshot row carried some other
    data_source (roster_hint, nfl_seed, ...). That delta measures the switch
    from an estimated stat line to a real one, not a change in play, so it
    is not a mover. Snapshots before 2026-09-08 carry no data_source — a
    missing prior value is treated as a pass."""
    if not prior:
        return False
    prev_src = prior.get("data_source")
    if prev_src in (None, ""):
        return False
    return (prev_src != MOVER_DATA_SOURCE
            and (p.get("data_source") or "") == MOVER_DATA_SOURCE)


def mover_scorable(p, prior=None):
    """Gate for risers/fallers: verified ESPN stat line with >= MOVER_MIN_GAMES
    on the CURRENT row, and — when the prior row records a data_source — the
    prior line was verified too (see first_verified_line)."""
    gp = mover_games(p)
    return (gp is not None and gp >= MOVER_MIN_GAMES
            and (p.get("data_source") or "") == MOVER_DATA_SOURCE
            and not first_verified_line(p, prior))


def mover_basis(p, prior=None):
    """Human-readable evidence basis for a mover row ('5 games',
    'prior-season carried', 'first verified line', ...) — shown inline in
    the movers UI."""
    if (p.get("data_source") or "") != MOVER_DATA_SOURCE:
        return "prior-season carried"
    if first_verified_line(p, prior):
        return "first verified line"
    gp = mover_games(p)
    if gp is None:
        return "no games logged"
    return f"{gp} game{'s' if gp != 1 else ''}"


def compute_movers(prospects, prior, generated_at):
    """Deltas vs the prior snapshot, shaped for the /api/movers contract.

    Players are matched by name+team (case-insensitive); players absent from
    the prior snapshot get no delta. delta_rank = prior_rank - new_rank, so
    positive means the player climbed the board.

    risers/fallers are GATED to scorable rows (see mover_scorable — the gate
    also looks at the PRIOR row's data_source, so a roster_hint -> espn_live
    flip is not a mover) and carry games_played / data_source /
    prior_data_source / basis; all_deltas stays ungated because it only feeds
    the per-row "trend" chip on /api/prospects.
    """
    movers = {"generated_at": generated_at, "since": None,
              "count": 0, "risers": [], "fallers": [],
              "gate": {"min_games": MOVER_MIN_GAMES,
                       "data_source": MOVER_DATA_SOURCE,
                       "excluded": 0,
                       # subset of excluded: current espn_live, prior not
                       "excluded_first_verified": 0}}
    prior_players = (prior or {}).get("players") or []
    if not prior_players:
        return movers

    prev = {
        ((q.get("name") or "").strip().lower(), (q.get("team") or "").strip().lower()): q
        for q in prior_players
    }
    all_deltas = []  # every matched player — merged into /api/prospects rows
    scored     = []  # rows shaped for risers/fallers
    gated      = 0   # non-zero deltas dropped by the scorability gate
    first_ver  = 0   # ...of which: first verified line (prior row unverified)
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
        if delta_prob == 0:
            continue
        if not mover_scorable(p, q):
            gated += 1
            first_ver += int(first_verified_line(p, q))
            continue
        scored.append({
            "name":                p.get("name"),
            "team":                p.get("team"),
            "position":            p.get("position"),
            "grade":               p.get("grade"),
            "success_probability": p.get("success_probability") or 0,
            "delta_prob":          delta_prob,
            "delta_rank":          delta_rank,
            "espn_team_id":        p.get("espn_team_id") or "",
            "games_played":        mover_games(p),
            "data_source":         p.get("data_source") or "unknown",
            # None when the prior snapshot predates the field (pre 2026-09-08)
            "prior_data_source":   q.get("data_source"),
            "basis":               mover_basis(p, q),
        })

    risers  = sorted((m for m in scored if m["delta_prob"] > 0),
                     key=lambda m: -m["delta_prob"])[:MOVERS_TOP_N]
    fallers = sorted((m for m in scored if m["delta_prob"] < 0),
                     key=lambda m: m["delta_prob"])[:MOVERS_TOP_N]
    movers["gate"]["excluded"] = gated
    movers["gate"]["excluded_first_verified"] = first_ver
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


def write_board_history(all_prospects, generated_at, rewrite=False, cache_path=None):
    """Snapshot today's board and write board_movers.json vs the prior one.

    rewrite=True is the post-process path (scripts/postprocess_board.py):
    today's snapshot already exists from the build, so it is REWRITTEN with
    the relabelled rows and movers are recomputed against the latest snapshot
    dated strictly before today — never against the file being replaced.
    """
    date_str = _snapshot_date_from_iso(generated_at)
    # Read the prior BEFORE writing today's snapshot — a same-day rerun then
    # computes deltas against the previous run instead of diffing the file
    # against itself.
    prior     = load_latest_snapshot(before=date_str if rewrite else None)
    snap_path = write_snapshot(all_prospects, generated_at, date_str, cache_path)
    movers    = compute_movers(all_prospects, prior, generated_at)
    with open(MOVERS_FILE, "w") as f:
        json.dump(movers, f, separators=(",", ":"))
    print(f"  Snapshot: {snap_path}{' (rewritten post-process)' if rewrite else ''}")
    print(f"  Movers:   {MOVERS_FILE} "
          f"({movers['count']} movers since {movers['since'] or 'n/a — first snapshot'}; "
          f"{movers['gate']['excluded']} non-zero deltas gated out: "
          f"< {MOVER_MIN_GAMES} games, not {MOVER_DATA_SOURCE}, or "
          f"{movers['gate']['excluded_first_verified']} first verified line)")
    return movers


def main():
    global OUTPUT_FILE, HISTORY_DIR, MOVERS_FILE

    parser = argparse.ArgumentParser(description="Build prospect prediction cache.")
    parser.add_argument("--api-url",   default="http://localhost:5001",
                        help="Base URL of the running prediction API")
    parser.add_argument("--max-teams", type=int, default=1000,
                        help="ESPN team-list fetch cap (default 1000 — the list "
                             "interleaves every division; 250 silently dropped "
                             "~90 FBS programs TWICE)")
    parser.add_argument("--delay",     type=float, default=0.25,
                        help="Seconds between /predict calls in serial mode "
                             "(default: 0.25; ignored when --workers > 1 — the "
                             "--rps ceiling governs pacing there)")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Concurrent /predict calls per roster (default 1 "
                             "= exact legacy serial behavior). Per-call HTTP "
                             "is I/O bound on the server's ESPN fetches, so "
                             "4-8 gives near-linear speedup against localhost.")
    parser.add_argument("--rps",       type=float, default=10.0,
                        help="Global /predict requests-per-second ceiling when "
                             "--workers > 1 (default 10 — keeps the server's "
                             "downstream ESPN traffic polite). 0 = no ceiling.")
    parser.add_argument("--output",    default=OUTPUT_FILE,
                        help=f"Cache output path (default {OUTPUT_FILE}). When "
                             "overridden, board_history/ and board_movers.json "
                             "are written NEXT TO the override, so smoke runs "
                             "never touch the real board files.")
    args = parser.parse_args()

    # Output override: redirect the snapshot/movers side-effects alongside it
    # so a test run is fully hermetic (never clobbers the real history/movers).
    out_abs = os.path.abspath(args.output)
    if out_abs != os.path.abspath(OUTPUT_FILE):
        OUTPUT_FILE = args.output
        out_dir     = os.path.dirname(out_abs) or "."
        HISTORY_DIR = os.path.join(out_dir, "board_history")
        MOVERS_FILE = os.path.join(out_dir, "board_movers.json")
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

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
    limiter       = RateLimiter(args.rps) if args.workers > 1 else None

    for idx, team in enumerate(teams):
        roster = fetch_roster(team["id"], team["name"])
        print(f"[{idx+1:3}/{len(teams)}] {team['name']:<35} {len(roster)} players", end="")

        # Dedupe by name+team BEFORE fan-out: a bare-name key once skipped
        # South Carolina's Dylan Stewart because Delaware also has one.
        todo = []
        for player in roster:
            norm = f"{player['name'].lower().strip()}|{player['team'].lower().strip()}"
            if norm in seen_names:
                continue
            seen_names.add(norm)
            todo.append(player)

        results, team_errors = grade_players(
            todo, args.api_url, args.workers, args.delay, limiter)
        errors += team_errors

        team_prospects = 0
        for result in results:
            # Skip FCS-level tier players (tier 9-10) to keep list P5/G5 focused
            if int(result.get("conference_tier") or 10) <= MAX_CONFERENCE_TIER:
                all_prospects.append(result)
                team_prospects += 1

        print(f"  → {team_prospects} cached")
        time.sleep(0.05)

    # Identity normalization: /predict can reassign a player's team (transfers,
    # or a same-name player elsewhere resolving to this identity). Re-key
    # espn_team_id to the RESOLVED team and drop hijack-duplicates, preferring
    # rows whose roster origin matches their resolved team (Jeremiah Smith of
    # Louisiana Tech once shipped wearing Ohio State's identity with a LA Tech
    # logo id).
    team_id_by_name = {t["name"]: t["id"] for t in teams}
    all_prospects.sort(
        key=lambda p: 0 if team_id_by_name.get(p["team"]) == p.get("espn_team_id") else 1)
    seen_final, cleaned = set(), []
    for p in all_prospects:
        key = (p["name"].lower().strip(), p["team"].lower().strip())
        if key in seen_final:
            continue
        seen_final.add(key)
        want = team_id_by_name.get(p["team"])
        if want:
            p["espn_team_id"] = want
        cleaned.append(p)
    if len(cleaned) != len(all_prospects):
        print(f"  identity cleanup: dropped {len(all_prospects) - len(cleaned)} hijack-duplicates")
    all_prospects = cleaned

    # Sort by grade then success probability
    all_prospects.sort(key=lambda p: (
        GRADE_ORDER.get(p.get("grade"), 9),
        -(p.get("success_probability") or 0),
    ))

    # Safety valve: a flaky network (DNS drops, ESPN timeouts) must never
    # replace a healthy board with a gutted one. On 2026-08-25 a mid-outage
    # run overwrote 12,862 rows with 0. Abort below half the existing size.
    try:
        with open(OUTPUT_FILE) as f:
            existing_total = int(json.load(f).get("total") or 0)
    except Exception:
        existing_total = 0
    if existing_total and len(all_prospects) < existing_total * 0.5:
        print(f"\n✗ ABORT: new board has {len(all_prospects)} rows vs {existing_total} "
              f"existing (<50%) — refusing to overwrite. Existing cache kept.")
        sys.exit(1)

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
