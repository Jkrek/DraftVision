#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strength-of-schedule enrichment from CFBD.

For each season 2000-2025 (2 API calls per season):
  GET /ratings/sp?year={season}                 -> per-team SP+ overall rating
  GET /games?year={season}&seasonType=regular   -> all regular-season games

Per team-season:
  sos_sp            = mean SP+ overall rating of that team's opponents
                      (only opponents that themselves have an SP+ rating that
                      season; unrated FCS opponents are skipped)
  n_rated_opponents = how many opponents contributed to the mean

Writes training_data/enrich_sos.csv with columns:
  team, season, sos_sp, n_rated_opponents

Then prints a join preview against training_data/combine_outcomes.csv:
(college, draft_year-1) vs (team, season), names normalized strip+casefold.

HARD BUDGET: max 60 CFBD calls.  The first response's x-calllimit-remaining
header is checked; if below 150 the script aborts immediately (the weekly
board refresh needs the remaining quota).

One extra budgeted call probes /ratings/sp?year=2026 to see whether
current-season SP+ is already published (serve-time question).

Usage:  .venv/bin/python scripts/add_sos_column.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import certifi

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import pandas as pd

import train_models as tm  # reuse TRAINING_DATA_PATH (do not hardcode)

API_BASE = "https://api.collegefootballdata.com"
OUTPUT_PATH = os.path.join(REPO_ROOT, "training_data", "enrich_sos.csv")
SEASONS = list(range(2000, 2026))
MAX_CALLS = 60
MIN_REMAINING = 150

_call_count = 0
_last_remaining: str | None = None


def load_key() -> str:
    with open(os.path.join(REPO_ROOT, ".env")) as fh:
        for line in fh:
            if line.startswith("CFBD_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("CFBD_API_KEY not found in .env")


def cfbd_get(path: str, params: dict, key: str):
    """One budgeted CFBD call. Returns parsed JSON. Tracks x-calllimit-remaining."""
    global _call_count, _last_remaining
    if _call_count >= MAX_CALLS:
        raise SystemExit(f"HARD BUDGET of {MAX_CALLS} calls reached — aborting.")
    _call_count += 1
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}",
                                              "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as resp:
        _last_remaining = resp.headers.get("x-calllimit-remaining")
        data = json.loads(resp.read().decode("utf-8"))
    if _call_count == 1:
        print(f"[budget] first call x-calllimit-remaining = {_last_remaining}")
        if _last_remaining is not None:
            try:
                if int(_last_remaining) < MIN_REMAINING:
                    raise SystemExit(
                        f"STOP: x-calllimit-remaining={_last_remaining} < "
                        f"{MIN_REMAINING}; not burning quota needed by the "
                        f"weekly board refresh.")
            except ValueError:
                print(f"[budget] unparseable remaining header: {_last_remaining!r}")
    time.sleep(0.25)
    return data


def _team_of(game: dict, side: str) -> str | None:
    """Handle both camelCase (current CFBD) and snake_case (legacy) keys."""
    return game.get(f"{side}Team") or game.get(f"{side}_team")


def main() -> None:
    key = load_key()

    rows = []
    absent_seasons = []
    for season in SEASONS:
        sp = cfbd_get("/ratings/sp", {"year": season}, key)
        ratings = {}
        for r in sp or []:
            team = r.get("team")
            rating = r.get("rating")
            if not team or team == "nationalAverages" or rating is None:
                continue
            ratings[team.strip().casefold()] = float(rating)

        games = cfbd_get("/games", {"year": season, "seasonType": "regular"}, key)

        if not ratings:
            absent_seasons.append(season)
            print(f"[{season}] SP+ ABSENT ({len(games or [])} games fetched, "
                  f"no ratings) — season skipped")
            continue

        opponents: dict[str, list[str]] = defaultdict(list)
        for g in games or []:
            home, away = _team_of(g, "home"), _team_of(g, "away")
            if not home or not away:
                continue
            opponents[home].append(away)
            opponents[away].append(home)

        n_teams = 0
        for team, opps in sorted(opponents.items()):
            rated = [ratings[o.strip().casefold()] for o in opps
                     if o.strip().casefold() in ratings]
            if not rated:
                continue
            rows.append({"team": team, "season": season,
                         "sos_sp": round(sum(rated) / len(rated), 4),
                         "n_rated_opponents": len(rated)})
            n_teams += 1
        print(f"[{season}] {len(ratings)} SP+ teams, {len(games or [])} games, "
              f"{n_teams} team-seasons with sos_sp")

    out = pd.DataFrame(rows, columns=["team", "season", "sos_sp",
                                      "n_rated_opponents"])
    out.to_csv(OUTPUT_PATH, index=False)
    covered = sorted(out["season"].unique().tolist())
    print(f"\nWrote {OUTPUT_PATH}: {len(out)} team-season rows")
    print(f"SP+ coverage window: {covered[0]}-{covered[-1]}"
          f" | absent seasons: {absent_seasons or 'none'}")

    # ---- serve-time probe: is current-season (2026) SP+ published? ----
    try:
        sp26 = cfbd_get("/ratings/sp", {"year": 2026}, key)
        n26 = sum(1 for r in (sp26 or [])
                  if r.get("team") and r.get("team") != "nationalAverages"
                  and r.get("rating") is not None)
        print(f"[2026 probe] /ratings/sp?year=2026 -> {n26} rated teams")
    except Exception as exc:  # a failed probe must not kill the join preview
        print(f"[2026 probe] failed: {exc}")

    print(f"[budget] total CFBD calls used: {_call_count} "
          f"(x-calllimit-remaining now {_last_remaining})")

    # ---- join preview against the training CSV ----
    train = pd.read_csv(tm.TRAINING_DATA_PATH)
    enrich_keys = {(t.strip().casefold(), int(s))
                   for t, s in zip(out["team"], out["season"])}
    cov_start = covered[0]

    def norm(c) -> str:
        return str(c).strip().casefold()

    window = train[train["draft_year"] >= cov_start + 1].copy()
    window["_key"] = [(norm(c), int(y) - 1)
                      for c, y in zip(window["college"], window["draft_year"])]
    window["_hit"] = [k in enrich_keys for k in window["_key"]]

    n_win, n_hit = len(window), int(window["_hit"].sum())
    print(f"\nJoin preview (training rows with draft_year >= {cov_start + 1}):")
    print(f"  matched {n_hit}/{n_win} = {100.0 * n_hit / n_win:.1f}% "
          f"of rows get a sos_sp")

    misses = Counter(str(c).strip()
                     for c, hit in zip(window["college"], window["_hit"])
                     if not hit)
    print("  top 15 unmatched college names by row count:")
    for name, cnt in misses.most_common(15):
        print(f"    {cnt:4d}  {name}")


if __name__ == "__main__":
    main()
