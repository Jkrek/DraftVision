#!/usr/bin/env python
"""Stage CFBD extras: transfer portal, player usage, player season PPA.

Uses the existing cached CFBD client (scripts/experiments/cfbd_client.py):
CFBD_API_KEY is read from .env at runtime only; every response is cached under
/tmp/dv_training_cache/cfbd/, so re-runs cost zero API calls.

Endpoints (verified against the live API):
  /player/portal        — transfer portal entries (data exists 2021+; earlier
                          years are probed and skipped if empty/unsupported)
  /player/usage         — share of team plays per player, 2015-2025
  /ppa/players/season   — opponent-adjusted Predicted Points Added, 2015-2025

Outputs (all keyed by player + school/team + season):
  training_data/staging/cfbd_transfer_portal.csv
    season, player, position, origin, destination, transfer_date, stars,
    rating, eligibility
  training_data/staging/cfbd_usage.csv
    season, player, school, position, conference, usage_overall, usage_pass,
    usage_rush, usage_first_down, usage_second_down, usage_third_down,
    usage_standard_downs, usage_passing_downs
  training_data/staging/cfbd_ppa.csv
    season, player, school, position, conference, avg_ppa_all, avg_ppa_pass,
    avg_ppa_rush, avg_ppa_first_down, avg_ppa_second_down, avg_ppa_third_down,
    avg_ppa_standard_downs, avg_ppa_passing_downs, total_ppa_all

License: CFBD free-tier terms (attribution; usage within key tier limits).

Run: .venv/bin/python scripts/data/fetch_cfbd_extras.py
"""

import os
import sys

import pandas as pd

from _common import SCRIPTS_DIR, write_csv

sys.path.insert(0, os.path.join(SCRIPTS_DIR, "experiments"))
from cfbd_client import get as cfbd_get  # noqa: E402

PORTAL_YEARS = range(2018, 2027)   # probed; empty/unsupported years skipped
USAGE_PPA_YEARS = range(2015, 2026)

_PPA_KEYS = ["all", "pass", "rush", "firstDown", "secondDown", "thirdDown",
             "standardDowns", "passingDowns"]
_SNAKE = {"firstDown": "first_down", "secondDown": "second_down",
          "thirdDown": "third_down", "standardDowns": "standard_downs",
          "passingDowns": "passing_downs", "all": "all", "pass": "pass",
          "rush": "rush"}


def stage_portal() -> None:
    rows = []
    for y in PORTAL_YEARS:
        try:
            data = cfbd_get("/player/portal", {"year": y})
        except RuntimeError as e:
            print(f"  portal {y}: unavailable ({str(e)[:80]}), skipped")
            continue
        if not data:
            print(f"  portal {y}: 0 entries, skipped")
            continue
        for r in data:
            rows.append({
                "season": r.get("season") or y,
                "player": f"{(r.get('firstName') or '').strip()} "
                          f"{(r.get('lastName') or '').strip()}".strip(),
                "position": r.get("position"),
                "origin": r.get("origin"),
                "destination": r.get("destination"),
                "transfer_date": r.get("transferDate"),
                "stars": r.get("stars"),
                "rating": r.get("rating"),
                "eligibility": r.get("eligibility"),
            })
        print(f"  portal {y}: {len(data)} entries")
    write_csv(pd.DataFrame(rows), "cfbd_transfer_portal.csv")


def _flatten(rows, y, block_key, prefix):
    out = []
    for r in rows:
        rec = {
            "season": r.get("season") or y,
            "player": r.get("name"),
            "school": r.get("team"),
            "position": r.get("position"),
            "conference": r.get("conference"),
        }
        block = r.get(block_key) or {}
        for k in _PPA_KEYS:
            rec[f"{prefix}_{_SNAKE[k]}"] = block.get(k)
        out.append(rec)
    return out


def stage_usage() -> None:
    rows = []
    for y in USAGE_PPA_YEARS:
        data = cfbd_get("/player/usage", {"year": y})
        rows.extend(_flatten(data, y, "usage", "usage"))
        print(f"  usage {y}: {len(data)} players")
    write_csv(pd.DataFrame(rows), "cfbd_usage.csv")


def stage_ppa() -> None:
    rows = []
    for y in USAGE_PPA_YEARS:
        data = cfbd_get("/ppa/players/season", {"year": y})
        flat = _flatten(data, y, "averagePPA", "avg_ppa")
        for rec, r in zip(flat, data):
            rec["total_ppa_all"] = (r.get("totalPPA") or {}).get("all")
        rows.extend(flat)
        print(f"  ppa {y}: {len(data)} players")
    write_csv(pd.DataFrame(rows), "cfbd_ppa.csv")


if __name__ == "__main__":
    stage_portal()
    stage_usage()
    stage_ppa()
