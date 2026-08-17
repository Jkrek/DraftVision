#!/usr/bin/env python
"""Fetch (with disk cache) all CFBD data needed for v3 experiments.

Endpoints (all confirmed to work with year-only params via probes):
  /recruiting/players?year=Y   Y=1999..2023  (1996-1998 probed empty)
  /stats/player/season?year=Y  Y=2004..2025  (returns ALL categories per year)
  /ratings/sp?year=Y           Y=1999..2025

Run: .venv/bin/python scripts/experiments/fetch_v3_data.py
"""
import collections
import sys

from cfbd_client import get

def main():
    print("== /recruiting/players ==")
    for y in range(1999, 2024):
        d = get("/recruiting/players", {"year": y})
        print(f"  {y}: {len(d)} recruits")

    print("== /ratings/sp ==")
    for y in range(1999, 2026):
        d = get("/ratings/sp", {"year": y})
        print(f"  {y}: {len(d)} teams")

    print("== /stats/player/season ==")
    for y in range(2004, 2026):
        d = get("/stats/player/season", {"year": y})
        cats = collections.Counter(r["category"] for r in d)
        print(f"  {y}: {len(d)} records, defensive={cats.get('defensive', 0)}, "
              f"passing={cats.get('passing', 0)}, rushing={cats.get('rushing', 0)}, "
              f"receiving={cats.get('receiving', 0)}, interceptions={cats.get('interceptions', 0)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
