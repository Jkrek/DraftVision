#!/usr/bin/env python3
"""Top up the prospect cache with teams the last build missed.

The ESPN team-list cap once dropped real FBS programs (Oregon, Miami,
South Carolina, Texas Tech, Cincinnati, SMU…). This grades ONLY rosters
of teams absent from the current cache and merges them in, so a missed
program never requires a full 2.5-hour rebuild.

Usage: .venv/bin/python scripts/topup_missing_teams.py [--api-url http://localhost:5001]
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")
import importlib.util

spec = importlib.util.spec_from_file_location("bpc", "build_prospect_cache.py")
bpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpc)

GRADE_ORDER = bpc.GRADE_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:5001")
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    cache = json.load(open(bpc.OUTPUT_FILE))
    have = {r["team"] for r in cache["prospects"]}
    seen_names = {r["name"].lower().strip() for r in cache["prospects"]}

    teams = bpc.fetch_teams()
    missing = [t for t in teams if t["name"] not in have]
    print(f"cache has {len(have)} teams; ESPN lists {len(teams)}; fetching {len(missing)} missing")

    added, skipped_tier = 0, 0
    for i, team in enumerate(missing):
        roster = bpc.fetch_roster(team["id"], team["name"])
        kept = 0
        for player in roster:
            if player["name"].lower().strip() in seen_names:
                continue
            result = bpc.call_predict(player, args.api_url)
            if result:
                if int(result.get("conference_tier") or 10) <= bpc.MAX_CONFERENCE_TIER:
                    seen_names.add(player["name"].lower().strip())
                    cache["prospects"].append(result)
                    kept += 1
                else:
                    skipped_tier += 1
                    break  # first sub-FBS hit: whole roster is sub-FBS, skip the rest
            time.sleep(args.delay)
        if kept:
            print(f"[{i+1}/{len(missing)}] {team['name']:<35} +{kept}")
            added += kept

    cache["prospects"].sort(key=lambda p: (
        GRADE_ORDER.get(p.get("grade"), 9), -(p.get("success_probability") or 0)))
    cache["total"] = len(cache["prospects"])
    with open(bpc.OUTPUT_FILE, "w") as f:
        json.dump(cache, f, separators=(",", ":"))
    print(f"\n✓ added {added} players; cache now {cache['total']}")


if __name__ == "__main__":
    main()
