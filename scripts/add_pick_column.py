#!/usr/bin/env python3
"""Join nflverse overall-pick numbers into training_data/combine_outcomes.csv.

Adds a `draft_pick` column (1-262 for drafted rows, blank for UDFA) matched on
normalized name + draft year, requiring round agreement so same-name players
in the same class can't cross-contaminate. One-time backfill for the
draft-position regressor; build_training_data.py owns this join going forward.
"""

import csv
import re
import sys
import urllib.request

SRC = "training_data/combine_outcomes.csv"
PICKS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "draft_picks/draft_picks.csv")


def norm(name: str) -> str:
    n = (name or "").lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?$", "", n).strip()
    return re.sub(r"[^a-z ]", "", n)


def main() -> int:
    try:
        with open("/tmp/draft_picks.csv") as f:
            picks = list(csv.DictReader(f))
    except FileNotFoundError:
        req = urllib.request.Request(PICKS_URL, headers={"User-Agent": "DraftVision"})
        with urllib.request.urlopen(req, timeout=120) as r, open("/tmp/draft_picks.csv", "wb") as f:
            f.write(r.read())
        with open("/tmp/draft_picks.csv") as f:
            picks = list(csv.DictReader(f))

    # (name, season, round) -> pick
    lookup = {}
    for p in picks:
        key = (norm(p.get("pfr_player_name")), p.get("season"), p.get("round"))
        if key[0]:
            lookup[key] = p.get("pick")

    with open(SRC) as f:
        rows = list(csv.DictReader(f))
        fields = f.fieldnames if hasattr(f, "fieldnames") else None
    fields = list(rows[0].keys())
    if "draft_pick" not in fields:
        fields.append("draft_pick")

    matched = drafted = 0
    for r in rows:
        rd = r.get("draft_round")
        if rd and rd not in ("8", "8.0"):
            drafted += 1
            pick = lookup.get((norm(r.get("name")), str(int(float(r.get("draft_year")))),
                               str(int(float(rd)))))
            r["draft_pick"] = pick or ""
            matched += bool(pick)
        else:
            r["draft_pick"] = ""

    with open(SRC, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"drafted rows: {drafted}, pick matched: {matched} ({100*matched/max(drafted,1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
