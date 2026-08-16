#!/usr/bin/env python3
"""Build empirical production-score percentile tables per position group.

Reads a prospect cache JSON (9k+ FBS players with raw compute_production_score
composites) and writes sorted raw-score arrays per position group to
training_data/production_percentiles.json.

At runtime, XGBOost.compute_production_score converts a raw composite into its
mean-rank percentile within the player's position group:
    pct = (count_less + 0.5 * count_equal) / n * 100
This removes the 100.0 saturation wall and ranks OL against OL (not WRs).

Usage:
    .venv/bin/python scripts/compute_production_percentiles.py [cache_path]

Default cache_path is training_data/prospect_cache.json; pass an explicit
snapshot path (e.g. /tmp/prospect_cache_OLD.json) to avoid a live file that is
being rewritten.
"""

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "training_data" / "production_percentiles.json"
DEFAULT_CACHE = REPO_ROOT / "training_data" / "prospect_cache.json"

# Keep in sync with XGBOost._PRODUCTION_GROUP_BY_POS
GROUP_BY_POS = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL", "OT": "OL", "OG": "OL", "G": "OL", "C": "OL", "LS": "OL",
    "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
}


def position_group(position: str) -> str:
    return GROUP_BY_POS.get((position or "").strip().upper(), "OTHER")


def main() -> int:
    cache_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CACHE
    with open(cache_path) as fh:
        cache = json.load(fh)

    prospects = cache["prospects"] if isinstance(cache, dict) else cache
    groups: dict[str, list[float]] = {}
    skipped = 0
    for p in prospects:
        score = p.get("production_score")
        if score is None:
            skipped += 1
            continue
        score = float(score)
        if not math.isfinite(score):
            skipped += 1
            continue
        groups.setdefault(position_group(p.get("position")), []).append(round(score, 2))

    for vals in groups.values():
        vals.sort()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(cache_path),
        "method": (
            "sorted raw compute_production_score composites per position group; "
            "runtime maps raw -> mean-rank percentile (count_less + 0.5*ties)/n*100"
        ),
        "groups": groups,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(payload, fh)

    print(f"Read {len(prospects)} prospects from {cache_path} ({skipped} skipped)")
    print(f"Wrote {OUTPUT_PATH}")
    for group in sorted(groups):
        vals = groups[group]
        ties_at_max = sum(1 for v in vals if v == vals[-1])
        print(
            f"  {group:6s} n={len(vals):5d}  min={vals[0]:7.2f}  "
            f"p50={vals[len(vals) // 2]:7.2f}  max={vals[-1]:7.2f}  "
            f"ties_at_max={ties_at_max}"
        )
    # Distinct positions seen, for auditability
    pos_counts = Counter((p.get("position") or "?").upper() for p in prospects)
    print(f"  positions: {dict(sorted(pos_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
