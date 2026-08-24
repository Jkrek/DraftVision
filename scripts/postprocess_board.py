#!/usr/bin/env python3
"""Post-rebuild board pass — run after build_prospect_cache.py.

1. Nominal picks: rank each row's raw model_pick within its draft class —
   the class's #1-ordered player is its projected #1 pick.
2. Grade cutoffs: recompute the letter-grade percentile cutoffs from the new
   success-probability distribution (print them; XGBOost.py's
   compute_prospect_grade constants must be updated to match).
3. Relabel every row: prospect grade from the new cutoffs, draft projection
   from the 11-bucket nominal-pick ladder (Generational = nominal <= 2 AND
   top-3 board success probability).
"""

import json
import sys

CACHE = "training_data/prospect_cache.json"

GRADE_PCTS = [("A+", 98), ("A", 95), ("A-", 90), ("B+", 80), ("B", 70),
              ("B-", 55), ("C+", 45), ("C", 35), ("C-", 10)]


def ladder(nominal, sp, gen_cut):
    if nominal is None:
        return None
    pk = float(nominal)
    if pk <= 2 and sp >= gen_cut: return "Generational"
    if pk <= 5:   return "Top 5 Pick"
    if pk <= 10:  return "Top 10 Pick"
    if pk <= 20:  return "Top 20 Pick"
    if pk <= 32:  return "1st Round"
    if pk <= 64:  return "2nd Round"
    if pk <= 105: return "3rd Round"
    if pk <= 145: return "4th Round"
    if pk <= 185: return "5th Round"
    if pk <= 262: return "Round 6–7"
    if pk <= 350: return "Priority UDFA"
    return "Undrafted"


def main() -> int:
    with open(CACHE) as f:
        cache = json.load(f)
    rows = cache["prospects"]

    # 1) nominal picks by class rank of raw model_pick
    by_class = {}
    for p in rows:
        if p.get("model_pick") is not None:
            by_class.setdefault(int(p.get("draft_class") or 0), []).append(float(p["model_pick"]))
    for v in by_class.values():
        v.sort()
    import bisect
    for p in rows:
        mp = p.get("model_pick")
        if mp is None:
            p["projected_pick"] = None
            continue
        arr = by_class.get(int(p.get("draft_class") or 0)) or []
        p["projected_pick"] = (1 + bisect.bisect_left(arr, float(mp))) if len(arr) >= 100 else None

    # 2) grade cutoffs from the new sp distribution
    sps = sorted((float(p.get("success_probability") or 0) for p in rows), reverse=True)
    n = len(sps)
    cuts = [(g, round(sps[min(n - 1, int(n * (100 - pct) / 100))], 1)) for g, pct in GRADE_PCTS]
    print("NEW GRADE CUTOFFS (update compute_prospect_grade in XGBOost.py):")
    for g, v in cuts:
        print(f"    if p >= {v}: return \"{g}\"")

    def grade_of(sp):
        for g, v in cuts:
            if sp >= v:
                return g
        return "D"

    # 3) relabel
    gen_cut = sps[2] if n >= 3 else 1e9
    for p in rows:
        sp = float(p.get("success_probability") or 0)
        p["grade"] = grade_of(sp)
        lab = ladder(p.get("projected_pick"), sp, gen_cut)
        if lab:
            p["draft_grade"] = lab

    with open(CACHE, "w") as f:
        json.dump(cache, f, separators=(",", ":"))

    from collections import Counter
    print("\nprojection distribution:", dict(Counter(p.get("draft_grade") for p in rows).most_common(14)))
    print("generational:", [p["name"] for p in rows if p.get("draft_grade") == "Generational"])
    print("top5:", [p["name"] for p in rows if p.get("draft_grade") == "Top 5 Pick"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
