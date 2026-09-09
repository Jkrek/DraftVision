#!/usr/bin/env python3
"""Post-rebuild board pass — run after build_prospect_cache.py.

1. Nominal picks: rank each row's raw model_pick within its draft class —
   the class's #1-ordered player is its projected #1 pick. The conformal
   interval gets the SAME treatment: pick_range_raw {lo, hi} (raw model-pick
   floats from /predict) is re-ranked against the rebuilt board and written
   to pick_lo / pick_hi. The served pick_range lo/hi were ranked against the
   board the server had loaded — the previous week's — and are used only as
   a fallback when pick_range_raw is absent (caches built before the field).
2. Grade cutoffs: recompute the letter-grade percentile cutoffs from the new
   success-probability distribution (print them; XGBOost.py's
   compute_prospect_grade constants must be updated to match).
3. Relabel every row: prospect grade from the new cutoffs, draft projection
   from the 11-bucket nominal-pick ladder (Generational = nominal <= 2 AND
   top-3 board success probability).
4. Rewrite today's board_history snapshot + board_movers.json from the
   relabelled rows (build_prospect_cache.write_board_history, rewrite=True)
   so the weekly ledger scores post-processed grades/picks, and movers are
   diffed against LAST week's snapshot rather than the pre-relabel file.
"""

import bisect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from build_prospect_cache import write_board_history  # noqa: E402

CACHE = "training_data/prospect_cache.json"

GRADE_PCTS = [("A+", 98), ("A", 95), ("A-", 90), ("B+", 80), ("B", 70),
              ("B-", 55), ("C+", 45), ("C", 35), ("C-", 10)]


# A class needs this many graded rows before its rank order is a pick number.
MIN_CLASS_ROWS = 100


def class_pick_arrays(rows):
    """draft_class -> sorted raw model_pick floats (the per-class rank index)."""
    by_class = {}
    for p in rows:
        if p.get("model_pick") is not None:
            by_class.setdefault(int(p.get("draft_class") or 0), []).append(float(p["model_pick"]))
    for v in by_class.values():
        v.sort()
    return by_class


def nominal_rank(arr, raw):
    """Raw model pick -> 1-based rank within its class, or None when the
    class is too thin to rank (mirrors XGBOost.nominal_draft_pick)."""
    if raw is None or not arr or len(arr) < MIN_CLASS_ROWS:
        return None
    return 1 + bisect.bisect_left(arr, float(raw))


def apply_nominal_picks(rows):
    """Write projected_pick and pick_lo / pick_hi onto every row from the
    per-class rank index of the rows themselves (the REBUILT board).

    pick_lo / pick_hi come from pick_range_raw re-ranked through the same
    index; when raw is absent or the class is unrankable they fall back to
    the served nominal pick_range (ranked against the previous board), and
    are left unset when neither exists — slim_board then omits them.
    Returns the class index for callers that want to inspect it."""
    by_class = class_pick_arrays(rows)
    for p in rows:
        arr = by_class.get(int(p.get("draft_class") or 0)) or []
        p["projected_pick"] = nominal_rank(arr, p.get("model_pick"))

        raw = p.get("pick_range_raw")
        raw = raw if isinstance(raw, dict) else {}
        lo, hi = nominal_rank(arr, raw.get("lo")), nominal_rank(arr, raw.get("hi"))
        if lo is None or hi is None:
            served = p.get("pick_range")
            served = served if isinstance(served, dict) else {}
            lo, hi = served.get("lo"), served.get("hi")
        if lo is None or hi is None:
            p.pop("pick_lo", None)
            p.pop("pick_hi", None)
            continue
        p["pick_lo"], p["pick_hi"] = min(lo, hi), max(lo, hi)
    return by_class


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

    # 1) nominal picks (+ interval bounds) by class rank of raw model_pick
    apply_nominal_picks(rows)
    n_bounds = sum(1 for p in rows if p.get("pick_lo") is not None)
    n_raw = sum(1 for p in rows if p.get("pick_lo") is not None
                and isinstance(p.get("pick_range_raw"), dict)
                and p.get("projected_pick") is not None)
    print(f"pick bounds: {n_bounds}/{len(rows)} rows "
          f"({n_raw} re-ranked from pick_range_raw against the rebuilt board, "
          f"{n_bounds - n_raw} fell back to the served pick_range)")

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

    # 4) ledger: rewrite today's snapshot (+ .sha256 sidecar) and movers from
    #    the post-processed rows. The builder's own snapshot was pre-relabel.
    write_board_history(rows, cache.get("generated_at") or "", rewrite=True,
                        cache_path=CACHE)

    from collections import Counter
    print("\nprojection distribution:", dict(Counter(p.get("draft_grade") for p in rows).most_common(14)))
    print("generational:", [p["name"] for p in rows if p.get("draft_grade") == "Generational"])
    print("top5:", [p["name"] for p in rows if p.get("draft_grade") == "Top 5 Pick"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
