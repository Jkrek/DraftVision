#!/usr/bin/env python
"""Build training_data/enrichment.json — CFBD features for CURRENT players.

The v3 model features (dv_features.V3_FEATURES) include recruiting pedigree,
the years-in-college age proxy, all-position production composites and SP+.
Training rows get them from scripts/build_training_data.py; live /predict
requests get them from this file, merged in XGBOost.build_success_features
(hot-reloaded by mtime). Players absent from this file degrade gracefully:
their new feature columns are NaN and the NaN-native ensemble grades them on
whatever it can see — exactly how unmatched players look in training.

Mapping:  "Name|Team"  (prospect-cache identity: ESPN display name + ESPN team
name, e.g. "Jeremiah Smith|Ohio State Buckeyes") →
  stars / rating / national_rank / recruit_year  — CFBD /recruiting/players,
      classes 2022-2026, matched by normalized name + committed school
      (falls back to name + position group for transfers; same matcher the
      training build uses),
  sp_plus       — the team's latest SP+ rating (/ratings/sp, 2025 else 2024),
  prod_fs_raw / prod_car_raw / car_seasons / fs_season — production composites
      (dv_features.production_composite) from CFBD /stats/player/season for the
      most recent recorded season (2025, else 2024) and the career to date,
      matched by normalized name + school. Raw values on purpose: serving
      z-scores them with the FROZEN training stats in models/feature_stats.json.

Player universe: training_data/prospect_cache.json (the current FBS board).
CFBD responses are cached under /tmp/dv_training_cache/cfbd/; the API key is
read from .env at runtime and never stored.

Run: .venv/bin/python scripts/build_enrichment.py
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import numpy as np

import build_training_data as btd
from dv_features import production_composite, _composite_group

PROSPECT_CACHE_PATH = os.path.join(REPO_ROOT, "training_data", "prospect_cache.json")
OUT_PATH = os.path.join(REPO_ROOT, "training_data", "enrichment.json")

RECRUIT_CLASSES = range(2022, 2027)   # current rosters: 5th-years back to true FR
STAT_SEASONS = range(2019, 2026)      # career window for anyone on a 2026 roster
SP_SEASONS = (2025, 2024)             # latest completed season first
MOST_RECENT_DRAFT_ANCHOR = 2026       # stats_for(D) uses seasons D-1 (2025), D-2

# match_recruit windows candidates to [dy-7, dy-3]; anchor 2029 makes that
# window exactly RECRUIT_CLASSES (2022-2026) without duplicating the matcher.
RECRUIT_MATCH_ANCHOR = 2029


def _clean(v):
    """np/NaN -> None for JSON; round floats for a compact file."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return round(f, 4)


def load_recruiting_current():
    idx = btd.defaultdict(list)
    for y in RECRUIT_CLASSES:
        for r in btd._cfbd_get("/recruiting/players", {"year": y}):
            nm = btd.norm_name(r.get("name"))
            if not nm:
                continue
            idx[nm].append({
                "year": int(r["year"]),
                "committed": btd._normalize_team(r.get("committedTo") or ""),
                "pos_group": btd._REC_POS_GROUP.get((r.get("position") or "").upper(), "OTHER"),
                "hs": (r.get("recruitType") == "HighSchool"),
                "stars": btd._num(r.get("stars")),
                "rating": btd._num(r.get("rating")),
                "ranking": btd._num(r.get("ranking")),
            })
    return idx


def load_sp_latest():
    sp = {}
    for y in SP_SEASONS:
        for r in btd._cfbd_get("/ratings/sp", {"year": y}):
            if r.get("team") and r.get("rating") is not None:
                sp.setdefault(btd._normalize_team(r["team"]), float(r["rating"]))
    return sp


def sp_latest_for(sp, team):
    t = btd._normalize_team(btd.clean_school(team))
    if not t:
        return None
    if t in sp:
        return sp[t]
    for k, v in sp.items():
        if btd._team_compatible(k, t):
            return v
    return None


def main() -> None:
    with open(PROSPECT_CACHE_PATH) as fh:
        prospects = json.load(fh).get("prospects", [])
    print(f"Prospect universe: {len(prospects)} players ({PROSPECT_CACHE_PATH})")

    print("Loading CFBD recruiting 2022-2026 / season stats 2019-2025 / SP+ (cached)…")
    rec_idx = load_recruiting_current()
    st_by_name, st_by_pid = btd.load_stats(seasons=STAT_SEASONS)
    sp = load_sp_latest()
    print(f"  recruiting names: {len(rec_idx)}; stat names: {len(st_by_name)}; "
          f"sp teams: {len(sp)}")

    out = {}
    n_rec = n_prod = n_sp = 0
    for p in prospects:
        name = str(p.get("name") or "").strip()
        team = str(p.get("team") or "").strip()
        pos = str(p.get("position") or "").strip().upper()
        if not name or not team:
            continue
        key = f"{name}|{team}"
        if key in out:
            continue

        rec, mtype = btd.match_recruit(rec_idx, name, team, pos, RECRUIT_MATCH_ANCHOR)

        st = btd.stats_for(st_by_name, st_by_pid, name, team, MOST_RECENT_DRAFT_ANCHOR)
        grp = _composite_group(pos)
        fs_raw = production_composite(grp, {k: st.get("fs_" + k) for k in btd.FS_STATS})
        car_raw = production_composite(grp, {k: st.get("car_" + k) for k in btd.FS_STATS})
        car_seasons = st.get("car_seasons")
        # stats_for anchors at D=2026: final-season = 2025 with 2024 fallback
        fs_season = "2025_or_2024" if math.isfinite(btd._num(st.get("fs_pass_yds"))) else None

        sp_plus = sp_latest_for(sp, team)

        entry = {
            "stars":         _clean(rec["stars"]) if rec else None,
            "rating":        _clean(rec["rating"]) if rec else None,
            "national_rank": _clean(rec["ranking"]) if rec else None,
            "recruit_year":  int(rec["year"]) if rec else None,
            "match_type":    mtype,
            "sp_plus":       _clean(sp_plus),
            "prod_fs_raw":   _clean(fs_raw),
            "prod_car_raw":  _clean(car_raw),
            "car_seasons":   _clean(car_seasons),
            "fs_season":     fs_season,
        }
        if all(entry[k] is None for k in
               ("stars", "rating", "national_rank", "sp_plus", "prod_fs_raw",
                "prod_car_raw", "car_seasons")):
            continue  # nothing to enrich — omit so lookups stay honest NaN
        out[key] = entry
        n_rec += rec is not None
        n_prod += entry["prod_fs_raw"] is not None or entry["prod_car_raw"] is not None
        n_sp += entry["sp_plus"] is not None

    payload = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "source": "CFBD recruiting 2022-2026, /stats/player/season 2019-2025, "
                  "/ratings/sp 2025 (fallback 2024)",
        "key": "Name|Team (prospect-cache identity)",
        "players": out,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, OUT_PATH)

    n = len(out)
    print(f"\nWrote {OUT_PATH}: {n} enriched of {len(prospects)} prospects")
    print(f"  recruiting matched: {n_rec} ({n_rec / max(n,1) * 100:.1f}% of enriched)")
    print(f"  production matched: {n_prod} ({n_prod / max(n,1) * 100:.1f}%)")
    print(f"  sp+ matched:        {n_sp} ({n_sp / max(n,1) * 100:.1f}%)")
    for probe in ("Jeremiah Smith|Ohio State Buckeyes",
                  "Colin Simmons|Texas Longhorns",
                  "Arch Manning|Texas Longhorns"):
        print(f"  {probe}: {json.dumps(out.get(probe))}")


if __name__ == "__main__":
    main()
