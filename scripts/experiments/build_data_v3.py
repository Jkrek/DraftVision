#!/usr/bin/env python
"""EXPERIMENT v3 dataset builder (does NOT touch production artifacts).

Extends build_data_v2.py:
  - draft classes 2000-2026 (v2: 2000-2020). 2021-2026 are TRAIN-ONLY modern
    rows for experiment E (frozen test stays 2019-2020; forward split tests
    2025-2026). Undrafted combine-invitee negatives included for all years.
  - A/B. recruiting pedigree from CFBD /recruiting/players (1999-2023):
    rec_stars, rec_rating, rec_ranking, rec_year, years_in_college,
    early_declare, rec_match_type (school|pos|none — reporting only).
  - C. all-position production from CFBD /stats/player/season (2004-2025):
    final-season (fs_*) and career-aggregate (car_*) raw stat columns for
    passing / rushing / receiving / defensive / interceptions. Composites +
    z-scores are computed in the harness (they depend on the split).
  - D. SP+ team rating of the final college season (sp_rating).

Base columns and measurables are built with the same imported production
logic as v2; the 2010-2020 slice is asserted byte-identical to the
production CSV.

Run: .venv/bin/python scripts/experiments/build_data_v3.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, HERE)

import build_training_data as btd  # noqa: E402
from build_data_v2 import MEASURABLE_COLS, measurables_from_combine_row, _num  # noqa: E402
from cfbd_client import get  # noqa: E402

OUT_PATH = os.path.join(REPO_ROOT, "training_data", "experiments", "combine_outcomes_v3.csv")
PROD_CSV = os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv")

FIRST_CLASS, LAST_CLASS = 2000, 2026
RECRUIT_YEARS = range(1999, 2024)
STAT_SEASONS = range(2004, 2026)
SP_YEARS = range(1999, 2026)

FS_STATS = ["pass_yds", "pass_td", "pass_int", "pass_att",
            "rush_yds", "rush_td", "rush_car",
            "rec", "rec_yds", "rec_td",
            "tackles", "tfl", "sacks", "pd", "def_int"]

# recruiting position -> coarse group (ATH matches every group)
_REC_POS_GROUP = {
    "QB": "QB", "PRO": "QB", "DUAL": "QB",
    "RB": "RB", "APB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "OT": "OL", "OG": "OL", "OC": "OL", "OL": "OL", "C": "OL",
    "WDE": "DL", "SDE": "DL", "DT": "DL", "DE": "DL", "DL": "DL", "NT": "DL", "EDGE": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB", "FS": "DB", "SS": "DB", "ATH": "ATH",
}
_DRAFT_POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "G": "OL", "T": "OL", "LS": "OL",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB", "FS": "DB", "SS": "DB",
}

_nt = btd._normalize_team


def _team_compatible(a: str, b: str) -> bool:
    """Prefix/substring compatibility between two normalized team spellings."""
    if not a or not b:
        return False
    return (a == b or a.startswith(b + " ") or b.startswith(a + " ")
            or a in b or b in a)


# ── A/B. recruiting index ─────────────────────────────────────────────────────

def load_recruiting():
    idx = defaultdict(list)
    for y in RECRUIT_YEARS:
        for r in get("/recruiting/players", {"year": y}):
            nm = btd.norm_name(r.get("name"))
            if not nm:
                continue
            idx[nm].append({
                "year": int(r["year"]),
                "committed": _nt(r.get("committedTo") or ""),
                "pos_group": _REC_POS_GROUP.get((r.get("position") or "").upper(), "OTHER"),
                "hs": (r.get("recruitType") == "HighSchool"),
                "stars": _num(r.get("stars")),
                "rating": _num(r.get("rating")),
                "ranking": _num(r.get("ranking")),
            })
    return idx


def match_recruit(idx, name, college, pos, draft_year):
    """Returns (record, match_type) or (None, 'none')."""
    cands = [c for c in idx.get(btd.norm_name(name), [])
             if draft_year - 7 <= c["year"] <= draft_year - 3]
    if not cands:
        return None, "none"
    school = _nt(btd.clean_school(college))
    by_school = [c for c in cands if _team_compatible(c["committed"], school)]
    if by_school:
        pick = sorted(by_school, key=lambda c: (not c["hs"], -(c["rating"] or 0)))[0]
        return pick, "school"
    grp = _DRAFT_POS_GROUP.get((pos or "").upper())
    by_pos = [c for c in cands if grp and (c["pos_group"] == grp or c["pos_group"] == "ATH")]
    if by_pos:
        pick = sorted(by_pos, key=lambda c: (not c["hs"], -(c["rating"] or 0)))[0]
        return pick, "pos"
    return None, "none"


# ── C. per-season stat index ─────────────────────────────────────────────────

_STAT_KEY = {
    ("passing", "YDS"): "pass_yds", ("passing", "TD"): "pass_td",
    ("passing", "INT"): "pass_int", ("passing", "ATT"): "pass_att",
    ("rushing", "YDS"): "rush_yds", ("rushing", "TD"): "rush_td",
    ("rushing", "CAR"): "rush_car",
    ("receiving", "REC"): "rec", ("receiving", "YDS"): "rec_yds",
    ("receiving", "TD"): "rec_td",
    ("defensive", "TOT"): "tackles", ("defensive", "TFL"): "tfl",
    ("defensive", "SACKS"): "sacks", ("defensive", "PD"): "pd",
    ("interceptions", "INT"): "def_int",
}


def load_stats():
    """Returns (by_name, by_pid).
    by_name: norm_name -> list of (season, pid, norm_team) ;
    by_pid:  pid -> {season: stats dict}."""
    by_name = defaultdict(list)
    by_pid = defaultdict(dict)
    for y in STAT_SEASONS:
        recs = get("/stats/player/season", {"year": y})
        seen = {}
        for r in recs:
            key = _STAT_KEY.get((r["category"], r["statType"]))
            if key is None:
                continue
            pid = r["playerId"]
            if pid not in seen:
                seen[pid] = (btd.norm_name(r["player"]), _nt(r["team"]))
            v = _num(r["stat"])
            if np.isfinite(v):
                d = by_pid[pid].setdefault(y, {})
                d[key] = d.get(key, 0.0) + v
        for pid, (nm, tm) in seen.items():
            if y in by_pid.get(pid, {}):
                by_name[nm].append((y, pid, tm))
    return by_name, by_pid


def stats_for(by_name, by_pid, name, college, draft_year):
    """Final-season (D-1, fallback D-2) + career sums for a prospect.
    Team must be compatible with the draft-file school when both known."""
    nm = btd.norm_name(name)
    school = _nt(btd.clean_school(college))
    cands = by_name.get(nm, [])
    pid = None
    fs = None
    for season in (draft_year - 1, draft_year - 2):
        hits = [(p, t) for (s, p, t) in cands if s == season
                and (not school or _team_compatible(t, school))]
        if hits:
            pid = hits[0][0]
            fs = by_pid[pid].get(season)
            break
    if pid is None:
        # any earlier season with a team match still identifies the player
        earlier = [(s, p) for (s, p, t) in cands
                   if s <= draft_year - 1 and school and _team_compatible(t, school)]
        if earlier:
            pid = max(earlier)[1]
    out = {}
    for k in FS_STATS:
        out["fs_" + k] = fs.get(k, 0.0) if fs else np.nan
    car_seasons = 0
    car = defaultdict(float)
    if pid is not None:
        for s, d in by_pid[pid].items():
            if s <= draft_year - 1:
                car_seasons += 1
                for k, v in d.items():
                    car[k] += v
    for k in FS_STATS:
        out["car_" + k] = car.get(k, 0.0) if car_seasons else np.nan
    out["car_seasons"] = car_seasons if car_seasons else np.nan
    return out


# ── D. SP+ ratings ───────────────────────────────────────────────────────────

def load_sp():
    sp = {}
    for y in SP_YEARS:
        for r in get("/ratings/sp", {"year": y}):
            if r.get("team") and r.get("rating") is not None:
                sp[(y, _nt(r["team"]))] = float(r["rating"])
    return sp


def sp_for(sp, college, draft_year, _cache={}):
    school = _nt(btd.clean_school(college))
    if not school:
        return np.nan
    for y in (draft_year - 1, draft_year - 2):
        if (y, school) in sp:
            return sp[(y, school)]
        ck = (y, school)
        if ck not in _cache:
            hit = np.nan
            for (yy, t), v in sp.items():
                if yy == y and _team_compatible(t, school):
                    hit = v
                    break
            _cache[ck] = hit
        if np.isfinite(_cache[ck]):
            return _cache[ck]
    return np.nan


# ── main build ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading nflverse data from cache…")
    draft = pd.read_csv(btd.fetch(btd.DRAFT_URL, "draft_picks.csv"))
    combine = pd.read_csv(btd.fetch(btd.COMBINE_URL, "combine.csv"))
    draft = draft[(draft.season >= FIRST_CLASS) & (draft.season <= LAST_CLASS)].copy()
    combine = combine[(combine.season >= FIRST_CLASS) & (combine.season <= LAST_CLASS)].copy()
    draft["av"] = draft["car_av"].fillna(draft["w_av"])

    comb_by_pfr, comb_by_name = {}, {}
    for _, r in combine.iterrows():
        if pd.notna(r.get("pfr_id")):
            comb_by_pfr[(r["pfr_id"], int(r["season"]))] = r
        comb_by_name[(btd.norm_name(r["player_name"]), int(r["season"]))] = r

    print("Loading cfbfastR play-level stats (base production_score)…")
    cfb_by_season = {}
    for season in range(btd.CFB_FIRST_SEASON, LAST_CLASS):  # 2014..2025
        try:
            cfb_by_season[season] = btd.load_cfb_season(season)
            print(f"  {season}: {len(cfb_by_season[season])} players")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not load {season} ({exc})")

    print("Loading CFBD recruiting / season stats / SP+ (cached)…")
    rec_idx = load_recruiting()
    st_by_name, st_by_pid = load_stats()
    sp = load_sp()
    print(f"  recruiting names: {len(rec_idx)}; stat names: {len(st_by_name)}; "
          f"sp entries: {len(sp)}")

    def cfbd_cols(name, college, pos, year):
        rec, mtype = match_recruit(rec_idx, name, college, pos, year)
        out = {
            "rec_stars": rec["stars"] if rec else np.nan,
            "rec_rating": rec["rating"] if rec else np.nan,
            "rec_ranking": rec["ranking"] if rec else np.nan,
            "rec_year": rec["year"] if rec else np.nan,
            "years_in_college": (year - rec["year"]) if rec else np.nan,
            "early_declare": (1.0 if year - rec["year"] <= 3 else 0.0) if rec else np.nan,
            "rec_match_type": mtype,
            "sp_rating": sp_for(sp, college, year),
        }
        out.update(stats_for(st_by_name, st_by_pid, name, college, year))
        return out

    rows = []
    drafted_pfr_ids = set(draft["pfr_player_id"].dropna())
    drafted_names = {(btd.norm_name(n), int(s)) for n, s in
                     zip(draft["pfr_player_name"], draft["season"])}

    print("Building drafted rows…")
    for _, r in draft.iterrows():
        year = int(r["season"])
        pos = btd.norm_pos(r["position"])
        name = r["pfr_player_name"]
        college = btd.clean_school(r["college"] if pd.notna(r["college"]) else "")
        crow = None
        if pd.notna(r.get("pfr_player_id")):
            crow = comb_by_pfr.get((r["pfr_player_id"], year))
        if crow is None:
            crow = comb_by_name.get((btd.norm_name(name), year))
        forty = None
        if crow is not None and pd.notna(crow.get("forty")):
            forty = float(crow["forty"])
        prod = btd.production_for(name, college, pos, year, cfb_by_season)
        to_season = r["to"] if pd.notna(r["to"]) else None
        exp = int(max(0, to_season - year + 1)) if to_season else 0
        probowls = int(r["probowls"]) if pd.notna(r["probowls"]) else 0
        sstart = int(r["seasons_started"]) if pd.notna(r["seasons_started"]) else 0
        av = float(r["av"]) if pd.notna(r["av"]) else 0.0
        rnd = int(r["round"])
        rows.append({
            "name": name, "draft_year": year, "position": pos, "college": college,
            "conference_tier": btd.classify_college_tier(college),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(btd.forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": rnd,
            "draft_grade": btd.draft_grade_for(rnd),
            "nfl_success": btd.success_label(probowls, sstart, av),
            "experience_years": exp, "pro_bowls": probowls,
            "seasons_started": sstart, "career_av": av,
            **measurables_from_combine_row(crow),
            "age": _num(r.get("age")),
            **cfbd_cols(name, college, pos, year),
        })

    print("Building undrafted combine-invitee rows…")
    for _, r in combine.iterrows():
        year = int(r["season"])
        if pd.notna(r.get("draft_round")):
            continue
        if pd.notna(r.get("pfr_id")) and r["pfr_id"] in drafted_pfr_ids:
            continue
        if (btd.norm_name(r["player_name"]), year) in drafted_names:
            continue
        pos = btd.norm_pos(r["pos"])
        name = r["player_name"]
        school = btd.clean_school(r["school"] if pd.notna(r["school"]) else "")
        forty = float(r["forty"]) if pd.notna(r["forty"]) else None
        prod = btd.production_for(name, school, pos, year, cfb_by_season)
        rows.append({
            "name": name, "draft_year": year, "position": pos, "college": school,
            "conference_tier": btd.classify_college_tier(school),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(btd.forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": 8, "draft_grade": 3, "nfl_success": 0,
            "experience_years": 0, "pro_bowls": 0, "seasons_started": 0,
            "career_av": 0.0,
            **measurables_from_combine_row(r),
            "age": np.nan,
            **cfbd_cols(name, school, pos, year),
        })

    base_cols = [
        "name", "draft_year", "position", "college", "conference_tier",
        "combine_forty", "combine_speed_score", "games_college",
        "production_score", "draft_round", "draft_grade", "nfl_success",
        "experience_years", "pro_bowls", "seasons_started", "career_av",
    ]
    new_cols = (["rec_stars", "rec_rating", "rec_ranking", "rec_year",
                 "years_in_college", "early_declare", "rec_match_type", "sp_rating"]
                + ["fs_" + k for k in FS_STATS] + ["car_" + k for k in FS_STATS]
                + ["car_seasons"])
    df = pd.DataFrame(rows, columns=base_cols + MEASURABLE_COLS + ["age"] + new_cols)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # ── Verify 2010-2020 slice matches the production CSV exactly ────────────
    prod_df = pd.read_csv(PROD_CSV)
    v3 = pd.read_csv(OUT_PATH)
    sl = v3[(v3.draft_year >= 2010) & (v3.draft_year <= 2020)][base_cols].reset_index(drop=True)
    same = len(prod_df) == len(sl) and sl.fillna("__NA__").astype(str).equals(
        prod_df[base_cols].fillna("__NA__").astype(str))
    print(f"\nWrote {OUT_PATH}: {len(df)} rows")
    print(f"2010-2020 slice identical to production CSV: {same} "
          f"(prod {len(prod_df)} vs slice {len(sl)})")

    # ── G. missingness report drafted vs UDFA ────────────────────────────────
    drafted = df[df.draft_round <= 7]
    udfa = df[df.draft_round == 8]
    print("\nCoverage (non-NaN %) drafted vs UDFA — full 2000-2026:")
    for c in ["rec_stars", "rec_rating", "rec_ranking", "years_in_college",
              "early_declare", "sp_rating", "fs_pass_yds", "fs_tackles",
              "car_seasons", "age"]:
        d = pd.to_numeric(drafted[c], errors="coerce").notna().mean() * 100
        u = pd.to_numeric(udfa[c], errors="coerce").notna().mean() * 100
        print(f"  {c:20s} drafted {d:5.1f}%  udfa {u:5.1f}%")
    m = df[(df.draft_year >= 2010)]
    dd, uu = m[m.draft_round <= 7], m[m.draft_round == 8]
    print("Coverage drafted vs UDFA — 2010+ only:")
    for c in ["rec_stars", "years_in_college", "sp_rating", "fs_tackles", "car_seasons"]:
        d = pd.to_numeric(dd[c], errors="coerce").notna().mean() * 100
        u = pd.to_numeric(uu[c], errors="coerce").notna().mean() * 100
        print(f"  {c:20s} drafted {d:5.1f}%  udfa {u:5.1f}%")
    print("Recruit match type share (2010+): ")
    print(m.rec_match_type.value_counts(normalize=True).round(3).to_dict())
    print("Rows per draft year:")
    for year, cnt in df.draft_year.value_counts().sort_index().items():
        print(f"  {year}: {cnt}")


if __name__ == "__main__":
    main()
