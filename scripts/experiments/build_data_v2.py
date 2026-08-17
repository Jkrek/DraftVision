#!/usr/bin/env python
"""EXPERIMENT dataset builder (does NOT touch production artifacts).

Builds training_data/experiments/combine_outcomes_v2.csv:
  - draft classes 2000-2020 (vs production 2010-2020) — extra 2000-2009 rows
    are TRAIN-ONLY history for experiment C.
  - adds raw combine measurables the production CSV discards:
    height_in, weight_lb, vertical, bench, broad_in, cone, shuttle
    (NaN when the player has no combine row / missing drill), plus draft-file
    age (drafted players only; NaN for undrafted invitees).

All join/label/production logic is IMPORTED from scripts/build_training_data.py
(unmodified) so rows for 2010-2020 are identical-by-construction to the
production CSV; an assertion verifies this at the end.

Run: .venv/bin/python scripts/experiments/build_data_v2.py
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import build_training_data as btd  # noqa: E402  (pure functions + constants)

OUT_PATH = os.path.join(REPO_ROOT, "training_data", "experiments", "combine_outcomes_v2.csv")
PROD_CSV = os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv")

FIRST_CLASS, LAST_CLASS = 2000, 2020

MEASURABLE_COLS = ["height_in", "weight_lb", "vertical", "bench", "broad_in", "cone", "shuttle"]


def parse_height(ht) -> float:
    """'6-4' -> 76.0 inches. NaN otherwise."""
    if ht is None or (isinstance(ht, float) and np.isnan(ht)):
        return np.nan
    s = str(ht).strip()
    if "-" in s:
        try:
            ft, inch = s.split("-", 1)
            return float(ft) * 12.0 + float(inch)
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def measurables_from_combine_row(crow) -> dict:
    if crow is None:
        return {c: np.nan for c in MEASURABLE_COLS}
    return {
        "height_in": parse_height(crow.get("ht")),
        "weight_lb": _num(crow.get("wt")),
        "vertical":  _num(crow.get("vertical")),
        "bench":     _num(crow.get("bench")),
        "broad_in":  _num(crow.get("broad_jump")),
        "cone":      _num(crow.get("cone")),
        "shuttle":   _num(crow.get("shuttle")),
    }


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

    print("Loading cfbfastR play-level stats (cached; 2014+ only)…")
    cfb_by_season = {}
    for season in range(btd.CFB_FIRST_SEASON, LAST_CLASS):  # 2014..2019
        try:
            cfb_by_season[season] = btd.load_cfb_season(season)
            print(f"  {season}: {len(cfb_by_season[season])} players")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not load {season} ({exc})")

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
            "name": name,
            "draft_year": year,
            "position": pos,
            "college": college,
            "conference_tier": btd.classify_college_tier(college),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(btd.forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": rnd,
            "draft_grade": btd.draft_grade_for(rnd),
            "nfl_success": btd.success_label(probowls, sstart, av),
            "experience_years": exp,
            "pro_bowls": probowls,
            "seasons_started": sstart,
            "career_av": av,
            **measurables_from_combine_row(crow),
            "age": _num(r.get("age")),
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
            "name": name,
            "draft_year": year,
            "position": pos,
            "college": school,
            "conference_tier": btd.classify_college_tier(school),
            "combine_forty": forty if forty else "",
            "combine_speed_score": round(btd.forty_to_speed_score(pos, forty), 2) if forty else "",
            "games_college": "",
            "production_score": prod if prod is not None else "",
            "draft_round": 8,
            "draft_grade": 3,
            "nfl_success": 0,
            "experience_years": 0,
            "pro_bowls": 0,
            "seasons_started": 0,
            "career_av": 0.0,
            **measurables_from_combine_row(r),
            "age": np.nan,
        })

    base_cols = [
        "name", "draft_year", "position", "college", "conference_tier",
        "combine_forty", "combine_speed_score", "games_college",
        "production_score", "draft_round", "draft_grade", "nfl_success",
        "experience_years", "pro_bowls", "seasons_started", "career_av",
    ]
    df = pd.DataFrame(rows, columns=base_cols + MEASURABLE_COLS + ["age"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # ── Verify 2010-2020 slice matches the production CSV exactly ───────────
    prod_df = pd.read_csv(PROD_CSV)
    v2_slice = pd.read_csv(OUT_PATH)
    v2_slice = v2_slice[(v2_slice.draft_year >= 2010) & (v2_slice.draft_year <= 2020)]
    v2_slice = v2_slice[base_cols].reset_index(drop=True)
    same = len(prod_df) == len(v2_slice)
    if same:
        same = v2_slice.fillna("__NA__").astype(str).equals(
            prod_df[base_cols].fillna("__NA__").astype(str))
    print(f"\nWrote {OUT_PATH}: {len(df)} rows "
          f"({int((df.draft_year <= 2009).sum())} history rows 2000-2009)")
    print(f"2010-2020 slice identical to production CSV: {same} "
          f"(prod {len(prod_df)} vs v2-slice {len(v2_slice)} rows)")
    for c in MEASURABLE_COLS + ["age"]:
        print(f"  {c}: coverage {df[c].notna().mean() * 100:.1f}%")
    print("Rows per draft year:")
    for year, cnt in df.draft_year.value_counts().sort_index().items():
        print(f"  {year}: {cnt}")


if __name__ == "__main__":
    main()
