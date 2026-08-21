#!/usr/bin/env python
"""Measure join match rates of every staged dataset against the repo's two
canonical player references:

  (a) training_data/combine_outcomes.csv   — training rows (2000-2026 classes)
  (b) training_data/prospect_cache.json    — live prospect names (current CFB)

Matching reuses the exact normalizers from scripts/build_training_data.py
(norm_name / clean_school / _normalize_team / _team_compatible), i.e. the same
fuzzy machinery as the recruiting join. A staged row "matches" when its
normalized name is found with a compatible draft-year window; school
compatibility is reported as the stricter tier.

Direction per dataset (stated in the output, chosen for honesty):
  * selective datasets (consensus boards, all-star invites, contracts,
    snap counts, portal) -> % of STAGED rows that resolve into the reference;
  * coverage datasets (cfbd_usage / cfbd_ppa, which contain every FBS player)
    -> % of REFERENCE rows in the applicable window that find a staged row
    (a raw staged->reference rate would just measure how many college players
    never reach the NFL, which is not a join-quality signal).

Datasets under 60%% on their headline rate are flagged with unmatched
examples.

Run: .venv/bin/python scripts/data/join_check.py
"""

import json
import os
from collections import defaultdict

import pandas as pd

from _common import (REPO_ROOT, STAGING_DIR, _normalize_team, _team_compatible,
                     clean_school, norm_name)

FLAG_THRESHOLD = 60.0


def _s(v) -> str:
    """NaN-safe string coercion for school/name cells."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def load_combine_outcomes():
    df = pd.read_csv(os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv"),
                     usecols=["name", "draft_year", "college", "position"])
    idx = defaultdict(list)  # norm_name -> [(draft_year, norm_school, position)]
    for r in df.itertuples(index=False):
        idx[norm_name(_s(r.name))].append(
            (int(r.draft_year), _normalize_team(clean_school(_s(r.college))), r.position))
    return df, idx


def load_prospect_cache():
    with open(os.path.join(REPO_ROOT, "training_data", "prospect_cache.json")) as fh:
        cache = json.load(fh)
    idx = defaultdict(set)  # norm_name -> {norm_team}
    for p in cache["prospects"]:
        idx[norm_name(p.get("name"))].add(_normalize_team(p.get("team") or ""))
    return cache["prospects"], idx


def match_ref_a(idx_a, name, year, school, year_lo=0, year_hi=0):
    """Return 'name+year+school' | 'name+year' | 'none' against combine_outcomes."""
    cands = [c for c in idx_a.get(norm_name(name), [])
             if year + year_lo <= c[0] <= year + year_hi]
    if not cands:
        return "none"
    school_n = _normalize_team(clean_school(_s(school)))
    if any(_team_compatible(c[1], school_n) for c in cands):
        return "name+year+school"
    return "name+year"


def match_ref_b(idx_b, name, school=None):
    teams = idx_b.get(norm_name(name))
    if not teams:
        return "none"
    if school is not None:
        school_n = _normalize_team(clean_school(_s(school)))
        if any(_team_compatible(t, school_n) for t in teams):
            return "name+school"
    return "name"


# ---------------------------------------------------------------------------
# Per-dataset checks
# ---------------------------------------------------------------------------

RESULTS = []


def report(dataset, direction, headline_pct, n_base, tiers, examples):
    RESULTS.append((dataset, direction, headline_pct, n_base, tiers, examples))
    flag = "  <-- UNDER 60%, review" if headline_pct < FLAG_THRESHOLD else ""
    print(f"\n== {dataset}")
    print(f"   direction : {direction}")
    print(f"   base rows : {n_base}")
    print(f"   match     : {headline_pct:.1f}%{flag}")
    for tier, n in tiers.items():
        print(f"     {tier:<18} {n}")
    if headline_pct < FLAG_THRESHOLD and examples:
        print("   unmatched examples:")
        for e in examples[:10]:
            print(f"     - {e}")


def check_staged_vs_a(df, idx_a, dataset, name_col, year_col, school_col,
                      year_lo=0, year_hi=0, idx_b=None, b_school_col=None):
    tiers = defaultdict(int)
    misses = []
    for r in df.itertuples(index=False):
        t = match_ref_a(idx_a, getattr(r, name_col), int(getattr(r, year_col)),
                        getattr(r, school_col) if school_col else "",
                        year_lo, year_hi)
        tiers[t] += 1
        if t == "none":
            misses.append(f"{getattr(r, name_col)} ({getattr(r, school_col) if school_col else '?'}, "
                          f"{getattr(r, year_col)})")
    n = len(df)
    matched = n - tiers["none"]
    report(dataset + "  vs combine_outcomes", "staged -> reference",
           100.0 * matched / n if n else 0.0, n, dict(tiers), misses)
    if idx_b is not None:
        tb = defaultdict(int)
        for r in df.itertuples(index=False):
            tb[match_ref_b(idx_b, getattr(r, name_col),
                           getattr(r, b_school_col) if b_school_col else None)] += 1
        mb = n - tb["none"]
        print(f"   (vs prospect cache: {100.0 * mb / n if n else 0:.1f}% "
              f"of staged names present — informational; cache only holds CURRENT college players)")


def main() -> None:
    _, idx_a = load_combine_outcomes()
    prospects, idx_b = load_prospect_cache()
    co = pd.read_csv(os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv"),
                     usecols=["name", "draft_year", "college"])

    s = lambda f: pd.read_csv(os.path.join(STAGING_DIR, f))

    # 1. Wide Left consensus (2024-2026 classes)
    check_staged_vs_a(s("consensus_wideleft.csv"), idx_a, "consensus_wideleft.csv",
                      "player", "draft_year", "school", idx_b=idx_b, b_school_col="school")

    # 2. ESPN historical consensus (2004-2021 in practice)
    check_staged_vs_a(s("consensus_espn.csv"), idx_a, "consensus_espn.csv",
                      "player", "draft_year", "school")

    # 3. All-star invites (game year == draft-class year)
    check_staged_vs_a(s("allstar_invites.csv"), idx_a, "allstar_invites.csv",
                      "player", "year", "school")

    # 4. Contracts — distinct players whose OTC draft_year falls in the
    #    training window. combine_outcomes only holds combine invitees +
    #    drafted players, so undrafted street free agents are structurally
    #    absent; the drafted subset is the honest join-quality number.
    con = s("contracts.csv")
    con_w = con[(con.draft_year >= 2000) & (con.draft_year <= 2026)].copy()
    con_w["draft_year"] = con_w["draft_year"].astype(int)
    con_p = con_w.drop_duplicates("otc_id")
    check_staged_vs_a(con_p, idx_a,
                      "contracts.csv (distinct players, classes 2000-2026)",
                      "player", "draft_year", "college")
    con_d = con_p[con_p.draft_round.notna()]
    check_staged_vs_a(con_d, idx_a,
                      "contracts.csv (distinct DRAFTED players 2000-2026)",
                      "player", "draft_year", "college")

    # 5. Snap counts — distinct players, matched on name with the draft year
    #    required to be <= the player's first snap season (+1 slack for
    #    January-drafted edge cases; no school column exists in this source).
    sn = s("snap_counts.csv")
    first = sn.groupby("player", as_index=False)["season"].min()
    tiers = defaultdict(int)
    misses = []
    for r in first.itertuples(index=False):
        cands = [c for c in idx_a.get(norm_name(_s(r.player)), []) if c[0] <= r.season + 1]
        if cands:
            tiers["name+year-window"] += 1
        else:
            tiers["none"] += 1
            misses.append(f"{r.player} (first snap season {r.season})")
    n = len(first)
    report("snap_counts.csv (distinct players)", "staged -> reference",
           100.0 * tiers["name+year-window"] / n, n, dict(tiers), misses)

    # 6. CFBD usage / PPA — these endpoints only publish OFFENSIVE skill
    #    players (QB/RB/WR/TE/FB) at FBS schools, so coverage is measured
    #    against combine_outcomes rows at those positions (classes 2016-2026);
    #    defenders/OL/FCS players are structurally absent, not join failures.
    co_pos = pd.read_csv(os.path.join(REPO_ROOT, "training_data", "combine_outcomes.csv"),
                         usecols=["name", "draft_year", "college", "position"])
    skill = co_pos[co_pos.position.isin(["QB", "RB", "WR", "TE", "FB"])]
    for fname in ("cfbd_usage.csv", "cfbd_ppa.csv"):
        st = s(fname)
        st_idx = defaultdict(list)  # norm_name -> [(season, norm_team)]
        for r in st.itertuples(index=False):
            st_idx[norm_name(_s(r.player))].append((int(r.season), _normalize_team(_s(r.school))))
        ref = skill[(skill.draft_year >= 2016) & (skill.draft_year <= 2026)]
        tiers = defaultdict(int)
        misses = []
        for r in ref.itertuples(index=False):
            school_n = _normalize_team(clean_school(_s(r.college)))
            cands = [c for c in st_idx.get(norm_name(_s(r.name)), [])
                     if r.draft_year - 6 <= c[0] <= r.draft_year - 1]
            if not cands:
                tiers["none"] += 1
                misses.append(f"{r.name} ({r.college}, {r.draft_year})")
            elif any(_team_compatible(c[1], school_n) for c in cands):
                tiers["name+season+school"] += 1
            else:
                tiers["name+season"] += 1
        n = len(ref)
        matched = n - tiers["none"]
        report(f"{fname} (combine_outcomes QB/RB/WR/TE/FB classes 2016-2026)",
               "reference -> staged (coverage)",
               100.0 * matched / n, n, dict(tiers), misses)
        # informational: current-class coverage vs prospect cache (2025 season)
        cur = {norm_name(p["name"]): _normalize_team(p.get("team") or "") for p in prospects}
        st25 = {nm for nm, lst in st_idx.items() if any(c[0] == 2025 for c in lst)}
        hit = sum(1 for nm in cur if nm in st25)
        print(f"   (prospect cache names with a 2025 {fname.split('_')[1].split('.')[0]} row: "
              f"{100.0 * hit / len(cur):.1f}% of {len(cur)} — cache includes deep FBS backups)")

    # 7. Transfer portal — event data; staged -> reference on name only
    #    (a transfer in season Y maps to draft classes Y..Y+4).
    po = s("cfbd_transfer_portal.csv")
    tiers = defaultdict(int)
    misses = []
    for r in po.itertuples(index=False):
        cands = [c for c in idx_a.get(norm_name(_s(r.player)), [])
                 if r.season <= c[0] <= r.season + 5]
        if cands:
            tiers["name+year-window"] += 1
        else:
            tiers["none"] += 1
            misses.append(f"{r.player} ({r.origin} -> {r.destination}, {r.season})")
    n = len(po)
    pct = 100.0 * tiers["name+year-window"] / n
    report("cfbd_transfer_portal.csv", "staged -> reference (most portal "
           "entrants never reach a draft class; low % is EXPECTED)",
           pct, n, dict(tiers), misses if pct < FLAG_THRESHOLD else [])
    tb = sum(1 for r in po.itertuples(index=False)
             if match_ref_b(idx_b, r.player, r.destination) != "none")
    print(f"   (vs prospect cache: {100.0 * tb / n:.1f}% of portal names present)")

    print("\nSummary")
    for ds, _, pct, n, _, _ in RESULTS:
        flag = "  ** UNDER 60%" if pct < FLAG_THRESHOLD else ""
        print(f"  {pct:5.1f}%  {ds}  (n={n}){flag}")


if __name__ == "__main__":
    main()
