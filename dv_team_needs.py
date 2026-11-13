#!/usr/bin/env python3
"""Team Needs: every NFL depth chart, with each player's contract end year.

Build (weekly job, needs pyarrow):
  .venv/bin/python dv_team_needs.py            # writes training_data/team_needs.json

Serve: XGBOost.py calls index_payload() / team_payload(abbr), which only read
that JSON. Nothing here fetches at request time.

What "contract ends" means: the last league year in which the player's
current contract pays him cash, from the year-by-year table OverTheCap
publishes. That skips void years (cap placeholders with no pay) and counts an
extension from the end of the deal it extends. It is NOT a prediction that he
leaves: options, franchise tags and restricted free agency are not modelled.

Data: nflverse-data release files (CC-BY 4.0) — depth charts (ESPN-sourced)
and contracts (OverTheCap-sourced). Release files only; neither site is
scraped. Draft slots come from dv_draft_order.py.
"""
import datetime
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(REPO_ROOT, "training_data", "team_needs.json")
DRAFT_ORDER_PATH = os.path.join(REPO_ROOT, "training_data", "draft_order.json")
RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
MAX_DEPTH = 3

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Position groups a draft board talks in. Depth-chart position -> group.
# The front decides who the edge rushers are: the ends in a 4-3, the outside
# linebackers in a 3-4 (whose "ends" are interior linemen).
GROUP_OF = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "LT": "OT", "RT": "OT", "LG": "IOL", "C": "IOL", "RG": "IOL",
    "LDE": "EDGE", "RDE": "EDGE", "LDT": "DT", "RDT": "DT", "NT": "DT",
    "WLB": "LB", "MLB": "LB", "SLB": "LB", "LILB": "LB", "RILB": "LB",
    "LCB": "CB", "RCB": "CB", "NB": "CB", "FS": "S", "SS": "S",
}
GROUP_OF_34 = GROUP_OF | {"LDE": "DT", "RDE": "DT", "WLB": "EDGE", "SLB": "EDGE"}
GROUP_LABEL = {"QB": "Quarterback", "RB": "Running back", "WR": "Wide receiver", "TE": "Tight end",
               "OT": "Offensive tackle", "IOL": "Interior line", "EDGE": "Edge rusher",
               "DT": "Defensive tackle", "LB": "Linebacker", "CB": "Cornerback", "S": "Safety"}
# College position label -> the groups it can fill. The board's generic
# labels (OL, DL, DB) could be either; they match whichever the team needs more.
PROSPECT_GROUPS = {
    "QB": ("QB",), "RB": ("RB",), "FB": ("RB",), "WR": ("WR",), "TE": ("TE",),
    "OT": ("OT",), "T": ("OT",), "G": ("IOL",), "OG": ("IOL",), "C": ("IOL",), "IOL": ("IOL",),
    "OL": ("OT", "IOL"), "EDGE": ("EDGE",), "DE": ("EDGE",), "OLB": ("EDGE",),
    "DT": ("DT",), "NT": ("DT",), "DL": ("EDGE", "DT"), "LB": ("LB",), "ILB": ("LB",),
    "CB": ("CB",), "S": ("S",), "FS": ("S",), "SS": ("S",), "DB": ("CB", "S"),
}
# FB and NB are on the chart but are not every-down starters.
NON_STARTER_SPOTS = {"FB", "NB"}


def _download(url, dest):
    import requests
    r = requests.get(url, timeout=120, headers={"User-Agent": "DraftVision/1.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    return dest


def _contract_end(season_history):
    """Last league year the contract pays cash (see module docstring)."""
    last = None
    for row in season_history if season_history is not None else []:
        try:
            year = int(row["year"])
        except (TypeError, ValueError):
            continue                       # the 'Total' row
        cash = row.get("cash_paid")
        if cash is not None and cash == cash and cash > 0:
            last = year if last is None else max(last, year)
    return last


def _cap_in(season_history, year):
    for row in season_history if season_history is not None else []:
        if str(row.get("year")) == str(year):
            v = row.get("cap_number")
            return round(float(v), 2) if v is not None and v == v else None
    return None


def _status(ends, season):
    if ends is None:
        return "unknown"
    if ends <= season:
        return "expiring"
    if ends == season + 1:
        return "next"
    return "signed"


def _num(v, nd=2):
    return round(float(v), nd) if v is not None and v == v else None


def contract_index(contracts, season):
    """gsis_id -> contract summary, from the active OverTheCap rows."""
    active = contracts[contracts.is_active & contracts.gsis_id.notna()]
    active = active.sort_values("year_signed").drop_duplicates("gsis_id", keep="last")
    out = {}
    for r in active.itertuples(index=False):
        ends = _contract_end(r.season_history)
        kind = None
        if r.contract_history is not None:
            for row in r.contract_history:
                if row.get("status") == "Active":
                    kind = row.get("contract_type")
        if ends is None and kind in (None, "Drafted", "UDFA") and r.years == r.years:
            # A fresh rookie deal has no year-by-year table yet; with no
            # earlier contract to extend, signing year + length is exact.
            ends = int(r.year_signed + r.years - 1)
        out[r.gsis_id] = {
            "ends": ends,
            "status": _status(ends, season),
            "signed": int(r.year_signed) if r.year_signed == r.year_signed else None,
            "apy": _num(r.apy),
            "guaranteed": _num(r.guaranteed),
            "cap_next": _cap_in(r.season_history, season + 1),
            "type": kind,
            "rookie_deal": kind in ("Drafted", "UDFA"),
            "otc": r.player_page if isinstance(r.player_page, str) else None,
        }
    return out


def build(depth, contracts, draft_order, season):
    latest = depth[depth.dt == depth.groupby("team").dt.transform("max")]
    cidx = contract_index(contracts, season)
    slots = {t["team"]: t for t in (draft_order or {}).get("teams", [])}
    teams = {}
    for abbr, tdf in latest.groupby("team"):
        if abbr not in TEAM_NAMES:
            continue
        spots = []
        front = "3-4" if tdf.pos_grp.str.startswith("Base 3-4").any() else "4-3"
        group_of = GROUP_OF_34 if front == "3-4" else GROUP_OF
        for (grp, pos, slot), sdf in tdf.groupby(["pos_grp", "pos_abb", "pos_slot"], sort=False):
            unit = "offense" if grp == "3WR 1TE" else "special" if grp == "Special Teams" else "defense"
            players = []
            for r in sdf.sort_values("pos_rank").head(MAX_DEPTH).itertuples(index=False):
                if not isinstance(r.player_name, str) or not r.player_name.strip():
                    continue               # ESPN lists the odd empty slot
                c = cidx.get(r.gsis_id)
                players.append({
                    "name": r.player_name,
                    "espn_id": str(r.espn_id) if r.espn_id == r.espn_id and r.espn_id else None,
                    "contract": c or {"ends": None, "status": "unknown"},
                })
            if not players:
                continue
            spots.append({"unit": unit, "pos": pos, "slot": int(slot), "players": players})
        # Group summary: every-down starters whose deal ends this season.
        groups = {}
        for s in spots:
            g = group_of.get(s["pos"])
            if s["unit"] == "special" or not g or s["pos"] in NON_STARTER_SPOTS or not s["players"]:
                continue
            st = s["players"][0]
            d = groups.setdefault(g, {"group": g, "label": GROUP_LABEL[g], "starters": 0,
                                      "expiring": 0, "next": 0, "names": []})
            d["starters"] += 1
            if st["contract"]["status"] == "expiring":
                d["expiring"] += 1
                d["names"].append(st["name"])
            elif st["contract"]["status"] == "next":
                d["next"] += 1
        ordered = sorted(groups.values(), key=lambda d: (-d["expiring"] / d["starters"], -d["expiring"], -d["next"]))
        draft = slots.get(abbr)
        teams[abbr] = {
            "team": abbr,
            "name": TEAM_NAMES[abbr],
            "front": front,
            "chart_updated": str(tdf.dt.max())[:10],
            "draft": draft,
            "starters": sum(d["starters"] for d in groups.values()),
            "starters_expiring": sum(d["expiring"] for d in groups.values()),
            "groups": ordered,
            "spots": spots,
        }
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "season": season,
        "draft_year": season + 1,
        "draft_through_week": (draft_order or {}).get("through_week"),
        "trades_as_of": (draft_order or {}).get("trades_as_of"),
        "trades_source": (draft_order or {}).get("trades_source"),
        "source": "nflverse-data (CC-BY 4.0): depth charts via ESPN, contracts via OverTheCap",
        "teams": teams,
    }


# ── Serving (read-only) ──────────────────────────────────────────────────────
_CACHE = {"mtime": None, "data": None}


def _load():
    try:
        mtime = os.path.getmtime(OUT_PATH)
    except OSError:
        return None
    if _CACHE["mtime"] != mtime:
        with open(OUT_PATH) as fh:
            _CACHE["data"] = json.load(fh)
        _CACHE["mtime"] = mtime
    return _CACHE["data"]


_META = ("generated_at", "season", "draft_year", "draft_through_week", "trades_as_of", "trades_source", "source")


def _first_pick(t):
    picks = (t.get("draft") or {}).get("picks")
    if picks is None:                       # file built before trades were applied
        return (t.get("draft") or {}).get("slot_mean", 99)
    return picks[0]["slot_mean"] if picks else 99


def index_payload():
    """Light list for the team picker, ordered by each team's first pick."""
    d = _load()
    if not d:
        return None
    teams = [{"team": t["team"], "name": t["name"], "draft": t["draft"],
              "starters": t["starters"], "starters_expiring": t["starters_expiring"],
              "top_group": (t["groups"][0]["group"] if t["groups"] and t["groups"][0]["expiring"] else None)}
             for t in d["teams"].values()]
    teams.sort(key=lambda t: (_first_pick(t), (t["draft"] or {}).get("slot_mean", 99)))
    return {k: d.get(k) for k in _META} | {"teams": teams}


def prospect_fits(team, prospects, draft_year, per_pick=4, max_picks=3):
    """For each first-round pick the team holds: prospects the board projects
    inside that pick's likely range who play a position where a starter's
    contract is running out.

    Need weight per position group = 2 x starters whose deal ends after this
    season + 1 x starters whose deal ends a year later. Within a pick the list
    is ordered by that weight, then by projected pick. Nothing here predicts a
    selection: it is the overlap of two lists.
    """
    need = {g["group"]: g for g in team.get("groups", []) if g["expiring"] or g["next"]}
    weight = {k: 2 * g["expiring"] + g["next"] for k, g in need.items()}
    pool = sorted((p for p in prospects or []
                   if p.get("draft_class") == draft_year and p.get("projected_pick")
                   and any(g in weight for g in PROSPECT_GROUPS.get((p.get("position") or "").upper(), ()))),
                  key=lambda p: p["projected_pick"])
    out, used = [], set()
    for pick in ((team.get("draft") or {}).get("picks") or [])[:max_picks]:
        rows = []
        for p in pool:
            if not pick["slot_p10"] <= p["projected_pick"] <= pick["slot_p90"]:
                continue
            key = (p.get("name"), p.get("team"))
            if key in used:
                continue
            g = max((g for g in PROSPECT_GROUPS[(p["position"] or "").upper()] if g in weight), key=weight.get)
            rows.append({"name": p.get("name"), "position": p.get("position"), "college": p.get("team"),
                         "projected_pick": p["projected_pick"], "grade": p.get("grade"),
                         "group": g, "group_label": GROUP_LABEL[g], "weight": weight[g],
                         "expiring": need[g]["expiring"], "next": need[g]["next"],
                         "starters": need[g]["starters"]})
        rows.sort(key=lambda r: (-r["weight"], r["projected_pick"]))
        rows = rows[:per_pick]
        used.update((r["name"], r["college"]) for r in rows)
        out.append({k: pick[k] for k in ("slot_p10", "slot_p50", "slot_p90", "from", "own")} | {"prospects": rows})
    return out


def team_payload(abbr, prospects=None):
    d = _load()
    if not d:
        return None
    t = d["teams"].get((abbr or "").upper())
    if not t:
        return None
    out = {k: d.get(k) for k in _META} | {"team": t}
    if prospects is not None:
        out["fits"] = prospect_fits(t, prospects, d["draft_year"])
    return out


_ORDER_CACHE = {"mtime": None, "data": None}


def draft_order_payload():
    """Round-one order for the mock draft: 32 picks, each with the team that
    holds it and the team whose record earned it."""
    try:
        mtime = os.path.getmtime(DRAFT_ORDER_PATH)
    except OSError:
        return None
    if _ORDER_CACHE["mtime"] != mtime:
        with open(DRAFT_ORDER_PATH) as fh:
            _ORDER_CACHE["data"] = json.load(fh)
        _ORDER_CACHE["mtime"] = mtime
    d = _ORDER_CACHE["data"]
    if not d.get("first_round"):
        return None
    return {k: d.get(k) for k in ("generated_at", "season", "draft_year", "through_week", "trades_as_of")} | {
        "first_round": [{"pick": p["pick"], "team": TEAM_NAMES[p["team"]], "from": TEAM_NAMES[p["from"]]}
                        for p in d["first_round"]]}


def main() -> int:
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depth", help="local depth_charts_<season>.parquet")
    ap.add_argument("--contracts", help="local historical_contracts.parquet")
    ap.add_argument("--season", type=int)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    draft_order = json.load(open(DRAFT_ORDER_PATH)) if os.path.exists(DRAFT_ORDER_PATH) else None
    season = args.season or (draft_order or {}).get("season")
    if not season:
        today = datetime.date.today()
        season = today.year if today.month >= 3 else today.year - 1
    with tempfile.TemporaryDirectory() as tmp:
        depth_path = args.depth or _download(f"{RELEASES}/depth_charts/depth_charts_{season}.parquet",
                                             os.path.join(tmp, "depth.parquet"))
        contracts_path = args.contracts or _download(f"{RELEASES}/contracts/historical_contracts.parquet",
                                                     os.path.join(tmp, "contracts.parquet"))
        depth = pd.read_parquet(depth_path)
        contracts = pd.read_parquet(contracts_path)
    payload = build(depth, contracts, draft_order, season)
    if len(payload["teams"]) != 32:
        print(f"expected 32 teams, got {len(payload['teams'])}; not writing", file=sys.stderr)
        return 2
    matched = [p["contract"]["status"] != "unknown"
               for t in payload["teams"].values() for s in t["spots"] for p in s["players"]]
    share = sum(matched) / max(len(matched), 1)
    if share < 0.8:
        print(f"only {share:.0%} of charted players matched a contract; not writing", file=sys.stderr)
        return 2
    json.dump(payload, open(args.out, "w"), separators=(",", ":"), allow_nan=False)
    print(f"{season}: 32 teams, {len(matched)} charted players, {share:.1%} with a contract -> {args.out}")
    for t in sorted(payload["teams"].values(), key=lambda t: -t["starters_expiring"])[:6]:
        top = ", ".join(f"{g['group']} {g['expiring']}/{g['starters']}" for g in t["groups"][:3] if g["expiring"])
        print(f"  {t['team']:<4}{t['starters_expiring']:>2}/{t['starters']} starters end after {season}: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
