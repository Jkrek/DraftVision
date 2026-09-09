#!/usr/bin/env python3
"""Sanity checks for the weekly board ledger — run after postprocess_board.py.

Asserts that the LATEST training_data/board_history/board_<date>.json carries
the scorable-ledger fields (provenance at the top level — model_git_sha must
be a real sha, not 'unknown', plus metadata_sha256 — and the REQUIRED row
fields on every row; OPTIONAL row fields are omitted when null, so their
coverage is printed rather than enforced), that its .sha256 sidecar matches,
and that no riser / faller in board_movers.json slipped past the scorability
gate (games_played >= MOVER_MIN_GAMES, data_source == 'espn_live', and the
prior row verified too when the prior snapshot records data_source).

It also exercises the pick-interval re-rank path
(postprocess_board.apply_nominal_picks -> slim_board) on synthetic rows every
run, because the live cache may predate pick_range_raw and would leave that
path untested — today's ledger legitimately has no pick_lo / pick_hi then.

    python scripts/check_snapshot.py               # check files on disk
    python scripts/check_snapshot.py --recompute   # also recompute movers from
                                                   # the current cache vs the
                                                   # prior-week snapshot and
                                                   # gate-check that in memory

Exit 0 on success, 1 with an itemised failure list otherwise.
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_prospect_cache as bpc  # noqa: E402
import postprocess_board as ppb     # noqa: E402

SNAPSHOT_TOP_FIELDS = ("generated_at", "snapshot_date", "model_git_sha",
                       "metadata_sha256", "features_n", "cache_sha256", "players")
SNAPSHOT_ROW_REQUIRED = bpc.SNAPSHOT_ROW_REQUIRED
SNAPSHOT_ROW_OPTIONAL = bpc.SNAPSHOT_ROW_OPTIONAL
SNAPSHOT_ROW_KNOWN = set(SNAPSHOT_ROW_REQUIRED) | set(SNAPSHOT_ROW_OPTIONAL)
MOVER_FIELDS = ("name", "team", "delta_prob", "delta_rank",
                "games_played", "data_source", "prior_data_source", "basis")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def latest_snapshot_path(history_dir):
    dated = []
    for fn in os.listdir(history_dir) if os.path.isdir(history_dir) else []:
        m = re.fullmatch(r"board_(\d{4}-\d{2}-\d{2})\.json", fn)
        if m:
            dated.append((m.group(1), os.path.join(history_dir, fn)))
    return sorted(dated)[-1][1] if dated else None


def check_snapshot(path, fails):
    base = os.path.basename(path)
    with open(path) as f:
        snap = json.load(f)
    for k in SNAPSHOT_TOP_FIELDS:
        if k not in snap:
            fails.append(f"{base}: missing top-level field '{k}'")
    if "cache_file" in snap:
        fails.append(f"{base}: stale top-level field 'cache_file' (dropped; cache_sha256 is the key)")
    sha = snap.get("model_git_sha")
    if sha in (None, "", "unknown"):
        fails.append(f"{base}: model_git_sha is {sha!r} — models/metadata.json was not "
                     f"readable at write time (MODEL_METADATA_FILE={bpc.MODEL_METADATA_FILE})")
    elif not SHA_RE.match(str(sha)):
        fails.append(f"{base}: model_git_sha {sha!r} is not a 40-hex git sha")
    if not snap.get("metadata_sha256"):
        fails.append(f"{base}: metadata_sha256 is empty")
    players = snap.get("players") or []
    if not players:
        fails.append(f"{base}: no players")
    missing, present, nulls, unknown = {}, {}, {}, {}
    for row in players:
        for k in SNAPSHOT_ROW_REQUIRED:
            if k not in row:
                missing[k] = missing.get(k, 0) + 1
        for k in SNAPSHOT_ROW_OPTIONAL:
            if k in row:
                present[k] = present.get(k, 0) + 1
        for k, v in row.items():
            if v is None:
                nulls[k] = nulls.get(k, 0) + 1
            if k not in SNAPSHOT_ROW_KNOWN:
                unknown[k] = unknown.get(k, 0) + 1
    for k, n in sorted(missing.items()):
        fails.append(f"{base}: required row field '{k}' missing on {n}/{len(players)} rows")
    for k, n in sorted(nulls.items()):
        fails.append(f"{base}: null-valued key '{k}' kept on {n} rows (nulls must be omitted)")
    for k, n in sorted(unknown.items()):
        fails.append(f"{base}: unknown row key '{k}' on {n} rows")
    for row in players:
        if any(isinstance(row.get(k), (dict, list)) for k in row):
            fails.append(f"{base}: nested blob on row {row.get('name')!r}")
            break
    if players:
        cov = ", ".join(f"{k}={present.get(k, 0)}" for k in SNAPSHOT_ROW_OPTIONAL)
        print(f"optional-field coverage ({len(players)} rows): {cov}")
    # sha256 sidecar
    side = bpc.snapshot_sidecar_path(path)
    if not os.path.exists(side):
        fails.append(f"missing sidecar {os.path.basename(side)}")
    else:
        with open(side) as f:
            recorded = (f.read().split() or [""])[0]
        actual = bpc._sha256_file(path)
        if recorded != actual:
            fails.append(f"{os.path.basename(side)}: recorded {recorded[:12]}… != actual {actual[:12]}…")
    return snap


def check_movers(movers, label, fails):
    rows = (movers.get("risers") or []) + (movers.get("fallers") or [])
    if len(rows) != movers.get("count"):
        fails.append(f"{label}: count={movers.get('count')} but {len(rows)} rows listed")
    for m in rows:
        for k in MOVER_FIELDS:
            if k not in m:
                fails.append(f"{label}: mover {m.get('name')!r} missing '{k}'")
        prior = {"data_source": m.get("prior_data_source")}
        if not bpc.mover_scorable(m, prior):
            fails.append(f"{label}: mover {m.get('name')!r} fails gate "
                         f"(games_played={m.get('games_played')}, "
                         f"data_source={m.get('data_source')}, "
                         f"prior_data_source={m.get('prior_data_source')})")
        if m.get("basis") == "first verified line":
            fails.append(f"{label}: mover {m.get('name')!r} is a first verified line "
                         f"(must be excluded from risers/fallers)")
    return len(rows)


# ── Unit-style checks on synthetic rows ──────────────────────────────────────

def _synthetic_rows():
    """Two rankable classes (>= MIN_CLASS_ROWS rows) plus one thin class.
    Raw model picks are spaced 2 apart so a raw bound between two rows ranks
    unambiguously; served pick_range values are deliberately WRONG (ranked
    against a fictitious previous board) so a test can tell which was used."""
    rows = []
    for cls in (2027, 2028):
        for i in range(ppb.MIN_CLASS_ROWS + 20):
            raw = 10.0 + 2 * i
            rows.append({"name": f"p{cls}-{i}", "team": "T", "position": "WR",
                         "grade": "B", "success_probability": 50.0,
                         "draft_class": cls, "model_pick": raw,
                         "pick_range_raw": {"lo": raw - 5, "hi": raw + 5},
                         "pick_range": {"lo": 999, "hi": 999},
                         "data_source": "espn_live", "games_played": 3})
    # raw absent -> served nominal fallback (raw pick past the grid so the
    # extra rows do not shift the ranks asserted below)
    rows.append({"name": "served-only", "team": "T", "position": "QB", "grade": "C",
                 "success_probability": 40.0, "draft_class": 2027, "model_pick": 1000.0,
                 "pick_range": {"lo": 7, "hi": 40}, "data_source": "roster_hint"})
    # neither -> bounds omitted from the ledger row
    rows.append({"name": "no-bounds", "team": "T", "position": "RB", "grade": "C",
                 "success_probability": 30.0, "draft_class": 2027, "model_pick": 1001.0,
                 "data_source": "roster_hint", "projected_career_av": None})
    # thin class: raw present but unrankable -> served fallback
    for i in range(5):
        rows.append({"name": f"thin-{i}", "team": "T", "position": "S", "grade": "C",
                     "success_probability": 30.0, "draft_class": 2030,
                     "model_pick": 50.0 + i,
                     "pick_range_raw": {"lo": 40.0, "hi": 60.0},
                     "pick_range": {"lo": 3, "hi": 9}, "data_source": "espn_live"})
    return rows


def check_rerank_path(fails):
    """apply_nominal_picks must rank pick_range_raw through the rebuilt-board
    index (never the served values), fall back to served only when raw is
    absent/unrankable, and slim_board must carry the result with nulls
    omitted."""
    rows = _synthetic_rows()
    ppb.apply_nominal_picks(rows)
    by_name = {r["name"]: r for r in rows}

    # class 2027 row i=10: raw 30 -> rank 11 (10 grid values below it);
    # lo raw 25 -> rank 9 (bisect_left over 10,12,...: 10..24 = 8 values
    # below, +1); hi raw 35 -> rank 14 (10..34 = 13 below, +1).
    r = by_name["p2027-10"]
    exp = (11, 9, 14)
    got = (r.get("projected_pick"), r.get("pick_lo"), r.get("pick_hi"))
    if got != exp:
        fails.append(f"rerank: p2027-10 (projected_pick, pick_lo, pick_hi) = {got}, expected {exp}")
    if r.get("pick_lo") == 999:
        fails.append("rerank: served pick_range used although pick_range_raw was present")
    # the same raw values rank identically in the other class (same spacing)
    r2 = by_name["p2028-10"]
    if (r2.get("pick_lo"), r2.get("pick_hi")) != (9, 14):
        fails.append(f"rerank: class isolation broken — p2028-10 bounds {r2.get('pick_lo')},{r2.get('pick_hi')}")
    # served-only (raw 1000, behind the 120 grid rows) ranks 121st but keeps
    # its SERVED bounds because it has no pick_range_raw
    s = by_name["served-only"]
    if (s.get("projected_pick"), s.get("pick_lo"), s.get("pick_hi")) != (121, 7, 40):
        fails.append(f"rerank: served fallback wrong — {s.get('projected_pick')},{s.get('pick_lo')},{s.get('pick_hi')}")
    nb = by_name["no-bounds"]
    if "pick_lo" in nb or "pick_hi" in nb:
        fails.append("rerank: bounds set on a row with neither raw nor served interval")
    t = by_name["thin-0"]
    if t.get("projected_pick") is not None or (t.get("pick_lo"), t.get("pick_hi")) != (3, 9):
        fails.append(f"rerank: thin class should be unrankable + served fallback, got "
                     f"{t.get('projected_pick')},{t.get('pick_lo')},{t.get('pick_hi')}")
    # lo/hi ordering is enforced even if raw came in reversed (no model_pick,
    # so the row does not enter the class index)
    rev = {"name": "rev", "team": "T", "draft_class": 2027, "model_pick": None,
           "pick_range_raw": {"lo": 35.0, "hi": 25.0}}
    ppb.apply_nominal_picks(rows + [rev])
    if (rev.get("pick_lo"), rev.get("pick_hi")) != (9, 14):
        fails.append(f"rerank: reversed raw bounds not normalised — {rev.get('pick_lo')},{rev.get('pick_hi')}")

    # ledger rows: bounds carried, nulls omitted, no nested blobs
    slim = {s["name"]: s for s in bpc.slim_board(rows)}
    if (slim["p2027-10"].get("pick_lo"), slim["p2027-10"].get("pick_hi")) != (9, 14):
        fails.append("slim_board: re-ranked pick_lo/pick_hi not carried onto the ledger row")
    if (slim["served-only"].get("pick_lo"), slim["served-only"].get("pick_hi")) != (7, 40):
        fails.append("slim_board: served fallback bounds not carried onto the ledger row")
    nbs = slim["no-bounds"]
    for k in ("pick_lo", "pick_hi", "projected_career_av", "games_played", "espn_id"):
        if k in nbs:
            fails.append(f"slim_board: null key '{k}' not omitted")
    for k in SNAPSHOT_ROW_REQUIRED:
        if k not in nbs:
            fails.append(f"slim_board: required key '{k}' missing")
    if any(isinstance(v, (dict, list)) for v in slim["p2027-10"].values()):
        fails.append("slim_board: nested blob leaked onto ledger row")

    # mover gate: a roster_hint -> espn_live flip is a first verified line,
    # a missing prior data_source passes, an espn_live prior passes
    cur = {"games_played": 3, "data_source": "espn_live"}
    cases = [({"data_source": "roster_hint"}, False, "first verified line"),
             ({}, True, "3 games"),
             ({"data_source": None}, True, "3 games"),
             ({"data_source": "espn_live"}, True, "3 games"),
             (None, True, "3 games")]
    for prior, want, basis in cases:
        if bpc.mover_scorable(cur, prior) != want or bpc.mover_basis(cur, prior) != basis:
            fails.append(f"mover gate: prior={prior} -> scorable={bpc.mover_scorable(cur, prior)} "
                         f"basis={bpc.mover_basis(cur, prior)!r}, expected {want}/{basis!r}")
    cur_snap = [dict(cur, name="X", team="T", success_probability=60.0)]
    prior_snap = {"generated_at": "t0", "players": [
        {"name": "X", "team": "T", "success_probability": 40.0, "rank": 1,
         "data_source": "roster_hint"}]}
    mv = bpc.compute_movers(cur_snap, prior_snap, "t1")
    if mv["count"] != 0 or mv["gate"]["excluded_first_verified"] != 1:
        fails.append(f"compute_movers: roster_hint->espn_live flip surfaced as a mover ({mv['gate']})")
    prior_snap["players"][0].pop("data_source")
    mv = bpc.compute_movers(cur_snap, prior_snap, "t1")
    if mv["count"] != 1 or mv["risers"][0].get("prior_data_source") is not None:
        fails.append("compute_movers: missing prior data_source should pass the gate")
    print("synthetic: re-rank path, null omission, and prior-row mover gate exercised")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history-dir", default=bpc.HISTORY_DIR)
    ap.add_argument("--movers", default=bpc.MOVERS_FILE)
    ap.add_argument("--cache", default=bpc.OUTPUT_FILE)
    ap.add_argument("--recompute", action="store_true",
                    help="Recompute movers from --cache vs the prior-week snapshot "
                         "in memory and gate-check the result as well.")
    args = ap.parse_args()
    os.chdir(ROOT)
    fails = []

    check_rerank_path(fails)

    snap_path = latest_snapshot_path(args.history_dir)
    if not snap_path:
        fails.append(f"no board_<date>.json in {args.history_dir}")
        snap = None
    else:
        snap = check_snapshot(snap_path, fails)
        print(f"snapshot: {snap_path} ({len(snap.get('players') or [])} rows, "
              f"{os.path.getsize(snap_path) / 1e6:.2f} MB, "
              f"model {str(snap.get('model_git_sha'))[:10]}, "
              f"metadata {str(snap.get('metadata_sha256'))[:10]}, "
              f"features_n={snap.get('features_n')})")

    if os.path.exists(args.movers):
        with open(args.movers) as f:
            movers = json.load(f)
        n = check_movers(movers, os.path.basename(args.movers), fails)
        gate = movers.get("gate") or {}
        print(f"movers:   {args.movers} ({n} rows, since {movers.get('since')}, "
              f"gate excluded={gate.get('excluded')}, "
              f"first-verified={gate.get('excluded_first_verified')})")
    else:
        fails.append(f"missing {args.movers}")

    if args.recompute:
        with open(args.cache) as f:
            cache = json.load(f)
        prospects = cache.get("prospects") or []
        date_str = bpc._snapshot_date_from_iso(cache.get("generated_at"), args.cache)
        bpc.HISTORY_DIR = args.history_dir
        prior = bpc.load_latest_snapshot(before=date_str)
        if not prior:
            fails.append(f"--recompute: no snapshot dated before {date_str}")
        else:
            live = bpc.compute_movers(prospects, prior, cache.get("generated_at"))
            n = check_movers(live, "recomputed movers", fails)
            print(f"recompute: {n} movers from {len(prospects)} cache rows vs "
                  f"{prior.get('generated_at')} (gate excluded={live['gate']['excluded']})")

    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for msg in fails:
            print(f"  - {msg}")
        return 1
    print("\nOK — ledger fields present, sidecar matches, every mover passes the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
