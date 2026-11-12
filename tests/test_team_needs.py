"""Team Needs: contract end year and the status buckets."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dv_team_needs as tn


def _rows(*pairs):
    rows = [{"year": str(y), "cash_paid": c, "cap_number": c} for y, c in pairs]
    return rows + [{"year": "Total", "cash_paid": 99.0, "cap_number": 99.0}]


def test_end_year_is_last_year_with_cash_not_a_void_year():
    # extension signed 2025 for 4 new years on top of a deal through 2025,
    # followed by two void years that carry cap but pay nothing
    hist = _rows((2024, 4.8), (2025, 40.6), (2026, 33.7), (2029, 44.8), (2030, 0.0), (2031, 0.0))
    assert tn._contract_end(hist) == 2029


def test_end_year_none_when_nothing_is_paid():
    assert tn._contract_end(_rows((2026, float("nan")))) is None
    assert tn._contract_end(None) is None


def test_status_buckets():
    assert tn._status(2025, 2026) == "expiring"      # already past: still open
    assert tn._status(2026, 2026) == "expiring"
    assert tn._status(2027, 2026) == "next"
    assert tn._status(2028, 2026) == "signed"
    assert tn._status(None, 2026) == "unknown"


def test_committed_file_is_complete_and_strict_json():
    import json
    with open(tn.OUT_PATH) as fh:
        d = json.load(fh, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert len(d["teams"]) == 32
    for t in d["teams"].values():
        keys = {s["pos"] for s in t["spots"]}
        assert {"QB", "LT", "C", "RT", "FS", "SS", "LCB", "RCB"} <= keys, t["team"]
        assert all(s["players"] and isinstance(s["players"][0]["name"], str) for s in t["spots"])
    assert tn.team_payload("cin")["team"]["team"] == "CIN"
    assert tn.team_payload("zzz") is None
    assert len(tn.index_payload()["teams"]) == 32


def test_prospect_fits_match_need_positions_inside_the_pick_range():
    team = {
        "groups": [{"group": "CB", "label": "Cornerback", "starters": 2, "expiring": 1, "next": 0},
                   {"group": "IOL", "label": "Interior line", "starters": 3, "expiring": 0, "next": 1},
                   {"group": "QB", "label": "Quarterback", "starters": 1, "expiring": 0, "next": 0}],
        "draft": {"picks": [{"slot_p10": 3, "slot_p50": 5, "slot_p90": 8, "from": "TEN", "own": True}]},
    }
    board = [
        {"name": "Q", "team": "A", "position": "QB", "draft_class": 2027, "projected_pick": 4},   # no need
        {"name": "G", "team": "A", "position": "OL", "draft_class": 2027, "projected_pick": 5},   # generic OL -> IOL
        {"name": "C", "team": "A", "position": "DB", "draft_class": 2027, "projected_pick": 7},   # generic DB -> CB
        {"name": "L", "team": "A", "position": "CB", "draft_class": 2027, "projected_pick": 9},   # outside range
        {"name": "Y", "team": "A", "position": "CB", "draft_class": 2028, "projected_pick": 5},   # wrong class
    ]
    fits = tn.prospect_fits(team, board, 2027)
    assert [(r["name"], r["group"]) for r in fits[0]["prospects"]] == [("C", "CB"), ("G", "IOL")]
    assert tn.prospect_fits({"groups": [], "draft": {"picks": []}}, board, 2027) == []


def test_edge_rushers_follow_the_front():
    assert tn.GROUP_OF["LDE"] == "EDGE" and tn.GROUP_OF["WLB"] == "LB"
    assert tn.GROUP_OF_34["LDE"] == "DT" and tn.GROUP_OF_34["WLB"] == "EDGE"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
