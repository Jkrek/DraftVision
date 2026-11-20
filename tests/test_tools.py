"""Tool layer: strict arguments, read-only results, one accuracy statement.
Rate limiter: windows, buckets, loopback."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dv_ratelimit as rl
import dv_tools as t

BOARD = [
    {"name": "Ada Quick", "team": "Texas Longhorns", "position": "QB", "draft_class": 2027, "grade": "A+",
     "success_probability": 80.0, "projected_pick": 1, "conference_tier": 1, "data_source": "espn_live",
     "production_score": 90.0, "projected_career_av": 50.0, "pick_range": {"lo": 1, "hi": 9, "confidence": 0.72,
                                                                          "confidence_target": 0.8}},
    {"name": "Ben Quick", "team": "Rice Owls", "position": "QB", "draft_class": 2027, "grade": "B",
     "success_probability": 30.0, "projected_pick": 40, "conference_tier": 5, "data_source": "roster_hint"},
    {"name": "Cy Rush", "team": "Texas Longhorns", "position": "EDGE", "draft_class": 2027, "grade": "A-",
     "success_probability": 55.0, "projected_pick": 12, "conference_tier": 1, "data_source": "espn_live"},
    {"name": "Di Later", "team": "Texas Longhorns", "position": "DE", "draft_class": 2028, "grade": "C",
     "success_probability": 10.0, "projected_pick": 3, "conference_tier": 1, "data_source": "espn_live"},
]


def _use_fixture():
    t.configure(board=lambda: (BOARD, "2099-01-01T00:00:00+00:00"))


def _raises(name, args, needle):
    try:
        t.call(name, args)
    except t.ToolError as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(f"{name}({args}) should have been rejected")


def test_every_tool_has_a_closed_schema_and_a_description():
    assert len(t.TOOLS) == 8
    for spec in t.tool_specs():
        s = spec["input_schema"]
        assert s["type"] == "object" and s["additionalProperties"] is False, spec["name"]
        assert set(s["required"]) <= set(s["properties"]) and len(spec["description"]) > 20
    json.dumps(t.tool_specs())


def test_bad_arguments_are_rejected_before_anything_runs():
    _use_fixture()
    _raises("query_board", {"pos": "QB"}, "unknown argument")
    _raises("query_board", {"limit": 500}, "between 1 and 50")
    _raises("query_board", {"limit": True}, "must be integer")
    _raises("query_board", {"position": "QUARTERBACK"}, "one of")
    _raises("search_players", {}, "missing required")
    _raises("search_players", {"query": "a"}, "characters")
    _raises("compare_players", {"players": ["ada-quick-texas-longhorns"]}, "2 to 4 items")
    _raises("drop_table", {}, "unknown tool")


def test_query_board_filters_and_sorts():
    _use_fixture()
    r = t.call("query_board", {"position": "QB", "draft_class": 2027})["result"]
    assert [p["name"] for p in r["players"]] == ["Ada Quick", "Ben Quick"] and r["total"] == 2
    r = t.call("query_board", {"position": "QB", "verified_only": True})["result"]
    assert [p["name"] for p in r["players"]] == ["Ada Quick"]
    # EDGE covers the board's DE label too; class filter still applies
    r = t.call("query_board", {"position": "EDGE", "draft_class": 2027})["result"]
    assert [p["name"] for p in r["players"]] == ["Cy Rush"]
    r = t.call("query_board", {"grade_min": "A-", "sort": "success_probability"})["result"]
    assert [p["name"] for p in r["players"]] == ["Ada Quick", "Cy Rush"]
    r = t.call("query_board", {"conference_tier_max": 1, "limit": 1})["result"]
    assert r["total"] == 3 and r["returned"] == 1
    assert all("data_source" in p for p in r["players"])


def test_players_resolve_by_id_or_exact_name_and_ambiguity_is_reported():
    _use_fixture()
    env = t.call("get_player_report", {"player": "ada-quick-texas-longhorns"})
    assert env["board_generated_at"].startswith("2099") and env["model_version"] == t.MODEL_VERSION
    r = env["result"]
    assert r["ranks"] == {"class_rank": 1, "position_rank": 1, "class_size": 3}
    assert r["pick_range"]["measured_coverage"] == 0.72
    assert t.call("get_player_report", {"player": "Cy Rush"})["result"]["player_id"] == "cy-rush-texas-longhorns"
    _raises("get_player_report", {"player": "Quick"}, "matches 2 players")
    _raises("get_player_report", {"player": "Nobody Atall"}, "no player matches")
    r = t.call("compare_players", {"players": ["Ada Quick", "Cy Rush"]})["result"]
    assert r["leader_by_metric"]["projected_pick"] == "ada-quick-texas-longhorns" and r["same_draft_class"]
    r = t.call("search_players", {"query": "quick"})["result"]
    assert [p["name"] for p in r["players"]] == ["Ada Quick", "Ben Quick"]


def test_model_card_matches_the_committed_evaluation_files():
    _use_fixture()
    card = t.call("get_model_card")["result"]
    meta = json.load(open(t.METADATA_PATH))
    cv = json.load(open(t.ROLLING_CV_PATH))
    assert card["frozen_holdout"]["draft_grade_accuracy"] == meta["evaluation"]["draft_grade"]["ensemble_raw_mean"]["accuracy"]
    assert card["frozen_holdout"]["success_auc"] == meta["evaluation"]["success"]["ensemble_calibrated"]["auc"]
    assert card["rolling_backtest"]["draft_grade_accuracy"]["mean"] == cv["summary"]["grade_acc"]["mean"]
    assert card["board"] == {"players": 4, "generated_at": "2099-01-01T00:00:00+00:00", "verified_stats": 3,
                             "verified_share": 0.75}
    assert len(card["limits"]) >= 4


def test_committed_board_answers_through_every_tool():
    t._PROVIDERS.clear()
    t._INDEX["key"] = None
    top = t.call("query_board", {"draft_class": 2027, "limit": 3})["result"]["players"]
    assert len(top) == 3 and top[0]["projected_pick"] == 1
    assert t.call("get_player_report", {"player": top[0]["player_id"]})["result"]["name"] == top[0]["name"]
    assert "risers" in t.call("get_movers", {"direction": "risers", "limit": 3})["result"]
    m = t.call("get_market_edge", {"limit": 3})["result"]
    assert "Analysis only" in m["note"] and len(m["rows"]) <= 3
    n = t.call("get_team_needs", {"team": "bengals"})["result"]
    assert n["team"] == "CIN" and n["position_groups"]
    _raises("get_team_needs", {"team": "zzz"}, "unknown NFL team")


def test_rate_limit_window_buckets_and_loopback():
    rl.reset()
    limit, window = rl.LIMITS["owner"]
    assert all(rl.check("owner", "1.2.3.4", now=100.0)[0] for _ in range(limit))
    ok, retry, _ = rl.check("owner", "1.2.3.4", now=100.5)
    assert not ok and 1 <= retry <= window
    assert rl.check("owner", "5.6.7.8", now=100.5)[0]            # another address is unaffected
    assert rl.check("tools", "1.2.3.4", now=100.5)[0]            # another bucket is unaffected
    assert rl.check("owner", "1.2.3.4", now=100.0 + window)[0]   # the window rolls over
    assert rl.bucket_for("POST", "/api/mock-draft/upload") == "owner"
    assert rl.bucket_for("POST", "/api/tools/query_board") == "tools"
    assert rl.bucket_for("POST", "/predict") == "predict"
    assert rl.bucket_for("GET", "/api/prospects") is None
    assert rl.bucket_for("GET", "/api/toolsx") is None
    assert rl.client_ip({"Fly-Client-IP": "9.9.9.9"}, "10.0.0.1") == "9.9.9.9"
    assert rl.client_ip({}, "10.0.0.1") == "10.0.0.1"
    rl.reset()


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
