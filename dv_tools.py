#!/usr/bin/env python3
"""Read-only tool layer over the committed board.

One contract for every consumer: the on-site agent, an MCP server, the weekly
jobs and the evals all call these eight functions and get the same JSON.

Rules the layer keeps:
  * Read-only. Nothing here writes a file, calls a model or fetches a URL.
    It reads what the weekly job already committed.
  * Strict inputs. Every tool has a JSON Schema (`TOOLS[name]["input_schema"]`)
    with additionalProperties: false; `call()` validates before running and
    raises ToolError with a message a model can act on.
  * Every player row carries `data_source`, so a caller can say when a grade
    rests on a roster placeholder instead of verified stats.
  * Every result is wrapped with the board date and model version.

Use
  import dv_tools
  dv_tools.call("query_board", {"position": "QB", "draft_class": 2027, "limit": 5})
  dv_tools.tool_specs()        # [{name, description, input_schema}] for an LLM
  python dv_tools.py query_board '{"position":"QB","limit":3}'
"""
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_P = lambda *parts: os.path.join(REPO_ROOT, *parts)  # noqa: E731
BOARD_PATH = _P("training_data", "prospect_cache.json")
MOVERS_PATH = _P("training_data", "board_movers.json")
LEDGER_PATH = _P("training_data", "edge_ledger.json")
METADATA_PATH = _P("models", "metadata.json")
ROLLING_CV_PATH = _P("models", "experiments", "rolling_cv_v4.json")
MODEL_VERSION = "v5"
MAX_LIMIT = 50

GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D"]
GRADE_RANK = {g: i for i, g in enumerate(GRADES)}
# Query position -> the board labels it covers. The board mixes specific and
# generic labels (OL next to OT, DL next to EDGE), so a group pulls in both.
POSITION_GROUPS = {
    "QB": {"QB"}, "RB": {"RB", "FB"}, "WR": {"WR"}, "TE": {"TE"},
    "OL": {"OL", "OT", "G", "C"}, "OT": {"OT", "OL"}, "IOL": {"G", "C", "OL"},
    "DL": {"DL", "DE", "DT", "NT", "EDGE"}, "EDGE": {"EDGE", "DE", "DL"}, "DT": {"DT", "NT", "DL"},
    "LB": {"LB"}, "DB": {"DB", "CB", "S"}, "CB": {"CB", "DB"}, "S": {"S", "DB"}, "K": {"PK"},
}
SORTS = {
    "projected_pick": ("projected_pick", False),
    "success_probability": ("success_probability", True),
    "production_score": ("production_score", True),
    "projected_career_av": ("projected_career_av", True),
}
DATA_SOURCE_NOTE = {
    "espn_live": "graded on verified live ESPN stats",
    "roster_hint": "on a roster but no verified stat line: the grade is a placeholder from position, class and program",
    "nfl_seed": "seed row",
}


class ToolError(ValueError):
    """Bad arguments or an unknown player: safe to show to the caller."""


# ── Data access ──────────────────────────────────────────────────────────────
# Files are read lazily and re-read when they change on disk. A host that
# already holds the board in memory (the Flask app) passes its own getter to
# configure() so the 15k rows are not loaded twice.
_FILES = {}
_PROVIDERS = {}
_INDEX = {"key": None, "by_slug": {}, "ranks": {}}


def configure(**providers):
    """providers: board=callable -> (rows, generated_at)."""
    _PROVIDERS.update(providers)
    _INDEX["key"] = None


def _read(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _FILES.get(path)
    if not hit or hit[0] != mtime:
        with open(path) as fh:
            hit = (mtime, json.load(fh))
        _FILES[path] = hit
    return hit[1]


def _board():
    if "board" in _PROVIDERS:
        rows, generated_at = _PROVIDERS["board"]()
        return rows or [], generated_at
    d = _read(BOARD_PATH) or {}
    return d.get("prospects", []), d.get("generated_at")


def slug(name, team):
    """Player id: the same encoding the site's /player/<slug> pages use."""
    s = f"{name or ''}-{team or ''}".lower()
    return re.sub(r"[^a-z0-9-]", "", re.sub(r"\s+", "-", s))


def _index():
    rows, generated_at = _board()
    key = (id(rows), len(rows), generated_at)
    if _INDEX["key"] != key:
        by_slug, ranks = {}, {}
        for p in rows:
            by_slug.setdefault(slug(p.get("name"), p.get("team")), p)
        # rank inside the draft class, overall and by position label
        classes = {}
        for p in rows:
            if p.get("projected_pick"):
                classes.setdefault(p.get("draft_class"), []).append(p)
        for members in classes.values():
            members.sort(key=lambda p: p["projected_pick"])
            seen = {}
            for i, p in enumerate(members):
                pos = p.get("position")
                seen[pos] = seen.get(pos, 0) + 1
                ranks[slug(p.get("name"), p.get("team"))] = {"class_rank": i + 1, "position_rank": seen[pos],
                                                              "class_size": len(members)}
        _INDEX.update(key=key, by_slug=by_slug, ranks=ranks)
    return _INDEX


def _row(p, full=False):
    """Compact, stable view of one board row."""
    sid = slug(p.get("name"), p.get("team"))
    out = {
        "player_id": sid,
        "name": p.get("name"),
        "position": p.get("position"),
        "college": p.get("team"),
        "draft_class": p.get("draft_class"),
        "class_year": p.get("class_year"),
        "grade": p.get("grade"),
        "success_probability": p.get("success_probability"),
        "draft_grade": p.get("draft_grade"),
        "projected_pick": p.get("projected_pick"),
        "data_source": p.get("data_source"),
    }
    if full:
        rng = p.get("pick_range") or {}
        out.update({
            "pick_range": {"lo": rng.get("lo"), "hi": rng.get("hi"), "measured_coverage": rng.get("confidence"),
                           "target_coverage": rng.get("confidence_target")} if rng else None,
            "projected_career_av": p.get("projected_career_av"),
            "production_score": p.get("production_score"),
            "games_played": p.get("games_played"),
            "conference_tier": p.get("conference_tier"),
            "is_all_american": bool(p.get("is_all_american")),
            "is_award_winner": bool(p.get("is_award_winner")),
            "data_source_note": DATA_SOURCE_NOTE.get(p.get("data_source")),
            "ranks": _index()["ranks"].get(sid),
            "page": f"/player/{sid}",
        })
    return out


def _resolve(player):
    """A player id (slug) or a name -> board row. Ambiguous names raise with
    the candidates, so the caller can pick one instead of guessing."""
    idx = _index()
    key = (player or "").strip()
    hit = idx["by_slug"].get(key.lower())
    if hit:
        return hit
    name = key.lower()
    exact = [p for p in _board()[0] if (p.get("name") or "").lower() == name]
    if len(exact) == 1:
        return exact[0]
    cands = exact or [p for p in _board()[0] if name and name in (p.get("name") or "").lower()]
    if not cands:
        raise ToolError(f"no player matches {player!r}; call search_players first")
    cands.sort(key=lambda p: -(p.get("success_probability") or 0))
    if len(cands) == 1:
        return cands[0]
    listing = "; ".join(f"{slug(p.get('name'), p.get('team'))} ({p.get('position')}, {p.get('team')})" for p in cands[:6])
    raise ToolError(f"{player!r} matches {len(cands)} players; pass one player_id: {listing}")


# ── Tools ────────────────────────────────────────────────────────────────────
def search_players(query, limit=8):
    q = query.strip().lower()
    scored = []
    for p in _board()[0]:
        name, team = (p.get("name") or "").lower(), (p.get("team") or "").lower()
        if q == name:
            s = 0
        elif name.startswith(q):
            s = 1
        elif any(part.startswith(q) for part in name.split()):
            s = 2
        elif q in name:
            s = 3
        elif q in team:
            s = 4
        else:
            continue
        scored.append((s, -(p.get("success_probability") or 0), p))
    scored.sort(key=lambda t: t[:2])
    return {"total": len(scored), "players": [_row(p) for _, _, p in scored[:limit]]}


def get_player_report(player):
    p = _resolve(player)
    out = _row(p, full=True)
    movers = _read(MOVERS_PATH) or {}
    key = ((p.get("name") or ""), (p.get("team") or ""))
    for d in movers.get("all_deltas", []):
        if (d.get("name"), d.get("team")) == key:
            out["week_change"] = {"since": movers.get("since"), "success_probability_points": d.get("delta_prob"),
                                  "board_spots": d.get("delta_rank")}
            break
    out["market"] = _market_rows(slug(p.get("name"), p.get("team")))
    out["explanation_available"] = False   # SHAP factors are computed per /predict call, not stored on the board
    return out


def compare_players(players):
    rows = [_row(_resolve(x), full=True) for x in players]
    if len({r["player_id"] for r in rows}) != len(rows):
        raise ToolError("the same player was passed more than once")
    fields = ("success_probability", "projected_pick", "projected_career_av", "production_score")
    leaders = {}
    for f in fields:
        have = [r for r in rows if r.get(f) is not None]
        if have:
            best = (min if f == "projected_pick" else max)(have, key=lambda r: r[f])
            leaders[f] = best["player_id"]
    return {"players": rows, "leader_by_metric": leaders,
            "same_draft_class": len({r["draft_class"] for r in rows}) == 1}


def query_board(position=None, draft_class=None, college=None, conference_tier_max=None, grade_min=None,
                grade_max=None, min_success_probability=None, verified_only=False,
                sort="projected_pick", descending=None, limit=10):
    rows = _board()[0]
    if position:
        labels = POSITION_GROUPS[position]
        rows = [p for p in rows if (p.get("position") or "").upper() in labels]
    if draft_class is not None:
        rows = [p for p in rows if p.get("draft_class") == draft_class]
    if college:
        c = college.strip().lower()
        rows = [p for p in rows if c in (p.get("team") or "").lower()]
    if conference_tier_max is not None:
        rows = [p for p in rows if (p.get("conference_tier") or 99) <= conference_tier_max]
    if grade_min:      # "at least B+": B+ or better
        rows = [p for p in rows if GRADE_RANK.get(p.get("grade"), 99) <= GRADE_RANK[grade_min]]
    if grade_max:      # "no better than B"
        rows = [p for p in rows if GRADE_RANK.get(p.get("grade"), -1) >= GRADE_RANK[grade_max]]
    if min_success_probability is not None:
        rows = [p for p in rows if (p.get("success_probability") or 0) >= min_success_probability]
    if verified_only:
        rows = [p for p in rows if p.get("data_source") == "espn_live"]
    field, default_desc = SORTS[sort]
    desc = default_desc if descending is None else descending
    have = [p for p in rows if p.get(field) is not None]
    have.sort(key=lambda p: p[field], reverse=desc)
    return {"total": len(rows), "returned": min(limit, len(have)), "sort": sort, "descending": desc,
            "players": [_row(p) for p in have[:limit]]}


def get_movers(direction="both", position=None, limit=10):
    movers = _read(MOVERS_PATH) or {}
    labels = POSITION_GROUPS[position] if position else None

    def pick(rows):
        rows = [r for r in rows if not labels or (r.get("position") or "").upper() in labels]
        return [{"player_id": slug(r.get("name"), r.get("team")), "name": r.get("name"), "position": r.get("position"),
                 "college": r.get("team"), "grade": r.get("grade"), "success_probability": r.get("success_probability"),
                 "success_probability_points": r.get("delta_prob"), "board_spots": r.get("delta_rank"),
                 "basis": r.get("basis"), "data_source": r.get("data_source"),
                 "data_source_changed": r.get("data_source") != r.get("prior_data_source")} for r in rows[:limit]]

    out = {"since": movers.get("since"), "generated_at": movers.get("generated_at"),
           "note": "Movers are gated to players with verified stats in both weeks. A move of thousands of "
                   "board spots is normal low on the board, where many players sit within a point of each other."}
    if direction in ("both", "risers"):
        out["risers"] = pick(movers.get("risers", []))
    if direction in ("both", "fallers"):
        out["fallers"] = pick(movers.get("fallers", []))
    return out


def get_model_card():
    """The one accuracy statement every surface should quote."""
    meta = _read(METADATA_PATH) or {}
    cv = (_read(ROLLING_CV_PATH) or {})
    ev = meta.get("evaluation", {})
    succ = ev.get("success", {}).get("ensemble_calibrated", {})
    grade = ev.get("draft_grade", {}).get("ensemble_raw_mean", {})
    pick = (meta.get("pick_eval") or {}).get("blend_50_50_SERVED", {})
    pi = (meta.get("pick_interval") or {})
    s = cv.get("summary", {})
    rows, generated_at = _board()
    verified = sum(1 for p in rows if p.get("data_source") == "espn_live")

    def ms(k):
        v = s.get(k) or {}
        return {"mean": v.get("mean"), "std": v.get("std")}

    return {
        "model_version": MODEL_VERSION,
        "trained_at": meta.get("generated_at"),
        "git_sha": str(meta.get("git_sha") or "")[:8],
        "features": len(meta.get("features") or []),
        "what_it_predicts": ["NFL success probability", "draft grade (4 buckets)", "projected pick with a range",
                             "projected career value"],
        "rolling_backtest": {
            "design": "six folds, each trained only on classes at least two years older than the class it is tested on",
            "test_classes": (cv.get("design") or {}).get("test_fold_years"),
            "success_auc": ms("success_auc"), "success_brier": ms("success_brier"),
            "draft_grade_accuracy": ms("grade_acc"), "pick_mae_drafted": ms("pick_mae_drafted"),
            "pick_spearman": ms("pick_spearman_all"),
        },
        "frozen_holdout": {
            "test_classes": ev.get("holdout_years"),
            "success_auc": succ.get("auc"), "success_brier": succ.get("brier"),
            "draft_grade_accuracy": grade.get("accuracy"),
            "pick_mae_drafted": pick.get("mae_picks_drafted"), "pick_spearman": pick.get("spearman_all"),
        },
        "pick_range": {"target_coverage": pi.get("coverage_target"),
                       "measured_coverage": (pi.get("eval") or {}).get("coverage"),
                       "measured_coverage_top64": (pi.get("eval") or {}).get("coverage_top64")},
        "board": {"players": len(rows), "generated_at": generated_at, "verified_stats": verified,
                  "verified_share": round(verified / len(rows), 3) if rows else None},
        "limits": [
            "Every backtested player had a consensus big-board rank. Only a few hundred of the players on the "
            "live board have one, so accuracy on the rest of the board is not measured by these numbers.",
            "The newest class in any backtest is 2020. The production model is trained on every class and has "
            "no untouched holdout.",
            "An earlier feature set without consensus ranks scored 0.44 to 0.47 draft-grade accuracy on the "
            "2025-26 classes (models/experiments/RESULTS.md). Treat that as the floor for players without a "
            "consensus rank.",
            "On players the consensus board ranks, the model roughly ties the consensus; it does not beat it.",
            f"{len(rows) - verified} of {len(rows)} players on the board are graded on a roster placeholder, "
            "not verified stats.",
        ],
        "source_files": ["models/metadata.json", "models/experiments/rolling_cv_v4.json"],
    }


def _market_rows(player_id=None):
    led = _read(LEDGER_PATH) or {}
    out = []
    for e in led.get("entries", []):
        if player_id and e.get("slug") != player_id:
            continue
        res = e.get("resolution") or {}
        out.append({
            "player_id": e.get("slug"), "player": e.get("player"), "question": (e.get("question") or {}).get("label"),
            "recorded_on": e.get("date"), "model_probability": e.get("model_prob"),
            "market_price_cents": e.get("market_price_cents"),
            "difference_points": e.get("edge"), "market_closes": e.get("close_time"),
            "settled": res.get("outcome") is not None, "outcome": res.get("outcome"),
        })
    return out


def get_market_edge(player=None, limit=20):
    pid = slug(*(lambda p: (p.get("name"), p.get("team")))(_resolve(player))) if player else None
    rows = _market_rows(pid)
    rows.sort(key=lambda r: -abs(r["difference_points"] or 0))
    settled = [r for r in rows if r["settled"]]
    return {
        "note": "Analysis only: a frozen weekly record of the model's probability next to the Kalshi price for "
                "the same question. Not advice, and nothing here can place or suggest a trade.",
        "recorded": len(rows), "settled": len(settled), "last_recorded": (_read(LEDGER_PATH) or {}).get("last_synced_at"),
        "rows": rows[:limit],
    }


def get_team_needs(team):
    import dv_team_needs
    t = (team or "").strip()
    abbr = t.upper()
    if abbr not in dv_team_needs.TEAM_NAMES:
        hits = [a for a, n in dv_team_needs.TEAM_NAMES.items() if t.lower() in n.lower()]
        if len(hits) != 1:
            raise ToolError(f"unknown NFL team {team!r}; use an abbreviation such as CIN, NYJ, LA, SF")
        abbr = hits[0]
    d = dv_team_needs.team_payload(abbr, prospects=_board()[0])
    if not d:
        raise ToolError("team needs have not been built yet")
    tm, dr = d["team"], d["team"].get("draft") or {}
    return {
        "team": abbr, "name": tm["name"], "season": d["season"], "draft_year": d["draft_year"],
        "record": dr.get("record"), "projected_wins": dr.get("projected_wins"),
        "first_round_picks": [{k: p.get(k) for k in ("slot_p10", "slot_p50", "slot_p90", "from", "own")}
                              for p in dr.get("picks") or []],
        "acquired_from": dr.get("acquired_from"), "sent_to": dr.get("sent_to"),
        "starters": tm["starters"], "starters_contract_ends_this_season": tm["starters_expiring"],
        "position_groups": [{"group": g["group"], "label": g["label"], "starters": g["starters"],
                             "contract_ends_this_season": g["expiring"], "contract_ends_next_season": g["next"],
                             "players_ending_this_season": g["names"]} for g in tm["groups"]],
        "prospects_in_range": [{"pick": f["slot_p50"], "range": [f["slot_p10"], f["slot_p90"]],
                                "prospects": [{"player_id": slug(r["name"], r["college"]), "name": r["name"],
                                               "position": r["position"], "college": r["college"],
                                               "projected_pick": r["projected_pick"], "need_group": r["group"]}
                                              for r in f["prospects"]]} for f in d.get("fits", [])],
        "note": "Contract end is the last season the deal pays cash, not a prediction that the player leaves. "
                "Prospects in range are an overlap of two lists, not predicted selections.",
        "page": f"/team-needs/{abbr}",
    }


# ── Registry + validation ────────────────────────────────────────────────────
_STR = {"type": "string", "minLength": 1, "maxLength": 80}
_LIMIT = lambda default: {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": default}  # noqa: E731
_POS = {"type": "string", "enum": sorted(POSITION_GROUPS)}
_GRADE = {"type": "string", "enum": GRADES}


def _schema(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = {
    "search_players": {
        "fn": search_players,
        "description": "Find players on the board by name (or college). Returns player_id values the other tools take.",
        "input_schema": _schema({"query": {**_STR, "minLength": 2}, "limit": _LIMIT(8)}, ["query"]),
    },
    "get_player_report": {
        "fn": get_player_report,
        "description": "Everything the board holds on one player: grade, success probability, projected pick and "
                       "range, ranks in class and position, this week's change, data source, market record.",
        "input_schema": _schema({"player": {**_STR, "description": "player_id from search_players, or an exact name"}},
                                ["player"]),
    },
    "compare_players": {
        "fn": compare_players,
        "description": "Two to four players side by side, with the leader on each metric.",
        "input_schema": _schema({"players": {"type": "array", "items": _STR, "minItems": 2, "maxItems": 4}}, ["players"]),
    },
    "query_board": {
        "fn": query_board,
        "description": "Filter and sort the board. Use this for any 'which players...' or 'top N...' question.",
        "input_schema": _schema({
            "position": _POS,
            "draft_class": {"type": "integer", "minimum": 2026, "maximum": 2032},
            "college": {**_STR, "description": "substring of the college name, e.g. 'Texas'"},
            "conference_tier_max": {"type": "integer", "minimum": 1, "maximum": 8,
                                    "description": "1 is the strongest conference tier; 2 keeps tiers 1 and 2"},
            "grade_min": {**_GRADE, "description": "this grade or better"},
            "grade_max": {**_GRADE, "description": "this grade or worse"},
            "min_success_probability": {"type": "number", "minimum": 0, "maximum": 100},
            "verified_only": {"type": "boolean", "default": False,
                              "description": "only players graded on verified live stats"},
            "sort": {"type": "string", "enum": sorted(SORTS), "default": "projected_pick"},
            "descending": {"type": "boolean"},
            "limit": _LIMIT(10),
        }),
    },
    "get_movers": {
        "fn": get_movers,
        "description": "This week's biggest risers and fallers on the board, versus the previous weekly snapshot.",
        "input_schema": _schema({"direction": {"type": "string", "enum": ["both", "risers", "fallers"], "default": "both"},
                                 "position": _POS, "limit": _LIMIT(10)}),
    },
    "get_model_card": {
        "fn": get_model_card,
        "description": "How accurate the model is and where that is and is not measured. Quote accuracy only from here.",
        "input_schema": _schema({}),
    },
    "get_market_edge": {
        "fn": get_market_edge,
        "description": "The recorded comparison of model probability and Kalshi market price. Analysis only.",
        "input_schema": _schema({"player": _STR, "limit": _LIMIT(20)}),
    },
    "get_team_needs": {
        "fn": get_team_needs,
        "description": "An NFL team's projected first-round picks, starters whose contracts are ending by position "
                       "group, and board prospects projected in its pick range at those positions.",
        "input_schema": _schema({"team": {**_STR, "description": "abbreviation (CIN) or team name (Bengals)"}}, ["team"]),
    },
}

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _check(value, schema, path):
    t = schema.get("type")
    if t and (not isinstance(value, _TYPES[t]) or (t in ("integer", "number") and isinstance(value, bool))):
        raise ToolError(f"{path} must be {t}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"{path} must be one of {schema['enum']}")
    if t == "string":
        if len(value.strip()) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 10 ** 6):
            raise ToolError(f"{path} must be {schema.get('minLength', 0)} to {schema.get('maxLength')} characters")
    if t in ("integer", "number"):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            raise ToolError(f"{path} must be between {schema.get('minimum')} and {schema.get('maximum')}")
    if t == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10 ** 6):
            raise ToolError(f"{path} must have {schema.get('minItems', 0)} to {schema.get('maxItems')} items")
        for i, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{i}]")


def validate(name, args):
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(TOOLS)}")
    schema = TOOLS[name]["input_schema"]
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    extra = set(args) - set(schema["properties"])
    if extra:
        raise ToolError(f"unknown argument(s) {sorted(extra)}; allowed: {sorted(schema['properties'])}")
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        raise ToolError(f"missing required argument(s) {missing}")
    for k, v in args.items():
        _check(v, schema["properties"][k], k)
    return args


def call(name, args=None):
    """Validate, run, wrap. Raises ToolError for anything the caller got wrong."""
    args = validate(name, dict(args or {}))
    result = TOOLS[name]["fn"](**args)
    return {"tool": name, "model_version": MODEL_VERSION, "board_generated_at": _board()[1], "result": result}


def tool_specs():
    """The list an LLM tool loop or an MCP server advertises."""
    return [{"name": n, "description": t["description"], "input_schema": t["input_schema"]} for n, t in TOOLS.items()]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps(tool_specs(), indent=2))
        sys.exit(0)
    try:
        print(json.dumps(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}), indent=2))
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
