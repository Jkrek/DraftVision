"""Regression tests for dv_edge against SAVED raw Kalshi payloads.

Fixtures under tests/fixtures/ are verbatim public-API responses
(GET /trade-api/v2/markets?series_ticker=…&status=open, 2026-09-09) plus the
prospect-cache rows those markets name, with pick_range attached from the
local v5 /predict. If Kalshi drifts its schema again (the 2026 move from
integer-cent `last_price` to string `last_price_dollars` silently zeroed the
board for weeks), the parser tests here fail loudly instead of the page
degrading to its "seasonal" empty state.

Run either way — pytest is not a project dependency:

    .venv/bin/python -m pytest tests/test_edge_parsing.py -q
    .venv/bin/python tests/test_edge_parsing.py

No network: DV_EDGE_NO_WARMUP is set before dv_edge is imported, and the
market cache is filled from the fixtures.
"""

import json
import os
import sys
import tempfile
import time

os.environ["DV_EDGE_NO_WARMUP"] = "1"  # must precede the import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dv_edge as E  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SERIES = ("KXNFLDRAFTTOP", "KXNFLDRAFTPICK", "KXHEISMAN")


def _load(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def _raw_markets():
    out = []
    for s in SERIES:
        for m in _load(f"kalshi_markets_{s}.json")["markets"]:
            out.append((s, m))
    return out


def _rows():
    return [E._market_row(m, "", s, m.get("event_ticker") or "") for s, m in _raw_markets()]


def _prospects():
    return _load("prospects_2027_subset.json")["prospects"]


def _find(rows, ticker):
    return next(r for r in rows if r["ticker"] == ticker)


# ── Schema drift guard ────────────────────────────────────────────────────────
def test_fixtures_carry_the_fields_the_parser_reads():
    raw = _raw_markets()
    assert len(raw) == 77, "fixture market count changed — regenerate or update"
    for _, m in raw:
        assert "ticker" in m and "event_ticker" in m and "title" in m
        assert "yes_sub_title" in m
        has_dollars = any(k in m for k in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars"))
        has_cents = any(k in m for k in ("last_price", "yes_bid", "yes_ask"))
        assert has_dollars or has_cents, f"{m['ticker']}: no price fields at all"


def test_every_fixture_market_with_a_quote_is_priced():
    unpriced = []
    for _, m in _raw_markets():
        quoted = any(
            E._cents_from_dollars(m.get(k)) not in (None, 0)
            for k in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars")
        )
        price = E._yes_price_cents(m)
        if quoted and price is None:
            unpriced.append(m["ticker"])
        if price is not None:
            assert 0 < price < 100
    assert not unpriced, f"quoted but unpriced: {unpriced}"
    priced = sum(E._yes_price_cents(m) is not None for _, m in _raw_markets())
    assert priced >= 0.95 * 77, f"only {priced}/77 fixture markets priced"


# ── Price parser ──────────────────────────────────────────────────────────────
def test_price_parser_dollars_strings():
    top = {m["ticker"]: m for _, m in _raw_markets()}
    assert E._yes_price(top["KXNFLDRAFTTOP-27-5-AMAN"]) == (77, "last")   # "0.7700"
    assert E._yes_price(top["KXNFLDRAFTTOP-27-5-JSMI"]) == (90, "last")
    assert E._yes_price(top["KXNFLDRAFTPICK-27-1-AMAN"]) == (29, "last")
    # last "0.0000", bid "0.0100", ask "0.6500" -> two-sided midpoint
    assert E._yes_price(top["KXNFLDRAFTTOP-27-5-JSEA"]) == (33, "mid")
    # last "0.0000", bid "0.0000", ask "0.3800" -> one-sided ask
    assert E._yes_price(top["KXNFLDRAFTTOP-27-5-QRHO"]) == (38, "ask")


def test_price_parser_legacy_cents_fallback():
    assert E._yes_price({"last_price": 42}) == (42, "last")
    assert E._yes_price({"yes_bid": 40, "yes_ask": 44}) == (42, "mid")
    assert E._yes_price({"yes_ask": 60}) == (60, "ask")
    # dollars win when both are present
    assert E._yes_price({"last_price": 42, "last_price_dollars": "0.5000"}) == (50, "last")


def test_price_parser_no_price_cases():
    assert E._yes_price({}) == (None, None)
    assert E._yes_price({"last_price_dollars": "0.0000", "yes_bid_dollars": "0.0000"}) == (None, None)
    assert E._yes_price({"last_price_dollars": "0.0000", "yes_bid_dollars": "0.0000",
                         "yes_ask_dollars": "1.0000"}) == (None, None)
    assert E._yes_price({"last_price_dollars": "garbage"}) == (None, None)
    assert E._cents_from_dollars("1.5") is None
    assert E._cents_from_int(True) is None
    # non-finite strings parse as floats and must not reach round()
    for bad in ("inf", "-inf", "nan", "Infinity", "NaN", "1e400"):
        assert E._cents_from_dollars(bad) is None, bad
    assert E._cents_from_int(float("inf")) is None
    assert E._cents_from_int(float("nan")) is None
    assert E._volume({"volume_fp": "inf"}) is None
    assert E._volume({"volume_fp": "12.50"}) == 12.5
    assert E._volume({"volume": 3}) == 3.0
    assert E._volume({}) is None


# ── Names, matching, tiebreak ─────────────────────────────────────────────────
def test_candidate_names_prefer_custom_strike_person():
    top = {m["ticker"]: m for _, m in _raw_markets()}
    assert E._candidate_names("", top["KXNFLDRAFTTOP-27-5-QRHO"])[0] == "Quincy Rhodes Jr."
    assert E._candidate_names("", top["KXHEISMAN-27-JSMIT"])[0] == "Jeremiah Smith"
    assert E._team_hint(top["KXHEISMAN-27-JSMIT"]) == "Ohio St."
    assert E._norm_name("Quincy Rhodes Jr.") == "quincy rhodes"
    assert E._norm_name("William “Pop” Watson III") == "william pop watson"


def test_ambiguous_names_are_tiebroken_not_refused():
    index = E._build_prospect_index(_prospects())
    assert len(index["jeremiah smith"]) == 2 and len(index["dylan stewart"]) == 2
    # both collisions are DECISIVE (a pick-1/pick-6 prospect vs a namesake
    # ~3,400 ranks down the board), so match_ambiguous is False
    row, amb = E._resolve_player(["Jeremiah Smith"], index, draft_year=2027, team_hint="")
    assert not amb and row["team"] == "Ohio State Buckeyes"
    row, amb = E._resolve_player(["Dylan Stewart"], index, draft_year=2027)
    assert not amb and row["team"] == "South Carolina Gamecocks"
    # a team hint can override the projected-pick tiebreak
    row, _ = E._resolve_player(["Jeremiah Smith"], index, team_hint="Louisiana Tech")
    assert row["team"] == "Louisiana Tech Bulldogs"
    # draft_class is NOT a tiebreak: a junior carries draft_class 2028 while
    # eligible for the 2027 draft, so the lower projected pick wins.
    fake = E._build_prospect_index([
        {"name": "A B", "team": "X", "draft_class": 2028, "projected_pick": 1},
        {"name": "A B", "team": "Y", "draft_class": 2027, "projected_pick": 50},
    ])
    row, amb = E._resolve_player(["A B"], fake, draft_year=2027)
    assert amb and row["team"] == "X"
    # ... but a draft_class EARLIER than the market year is an exclusion
    fake = E._build_prospect_index([
        {"name": "A B", "team": "X", "draft_class": 2026, "projected_pick": 1},
        {"name": "A B", "team": "Y", "draft_class": 2027, "projected_pick": 50},
    ])
    row, amb = E._resolve_player(["A B"], fake, draft_year=2027)
    assert not amb and row["team"] == "Y"
    only_old = E._build_prospect_index([{"name": "C D", "team": "Z", "draft_class": 2026}])
    assert E._resolve_player(["C D"], only_old, draft_year=2027) == (None, False)
    assert E._resolve_player(["C D"], only_old, draft_year=None)[0]["team"] == "Z"
    assert E._resolve_player(["Nobody Here"], index) == (None, False)


# ── Question parsing ──────────────────────────────────────────────────────────
def test_market_question_kinds():
    rows = _rows()
    assert _find(rows, "KXNFLDRAFTTOP-27-5-AMAN")["_question"] == {"kind": "top_n", "n": 5}
    assert _find(rows, "KXNFLDRAFTPICK-27-1-AMAN")["_question"] == {"kind": "pick_eq", "n": 1}
    assert _find(rows, "KXHEISMAN-27-AMANN")["_question"] == {"kind": "heisman"}
    assert _find(rows, "KXNFLDRAFTTOP-27-5-AMAN")["draft_year"] == 2027
    # every ticker shape carries the two-digit year
    for tick in ("KXNFLDRAFTTOP-27-5", "KXNFLDRAFTPICK-27-1", "KXHEISMAN-27",
                 "KXNFLDRAFTQB-27P2", "KXNFLDRAFTOU-27", "KXNFLDRAFTTOP-27-R1"):
        assert E._draft_year(tick) == 2027, tick
    assert E._draft_year("KXNFLDRAFTTOP-26-R1") == 2026
    assert E._draft_year("") is None
    q = E._market_question
    assert q("KXNFLDRAFTTOP", "KXNFLDRAFTTOP-26-R1", "Drafted in the 1st Round", {}) == {"kind": "top_n", "n": 32}
    assert q("KXNFLDRAFTQB", "KXNFLDRAFTQB-27P2", "2nd Quarterback drafted", {}) == {"kind": "nth_position"}
    assert q("KXNFLDRAFTOU", "KXNFLDRAFTOU-27", "", {"strike_type": "less", "cap_strike": 10.5}) == {"kind": "top_n", "n": 10}
    assert q("KXNFLDRAFTOU", "KXNFLDRAFTOU-27", "", {"strike_type": "greater", "floor_strike": 10.5}) == {"kind": "after_n", "n": 10}
    assert q("KXNFLDRAFTOU", "KXNFLDRAFTOU-27", "Drafted under pick 10.5?", {}) == {"kind": "top_n", "n": 10}
    assert q("KXNFLDRAFTOU", "KXNFLDRAFTOU-27", "Drafted over pick 10.5?", {}) == {"kind": "after_n", "n": 10}
    assert q("KXWHATEVER", "KXWHATEVER-27", "Will X be a top 10 pick?", {}) == {"kind": "top_n", "n": 10}
    assert q("KXWHATEVER", "KXWHATEVER-27", "Something else", {}) == {"kind": "unknown"}


def test_relevance_guard_excludes_team_and_supplemental_series():
    assert E._is_relevant("KXNFLDRAFTTOP", "", "")
    assert E._is_relevant("KXHEISMAN", "", "")
    assert not E._is_relevant("KXNFLDRAFT1ST", "", "Will Arizona make the 1st Overall Pick?")
    assert not E._is_relevant("KXNFLSDRAFTTOP", "", "")
    assert not E._is_relevant("KXNCAAFTEAMRECTD", "", "Arizona St.: 1+ receiving touchdowns")
    assert "KXNCAAFTEAMRECTD" not in E._KNOWN_SERIES
    assert "KXNFLSDRAFTTOP" not in E._KNOWN_SERIES
    assert "KXNFLDRAFT1ST" not in E._KNOWN_SERIES
    for s in ("KXNFLDRAFTTOP", "KXNFLDRAFTPICK", "KXNFLDRAFTOU", "KXNFLDRAFTQB", "KXNFLDRAFTEDGE", "KXHEISMAN"):
        assert s in E._KNOWN_SERIES


# ── Model-implied probabilities ──────────────────────────────────────────────
def test_pick_cdf_is_a_cdf():
    for lo, hi in ((1, 18), (1, 7), (3, 3884), (12, 1270), (5, 5)):
        prev = 0.0
        for x in range(1, 400):
            p = E._pick_cdf(x, lo, hi)
            assert 0.0 <= p <= 1.0 and p >= prev - 1e-12
            prev = p
        # the interval holds ~its nominal mass (up to the continuity correction)
        inside = E._pick_cdf(hi + 0.5, lo, hi) - E._pick_cdf(max(lo - 0.5, 0.5), lo, hi)
        assert inside >= 0.79, (lo, hi, inside)
        assert E._pick_cdf(E._RANK_CEILING, lo, hi) == 1.0
    assert E._pick_cdf(0.4, 1, 18) == 0.0
    assert abs(E._p_pick_le(5, 1, 18, 0.8) - 0.598) < 0.002
    # exact-pick mass is deliberately NOT derivable from the interval
    assert not hasattr(E, "_p_pick_eq")
    # a wide interval reaching far past the draft does not fold mass back in
    assert E._p_pick_le(5, 8, 3884, 0.8) < 0.10
    assert E._p_pick_le(260, 8, 3884, 0.8) < 0.60


def test_model_fields_policy():
    rows = _rows()
    index = E._build_prospect_index(_prospects())
    manning = index["arch manning"][0]
    assert manning["pick_range"]["lo"] == 1 and manning["pick_range"]["hi"] == 18
    top5 = _find(rows, "KXNFLDRAFTTOP-27-5-AMAN")
    prob, edge, basis = E._model_fields(top5, manning)
    assert prob == 59.8 and edge == round(59.8 - 77, 1)
    assert basis.startswith("model-implied P(pick <= 5)") and "[1, 18]" in basis
    # exact-pick markets are listed only — never priced from an interval
    pick1 = _find(rows, "KXNFLDRAFTPICK-27-1-AMAN")
    prob, edge, basis = E._model_fields(pick1, manning)
    assert prob is None and edge is None
    assert basis == "not priced: exact-pick probabilities are not identifiable from an interval"
    # the event-coherence scale multiplies P(pick <= n) and is recorded
    prob, edge, basis = E._model_fields(top5, manning, scale=0.5)
    assert prob == 29.9 and edge == round(29.9 - 77, 1)
    assert "scaled x0.500" in basis and "sums to 5" in basis
    for bad in (0.0, -1.0, 2.0, float("nan"), "x", None):
        assert E._model_fields(top5, manning, scale=bad)[0] == 59.8, bad
    # Heisman: listed, never priced
    prob, edge, basis = E._model_fields(_find(rows, "KXHEISMAN-27-AMANN"), manning)
    assert prob is None and edge is None and basis.startswith("not priced")
    # no pick_range -> not priced (the old success_probability proxy is gone)
    bare = {k: v for k, v in manning.items() if k != "pick_range"}
    prob, edge, basis = E._model_fields(top5, bare)
    assert prob is None and "pick_range" in basis
    # no quote -> not priced
    prob, edge, basis = E._model_fields(dict(top5, yes_price_cents=None), manning)
    assert prob is None and "quote" in basis


# ── End-to-end payload (no network) ──────────────────────────────────────────
def _with_cache(rows, prospects, ledger_path, **kwargs):
    saved = (dict(E._kalshi_cache), E.LEDGER_PATH, dict(E._LEDGER), E._LEDGER_MTIME)
    E._kalshi_cache.update({"markets": rows, "fetched_at": time.monotonic(),
                            "fetched_at_utc": "test", "series_counts": {"TEST": len(rows or [])},
                            "error": None, "refreshing": False})
    E.LEDGER_PATH = ledger_path
    E._LEDGER = {"entries": []}
    E._LEDGER_MTIME = 0.0
    try:
        return E.edge_payload(prospects, **kwargs)
    finally:
        E._kalshi_cache.update(saved[0])
        E.LEDGER_PATH, E._LEDGER, E._LEDGER_MTIME = saved[1], saved[2], saved[3]


def test_edge_payload_shape_and_counts():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        p = _with_cache(_rows(), _prospects(), ledger)
        assert set(p) == {"generated_at", "markets", "note", "discovery", "summary"}
        assert p["note"] != "seasonal" and p["note"] != "warming"
        assert p["discovery"]["state"] == "ok"
        row_keys = {"ticker", "title", "event_ticker", "series", "yes_price_cents",
                    "yes_price_source", "spread_cents", "ledger_eligible", "matched_player",
                    "matched_team", "match_ambiguous", "model_prob", "edge", "model_basis", "url"}
        # futures spec api_contract: ADDITIVE fields only — the legacy keys
        # stay, exactly these are added.
        added_keys = {"question", "draft_year", "volume", "close_time", "model_interval",
                      "event_scale", "wide_interval", "matched_espn_id", "matched_espn_team_id",
                      "matched_slug", "call"}
        for r in p["markets"]:
            assert set(r) == row_keys | added_keys, set(r) ^ (row_keys | added_keys)
            assert r["model_basis"]  # every row explains itself
        s = p["summary"]
        assert s["markets"] == 77 and s["priced"] == 77
        assert s["matched"] == 70, s  # 65 exact + 5 recovered by the tiebreak
        assert s["modeled"] == 18, s  # 19 top-5 minus 1 not in cache; #1-pick rows are listed only
        assert s["edges_over_threshold"] >= 5
        assert {"eligible_over_threshold", "on_ledger", "by_kind", "draft_year",
                "top_gap_ticker"} <= set(s)
        heisman = [r for r in p["markets"] if r["series"] == "KXHEISMAN"]
        assert heisman and all(r["edge"] is None for r in heisman)
        pick1 = [r for r in p["markets"] if r["series"] == "KXNFLDRAFTPICK"]
        assert len(pick1) == 20 and all(r["model_prob"] is None for r in pick1)
        assert all(r["model_basis"] == E._NOT_PRICED_EXACT for r in pick1 if r["matched_player"])
        smith = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
        assert smith["matched_team"] == "Ohio State Buckeyes" and not smith["match_ambiguous"]
        # sorted: edges first by |edge| desc
        edges = [abs(r["edge"]) for r in p["markets"] if r["edge"] is not None]
        assert edges == sorted(edges, reverse=True)
        # ledger: one row per qualifying ticker, two-sided quotes only
        with open(ledger) as f:
            entries = json.load(f)["entries"]
        assert entries and len(entries) == len({e["ticker"] for e in entries})
        assert all(e["price_source"] in ("last", "mid") for e in entries)
        assert all(abs(e["edge"]) >= E.LEDGER_EDGE_THRESHOLD for e in entries)
        assert all(e["model_basis"].startswith("model-implied") for e in entries)
        by_ticker = {r["ticker"]: r for r in p["markets"]}
        assert all(by_ticker[e["ticker"]]["ledger_eligible"] for e in entries)
        qualifying_one_sided = [r for r in p["markets"] if r["edge"] is not None
                                and abs(r["edge"]) >= 10 and r["yes_price_source"] == "ask"]
        assert qualifying_one_sided, "fixture should contain a one-sided qualifying quote"
        assert not any(e["ticker"] == qualifying_one_sided[0]["ticker"] for e in entries)
        # JSEA / CCOL: last 0, bid 1c, ask 54-65c, zero volume -> a placeholder
        # book. Its midpoint stays on the board tagged mid_wide, never ledgered.
        for tick in ("KXNFLDRAFTTOP-27-5-JSEA", "KXNFLDRAFTTOP-27-5-CCOL"):
            r = by_ticker[tick]
            assert r["yes_price_source"] == "mid_wide" and r["ledger_eligible"] is False, r
            assert r["spread_cents"] > E.LEDGER_MAX_SPREAD_CENTS
            assert r["edge"] is not None and abs(r["edge"]) >= E.LEDGER_EDGE_THRESHOLD, r
            assert not any(e["ticker"] == tick for e in entries), tick
        # re-running the same day adds nothing
        _with_cache(_rows(), _prospects(), ledger)
        with open(ledger) as f:
            assert len(json.load(f)["entries"]) == len(entries)


def test_event_coherence_top_n_mass_sums_to_at_most_n():
    """Within KXNFLDRAFTTOP-27-5 at most five players can go top-5, so the
    matched players' P(pick <= 5) must sum to <= 5. The raw marginals from
    independent intervals overshoot on this fixture; the payload scales them."""
    rows = _rows()
    index = E._build_prospect_index(_prospects())
    matched = []
    for src in rows:
        if src["series"] != "KXNFLDRAFTTOP":
            continue
        player, _ = E._resolve_player(src["_market_raw_names"], index,
                                      draft_year=src["draft_year"], team_hint=src["_team_hint"])
        if player:
            matched.append((src, player))
    raw = [E._raw_p_le(s, pl) for s, pl in matched]
    raw_sum = sum(x for x in raw if x is not None)
    assert len(matched) == 18 and raw_sum > 5, raw_sum
    scales = E._event_scale_factors(matched)
    assert set(scales) == {("KXNFLDRAFTTOP-27-5", 5)}
    assert abs(scales[("KXNFLDRAFTTOP-27-5", 5)] - 5 / raw_sum) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        p = _with_cache(rows, _prospects(), os.path.join(tmp, "ledger.json"), dry_run=True)
    top5 = [r for r in p["markets"] if r["event_ticker"] == "KXNFLDRAFTTOP-27-5"
            and r["model_prob"] is not None]
    assert len(top5) == 18
    total = sum(r["model_prob"] for r in top5) / 100.0
    assert total <= 5.0 + 0.01, total  # rounding slack: 18 rows at 0.05 pt each
    assert total > 4.9, total  # scaled to N, not crushed
    assert all("scaled x" in r["model_basis"] for r in top5)
    # #1-pick rows in the same fixture: none priced, so no sum to check (n/a)
    assert all(r["model_prob"] is None for r in p["markets"]
               if r["event_ticker"] == "KXNFLDRAFTPICK-27-1")
    # an event whose marginals already sum below N is left untouched
    under = [(dict(rows[0], event_ticker="EV-X", ticker="EV-X-A",
                   _question={"kind": "top_n", "n": 5}),
              {"name": "A", "team": "T", "pick_range": {"lo": 1, "hi": 18, "confidence": 0.8}}),
             (dict(rows[0], event_ticker="EV-X", ticker="EV-X-B",
                   _question={"kind": "after_n", "n": 5}),
              {"name": "B", "team": "T", "pick_range": {"lo": 1, "hi": 18, "confidence": 0.8}})]
    assert E._event_scale_factors(under) == {}
    # over/under rows count their P(pick <= n) toward the same event sum
    over = [(dict(u[0], _question={"kind": "after_n", "n": 1}), u[1]) for u in under] * 3
    sc = E._event_scale_factors([(dict(s, ticker=f"t{i}"), dict(pl, name=f"P{i}"))
                                 for i, (s, pl) in enumerate(over)])
    assert ("EV-X", 1) in sc and 0 < sc[("EV-X", 1)] < 1


def test_probe_dry_run_never_writes_the_ledger():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        p = _with_cache(_rows(), _prospects(), ledger, dry_run=True)
        assert p["ledger_dry_run"], "fixture should yield would-be ledger rows"
        assert not os.path.exists(ledger)
        assert all(_find(p["markets"], e["ticker"])["ledger_eligible"] for e in p["ledger_dry_run"])
        # the real path writes exactly those rows
        _with_cache(_rows(), _prospects(), ledger)
        with open(ledger) as f:
            written = {e["ticker"] for e in json.load(f)["entries"]}
        assert written == {e["ticker"] for e in p["ledger_dry_run"]}


def test_discovery_stale_state_is_reachable_after_failures():
    saved = dict(E._kalshi_cache)
    fetch = E._fetch_relevant_markets

    def boom():
        raise RuntimeError("upstream down")

    E._fetch_relevant_markets = boom
    try:
        old = time.monotonic() - 3 * E.CACHE_TTL_S
        E._kalshi_cache.update({"markets": _rows(), "fetched_at": old, "fetched_at_utc": "then",
                                "series_counts": {}, "error": None, "refreshing": False})
        E._refresh_markets_once()
        st = E._discovery_status()
        assert st["state"] == "stale" and st["error"] == "upstream down", st
        assert E._kalshi_cache["fetched_at"] == old  # not advanced on failure
        assert E._kalshi_cache["fetched_at_utc"] == "then"
        assert len(E._kalshi_cache["markets"]) == 77  # last good pass kept
        # a failure before any success: [] + error -> 'error', not 'ok'
        E._kalshi_cache.update({"markets": None, "fetched_at": 0.0, "fetched_at_utc": None,
                                "error": None})
        E._refresh_markets_once()
        st = E._discovery_status()
        assert st["state"] == "error" and E._kalshi_cache["markets"] == []
    finally:
        E._fetch_relevant_markets = fetch
        E._kalshi_cache.clear()
        E._kalshi_cache.update(saved)


def test_one_malformed_market_does_not_abort_the_pass():
    good = [m for _, m in _raw_markets() if m["ticker"].startswith("KXNFLDRAFTTOP")]
    bad = dict(good[0], ticker="KXNFLDRAFTTOP-27-5-BAD", title=12345)  # .strip() on an int
    per_series = {"KXNFLDRAFTTOP": [bad, "not a dict"] + good}
    fetch = E._fetch_series_markets
    E._fetch_series_markets = lambda session, series: per_series.get(series, [])
    try:
        rows, counts = E._fetch_relevant_markets()
    finally:
        E._fetch_series_markets = fetch
    assert counts["KXNFLDRAFTTOP"] == len(good) and len(rows) == len(good)
    assert "KXNFLDRAFTTOP-27-5-BAD" not in {r["ticker"] for r in rows}


def test_fork_hook_replaces_locks_and_forgets_the_thread():
    k, l = E._kalshi_lock, E._ledger_lock
    E._kalshi_cache["refreshing"] = True
    E._after_fork_in_child()
    assert E._kalshi_lock is not k and E._ledger_lock is not l
    assert E._warm_thread is None and E._kalshi_cache["refreshing"] is False
    assert not E._kalshi_lock.locked() and not E._ledger_lock.locked()


def test_edge_payload_warming_and_seasonal_states():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        p = _with_cache(None, _prospects(), ledger)
        assert p["note"] == "warming" and p["markets"] == [] and p["discovery"]["state"] == "warming"
        p = _with_cache([], _prospects(), ledger)
        assert p["note"] == "seasonal" and p["markets"] == []
        assert not os.path.exists(ledger)


def test_no_background_thread_under_tests():
    assert E.ensure_background_refresh() is False
    assert E._warm_thread is None


# ── Futures spec: additive api_contract ──────────────────────────────────────
def _board_ctx(rows, ledger_path, history_path=None):
    """Context that installs fixture markets + a scratch ledger/history and
    restores module state afterwards (the memoised board is dropped too)."""
    class _Ctx:
        def __enter__(self):
            self.saved = (dict(E._kalshi_cache), E.LEDGER_PATH, dict(E._LEDGER), E._LEDGER_MTIME,
                          E.HISTORY_PATH, dict(E._HISTORY), E._HISTORY_MTIME,
                          dict(E._BOARD_MEMO))
            E._kalshi_cache.update({"markets": rows, "fetched_at": time.monotonic(),
                                    "fetched_at_utc": "2026-09-10T14:14:00+00:00",
                                    "series_counts": {"TEST": len(rows or [])},
                                    "error": None, "refreshing": False})
            E.LEDGER_PATH = ledger_path
            E._LEDGER = {"entries": [], "last_synced_at": None}
            E._LEDGER_MTIME = 0.0
            if history_path:
                E.HISTORY_PATH = history_path
                E._HISTORY = {"tickers": {}}
                E._HISTORY_MTIME = 0.0
            return self

        def __exit__(self, *exc):
            E._kalshi_cache.clear()
            E._kalshi_cache.update(self.saved[0])
            E.LEDGER_PATH, E._LEDGER, E._LEDGER_MTIME = self.saved[1], self.saved[2], self.saved[3]
            E.HISTORY_PATH, E._HISTORY, E._HISTORY_MTIME = self.saved[4], self.saved[5], self.saved[6]
            E._BOARD_MEMO.clear()
            E._BOARD_MEMO.update(self.saved[7])
            return False
    return _Ctx()


def test_question_public_classification_and_labels():
    rows = _rows()
    qp = E._question_public
    assert qp(_find(rows, "KXNFLDRAFTTOP-27-5-AMAN")["_question"]) == \
        {"kind": "top_n", "n": 5, "label": "Top-5 pick"}
    assert qp(_find(rows, "KXNFLDRAFTPICK-27-1-AMAN")["_question"]) == \
        {"kind": "pick_eq", "n": 1, "label": "#1 overall"}
    assert qp(_find(rows, "KXHEISMAN-27-AMANN")["_question"]) == \
        {"kind": "heisman", "n": None, "label": "Heisman"}
    assert qp({"kind": "after_n", "n": 10}) == {"kind": "after_n", "n": 10, "label": "After pick 10"}
    assert qp({"kind": "nth_position"})["label"] == "Positional order"
    assert qp({"kind": "unknown"}) == {"kind": "unknown", "n": None, "label": "Other"}
    assert qp(None) == {"kind": "unknown", "n": None, "label": "Other"}
    assert qp({"kind": "top_n", "n": "7"})["n"] == 7  # coerced
    # every fixture row lands in one of the spec's by_kind buckets
    kinds = {qp(r["_question"])["kind"] for r in rows}
    assert kinds == {"top_n", "pick_eq", "heisman"}
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(_prospects(), dry_run=True)
    assert p["summary"]["by_kind"] == {"top_n": 19, "pick_eq": 20, "heisman": 38, "other": 0}
    assert p["summary"]["draft_year"] == 2027
    for r in p["markets"]:
        assert r["question"]["kind"] in ("top_n", "pick_eq", "heisman")
        assert r["draft_year"] == 2027 and r["close_time"]
        assert r["wide_interval"] == bool(r["model_interval"] and r["model_interval"]["hi"] > 100)
        if r["model_prob"] is not None:
            assert r["model_interval"] and r["event_scale"] is not None
            assert 0 < r["event_scale"] <= 1.0
        else:
            assert r["event_scale"] is None
    # KXNFLDRAFTTOP-27-5: 18 rows scaled by the same factor, recorded on each
    top5 = [r for r in p["markets"] if r["event_ticker"] == "KXNFLDRAFTTOP-27-5" and r["model_prob"] is not None]
    assert len({r["event_scale"] for r in top5}) == 1 and top5[0]["event_scale"] < 1.0
    manning = _find(p["markets"], "KXNFLDRAFTTOP-27-5-AMAN")
    assert manning["model_interval"] == {"lo": 1, "hi": 18, "coverage": 0.8}
    assert manning["wide_interval"] is False and manning["volume"] is not None
    assert manning["matched_espn_id"] == "4870906" and manning["matched_espn_team_id"] == "251"
    assert manning["matched_slug"] == "arch-manning-texas-longhorns"
    # summary.top_gap_ticker: largest |edge| among eligible & unambiguous only
    clean = [r for r in p["markets"] if r["edge"] is not None and abs(r["edge"]) >= 10
             and r["ledger_eligible"] and not r["match_ambiguous"]]
    assert p["summary"]["top_gap_ticker"] == max(clean, key=lambda r: abs(r["edge"]))["ticker"]
    assert p["summary"]["eligible_over_threshold"] == sum(
        1 for r in p["markets"] if r["edge"] is not None and abs(r["edge"]) >= 10 and r["ledger_eligible"])
    assert p["summary"]["eligible_over_threshold"] < p["summary"]["edges_over_threshold"]  # mid_wide/ask excluded
    assert p["summary"]["on_ledger"] == 0  # dry run: nothing written


def test_player_slug_matches_the_js_port():
    slug = E._player_slug
    assert slug("Arch Manning", "Texas Longhorns") == "arch-manning-texas-longhorns"
    assert slug("Quincy Rhodes Jr.", "Arkansas Razorbacks") == "quincy-rhodes-jr-arkansas-razorbacks"
    assert slug("William “Pop” Watson III", "Ohio St.") == "william-pop-watson-iii-ohio-st"
    assert slug("  Two   Spaces ", "T") == "-two-spaces--t"  # whitespace runs → '-', like the JS
    assert slug("Ka'imi O'Neal", "Hawai'i") == "kaimi-oneal-hawaii"
    assert slug(None, None) == "-"
    assert E._player_slug_key("Arch Manning-Texas Longhorns") == "arch-manning-texas-longhorns"
    assert E._player_slug_key("arch-manning-texas-longhorns") == "arch-manning-texas-longhorns"


def test_player_lookup_states_and_ordering():
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(rows, ledger):
            by_nt = E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")
            assert by_nt["state"] == "ok" and by_nt["as_of"] and by_nt["discovery"]["state"] == "ok"
            assert by_nt["player"] == {"name": "Arch Manning", "team": "Texas Longhorns",
                                       "espn_id": "4870906", "espn_team_id": "251",
                                       "slug": "arch-manning-texas-longhorns"}
            kinds = [m["question"]["kind"] for m in by_nt["markets"]]
            assert kinds == ["top_n", "pick_eq", "heisman"], kinds  # spec order
            assert all("call" in m and "matched_slug" in m for m in by_nt["markets"])
            assert by_nt["on_ledger"] is False and by_nt["ledger_since"] is None
            # normalisation: case, punctuation, 'St.' → 'state'
            alt = E.player_payload(prospects, name="arch manning", team="texas longhorns")
            assert [m["ticker"] for m in alt["markets"]] == [m["ticker"] for m in by_nt["markets"]]
            # slug and espn_id routes hit the same rows
            assert E.player_payload(prospects, slug="arch-manning-texas-longhorns")["markets"] == by_nt["markets"]
            assert E.player_payload(prospects, espn_id="4870906")["markets"] == by_nt["markets"]
            # wrong team → no_markets (never a different player's rows)
            miss = E.player_payload(prospects, name="Arch Manning", team="Oregon Ducks")
            assert miss["state"] == "no_markets" and miss["markets"] == [] and miss["player"] is None
            assert E.player_payload(prospects, name="Nobody Here", team="X")["state"] == "no_markets"
            assert E.player_payload(prospects, slug="nobody-here-x")["state"] == "no_markets"
            assert E.player_payload(prospects)["state"] == "no_markets"
            # bare name: ok only when it maps to exactly one player on the board
            assert E.player_payload(prospects, name="Dante Moore")["state"] == "ok"
            # the tiebroken Jeremiah Smith rows all resolve to Ohio State, so a
            # bare name is unambiguous on the BOARD even though the cache has two
            smith = E.player_payload(prospects, name="Jeremiah Smith", team="Ohio State Buckeyes")
            assert smith["state"] == "ok" and not any(m["match_ambiguous"] for m in smith["markets"])
            assert E.player_payload(prospects, name="Jeremiah Smith", team="Louisiana Tech Bulldogs")["state"] == "no_markets"
            # ledger linkage: the FIRST serving response already carries call{}
            # and on_ledger for the tickers it just recorded (record-before-read)
            first = E.edge_payload(prospects)
            assert first["summary"]["on_ledger"] == len(E._LEDGER["entries"]) > 0
            recorded = {e["ticker"] for e in E._LEDGER["entries"]}
            assert all(_find(first["markets"], t)["call"] for t in recorded)
            on = E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")
            assert on["on_ledger"] is True and on["ledger_since"] == time.strftime("%Y-%m-%d", time.gmtime())
            top5 = on["markets"][0]
            assert top5["call"] == {"first_date": on["ledger_since"], "price_at_call": 77,
                                    "model_at_call": top5["model_prob"]}
            assert E.edge_payload(prospects)["summary"]["on_ledger"] == len(E._LEDGER["entries"])
            # a request with (name, team) after the memo is warm does no re-matching
            before = E._BOARD_MEMO["board"]
            E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")
            assert E._BOARD_MEMO["board"] is before
        # warming / seasonal states carry the state, never rows
        with _board_ctx(None, ledger):
            w = E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")
            assert w["state"] == "warming" and w["markets"] == [] and w["player"] is None
        with _board_ctx([], ledger):
            assert E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")["state"] == "seasonal"


def test_spotlight_selection_excludes_mid_wide_and_ambiguous():
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(rows, ledger):
            sp = E.spotlight_payload(prospects, n=3)
            assert sp["state"] == "ok" and len(sp["rows"]) == 3
            assert sp["headline"] == sp["rows"][0]
            board = E.edge_payload(prospects, dry_run=True)["markets"]
            want = sorted((r for r in board if r["model_prob"] is not None and r["ledger_eligible"]
                           and not r["match_ambiguous"]), key=lambda r: -abs(r["edge"]))
            assert [r["ticker"] for r in sp["rows"]] == [r["ticker"] for r in want[:3]]
            for r in sp["rows"]:
                assert r["match_ambiguous"] is False and r["ledger_eligible"] is True
                assert r["price_source"] in ("last", "mid")
                assert r["espn_id"] and r["espn_team_id"] and r["slug"] and r["question"]["label"]
                assert set(r) == {"ticker", "player", "team", "slug", "espn_id", "espn_team_id",
                                  "question", "market", "price_source", "model", "gap",
                                  "wide_interval", "ledger_eligible", "match_ambiguous", "url"}
            # the decisively tiebroken Jeremiah Smith (90c vs the model, the
            # board's largest gap) IS the headline; a non-decisive collision
            # would still be excluded (see test_decisive_tiebreak_is_not_ambiguous)
            assert sp["headline"]["player"] == "Jeremiah Smith" and sp["headline"]["market"] == 90
            assert sp["headline"]["team"] == "Ohio State Buckeyes" and sp["headline"]["match_ambiguous"] is False
            assert E.edge_payload(prospects, dry_run=True)["summary"]["top_gap_ticker"] == sp["headline"]["ticker"]
            gaps = [abs(r["gap"]) for r in sp["rows"]]
            assert gaps == sorted(gaps, reverse=True)
            # n clamps to 1..6; the fill path (n > eligible count) still never
            # admits mid_wide / ask / bid or ambiguous rows
            assert len(E.spotlight_payload(prospects, n=0)["rows"]) == 1
            assert len(E.spotlight_payload(prospects, n="x")["rows"]) == 3
            big = E.spotlight_payload(prospects, n=99)["rows"]
            assert len(big) == 6
            wide = E.spotlight_payload(prospects, n=6)
            assert all(r["price_source"] in ("last", "mid") and not r["match_ambiguous"] for r in wide["rows"])
            s = sp["summary"]
            assert set(s) == {"markets", "modeled", "disagreements", "on_ledger", "draft_year"}
            assert s["markets"] == 77 and s["modeled"] == 18 and s["draft_year"] == 2027
            assert s["disagreements"] == E.edge_payload(prospects, dry_run=True)["summary"]["eligible_over_threshold"]
            rec = sp["record"]
            assert rec["calls"] == 0 and rec["status"] == "open" and rec["brier"] is None
            # a board where only mid_wide rows carry a big gap → fill stays empty
            only_wide = [dict(r) for r in rows]
            for r in only_wide:
                if r["yes_price_source"] in ("last", "mid"):
                    r["yes_price_source"], r["ledger_eligible"] = "mid_wide", False
            with _board_ctx(only_wide, ledger):
                empty = E.spotlight_payload(prospects, n=3)
                assert empty["rows"] == [] and empty["headline"] is None
        with _board_ctx(None, ledger):
            w = E.spotlight_payload(prospects)
            assert w["state"] == "warming" and w["headline"] is None and w["rows"] == []
            assert w["record"]["calls"] == 0
        with _board_ctx([], ledger):
            assert E.spotlight_payload(prospects)["state"] == "seasonal"


def test_ledger_payload_enrichment_and_backfill_is_idempotent():
    rows, prospects = _rows(), _prospects()
    legacy = [{  # the shape of the 11 entries committed before the spec
        "date": "2026-09-10", "edge": -34.5, "market_price_cents": 77,
        "model_basis": ("model-implied P(pick <= 5) from the 72% conformal pick interval "
                        "[1, 18], log-uniform mass; scaled x0.742 so the event's matched "
                        "P(pick <= 5) sums to 5 (raw marginals overshot)"),
        "model_prob": 42.5, "player": "Arch Manning", "price_source": "last",
        "ticker": "KXNFLDRAFTTOP-27-5-AMAN",
        "title": "Will Arch Manning be a top 5 draft pick in 2027?",
    }, {
        "date": "2026-09-10", "edge": -40.1, "market_price_cents": 82,
        "model_basis": "model-implied P(pick <= 5) from the 72% conformal pick interval [1, 19], log-uniform mass",
        "model_prob": 41.9, "player": "Dylan Stewart", "price_source": "last",
        "ticker": "KXNFLDRAFTTOP-27-5-DSTE",
        "title": "Will Dylan Stewart be a top 5 draft pick in 2027?",
    }, {  # off-board ticker: identity must still come from the prospect index
        "date": "2026-09-09", "edge": 12.0, "market_price_cents": 30,
        "model_basis": "model-implied P(pick <= 10) from the 80% conformal pick interval [1, 7], log-uniform mass",
        "model_prob": 42.0, "player": "Dante Moore", "price_source": "mid",
        "ticker": "KXNFLDRAFTTOP-27-10-DMOO",
        "title": "Will Dante Moore be a top 10 draft pick in 2027?",
    }]
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with open(ledger, "w") as f:
            json.dump({"entries": legacy}, f)
        with _board_ctx(rows, ledger):
            # legacy rows serve the full shape with nulls before any backfill
            lp = E.ledger_payload()
            assert set(lp) == {"entries", "summary", "scoring", "note"}
            e0 = next(e for e in lp["entries"] if e["ticker"] == "KXNFLDRAFTTOP-27-5-AMAN")
            assert e0["team"] is None and e0["current_price_cents"] == 77  # live from the cache
            assert e0["resolution"] == {"outcome": None, "settled_at": None, "brier_model": None,
                                        "brier_market": None, "closer": None, "note": None}
            off = next(e for e in lp["entries"] if e["ticker"] == "KXNFLDRAFTTOP-27-10-DMOO")
            assert off["current_price_cents"] is None
            s = lp["summary"]
            assert s["calls"] == 3 and s["players"] == 3 and s["open"] == 3 and s["resolved"] == 0
            assert s["scored"] == 0 and s["brier"] is None and s["model_brier"] is None
            assert s["first_date"] == "2026-09-09" and s["last_date"] == "2026-09-10"
            assert s["avg_abs_gap"] == round((34.5 + 40.1 + 12.0) / 3, 1)
            assert s["scoring_note"] == "scored on draft night — Brier vs the real 2027 draft order"
            assert lp["scoring"] == {"rule": "brier", "settles_on": "2027-05-22",
                                     "status": "open", "last_synced_at": None}
            assert lp["entries"][0]["date"] >= lp["entries"][-1]["date"]  # newest first
            # backfill: fills only what is missing, from the basis string first
            assert E.backfill_ledger(prospects) == 3
            with open(ledger) as f:
                on_disk = {e["ticker"]: e for e in json.load(f)["entries"]}
            am = on_disk["KXNFLDRAFTTOP-27-5-AMAN"]
            assert am["team"] == "Texas Longhorns" and am["slug"] == "arch-manning-texas-longhorns"
            assert am["espn_id"] == "4870906" and am["espn_team_id"] == "251"
            assert am["event_ticker"] == "KXNFLDRAFTTOP-27-5"
            assert am["question"] == {"kind": "top_n", "n": 5, "label": "Top-5 pick"}
            assert am["model_interval"] == {"lo": 1, "hi": 18, "coverage": 0.72}  # record-time, not the cache's 0.8
            assert am["event_scale"] == 0.742 and am["close_time"] == "2027-05-22T14:00:00Z"
            assert am["resolution"]["outcome"] is None
            # untouched legacy values
            assert am["model_prob"] == 42.5 and am["market_price_cents"] == 77 and am["edge"] == -34.5
            ds = on_disk["KXNFLDRAFTTOP-27-5-DSTE"]
            assert ds["team"] == "South Carolina Gamecocks" and ds["event_scale"] == 1.0
            dm = on_disk["KXNFLDRAFTTOP-27-10-DMOO"]
            assert dm["team"] == "Oregon Ducks" and dm["espn_id"] == "4870921"
            assert dm["question"] == {"kind": "top_n", "n": 10, "label": "Top-10 pick"}
            assert dm["model_interval"] == {"lo": 1, "hi": 7, "coverage": 0.8}
            assert dm["event_ticker"] == "KXNFLDRAFTTOP-27-10" and "close_time" not in dm
            # idempotent: a second pass changes nothing and does not rewrite
            mtime = os.path.getmtime(ledger)
            assert E.backfill_ledger(prospects) == 0
            assert os.path.getmtime(ledger) == mtime
            with open(ledger) as f:
                assert {e["ticker"]: e for e in json.load(f)["entries"]} == on_disk
            # new entries are written enriched at record time
            E.edge_payload(prospects)
            with open(ledger) as f:
                fresh = [e for e in json.load(f)["entries"] if e["date"] != "2026-09-10" or e["ticker"] not in on_disk]
            assert fresh
            for e in fresh:
                assert e["team"] and e["slug"] and e["question"]["kind"] == "top_n"
                assert e["model_interval"] and e["event_scale"] and e["event_ticker"]
                assert e["resolution"]["outcome"] is None
            assert E.backfill_ledger(prospects) == 0
            # the on-board count in /api/edge counts tickers, not rows
            assert E.edge_payload(prospects, dry_run=True)["summary"]["on_ledger"] == \
                len({e["ticker"] for e in E._LEDGER["entries"]} & {r["ticker"] for r in rows})


def test_history_capture_bounds_and_prunes():
    from datetime import datetime, timedelta, timezone
    rows = _rows()
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, "history.json")
        with _board_ctx(rows, os.path.join(tmp, "ledger.json"), history_path=hist):
            t0 = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
            priced = [r for r in rows if r["yes_price_cents"] is not None]
            assert E._capture_history(rows, now=t0) == len(priced)
            assert os.path.exists(hist)
            h = E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")
            assert h == {"ticker": "KXNFLDRAFTTOP-27-5-AMAN", "bounded": True,
                         "points": [["2026-09-10", 77, "last"]]}
            assert E.history_payload("NOPE")["points"] == []  # 404-free
            assert E.history_payload(None)["points"] == []
            # same day, unchanged price → nothing; a 2c move → nothing; 3c → a point
            assert E._capture_history(rows, now=t0 + timedelta(minutes=10)) == 0
            aman = dict(_find(rows, "KXNFLDRAFTTOP-27-5-AMAN"))
            aman["yes_price_cents"] = 79
            assert E._capture_history([aman], now=t0 + timedelta(minutes=20)) == 0
            aman["yes_price_cents"] = 80
            assert E._capture_history([aman], now=t0 + timedelta(minutes=30)) == 1
            # a new UTC date → a point even with no move
            assert E._capture_history([aman], now=t0 + timedelta(days=1)) == 1
            pts = E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"]
            assert pts == [["2026-09-10", 77, "last"], ["2026-09-10", 80, "last"], ["2026-09-11", 80, "last"]]
            # unpriced rows never produce points
            assert E._capture_history([dict(aman, yes_price_cents=None)], now=t0 + timedelta(days=2)) == 0
            # hard bound: 400 points, oldest dropped (close pushed out so the
            # 30-days-past-close prune does not fire inside the loop)
            aman["close_time"] = "2030-01-01T00:00:00Z"
            for i in range(500):
                E._capture_history([aman], now=t0 + timedelta(days=2 + i))
            pts = E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"]
            assert len(pts) == E.HISTORY_MAX_POINTS == 400
            assert pts[-1][0] == (t0 + timedelta(days=501)).date().isoformat()
            assert pts[0][0] > "2026-09-11"  # the earliest points are gone
            # prune: every other fixture ticker was absent for the whole loop
            # (> 30 days) and is gone; the live one stays
            assert E.history_payload("KXNFLDRAFTTOP-27-5-JSMI")["points"] == []
            assert E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"]
            # a ticker seen once, then absent: kept at 29 days, pruned at 31
            t1 = t0 + timedelta(days=600)
            fresh = dict(aman, ticker="KXNFLDRAFTTOP-27-5-FRSH", close_time="2030-01-01T00:00:00Z")
            E._capture_history([aman, fresh], now=t1)
            E._capture_history([aman], now=t1 + timedelta(days=29))
            assert E.history_payload("KXNFLDRAFTTOP-27-5-FRSH")["points"]
            E._capture_history([aman], now=t1 + timedelta(days=31))
            assert E.history_payload("KXNFLDRAFTTOP-27-5-FRSH")["points"] == []
            assert E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"]
            # ... and a ticker 30 days past its close_time is pruned too
            closed = dict(aman, ticker="KXNFLDRAFTTOP-27-5-ZZZZ", close_time="2026-01-01T00:00:00Z")
            E._capture_history([closed], now=t0 + timedelta(days=600))
            assert E.history_payload("KXNFLDRAFTTOP-27-5-ZZZZ")["points"] == []
            # hot-reload: a file another worker wrote is picked up by mtime
            with open(hist) as f:
                data = json.load(f)
            data["tickers"]["KXNFLDRAFTTOP-27-5-AMAN"]["points"].append(["2028-01-01", 50, "last"])
            with open(hist, "w") as f:
                json.dump(data, f)
            os.utime(hist, (time.time() + 5, time.time() + 5))
            assert E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"][-1] == ["2028-01-01", 50, "last"]
            # the warm refresh never captures under DV_EDGE_NO_WARMUP (tests / --probe)
            fetch = E._fetch_relevant_markets
            E._fetch_relevant_markets = lambda: (rows, {})
            try:
                before = os.path.getmtime(hist)
                E._refresh_markets_once()
                assert os.path.getmtime(hist) == before
            finally:
                E._fetch_relevant_markets = fetch


# ── match_ambiguous == "the tiebreak was NOT decisive" ───────────────────────
def test_decisive_tiebreak_is_not_ambiguous():
    """The Smith / Louisiana-Tech shape: two cache rows share a name, the
    chosen one is a pick-1 prospect and the other sits ~3,700 ranks down the
    board. The tiebreak is decisive, so the market is NOT ambiguous and may
    headline. Two live prospects sharing a name stay ambiguous."""
    smith_osu = {"name": "Jeremiah Smith", "team": "Ohio State Buckeyes", "position": "WR",
                 "draft_class": 2028, "projected_pick": 1,
                 "pick_range": {"lo": 1, "hi": 30, "confidence": 0.8}}
    smith_lat = {"name": "Jeremiah Smith", "team": "Louisiana Tech Bulldogs", "position": "LB",
                 "draft_class": 2030, "projected_pick": 3725,
                 "pick_range": {"lo": 18, "hi": 4838, "confidence": 0.8}}
    index = E._build_prospect_index([smith_lat, smith_osu])
    row, amb = E._resolve_player(["Jeremiah Smith"], index, draft_year=2027)
    assert row is smith_osu and amb is False
    # a team hint pointing at the far-off row still resolves (and, chosen row
    # at pick 3725, is NOT decisive → ambiguous)
    row, amb = E._resolve_player(["Jeremiah Smith"], index, draft_year=2027, team_hint="Louisiana Tech")
    assert row is smith_lat and amb is True
    # alternative inside the top 300 → not decisive
    near = dict(smith_lat, projected_pick=250)
    row, amb = E._resolve_player(["Jeremiah Smith"], E._build_prospect_index([smith_osu, near]), draft_year=2027)
    assert row is smith_osu and amb is True
    # chosen row outside the top 100 → not decisive even with a far alternative
    late = dict(smith_osu, projected_pick=140)
    row, amb = E._resolve_player(["Jeremiah Smith"], E._build_prospect_index([late, smith_lat]), draft_year=2027)
    assert row is late and amb is True
    # alternative in another draft class with no market-eligible profile
    # (no pick_range) is decisive even at a near pick
    other_class = {"name": "Jeremiah Smith", "team": "X", "draft_class": 2029, "projected_pick": 120}
    row, amb = E._resolve_player(["Jeremiah Smith"], E._build_prospect_index([smith_osu, other_class]), draft_year=2027)
    assert row is smith_osu and amb is False
    # ... but the same row in the market's own class is a live alternative
    same_class = dict(other_class, draft_class=2027)
    row, amb = E._resolve_player(["Jeremiah Smith"], E._build_prospect_index([smith_osu, same_class]), draft_year=2027)
    assert amb is True
    assert E._tiebreak_decisive(smith_osu, [], 2027) is True
    # on the full fixture board every Smith / Stewart row is unambiguous and
    # the Smith top-5 row is the spotlight headline
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(prospects, dry_run=True)
            for r in p["markets"]:
                if r["matched_player"] in ("Jeremiah Smith", "Dylan Stewart"):
                    assert r["match_ambiguous"] is False, r["ticker"]
            assert p["summary"]["top_gap_ticker"] == "KXNFLDRAFTTOP-27-5-JSMI"
            assert E.spotlight_payload(prospects, n=1)["headline"]["ticker"] == "KXNFLDRAFTTOP-27-5-JSMI"


# ── --score-ledger: resolution + Brier, aggregated into summary/record ───────
def _ledger_entry(ticker, player, model_prob, price, **extra):
    return dict({"date": "2026-09-10", "ticker": ticker, "player": player,
                 "title": f"Will {player} be a top 5 draft pick in 2027?",
                 "model_prob": model_prob, "market_price_cents": price, "price_source": "last",
                 "edge": round(model_prob - price, 1), "model_basis": "model-implied P(pick <= 5) x",
                 "close_time": "2027-05-22T14:00:00Z"}, **extra)


def test_score_ledger_fills_resolution_and_summary_aggregates():
    entries = [
        _ledger_entry("KXNFLDRAFTTOP-27-5-JSMI", "Jeremiah Smith", 37.3, 90),
        _ledger_entry("KXNFLDRAFTTOP-27-5-AMAN", "Arch Manning", 42.5, 77),
        _ledger_entry("KXNFLDRAFTTOP-27-5-AMAN", "Arch Manning", 40.0, 80, date="2026-09-11"),
        _ledger_entry("KXNFLDRAFTTOP-27-5-GONE", "Nobody Listed", 20.0, 40),
        _ledger_entry("KXNFLDRAFTTOP-27-5-OPEN", "Still Open", 30.0, 50),
        _ledger_entry("KXNFLDRAFTTOP-27-5-DOWN", "Upstream Down", 30.0, 50),
    ]
    results = {
        "KXNFLDRAFTTOP-27-5-JSMI": ("no", "2027-04-24T03:00:00Z", None),
        "KXNFLDRAFTTOP-27-5-AMAN": ("yes", "2027-04-24T03:00:00Z", None),
        "KXNFLDRAFTTOP-27-5-GONE": ("void", None, "market no longer listed on Kalshi"),
        "KXNFLDRAFTTOP-27-5-OPEN": (None, None, None),
    }
    fetched = []

    def fake_fetch(ticker):
        fetched.append(ticker)
        if ticker == "KXNFLDRAFTTOP-27-5-DOWN":
            raise RuntimeError("503")
        return results[ticker]

    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with open(ledger, "w") as f:
            json.dump({"entries": entries}, f)
        with _board_ctx(_rows(), ledger):
            c = E.score_ledger(fetch_result=fake_fetch)
            assert c == {"tickers": 5, "scored": 2, "void": 1, "open": 1, "errors": 1}, c
            assert sorted(fetched) == sorted(list(results) + ["KXNFLDRAFTTOP-27-5-DOWN"])
            assert len(set(fetched)) == len(fetched)  # one GET per distinct ticker
            lp = E.ledger_payload()
            by = {(e["ticker"], e["date"]): e for e in lp["entries"]}
            smith = by[("KXNFLDRAFTTOP-27-5-JSMI", "2026-09-10")]["resolution"]
            assert smith == {"outcome": "no", "settled_at": "2027-04-24T03:00:00Z",
                             "brier_model": round((0.373 - 0) ** 2, 4),
                             "brier_market": round((0.90 - 0) ** 2, 4),
                             "closer": "model", "note": None}
            am1 = by[("KXNFLDRAFTTOP-27-5-AMAN", "2026-09-10")]["resolution"]
            am2 = by[("KXNFLDRAFTTOP-27-5-AMAN", "2026-09-11")]["resolution"]
            assert am1["outcome"] == "yes" and am1["closer"] == "market"
            assert am1["brier_model"] == round((0.425 - 1) ** 2, 4) and am1["brier_market"] == round((0.77 - 1) ** 2, 4)
            assert am2["brier_model"] == round((0.40 - 1) ** 2, 4)  # scored on its own numbers
            gone = by[("KXNFLDRAFTTOP-27-5-GONE", "2026-09-10")]["resolution"]
            assert gone["outcome"] == "void" and gone["brier_model"] is None and gone["closer"] is None
            assert gone["settled_at"] and gone["note"] == "market no longer listed on Kalshi"
            for t in ("KXNFLDRAFTTOP-27-5-OPEN", "KXNFLDRAFTTOP-27-5-DOWN"):
                assert by[(t, "2026-09-10")]["resolution"]["outcome"] is None
            # summary / record aggregate the resolution fields
            s = lp["summary"]
            assert s["calls"] == 6 and s["resolved"] == 3 and s["scored"] == 3 and s["void"] == 1
            assert s["open"] == 2 and s["model_closer"] == 1 and s["market_closer"] == 2 and s["push"] == 0
            bm = [smith["brier_model"], am1["brier_model"], am2["brier_model"]]
            bk = [smith["brier_market"], am1["brier_market"], am2["brier_market"]]
            assert s["model_brier"] == round(sum(bm) / 3, 4) and s["market_brier"] == round(sum(bk) / 3, 4)
            assert s["brier"] == s["model_brier"]
            assert lp["scoring"]["status"] == "partial"
            assert lp["scoring"]["settles_on"] == "2027-05-22"  # only open tickers count
            rec = E.spotlight_payload(_prospects(), n=1)["record"]
            assert rec["scored"] == 3 and rec["resolved"] == 3 and rec["void"] == 1
            assert rec["model_closer"] == 1 and rec["market_closer"] == 2 and rec["status"] == "partial"
            assert rec["model_brier"] == s["model_brier"] and rec["market_brier"] == s["market_brier"]
            # idempotent: settled entries are never re-fetched or rewritten
            fetched.clear()
            mtime = os.path.getmtime(ledger)
            results["KXNFLDRAFTTOP-27-5-JSMI"] = ("yes", "later", None)  # would flip if re-scored
            c = E.score_ledger(fetch_result=fake_fetch)
            assert c == {"tickers": 2, "scored": 0, "void": 0, "open": 1, "errors": 1}
            assert sorted(fetched) == ["KXNFLDRAFTTOP-27-5-DOWN", "KXNFLDRAFTTOP-27-5-OPEN"]
            assert os.path.getmtime(ledger) == mtime
            assert E.ledger_payload()["summary"]["model_closer"] == 1
            # once everything settles the status is 'scored'
            results["KXNFLDRAFTTOP-27-5-OPEN"] = ("no", None, None)
            E.score_ledger(fetch_result=lambda t: results.get(t, ("void", None, "gone")))
            lp = E.ledger_payload()
            assert lp["summary"]["open"] == 0 and lp["scoring"]["status"] == "scored"
            assert lp["scoring"]["settles_on"] is None
    # status derivation: void-only is not 'open'
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with open(ledger, "w") as f:
            json.dump({"entries": [_ledger_entry("T-1", "A", 30.0, 50)]}, f)
        with _board_ctx(_rows(), ledger):
            E.score_ledger(fetch_result=lambda t: ("void", None, "gone"))
            lp = E.ledger_payload()
            assert lp["summary"]["void"] == 1 and lp["summary"]["open"] == 0
            assert lp["scoring"]["status"] == "scored" and lp["summary"]["model_brier"] is None
            assert E.score_ledger(fetch_result=lambda t: (_ for _ in ()).throw(AssertionError("refetched"))) == \
                {"tickers": 0, "scored": 0, "void": 0, "open": 0, "errors": 0}
    # a fresh pre-draft ledger is a no-op
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(_rows(), os.path.join(tmp, "ledger.json")):
            assert E.score_ledger(fetch_result=lambda t: ("yes", None, None))["tickers"] == 0


def test_fetch_market_result_reads_public_market_shape():
    class _Resp:
        def __init__(self, status, body=None):
            self.status_code, self._body = status, body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._body

    class _Session:
        def __init__(self, resp):
            self.resp, self.urls = resp, []

        def get(self, url, timeout=None):
            self.urls.append(url)
            return self.resp

    sess = _Session(_Resp(200, {"market": {"status": "finalized", "result": "yes",
                                           "settlement_ts": "2027-04-24T03:00:00Z"}}))
    assert E._fetch_market_result(sess, "T-1") == ("yes", "2027-04-24T03:00:00Z", None)
    assert sess.urls == [E.KALSHI_BASE + "/markets/T-1"]  # public market read, nothing else
    assert "/orders" not in sess.urls[0] and "/portfolio" not in sess.urls[0]
    sess = _Session(_Resp(200, {"market": {"status": "open", "result": ""}}))
    assert E._fetch_market_result(sess, "T-1") == (None, None, None)
    sess = _Session(_Resp(404, {"error": {"code": "not_found"}}))
    assert E._fetch_market_result(sess, "T-1")[0] == "void"
    sess = _Session(_Resp(200, {"market": {"status": "settled", "result": "void", "settlement_ts": "x"}}))
    assert E._fetch_market_result(sess, "T-1") == ("void", "x", "market voided by Kalshi")
    sess = _Session(_Resp(503, {}))
    try:
        E._fetch_market_result(sess, "T-1")
        raise AssertionError("503 should raise so the call stays open")
    except RuntimeError:
        pass
    # the module CODE (past the policy docstring, which names them as the
    # things it never touches) references no order/portfolio endpoint or key
    with open(os.path.join(ROOT, "dv_edge.py")) as f:
        code = f.read().split('"""', 2)[2]
    assert "/orders" not in code and "/portfolio" not in code and "api_key" not in code.lower()


# ── --sync-history: pull the remote capture into the local file ──────────────
def test_sync_history_merges_remote_points_bounded():
    remote_board = {"markets": [{"ticker": "KXNFLDRAFTTOP-27-5-AMAN"}, {"ticker": "KXNFLDRAFTTOP-27-5-JSMI"}]}
    remote_ledger = {"entries": [{"ticker": "KXNFLDRAFTTOP-27-10-DMOO"}]}
    remote_hist = {
        "KXNFLDRAFTTOP-27-5-AMAN": [["2026-09-09", 75, "last"], ["2026-09-10", 77, "last"], ["2026-09-10", 80, "last"]],
        "KXNFLDRAFTTOP-27-5-JSMI": [[f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", 50 + i % 40, "last"] for i in range(220)],
        "KXNFLDRAFTTOP-27-10-DMOO": [],
    }
    calls = []

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            calls.append((url, dict(params or {})))
            if url.endswith("/api/edge"):
                return _Resp(remote_board)
            if url.endswith("/api/edge/ledger"):
                return _Resp(remote_ledger)
            assert url.endswith("/api/edge/history"), url
            return _Resp({"ticker": params["ticker"], "points": remote_hist[params["ticker"]], "bounded": True})

    saved = E.requests.Session
    E.requests.Session = _Session
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hist = os.path.join(tmp, "history.json")
            with _board_ctx(_rows(), os.path.join(tmp, "ledger.json"), history_path=hist):
                # a local point the remote lacks (a different serving process
                # captured it) survives the merge; the remote order wins in-day
                E._HISTORY["tickers"]["KXNFLDRAFTTOP-27-5-AMAN"] = {
                    "points": [["2026-09-10", 77, "last"], ["2026-09-08", 70, "mid"]],
                    "close_time": "2027-05-22T14:00:00Z", "last_seen": "2026-09-10"}
                E._write_json_atomic(hist, {"tickers": E._HISTORY["tickers"]})
                E._HISTORY_MTIME = os.path.getmtime(hist)
                # 300 local points on JSMI + 220 remote → bounded to 400
                E._HISTORY["tickers"]["KXNFLDRAFTTOP-27-5-JSMI"] = {
                    "points": [[f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}", 30, "last"] for i in range(300)],
                    "close_time": None, "last_seen": "2025-11-01"}
                E._write_json_atomic(hist, {"tickers": E._HISTORY["tickers"]})
                E._HISTORY_MTIME = os.path.getmtime(hist)
                added = E.sync_history_from("https://example.test/")
                assert added == 2 + 100  # AMAN: 2 new; JSMI: 300 + 220 capped at 400
                urls = [u for u, _ in calls]
                assert urls[:2] == ["https://example.test/api/edge", "https://example.test/api/edge/ledger"]
                assert sorted(p["ticker"] for u, p in calls if u.endswith("/history")) == \
                    ["KXNFLDRAFTTOP-27-10-DMOO", "KXNFLDRAFTTOP-27-5-AMAN", "KXNFLDRAFTTOP-27-5-JSMI"]
                aman = E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"]
                assert aman == [["2026-09-08", 70, "mid"], ["2026-09-09", 75, "last"],
                                ["2026-09-10", 77, "last"], ["2026-09-10", 80, "last"]]
                jsmi = E.history_payload("KXNFLDRAFTTOP-27-5-JSMI")["points"]
                assert len(jsmi) == E.HISTORY_MAX_POINTS and jsmi[-1][0].startswith("2026-")
                assert E.history_payload("KXNFLDRAFTTOP-27-10-DMOO")["points"] == []  # empty remote: nothing created
                with open(hist) as f:
                    on_disk = json.load(f)
                assert "KXNFLDRAFTTOP-27-10-DMOO" not in on_disk["tickers"]
                rec = on_disk["tickers"]["KXNFLDRAFTTOP-27-5-AMAN"]
                assert rec["close_time"] == "2027-05-22T14:00:00Z" and rec["last_seen"] == "2026-09-10"
                assert on_disk["tickers"]["KXNFLDRAFTTOP-27-5-JSMI"]["last_seen"] == jsmi[-1][0]
                assert on_disk["last_synced_at"]
                # idempotent: a second sync adds nothing and keeps the shape
                assert E.sync_history_from("https://example.test") == 0
                assert E.history_payload("KXNFLDRAFTTOP-27-5-AMAN")["points"] == aman
    finally:
        E.requests.Session = saved
    assert E._merge_history_points([], [["2026-01-01", "77", "last"], ["bad"], ["2026-01-01", 77, "last"]]) == \
        [["2026-01-01", 77, "last"]]


# ── Cache-Control per payload (the XGBOost.py after_request hook reads this) ─
def test_cache_control_follows_payload_state():
    cc = E.cache_control_for
    assert cc({"discovery": {"state": "ok"}, "note": "x"}) == "public, max-age=120"
    assert cc({"discovery": {"state": "stale"}}) == "public, max-age=120"
    assert cc({"discovery": {"state": "warming"}, "note": "warming"}) == "no-cache"
    assert cc({"discovery": {"state": "error"}, "note": "seasonal"}) == "no-cache"
    assert cc({"state": "warming", "discovery": {"state": "warming", "age_s": None}}) == "no-cache"
    assert cc({"state": "seasonal", "discovery": {"state": "ok"}}) == "public, max-age=120"
    assert cc({"state": "no_markets", "discovery": {"state": "ok"}}) == "public, max-age=120"
    assert cc({"entries": [], "summary": {}}) == "public, max-age=120"  # ledger/history shapes
    assert cc({"ticker": "T", "points": [], "bounded": True}) == "public, max-age=120"
    assert cc(None) == "no-cache"
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(None, ledger):
            assert cc(E.edge_payload(_prospects())) == "no-cache"
            assert cc(E.spotlight_payload(_prospects())) == "no-cache"
            assert cc(E.player_payload(_prospects(), name="Arch Manning", team="Texas Longhorns")) == "no-cache"
        with _board_ctx(_rows(), ledger):
            assert cc(E.edge_payload(_prospects(), dry_run=True)) == "public, max-age=120"
            assert cc(E.spotlight_payload(_prospects())) == "public, max-age=120"
        with _board_ctx([], ledger):
            E._kalshi_cache.update({"fetched_at_utc": None, "error": "boom"})
            assert E._discovery_status()["state"] == "error"
            assert cc(E.edge_payload(_prospects())) == "no-cache"
            assert cc(E.spotlight_payload(_prospects())) == "no-cache"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
