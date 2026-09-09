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
    row, amb = E._resolve_player(["Jeremiah Smith"], index, draft_year=2027, team_hint="")
    assert amb and row["team"] == "Ohio State Buckeyes"
    row, amb = E._resolve_player(["Dylan Stewart"], index, draft_year=2027)
    assert amb and row["team"] == "South Carolina Gamecocks"
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
        for r in p["markets"]:
            assert set(r) == row_keys, set(r) ^ row_keys
            assert r["model_basis"]  # every row explains itself
        s = p["summary"]
        assert s["markets"] == 77 and s["priced"] == 77
        assert s["matched"] == 70, s  # 65 exact + 5 recovered by the tiebreak
        assert s["modeled"] == 18, s  # 19 top-5 minus 1 not in cache; #1-pick rows are listed only
        assert s["edges_over_threshold"] >= 5
        heisman = [r for r in p["markets"] if r["series"] == "KXHEISMAN"]
        assert heisman and all(r["edge"] is None for r in heisman)
        pick1 = [r for r in p["markets"] if r["series"] == "KXNFLDRAFTPICK"]
        assert len(pick1) == 20 and all(r["model_prob"] is None for r in pick1)
        assert all(r["model_basis"] == E._NOT_PRICED_EXACT for r in pick1 if r["matched_player"])
        smith = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
        assert smith["matched_team"] == "Ohio State Buckeyes" and smith["match_ambiguous"]
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
