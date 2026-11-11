"""Regression tests for dv_edge against SAVED raw Kalshi payloads.

Fixtures under tests/fixtures/ are verbatim public-API responses
(GET /trade-api/v2/markets?series_ticker=…&status=open, 2026-09-09) plus the
prospect-cache rows those markets name, with pick_range attached from the
local v5 /predict. The served top-N probabilities come from the committed
models/topn_calibration.json (rank calibration), which the tests read too. If Kalshi drifts its schema again (the 2026 move from
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
import math
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


def _crowded_prospects():
    """The fixture cache with every 2027-class player ranked 1-8 moved to
    rank 1: eight ~95% top-5 marginals, so the KXNFLDRAFTTOP-27-5 event
    overshoots 5 and the coherence scaling must engage. (On the real fixture
    ranks the event sums to about 4.5 and is left alone.)"""
    return [dict(r, projected_pick=1) if (r["draft_class"] == 2027 and r["projected_pick"] <= 8) else r
            for r in _prospects()]


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
    assert E._yes_price(top["KXNFLDRAFTTOP-27-5-AMAN"]) == (77, "last")   # "0.7700", book 66/77
    # last "0.9000" sits ABOVE a live 72/89 book -> a stale print; the book's
    # midpoint is the price (see test_stale_last_outside_live_book_prices_at_mid)
    jsmi = top["KXNFLDRAFTTOP-27-5-JSMI"]
    assert (jsmi["last_price_dollars"], jsmi["yes_bid_dollars"], jsmi["yes_ask_dollars"]) == \
        ("0.9000", "0.7200", "0.8900")
    assert E._yes_price(jsmi) == (80, "mid")
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


def _curves():
    """The committed calibration curves, read independently of dv_edge."""
    with open(os.path.join(ROOT, "models", "topn_calibration.json")) as f:
        return {int(k): (v["a"], v["b"]) for k, v in json.load(f)["curves"].items()}


def _curve_p(n, rank):
    a, b = _curves()[n]
    return 1.0 / (1.0 + math.exp(-(a + b * math.log(rank))))


def test_model_fields_policy():
    rows = _rows()
    index = E._build_prospect_index(_prospects())
    manning = index["arch manning"][0]
    assert manning["projected_pick"] == 3
    top5 = _find(rows, "KXNFLDRAFTTOP-27-5-AMAN")
    want = round(100.0 * _curve_p(5, 3), 1)  # the rank-3 point of the top-5 curve
    assert 60.0 < want < 66.0, want
    prob, edge, basis = E._model_fields(top5, manning)
    assert prob == want and edge == round(want - 77, 1)
    assert basis.startswith("model P(pick <= 5) for its class-rank-3 player")
    assert "conformal" not in basis and "log-uniform" not in basis  # the interval formula is gone
    # exact-pick markets are listed only — never priced
    pick1 = _find(rows, "KXNFLDRAFTPICK-27-1-AMAN")
    prob, edge, basis = E._model_fields(pick1, manning)
    assert prob is None and edge is None
    assert basis == "not priced: exact-pick probabilities are not identifiable from an interval"
    # the event-coherence scale multiplies P(pick <= n) and is recorded
    half = round(100.0 * _curve_p(5, 3) * 0.5, 1)
    prob, edge, basis = E._model_fields(top5, manning, scale=0.5)
    assert prob == half and edge == round(half - 77, 1)
    assert "scaled x0.500" in basis and "sums to 5" in basis
    for bad in (0.0, -1.0, 2.0, float("nan"), "x", None):
        assert E._model_fields(top5, manning, scale=bad)[0] == want, bad
    # after_n is the complement of the same curve
    after5 = dict(top5, _question={"kind": "after_n", "n": 5})
    prob, edge, basis = E._model_fields(after5, manning)
    assert prob == round(100.0 * (1.0 - _curve_p(5, 3)), 1)
    assert basis.startswith("model P(pick > 5) for its class-rank-3 player")
    # Heisman: listed, never priced
    prob, edge, basis = E._model_fields(_find(rows, "KXHEISMAN-27-AMANN"), manning)
    assert prob is None and edge is None and basis.startswith("not priced")
    # the probability no longer depends on the conformal interval at all: a
    # row with no pick_range (or a wildly different one) prices identically
    bare = {k: v for k, v in manning.items() if k != "pick_range"}
    assert E._model_fields(top5, bare)[0] == want
    wide = dict(manning, pick_range={"lo": 1, "hi": 3000, "confidence": 0.8})
    assert E._model_fields(top5, wide)[0] == want
    # the rank is what matters: _market_rank (set by _build_board) wins over
    # the row's projected_pick; without either the row is not priced
    prob, edge, basis = E._model_fields(dict(top5, _market_rank=2), manning)
    assert prob == round(100.0 * _curve_p(5, 2), 1) > want and "class-rank-2 player" in basis
    # ... and the ceiling applies: a rank-1 player is 95.0, not the raw curve's 95.1
    assert E._model_fields(dict(top5, _market_rank=1), manning)[0] == 95.0
    assert E._class_rank(manning, {"_market_rank": 12}) == 12
    assert E._class_rank(manning, {}) == 3 and E._class_rank(manning) == 3
    for no_rank in (None, 0, -4, "x"):
        norank = dict(manning, projected_pick=no_rank)
        assert E._class_rank(norank, {}) is None, no_rank
        prob, edge, basis = E._model_fields(top5, norank)
        assert prob is None and edge is None and basis.startswith("not priced: no rank for this player")
    # a threshold no curve was fit for is not priced (never interpolated)
    top7 = dict(top5, _question={"kind": "top_n", "n": 7})
    prob, edge, basis = E._model_fields(top7, manning)
    assert prob is None and edge is None
    assert basis == "not priced: no calibration curve for a top-7 question"
    assert E._raw_p_le(top7, manning) is None
    assert abs(E._raw_p_le(top5, manning) - _curve_p(5, 3)) < 1e-12
    assert E._raw_p_le(pick1, manning) is None
    # no quote -> not priced
    prob, edge, basis = E._model_fields(dict(top5, yes_price_cents=None), manning)
    assert prob is None and "quote" in basis


def test_p_top_n_curve_properties():
    curves = _curves()
    assert set(curves) == {1, 3, 5, 10, 16, 32, 64, 100}
    assert set(E._topn_curves()) == set(curves)
    lo, hi = E._TOPN_P_FLOOR, E._TOPN_P_CEIL
    assert (lo, hi) == (0.005, 0.95)
    for n in sorted(curves):
        prev = 1.0
        for rank in list(range(1, 301)) + [500, 1000, 4000]:
            p = E._p_top_n(n, rank)
            assert lo <= p <= hi, (n, rank, p)          # clipped: never certainty
            assert p <= prev + 1e-12, (n, rank, p, prev)  # monotone non-increasing in rank
            prev = p
            # equals the fitted curve wherever the clip is not binding
            raw = _curve_p(n, rank)
            assert abs(p - min(max(raw, lo), hi)) < 1e-9, (n, rank)
        assert E._p_top_n(n, 4000) == lo  # far off the board hits the floor, not 0
    # strictly decreasing where unclipped
    assert E._p_top_n(5, 1) > E._p_top_n(5, 3) > E._p_top_n(5, 10) > E._p_top_n(5, 30) > lo
    # increasing in N at a fixed rank (a top-10 pick is at least as likely as a top-5)
    for rank in (1, 2, 3, 5, 8, 15, 30, 60, 150, 400):
        ps = [E._p_top_n(n, rank) for n in sorted(curves)]
        assert ps == sorted(ps), (rank, ps)
    assert E._p_top_n(1, 10) < E._p_top_n(10, 10) < E._p_top_n(100, 10)
    # both clips really bind: the raw curve overshoots at the top / undershoots at the bottom
    assert _curve_p(32, 1) > hi and E._p_top_n(32, 1) == hi
    assert _curve_p(100, 1) > hi and E._p_top_n(100, 1) == hi
    assert _curve_p(1, 300) < lo and E._p_top_n(1, 300) == lo
    assert E._p_top_n(1, 1) < 0.5  # even the #1-ranked player is not a favourite to go first
    # no curve for that N -> None (no interpolation between 5 and 10); bad ranks -> None
    for n in (2, 4, 7, 20, 50, 260):
        assert E._p_top_n(n, 1) is None, n
    assert E._p_top_n(5, None) is None and E._p_top_n(5, 0) is None and E._p_top_n(5, 0.5) is None
    assert E._p_top_n("5", 3) == E._p_top_n(5, 3)
    # a missing / unreadable calibration file prices nothing instead of raising
    saved = (E.TOPN_CALIBRATION_PATH, E._TOPN_CURVES)
    try:
        E.TOPN_CALIBRATION_PATH, E._TOPN_CURVES = os.path.join(ROOT, "models", "nope.json"), None
        assert E._topn_curves() == {} and E._p_top_n(5, 1) is None
        top5 = _find(_rows(), "KXNFLDRAFTTOP-27-5-AMAN")
        assert E._model_fields(top5, {"name": "A", "projected_pick": 1})[0] is None
        # a curve with a non-negative slope (would rise with rank) is rejected
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "cal.json")
            with open(bad, "w") as f:
                json.dump({"curves": {"5": {"a": 1.0, "b": 0.5}, "10": {"a": 5.0, "b": -2.0},
                                      "16": {"a": "x", "b": -1.0}}}, f)
            E.TOPN_CALIBRATION_PATH, E._TOPN_CURVES = bad, None
            assert set(E._topn_curves()) == {10}
    finally:
        E.TOPN_CALIBRATION_PATH, E._TOPN_CURVES = saved
    assert set(E._topn_curves()) == set(curves)


# ── Quote quality: stale prints, live books, disagreement ────────────────────
def test_stale_last_outside_live_book_prices_at_mid():
    yp = E._yes_price
    # last above the ask / below the bid of a live two-sided book -> midpoint
    assert yp({"last_price": 90, "yes_bid": 72, "yes_ask": 89}) == (80, "mid")
    assert yp({"last_price": 10, "yes_bid": 40, "yes_ask": 44}) == (42, "mid")
    assert yp({"last_price_dollars": "0.9000", "yes_bid_dollars": "0.2000",
               "yes_ask_dollars": "0.3000"}) == (25, "mid")
    # inside the book, or exactly on either edge, the last trade stands
    assert yp({"last_price": 42, "yes_bid": 40, "yes_ask": 44}) == (42, "last")
    assert yp({"last_price": 40, "yes_bid": 40, "yes_ask": 44}) == (40, "last")
    assert yp({"last_price": 44, "yes_bid": 40, "yes_ask": 44}) == (44, "last")
    assert yp({"last_price": 45, "yes_bid": 40, "yes_ask": 44}) == (42, "mid")
    assert yp({"last_price": 39, "yes_bid": 40, "yes_ask": 44}) == (42, "mid")
    # no LIVE two-sided book to contradict it -> the last trade is all there is
    assert yp({"last_price": 90, "yes_bid": 0, "yes_ask": 50}) == (90, "last")    # nobody bidding
    assert yp({"last_price": 90, "yes_bid": 20, "yes_ask": 100}) == (90, "last")  # nobody offering
    assert yp({"last_price": 90, "yes_ask": 50}) == (90, "last")
    assert yp({"last_price": 90, "yes_bid": 50}) == (90, "last")
    assert yp({"last_price": 90, "yes_bid": 60, "yes_ask": 50}) == (90, "last")   # crossed book
    assert yp({"last_price": 90}) == (90, "last")
    # the market row carries the corrected price, the book, and a mid/mid_wide tag
    raw = {m["ticker"]: m for _, m in _raw_markets()}["KXNFLDRAFTTOP-27-5-JSMI"]
    row = E._market_row(raw, "", "KXNFLDRAFTTOP", raw["event_ticker"])
    assert (row["yes_price_cents"], row["bid_cents"], row["ask_cents"]) == (80, 72, 89)
    assert row["yes_price_source"] == "mid_wide" and row["spread_cents"] == 17  # 17c > the tight limit
    assert row["ledger_eligible"] is True  # a live two-sided book, even though wide
    tight = E._market_row(dict(raw, yes_bid_dollars="0.8000", yes_ask_dollars="0.8800"),
                          "", "KXNFLDRAFTTOP", raw["event_ticker"])
    assert (tight["yes_price_cents"], tight["yes_price_source"]) == (84, "mid")
    # every fixture row: a 'last' price is never outside its own live book
    stale = 0
    for _, m in _raw_markets():
        r = E._market_row(m, "", "X", m.get("event_ticker") or "")
        bid, ask = r["bid_cents"], r["ask_cents"]
        live = bid is not None and ask is not None and 0 < bid <= ask < 100
        last = E._price_field_cents(m, "last_price")
        if live and last is not None and 0 < last < 100 and not (bid <= last <= ask):
            stale += 1
            assert r["yes_price_source"] in ("mid", "mid_wide"), m["ticker"]
            assert bid <= r["yes_price_cents"] <= ask
        if live and r["yes_price_source"] == "last":
            assert bid <= r["yes_price_cents"] <= ask, m["ticker"]
    assert stale >= 1, "fixture should contain at least one stale last print"


def test_volume_only_book_is_not_ledger_eligible():
    def qq(m):
        price, source = E._yes_price(m)
        return E._quote_quality(m, price, source)

    # traded volume with nobody bidding: the old rule took this row; no more
    src, spread, vol, elig = qq({"last_price": 17, "yes_bid": 0, "yes_ask": 85, "volume": 4})
    assert (src, vol, elig) == ("last", 4.0, False)
    # huge lifetime volume, bid below the minimum -> still not a live book
    assert qq({"last_price": 30, "yes_bid": 1, "yes_ask": 60, "volume": 5000})[3] is False
    assert E.LEDGER_MIN_BID_CENTS == 2
    assert qq({"last_price": 30, "yes_bid": 2, "yes_ask": 60, "volume": 0})[3] is True  # ... at the minimum it is
    # nobody offering (ask 100 / missing) -> not eligible whatever the volume
    assert qq({"last_price": 50, "yes_bid": 40, "yes_ask": 100, "volume": 900})[3] is False
    assert qq({"last_price": 50, "yes_bid": 40, "volume": 900})[3] is False
    assert qq({"last_price": 50, "volume": 900})[3] is False
    # one-sided sources are never eligible
    assert qq({"yes_ask": 38, "volume": 10}) == ("ask", None, 10.0, False)
    assert qq({"yes_bid": 38, "volume": 10})[0] == "bid" and qq({"yes_bid": 38, "volume": 10})[3] is False
    # a live two-sided book qualifies with ZERO volume, tight or wide
    assert qq({"yes_bid": 40, "yes_ask": 44, "volume": 0}) == ("mid", 4, 0.0, True)
    assert qq({"yes_bid": 20, "yes_ask": 60}) == ("mid_wide", 40, None, True)  # mid_wide CAN be eligible
    assert qq({"last_price": 30, "yes_bid": 20, "yes_ask": 60}) == ("last", 40, None, True)
    # ... but a placeholder 1c bid under a wide ask is mid_wide and NOT eligible
    assert qq({"yes_bid": 1, "yes_ask": 65, "volume": 0}) == ("mid_wide", 64, 0.0, False)
    assert E._quote_quality({"yes_bid": 40, "yes_ask": 44}, None, None)[3] is False
    # on the fixtures: eligibility is exactly "live two-sided book right now"
    seen_volume_only = False
    for r in _rows():
        bid, ask = r["bid_cents"], r["ask_cents"]
        live = (bid is not None and ask is not None and E.LEDGER_MIN_BID_CENTS <= bid <= ask < 100
                and r["yes_price_source"] in ("last", "mid", "mid_wide"))
        assert r["ledger_eligible"] is live, r["ticker"]
        if (r["volume"] or 0) > 0 and not live:
            seen_volume_only = True
    assert seen_volume_only, "fixture should contain a traded market without a live book"
    rwil = _find(_rows(), "KXNFLDRAFTTOP-27-5-RWIL")  # last 17c, bid 0, ask 85, 4 contracts traded
    assert rwil["volume"] == 4.0 and rwil["bid_cents"] == 0 and rwil["ledger_eligible"] is False


def test_is_disagreement_boundary():
    d = E._is_disagreement
    assert E.LEDGER_EDGE_THRESHOLD == 10.0
    base = {"ledger_eligible": True, "bid_cents": 20, "ask_cents": 43}
    # at least 10 points ABOVE the ask
    assert d(dict(base, model_prob=53.0)) is True       # exactly ask + 10 counts
    assert d(dict(base, model_prob=52.9)) is False
    assert d(dict(base, model_prob=90.0)) is True
    # at least 10 points BELOW the bid
    assert d(dict(base, model_prob=10.0)) is True       # exactly bid - 10 counts
    assert d(dict(base, model_prob=10.1)) is False
    assert d(dict(base, model_prob=0.5)) is True
    # inside the book, on its edges, or just outside: the model agrees with a quote
    for p in (20.0, 31.5, 43.0, 45.0, 19.0, 11.0, 52.0):
        assert d(dict(base, model_prob=p)) is False, p
    # far from the MIDPOINT is not enough: 20/43 book (mid 31.5), model 45 is
    # 13.5 off the mid but only 2 above the ask
    assert abs(45.0 - 31.5) >= E.LEDGER_EDGE_THRESHOLD and d(dict(base, model_prob=45.0)) is False
    # not a live book / not modeled / book missing -> never a disagreement
    assert d(dict(base, model_prob=90.0, ledger_eligible=False)) is False
    assert d(dict(base, model_prob=None)) is False
    assert d({"ledger_eligible": True, "model_prob": 90.0}) is False
    assert d(dict(base, model_prob=90.0, bid_cents=None)) is False
    assert d(dict(base, model_prob=90.0, ask_cents=None)) is False
    assert d({}) is False
    # a bid under 10c can never be undercut by 10 points; only the ask side can fire
    low = {"ledger_eligible": True, "bid_cents": 5, "ask_cents": 9}
    assert d(dict(low, model_prob=0.5)) is False and d(dict(low, model_prob=19.0)) is True
    # every board row's flag is this function of the row, and the summary counts it
    rows = _rows()
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(_prospects())
    flagged = [r for r in p["markets"] if r["disagreement"]]
    assert flagged and all(isinstance(r["disagreement"], bool) for r in p["markets"])
    for r in p["markets"]:
        assert r["disagreement"] is d(r), r["ticker"]
        if r["disagreement"]:
            assert r["ledger_eligible"] and r["model_prob"] is not None
            assert (r["model_prob"] >= r["ask_cents"] + 10) or (r["model_prob"] <= r["bid_cents"] - 10)
            assert abs(r["edge"]) >= E.LEDGER_EDGE_THRESHOLD  # the price sits inside the book
    assert p["summary"]["eligible_over_threshold"] == len(flagged)
    # a big |edge| on a live book that the model sits INSIDE is not a disagreement
    inside = [r for r in p["markets"] if r["edge"] is not None and abs(r["edge"]) >= 10
              and r["ledger_eligible"] and not r["disagreement"]]
    assert inside, "fixture should contain a wide live book the model sits inside (or within 10 of)"
    assert {c["ticker"] for c in p["ledger_dry_run"]} == \
        {r["ticker"] for r in flagged if not r["match_ambiguous"]}


# ── End-to-end payload (no network) ──────────────────────────────────────────
def _with_cache(rows, prospects, ledger_path, **kwargs):
    """One edge_payload() call against fixture markets and a scratch ledger
    path. kwargs go to edge_payload — with none this is exactly what the
    GET /api/edge handler does."""
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
        # a plain GET-style call is read-only and reports what it WOULD record
        assert set(p) == {"generated_at", "markets", "note", "discovery", "summary", "ledger_dry_run"}
        assert not os.path.exists(ledger)
        assert p["note"] != "seasonal" and p["note"] != "warming"
        assert p["discovery"]["state"] == "ok"
        row_keys = {"ticker", "title", "event_ticker", "series", "yes_price_cents",
                    "yes_price_source", "spread_cents", "ledger_eligible", "matched_player",
                    "matched_team", "match_ambiguous", "model_prob", "edge", "model_basis", "url"}
        # futures spec api_contract: ADDITIVE fields only — the legacy keys
        # stay, exactly these are added.
        added_keys = {"question", "draft_year", "volume", "close_time", "model_interval",
                      "event_scale", "wide_interval", "matched_espn_id", "matched_espn_team_id",
                      "matched_slug", "call",
                      # rank-calibration wave: the live book and the model's verdict on it
                      "bid_cents", "ask_cents", "disagreement", "model_rank"}
        raw = {m["ticker"]: m for _, m in _raw_markets()}
        for r in p["markets"]:
            assert set(r) == row_keys | added_keys, set(r) ^ (row_keys | added_keys)
            assert r["model_basis"]  # every row explains itself
            assert r["bid_cents"] == E._price_field_cents(raw[r["ticker"]], "yes_bid")
            assert r["ask_cents"] == E._price_field_cents(raw[r["ticker"]], "yes_ask")
            assert isinstance(r["disagreement"], bool)
            if r["model_prob"] is not None:
                assert isinstance(r["model_rank"], int) and r["model_rank"] >= 1
                assert f"class-rank-{r['model_rank']} player" in r["model_basis"]
            else:
                assert r["model_rank"] is None and r["disagreement"] is False
        s = p["summary"]
        assert s["markets"] == 77 and s["priced"] == 77
        assert s["matched"] == 70, s  # 65 exact + 5 recovered by the tiebreak
        # Quincy Rhodes Jr.: off the board (pick 538) with no games recorded yet,
        # but a real ESPN-listed Power-conference player — the identity guard
        # keeps him (a missing games_played is not "no stats"; roster_hint is)
        for tick in ("KXNFLDRAFTTOP-27-5-QRHO", "KXNFLDRAFTPICK-27-1-QRHO"):
            assert _find(p["markets"], tick)["matched_team"] == "Arkansas Razorbacks"
        assert _find(p["markets"], "KXNFLDRAFTTOP-27-5-QRHO")["model_rank"] == 538
        # 19 top-5 markets, minus 1 not in the cache (RWIL), minus 1 matched to
        # a player of ANOTHER draft class (Mestemaker, 2028): the fixture cache
        # is too small to rank him in the 2027 pool, and his own-class
        # projected_pick is not a 2027 rank, so he is matched but not priced.
        # #1-pick rows are listed only.
        assert s["modeled"] == 17, s
        dmes = _find(p["markets"], "KXNFLDRAFTTOP-27-5-DMES")
        assert dmes["matched_player"] == "Drew Mestemaker" and dmes["model_prob"] is None
        assert dmes["model_rank"] is None and dmes["disagreement"] is False
        assert dmes["model_basis"].startswith("not priced")
        assert s["edges_over_threshold"] >= 5
        assert {"eligible_over_threshold", "on_ledger", "by_kind", "draft_year",
                "top_gap_ticker"} <= set(s)
        assert 0 < s["eligible_over_threshold"] < s["edges_over_threshold"]
        assert s["on_ledger"] == 0 and all(r["call"] is None for r in p["markets"])
        heisman = [r for r in p["markets"] if r["series"] == "KXHEISMAN"]
        assert heisman and all(r["edge"] is None for r in heisman)
        pick1 = [r for r in p["markets"] if r["series"] == "KXNFLDRAFTPICK"]
        assert len(pick1) == 20 and all(r["model_prob"] is None for r in pick1)
        assert all(r["model_basis"] == E._NOT_PRICED_EXACT for r in pick1 if r["matched_player"])
        smith = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
        assert smith["matched_team"] == "Ohio State Buckeyes" and not smith["match_ambiguous"]
        # the fixture cache is far smaller than a draft class, so the rank is
        # the row's projected_pick (see test_market_rank_uses_the_markets_draft_pool)
        assert smith["model_rank"] == 1 and _find(p["markets"], "KXNFLDRAFTTOP-27-5-AMAN")["model_rank"] == 3
        # sorted: edges first by |edge| desc
        edges = [abs(r["edge"]) for r in p["markets"] if r["edge"] is not None]
        assert edges == sorted(edges, reverse=True)
        # ledger: only record=True writes — one row per disagreeing ticker
        rec = _with_cache(_rows(), _prospects(), ledger, record=True)
        assert "ledger_dry_run" not in rec
        with open(ledger) as f:
            entries = json.load(f)["entries"]
        assert entries and len(entries) == len({e["ticker"] for e in entries})
        assert rec["ledger_recorded"] == len(entries) == s["eligible_over_threshold"]
        assert [e["ticker"] for e in entries] == [c["ticker"] for c in p["ledger_dry_run"]]
        assert all(e["method"] == E.LEDGER_METHOD == "rank-calibration-v2" for e in entries)
        assert all(e["price_source"] in ("last", "mid", "mid_wide") for e in entries)
        assert all(abs(e["edge"]) >= E.LEDGER_EDGE_THRESHOLD for e in entries)
        assert all(e["model_basis"].startswith("model P(pick <= 5) for its class-rank-") for e in entries)
        by_ticker = {r["ticker"]: r for r in p["markets"]}
        for e in entries:
            r = by_ticker[e["ticker"]]
            assert r["ledger_eligible"] and r["disagreement"] and not r["match_ambiguous"]
            # the frozen row carries the book it disagreed with and the rank it used
            assert (e["bid_cents"], e["ask_cents"], e["model_rank"]) == \
                (r["bid_cents"], r["ask_cents"], r["model_rank"])
            assert e["bid_cents"] >= E.LEDGER_MIN_BID_CENTS and e["ask_cents"] < 100
            assert (e["model_prob"] >= e["ask_cents"] + 10) or (e["model_prob"] <= e["bid_cents"] - 10)
            assert e["model_prob"] == r["model_prob"] and e["market_price_cents"] == r["yes_price_cents"]
        # every disagreement on the board was recorded, nothing else
        assert {e["ticker"] for e in entries} == {r["ticker"] for r in p["markets"] if r["disagreement"]}
        qualifying_one_sided = [r for r in p["markets"] if r["edge"] is not None
                                and abs(r["edge"]) >= 10 and r["yes_price_source"] == "ask"]
        assert qualifying_one_sided, "fixture should contain a one-sided qualifying quote"
        assert not any(e["ticker"] == qualifying_one_sided[0]["ticker"] for e in entries)
        # JSEA / CCOL: last 0, bid 1c, ask 54-65c, zero volume -> a placeholder
        # book. Its midpoint stays on the board tagged mid_wide, never ledgered.
        for tick in ("KXNFLDRAFTTOP-27-5-JSEA", "KXNFLDRAFTTOP-27-5-CCOL"):
            r = by_ticker[tick]
            assert r["yes_price_source"] == "mid_wide" and r["ledger_eligible"] is False, r
            assert r["bid_cents"] < E.LEDGER_MIN_BID_CENTS
            assert r["spread_cents"] > E.LEDGER_MAX_SPREAD_CENTS
            assert r["edge"] is not None and abs(r["edge"]) >= E.LEDGER_EDGE_THRESHOLD, r
            assert not r["disagreement"]
            assert not any(e["ticker"] == tick for e in entries), tick
        # DMEN: last 51c inside a live 7/82 book with real volume — the old
        # "traded volume" rule ledgered this 50-point gap; the model (under 1%)
        # is not 10 points below the 7c bid, so it agrees with someone quoting.
        dmen = by_ticker["KXNFLDRAFTTOP-27-5-DMEN"]
        assert (dmen["bid_cents"], dmen["ask_cents"], dmen["yes_price_source"]) == (7, 82, "last")
        assert dmen["ledger_eligible"] and abs(dmen["edge"]) >= 40 and not dmen["disagreement"]
        assert not any(e["ticker"] == dmen["ticker"] for e in entries)
        # a wide live book (mid_wide) IS eligible now; it is just not a
        # disagreement unless the model is outside it
        wide_live = [r for r in p["markets"] if r["yes_price_source"] == "mid_wide" and r["ledger_eligible"]]
        assert wide_live and all(r["bid_cents"] >= E.LEDGER_MIN_BID_CENTS for r in wide_live)
        # re-running (same day or any later one) adds nothing, and GETs never do
        again = _with_cache(_rows(), _prospects(), ledger, record=True)
        assert again["ledger_recorded"] == 0
        assert again["summary"]["on_ledger"] == len(entries)
        _with_cache(_rows(), _prospects(), ledger)
        with open(ledger) as f:
            assert json.load(f)["entries"] == entries


def test_event_coherence_top_n_mass_sums_to_at_most_n():
    """Within KXNFLDRAFTTOP-27-5 at most five players can go top-5, so the
    matched players' P(pick <= 5) must sum to <= 5. When the raw
    rank-calibrated marginals overshoot, the payload scales them; when they
    do not (the fixture's real ranks), they are left alone."""
    rows = _rows()

    def marginals(prospects):
        index = E._build_prospect_index(prospects)
        matched = []
        for src in rows:
            if src["series"] != "KXNFLDRAFTTOP":
                continue
            player, _ = E._resolve_player(src["_market_raw_names"], index,
                                          draft_year=src["draft_year"], team_hint=src["_team_hint"],
                                          require_prospect=True)  # as _build_board does for top-N
            if player:
                matched.append((src, player))
        assert len(matched) == 18
        # priced = matched players of the market's own class (the fixture cache
        # is too small for a pool rank); the inputs are the calibrated curve at
        # each player's rank, nothing else
        priced = []
        for src, pl in matched:
            x = E._raw_p_le(src, pl)
            if pl["draft_class"] != 2027:
                assert x is None, pl["name"]
                continue
            assert abs(x - min(max(_curve_p(5, pl["projected_pick"]), 0.005), 0.95)) < 1e-9, pl["name"]
            priced.append((src, pl, x))
        assert len(priced) == 17
        return matched, priced

    # the real fixture ranks: the marginals already sum below 5 → untouched
    matched, priced = marginals(_prospects())
    raw_sum = sum(x for _, _, x in priced)
    assert 4.0 < raw_sum < 5.0, raw_sum
    assert E._event_scale_factors(matched) == {}
    with tempfile.TemporaryDirectory() as tmp:
        p = _with_cache(rows, _prospects(), os.path.join(tmp, "ledger.json"), dry_run=True)
    top5 = [r for r in p["markets"] if r["event_ticker"] == "KXNFLDRAFTTOP-27-5"
            and r["model_prob"] is not None]
    assert len(top5) == 17
    assert all(r["event_scale"] == 1.0 and "scaled x" not in r["model_basis"] for r in top5)
    for src, pl, x in priced:
        assert _find(top5, src["ticker"])["model_prob"] == round(100.0 * x, 1)
    assert abs(sum(r["model_prob"] for r in top5) / 100.0 - raw_sum) < 0.01

    # a crowded top of the board: eight players at rank 1 → the raw marginals
    # overshoot 5 and every one is scaled by the same 5 / sum
    crowded = _crowded_prospects()
    matched, priced = marginals(crowded)
    raw_sum = sum(x for _, _, x in priced)
    assert sum(1 for _, pl, _ in priced if pl["projected_pick"] == 1) == 8 and raw_sum > 7, raw_sum
    scales = E._event_scale_factors(matched)
    assert set(scales) == {("KXNFLDRAFTTOP-27-5", 5)}
    assert abs(scales[("KXNFLDRAFTTOP-27-5", 5)] - 5 / raw_sum) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        p = _with_cache(rows, crowded, os.path.join(tmp, "ledger.json"), dry_run=True)
    top5 = [r for r in p["markets"] if r["event_ticker"] == "KXNFLDRAFTTOP-27-5"
            and r["model_prob"] is not None]
    assert len(top5) == 17
    total = sum(r["model_prob"] for r in top5) / 100.0
    assert total <= 5.0 + 0.01, total  # rounding slack: 17 rows at 0.05 pt each
    assert total > 4.9, total  # scaled to N, not crushed
    assert all("scaled x" in r["model_basis"] for r in top5)
    want_scale = round(5 / raw_sum, 4)
    assert all(r["event_scale"] == want_scale for r in top5)
    for src, pl, x in priced:  # each row is its raw marginal times the one factor
        assert _find(top5, src["ticker"])["model_prob"] == round(100.0 * x * (5 / raw_sum), 1)
    # scaling preserves the model's ordering: probability falls as rank rises
    by_rank = sorted(top5, key=lambda r: r["model_rank"])
    probs = [r["model_prob"] for r in by_rank]
    assert probs == sorted(probs, reverse=True)
    # the disagreement test runs on the SCALED number
    for r in top5:
        assert r["disagreement"] is E._is_disagreement(r)
    # #1-pick rows in the same fixture: none priced, so no sum to check (n/a)
    assert all(r["model_prob"] is None for r in p["markets"]
               if r["event_ticker"] == "KXNFLDRAFTPICK-27-1")
    # an event whose marginals already sum below N is left untouched
    under = [(dict(rows[0], event_ticker="EV-X", ticker="EV-X-A",
                   _question={"kind": "top_n", "n": 5}),
              {"name": "A", "team": "T", "projected_pick": 1}),
             (dict(rows[0], event_ticker="EV-X", ticker="EV-X-B",
                   _question={"kind": "after_n", "n": 5}),
              {"name": "B", "team": "T", "projected_pick": 2})]
    assert all(E._raw_p_le(s, pl) is not None for s, pl in under)
    assert E._event_scale_factors(under) == {}
    # over/under rows count their P(pick <= n) toward the same event sum
    over = [(dict(u[0], _question={"kind": "after_n", "n": 1}), u[1]) for u in under] * 3
    sc = E._event_scale_factors([(dict(s, ticker=f"t{i}"), dict(pl, name=f"P{i}"))
                                 for i, (s, pl) in enumerate(over)])
    assert ("EV-X", 1) in sc and 0 < sc[("EV-X", 1)] < 1
    # six players at ranks 1,2,1,2,1,2 on a top-1 question: 3*(p1 + p2) > 1
    assert abs(sc[("EV-X", 1)] - 1 / (3 * (_curve_p(1, 1) + _curve_p(1, 2)))) < 1e-12
    # a player without a rank, or a threshold without a curve, adds no mass
    assert E._event_scale_factors([(under[0][0], {"name": "Z", "team": "T"})]) == {}
    no_curve = [(dict(under[0][0], ticker=f"n{i}", _question={"kind": "top_n", "n": 2}),
                 {"name": f"Q{i}", "team": "T", "projected_pick": 1}) for i in range(9)]
    assert E._event_scale_factors(no_curve) == {}


def test_probe_dry_run_never_writes_the_ledger():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        p = _with_cache(_rows(), _prospects(), ledger, dry_run=True)
        assert p["ledger_dry_run"], "fixture should yield would-be ledger rows"
        assert "ledger_recorded" not in p
        assert not os.path.exists(ledger)
        assert all(_find(p["markets"], e["ticker"])["disagreement"] for e in p["ledger_dry_run"])
        # a plain edge_payload(prospects) — the GET /api/edge call — is the
        # same read-only pass: the default is dry_run=True, record=False
        g = _with_cache(_rows(), _prospects(), ledger)
        assert g["ledger_dry_run"] == p["ledger_dry_run"] and not os.path.exists(ledger)
        # dry_run=False alone does NOT write either (it only drops the preview)
        q = _with_cache(_rows(), _prospects(), ledger, dry_run=False)
        assert "ledger_dry_run" not in q and "ledger_recorded" not in q
        assert not os.path.exists(ledger) and os.listdir(tmp) == []
        # the operator CLI: --probe runs one (stubbed) discovery pass and prints;
        # it must leave no ledger and no history file behind
        hist = os.path.join(tmp, "history.json")
        fixture_cache = os.path.join(FIXTURES, "prospects_2027_subset.json")
        rows = _rows()
        fetch = E._fetch_relevant_markets
        E._fetch_relevant_markets = lambda: (rows, {"TEST": len(rows)})
        try:
            with _board_ctx(None, ledger, history_path=hist):
                assert E._cli(["--probe", "--cache", fixture_cache]) == 0
                assert E._kalshi_cache["markets"] is rows  # the pass really ran
                assert E._LEDGER["entries"] == []
        finally:
            E._fetch_relevant_markets = fetch
        assert not os.path.exists(ledger) and not os.path.exists(hist)
        assert os.listdir(tmp) == []
        # only record=True writes — exactly the rows the dry run previewed
        r = _with_cache(_rows(), _prospects(), ledger, record=True)
        assert r["ledger_recorded"] == len(p["ledger_dry_run"])
        with open(ledger) as f:
            written = {e["ticker"] for e in json.load(f)["entries"]}
        assert written == {e["ticker"] for e in p["ledger_dry_run"]}
        # ... and --probe / GETs against an EXISTING ledger leave it byte-identical
        with open(ledger, "rb") as f:
            before = f.read()
        mtime = os.path.getmtime(ledger)
        E._fetch_relevant_markets = lambda: (rows, {"TEST": len(rows)})
        try:
            with _board_ctx(None, ledger, history_path=hist):
                assert E._cli(["--probe", "--cache", fixture_cache]) == 0
                E.edge_payload(_prospects())
                E.spotlight_payload(_prospects())
                E.player_payload(_prospects(), name="Arch Manning", team="Texas Longhorns")
                E.ledger_payload()
        finally:
            E._fetch_relevant_markets = fetch
        with open(ledger, "rb") as f:
            assert f.read() == before
        assert os.path.getmtime(ledger) == mtime and not os.path.exists(hist)


def test_get_never_writes_and_record_writes_once_per_ticker():
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(rows, ledger):
            # any number of page loads: nothing on disk, nothing in memory
            for _ in range(3):
                g = E.edge_payload(prospects)
                assert "ledger_recorded" not in g and g["summary"]["on_ledger"] == 0
                assert all(r["call"] is None for r in g["markets"])
            assert not os.path.exists(ledger) and E._LEDGER["entries"] == []
            would = g["ledger_dry_run"]
            assert len(would) == g["summary"]["eligible_over_threshold"] > 0
            # the weekly job: one write, one row per disagreeing ticker
            r1 = E.edge_payload(prospects, record=True)
            assert r1["ledger_recorded"] == len(would) and "ledger_dry_run" not in r1
            with open(ledger) as f:
                first = json.load(f)["entries"]
            assert sorted(e["ticker"] for e in first) == sorted(c["ticker"] for c in would)
            assert len({e["ticker"] for e in first}) == len(first)
            # record-before-read: the recording response already links the calls
            assert r1["summary"]["on_ledger"] == len(first)
            assert all(_find(r1["markets"], e["ticker"])["call"] for e in first)
            # a second record adds 0 rows and does not rewrite the file
            mtime = os.path.getmtime(ledger)
            r2 = E.edge_payload(prospects, record=True)
            assert r2["ledger_recorded"] == 0 and os.path.getmtime(ledger) == mtime
            with open(ledger) as f:
                assert json.load(f)["entries"] == first
            # record wins over dry_run when both are passed
            r3 = E.edge_payload(prospects, dry_run=True, record=True)
            assert r3["ledger_recorded"] == 0 and "ledger_dry_run" not in r3
            # GETs after the record still write nothing
            E.edge_payload(prospects)
            assert os.path.getmtime(ledger) == mtime
        # a LATER week: the same disagreements on a different date add nothing —
        # the key is (ticker, method), not (ticker, date)
        aged = [dict(e, date="2026-09-10") for e in first]
        with open(ledger, "w") as f:
            json.dump({"entries": aged}, f)
        with _board_ctx(rows, ledger):
            assert time.strftime("%Y-%m-%d", time.gmtime()) != "2026-09-10"
            assert E.edge_payload(prospects, record=True)["ledger_recorded"] == 0
            assert [e["date"] for e in E._LEDGER["entries"]] == ["2026-09-10"] * len(first)
            # the frozen call is the first week's, even if the price has moved since
            moved = [dict(r) for r in rows]
            t = first[0]["ticker"]
            _find(moved, t)["yes_price_cents"] = first[0]["market_price_cents"] + 1
        with _board_ctx(moved, ledger):
            assert E.edge_payload(prospects, record=True)["ledger_recorded"] == 0
            call = _find(E.edge_payload(prospects)["markets"], t)["call"]
            assert call["first_date"] == "2026-09-10"
            assert call["price_at_call"] == first[0]["market_price_cents"]
        # a ticker with only an interval-v1 (pre-calibration) row gets exactly
        # one current-method row beside it, once
        legacy = [{k: v for k, v in dict(e, model_prob=40.0).items() if k != "method"} for e in aged]
        with open(ledger, "w") as f:
            json.dump({"entries": legacy}, f)
        with _board_ctx(rows, ledger):
            assert E.edge_payload(prospects, record=True)["ledger_recorded"] == len(legacy)
            assert E.edge_payload(prospects, record=True)["ledger_recorded"] == 0
            assert len(E._LEDGER["entries"]) == 2 * len(legacy)
            methods = sorted(E._method_of(e) for e in E._LEDGER["entries"] if e["ticker"] == t)
            assert methods == ["interval-v1", "rank-calibration-v2"]
            # the board's call{} is the current method's row, not the older legacy one
            call = _find(E.edge_payload(prospects)["markets"], t)["call"]
            assert call["first_date"] == time.strftime("%Y-%m-%d", time.gmtime())
            assert call["model_at_call"] != 40.0
            # on_ledger counts tickers, not rows
            assert E.edge_payload(prospects)["summary"]["on_ledger"] == len(legacy)
    # ambiguous matches are never recorded even when they disagree
    amb_prospects = [dict(r) for r in prospects]
    for r in amb_prospects:  # make the Louisiana Tech namesake a live top-300 prospect
        if r["name"] == "Jeremiah Smith" and r["team"].startswith("Louisiana"):
            r["projected_pick"] = 250
    cheap = [dict(r) for r in rows]
    jsmi = _find(cheap, "KXNFLDRAFTTOP-27-5-JSMI")
    jsmi.update(yes_price_cents=22, yes_price_source="last", bid_cents=20, ask_cents=24,
                spread_cents=4, ledger_eligible=True)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(cheap, ledger):
            p = E.edge_payload(amb_prospects, record=True)
            row = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
            assert row["match_ambiguous"] is True and row["disagreement"] is True
            assert p["ledger_recorded"] > 0 and row["call"] is None
            assert "KXNFLDRAFTTOP-27-5-JSMI" not in {e["ticker"] for e in E._LEDGER["entries"]}
            assert p["summary"]["top_gap_ticker"] != "KXNFLDRAFTTOP-27-5-JSMI"
        # the same book with a decisive match IS recorded
        with _board_ctx(cheap, os.path.join(tmp, "ledger2.json")):
            p = E.edge_payload(prospects, record=True)
            assert "KXNFLDRAFTTOP-27-5-JSMI" in {e["ticker"] for e in E._LEDGER["entries"]}
            assert p["summary"]["top_gap_ticker"] == "KXNFLDRAFTTOP-27-5-JSMI"


def test_ledger_dedupe_is_per_ticker_per_method():
    def row(ticker, date, **extra):
        return dict({"ticker": ticker, "date": date, "model_prob": 50.0,
                     "market_price_cents": 30, "espn_id": "1"}, **extra)

    assert E.LEDGER_METHOD == "rank-calibration-v2" and E.LEDGER_METHOD_LEGACY == "interval-v1"
    assert E._method_of({}) == "interval-v1" and E._method_of({"method": None}) == "interval-v1"
    assert E._method_of({"method": ""}) == "interval-v1"
    assert E._method_of({"method": "rank-calibration-v2"}) == "rank-calibration-v2"
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(_rows(), ledger):
            new = E.LEDGER_METHOD
            assert E._record_ledger_entries([]) == 0 and not os.path.exists(ledger)
            assert E._record_ledger_entries([row("T-1", "2026-11-01", method=new),
                                             row("T-2", "2026-11-01", method=new)]) == 2
            # same ticker + method on a later date: refused (the old key let it in)
            assert E._record_ledger_entries([row("T-1", "2026-11-08", method=new)]) == 0
            # duplicates inside one batch collapse; junk rows are skipped
            assert E._record_ledger_entries([row("T-3", "2026-11-08", method=new),
                                             row("T-3", "2026-11-08", method=new),
                                             {"ticker": "T-4"}, {"date": "2026-11-08"}, "junk"]) == 1
            # a different method for the same ticker is its own call
            assert E._record_ledger_entries([row("T-1", "2026-09-10")]) == 1          # interval-v1
            assert E._record_ledger_entries([row("T-1", "2026-09-17")]) == 0
            with open(ledger) as f:
                on_disk = json.load(f)["entries"]
            assert sorted((e["ticker"], E._method_of(e), e["date"]) for e in on_disk) == [
                ("T-1", "interval-v1", "2026-09-10"), ("T-1", new, "2026-11-01"),
                ("T-2", new, "2026-11-01"), ("T-3", new, "2026-11-08")]
            # _ledger_calls: the CURRENT method's row is the call, even though
            # the legacy row is older
            calls = E._ledger_calls()
            assert calls["T-1"]["first_date"] == "2026-11-01"
            assert set(calls) == {"T-1", "T-2", "T-3"}
        # a ticker with only legacy rows falls back to its first legacy row
        with open(ledger, "w") as f:
            json.dump({"entries": [row("T-9", "2026-09-17", model_prob=41.0),
                                   row("T-9", "2026-09-10", model_prob=42.0)]}, f)
        with _board_ctx(_rows(), ledger):
            E._load_ledger()
            c = E._ledger_calls()["T-9"]
            assert c == {"first_date": "2026-09-10", "price_at_call": 30, "model_at_call": 42.0}
            # a write merges against what is on disk right now
            assert E._record_ledger_entries([row("T-9", "2026-11-11", method=E.LEDGER_METHOD,
                                                 model_prob=77.0)]) == 1
            assert E._ledger_calls()["T-9"]["model_at_call"] == 77.0
            assert len(E._LEDGER["entries"]) == 3
        # a failed write is no longer swallowed: the weekly job must see it
        missing = os.path.join(tmp, "no-such-dir", "ledger.json")
        with _board_ctx(_rows(), missing):
            try:
                E._record_ledger_entries([row("T-1", "2026-11-01", method=E.LEDGER_METHOD)])
                raise AssertionError("a ledger write failure must propagate")
            except OSError:
                pass
            try:
                E.edge_payload(_prospects(), record=True)
                raise AssertionError("record=True must surface a ledger write failure")
            except OSError:
                pass
            # ... while a read-only GET on the same broken path still serves
            assert E.edge_payload(_prospects())["markets"]


def test_record_ledger_cli_records_once_and_skips_without_discovery():
    rows = _rows()
    fixture_cache = os.path.join(FIXTURES, "prospects_2027_subset.json")
    fetch = E._fetch_relevant_markets
    with tempfile.TemporaryDirectory() as tmp:
        ledger, hist = os.path.join(tmp, "ledger.json"), os.path.join(tmp, "history.json")
        try:
            # discovery down: exit 2, nothing written
            def boom():
                raise RuntimeError("upstream down")

            E._fetch_relevant_markets = boom
            with _board_ctx(None, ledger, history_path=hist):
                assert E._cli(["--record-ledger", "--cache", fixture_cache]) == 2
            # an empty (off-season) book: also skipped
            E._fetch_relevant_markets = lambda: ([], {})
            with _board_ctx(None, ledger, history_path=hist):
                assert E._cli(["--record-ledger", "--cache", fixture_cache]) == 2
            assert os.listdir(tmp) == []
            # a good pass: disagreements frozen, price history captured
            E._fetch_relevant_markets = lambda: (rows, {"TEST": len(rows)})
            with _board_ctx(None, ledger, history_path=hist):
                assert E._cli(["--record-ledger", "--cache", fixture_cache]) == 0
                with open(ledger) as f:
                    first = json.load(f)["entries"]
                assert first and all(e["method"] == E.LEDGER_METHOD for e in first)
                assert len({e["ticker"] for e in first}) == len(first)
                with open(hist) as f:
                    assert len(json.load(f)["tickers"]) == sum(r["yes_price_cents"] is not None for r in rows)
                want = E.edge_payload(_prospects())
                assert {e["ticker"] for e in first} == {r["ticker"] for r in want["markets"] if r["disagreement"]}
            # the next weekly run: one call per ticker, so nothing new
            with _board_ctx(None, ledger, history_path=hist):
                mtime = os.path.getmtime(ledger)
                assert E._cli(["--record-ledger", "--cache", fixture_cache]) == 0
                assert os.path.getmtime(ledger) == mtime
                with open(ledger) as f:
                    assert json.load(f)["entries"] == first
        finally:
            E._fetch_relevant_markets = fetch


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
    # KXNFLDRAFTTOP-27-5: 17 priced rows share one event factor, recorded on
    # each — 1.0 here (the marginals sum below 5), below 1 on a crowded board
    top5 = [r for r in p["markets"] if r["event_ticker"] == "KXNFLDRAFTTOP-27-5" and r["model_prob"] is not None]
    assert len(top5) == 17 and {r["event_scale"] for r in top5} == {1.0}
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            crowded = E.edge_payload(_crowded_prospects(), dry_run=True)["markets"]
    ctop5 = [r for r in crowded if r["event_ticker"] == "KXNFLDRAFTTOP-27-5" and r["model_prob"] is not None]
    assert len(ctop5) == 17 and len({r["event_scale"] for r in ctop5}) == 1
    assert 0 < ctop5[0]["event_scale"] < 1.0
    manning = _find(p["markets"], "KXNFLDRAFTTOP-27-5-AMAN")
    assert manning["model_interval"] == {"lo": 1, "hi": 18, "coverage": 0.8}
    assert manning["wide_interval"] is False and manning["volume"] is not None
    assert manning["matched_espn_id"] == "4870906" and manning["matched_espn_team_id"] == "251"
    assert manning["matched_slug"] == "arch-manning-texas-longhorns"
    # summary.top_gap_ticker: largest |edge| among unambiguous DISAGREEMENTS
    # (model >= 10 points outside a live book) only — recomputed here from the
    # row's book, not from the row's own flag
    def outside(r):
        return (r["model_prob"] is not None and r["ledger_eligible"]
                and (r["model_prob"] >= r["ask_cents"] + 10 or r["model_prob"] <= r["bid_cents"] - 10))

    clean = [r for r in p["markets"] if outside(r) and not r["match_ambiguous"]]
    assert clean
    assert p["summary"]["top_gap_ticker"] == max(clean, key=lambda r: abs(r["edge"]))["ticker"]
    assert p["summary"]["eligible_over_threshold"] == sum(1 for r in p["markets"] if outside(r))
    assert p["summary"]["top_gap_ticker"] == "KXNFLDRAFTTOP-27-5-JSAY"  # rank 2 vs a 2/28 book
    # a 50-point |edge| against a last print INSIDE a wide live book (Mensah,
    # 7/82) is not a disagreement and could never be the top gap, even though
    # it is larger than the second real disagreement's gap
    dmen = _find(p["markets"], "KXNFLDRAFTTOP-27-5-DMEN")
    assert dmen["ledger_eligible"] and not outside(dmen) and not dmen["match_ambiguous"]
    assert abs(dmen["edge"]) > min(abs(r["edge"]) for r in clean)
    assert dmen["ticker"] not in {r["ticker"] for r in clean}
    # fewer than the old "eligible and |edge| >= 10" count, and than all big edges
    old_rule = sum(1 for r in p["markets"] if r["edge"] is not None and abs(r["edge"]) >= 10
                   and r["ledger_eligible"])
    assert p["summary"]["eligible_over_threshold"] < old_rule < p["summary"]["edges_over_threshold"]
    assert p["summary"]["on_ledger"] == 0  # read-only: nothing written


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
            # serving the page (any number of GETs) puts nobody on the ledger
            served = E.edge_payload(prospects)
            assert served["summary"]["on_ledger"] == 0 and E._LEDGER["entries"] == []
            assert not os.path.exists(ledger)
            assert E.player_payload(prospects, name="Julian Sayin", team="Ohio State Buckeyes")["on_ledger"] is False
            # ledger linkage: the RECORDING response already carries call{}
            # and on_ledger for the tickers it just recorded (record-before-read)
            first = E.edge_payload(prospects, record=True)
            assert first["ledger_recorded"] == len(E._LEDGER["entries"]) > 0
            assert first["summary"]["on_ledger"] == len(E._LEDGER["entries"])
            recorded = {e["ticker"] for e in E._LEDGER["entries"]}
            assert all(_find(first["markets"], t)["call"] for t in recorded)
            assert all(r["call"] is None for r in first["markets"] if r["ticker"] not in recorded)
            # Sayin: rank 2, a live 2/28 book, the model far above the ask → on the record
            on = E.player_payload(prospects, name="Julian Sayin", team="Ohio State Buckeyes")
            assert on["on_ledger"] is True and on["ledger_since"] == time.strftime("%Y-%m-%d", time.gmtime())
            top5 = on["markets"][0]
            assert top5["question"]["kind"] == "top_n" and top5["disagreement"] is True
            assert (top5["bid_cents"], top5["ask_cents"], top5["model_rank"]) == (2, 28, 2)
            assert top5["call"] == {"first_date": on["ledger_since"], "price_at_call": 28,
                                    "model_at_call": top5["model_prob"]}
            # Manning: the model sits within 10 points of his 66/77 book → listed, not recorded
            off = E.player_payload(prospects, name="Arch Manning", team="Texas Longhorns")
            assert off["state"] == "ok" and off["on_ledger"] is False and off["ledger_since"] is None
            assert off["markets"][0]["disagreement"] is False and off["markets"][0]["call"] is None
            assert (off["markets"][0]["bid_cents"], off["markets"][0]["ask_cents"]) == (66, 77)
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


def test_spotlight_selection_prefers_disagreements_and_excludes_ambiguous():
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        with _board_ctx(rows, ledger):
            sp = E.spotlight_payload(prospects, n=3)
            assert sp["state"] == "ok" and len(sp["rows"]) == 3
            assert sp["headline"] == sp["rows"][0]
            board = E.edge_payload(prospects, dry_run=True)["markets"]
            # primary rows: unambiguous DISAGREEMENTS (model >= 10 points outside
            # a live book), largest gap first — exactly two on this fixture
            # (Sayin far above a 2/28 book, Carr far below a 25/59 book); the
            # third slot is filled from the remaining last/mid rows
            want = sorted((r for r in board if r["model_prob"] is not None and r["disagreement"]
                           and not r["match_ambiguous"]), key=lambda r: -abs(r["edge"]))
            assert [r["ticker"] for r in want] == ["KXNFLDRAFTTOP-27-5-JSAY", "KXNFLDRAFTTOP-27-5-CCAR"]
            primary = sp["rows"][:2]
            assert [r["ticker"] for r in primary] == [r["ticker"] for r in want]
            assert sp["rows"][2]["disagreement"] is False
            by_ticker = {r["ticker"]: r for r in board}
            for r in primary:
                assert r["ledger_eligible"] is True and r["disagreement"] is True
                assert r["price_source"] in ("last", "mid", "mid_wide")
                assert r["model"] >= r["ask"] + 10 or r["model"] <= r["bid"] - 10
                assert r["bid"] <= r["market"] <= r["ask"]
            for r in sp["rows"]:
                assert r["match_ambiguous"] is False
                assert (r["bid"], r["ask"]) == (by_ticker[r["ticker"]]["bid_cents"],
                                                by_ticker[r["ticker"]]["ask_cents"])
                assert r["espn_id"] and r["espn_team_id"] and r["slug"] and r["question"]["label"]
                assert set(r) == {"ticker", "player", "team", "slug", "espn_id", "espn_team_id",
                                  "question", "market", "price_source", "model", "gap",
                                  "wide_interval", "ledger_eligible", "match_ambiguous", "url",
                                  "disagreement", "bid", "ask"}
            # the headline is the summary's top gap
            assert E.edge_payload(prospects, dry_run=True)["summary"]["top_gap_ticker"] == sp["headline"]["ticker"]
            assert sp["headline"]["player"] == "Julian Sayin" and sp["headline"]["team"] == "Ohio State Buckeyes"
            # a last print inside a wide live book with a huge raw |edge| is not a
            # disagreement: it can only FILL, below every real disagreement —
            # with n=2 it is not shown at all
            wide_gap = max((r for r in board if r["edge"] is not None and not r["disagreement"]
                            and r["yes_price_source"] in ("last", "mid") and not r["match_ambiguous"]),
                           key=lambda r: abs(r["edge"]))
            assert abs(wide_gap["edge"]) > abs(want[-1]["edge"])
            assert sp["rows"][2]["ticker"] == wide_gap["ticker"]
            assert [r["ticker"] for r in E.spotlight_payload(prospects, n=2)["rows"]] == \
                [r["ticker"] for r in want]
            gaps = [abs(r["gap"]) for r in primary]
            assert gaps == sorted(gaps, reverse=True)
            # n clamps to 1..6; the fill path (n > disagreement count) never
            # admits mid_wide / ask / bid or ambiguous rows, and every
            # disagreement still ranks ahead of every fill row
            assert len(E.spotlight_payload(prospects, n=0)["rows"]) == 1
            assert len(E.spotlight_payload(prospects, n="x")["rows"]) == 3
            big = E.spotlight_payload(prospects, n=99)["rows"]
            assert len(big) == 6
            wide = E.spotlight_payload(prospects, n=6)
            assert [r["ticker"] for r in wide["rows"][:3]] == [r["ticker"] for r in sp["rows"]]
            fill = wide["rows"][2:]
            assert len(fill) == 4
            assert fill and all(r["disagreement"] is False for r in fill)
            assert all(r["price_source"] in ("last", "mid") and not r["match_ambiguous"] for r in fill)
            fill_gaps = [abs(r["gap"]) for r in fill]
            assert fill_gaps == sorted(fill_gaps, reverse=True)
            assert fill_gaps[0] > gaps[-1]  # a bigger raw gap still ranks BELOW a real disagreement
            s = sp["summary"]
            assert set(s) == {"markets", "modeled", "disagreements", "on_ledger", "draft_year"}
            assert s["markets"] == 77 and s["modeled"] == 17 and s["draft_year"] == 2027
            assert s["disagreements"] == 2 == \
                E.edge_payload(prospects, dry_run=True)["summary"]["eligible_over_threshold"]
            rec = sp["record"]
            assert rec["calls"] == 0 and rec["status"] == "open" and rec["brier"] is None
            assert not os.path.exists(ledger)  # the spotlight never writes either
            # a board where only mid_wide rows carry a big gap and none has a
            # live book → no primary rows, and the fill stays empty
            only_wide = [dict(r) for r in rows]
            for r in only_wide:
                if r["yes_price_source"] in ("last", "mid"):
                    r["yes_price_source"] = "mid_wide"
                r["ledger_eligible"] = False
            with _board_ctx(only_wide, ledger):
                empty = E.spotlight_payload(prospects, n=3)
                assert empty["rows"] == [] and empty["headline"] is None
                assert empty["summary"]["disagreements"] == 0
            # ... but a mid_wide row on a LIVE wide book the model sits outside
            # is a disagreement and may headline (mid_wide can be eligible now)
            live_wide = [dict(r) for r in only_wide]
            lmoo = _find(live_wide, "KXNFLDRAFTTOP-27-5-LMOO")
            assert lmoo["yes_price_source"] == "mid_wide"
            lmoo.update(bid_cents=3, ask_cents=17, spread_cents=14, yes_price_cents=10, ledger_eligible=True)
            with _board_ctx(live_wide, ledger):
                one = E.spotlight_payload(prospects, n=3)
                assert [r["ticker"] for r in one["rows"]] == ["KXNFLDRAFTTOP-27-5-LMOO"]
                h = one["headline"]
                assert h["price_source"] == "mid_wide" and h["disagreement"] is True
                assert (h["bid"], h["ask"], h["market"]) == (3, 17, 10) and h["model"] >= 27
                assert one["summary"]["disagreements"] == 1
            # an ambiguous disagreement is counted in the summary but never shown
            amb = [dict(r) for r in prospects]
            for r in amb:
                if r["name"] == "Julian Sayin":
                    twin = dict(r, team="Somewhere Else", espn_id="1", projected_pick=40)
            amb.append(twin)
            with _board_ctx(rows, ledger):
                sp2 = E.spotlight_payload(amb, n=3)
                jsay = _find(E.edge_payload(amb)["markets"], "KXNFLDRAFTTOP-27-5-JSAY")
                assert jsay["match_ambiguous"] is True and jsay["disagreement"] is True
                assert "KXNFLDRAFTTOP-27-5-JSAY" not in {r["ticker"] for r in sp2["rows"]}
                assert all(not r["match_ambiguous"] for r in sp2["rows"])
                assert sp2["summary"]["disagreements"] == 2
                assert [r["ticker"] for r in sp2["rows"] if r["disagreement"]] == ["KXNFLDRAFTTOP-27-5-CCAR"]
                assert sp2["headline"]["ticker"] == "KXNFLDRAFTTOP-27-5-CCAR"
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
            # a page load records nothing ...
            E.edge_payload(prospects)
            assert os.path.getmtime(ledger) == mtime
            # ... the weekly record writes new entries enriched at record time,
            # tagged with the current method, beside the untouched legacy rows
            added = E.edge_payload(prospects, record=True)["ledger_recorded"]
            with open(ledger) as f:
                after = json.load(f)["entries"]
            fresh = [e for e in after if e.get("method") == E.LEDGER_METHOD]
            assert fresh and len(fresh) == added and len(after) == 3 + added
            assert {e["ticker"]: e for e in after if "method" not in e} == on_disk
            today = time.strftime("%Y-%m-%d", time.gmtime())
            for e in fresh:
                assert e["date"] == today
                assert e["team"] and e["slug"] and e["question"]["kind"] == "top_n"
                assert e["espn_id"] and e["espn_team_id"]
                assert e["model_interval"] and e["event_scale"] and e["event_ticker"]
                assert e["bid_cents"] is not None and e["ask_cents"] is not None and e["model_rank"] >= 1
                assert e["model_basis"].startswith("model P(pick <= 5) for its class-rank-")
                assert e["resolution"]["outcome"] is None
            assert E.backfill_ledger(prospects) == 0
            # legacy interval-v1 rows keep their frozen numbers and still read
            # as the old method
            legacy_rows = [e for e in E._LEDGER["entries"] if "method" not in e]
            assert len(legacy_rows) == 3 and all(E._method_of(e) == "interval-v1" for e in legacy_rows)
            assert next(e for e in legacy_rows if e["ticker"] == "KXNFLDRAFTTOP-27-5-AMAN")["model_prob"] == 42.5
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
    # on the full fixture board every Smith / Stewart row is unambiguous
    rows, prospects = _rows(), _prospects()
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(prospects, dry_run=True)
            seen = set()
            for r in p["markets"]:
                if r["matched_player"] in ("Jeremiah Smith", "Dylan Stewart"):
                    seen.add(r["matched_team"])
                    assert r["match_ambiguous"] is False, r["ticker"]
            assert seen == {"Ohio State Buckeyes", "South Carolina Gamecocks"}
            # Smith's real book (72/89) contains the model's number, so he is
            # not a disagreement and does not headline on this fixture
            smith = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
            assert smith["disagreement"] is False
            assert p["summary"]["top_gap_ticker"] != "KXNFLDRAFTTOP-27-5-JSMI"
        # ... but a decisively tiebroken match MAY headline: give the Smith
        # top-5 market a live 20/24 book the model sits far above, and it is
        # the board's top gap and the spotlight headline
        cheap = [dict(r) for r in rows]
        _find(cheap, "KXNFLDRAFTTOP-27-5-JSMI").update(
            yes_price_cents=22, yes_price_source="last", bid_cents=20, ask_cents=24,
            spread_cents=4, ledger_eligible=True)
        with _board_ctx(cheap, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(prospects, dry_run=True)
            smith = _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")
            assert smith["match_ambiguous"] is False and smith["disagreement"] is True
            assert p["summary"]["top_gap_ticker"] == "KXNFLDRAFTTOP-27-5-JSMI"
            head = E.spotlight_payload(prospects, n=1)["headline"]
            assert head["ticker"] == "KXNFLDRAFTTOP-27-5-JSMI" and head["team"] == "Ohio State Buckeyes"
            assert head["match_ambiguous"] is False and (head["bid"], head["ask"]) == (20, 24)
            # the same book with a NON-decisive collision is excluded from both
            near = [dict(r, projected_pick=250) if (r["name"] == "Jeremiah Smith"
                                                    and r["team"].startswith("Louisiana")) else r
                    for r in prospects]
            p = E.edge_payload(near, dry_run=True)
            assert _find(p["markets"], "KXNFLDRAFTTOP-27-5-JSMI")["match_ambiguous"] is True
            assert p["summary"]["top_gap_ticker"] != "KXNFLDRAFTTOP-27-5-JSMI"
            assert E.spotlight_payload(near, n=6)["headline"]["ticker"] != "KXNFLDRAFTTOP-27-5-JSMI"
            assert "KXNFLDRAFTTOP-27-5-JSMI" not in {c["ticker"] for c in p["ledger_dry_run"]}


# ── Identity guards: unmatched is honest, mismatched is not ──────────────────
def test_identity_guard_refuses_offboard_namesakes_for_draft_slot_markets():
    ok = E._plausible_subject
    stats = {"games_played": 9, "data_source": "espn_live", "conference_tier": 2}
    # on the board (projected pick <= 300): always plausible, stats or not
    assert ok({"projected_pick": 6, "games_played": None, "data_source": "roster_hint", "conference_tier": 9})
    assert ok({"projected_pick": 300, "games_played": None, "conference_tier": 9})
    # off the board with recorded stats at a Power-conference program: kept —
    # the model ranking a real, listed QB outside the draft is a disagreement
    assert ok(dict(stats, projected_pick=966))
    assert ok(dict(stats, projected_pick=301, conference_tier=5))      # tier 5 is still allowed
    assert ok(dict(stats, projected_pick=966, conference_tier=None))   # unknown tier is not held against him
    # a missing games_played alone is NOT "no stats": a real ESPN-listed
    # Power-conference player who has not appeared yet is still the subject
    assert ok(dict(stats, projected_pick=301, games_played=None))
    assert ok(dict(stats, projected_pick=538, games_played=None, conference_tier=3))
    # off the board AND (a roster-hint row with no stats OR a bottom-tier program): not the subject
    assert not ok(dict(stats, projected_pick=301, data_source="roster_hint"))
    assert not ok(dict(stats, projected_pick=1704, data_source="roster_hint"))
    assert not ok(dict(stats, projected_pick=1704, data_source="roster_hint", games_played=None))
    assert not ok(dict(stats, projected_pick=1704, conference_tier=6))
    assert not ok(dict(stats, projected_pick=1704, conference_tier=8, games_played=None,
                       data_source="roster_hint"))
    # no projected pick counts as off the board, so the same two tests decide
    assert ok({"games_played": 9, "conference_tier": 1})
    assert not ok({"projected_pick": None, "data_source": "roster_hint"})
    assert not ok({"games_played": 9, "conference_tier": 7})

    sc = {"name": "Dylan Stewart", "team": "South Carolina Gamecocks", "position": "DE",
          "draft_class": 2028, "projected_pick": 6, "conference_tier": 3, "games_played": 6,
          "data_source": "espn_live", "espn_id": "5079709"}
    del_ = {"name": "Dylan Stewart", "team": "Delaware Blue Hens", "position": "OL",
            "draft_class": 2029, "projected_pick": 1704, "conference_tier": 8, "games_played": None,
            "data_source": "roster_hint", "espn_id": "5296334"}
    mateer = {"name": "John Mateer", "team": "Oklahoma Sooners", "position": "QB",
              "draft_class": 2027, "projected_pick": 966, "conference_tier": 2, "games_played": 1,
              "data_source": "espn_live", "espn_id": "4915980"}
    # the 2026-09-15 incident: the real player drops out of the cache and only
    # the no-stats namesake is left
    only_namesake = E._build_prospect_index([del_, mateer])
    assert E._resolve_player(["Dylan Stewart"], only_namesake, draft_year=2027)[0] is del_   # unguarded
    assert E._resolve_player(["Dylan Stewart"], only_namesake, draft_year=2027,
                             require_prospect=True) == (None, False)
    # an off-board Power-conference player WITH stats is kept
    assert E._resolve_player(["John Mateer"], only_namesake, draft_year=2027,
                             require_prospect=True) == (mateer, False)
    # with both rows present the guard drops the namesake before the tiebreak:
    # a single clean match, even when a team hint points at the namesake
    both = E._build_prospect_index([del_, sc])
    assert E._resolve_player(["Dylan Stewart"], both, draft_year=2027, require_prospect=True) == (sc, False)
    assert E._resolve_player(["Dylan Stewart"], both, draft_year=2027, team_hint="Delaware",
                             require_prospect=True) == (sc, False)
    assert E._resolve_player(["Dylan Stewart"], both, draft_year=2027, team_hint="Delaware")[0] is del_
    # a refused first candidate falls through to the next candidate name
    assert E._resolve_player(["Dylan Stewart", "John Mateer"], only_namesake, draft_year=2027,
                             require_prospect=True) == (mateer, False)
    assert set(E._DRAFT_SLOT_KINDS) == {"top_n", "after_n", "pick_eq", "nth_position"}

    # board level, on the fixture cache minus South Carolina's Dylan Stewart
    rows = _rows()
    prospects = [r for r in _prospects()
                 if not (r["name"] == "Dylan Stewart" and r["team"].startswith("South Carolina"))]
    assert any(r["name"] == "Dylan Stewart" and r["team"] == "Delaware Blue Hens" for r in prospects)
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            p = E.edge_payload(prospects)
            by = {r["ticker"]: r for r in p["markets"]}
            for tick in ("KXNFLDRAFTTOP-27-5-DSTE", "KXNFLDRAFTPICK-27-1-DSTE"):
                r = by[tick]
                assert r["matched_player"] is None and r["matched_team"] is None, r
                assert r["matched_espn_id"] is None and r["model_prob"] is None and r["edge"] is None
                assert r["disagreement"] is False and r["model_rank"] is None
                assert r["model_basis"] == "not priced: player not in the prospect cache"
            assert "Delaware Blue Hens" not in {r["matched_team"] for r in p["markets"]
                                                if r["question"]["kind"] != "heisman"}
            assert not any(c["player"] == "Dylan Stewart" for c in p["ledger_dry_run"])
            # off-board Power-conference QBs with stats stay matched and priced
            # (rank 966 → the floor), on the board as the model's opinion
            mat = by["KXNFLDRAFTTOP-27-5-JMAT"]
            assert mat["matched_player"] == "John Mateer" and mat["matched_team"] == "Oklahoma Sooners"
            assert mat["model_rank"] == 966 and mat["model_prob"] is not None and mat["model_prob"] <= 0.5
            assert by["KXNFLDRAFTPICK-27-1-JMAT"]["matched_player"] == "John Mateer"
            # with the real player back in the cache the same market matches him
            full = E.edge_payload(_prospects())
            dste = _find(full["markets"], "KXNFLDRAFTTOP-27-5-DSTE")
            assert dste["matched_team"] == "South Carolina Gamecocks" and dste["matched_espn_id"] == "5079709"
            assert dste["match_ambiguous"] is False and dste["model_rank"] == 6
    # Quincy Rhodes Jr. on the fixture: pick 538, tier 3, espn_live, no games
    # recorded — plausible, so both of his draft-slot markets match him
    qrho = next(r for r in _prospects() if r["name"] == "Quincy Rhodes Jr.")
    assert qrho["projected_pick"] > 300 and qrho["games_played"] is None
    assert qrho["data_source"] == "espn_live" and ok(qrho)
    assert dste["matched_player"] and _find(full["markets"], "KXNFLDRAFTTOP-27-5-QRHO")["matched_espn_id"] == "4918136"
    assert _find(full["markets"], "KXNFLDRAFTPICK-27-1-QRHO")["matched_player"] == "Quincy Rhodes Jr."
    # the guard applies to draft-slot questions only: a Heisman market naming
    # the off-board roster-hint namesake is still matched (and never priced)
    src = dict(_find(rows, "KXNFLDRAFTTOP-27-5-DSTE"))
    as_heisman = dict(src, ticker="KXHEISMAN-27-DSTE", series="KXHEISMAN", event_ticker="KXHEISMAN-27",
                      _question={"kind": "heisman"})
    board = E._build_board([src, as_heisman], prospects)
    got = {r["ticker"]: (r["matched_player"], r["matched_team"], r["model_prob"]) for r in board["rows"]}
    assert got == {"KXNFLDRAFTTOP-27-5-DSTE": (None, None, None),
                   "KXHEISMAN-27-DSTE": ("Dylan Stewart", "Delaware Blue Hens", None)}


def test_ledger_pin_refuses_a_different_espn_id_for_a_pinned_ticker():
    rows, prospects = _rows(), _prospects()
    tick = "KXNFLDRAFTTOP-27-5-DMOO"
    real = next(r for r in prospects if r["name"] == "Dante Moore")
    assert real["espn_id"] == "4870921" and real["projected_pick"] == 1  # passes the prospect guard on his own

    def pin_row(espn_id, date="2026-09-10", **extra):
        return dict({"ticker": tick, "date": date, "player": "Dante Moore", "espn_id": espn_id,
                     "model_prob": 40.0, "market_price_cents": 60, "edge": -20.0,
                     "price_source": "last", "model_basis": "x", "title": "t"}, **extra)

    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "ledger.json")
        # pinned to somebody else: today's "Dante Moore" is a namesake → refused
        with open(ledger, "w") as f:
            json.dump({"entries": [pin_row("999999")]}, f)
        with _board_ctx(rows, ledger):
            assert E._ledger_pins() == {}  # nothing loaded yet; the board build reloads by mtime
            p = E.edge_payload(prospects)
            assert E._ledger_pins() == {tick: "999999"}
            r = _find(p["markets"], tick)
            assert r["matched_player"] is None and r["matched_espn_id"] is None and r["matched_team"] is None
            assert r["model_prob"] is None and r["edge"] is None and r["disagreement"] is False
            assert r["match_ambiguous"] is False
            assert r["model_basis"].startswith("not priced") and "namesake was refused" in r["model_basis"]
            assert r["call"]  # the frozen call is still shown against the ticker
            # the pin is per TICKER: his other markets (no call on record) still match
            assert _find(p["markets"], "KXNFLDRAFTPICK-27-1-DMOO")["matched_espn_id"] == "4870921"
            # nothing is recorded for the refused row, whatever its book
            assert tick not in {c["ticker"] for c in p["ledger_dry_run"]}
            assert E.edge_payload(prospects, record=True)["ledger_recorded"] == len(p["ledger_dry_run"])
            assert [e for e in E._LEDGER["entries"] if e["ticker"] == tick] == [pin_row("999999")]
            # every other row is unaffected by the pin
            base_ctx_rows = {x["ticker"]: x["matched_espn_id"] for x in p["markets"]}
        with _board_ctx(rows, os.path.join(tmp, "empty.json")):
            unpinned = E.edge_payload(prospects)
            free = _find(unpinned["markets"], tick)
            assert free["matched_espn_id"] == "4870921" and free["model_prob"] is not None
            assert "namesake" not in free["model_basis"]
            diff = {t for t, pid in base_ctx_rows.items()
                    if pid != _find(unpinned["markets"], t)["matched_espn_id"]}
            assert diff == {tick}
        # a pin that lands on disk AFTER the board was memoised (another
        # worker's record / a --sync-ledger) takes effect on the next request
        # for the very same market + prospect lists
        late = os.path.join(tmp, "late.json")
        with _board_ctx(rows, late):
            assert _find(E.edge_payload(prospects)["markets"], tick)["matched_player"] == "Dante Moore"
            memo = E._BOARD_MEMO["board"]
            E.edge_payload(prospects)
            assert E._BOARD_MEMO["board"] is memo  # ledger unchanged → no rebuild
            with open(late, "w") as f:
                json.dump({"entries": [pin_row("999999")]}, f)
            r = _find(E.edge_payload(prospects)["markets"], tick)
            assert E._BOARD_MEMO["board"] is not memo
            assert r["matched_player"] is None and "namesake was refused" in r["model_basis"]
            # ... and so does the pin going away again (mtime forced forward)
            memo = E._BOARD_MEMO["board"]
            with open(late, "w") as f:
                json.dump({"entries": []}, f)
            os.utime(late, (time.time() + 5, time.time() + 5))
            assert _find(E.edge_payload(prospects)["markets"], tick)["matched_player"] == "Dante Moore"
            assert E._BOARD_MEMO["board"] is not memo
        # pinned to the SAME espn_id: matched as usual (ids compare as strings)
        for same in ("4870921", 4870921):
            with open(ledger, "w") as f:
                json.dump({"entries": [pin_row(same)]}, f)
            with _board_ctx(rows, ledger):
                r = _find(E.edge_payload(prospects)["markets"], tick)
                assert r["matched_player"] == "Dante Moore" and r["matched_espn_id"] == "4870921"
                assert r["model_prob"] is not None and "namesake" not in r["model_basis"]
        # the pin is the EARLIEST dated row carrying an espn_id; rows without
        # one (un-backfilled legacy rows) pin nothing
        with open(ledger, "w") as f:
            json.dump({"entries": [pin_row("4870921", date="2026-11-01", method=E.LEDGER_METHOD),
                                   pin_row("999999", date="2026-09-10"),
                                   pin_row(None, date="2026-09-01")]}, f)
        with _board_ctx(rows, ledger):
            E._load_ledger()
            assert E._ledger_pins() == {tick: "999999"}
            assert _find(E.edge_payload(prospects)["markets"], tick)["matched_player"] is None
        with open(ledger, "w") as f:
            json.dump({"entries": [pin_row(None)]}, f)
        with _board_ctx(rows, ledger):
            E._load_ledger()
            assert E._ledger_pins() == {}
            assert _find(E.edge_payload(prospects)["markets"], tick)["matched_player"] == "Dante Moore"
        # a matched cache row with no espn_id cannot be contradicted by the pin
        anon = [dict(r, espn_id=None) if r["name"] == "Dante Moore" else r for r in prospects]
        with open(ledger, "w") as f:
            json.dump({"entries": [pin_row("999999")]}, f)
        with _board_ctx(rows, ledger):
            assert _find(E.edge_payload(anon)["markets"], tick)["matched_player"] == "Dante Moore"


def test_market_rank_uses_the_markets_draft_pool():
    """A full cache: the player's raw model pick is ranked against every row
    of the MARKET's draft year, so an underclassman bucketed a class late is
    not read as "rank 1" off his own class's projected_pick."""
    rows = [r for r in _rows() if r["ticker"] in ("KXNFLDRAFTTOP-27-5-AMAN", "KXNFLDRAFTTOP-27-5-DMOO",
                                                  "KXNFLDRAFTTOP-27-5-DMES", "KXNFLDRAFTPICK-27-1-AMAN")]
    assert len(rows) == 4
    base = {r["name"]: r for r in _prospects()}
    manning = dict(base["Arch Manning"], draft_class=2028, projected_pick=3, model_pick=11.0)
    moore = dict(base["Dante Moore"], draft_class=2027, projected_pick=1, model_pick=12.0)
    mest = dict(base["Drew Mestemaker"], draft_class=2029, projected_pick=1, model_pick=39.9)
    # 120 other 2027-class rows at model picks 5, 10, 15, ... (and noise from other classes)
    pool = [{"name": f"Pool Player {i}", "team": f"Team {i}", "draft_class": 2027,
             "projected_pick": 500 + i, "model_pick": 5.0 * (i + 1)} for i in range(120)]
    noise = [{"name": f"Other Class {i}", "team": f"Team {i}", "draft_class": 2028,
              "projected_pick": i + 1, "model_pick": 1.0 + i} for i in range(50)]
    noise.append({"name": "No Model Pick", "team": "T", "draft_class": 2027, "projected_pick": 2})
    prospects = pool + noise + [manning, moore, mest]
    with tempfile.TemporaryDirectory() as tmp:
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            by = {r["ticker"]: r for r in E.edge_payload(prospects)["markets"]}
            # 2027 pool model picks: 5, 10, [Moore 12.0], 15, 20, ..., 600
            # Manning 11.0 → 2 below him → rank 3;  Moore 12.0 → rank 3 too
            # (ties/near-ties share the insertion point: 5 and 10 are below);
            # Mestemaker 39.9 → 5,10,12,15,...,35 below (8 rows) → rank 9
            assert by["KXNFLDRAFTTOP-27-5-AMAN"]["model_rank"] == 3
            assert by["KXNFLDRAFTTOP-27-5-DMOO"]["model_rank"] == 3
            mes = by["KXNFLDRAFTTOP-27-5-DMES"]
            assert mes["model_rank"] == 9 and "class-rank-9 player" in mes["model_basis"]
            # NOT his own-class projected_pick of 1 (which would be ~95%)
            scale = mes["event_scale"]
            assert mes["model_prob"] == round(100.0 * _curve_p(5, 9) * scale, 1) < 20
            # unpriced rows carry no rank; the scratch key never leaks to the payload
            assert by["KXNFLDRAFTPICK-27-1-AMAN"]["model_rank"] is None
            assert all("_market_rank" not in r for r in by.values())
        # under 100 rows in the market's class → no pool rank. A same-class
        # player falls back to his projected_pick; a player of ANOTHER class is
        # not priced at all (his projected_pick ranks him in a different pool)
        small = pool[:98] + noise + [manning, moore, mest]  # 98 + Moore = 99 rows with a model pick
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            by = {r["ticker"]: r for r in E.edge_payload(small)["markets"]}
            assert by["KXNFLDRAFTTOP-27-5-DMOO"]["model_rank"] == 1          # class 2027: projected_pick
            assert by["KXNFLDRAFTTOP-27-5-DMOO"]["model_prob"] is not None
            for tick, who in (("KXNFLDRAFTTOP-27-5-AMAN", "Arch Manning"),         # class 2028
                              ("KXNFLDRAFTTOP-27-5-DMES", "Drew Mestemaker")):     # class 2029
                r = by[tick]
                assert r["matched_player"] == who, r
                assert r["model_rank"] is None and r["model_prob"] is None and r["edge"] is None
                assert r["disagreement"] is False and r["model_basis"].startswith("not priced")
            assert all("_market_rank" not in r for r in rows)  # stale ranks are cleared from the cached markets
        # exactly 100 rows is enough
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            by = {r["ticker"]: r for r in E.edge_payload(pool[:99] + [manning, moore, mest])["markets"]}
            assert by["KXNFLDRAFTTOP-27-5-DMES"]["model_rank"] == 9
            assert by["KXNFLDRAFTTOP-27-5-AMAN"]["model_rank"] == 3
        # with a full pool but no model_pick there is nothing to rank: again
        # same-class → projected_pick, other class → not priced
        nomp = {k: v for k, v in mest.items() if k != "model_pick"}
        moore_nomp = {k: v for k, v in moore.items() if k != "model_pick"}
        with _board_ctx(rows, os.path.join(tmp, "ledger.json")):
            by = {r["ticker"]: r for r in E.edge_payload(pool + [manning, moore_nomp, nomp])["markets"]}
            assert by["KXNFLDRAFTTOP-27-5-DMES"]["model_rank"] is None
            assert by["KXNFLDRAFTTOP-27-5-DMES"]["model_prob"] is None
            assert by["KXNFLDRAFTTOP-27-5-DMOO"]["model_rank"] == 1
            assert by["KXNFLDRAFTTOP-27-5-AMAN"]["model_rank"] == 3  # 5 and 10 sit below his 11.0
    # _class_rank directly: the pool rank always wins; the projected_pick
    # fallback needs the player's class to be the market's year (or unknown)
    cr = E._class_rank
    assert cr({"projected_pick": 4, "draft_class": 2028}, {"_market_rank": 12, "draft_year": 2027}) == 12
    assert cr({"projected_pick": 4, "draft_class": 2027}, {"draft_year": 2027}) == 4
    assert cr({"projected_pick": 4, "draft_class": "2027"}, {"draft_year": 2027}) == 4
    assert cr({"projected_pick": 4, "draft_class": 2028}, {"draft_year": 2027}) is None
    assert cr({"projected_pick": 4, "draft_class": 2026}, {"draft_year": 2027}) is None
    assert cr({"projected_pick": 4}, {"draft_year": 2027}) == 4                    # class unknown
    assert cr({"projected_pick": 4, "draft_class": 2028}, {}) == 4                 # market year unknown
    assert cr({"projected_pick": 4, "draft_class": 2028}) == 4
    top5 = _find(_rows(), "KXNFLDRAFTTOP-27-5-AMAN")
    other = {"name": "A", "team": "T", "projected_pick": 1, "draft_class": 2028}
    prob, edge, basis = E._model_fields(top5, other)
    assert prob is None and edge is None and basis.startswith("not priced")
    assert E._raw_p_le(top5, other) is None
    assert E._model_fields(top5, dict(other, draft_class=2027))[0] == 95.0


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
