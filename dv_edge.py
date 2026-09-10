"""Kalshi market-edge board — read-only analysis of public prediction-market
prices vs DraftVision model output.

Wired from XGBOost.py via:

    import dv_edge
    dv_edge.edge_payload(_PROSPECT_CACHE)              # -> GET /api/edge
    dv_edge.player_payload(_PROSPECT_CACHE, name=, team=, slug=, espn_id=)
                                                       # -> GET /api/edge/player
    dv_edge.spotlight_payload(_PROSPECT_CACHE, n=3)    # -> GET /api/edge/spotlight
    dv_edge.ledger_payload()                           # -> GET /api/edge/ledger
    dv_edge.history_payload(ticker)                    # -> GET /api/edge/history

Operator CLI (all GET-only):

    .venv/bin/python dv_edge.py --probe                 # discovery + match counts
    .venv/bin/python dv_edge.py --sync-ledger https://draft.jkrek.com
                                                        # pull prod paper ledger
                                                        # into training_data/
    .venv/bin/python dv_edge.py --backfill-ledger       # fill identity/question
                                                        # fields on old entries
                                                        # (idempotent)
    .venv/bin/python dv_edge.py --score-ledger          # after the draft: GET
                                                        # /markets/{ticker} per
                                                        # open call, fill
                                                        # resolution + Brier
                                                        # (idempotent, no-op
                                                        # until markets settle)
    .venv/bin/python dv_edge.py --sync-history https://draft.jkrek.com
                                                        # pull captured price
                                                        # history into
                                                        # training_data/

Request-path cost: the 77-row match/price pass is memoised per (market
list, prospect list) in _BOARD_MEMO and every /api/edge* endpoint reads
those rows; per-player lookups (/api/edge/player) go through an index keyed
by normalised name+team, slug and ESPN id that is rebuilt with the memo.
Price history (training_data/edge_history.json) is captured by the warm
thread after each successful pass — see _capture_history.

Design constraints (deliberate):
- READ-ONLY: we hit only Kalshi's PUBLIC market-data endpoints (no auth, no
  order placement — we never take or facilitate bets). This page is analysis
  of CFTC-regulated market prices, nothing more. There is no API key anywhere
  in this module and no code path that touches /orders or /portfolio.
- Gentle on the API: discovery probes a fixed list of draft series via
  GET /markets?series_ticker=…, paced, short timeouts, bounded pages, and the
  result is held in memory for ~10 minutes — a burst of page loads produces
  at most one upstream pass.
- Off the request thread: discovery runs in a daemon thread started at import
  (and lazily re-armed on first request if the process forked after import).
  Requests never block on Kalshi; they read whatever the thread last stored
  (stale-while-revalidating). Before the first pass completes the payload
  carries note "warming" and the page polls.
- Fork-safe: os.register_at_fork(after_in_child=…) swaps in fresh module
  locks and forgets the parent's thread handle, so a gunicorn/multiprocessing
  child never inherits a lock held mid-refresh or believes a thread that
  only exists in the parent is alive.
- Fail-soft: any upstream failure (network, schema drift, rate limit) keeps
  the last good discovery; a failure before any success degrades to
  {"markets": [], "note": "seasonal"} with HTTP 200. Nothing 500s.

── Honest-mapping policy (read this before "improving" the edge math) ─────────
A cache row (training_data/prospect_cache.json) carries these model outputs:
  * success_probability      calibrated P(NFL success), 0-100
  * draft_grade_class        argmax bucket (0 = "Top 50 Pick", …)
  * projected_pick           board-relative nominal pick (rank within class)
  * model_pick               raw pick-head point estimate
  * pick_range {lo, hi, confidence}
                             conformalized nominal pick interval from the v5
                             quantile heads (target coverage 0.80). Emitted by
                             POST /predict; the cache builder must persist it
                             for markets to be priced (see _model_fields).

Draft-POSITION *threshold* questions are the only ones the model answers:

  * "Drafted in the Top N" (KXNFLDRAFTTOP, first-round Y/N)   -> P(pick <= N)
  * Draft-position over/under (KXNFLDRAFTOU strike fields)      -> P(pick <= N)
                                                                   / P(pick > N)
  * "#K overall pick" (KXNFLDRAFTPICK) is LISTED ONLY. An exact-pick
    probability is not identifiable from a coverage interval — any
    P(pick == K) read off it is purely an artefact of the interior-mass
    assumption, not a model output — so those rows carry model_basis
    "not priced: exact-pick probabilities are not identifiable from an
    interval" and are never ledgered.

P(pick <= N) is derived from the player's conformal interval [lo, hi] by
assuming the interval's nominal mass (its `confidence`, 0.80) is spread
LOG-UNIFORMLY between lo and hi, with the remaining mass split equally into
the two tails (also log-uniform, down to pick 1 and up to the undrafted
ceiling). This is a *model-implied* number, not a persisted posterior — the
pick head was trained on a point target and the interval is a coverage
guarantee, not a distribution. Every priced row therefore carries a
`model_basis` string that says exactly this, and the UI labels the column
"model-implied". The older shortcut (P(top-5) ~ share of the nominal
interval below 5, or success_probability as a bucket proxy) is NOT used.

Event coherence: the per-player numbers are marginals from independent
intervals and nothing forces them to be jointly consistent, but within one
event (e.g. KXNFLDRAFTTOP-27-5) at most N players can go in the top N, so
the matched players' P(pick <= N) must sum to <= N. When the raw marginals
sum past N every one of them is scaled by N/sum (over/under rows in the same
event scale their P(pick <= N) the same way) and the factor is recorded in
model_basis. Sums below N are left alone — unmatched players hold the rest.

Questions the cache cannot answer stay listed with null model fields:
  * Heisman (KXHEISMAN) — an award, not a draft slot.
  * "Nth QB/WR/… drafted" (KXNFLDRAFT{QB,…}) — needs a joint positional
    order distribution we do not persist.
  * Team markets (KXNFLDRAFT1ST "which team picks 1st") — not probed at all.
  * Any player whose cache row lacks pick_range.

── Paper ledger ───────────────────────────────────────────────────────────────
When a matched market shows |edge| >= LEDGER_EDGE_THRESHOLD points on a REAL
two-sided quote we append one row per (ticker, day) to
training_data/edge_ledger.json. "Real" means: the price is a last trade or a
bid/ask midpoint (never a one-sided ask) AND the book is tight (bid >=
LEDGER_MIN_BID_CENTS and ask - bid <= LEDGER_MAX_SPREAD_CENTS) or the market
has traded volume. A 1c-bid / 65c-ask book on zero volume is a placeholder,
not an opinion: its midpoint stays on the board tagged `mid_wide` ("mid
(wide)" in the UI) and is excluded from the ledger. No
money moves — the ledger exists purely to build a verifiable public track
record of the model's calls BEFORE anyone is asked to trust an edge. Atomic
write (temp + os.replace) and mtime hot-reload, same pattern as the other
training_data caches.

Durability: the Fly container has no volume, so the ledger a production
worker appends to dies on the next `fly deploy`, and the copy in git is what
comes back. To keep the track record the weekly refresh job should, before
its commit step, run

    python dv_edge.py --sync-ledger https://draft.jkrek.com

(a single GET of /api/edge/ledger, merged into training_data/edge_ledger.json
deduped on (ticker, date)) and add `training_data/edge_ledger.json` to the
`git add` list in .github/workflows/refresh-board.yml. The file already
lives under training_data/ and is tracked, so the commit step needs no other
change. This module deliberately does not edit the workflow itself.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime, timezone

import requests

# ── Config ────────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
LEDGER_PATH = "training_data/edge_ledger.json"
HISTORY_PATH = "training_data/edge_history.json"

# Price-history capture (GET /api/edge/history): a point is appended on the
# warm-thread refresh when the UTC date changed or the price moved >= 3c;
# each ticker keeps at most 400 points and is pruned 30 days after it was
# last seen (or 30 days after its close_time).
HISTORY_MAX_POINTS = 400
HISTORY_MIN_MOVE_CENTS = 3
HISTORY_PRUNE_DAYS = 30

LEDGER_SCORING_NOTE = "scored on draft night — Brier vs the real 2027 draft order"

CACHE_TTL_S = 600            # background refresh cadence (~10 min)
REQUEST_TIMEOUT_S = 8        # per-request HTTP timeout
MAX_SERIES_PAGES = 3         # hard bound on pagination per series
PAGE_LIMIT = 200             # Kalshi max page size
INTER_REQUEST_DELAY_S = 0.6  # Kalshi 429s rapid-fire pagination; pace ourselves

# Set DV_EDGE_NO_WARMUP=1 to suppress the background discovery thread (tests,
# one-off scripts). The request path then serves whatever is in the cache.
_NO_WARMUP_ENV = "DV_EDGE_NO_WARMUP"

# Main-draft series families verified live on 2026-09-09 (see
# models/experiments/kalshi_audit.md). Probed directly — the /events walk
# never reached them (draft events sit ~2,400 events deep, past the page cap).
#   KXNFLDRAFTTOP   "Will <player> be a top N draft pick in <year>?"
#   KXNFLDRAFTPICK  "Who will be picked Kth in the Pro Football Draft?"
#   KXNFLDRAFTOU    draft-position over/under (2026 event exists; 2027 TBD)
#   KXNFLDRAFT<POS> "Nth <position> drafted" (listed, not priced)
#   KXHEISMAN       Heisman winner (listed, not priced)
# Deliberately NOT probed: KXNFLSDRAFTTOP (supplemental draft),
# KXNFLDRAFT1ST (team-to-pick-first — no players), KXNCAAFTEAMRECTD (team
# receiving-TD game props — no players, was filling the board with noise).
_KNOWN_SERIES = (
    "KXNFLDRAFTTOP",
    "KXNFLDRAFTPICK",
    "KXNFLDRAFTOU",
    "KXNFLDRAFTQB",
    "KXNFLDRAFTRB",
    "KXNFLDRAFTWR",
    "KXNFLDRAFTTE",
    "KXNFLDRAFTOL",
    "KXNFLDRAFTEDGE",
    "KXNFLDRAFTDT",
    "KXNFLDRAFTLB",
    "KXNFLDRAFTDB",
    "KXHEISMAN",
)

# Relevance guard applied to every probed market (defensive: a family that
# grows a team-market event should not leak onto a player board).
_SERIES_PREFIXES = ("KXNFLDRAFT", "KXHEISMAN")
_EXCLUDED_SERIES_PREFIXES = ("KXNFLDRAFT1ST", "KXNFLSDRAFT")
_TITLE_RELEVANCE_RE = re.compile(
    r"\bnfl draft\b|\bpro football draft\b|\bbe drafted\b|\bdraft pick\b|\bheisman\b",
    re.I,
)

LEDGER_EDGE_THRESHOLD = 10.0  # points of |edge| before a paper-ledger entry
# Ledger rows are only taken on REAL two-sided quotes: a last trade or a
# midpoint, on a book that is tight (bid >= 2c, spread <= 10c) or has traded.
# A one-sided ask, or a 1c/65c placeholder book on zero volume, is not a
# market opinion worth putting on the record.
_LEDGER_PRICE_SOURCES = ("last", "mid")
LEDGER_MIN_BID_CENTS = 2
LEDGER_MAX_SPREAD_CENTS = 10

# pick_range is in NOMINAL board-rank space (rank within the whole prospect
# board, ~15k rows), so `hi` can sit far past the last real draft slot. Any
# rank beyond ~260 means "undrafted"; the high tail is spread log-uniformly
# up to this ceiling, which no draft-position question ever queries.
_RANK_CEILING = 20000.0
_DEFAULT_INTERVAL_COVERAGE = 0.8

# ── In-memory Kalshi cache (owned by the background thread) ──────────────────
_kalshi_lock = threading.Lock()
_kalshi_cache: dict = {
    "fetched_at": 0.0,        # time.monotonic() of the last completed pass
    "fetched_at_utc": None,   # ISO timestamp of the last SUCCESSFUL pass
    "markets": None,          # list | None (None = never fetched)
    "series_counts": {},      # series ticker -> markets found (last pass)
    "error": None,            # last pass error string, or None
    "refreshing": False,
}
_warm_thread = None

# ── Paper ledger (atomic write + mtime hot-reload) ────────────────────────────
_ledger_lock = threading.Lock()
_LEDGER: dict = {"entries": [], "last_synced_at": None}
_LEDGER_MTIME: float = 0.0

# ── Price history (same atomic write + mtime hot-reload pattern) ─────────────
_history_lock = threading.Lock()
_HISTORY: dict = {"tickers": {}}
_HISTORY_MTIME: float = 0.0

# ── Resolved board memo (markets × prospects → rows + player index) ──────────
# The market list changes every ~10 min and the prospect cache weekly, so the
# 77-row match/price pass is memoised on the identity of both inputs (the
# memo holds references, so an id() can never be recycled under it). Every
# /api/edge* request then reads the same rows; per-player lookups go through
# the index instead of re-matching.
_board_lock = threading.Lock()
_BOARD_MEMO: dict = {"markets": None, "prospects": None, "board": None}


def _after_fork_in_child() -> None:
    """A forked child inherits the parent's memory but none of its threads:
    a lock the refresh thread held at fork time would stay locked forever,
    and the parent's Thread object reports is_alive() for a thread that does
    not exist here. Swap in fresh locks and forget the thread so
    ensure_background_refresh() re-arms discovery in the child. The
    inherited market list is kept — it is a valid last-good pass."""
    global _kalshi_lock, _ledger_lock, _history_lock, _board_lock, _warm_thread
    _kalshi_lock = threading.Lock()
    _ledger_lock = threading.Lock()
    _history_lock = threading.Lock()
    _board_lock = threading.Lock()
    _warm_thread = None
    _kalshi_cache["refreshing"] = False


if hasattr(os, "register_at_fork"):  # POSIX; absent on Windows
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _load_ledger() -> None:
    global _LEDGER, _LEDGER_MTIME
    if not os.path.exists(LEDGER_PATH):
        return
    try:
        mtime = os.path.getmtime(LEDGER_PATH)  # capture BEFORE reading
        with open(LEDGER_PATH) as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        synced = data.get("last_synced_at") if isinstance(data, dict) else None
        _LEDGER = {"entries": [e for e in entries if isinstance(e, dict)],
                   "last_synced_at": synced if isinstance(synced, str) else None}
        _LEDGER_MTIME = mtime
    except Exception as exc:  # file mid-rewrite etc. — retried next request
        print(f"Edge ledger load failed (will retry): {exc}")


def _ledger_file_dict() -> dict:
    """The on-disk shape: entries plus the --sync-ledger stamp."""
    out = {"entries": _LEDGER["entries"]}
    if _LEDGER.get("last_synced_at"):
        out["last_synced_at"] = _LEDGER["last_synced_at"]
    return out


def _maybe_reload_ledger() -> None:
    """Cheap mtime check so every worker picks up appends without restarts."""
    try:
        mtime = os.path.getmtime(LEDGER_PATH)
    except OSError:
        return
    if mtime == _LEDGER_MTIME:
        return
    with _ledger_lock:
        if mtime == _LEDGER_MTIME:  # another thread already reloaded
            return
        _load_ledger()


def _write_ledger_atomic(data: dict) -> None:
    # Atomic write: temp file + os.replace, so a reader (or the mtime
    # hot-reload in another worker) never sees a half-written file.
    _write_json_atomic(LEDGER_PATH, data)


def _write_json_atomic(path: str, data: dict) -> None:
    """temp file + fsync + os.replace; sorted keys so git diffs stay stable
    (the file is committed weekly by the refresh job)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _merge_ledger_entries(candidates: list) -> int:
    """Merge rows into the on-disk ledger, deduped on (ticker, date). Caller
    holds _ledger_lock. Returns the number of rows added."""
    global _LEDGER_MTIME
    _load_ledger()  # merge against whatever is on disk right now
    seen = {(e.get("ticker"), e.get("date")) for e in _LEDGER["entries"]}
    added = 0
    for row in candidates:
        if not isinstance(row, dict) or not row.get("ticker") or not row.get("date"):
            continue
        key = (row["ticker"], row["date"])
        if key in seen:
            continue
        _LEDGER["entries"].append(row)
        seen.add(key)
        added += 1
    if added:
        _write_ledger_atomic(_ledger_file_dict())
        _LEDGER_MTIME = os.path.getmtime(LEDGER_PATH)
    return added


def _record_ledger_entries(candidates: list) -> None:
    """Append paper-ledger rows, deduped to one per (ticker, day).

    Best-effort: a write failure is logged and swallowed — the board must
    render regardless.
    """
    if not candidates:
        return
    with _ledger_lock:
        try:
            _merge_ledger_entries(candidates)
        except Exception as exc:
            print(f"Edge ledger append failed: {exc}")


def sync_ledger_from(base_url: str, timeout: float = 60.0) -> int:
    """Pull the paper ledger a running deployment has accumulated (one GET of
    /api/edge/ledger) and merge it into the local LEDGER_PATH. Used by the
    weekly refresh runner so the track record survives `fly deploy`. Returns
    the number of rows added. Read-only against the remote."""
    url = base_url.rstrip("/") + "/api/edge/ledger"
    resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    global _LEDGER_MTIME
    data = resp.json()
    entries = data.get("entries", []) if isinstance(data, dict) else []
    with _ledger_lock:
        added = _merge_ledger_entries([e for e in entries if isinstance(e, dict)])
        # Stamp the sync time into the file (served as scoring.last_synced_at)
        # even when nothing new merged — the stamp is the receipt.
        _LEDGER["last_synced_at"] = datetime.now(timezone.utc).isoformat()
        _write_ledger_atomic(_ledger_file_dict())
        _LEDGER_MTIME = os.path.getmtime(LEDGER_PATH)
        return added


_EMPTY_RESOLUTION = {
    "outcome": None,        # null until draft night; then 'yes' | 'no' | 'void'
    "settled_at": None,
    "brier_model": None,    # (model_prob/100 - y)^2
    "brier_market": None,   # (market_price_cents/100 - y)^2
    "closer": None,         # 'model' | 'market' | 'push'
    "note": None,
}


def _resolution_of(entry: dict) -> dict:
    res = entry.get("resolution")
    out = dict(_EMPTY_RESOLUTION)
    if isinstance(res, dict):
        out.update({k: res.get(k) for k in _EMPTY_RESOLUTION if k in res})
    if out["outcome"] not in (None, "yes", "no", "void"):
        out["outcome"] = None
    return out


def _ledger_by_ticker() -> dict:
    """ticker -> entries sorted by date (oldest first). Caller has reloaded."""
    by: dict = {}
    for e in _LEDGER["entries"]:
        by.setdefault(str(e.get("ticker") or ""), []).append(e)
    for rows in by.values():
        rows.sort(key=lambda e: str(e.get("date") or ""))
    return by


def _ledger_calls() -> dict:
    """ticker -> {first_date, price_at_call, model_at_call} from the earliest
    ledger row (the frozen call the record is scored on)."""
    out = {}
    for ticker, rows in _ledger_by_ticker().items():
        first = rows[0]
        out[ticker] = {
            "first_date": first.get("date"),
            "price_at_call": first.get("market_price_cents"),
            "model_at_call": first.get("model_prob"),
        }
    return out


def _date_of_iso(ts) -> str:
    return str(ts)[:10] if ts else None


def _ledger_summary(entries: list, market_by_ticker: dict) -> tuple:
    """(summary, scoring) for the ledger response. Resolution fields are all
    null until --score-ledger runs after the draft, so every count but
    `calls`/`open` is 0 — the intended pre-draft state."""
    calls = len(entries)
    resolved = void = model_closer = market_closer = push = 0
    briers_m, briers_k = [], []
    for e in entries:
        res = _resolution_of(e)
        if res["outcome"] == "void":
            void += 1
        elif res["outcome"] in ("yes", "no"):
            resolved += 1
            if res["closer"] == "model":
                model_closer += 1
            elif res["closer"] == "market":
                market_closer += 1
            elif res["closer"] == "push":
                push += 1
            if isinstance(res["brier_model"], (int, float)):
                briers_m.append(float(res["brier_model"]))
            if isinstance(res["brier_market"], (int, float)):
                briers_k.append(float(res["brier_market"]))
    open_n = calls - resolved - void
    gaps = [abs(float(e["edge"])) for e in entries
            if isinstance(e.get("edge"), (int, float))]
    dates = sorted(str(e.get("date")) for e in entries if e.get("date"))
    # settles_on: the earliest close_time over still-open tickers — live from
    # the market cache, else the close_time frozen on the entry.
    closes = []
    for e in entries:
        if _resolution_of(e)["outcome"] is not None:
            continue
        live = market_by_ticker.get(e.get("ticker"))
        ct = (live or {}).get("close_time") or e.get("close_time")
        if ct:
            closes.append(str(ct))
    summary = {
        "calls": calls,
        "players": len({str(e.get("player") or "") for e in entries if e.get("player")}),
        "open": open_n,
        "resolved": resolved,
        "scored": resolved,
        "void": void,
        "model_closer": model_closer,
        "market_closer": market_closer,
        "push": push,
        "avg_abs_gap": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "model_brier": round(sum(briers_m) / len(briers_m), 4) if briers_m else None,
        "market_brier": round(sum(briers_k) / len(briers_k), 4) if briers_k else None,
        "brier": None if not briers_m else round(sum(briers_m) / len(briers_m), 4),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "scoring_note": LEDGER_SCORING_NOTE,
    }
    # 'open' only while nothing has resolved or voided; 'scored' once no call
    # is still open; anything in between is 'partial'.
    if resolved == 0 and void == 0:
        status = "open"
    elif open_n == 0:
        status = "scored"
    else:
        status = "partial"
    scoring = {
        "rule": "brier",
        "settles_on": _date_of_iso(min(closes)) if closes else None,
        "status": status,
        "last_synced_at": _LEDGER.get("last_synced_at"),
    }
    return summary, scoring


def ledger_payload() -> dict:
    _maybe_reload_ledger()
    markets = _peek_markets_cached() or []
    by_ticker = {m["ticker"]: m for m in markets}
    entries = []
    for e in sorted(
            _LEDGER["entries"],
            key=lambda e: (str(e.get("date") or ""), str(e.get("ticker") or "")),
            reverse=True):
        row = dict(e)
        # Identity/question fields are written at record time; older rows
        # (pre-backfill) still serve nulls so the shape is stable.
        for key in ("team", "slug", "espn_id", "espn_team_id", "event_ticker",
                    "question", "model_interval", "event_scale"):
            row.setdefault(key, None)
        live = by_ticker.get(e.get("ticker"))
        row["current_price_cents"] = live.get("yes_price_cents") if live else None
        row["resolution"] = _resolution_of(e)
        entries.append(row)
    summary, scoring = _ledger_summary(_LEDGER["entries"], by_ticker)
    return {
        "entries": entries,
        "summary": summary,
        "scoring": scoring,
        "note": (
            "Paper ledger only — no positions are ever taken. Rows are "
            "recorded automatically when the model-implied probability and "
            "a two-sided market quote disagree by 10+ points, to build a "
            "verifiable public track record."
        ),
    }


# ── Ledger backfill (operator, idempotent) ───────────────────────────────────
_BASIS_INTERVAL_RE = re.compile(r"(\d{1,3})% conformal pick interval \[(\d+), (\d+)\]")
_BASIS_SCALE_RE = re.compile(r"scaled x(\d+(?:\.\d+)?)")


def _interval_from_basis(basis: str):
    m = _BASIS_INTERVAL_RE.search(basis or "")
    if not m:
        return None
    return {"lo": int(m.group(2)), "hi": int(m.group(3)),
            "coverage": round(int(m.group(1)) / 100.0, 2)}


def _scale_from_basis(basis: str):
    m = _BASIS_SCALE_RE.search(basis or "")
    if m:
        return float(m.group(1))
    return 1.0 if (basis or "").startswith("model-implied") else None


def _enrich_ledger_entry(entry: dict, board_row, index: dict) -> bool:
    """Fill the identity/question fields of one ledger entry in place from
    the live board row (preferred) or, off-season, by re-resolving the
    player against the prospect index and parsing the ticker/basis. Only
    missing (None) keys are written, so a second pass is a no-op. Returns
    True when anything changed."""
    ticker = str(entry.get("ticker") or "")
    event_ticker = entry.get("event_ticker") or (board_row or {}).get("event_ticker") \
        or (ticker.rsplit("-", 1)[0] if "-" in ticker else None)
    series = (board_row or {}).get("series") or (event_ticker or "").split("-", 1)[0]
    want = {}
    if board_row and board_row.get("matched_player"):
        want.update({
            "team": board_row.get("matched_team"),
            "slug": board_row.get("matched_slug"),
            "espn_id": board_row.get("matched_espn_id"),
            "espn_team_id": board_row.get("matched_espn_team_id"),
        })
    else:
        player, _ = _resolve_player([entry.get("player") or ""], index,
                                    draft_year=_draft_year(event_ticker or ""))
        if player:
            want.update({
                "team": player.get("team"),
                "slug": _player_slug(player.get("name"), player.get("team")),
                "espn_id": _str_or_none(player.get("espn_id")),
                "espn_team_id": _str_or_none(player.get("espn_team_id")),
            })
    want["event_ticker"] = event_ticker
    want["question"] = (board_row or {}).get("question") or _question_public(
        _market_question(series, event_ticker or "", entry.get("title") or "", {}))
    # Record-time values come from the basis string frozen on the entry, so
    # the backfill cannot silently swap in a newer interval.
    want["model_interval"] = _interval_from_basis(entry.get("model_basis")) \
        or (board_row or {}).get("model_interval")
    want["event_scale"] = _scale_from_basis(entry.get("model_basis"))
    if want["event_scale"] is None:
        want["event_scale"] = (board_row or {}).get("event_scale")
    want["close_time"] = (board_row or {}).get("close_time")
    if entry.get("resolution") is None:
        want["resolution"] = dict(_EMPTY_RESOLUTION)
    changed = False
    for key, val in want.items():
        if entry.get(key) is None and val is not None:
            entry[key] = val
            changed = True
    return changed


def backfill_ledger(prospects: list) -> int:
    """Enrich every on-disk ledger entry with the identity/question/interval
    fields newer entries carry at record time. Idempotent: returns the number
    of entries changed (0 on a second run). Uses the current market cache
    when one exists, else the prospect index + ticker parsing."""
    global _LEDGER_MTIME
    markets = _peek_markets_cached()
    board_rows = {}
    if markets:
        board_rows = {r["ticker"]: r for r in _resolve_board(prospects)["rows"]}
    index = _build_prospect_index(prospects)
    with _ledger_lock:
        _load_ledger()
        changed = 0
        for entry in _LEDGER["entries"]:
            if _enrich_ledger_entry(entry, board_rows.get(entry.get("ticker")), index):
                changed += 1
        if changed:
            _write_ledger_atomic(_ledger_file_dict())
            _LEDGER_MTIME = os.path.getmtime(LEDGER_PATH)
    return changed


# ── Ledger scoring (operator, after draft night; GET-only, idempotent) ───────
# Pre-registered rule (published on /futures before the draft): each call is
# scored by Brier against the settled YES/NO result of its market —
#   brier_model  = (model_prob / 100 − y)²
#   brier_market = (market_price_cents / 100 − y)²     y ∈ {0, 1}
# and `closer` is whichever is lower ('push' on a tie). Voided or unlisted
# markets are marked 'void' and excluded from both means.
_SETTLED_RESULTS = {"yes": 1, "no": 0}
_VOID_RESULTS = ("void", "voided", "cancelled", "canceled")


def _brier(p, y):
    p = _num(p)
    if p is None:
        return None
    return round((p / 100.0 - float(y)) ** 2, 4)


def _resolution_for(entry: dict, outcome: str, settled_at, note=None) -> dict:
    """The resolution block for one ledger entry given a settled outcome."""
    res = dict(_EMPTY_RESOLUTION, outcome=outcome, settled_at=settled_at, note=note)
    if outcome in _SETTLED_RESULTS:
        y = _SETTLED_RESULTS[outcome]
        bm, bk = _brier(entry.get("model_prob"), y), _brier(entry.get("market_price_cents"), y)
        res["brier_model"], res["brier_market"] = bm, bk
        if bm is not None and bk is not None:
            res["closer"] = "push" if bm == bk else ("model" if bm < bk else "market")
    return res


def _fetch_market_result(session, ticker: str, timeout: float = REQUEST_TIMEOUT_S):
    """GET /markets/{ticker} (public, unauthenticated). Returns
    (outcome | None, settled_at, note): 'yes'/'no' once settled, 'void' for a
    voided or unlisted (404) market, None while still open. Raises on any
    other HTTP/network failure so the caller leaves the call open."""
    url = f"{KALSHI_BASE}/markets/{ticker}"
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        return "void", None, "market no longer listed on Kalshi"
    resp.raise_for_status()
    data = resp.json()
    m = data.get("market") if isinstance(data, dict) else None
    if not isinstance(m, dict):
        raise ValueError(f"unexpected payload for {ticker}")
    result = str(m.get("result") or "").strip().lower()
    settled_at = _str_or_none(m.get("settlement_ts")) or _str_or_none(m.get("expiration_time"))
    status = str(m.get("status") or "").lower()
    if result in _SETTLED_RESULTS:
        return result, settled_at, None
    if result in _VOID_RESULTS:
        return "void", settled_at, "market voided by Kalshi"
    if status in ("settled", "finalized") and result:
        # e.g. 'scalar' / 'all_no' — not a YES/NO outcome our rule can score
        return "void", settled_at, f"unscoreable settlement result '{result}'"
    return None, None, None


def score_ledger(fetch_result=None, now: datetime = None, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """Fill `resolution` on every still-open ledger call whose market has
    settled. One public GET per distinct open ticker (never an order or
    portfolio endpoint — there is no key to send). Idempotent: entries that
    already carry an outcome are never re-fetched or rewritten, and a run
    before the draft is a no-op. Returns counts:
    {tickers, scored, void, open, errors}."""
    global _LEDGER_MTIME
    now = now or datetime.now(timezone.utc)
    pace = fetch_result is None  # live GETs are paced; an injected fetcher (tests) is not
    if fetch_result is None:
        session = requests.Session()
        session.headers["Accept"] = "application/json"

        def fetch_result(ticker):
            return _fetch_market_result(session, ticker, timeout=timeout)
    counts = {"tickers": 0, "scored": 0, "void": 0, "open": 0, "errors": 0}
    with _ledger_lock:
        _load_ledger()
        open_by_ticker: dict = {}
        for e in _LEDGER["entries"]:
            if _resolution_of(e)["outcome"] is None and e.get("ticker"):
                open_by_ticker.setdefault(str(e["ticker"]), []).append(e)
        counts["tickers"] = len(open_by_ticker)
        changed = False
        for i, (ticker, entries) in enumerate(sorted(open_by_ticker.items())):
            if i and pace:
                time.sleep(INTER_REQUEST_DELAY_S)
            try:
                outcome, settled_at, note = fetch_result(ticker)
            except Exception as exc:
                counts["errors"] += 1
                print(f"ledger score: {ticker} left open ({exc})")
                continue
            if outcome is None:
                counts["open"] += 1
                continue
            settled_at = settled_at or now.isoformat()
            for e in entries:
                e["resolution"] = _resolution_for(e, outcome, settled_at, note)
                changed = True
            counts["scored" if outcome in _SETTLED_RESULTS else "void"] += 1
        if changed:
            _write_ledger_atomic(_ledger_file_dict())
            _LEDGER_MTIME = os.path.getmtime(LEDGER_PATH)
    return counts


# ── Price history (capture on the warm refresh; served per ticker) ───────────
def _load_history() -> None:
    global _HISTORY, _HISTORY_MTIME
    if not os.path.exists(HISTORY_PATH):
        return
    try:
        mtime = os.path.getmtime(HISTORY_PATH)
        with open(HISTORY_PATH) as f:
            data = json.load(f)
        tickers = data.get("tickers", {}) if isinstance(data, dict) else {}
        _HISTORY = {"tickers": {k: v for k, v in tickers.items()
                                if isinstance(v, dict) and isinstance(v.get("points"), list)}}
        _HISTORY_MTIME = mtime
    except Exception as exc:
        print(f"Edge history load failed (will retry): {exc}")


def _maybe_reload_history() -> None:
    try:
        mtime = os.path.getmtime(HISTORY_PATH)
    except OSError:
        return
    if mtime == _HISTORY_MTIME:
        return
    with _history_lock:
        if mtime == _HISTORY_MTIME:
            return
        _load_history()


def _capture_history(markets: list, now: datetime = None) -> int:
    """Append one point per market whose UTC date differs from its last
    stored point or whose price moved >= HISTORY_MIN_MOVE_CENTS; bound each
    ticker to HISTORY_MAX_POINTS (oldest dropped); prune tickers not seen for
    HISTORY_PRUNE_DAYS or closed that long ago. Atomic write only when
    something changed. Returns the number of points appended."""
    global _HISTORY_MTIME
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    cutoff = now.timestamp() - HISTORY_PRUNE_DAYS * 86400
    appended, changed = 0, False
    with _history_lock:
        _load_history()  # merge against the file another worker may have written
        tickers = _HISTORY["tickers"]
        for m in markets or []:
            ticker, price = m.get("ticker"), m.get("yes_price_cents")
            if not ticker or price is None:
                continue
            rec = tickers.get(ticker)
            if rec is None:
                rec = tickers[ticker] = {"points": [], "close_time": None, "last_seen": None}
            if rec.get("last_seen") != today or rec.get("close_time") != m.get("close_time"):
                rec["last_seen"], rec["close_time"] = today, m.get("close_time")
                changed = True
            pts = rec["points"]
            last = pts[-1] if pts else None
            if last is None or last[0] != today or abs(int(last[1]) - int(price)) >= HISTORY_MIN_MOVE_CENTS:
                pts.append([today, int(price), m.get("yes_price_source")])
                appended += 1
                changed = True
                if len(pts) > HISTORY_MAX_POINTS:
                    del pts[:len(pts) - HISTORY_MAX_POINTS]
        for ticker in list(tickers):
            rec = tickers[ticker]
            seen_ts = _iso_ts(rec.get("last_seen"))
            close_ts = _iso_ts(rec.get("close_time"))
            if (seen_ts is not None and seen_ts < cutoff) or (close_ts is not None and close_ts < cutoff):
                del tickers[ticker]
                changed = True
        if changed:
            try:
                _write_json_atomic(HISTORY_PATH, {
                    "tickers": tickers, "updated_at": now.isoformat()})
                _HISTORY_MTIME = os.path.getmtime(HISTORY_PATH)
            except Exception as exc:  # best-effort: the board never depends on it
                print(f"Edge history write failed: {exc}")
    return appended


def _iso_ts(s):
    """POSIX seconds for 'YYYY-MM-DD' or an ISO datetime; None if unparseable."""
    if not s:
        return None
    try:
        txt = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def history_payload(ticker: str) -> dict:
    """GET /api/edge/history?ticker= — unknown ticker → empty points, never 404."""
    _maybe_reload_history()
    ticker = (ticker or "").strip().upper()
    rec = _HISTORY["tickers"].get(ticker) or {}
    points = [list(p) for p in rec.get("points", [])][-HISTORY_MAX_POINTS:]
    return {"ticker": ticker, "points": points, "bounded": True}


def _merge_history_points(local: list, remote: list) -> list:
    """Union of two point lists (exact [date, cents, source] triples deduped),
    ordered by date with the remote order winning inside a day (the serving
    process is the one that captured them), bounded to HISTORY_MAX_POINTS."""
    seen, out = set(), []
    for p in list(remote or []) + list(local or []):
        if not (isinstance(p, (list, tuple)) and len(p) >= 2 and p[0]):
            continue
        key = (str(p[0]), int(p[1]) if _num(p[1]) is not None else None, p[2] if len(p) > 2 else None)
        if key[1] is None or key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1], key[2]])
    out.sort(key=lambda p: p[0])  # stable: remote-then-local order kept within a day
    return out[-HISTORY_MAX_POINTS:]


def sync_history_from(base_url: str, timeout: float = 60.0, tickers: list = None) -> int:
    """Pull the price history a running deployment captured (GET
    /api/edge/history?ticker= for every ticker on its board and ledger) and
    merge it into HISTORY_PATH, bounded per ticker. Read-only against the
    remote; used by the weekly refresh next to --sync-ledger because the Fly
    container has no volume. Returns the number of points added."""
    global _HISTORY_MTIME
    base = base_url.rstrip("/")
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    if tickers is None:
        # The first request may wake a scaled-to-zero machine, hence the long
        # timeout; the board GET is the source of "every ticker in the remote
        # payload", the ledger GET adds calls whose markets have since closed.
        resp = session.get(base + "/api/edge", timeout=timeout)
        resp.raise_for_status()
        board = resp.json()
        found = [m.get("ticker") for m in (board.get("markets") or []) if isinstance(m, dict)]
        try:
            lr = session.get(base + "/api/edge/ledger", timeout=timeout)
            lr.raise_for_status()
            found += [e.get("ticker") for e in (lr.json().get("entries") or []) if isinstance(e, dict)]
        except Exception as exc:
            print(f"history sync: ledger ticker list skipped ({exc})")
        tickers = sorted({str(t).strip().upper() for t in found if t})
    added = 0
    now = datetime.now(timezone.utc)
    with _history_lock:
        _load_history()
        store = _HISTORY["tickers"]
        for ticker in tickers:
            try:
                r = session.get(base + "/api/edge/history", params={"ticker": ticker}, timeout=timeout)
                r.raise_for_status()
                remote = (r.json() or {}).get("points") or []
            except Exception as exc:
                print(f"history sync: {ticker} skipped ({exc})")
                continue
            if not remote:
                continue
            rec = store.get(ticker)
            if rec is None:
                rec = store[ticker] = {"points": [], "close_time": None, "last_seen": None}
            before = len(rec["points"])
            merged = _merge_history_points(rec["points"], remote)
            added += max(len(merged) - before, 0)
            rec["points"] = merged
            last_point_day = merged[-1][0] if merged else None
            if last_point_day and (not rec.get("last_seen") or rec["last_seen"] < last_point_day):
                rec["last_seen"] = last_point_day
        _write_json_atomic(HISTORY_PATH, {"tickers": store, "updated_at": now.isoformat(),
                                          "last_synced_at": now.isoformat()})
        _HISTORY_MTIME = os.path.getmtime(HISTORY_PATH)
    return added


# ── Name normalization + player matching ──────────────────────────────────────
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(name: str) -> str:
    """Accent-fold, lowercase, strip punctuation and Jr/III-style suffixes."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t not in _NAME_SUFFIXES]
    return " ".join(tokens)


def _norm_team(team: str) -> str:
    t = _norm_name(team)
    return re.sub(r"\bst\b", "state", t)


def _player_slug(name, team) -> str:
    """Python port of src/lib/edgeData.js playerSlug: `${name}-${team}`
    lower-cased, whitespace runs → '-', then everything outside [a-z0-9-]
    stripped. Must stay byte-identical to the JS so /player/<slug> links and
    ?slug= lookups agree."""
    s = f"{name or ''}-{team or ''}".lower()
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"[^a-z0-9-]", "", s)


def _str_or_none(v):
    if v is None or v == "":
        return None
    return str(v)


_QUESTION_KIND_ORDER = {"top_n": 0, "pick_eq": 1, "heisman": 2}


def _question_label(kind: str, n) -> str:
    if kind == "top_n":
        return f"Top-{n} pick" if n is not None else "Top pick"
    if kind == "pick_eq":
        return f"#{n} overall" if n is not None else "Exact pick"
    if kind == "after_n":
        return f"After pick {n}" if n is not None else "After pick"
    if kind == "heisman":
        return "Heisman"
    if kind == "nth_position":
        return "Positional order"
    return "Other"


def _question_public(q) -> dict:
    """The row-level `question` object: {kind, n, label}."""
    q = q or {}
    kind = q.get("kind") or "unknown"
    n = q.get("n")
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    return {"kind": kind, "n": n, "label": _question_label(kind, n)}


def _kind_rank(row: dict) -> int:
    return _QUESTION_KIND_ORDER.get(((row.get("question") or {}).get("kind")), 3)


def _build_prospect_index(prospects: list) -> dict:
    """normalized name -> [cache rows], one per distinct (name, team).

    Collisions are kept, not refused: _resolve_player tiebreaks them against
    the market's draft year / team hint / projected pick and logs the
    ambiguity. (Refusing dropped the two most-traded names on the board —
    Jeremiah Smith and Dylan Stewart — see kalshi_audit.md.)
    """
    index: dict = {}
    for row in prospects or []:
        key = _norm_name(row.get("name", ""))
        if not key:
            continue
        bucket = index.setdefault(key, [])
        if any(r.get("team") == row.get("team") for r in bucket):
            continue  # same player duplicated across rows — keep the first
        bucket.append(row)
    return index


def _pick_sort_key(row: dict) -> float:
    try:
        p = float(row.get("projected_pick"))
        return p if p > 0 else math.inf
    except (TypeError, ValueError):
        return math.inf


_ambiguity_logged: set = set()

# Tiebreak decisiveness (match_ambiguous semantics): a name collision is
# AMBIGUOUS only when the tiebreak was not decisive. It is decisive when the
# chosen row is a real draft prospect (projected_pick <= _DECISIVE_CHOSEN_PICK)
# and every alternative is nowhere near the draft (projected_pick >
# _DECISIVE_OTHER_PICK) or belongs to another draft class with no
# market-eligible profile (no pick_range). Jeremiah Smith (Ohio State WR, pick
# 1) vs Jeremiah Smith (Louisiana Tech LB, pick ~3,700) is decisive; two
# prospects both inside the top 300 are not.
_DECISIVE_CHOSEN_PICK = 100
_DECISIVE_OTHER_PICK = 300


def _tiebreak_decisive(chosen: dict, others: list, draft_year) -> bool:
    if _pick_sort_key(chosen) > _DECISIVE_CHOSEN_PICK:
        return False
    for r in others:
        if _pick_sort_key(r) > _DECISIVE_OTHER_PICK:
            continue
        other_class = _num(r.get("draft_class"))
        if (_player_interval(r) is None and draft_year
                and other_class is not None and int(other_class) != int(draft_year)):
            continue
        return False
    return True


def _class_before(row: dict, draft_year) -> bool:
    """True when the cache row's draft_class is EARLIER than the market's
    draft year — that player was already drafted (or the row is stale) and
    cannot be the subject of this market."""
    if not draft_year:
        return False
    try:
        return int(row.get("draft_class")) < int(draft_year)
    except (TypeError, ValueError):
        return False


def _resolve_player(cands: list, index: dict, draft_year=None, team_hint: str = ""):
    """First candidate name that hits the index wins. Rows whose draft_class
    is earlier than the market's draft year are excluded outright; otherwise
    draft_class is IGNORED — a true junior carries draft_class 2028 while
    being fully eligible for (and priced in) the 2027 draft, so "class ==
    market year" is not a tiebreak, it is a trap. If several distinct
    players remain, tiebreak: team hint from the market, then the higher
    (lower-numbered) projected pick. Returns (row | None, ambiguous: bool)
    where ambiguous means the tiebreak was NOT decisive (see
    _tiebreak_decisive) — a collision with a far-off-board namesake resolves
    cleanly and is eligible for the headline/spotlight."""
    for cand in cands:
        key = _norm_name(cand)
        rows = index.get(key)
        if not rows:
            continue
        rows = [r for r in rows if not _class_before(r, draft_year)]
        if not rows:
            continue
        if len(rows) == 1:
            return rows[0], False
        pool = rows
        hint = _norm_team(team_hint)
        if hint:
            by_team = [r for r in pool if hint in _norm_team(r.get("team", ""))]
            if by_team:
                pool = by_team
        chosen = sorted(pool, key=_pick_sort_key)[0]
        alternatives = [r for r in rows if r is not chosen]
        decisive = _tiebreak_decisive(chosen, alternatives, draft_year)
        log_key = (key, chosen.get("team"), decisive)
        if log_key not in _ambiguity_logged:
            _ambiguity_logged.add(log_key)
            others = ", ".join(
                f"{r.get('team')} ({r.get('position')}, pick {r.get('projected_pick')})"
                for r in alternatives
            )
            print(
                f"Edge name tiebreak: '{cand}' matches {len(rows)} cache rows; "
                f"chose {chosen.get('team')} ({chosen.get('position')}, pick "
                f"{chosen.get('projected_pick')}) over {others} — "
                + ("decisive (alternatives are off the draft board), match_ambiguous=false"
                   if decisive else
                   "NOT decisive (an alternative is a live prospect), match_ambiguous=true")
            )
        return chosen, not decisive
    return None, False


# Title shapes seen on draft markets; ordered most- to least-specific.
_NAME_PATTERNS = (
    re.compile(r"\bwill\s+(.+?)\s+be\s+(?:drafted|selected|picked|a\s+top)\b", re.I),
    re.compile(r"^(.+?)\s+(?:drafted|selected)\s+(?:in|by|top)\b", re.I),
    re.compile(r"\bdrafted:\s*(.+?)\s*\??$", re.I),
)


def _candidate_names(event_title: str, market: dict) -> list:
    """Best-effort player-name candidates from a market. Kalshi player
    markets carry the name in custom_strike.Person and yes_sub_title;
    single-market events phrase it in the title. Candidates are only
    *candidates* — the prospect index lookup is the real filter."""
    cands = []
    strike = market.get("custom_strike")
    if isinstance(strike, dict):
        person = (strike.get("Person") or "").strip()
        if person:
            cands.append(person)
    for key in ("yes_sub_title", "no_sub_title"):
        v = (market.get(key) or "").strip()
        if v:
            cands.append(v)
    sub = (market.get("subtitle") or "").strip()
    if sub:
        # "Jeremiah Smith:: Ohio St." / ":: Louisville" / "Arch Manning"
        head = sub.split("::", 1)[0].strip()
        if head:
            cands.append(head)
    for text in ((market.get("title") or ""), (event_title or "")):
        for pat in _NAME_PATTERNS:
            m = pat.search(text)
            if m:
                cands.append(m.group(1))
        # Last resort: capitalized First Last(-ish) runs inside the title.
        cands.extend(re.findall(r"\b([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,2})\b", text))
    seen, out = set(), []
    for c in cands:
        k = _norm_name(c)
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _team_hint(market: dict) -> str:
    """Kalshi puts the school after '::' in subtitle on multi-player events."""
    sub = (market.get("subtitle") or "")
    if "::" in sub:
        return sub.split("::", 1)[1].strip()
    return ""


# ── Market-question parsing (what can the model honestly answer?) ─────────────
# "-27-5", "-27", "-27P2" (KXNFLDRAFTQB-27P2) all carry the two-digit year.
_EVENT_YEAR_RE = re.compile(r"-(\d{2})(?:\D|$)")
_EVENT_TOP_RE = re.compile(r"^KXNFLDRAFTTOP-\d{2}-(\d{1,3})$")
_EVENT_R1_RE = re.compile(r"^KXNFLDRAFTTOP-\d{2}-R1$")
_EVENT_PICK_RE = re.compile(r"^KXNFLDRAFTPICK-\d{2}-(\d{1,3})$")
_TOP_N_TITLE_RE = re.compile(r"\btop[\s\-]*(\d{1,3})\b", re.I)
_PICKED_K_TITLE_RE = re.compile(r"\bpicked\s+(\d{1,3})(?:st|nd|rd|th)\b", re.I)
_OU_TITLE_RE = re.compile(
    r"\b(over|under|before|after)\s+(?:pick\s*)?#?\s*(\d{1,3}(?:\.5)?)\b", re.I
)


def _draft_year(event_ticker: str):
    m = _EVENT_YEAR_RE.search((event_ticker or "").upper())
    return 2000 + int(m.group(1)) if m else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _market_question(series: str, event_ticker: str, title: str, m: dict):
    """Classify a market into a draft-position question the model can price,
    or a listed-only kind. Returns a dict with "kind" in
    {"top_n", "pick_eq", "after_n", "heisman", "nth_position", "unknown"}
    and, for the priceable kinds, "n"."""
    series = (series or "").upper()
    event_ticker = (event_ticker or "").upper()
    title = title or ""

    if series.startswith("KXHEISMAN"):
        return {"kind": "heisman"}

    if series.startswith("KXNFLDRAFTTOP"):
        em = _EVENT_TOP_RE.match(event_ticker)
        if em:
            return {"kind": "top_n", "n": int(em.group(1))}
        if _EVENT_R1_RE.match(event_ticker):
            return {"kind": "top_n", "n": 32}

    if series.startswith("KXNFLDRAFTPICK"):
        em = _EVENT_PICK_RE.match(event_ticker)
        if em:
            return {"kind": "pick_eq", "n": int(em.group(1))}
        tm = _PICKED_K_TITLE_RE.search(title)
        if tm:
            return {"kind": "pick_eq", "n": int(tm.group(1))}

    if series.startswith("KXNFLDRAFTOU"):
        st = (m.get("strike_type") or "").lower()
        cap, floor = _num(m.get("cap_strike")), _num(m.get("floor_strike"))
        if st in ("less", "less_or_equal") and cap is not None:
            n = int(math.floor(cap)) if st == "less_or_equal" else int(math.ceil(cap)) - 1
            return {"kind": "top_n", "n": max(n, 1)}
        if st in ("greater", "greater_or_equal") and floor is not None:
            n = int(math.floor(floor)) if st == "greater" else int(math.ceil(floor)) - 1
            return {"kind": "after_n", "n": max(n, 0)}
        om = _OU_TITLE_RE.search(title)
        if om:
            word, x = om.group(1).lower(), float(om.group(2))
            if word in ("under", "before"):
                # "under 10.5" -> pick <= 10; "under 10" -> pick <= 9
                n = int(math.floor(x)) if x % 1 else int(x) - 1
                return {"kind": "top_n", "n": max(n, 1)}
            return {"kind": "after_n", "n": int(math.floor(x))}

    if series.startswith("KXNFLDRAFT") and re.match(
            r"^KXNFLDRAFT(QB|RB|WR|TE|OL|EDGE|DT|LB|DB)\b", series):
        return {"kind": "nth_position"}

    # Generic title fallbacks for families we have not seen yet.
    tm = _TOP_N_TITLE_RE.search(title)
    if tm:
        return {"kind": "top_n", "n": int(tm.group(1))}
    if re.search(r"\b(?:first|1st)\s+round\b", title, re.I):
        return {"kind": "top_n", "n": 32}
    return {"kind": "unknown"}


# ── Model-implied pick probabilities ──────────────────────────────────────────
def _log_frac(x: float, a: float, b: float) -> float:
    """Share of a log-uniform segment [a, b] lying below x (clamped 0-1)."""
    if b <= a:
        return 1.0 if x >= b else 0.0
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (math.log(x) - math.log(a)) / (math.log(b) - math.log(a))


def _pick_cdf(x: float, lo: float, hi: float, coverage: float = _DEFAULT_INTERVAL_COVERAGE) -> float:
    """Model-implied P(pick < x) for a continuous pick position x, given a
    conformal interval [lo, hi] holding `coverage` of the mass.

    Assumptions (stated on every priced row via model_basis):
      * mass inside [lo, hi] is log-uniform;
      * the remaining (1 - coverage) is split equally into the tails, each
        also log-uniform — the low tail over [pick 1, lo], the high tail
        over [hi, _RANK_CEILING]. When lo is already pick 1 the low tail has
        no room, gets no mass, and its share goes to the interior. The high
        tail always has room (nominal ranks run to the board size), so mass
        past `hi` stays past `hi` — it is never folded back into the draft.
    """
    coverage = min(max(float(coverage or _DEFAULT_INTERVAL_COVERAGE), 0.05), 0.99)
    floor = 0.5  # pick k occupies [k-0.5, k+0.5] in continuous space
    L = max(float(lo) - 0.5, floor)
    H = min(max(float(hi) + 0.5, L), _RANK_CEILING)
    tail = (1.0 - coverage) / 2.0
    tail_lo = tail if L > floor else 0.0
    tail_hi = tail
    interior = 1.0 - tail_lo - tail_hi
    if x <= floor:
        return 0.0
    if x <= L:
        return tail_lo * _log_frac(x, floor, L)
    if x <= H:
        return tail_lo + interior * _log_frac(x, L, H)
    if x < _RANK_CEILING:
        return tail_lo + interior + tail_hi * _log_frac(x, H, _RANK_CEILING)
    return 1.0


def _p_pick_le(n: int, lo: float, hi: float, coverage: float) -> float:
    return _pick_cdf(n + 0.5, lo, hi, coverage)


# There is deliberately no _p_pick_eq: P(pick == K) is not identifiable from
# a coverage interval (see the module docstring) and must not be derived.

_NOT_PRICED_EXACT = "not priced: exact-pick probabilities are not identifiable from an interval"
_PRICEABLE_KINDS = ("top_n", "after_n")


def _player_interval(player: dict):
    """(lo, hi, coverage) from a cache row's pick_range, or None."""
    pr = player.get("pick_range")
    if not isinstance(pr, dict):
        return None
    lo, hi = _num(pr.get("lo")), _num(pr.get("hi"))
    if lo is None or hi is None or lo < 1 or hi < lo:
        return None
    cov = _num(pr.get("confidence")) or _DEFAULT_INTERVAL_COVERAGE
    return lo, hi, cov


def _raw_p_le(market_row: dict, player: dict):
    """The unscaled model-implied P(pick <= n) behind a priceable threshold
    market, or None when the row cannot be priced (used for the per-event
    coherence sum, which counts every matched player with an interval even
    when the market itself has no quote)."""
    q = market_row.get("_question") or {}
    if q.get("kind") not in _PRICEABLE_KINDS or q.get("n") is None:
        return None
    iv = _player_interval(player)
    if iv is None:
        return None
    lo, hi, cov = iv
    return _p_pick_le(int(q["n"]), lo, hi, cov)


def _event_scale_factors(matched: list) -> dict:
    """Per-event coherence: (event_ticker, n) -> factor in (0, 1].

    `matched` is a list of (market_row, player). Within one event at most n
    players can be drafted in the top n, so the matched players' P(pick <= n)
    must sum to <= n; when the raw marginals overshoot, every one of them is
    scaled by n / sum. Each player counts once per event."""
    sums: dict = {}
    seen: set = set()
    for src, player in matched:
        p = _raw_p_le(src, player)
        if p is None:
            continue
        q = src["_question"]
        key = (src.get("event_ticker") or src.get("ticker"), int(q["n"]))
        who = (key, player.get("name"), player.get("team"))
        if who in seen:
            continue
        seen.add(who)
        sums[key] = sums.get(key, 0.0) + p
    return {key: (n / total) for (key, total) in sums.items()
            for n in (key[1],) if total > n and total > 0}


def _model_fields(market_row: dict, player: dict, scale: float = 1.0):
    """Return (model_prob, edge, basis). model_prob/edge are None with a
    basis string explaining why when the question cannot be priced.

    Only draft-position THRESHOLD questions (top-N, over/under) are priced,
    and only from a persisted conformal pick_range — see the module
    docstring. `scale` (0, 1] is the event-coherence factor from
    _event_scale_factors; it multiplies P(pick <= n) and is written into the
    basis when it bites."""
    q = market_row.get("_question") or {"kind": "unknown"}
    kind = q.get("kind")
    if kind == "heisman":
        return None, None, "not priced: Heisman is an award, not a draft slot"
    if kind == "pick_eq":
        return None, None, _NOT_PRICED_EXACT
    if kind == "nth_position":
        return None, None, "not priced: positional draft order is not a persisted model output"
    if kind not in _PRICEABLE_KINDS:
        return None, None, "not priced: market question not recognised as a draft-position question"
    iv = _player_interval(player)
    if iv is None:
        return None, None, "not priced: cache row has no conformal pick_range"
    price = market_row.get("yes_price_cents")
    if price is None:
        return None, None, "not priced: market has no quote"
    lo, hi, cov = iv
    n = int(q["n"])
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 1.0
    if not (0.0 < scale <= 1.0) or not math.isfinite(scale):
        scale = 1.0
    p_le = _p_pick_le(n, lo, hi, cov) * scale
    if kind == "top_n":
        p = p_le
        what = f"P(pick <= {n})"
    else:
        p = 1.0 - p_le
        what = f"P(pick > {n})"
    model_prob = round(100.0 * p, 1)
    edge = round(model_prob - float(price), 1)
    basis = (
        f"model-implied {what} from the {int(round(cov * 100))}% conformal pick "
        f"interval [{int(lo)}, {int(hi)}], log-uniform mass"
    )
    if scale < 1.0:
        basis += (f"; scaled x{scale:.3f} so the event's matched P(pick <= {n}) "
                  f"sums to {n} (raw marginals overshot)")
    return model_prob, edge, basis


# ── Kalshi price parsing ──────────────────────────────────────────────────────
def _cents_from_dollars(v):
    """'0.7700' -> 77. None for missing/unparseable/non-finite/out-of-range
    ('inf', '-inf', 'nan' all parse as floats and must not reach round())."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).strip())
        if not math.isfinite(x):
            return None
        c = int(round(x * 100))
    except (TypeError, ValueError, OverflowError):
        return None
    return c if 0 <= c <= 100 else None


def _cents_from_int(v):
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        if not math.isfinite(v):
            return None
        c = int(round(v))
    except (TypeError, ValueError, OverflowError):
        return None
    return c if 0 <= c <= 100 else None


def _finite_num(v):
    """float(v) when it is a finite number (string or numeric), else None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).strip()) if isinstance(v, str) else float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _volume(m: dict):
    """Traded volume when the payload carries one (`volume_fp` string on the
    2026 schema, integer `volume` before it), else None."""
    for key in ("volume_fp", "volume"):
        if key in m:
            v = _finite_num(m.get(key))
            if v is not None:
                return max(v, 0.0)
    return None


def _price_field_cents(m: dict, base: str):
    """Kalshi now serves `<base>_dollars` strings; the integer-cent `<base>`
    key is the legacy fallback."""
    c = _cents_from_dollars(m.get(base + "_dollars"))
    if c is None:
        c = _cents_from_int(m.get(base))
    return c


def _yes_price(m: dict):
    """(cents, source) for the best available YES price: last trade, else
    bid/ask midpoint, else whichever side is quoted. (None, None) when the
    book is empty — a 0.0000 bid with no last trade is "no price", not 0."""
    last = _price_field_cents(m, "last_price")
    if last is not None and 0 < last < 100:
        return last, "last"
    bid, ask = _price_field_cents(m, "yes_bid"), _price_field_cents(m, "yes_ask")
    if bid is not None and ask is not None and 0 < bid <= ask < 100:
        return int(round((bid + ask) / 2)), "mid"
    if ask is not None and 0 < ask < 100:
        return ask, "ask"
    if bid is not None and 0 < bid < 100:
        return bid, "bid"
    return None, None


def _yes_price_cents(m: dict):
    return _yes_price(m)[0]


def _quote_quality(m: dict, price, source):
    """Classify the book behind a price for the paper ledger.

    Returns (source, spread_cents, volume, ledger_eligible). A midpoint on a
    wide book (spread > LEDGER_MAX_SPREAD_CENTS) is re-tagged `mid_wide` —
    it stays on the board, labelled, but is not a quote anyone traded.
    ledger_eligible requires a two-sided source (last or mid) AND either a
    tight book (bid >= LEDGER_MIN_BID_CENTS and spread <= LEDGER_MAX_SPREAD_
    CENTS) or traded volume > 0 when the payload reports volume."""
    bid, ask = _price_field_cents(m, "yes_bid"), _price_field_cents(m, "yes_ask")
    spread = (ask - bid) if (bid is not None and ask is not None and ask >= bid) else None
    volume = _volume(m)
    if price is None or source is None:
        return source, spread, volume, False
    tight = (bid is not None and spread is not None
             and bid >= LEDGER_MIN_BID_CENTS and spread <= LEDGER_MAX_SPREAD_CENTS)
    traded = volume is not None and volume > 0
    if source == "mid" and not tight:
        source = "mid_wide"
    eligible = source in _LEDGER_PRICE_SOURCES and (tight or traded)
    return source, spread, volume, eligible


# ── Kalshi discovery ──────────────────────────────────────────────────────────
def _is_relevant(series_ticker: str, event_ticker: str, title: str) -> bool:
    tick = (series_ticker or event_ticker or "").upper()
    if tick.startswith(_EXCLUDED_SERIES_PREFIXES):
        return False
    if tick.startswith(_SERIES_PREFIXES):
        return True
    return bool(_TITLE_RELEVANCE_RE.search(title or ""))


def _market_url(series_ticker: str, event_ticker: str) -> str:
    # kalshi.com/markets/<series-ticker> is the stable public series page;
    # per-event slugs are not derivable from the API payload alone.
    slug = (series_ticker or event_ticker or "").lower()
    return f"https://kalshi.com/markets/{slug}" if slug else "https://kalshi.com"


def _market_row(m: dict, event_title: str, series: str, event_ticker: str) -> dict:
    m_title = (m.get("title") or "").strip() or event_title
    # Nested-market titles are often just the strike; prefix the event
    # question so the row reads standalone.
    if event_title and m_title != event_title and len(m_title) < 25:
        m_title = f"{event_title} — {m_title}"
    # Multi-player events reuse one title ("Who will win Heisman Trophy?");
    # append the player so rows are distinguishable.
    person = (m.get("yes_sub_title") or "").strip()
    if person and person.lower() not in m_title.lower():
        m_title = f"{m_title} — {person}"
    price, source = _yes_price(m)
    source, spread, volume, eligible = _quote_quality(m, price, source)
    return {
        "ticker": m.get("ticker") or event_ticker,
        "title": m_title,
        "event_title": event_title,
        "event_ticker": event_ticker,
        "series": series,
        "draft_year": _draft_year(event_ticker),
        "yes_price_cents": price,
        "yes_price_source": source,
        "spread_cents": spread,
        "volume": volume,
        "close_time": _str_or_none(m.get("close_time")),
        "ledger_eligible": bool(eligible),
        "url": _market_url(series, event_ticker),
        "_market_raw_names": _candidate_names(event_title, m),
        "_team_hint": _team_hint(m),
        "_question": _market_question(series, event_ticker, m.get("title") or event_title, m),
    }


def _fetch_series_markets(session, series: str) -> list:
    """All open markets of one series (paginated, bounded). Raises on HTTP
    error so the caller can count the series as failed."""
    out, cursor = [], None
    for _ in range(MAX_SERIES_PAGES):
        params = {"series_ticker": series, "status": "open", "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        resp = session.get(f"{KALSHI_BASE}/markets", params=params,
                           timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out.extend(data.get("markets", []) or [])
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(INTER_REQUEST_DELAY_S)
    return out


def _fetch_relevant_markets():
    """One bounded, politely paced discovery pass over _KNOWN_SERIES.

    Returns (rows, series_counts). Fail-soft per series: a 429/network error
    on one family keeps whatever the others returned. Raises only if EVERY
    series errored — the caller keeps the previous good pass.
    """
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    rows, counts, errors = [], {}, 0

    for series in _KNOWN_SERIES:
        try:
            markets = _fetch_series_markets(session, series)
        except Exception as exc:
            errors += 1
            counts[series] = None
            print(f"Kalshi series probe {series} failed: {exc}")
            time.sleep(INTER_REQUEST_DELAY_S)
            continue
        kept = 0
        for m in markets:
            if not isinstance(m, dict):
                continue
            e_tick = m.get("event_ticker") or ""
            try:
                if not _is_relevant(series, e_tick, m.get("title") or ""):
                    continue
                rows.append(_market_row(m, "", series, e_tick))
            except Exception as exc:  # one malformed market must not abort the pass
                print(f"Kalshi market {m.get('ticker') or e_tick or '?'} skipped: {exc!r}")
                continue
            kept += 1
        counts[series] = kept
        time.sleep(INTER_REQUEST_DELAY_S)

    if errors == len(_KNOWN_SERIES):
        raise RuntimeError("Kalshi discovery: every series probe failed")

    seen, unique = set(), []
    for r in rows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        unique.append(r)
    return unique, counts


def _refresh_markets_once() -> None:
    """One discovery pass into the cache. Never raises. A failed pass keeps
    the last good market list (stale-while-revalidate); a failure before
    any success stores [] so the page shows the seasonal state."""
    with _kalshi_lock:
        if _kalshi_cache["refreshing"]:
            return
        _kalshi_cache["refreshing"] = True
    try:
        markets, counts = _fetch_relevant_markets()
        with _kalshi_lock:
            _kalshi_cache["markets"] = markets
            _kalshi_cache["series_counts"] = counts
            _kalshi_cache["fetched_at"] = time.monotonic()
            _kalshi_cache["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
            _kalshi_cache["error"] = None
        # Price-history capture rides the successful pass. Skipped under
        # DV_EDGE_NO_WARMUP (tests, --probe) so a diagnostic never writes the
        # git-tracked file; only the serving process captures.
        if not os.environ.get(_NO_WARMUP_ENV):
            try:
                _capture_history(markets)
            except Exception as exc:
                print(f"Edge history capture failed: {exc}")
    except Exception as exc:  # network/HTTP/schema — keep the last good pass
        print(f"Kalshi discovery failed (serving last good pass): {exc}")
        with _kalshi_lock:
            # fetched_at is NOT advanced here: it marks the last SUCCESSFUL
            # pass, so repeated failures age the cache into the 'stale' state.
            _kalshi_cache["error"] = str(exc)
            if _kalshi_cache["markets"] is None:
                _kalshi_cache["markets"] = []
    finally:
        with _kalshi_lock:
            _kalshi_cache["refreshing"] = False


def _warm_loop() -> None:
    while True:
        _refresh_markets_once()
        time.sleep(CACHE_TTL_S)


def ensure_background_refresh() -> bool:
    """Start (or restart after a fork) the daemon discovery thread. Returns
    True if a thread is running. No-op when DV_EDGE_NO_WARMUP is set."""
    global _warm_thread
    if os.environ.get(_NO_WARMUP_ENV):
        return False
    if _warm_thread is not None and _warm_thread.is_alive():
        return True
    with _kalshi_lock:
        if _warm_thread is not None and _warm_thread.is_alive():
            return True
        _warm_thread = threading.Thread(
            target=_warm_loop, name="dv-edge-kalshi-refresh", daemon=True)
        _warm_thread.start()
    return True


def _get_markets_cached():
    """Whatever the background thread last stored — never blocks on Kalshi.
    None means no pass has completed yet (page shows 'warming')."""
    ensure_background_refresh()
    with _kalshi_lock:
        return _kalshi_cache["markets"]


def _peek_markets_cached():
    """The cached market list WITHOUT arming the refresh thread (ledger
    reads, operator backfill)."""
    with _kalshi_lock:
        return _kalshi_cache["markets"]


def _discovery_status() -> dict:
    with _kalshi_lock:
        markets = _kalshi_cache["markets"]
        succeeded = markets is not None and _kalshi_cache["fetched_at_utc"] is not None
        age = (time.monotonic() - _kalshi_cache["fetched_at"]) if succeeded else None
        state = "warming" if markets is None else (
            "error" if _kalshi_cache["error"] and not succeeded else
            "stale" if age is not None and age > 2 * CACHE_TTL_S else "ok")
        return {
            "state": state,
            "fetched_at_utc": _kalshi_cache["fetched_at_utc"],
            "age_s": None if age is None else int(age),
            "series_counts": dict(_kalshi_cache["series_counts"]),
            "error": _kalshi_cache["error"],
            "refreshing": _kalshi_cache["refreshing"],
        }


# ── Public payload ────────────────────────────────────────────────────────────
_PAYLOAD_NOTE = (
    "Read-only analysis of public Kalshi prices. Model numbers are "
    "model-implied draft-position probabilities derived from each player's "
    "conformal pick interval (log-uniform mass assumption), renormalized so "
    "each event's top-N mass sums to at most N, and are shown only for "
    "top-N / over-under questions; exact-pick (#K overall) markets and every "
    "other relevant market are listed without a model number."
)


def edge_payload(prospects: list, dry_run: bool = False) -> dict:
    """Payload for GET /api/edge. Always 200; empty book → note 'seasonal';
    before the first discovery pass → note 'warming'.

    dry_run=True (operator --probe) never touches the git-tracked ledger:
    the rows that WOULD have been appended are returned under
    "ledger_dry_run" instead."""
    generated_at = datetime.now(timezone.utc).isoformat()
    markets = _get_markets_cached()
    status = _discovery_status()
    if markets is None:
        return {"generated_at": generated_at, "markets": [], "note": "warming",
                "discovery": status, "summary": _summary([])}
    if not markets:
        return {"generated_at": generated_at, "markets": [], "note": "seasonal",
                "discovery": status, "summary": _summary([])}

    board = _resolve_board(prospects, markets=markets)
    today = datetime.now(timezone.utc).date().isoformat()
    ledger_candidates = [dict(c, date=today) for c in board["ledger_candidates"]]
    # Record BEFORE reading calls so the very first response already carries
    # call{} and on_ledger for the tickers it just put on the ledger.
    if not dry_run:
        _record_ledger_entries(ledger_candidates)
    _maybe_reload_ledger()
    calls = _ledger_calls()
    out = _attach_calls(board["rows"], calls)
    payload = {
        "generated_at": generated_at,
        "markets": out,
        "note": _PAYLOAD_NOTE,
        "discovery": status,
        "summary": _summary(out, ledger_tickers=set(calls)),
    }
    if dry_run:
        payload["ledger_dry_run"] = ledger_candidates
    return payload


def _attach_calls(rows: list, calls: dict) -> list:
    """Shallow-copy the memoised rows with the per-ticker ledger `call`
    (the only per-request field — the ledger can change between refreshes)."""
    return [dict(r, call=calls.get(r["ticker"])) for r in rows]


def _resolve_board(prospects: list, markets: list = None) -> dict:
    """Match + price every cached market against the prospect list, once per
    (market list, prospect list) pair. Returns {rows, ledger_candidates,
    index} where rows are the public /api/edge rows (sorted, without `call`)
    and index maps ('nt', norm_name, norm_team) / ('name', norm_name) /
    ('slug', slug) / ('espn', espn_id) → [row, …] for O(1) player lookups."""
    if markets is None:
        markets = _get_markets_cached()
    markets = markets or []
    with _board_lock:
        memo = _BOARD_MEMO
        if memo["board"] is not None and memo["markets"] is markets and memo["prospects"] is prospects:
            return memo["board"]
    board = _build_board(markets, prospects)
    with _board_lock:
        _BOARD_MEMO.update({"markets": markets, "prospects": prospects, "board": board})
    return board


def _build_board(markets: list, prospects: list) -> dict:
    index = _build_prospect_index(prospects)
    out, ledger_candidates = [], []

    # Pass 1: match. Pass 2: per-event coherence scale over the matched set.
    resolved = []
    for src in markets:
        player, ambiguous = _resolve_player(
            src["_market_raw_names"], index,
            draft_year=src.get("draft_year"), team_hint=src.get("_team_hint", ""))
        resolved.append((src, player, ambiguous))
    scales = _event_scale_factors([(s, p) for s, p, _ in resolved if p])

    for src, player, ambiguous in resolved:
        question = _question_public(src.get("_question"))
        interval = _player_interval(player) if player else None
        model_interval = None
        if interval is not None:
            lo, hi, cov = interval
            model_interval = {"lo": int(lo), "hi": int(hi), "coverage": round(float(cov), 2)}
        row = {
            "ticker": src["ticker"],
            "title": src["title"],
            "event_ticker": src.get("event_ticker"),
            "series": src.get("series"),
            "yes_price_cents": src["yes_price_cents"],
            "yes_price_source": src.get("yes_price_source"),
            "spread_cents": src.get("spread_cents"),
            "ledger_eligible": bool(src.get("ledger_eligible")),
            "matched_player": player.get("name") if player else None,
            # team enables the frontend's /player/<name-team> profile links
            "matched_team": player.get("team") if player else None,
            "match_ambiguous": bool(ambiguous),
            "model_prob": None,
            "edge": None,
            "model_basis": None,
            "url": src["url"],
            # ── additive (futures spec api_contract) ──
            "question": question,
            "draft_year": src.get("draft_year"),
            "volume": src.get("volume"),
            "close_time": src.get("close_time"),
            "model_interval": model_interval,
            "event_scale": None,
            "wide_interval": bool(model_interval and model_interval["hi"] > 100),
            "matched_espn_id": _str_or_none(player.get("espn_id")) if player else None,
            "matched_espn_team_id": _str_or_none(player.get("espn_team_id")) if player else None,
            "matched_slug": _player_slug(player.get("name"), player.get("team")) if player else None,
        }
        if player:
            q = src.get("_question") or {}
            scale = 1.0
            if q.get("kind") in _PRICEABLE_KINDS and q.get("n") is not None:
                scale = scales.get((src.get("event_ticker") or src.get("ticker"), int(q["n"])), 1.0)
            row["model_prob"], row["edge"], row["model_basis"] = _model_fields(src, player, scale)
            if row["model_prob"] is not None:
                row["event_scale"] = round(float(scale), 4)
            if (row["edge"] is not None
                    and abs(row["edge"]) >= LEDGER_EDGE_THRESHOLD
                    and row["ledger_eligible"]):
                ledger_candidates.append({
                    "ticker": row["ticker"],
                    "title": row["title"],
                    "player": row["matched_player"],
                    "model_prob": row["model_prob"],
                    "market_price_cents": row["yes_price_cents"],
                    "price_source": row["yes_price_source"],
                    "edge": row["edge"],
                    "model_basis": row["model_basis"],
                    # identity/question frozen at record time (spec: ledger)
                    "team": row["matched_team"],
                    "slug": row["matched_slug"],
                    "espn_id": row["matched_espn_id"],
                    "espn_team_id": row["matched_espn_team_id"],
                    "event_ticker": row["event_ticker"],
                    "question": row["question"],
                    "model_interval": row["model_interval"],
                    "event_scale": row["event_scale"],
                    "close_time": row["close_time"],
                    "resolution": dict(_EMPTY_RESOLUTION),
                })
        else:
            q = (src.get("_question") or {}).get("kind")
            row["model_basis"] = (
                "not priced: player not in the prospect cache" if q in ("top_n", "pick_eq", "after_n")
                else "not priced: no matched player")
        out.append(row)

    # Matched-with-edge first (largest |edge|), then matched-no-edge, then rest.
    out.sort(key=lambda r: (
        0 if r["edge"] is not None else (1 if r["matched_player"] else 2),
        -(abs(r["edge"]) if r["edge"] is not None else 0),
        r["title"],
    ))

    lookup: dict = {}
    for r in out:
        if not r["matched_player"]:
            continue
        nm, tm = _norm_name(r["matched_player"]), _norm_team(r["matched_team"])
        for key in (("nt", nm, tm), ("name", nm), ("slug", r["matched_slug"]),
                    ("espn", r["matched_espn_id"])):
            if key[-1]:
                lookup.setdefault(key, []).append(r)
    return {"rows": out, "ledger_candidates": ledger_candidates, "index": lookup}


def _summary(rows: list, ledger_tickers=None) -> dict:
    priced = [r for r in rows if r.get("yes_price_cents") is not None]
    matched = [r for r in rows if r.get("matched_player")]
    edged = [r for r in rows if r.get("edge") is not None]
    eligible = [r for r in edged if abs(r["edge"]) >= LEDGER_EDGE_THRESHOLD and r.get("ledger_eligible")]
    clean = [r for r in eligible if not r.get("match_ambiguous")]
    by_kind = {"top_n": 0, "pick_eq": 0, "heisman": 0, "other": 0}
    years: dict = {}
    for r in rows:
        kind = (r.get("question") or {}).get("kind")
        by_kind[kind if kind in by_kind else "other"] += 1
        if r.get("draft_year"):
            years[r["draft_year"]] = years.get(r["draft_year"], 0) + 1
    ledger_tickers = ledger_tickers or set()
    return {
        "markets": len(rows),
        "priced": len(priced),
        "matched": len(matched),
        "modeled": len(edged),
        "edges_over_threshold": sum(1 for r in edged if abs(r["edge"]) >= LEDGER_EDGE_THRESHOLD),
        # ── additive ──
        "eligible_over_threshold": len(eligible),
        "on_ledger": sum(1 for r in rows if r.get("ticker") in ledger_tickers),
        "by_kind": by_kind,
        "draft_year": max(years, key=lambda y: (years[y], y)) if years else None,
        "top_gap_ticker": max(clean, key=lambda r: abs(r["edge"]))["ticker"] if clean else None,
    }


def _discovery_brief(status: dict) -> dict:
    return {"state": status.get("state"), "age_s": status.get("age_s"),
            "fetched_at_utc": status.get("fetched_at_utc")}


def _board_state(markets, status: dict) -> str:
    if markets is None:
        return "warming"
    if not markets:
        return "seasonal"
    return "ok"


CACHE_CONTROL_SHARED = "public, max-age=120"
CACHE_CONTROL_NONE = "no-cache"
_UNCACHEABLE_STATES = ("warming", "error")


def cache_control_for(payload) -> str:
    """Cache-Control for one /api/edge* payload: a warming or errored pass
    must not be shared for two minutes (pollers would keep seeing it), so
    those go out `no-cache`; everything else is `public, max-age=120`. Reads
    discovery.state and the top-level state/note, whichever is present."""
    if not isinstance(payload, dict):
        return CACHE_CONTROL_NONE
    disc = payload.get("discovery")
    states = {payload.get("state"), payload.get("note")}
    if isinstance(disc, dict):
        states.add(disc.get("state"))
    if states & set(_UNCACHEABLE_STATES):
        return CACHE_CONTROL_NONE
    return CACHE_CONTROL_SHARED


def player_payload(prospects: list, name: str = None, team: str = None,
                   slug: str = None, espn_id: str = None) -> dict:
    """GET /api/edge/player — every market row matched to one player.

    Lookup is by (name, team) — both normalised the way the matcher
    normalises them — or by the player-page slug or ESPN id. A bare name is
    honoured only when it maps to exactly one player on the board (two
    'Jeremiah Smith's exist), otherwise it is 'no_markets'. Always HTTP 200;
    the UI renders nothing for 'no_markets' | 'warming' | 'seasonal'."""
    markets = _get_markets_cached()
    status = _discovery_status()
    state = _board_state(markets, status)
    base = {"state": state, "as_of": status.get("fetched_at_utc"),
            "discovery": _discovery_brief(status), "player": None, "markets": [],
            "on_ledger": False, "ledger_since": None}
    if state != "ok":
        return base
    board = _resolve_board(prospects, markets=markets)
    lookup = board["index"]
    rows = []
    if slug:
        rows = lookup.get(("slug", _player_slug_key(slug)), [])
    elif espn_id:
        rows = lookup.get(("espn", str(espn_id).strip()), [])
    elif name and team:
        rows = lookup.get(("nt", _norm_name(name), _norm_team(team)), [])
    elif name:
        cand = lookup.get(("name", _norm_name(name)), [])
        if len({(r["matched_player"], r["matched_team"]) for r in cand}) == 1:
            rows = cand
    if not rows:
        return dict(base, state="no_markets")
    _maybe_reload_ledger()
    calls = _ledger_calls()
    rows = sorted(rows, key=lambda r: (
        _kind_rank(r),
        -(abs(r["edge"]) if r["edge"] is not None else -1),
        -(r["yes_price_cents"] if r["yes_price_cents"] is not None else -1),
        r["title"]))
    first = rows[0]
    since = [calls[r["ticker"]]["first_date"] for r in rows
             if r["ticker"] in calls and calls[r["ticker"]]["first_date"]]
    return dict(base, player={
        "name": first["matched_player"],
        "team": first["matched_team"],
        "espn_id": first["matched_espn_id"],
        "espn_team_id": first["matched_espn_team_id"],
        "slug": first["matched_slug"],
    }, markets=_attach_calls(rows, calls), on_ledger=bool(since),
        ledger_since=min(since) if since else None)


def _player_slug_key(slug: str) -> str:
    """Slugs are already in the slug alphabet; re-applying the port makes
    '?slug=Arch Manning-Texas' and a URL-decoded slug both hit."""
    s = str(slug or "").strip().lower()
    if re.fullmatch(r"[a-z0-9-]+", s):
        return s
    return _player_slug(s, "").rstrip("-")


_SPOTLIGHT_SOURCES = ("last", "mid")


def _spotlight_row(r: dict) -> dict:
    return {
        "ticker": r["ticker"],
        "player": r["matched_player"],
        "team": r["matched_team"],
        "slug": r["matched_slug"],
        "espn_id": r["matched_espn_id"],
        "espn_team_id": r["matched_espn_team_id"],
        "question": r["question"],
        "market": r["yes_price_cents"],
        "price_source": r["yes_price_source"],
        "model": r["model_prob"],
        "gap": r["edge"],
        "wide_interval": bool(r.get("wide_interval")),
        "ledger_eligible": bool(r.get("ledger_eligible")),
        "match_ambiguous": bool(r.get("match_ambiguous")),
        "url": r["url"],
    }


def _record_from_ledger() -> dict:
    """The home-band scoreboard: counts from the ledger, all resolution
    fields null until draft night."""
    _maybe_reload_ledger()
    markets = _peek_markets_cached() or []
    summary, scoring = _ledger_summary(_LEDGER["entries"], {m["ticker"]: m for m in markets})
    return {
        "calls": summary["calls"],
        "open": summary["open"],
        "resolved": summary["resolved"],
        "scored": summary["scored"],
        "void": summary["void"],
        "model_closer": summary["model_closer"],
        "market_closer": summary["market_closer"],
        "brier": summary["model_brier"],
        "model_brier": summary["model_brier"],
        "market_brier": summary["market_brier"],
        "settles_on": scoring["settles_on"],
        "status": scoring["status"],
    }


def spotlight_payload(prospects: list, n: int = 3) -> dict:
    """GET /api/edge/spotlight?n= — the home band. Rows: model_prob != null
    AND ledger_eligible AND !match_ambiguous, |edge| desc, top n; headline =
    rows[0]. Short → fill from modeled, unambiguous rows on a `last`/`mid`
    quote (never mid_wide/ask/bid: a one-sided or placeholder book must not
    become the screenshot). n clamped to 1..6."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 3
    n = max(1, min(n, 6))
    markets = _get_markets_cached()
    status = _discovery_status()
    state = _board_state(markets, status)
    record = _record_from_ledger()
    if state != "ok":
        return {"state": state, "as_of": status.get("fetched_at_utc"),
                "discovery": _discovery_brief(status), "headline": None, "rows": [],
                "summary": {"markets": 0, "modeled": 0, "disagreements": 0,
                            "on_ledger": 0, "draft_year": None},
                "record": record}
    board = _resolve_board(prospects, markets=markets)
    _maybe_reload_ledger()
    calls = _ledger_calls()
    rows = board["rows"]
    modeled = [r for r in rows if r["model_prob"] is not None and not r["match_ambiguous"]]
    primary = sorted((r for r in modeled if r["ledger_eligible"]),
                     key=lambda r: -abs(r["edge"]))
    chosen = primary[:n]
    if len(chosen) < n:
        taken = {r["ticker"] for r in chosen}
        fill = sorted((r for r in modeled if r["ticker"] not in taken
                       and r["yes_price_source"] in _SPOTLIGHT_SOURCES),
                      key=lambda r: -abs(r["edge"]))
        chosen.extend(fill[:n - len(chosen)])
    out_rows = [_spotlight_row(r) for r in chosen]
    s = _summary(rows, ledger_tickers=set(calls))
    return {
        "state": state,
        "as_of": status.get("fetched_at_utc"),
        "discovery": _discovery_brief(status),
        "headline": out_rows[0] if out_rows else None,
        "rows": out_rows,
        "summary": {"markets": s["markets"], "modeled": s["modeled"],
                    "disagreements": s["eligible_over_threshold"],
                    "on_ledger": s["on_ledger"], "draft_year": s["draft_year"]},
        "record": record,
    }


# ── CLI (operator utilities; GET-only) ───────────────────────────────────────
def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true",
                    help="run one discovery pass and print series/match/edge counts")
    ap.add_argument("--sync-ledger", metavar="BASE_URL",
                    help="GET <BASE_URL>/api/edge/ledger and merge into the local ledger")
    ap.add_argument("--backfill-ledger", action="store_true",
                    help="fill identity/question/interval fields on existing ledger "
                         "entries from the current cache (idempotent; one discovery GET)")
    ap.add_argument("--score-ledger", action="store_true",
                    help="GET /markets/{ticker} for every open ledger call and fill "
                         "resolution + Brier once settled (idempotent; no-op pre-draft)")
    ap.add_argument("--sync-history", metavar="BASE_URL",
                    help="GET <BASE_URL>/api/edge/history per remote ticker and merge into "
                         "the local price history")
    ap.add_argument("--cache", default="training_data/prospect_cache.json",
                    help="prospect cache to match against (with --probe / --backfill-ledger)")
    args = ap.parse_args(argv)
    os.environ[_NO_WARMUP_ENV] = "1"

    def _load_prospects():
        if not os.path.exists(args.cache):
            return []
        with open(args.cache) as f:
            data = json.load(f)
        return data.get("prospects", []) if isinstance(data, dict) else data

    if args.backfill_ledger:
        _refresh_markets_once()  # one GET pass; off-season the parser fallback runs
        status = _discovery_status()
        changed = backfill_ledger(_load_prospects())
        total = len(_LEDGER["entries"])
        print(f"ledger backfill (discovery {status['state']}): {changed} of {total} "
              f"entr{'y' if total == 1 else 'ies'} enriched in {LEDGER_PATH}")
    if args.sync_ledger:
        # Fly scales to zero; the first request can take ~20 s to wake the
        # machine, hence the generous timeout. A failed sync leaves the
        # committed ledger as-is (merge is idempotent), so callers may treat
        # exit 2 as non-fatal.
        try:
            added = sync_ledger_from(args.sync_ledger)
        except Exception as exc:
            print(f"ledger sync failed ({exc}); {LEDGER_PATH} left unchanged")
            return 2
        print(f"ledger sync: {added} new row(s) merged into {LEDGER_PATH}")
    if args.score_ledger:
        try:
            c = score_ledger()
        except Exception as exc:
            print(f"ledger scoring failed ({exc}); {LEDGER_PATH} left unchanged")
            return 2
        print(f"ledger score: {c['tickers']} open ticker(s) checked — {c['scored']} scored, "
              f"{c['void']} void, {c['open']} still open, {c['errors']} error(s)")
    if args.sync_history:
        try:
            added = sync_history_from(args.sync_history)
        except Exception as exc:
            print(f"history sync failed ({exc}); {HISTORY_PATH} left unchanged")
            return 2
        print(f"history sync: {added} new point(s) merged into {HISTORY_PATH}")
    if args.probe:
        t0 = time.monotonic()
        _refresh_markets_once()
        status = _discovery_status()
        print(f"discovery: {status['state']} in {time.monotonic() - t0:.1f}s; "
              f"series_counts={json.dumps(status['series_counts'])}")
        prospects = _load_prospects()
        # dry_run: --probe is a read-only diagnostic and must never write the
        # git-tracked ledger (only the serving process records calls).
        payload = edge_payload(prospects, dry_run=True)
        print(f"payload: note={payload['note']!r} summary={json.dumps(payload['summary'])}")
        spot = spotlight_payload(prospects, n=3)
        head = spot.get("headline")
        print(f"spotlight: state={spot['state']} headline="
              + (f"{head['player']} {head['market']}c vs {head['model']}% ({head['ticker']})"
                 if head else "none")
              + f" record={json.dumps(spot['record'])}")
        for r in payload["markets"]:
            if r["edge"] is not None:
                print(f"  edge {r['edge']:+6.1f}  {r['yes_price_cents']:>3}c/{r['yes_price_source']:<8} "
                      f"model {r['model_prob']:5.1f}  {r['ticker']}  {r['matched_player']}")
        would = payload.get("ledger_dry_run") or []
        print(f"ledger (dry run, nothing written): {len(would)} row(s) would be appended")
        for e in would:
            print(f"  {e['date']}  edge {e['edge']:+6.1f}  {e['market_price_cents']:>3}c/{e['price_source']:<8} "
                  f"model {e['model_prob']:5.1f}  {e['ticker']}  {e['player']}")
    return 0


_load_ledger()
_load_history()
if __name__ == "__main__":
    sys.exit(_cli())
else:
    ensure_background_refresh()
