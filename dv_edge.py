"""Kalshi market-edge board — read-only analysis of public prediction-market
prices vs DraftVision model output.

Wired from XGBOost.py via:

    import dv_edge
    dv_edge.edge_payload(_PROSPECT_CACHE)   # -> GET /api/edge
    dv_edge.ledger_payload()                # -> GET /api/edge/ledger

Design constraints (deliberate):
- READ-ONLY: we hit only Kalshi's PUBLIC market-data endpoints (no auth, no
  order placement — we never take or facilitate bets). This page is analysis
  of CFTC-regulated market prices, nothing more.
- Gentle on the API: discovery paginates GET /events?status=open (with nested
  markets) a bounded number of pages, one attempt each, short timeouts, and
  the result — success OR failure — is cached in-memory for ~10 minutes, so a
  burst of page loads produces at most one upstream fetch.
- Fail-soft: any upstream failure (network, schema drift, rate limit) or an
  off-season empty book degrades to {"markets": [], "note": "seasonal"} with
  HTTP 200. The frontend renders a friendly empty state; nothing 500s.

── Honest-mapping policy (read this before "improving" the edge math) ─────────
The prospect cache rows carry exactly two model outputs:
  * success_probability  — calibrated P(NFL success), 0-100
  * draft_grade / draft_grade_class — argmax bucket over
    ["Top 50 Pick", "Day 2 Pick", "Late Round Pick", "Undrafted Prospect"]
Per-bucket posteriors are NOT persisted, so most market questions cannot be
honestly priced by the model. We therefore only compute an `edge` when the
market's question genuinely corresponds to something the cache answers:

  * "Drafted in the Top X" markets with 32 <= X <= 50 map to our "Top 50 Pick"
    bucket. When the matched player's bucket IS "Top 50 Pick" we use
    success_probability as the model's confidence proxy for that bucket
    (it is the calibrated score that drives the bucket assignment). This is a
    proxy, and the UI copy says so.
  * Top-3 / Top-10 style markets: our Top-50 bucket cannot distinguish pick 5
    from pick 45 → model fields stay null.
  * "Will player be drafted at all" markets (e.g. KXNFLDRAFTWR): the cache
    does not persist P(drafted), and success_probability is a different
    quantity → null.
  * Season stat markets (e.g. KXNCAAFTEAMRECTD receiving TDs): listed for
    context, never priced → null.

Unmatched / unmappable markets are still listed with null model fields, so
the board never pretends to know more than it does.

── Paper ledger ───────────────────────────────────────────────────────────────
When a matched market shows |edge| >= 10 points, we append one row per
(ticker, day) to training_data/edge_ledger.json. No money moves — the ledger
exists purely to build a verifiable public track record of the model's calls
BEFORE anyone is asked to trust an edge. Atomic write (temp + os.replace) and
mtime hot-reload, same pattern as the other training_data caches.
"""

import json
import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone

import requests

# ── Config ────────────────────────────────────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
LEDGER_PATH = "training_data/edge_ledger.json"

CACHE_TTL_S = 600          # ~10 min in-memory cache of the discovery result
REQUEST_TIMEOUT_S = 8      # per-page HTTP timeout
MAX_EVENT_PAGES = 6        # hard bound on pagination — never hammer the API
PAGE_LIMIT = 200           # Kalshi max page size
INTER_REQUEST_DELAY_S = 0.6  # Kalshi 429s rapid-fire pagination; pace ourselves

# Verified-real series families probed directly via /markets?series_ticker= —
# guarantees the draft books are found even if they sit deep in /events
# pagination (the public API caps how far we can politely walk).
_KNOWN_SERIES = ("KXNFLDRAFTWR", "KXNFLSDRAFTTOP", "KXNCAAFTEAMRECTD")

LEDGER_EDGE_THRESHOLD = 10.0  # points of |edge| before a paper-ledger entry

# Series families verified to exist (relevance seeds, matched as prefixes):
#   KXNFLDRAFTWR   "Pro Football Wide Receiver Drafted"
#   KXNFLSDRAFTTOP "Will player be drafted Top X"
#   KXNCAAFTEAMRECTD  college receiving TDs
# NOTE: deliberately NOT a bare "KXNCAAF" prefix — that family also contains
# team/game markets (spreads, winners) which are not player markets and would
# flood the board once the season starts. NCAAF relevance instead comes from
# the specific stat-series prefix plus the player-stat title patterns below.
_SERIES_PREFIXES = ("KXNFLDRAFT", "KXNFLSDRAFT", "KXNCAAFTEAMREC")

# Title fallback for series we have not seen yet — draft-question or
# college-football player-stat phrasing.
_TITLE_RELEVANCE_RE = re.compile(
    r"\bnfl draft\b|\bbe drafted\b|\bdrafted (?:in|by|top)\b|\bdraft pick\b"
    r"|\bheisman\b|\b(?:receiving|rushing|passing) (?:yards|tds|touchdowns)\b",
    re.I,
)

# ── In-memory Kalshi cache ────────────────────────────────────────────────────
_kalshi_lock = threading.Lock()
_kalshi_cache: dict = {"fetched_at": 0.0, "markets": None}  # markets: list|None

# ── Paper ledger (atomic write + mtime hot-reload) ────────────────────────────
_ledger_lock = threading.Lock()
_LEDGER: dict = {"entries": []}
_LEDGER_MTIME: float = 0.0


def _load_ledger() -> None:
    global _LEDGER, _LEDGER_MTIME
    if not os.path.exists(LEDGER_PATH):
        return
    try:
        mtime = os.path.getmtime(LEDGER_PATH)  # capture BEFORE reading
        with open(LEDGER_PATH) as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        _LEDGER = {"entries": [e for e in entries if isinstance(e, dict)]}
        _LEDGER_MTIME = mtime
    except Exception as exc:  # file mid-rewrite etc. — retried next request
        print(f"Edge ledger load failed (will retry): {exc}")


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
    tmp_path = LEDGER_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, LEDGER_PATH)


def _record_ledger_entries(candidates: list) -> None:
    """Append paper-ledger rows, deduped to one per (ticker, day).

    Best-effort: a write failure is logged and swallowed — the board must
    render regardless.
    """
    if not candidates:
        return
    with _ledger_lock:
        try:
            _load_ledger()  # merge against whatever is on disk right now
            seen = {(e.get("ticker"), e.get("date")) for e in _LEDGER["entries"]}
            added = False
            for row in candidates:
                key = (row["ticker"], row["date"])
                if key in seen:
                    continue
                _LEDGER["entries"].append(row)
                seen.add(key)
                added = True
            if added:
                _write_ledger_atomic({"entries": _LEDGER["entries"]})
                global _LEDGER_MTIME
                _LEDGER_MTIME = os.path.getmtime(LEDGER_PATH)
        except Exception as exc:
            print(f"Edge ledger append failed: {exc}")


def ledger_payload() -> dict:
    _maybe_reload_ledger()
    entries = sorted(
        _LEDGER["entries"],
        key=lambda e: (str(e.get("date") or ""), str(e.get("ticker") or "")),
        reverse=True,
    )
    return {
        "entries": entries,
        "note": (
            "Paper ledger only — no positions are ever taken. Rows are "
            "recorded automatically when the model and the market disagree "
            "by 10+ points, to build a verifiable public track record."
        ),
    }


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


def _build_prospect_index(prospects: list) -> dict:
    """normalized name -> cache row. Ambiguous names (two different players
    normalizing identically) are refused outright — a wrong match is worse
    than no match on a page about trust."""
    index: dict = {}
    ambiguous: set = set()
    for row in prospects or []:
        key = _norm_name(row.get("name", ""))
        if not key or key in ambiguous:
            continue
        if key in index:
            if index[key].get("team") != row.get("team"):
                ambiguous.add(key)
                index.pop(key)
            continue  # same player duplicated across rows — keep the first
        index[key] = row
    return index


# Title shapes seen on draft markets; ordered most- to least-specific.
_NAME_PATTERNS = (
    re.compile(r"\bwill\s+(.+?)\s+be\s+(?:drafted|selected|picked)\b", re.I),
    re.compile(r"^(.+?)\s+(?:drafted|selected)\s+(?:in|by|top)\b", re.I),
    re.compile(r"\bdrafted:\s*(.+?)\s*\??$", re.I),
)


def _candidate_names(event_title: str, market: dict) -> list:
    """Best-effort player-name candidates from a market. Kalshi multi-market
    events usually put the player in yes_sub_title/subtitle; single-market
    events phrase it in the title. Candidates are only *candidates* — the
    prospect index lookup is the real filter."""
    cands = []
    for key in ("yes_sub_title", "subtitle"):
        v = (market.get(key) or "").strip()
        if v:
            cands.append(v)
    for text in ((market.get("title") or ""), (event_title or "")):
        for pat in _NAME_PATTERNS:
            m = pat.search(text)
            if m:
                cands.append(m.group(1))
        # Last resort: capitalized First Last(-ish) runs inside the title.
        cands.extend(re.findall(r"\b([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,2})\b", text))
    return cands


# ── Market-question parsing (what can the model honestly answer?) ─────────────
_TOP_N_TITLE_RE = re.compile(r"\btop[\s\-]*(\d{1,3})\b", re.I)
_TOP_N_TICKER_RE = re.compile(r"TOP(\d{1,3})|-T(\d{1,3})(?:$|-)")


def _extract_top_n(ticker: str, title: str):
    m = _TOP_N_TITLE_RE.search(title or "")
    if m:
        return int(m.group(1))
    m = _TOP_N_TICKER_RE.search((ticker or "").upper())
    if m:
        return int(m.group(1) or m.group(2))
    if re.search(r"\b(?:first|1st)\s+round\b", title or "", re.I):
        return 32
    return None


def _model_fields(market_row: dict, player: dict):
    """Return (model_prob, edge) or (None, None).

    See the module docstring: an edge is computed ONLY for top-X markets with
    32 <= X <= 50 where the matched player's bucket is "Top 50 Pick" — the one
    question the cached outputs genuinely answer. success_probability serves
    as the bucket-confidence proxy (per-bucket posteriors are not persisted).
    """
    top_n = _extract_top_n(market_row.get("ticker", ""), market_row.get("title", ""))
    if top_n is None or not (32 <= top_n <= 50):
        return None, None
    if player.get("draft_grade_class") != 0:  # bucket is not "Top 50 Pick"
        return None, None
    prob = player.get("success_probability")
    price = market_row.get("yes_price_cents")
    if prob is None or price is None:
        return None, None
    model_prob = round(float(prob), 1)
    edge = round(model_prob - float(price), 1)
    return model_prob, edge


# ── Kalshi discovery ──────────────────────────────────────────────────────────
def _yes_price_cents(m: dict):
    """Best available YES price in cents: last trade, else bid/ask midpoint,
    else whichever side exists. None when the book is empty."""
    last = m.get("last_price")
    if isinstance(last, (int, float)) and 0 < last < 100:
        return int(last)
    bid, ask = m.get("yes_bid"), m.get("yes_ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and 0 < bid and ask < 100:
        return int(round((bid + ask) / 2))
    for v in (ask, bid):
        if isinstance(v, (int, float)) and 0 < v < 100:
            return int(v)
    return None


def _is_relevant(series_ticker: str, event_ticker: str, title: str) -> bool:
    tick = (series_ticker or event_ticker or "").upper()
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
    return {
        "ticker": m.get("ticker") or event_ticker,
        "title": m_title,
        "event_title": event_title,
        "yes_price_cents": _yes_price_cents(m),
        "url": _market_url(series, event_ticker),
        "_market_raw_names": _candidate_names(event_title, m),
    }


def _fetch_relevant_markets() -> list:
    """Bounded, politely paced discovery pass.

    Two stages, each fail-soft (Kalshi 429s rapid-fire pagination, so a
    mid-pass rate limit keeps whatever was already collected instead of
    discarding the whole pass):
      1. Direct probe of the known draft series via /markets?series_ticker=.
      2. A capped walk of /events?status=open (nested markets) to discover
         series families we don't know about yet.
    Raises only if BOTH stages produced nothing AND an error occurred —
    the caller maps that to the seasonal empty state anyway.
    """
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    rows, failed = [], False

    # Stage 1 — known series, one page each.
    for series in _KNOWN_SERIES:
        try:
            resp = session.get(
                f"{KALSHI_BASE}/markets",
                params={"series_ticker": series, "status": "open", "limit": PAGE_LIMIT},
                timeout=REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            for m in resp.json().get("markets", []) or []:
                rows.append(_market_row(m, "", series, m.get("event_ticker") or ""))
        except Exception as exc:
            failed = True
            print(f"Kalshi series probe {series} failed: {exc}")
        time.sleep(INTER_REQUEST_DELAY_S)

    # Stage 2 — general event walk (bounded pages, paced).
    cursor = None
    for _ in range(MAX_EVENT_PAGES):
        try:
            params = {"status": "open", "limit": PAGE_LIMIT,
                      "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            resp = session.get(f"{KALSHI_BASE}/events", params=params,
                               timeout=REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # 429/network — keep what we have
            failed = True
            print(f"Kalshi event walk stopped early: {exc}")
            break
        for event in data.get("events", []) or []:
            e_title = event.get("title") or ""
            series = event.get("series_ticker") or ""
            e_tick = event.get("event_ticker") or ""
            if not _is_relevant(series, e_tick, e_title):
                continue
            for m in event.get("markets", []) or []:
                rows.append(_market_row(m, e_title, series, e_tick))
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(INTER_REQUEST_DELAY_S)

    if not rows and failed:
        raise RuntimeError("Kalshi discovery produced nothing and errored")

    # Dedupe (a known-series market can also surface in the event walk).
    seen, unique = set(), []
    for r in rows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        unique.append(r)
    return unique


def _get_markets_cached() -> list:
    """Discovery result (possibly empty) under a ~10-min in-memory cache.
    Failures are cached too, so an outage can't turn into a request storm."""
    now = time.monotonic()
    with _kalshi_lock:
        if (_kalshi_cache["markets"] is not None
                and now - _kalshi_cache["fetched_at"] < CACHE_TTL_S):
            return _kalshi_cache["markets"]
        try:
            markets = _fetch_relevant_markets()
        except Exception as exc:  # network/HTTP/schema — degrade to seasonal
            print(f"Kalshi fetch failed (serving seasonal state): {exc}")
            markets = []
        _kalshi_cache["markets"] = markets
        _kalshi_cache["fetched_at"] = now
        return markets


# ── Public payload ────────────────────────────────────────────────────────────
def edge_payload(prospects: list) -> dict:
    """Payload for GET /api/edge. Always 200; empty book → note 'seasonal'."""
    generated_at = datetime.now(timezone.utc).isoformat()
    markets = _get_markets_cached()
    if not markets:
        return {"generated_at": generated_at, "markets": [], "note": "seasonal"}

    index = _build_prospect_index(prospects)
    out, ledger_candidates = [], []
    today = datetime.now(timezone.utc).date().isoformat()
    for src in markets:
        player = None
        for cand in src["_market_raw_names"]:
            player = index.get(_norm_name(cand))
            if player:
                break
        row = {
            "ticker": src["ticker"],
            "title": src["title"],
            "yes_price_cents": src["yes_price_cents"],
            "matched_player": player.get("name") if player else None,
            # team enables the frontend's /player/<name-team> profile links
            "matched_team": player.get("team") if player else None,
            "model_prob": None,
            "edge": None,
            "url": src["url"],
        }
        if player:
            row["model_prob"], row["edge"] = _model_fields(src, player)
            if row["edge"] is not None and abs(row["edge"]) >= LEDGER_EDGE_THRESHOLD:
                ledger_candidates.append({
                    "date": today,
                    "ticker": row["ticker"],
                    "title": row["title"],
                    "player": row["matched_player"],
                    "model_prob": row["model_prob"],
                    "market_price_cents": row["yes_price_cents"],
                    "edge": row["edge"],
                })
        out.append(row)

    # Matched-with-edge first (largest |edge|), then matched-no-edge, then rest.
    out.sort(key=lambda r: (
        0 if r["edge"] is not None else (1 if r["matched_player"] else 2),
        -(abs(r["edge"]) if r["edge"] is not None else 0),
        r["title"],
    ))
    _record_ledger_entries(ledger_candidates)
    return {
        "generated_at": generated_at,
        "markets": out,
        "note": (
            "Read-only analysis of public Kalshi prices. Edges appear only "
            "where a market question maps directly onto the model's Top-50 "
            "bucket; everything else is listed without a model number."
        ),
    }


_load_ledger()
