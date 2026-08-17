"""DraftVision usage analytics + login-aware entitlement scaffolding.

Wired from XGBOost.py via:

    import dv_analytics
    dv_analytics.init_app(app, get_conn=_get_conn, placeholder=_placeholder,
                          use_postgres=USE_POSTGRES)

Design constraints (deliberate):
- Privacy-light: NO raw IPs, NO user agents are ever stored. A visitor is
  identified only by an Auth0 `sub` (when logged in) or the frontend's
  self-assigned `X-DV-Anon` uuid (see src/lib/api.js).
- Fail-open: an analytics/auth failure must never surface as a 500 to users.
  Every hook body is wrapped; worst case we simply don't log the event.
- Zero added request latency: events go into an in-memory queue flushed by a
  background thread (every FLUSH_MAX_EVENTS events or FLUSH_INTERVAL_S
  seconds), plus an atexit flush.

Auth (optional-on): when AUTH0_DOMAIN + AUTH0_AUDIENCE are set, Bearer tokens
are verified against Auth0's JWKS (cached 1h) using PyJWT[crypto]. A valid
token upserts a `users` row and attaches g.user_sub. Invalid or absent tokens
mean "anonymous" — public routes never 401.

── Eventual Stripe flow (documented, NOT implemented) ─────────────────────────
The `users.tier` column is the single source of truth for entitlements.
Planned flow:
  1. Frontend calls POST /api/billing/checkout (auth required) → backend
     creates a Stripe Checkout Session (mode=subscription, price=PRO_PRICE_ID,
     client_reference_id=<auth0 sub>) and returns the session URL.
  2. Stripe redirects the user back; entitlement does NOT flip here.
  3. A Stripe webhook (checkout.session.completed / customer.subscription.
     updated|deleted), verified via STRIPE_WEBHOOK_SECRET, maps the event to
     the auth0 sub and sets users.tier='pro' (or back to 'free' on
     cancellation/expiry).
  4. require_tier('pro') then gates premium routes automatically.
Until then, tiers are flipped by hand:  UPDATE users SET tier='pro' WHERE sub=...;
───────────────────────────────────────────────────────────────────────────────
"""

import atexit
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import g, jsonify, request

try:
    import jwt as pyjwt  # PyJWT[crypto]
    from jwt import PyJWKClient
    _JWT_AVAILABLE = True
except Exception:  # pragma: no cover — auth simply stays off
    _JWT_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").strip()
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "").strip()
AUTH_ENABLED = bool(AUTH0_DOMAIN and AUTH0_AUDIENCE and _JWT_AVAILABLE)

FLUSH_MAX_EVENTS = 20
FLUSH_INTERVAL_S = 10.0
JWKS_CACHE_LIFESPAN_S = 3600  # 1h
USER_UPSERT_THROTTLE_S = 300  # refresh last_seen at most every 5 min per sub

TRACKED_EXACT = {"/predict", "/search", "/init"}
UNTRACKED_PREFIXES = ("/api/analytics/",)  # don't let admin polling pollute stats

_TIER_RANK = {"free": 0, "pro": 1}

# DB layer handed to us by XGBOost.py (SQLite locally, Postgres in prod).
_get_conn = None
_placeholder = None
_use_postgres = False

# ── Event queue (flushed off the request path) ────────────────────────────────
_queue = []            # items: ("event", (...cols)) | ("user", (sub, email, name, now_iso))
_queue_lock = threading.Lock()
_queue_wakeup = threading.Event()
_flusher_started = False

_user_last_upsert = {}  # sub -> monotonic seconds of last queued upsert
_jwk_client = None
_jwk_client_lock = threading.Lock()


def _now_iso():
    """UTC ISO-8601 with a lexicographically sortable 'YYYY-MM-DDTHH:MM:SS' prefix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── Schema ────────────────────────────────────────────────────────────────────
def _init_schema():
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                sub TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                tier TEXT NOT NULL DEFAULT 'free',
                created_at TEXT,
                last_seen TEXT
            )
            """
        )
        id_col = ("id SERIAL PRIMARY KEY" if _use_postgres
                  else "id INTEGER PRIMARY KEY AUTOINCREMENT")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS events (
                {id_col},
                ts TEXT,
                route TEXT,
                method TEXT,
                status INTEGER,
                user_sub TEXT,
                anon_id TEXT,
                dur_ms INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ── Flush worker ──────────────────────────────────────────────────────────────
def _flush_once():
    with _queue_lock:
        if not _queue:
            return
        batch, _queue[:] = _queue[:], []

    events = [item[1] for item in batch if item[0] == "event"]
    users = {}  # sub -> (email, name, ts): last write wins, one upsert per sub
    for kind, payload in batch:
        if kind == "user":
            users[payload[0]] = payload

    ph = _placeholder()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        if events:
            cur.executemany(
                f"INSERT INTO events (ts, route, method, status, user_sub, anon_id, dur_ms) "
                f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
                events,
            )
        for sub, email, name, now in users.values():
            cur.execute(
                f"""
                INSERT INTO users (sub, email, name, created_at, last_seen)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                ON CONFLICT (sub) DO UPDATE SET
                    email = COALESCE(EXCLUDED.email, users.email),
                    name = COALESCE(EXCLUDED.name, users.name),
                    last_seen = EXCLUDED.last_seen
                """,
                (sub, email, name, now, now),
            )
        conn.commit()
    finally:
        conn.close()


def _flush_safe():
    try:
        _flush_once()
    except Exception as exc:  # fail-open: drop the batch, keep serving
        print(f"[dv_analytics] flush failed (dropped batch): {exc}")


def _flusher_loop():
    while True:
        _queue_wakeup.wait(timeout=FLUSH_INTERVAL_S)
        _queue_wakeup.clear()
        _flush_safe()


def _enqueue(item):
    with _queue_lock:
        _queue.append(item)
        should_wake = len(_queue) >= FLUSH_MAX_EVENTS
    if should_wake:
        _queue_wakeup.set()


# ── Auth (Auth0 JWKS via PyJWT) ───────────────────────────────────────────────
def _get_jwk_client():
    global _jwk_client
    if _jwk_client is None:
        with _jwk_client_lock:
            if _jwk_client is None:
                _jwk_client = PyJWKClient(
                    f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
                    cache_jwk_set=True,
                    lifespan=JWKS_CACHE_LIFESPAN_S,
                )
    return _jwk_client


def _resolve_user():
    """Verify a Bearer token if present. Anonymous on any failure — never 401s."""
    if not AUTH_ENABLED:
        return
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return
    token = auth_header[7:].strip()
    if not token:
        return
    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        claims = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
    except Exception:
        return  # invalid/expired token == anonymous
    sub = claims.get("sub")
    if not sub:
        return
    g.user_sub = sub
    now_mono = time.monotonic()
    last = _user_last_upsert.get(sub)
    if last is None or now_mono - last >= USER_UPSERT_THROTTLE_S:
        _user_last_upsert[sub] = now_mono
        _enqueue(("user", (sub, claims.get("email"), claims.get("name"), _now_iso())))


# ── Entitlement scaffold (paywall-ready, applied to nothing yet) ──────────────
def get_user_tier(sub):
    """Read the tier for a signed-in user. 'free' when unknown."""
    if not sub:
        return "free"
    ph = _placeholder()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT tier FROM users WHERE sub = {ph}", (sub,))
        row = cur.fetchone()
        return (row[0] or "free") if row else "free"
    finally:
        conn.close()


def require_tier(tier):
    """Decorator: 402 {"error": "upgrade_required"} unless the caller's tier
    meets `tier`. Not applied to any route yet — see the Stripe flow note in
    the module docstring for how tiers will eventually get set."""
    required_rank = _TIER_RANK.get(tier, 99)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            sub = getattr(g, "user_sub", None)
            try:
                current_rank = _TIER_RANK.get(get_user_tier(sub), 0)
            except Exception:
                current_rank = 0
            if current_rank < required_rank:
                return jsonify({"error": "upgrade_required"}), 402
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── Request hooks ─────────────────────────────────────────────────────────────
def _should_track(path):
    if path.startswith(UNTRACKED_PREFIXES):
        return False
    return path.startswith("/api/") or path in TRACKED_EXACT


def _before_request():
    try:
        g._dv_t0 = time.perf_counter()
        _resolve_user()
    except Exception as exc:
        print(f"[dv_analytics] before_request error (ignored): {exc}")


def _after_request(response):
    try:
        path = request.path
        if _should_track(path):
            t0 = getattr(g, "_dv_t0", None)
            dur_ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else None
            anon_id = (request.headers.get("X-DV-Anon") or "").strip()[:64] or None
            _enqueue((
                "event",
                (_now_iso(), path, request.method, response.status_code,
                 getattr(g, "user_sub", None), anon_id, dur_ms),
            ))
    except Exception as exc:
        print(f"[dv_analytics] after_request error (ignored): {exc}")
    return response


# ── Insight endpoint ──────────────────────────────────────────────────────────
def _visitor_expr():
    return "COALESCE(user_sub, anon_id)"


def _analytics_summary():
    # Invisible until ANALYTICS_KEY is configured; wrong key is also a 404 so
    # the endpoint's existence can't be probed.
    configured = os.getenv("ANALYTICS_KEY", "").strip()
    if not configured:
        return jsonify({"error": "not found"}), 404
    supplied = request.args.get("key") or request.headers.get("X-Analytics-Key") or ""
    if supplied != configured:
        return jsonify({"error": "not found"}), 404

    ph = _placeholder()
    now = datetime.now(timezone.utc)
    day_start = now.strftime("%Y-%m-%d")             # ts is ISO text: prefix-comparable
    cut_7d = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    cut_30d = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    vis = _visitor_expr()

    conn = _get_conn()
    try:
        cur = conn.cursor()

        def distinct_visitors(cutoff):
            cur.execute(
                f"SELECT COUNT(DISTINCT {vis}) FROM events WHERE ts >= {ph}",
                (cutoff,),
            )
            return cur.fetchone()[0] or 0

        dau = distinct_visitors(day_start)
        wau = distinct_visitors(cut_7d)
        mau = distinct_visitors(cut_30d)

        # Visitors seen on 2+ distinct days within the last 7 days.
        # substr(ts, 1, 10) == the date part; works on SQLite and Postgres.
        cur.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {vis} AS visitor
                FROM events
                WHERE ts >= {ph} AND {vis} IS NOT NULL
                GROUP BY {vis}
                HAVING COUNT(DISTINCT substr(ts, 1, 10)) >= 2
            ) returning_visitors
            """,
            (cut_7d,),
        )
        returning_7d = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM users")
        signups_total = cur.fetchone()[0] or 0

        cur.execute(
            f"SELECT COUNT(DISTINCT user_sub) FROM events "
            f"WHERE ts >= {ph} AND user_sub IS NOT NULL",
            (cut_7d,),
        )
        signed_in_wau = cur.fetchone()[0] or 0

        cur.execute(
            f"""
            SELECT route, COUNT(*) AS hits, COUNT(DISTINCT {vis}) AS uniques
            FROM events
            WHERE ts >= {ph}
            GROUP BY route
            ORDER BY hits DESC
            LIMIT 10
            """,
            (cut_7d,),
        )
        top_routes_7d = [
            {"route": r[0], "hits": r[1], "uniques": r[2]} for r in cur.fetchall()
        ]

        cur.execute(
            f"""
            SELECT substr(ts, 1, 10) AS day,
                   COUNT(DISTINCT {vis}) AS visitors,
                   COUNT(*) AS hits
            FROM events
            WHERE ts >= {ph}
            GROUP BY substr(ts, 1, 10)
            ORDER BY substr(ts, 1, 10)
            """,
            (cut_30d,),
        )
        daily_series_30d = [
            {"date": r[0], "visitors": r[1], "hits": r[2]} for r in cur.fetchall()
        ]
    finally:
        conn.close()

    return jsonify({
        "dau": dau,
        "wau": wau,
        "mau": mau,
        "returning_7d": returning_7d,
        "signups_total": signups_total,
        "signed_in_wau": signed_in_wau,
        "top_routes_7d": top_routes_7d,
        "daily_series_30d": daily_series_30d,
    })


# ── Wiring ────────────────────────────────────────────────────────────────────
def init_app(app, get_conn, placeholder, use_postgres):
    """Register analytics hooks + endpoint on the Flask app. Fail-open: if the
    schema can't be created (e.g. DB down), the app still boots and serves."""
    global _get_conn, _placeholder, _use_postgres, _flusher_started
    _get_conn = get_conn
    _placeholder = placeholder
    _use_postgres = use_postgres

    try:
        _init_schema()
    except Exception as exc:
        print(f"[dv_analytics] schema init failed (analytics disabled-ish): {exc}")

    if not _flusher_started:
        _flusher_started = True
        threading.Thread(target=_flusher_loop, name="dv-analytics-flush",
                         daemon=True).start()
        atexit.register(_flush_safe)

    app.before_request(_before_request)
    app.after_request(_after_request)
    app.add_url_rule("/api/analytics/summary", "dv_analytics_summary",
                     _analytics_summary, methods=["GET"])

    print(f"[dv_analytics] initialized (auth {'on' if AUTH_ENABLED else 'off'}, "
          f"backend {'postgres' if use_postgres else 'sqlite'})")
