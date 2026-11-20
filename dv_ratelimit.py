"""Per-IP rate limits for the routes that cost something.

A fixed-window counter held in process memory: no Redis, no new dependency.
Each gunicorn worker counts on its own, so with two workers a client can get
up to twice the stated limit. That is fine for the job here, which is to stop
one address from running up cost or guessing the owner key, not to meter usage
precisely. Anything that needs exact metering (paid plans) belongs in the
database, keyed by account.

The client address is Fly's `Fly-Client-IP` header, which the Fly proxy sets
itself and a client cannot forge through it. Off Fly it falls back to the
socket address. Loopback is never limited, so the weekly job (which runs the
server on 127.0.0.1 and makes ~12,000 /predict calls) is unaffected.
"""
import threading
import time

# bucket -> (requests, per seconds)
LIMITS = {
    "owner": (10, 60),     # key-guarded writes: also caps key guessing
    "predict": (30, 60),   # resolves a player through ESPN on each call
    "tools": (60, 60),     # read-only tool layer
    "search": (240, 60),   # type-ahead
}
# (method, path prefix) -> bucket. First match wins.
RULES = [
    ("POST", "/api/mock-draft/upload", "owner"),
    ("POST", "/sync/college-prospects", "owner"),
    ("POST", "/api/big-board", "owner"),
    ("GET", "/api/big-board/export", "owner"),
    ("GET", "/api/analytics", "owner"),
    ("POST", "/predict", "predict"),
    ("POST", "/api/tools", "tools"),
    ("GET", "/api/tools", "tools"),
    ("GET", "/search", "search"),
]
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
MAX_KEYS = 20000

_lock = threading.Lock()
_hits = {}   # (bucket, ip) -> [window_start, count]


def bucket_for(method, path):
    for m, prefix, bucket in RULES:
        if method == m and (path == prefix or path.startswith(prefix + "/")):
            return bucket
    return None


def client_ip(headers, remote_addr):
    return (headers.get("Fly-Client-IP") or remote_addr or "").strip()


def check(bucket, ip, now=None):
    """-> (allowed, retry_after_seconds, remaining)."""
    limit, window = LIMITS[bucket]
    now = time.time() if now is None else now
    with _lock:
        if len(_hits) > MAX_KEYS:        # drop expired windows before growing further
            for k in [k for k, v in _hits.items() if now - v[0] >= LIMITS[k[0]][1]]:
                del _hits[k]
        slot = _hits.get((bucket, ip))
        if not slot or now - slot[0] >= window:
            slot = _hits[(bucket, ip)] = [now, 0]
        if slot[1] >= limit:
            return False, max(1, int(window - (now - slot[0]) + 0.999)), 0
        slot[1] += 1
        return True, 0, limit - slot[1]


def reset():
    with _lock:
        _hits.clear()


def install(app):
    from flask import jsonify, request

    @app.before_request
    def _rate_limit():
        bucket = bucket_for(request.method, request.path)
        if not bucket:
            return None
        ip = client_ip(request.headers, request.remote_addr)
        if ip in LOOPBACK:
            return None
        ok, retry, _ = check(bucket, ip)
        if ok:
            return None
        resp = jsonify({"error": "rate limit exceeded", "retry_after_seconds": retry})
        resp.status_code = 429
        resp.headers["Retry-After"] = str(retry)
        resp.headers["Cache-Control"] = "no-store"
        return resp
