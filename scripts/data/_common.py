"""Shared helpers for the staging-data fetchers (scripts/data/*).

Reuses the EXACT normalization machinery from scripts/build_training_data.py
(norm_name, clean_school, _normalize_team, _team_compatible, norm_pos) so the
staged datasets join on the same keys the training build uses.

All network downloads are cached under /tmp/dv_training_cache/ (override with
DV_DATA_CACHE). Fetchers are idempotent: re-running never re-downloads a file
that is already cached, and output CSVs are rewritten atomically.

Run everything with .venv/bin/python.
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
STAGING_DIR = os.path.join(REPO_ROOT, "training_data", "staging")
CACHE_ROOT = os.environ.get("DV_DATA_CACHE", "/tmp/dv_training_cache")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Re-exported repo-canonical normalizers (import is main-guarded and safe).
from build_training_data import (  # noqa: E402,F401
    _normalize_team,
    _team_compatible,
    clean_school,
    norm_name,
    norm_pos,
)

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    SSL_CTX = ssl.create_default_context()

USER_AGENT = "DraftVision-staging/1.0 (research; contact repo owner)"


def cache_dir(name: str) -> str:
    d = os.path.join(CACHE_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def cached_download(url: str, subdir: str, fname: str, min_bytes: int = 200) -> str:
    """Download url into the cache once; return the local path."""
    path = os.path.join(cache_dir(subdir), fname)
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return path
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path + ".tmp"
    with urllib.request.urlopen(req, timeout=180, context=SSL_CTX) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    os.replace(tmp, path)
    return path


def write_csv(df, fname: str) -> str:
    """Atomically write a DataFrame to training_data/staging/<fname>."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    path = os.path.join(STAGING_DIR, fname)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(df)} rows)")
    return path
