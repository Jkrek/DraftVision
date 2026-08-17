#!/usr/bin/env python
"""Cached CFBD API client for v3 experiments.

- Reads CFBD_API_KEY from the repo .env at runtime (never stored anywhere else).
- Caches every response as JSON under /tmp/dv_training_cache/cfbd/ keyed by
  endpoint+params; checks the cache before hitting the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = "/tmp/dv_training_cache/cfbd"
BASE = "https://api.collegefootballdata.com"


def _api_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key
    with open(os.path.join(REPO_ROOT, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("CFBD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("CFBD_API_KEY not found in environment or .env")


def _cache_path(endpoint: str, params: dict) -> str:
    q = urllib.parse.urlencode(sorted(params.items()))
    slug = endpoint.strip("/").replace("/", "_")
    h = hashlib.md5(f"{endpoint}?{q}".encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{slug}__{h}.json")


def get(endpoint: str, params: dict, max_retries: int = 3):
    """GET endpoint with params; returns parsed JSON. Cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(endpoint, params)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    headers = {"Authorization": f"Bearer {_api_key()}"}
    for attempt in range(max_retries):
        resp = requests.get(BASE + endpoint, params=params, headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
            time.sleep(0.15)  # be polite
            return data
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"CFBD {endpoint} {params} -> HTTP {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError(f"CFBD {endpoint} {params} -> rate-limited after {max_retries} tries")


if __name__ == "__main__":
    # quick probe
    import sys
    ep = sys.argv[1] if len(sys.argv) > 1 else "/ratings/sp"
    params = dict(kv.split("=", 1) for kv in sys.argv[2:])
    data = get(ep, params)
    print(f"{ep} {params}: {len(data) if isinstance(data, list) else type(data)} items")
    if isinstance(data, list) and data:
        print(json.dumps(data[0], indent=2)[:800])
