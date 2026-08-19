# DraftVision — Railway Deployment Guide

## Prerequisites
- [Railway account](https://railway.app) (free tier works)
- A domain name (e.g., from Namecheap, Google Domains, Porkbun)
- Git repo pushed to GitHub

## Step 1 — Push to GitHub

```bash
# One-time setup
git remote add origin https://github.com/YOUR_USERNAME/draftvision.git
git push -u origin main
```

## Step 2 — Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Choose **Deploy from GitHub repo** → select your repo
3. Railway auto-detects `railway.toml` and runs:
   - Build: `pip install -r requirements.txt && npm ci && npm run build`
   - Start: `gunicorn -w 2 -b 0.0.0.0:$PORT --timeout 120 XGBOost:app`

## Step 3 — Add Postgres

1. In your Railway project → **+ New** → **Database** → **PostgreSQL**
2. Railway automatically injects `DATABASE_URL` into your service — no config needed

## Step 4 — Set environment variables

In Railway → your service → **Variables**, add:

| Variable | Value |
|----------|-------|
| `SECRET_KEY` | a long random string (e.g., `openssl rand -hex 32`) |
| `FLASK_ENV` | `production` |
| `FRONTEND_ORIGIN` | `https://draft.jkrek.com` |
| `CANONICAL_HOST` | `draft.jkrek.com` |
| `FORCE_HTTPS` | `true` |
| `AUTO_SYNC_COLLEGE_PROSPECTS` | `false` (set to `true` on first deploy to seed data) |

`DATABASE_URL` and `PORT` are set automatically by Railway.

### Optional — auth + analytics (dv_analytics.py)

All three are optional; each feature stays dormant until its variable is set.

| Variable | Value |
|----------|-------|
| `AUTH0_DOMAIN` | your Auth0 tenant domain, e.g. `dev-xyz123.us.auth0.com` (no `https://`) |
| `AUTH0_AUDIENCE` | the API identifier configured in Auth0, e.g. `https://api.jkrek.com` |
| `ANALYTICS_KEY` | a long random string (e.g., `openssl rand -hex 32`) |

- **`AUTH0_DOMAIN` + `AUTH0_AUDIENCE`** (both required together): enables
  verification of `Authorization: Bearer` tokens from the frontend's
  Auth0Provider. Valid tokens upsert a `users` row (sub/email/name, tier
  defaults to `free`) and tie usage events to the signed-in user. Unset =
  everyone is anonymous; the app never 401s public routes either way. These
  must match the frontend's `REACT_APP_AUTH0_*` values.
- **`ANALYTICS_KEY`**: unlocks `GET /api/analytics/summary` (pass it as
  `?key=...` or the `X-Analytics-Key` header) returning DAU/WAU/MAU,
  returning visitors, signups, top routes, and a 30-day daily series. While
  unset (or on a wrong key) the endpoint answers 404, so it is invisible in
  prod until configured.

No raw IPs or user agents are stored — visitors are counted by Auth0 `sub`
or the browser's self-assigned `X-DV-Anon` uuid only.

## Step 5 — Connect your domain

1. Railway project → **Settings** → **Domains** → **Add Custom Domain**
2. Add the domain:
   - `draft.jkrek.com`  (the apex `jkrek.com` belongs to JKClips)
3. Railway shows a CNAME target. In Cloudflare (jkrek.com's DNS), EDIT the
   existing `draft` record — it currently points at a stale
   `ghs.googlehosted.com` — to a CNAME of the Railway hostname. Set it to
   "DNS only" (grey cloud) until the certificate is issued, then proxying
   can be re-enabled if desired.
4. In Railway variables, set:
   - `FRONTEND_ORIGIN=https://draft.jkrek.com`
   - `CANONICAL_HOST=draft.jkrek.com`
   - `FORCE_HTTPS=true`
5. Wait for DNS propagation and SSL issuance
6. Verify:
   - `http://draft.jkrek.com` → `https://draft.jkrek.com`

## Step 6 — First deploy

1. Merge any changes → push to `main` → Railway auto-deploys
2. Watch build logs in Railway dashboard
3. Hit `/health` on your domain to confirm the app is running:
   ```
   https://draft.jkrek.com/health
   ```

## Subsequent deploys

```bash
git add .
git commit -m "your message"
git push origin main   # Railway auto-deploys on every push to main
```

## Syncing college prospects

After first deploy, trigger a prospect sync via curl or the Railway shell:

```bash
curl -X POST https://yourdomain.com/sync/college-prospects \
  -H "Content-Type: application/json" \
  -d '{"max_teams": 250, "max_players": 4000}'
```

Or set `AUTO_SYNC_COLLEGE_PROSPECTS=true` for the first boot, then flip it back to `false`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| White screen | Check `/health` — if 200, clear browser cache; if 500, check build logs |
| `players` table missing | Postgres migration runs automatically on startup via `initialize_player_database()` |
| ESPN stats not loading | Stats are cached for 1 hour; check Railway logs for fetch errors |
| Slow first load | Cold start on Railway free tier takes ~20s; paid tier eliminates this |

## Weekly board refresh (risers / fallers)

The prospect board refreshes itself once a week during the college football
season via GitHub Actions (`.github/workflows/refresh-board.yml`). The whole
job is **self-contained inside the Actions runner** — it boots its own copy of
the prediction server from the checked-out repo and grades every player
against `127.0.0.1`, so **production receives zero traffic** during a refresh
(the old design fired ~12,300 `/predict` calls at the 1 GB Fly box over 8+
hours; the new one finishes in ~25–40 minutes).

- **Schedule:** Tuesdays 09:00 UTC, August through January (cron `0 9 * 8-12,1 2`).
  It can also be run on demand from the Actions tab (`workflow_dispatch`).
- **What it does:**
  1. installs `requirements.txt` and starts `gunicorn -w 4 XGBOost:app` on
     `127.0.0.1:5001` (models load from the committed artifacts;
     `AUTO_SYNC_COLLEGE_PROSPECTS=false`, no `ANALYTICS_KEY` so analytics
     stays dormant), then waits for `/health`;
  2. *(optional)* refreshes `training_data/enrichment.json` via
     `scripts/build_enrichment.py` — only when the `CFBD_API_KEY` repo secret
     is configured, and `continue-on-error` so a CFBD hiccup never blocks the
     board (the committed `enrichment.json` keeps serving otherwise);
  3. runs `build_prospect_cache.py --workers 6 --delay 0` against localhost —
     6 threads paced by the builder's global ~10 requests/sec ceiling
     (~12.3k predicts ≈ 21 min + roster fetches). This rebuilds
     `training_data/prospect_cache.json`, snapshots the slim board to
     `training_data/board_history/board_<YYYY-MM-DD>.json`, and writes
     `training_data/board_movers.json` (top-15 risers/fallers plus per-player
     deltas, served by `GET /api/movers` and merged into `/api/prospects` rows
     as an optional `trend` field);
  4. runs `scripts/generate_weekly_recap.py` (→ `content/recaps/`), stops the
     server, and commits the refreshed files back to `main` as the board bot.
- **Failure = no commit:** the commit step only runs when every prior step
  succeeded, so a broken refresh publishes nothing and the old board stays
  live. The job has a 120-minute timeout and a `refresh-board` concurrency
  guard so runs never overlap.
- **Publishing — Fly does NOT auto-deploy on push.** The weekly commit updates
  the repo; production picks it up in one of three ways:
  1. *Recommended:* add a `FLY_API_TOKEN` repo secret (create one with
     `fly tokens create deploy -a draftvision`) — the workflow's final,
     `continue-on-error` step then runs `flyctl deploy --remote-only` and the
     new board goes live automatically. Without the secret the step is a no-op.
  2. Manual: run `fly deploy` whenever; the image ships the committed JSON.
  3. Hot-reload: the Flask app stats these JSON files on every
     `/api/prospects` / `/api/movers` request and reloads them on mtime
     change, so a cache uploaded directly to a running machine (e.g.
     `fly ssh sftp`) takes effect immediately, no restart needed.
- **Secrets (both optional — the job works with neither):**

  | Repo secret | Effect when set |
  |-------------|-----------------|
  | `CFBD_API_KEY` | enables the enrichment-refresh step (recruiting/SP+/production features); unset = committed `enrichment.json` is used as-is |
  | `FLY_API_TOKEN` | auto-deploys to Fly after the bot commit; unset = deploy stays manual |

- **Local refresh:** start the API (`python XGBOost.py`), then run
  `scripts/refresh_board.sh` (defaults to `http://localhost:5001`; override
  with `BOARD_API_URL`; extra flags like `--workers 4 --delay 0` pass
  through). A hermetic test run that never touches the real board files:
  `python build_prospect_cache.py --max-teams 3 --workers 4 --output /tmp/dv_smoke/prospect_cache.json`.
- **First run:** history starts empty, so `build_prospect_cache.py` bootstraps
  it by snapshotting the pre-existing `prospect_cache.json` (dated from its
  `generated_at`) before overwriting it — the first refreshed board therefore
  already has deltas. Until a refresh has run, `/api/movers` returns a valid
  empty payload (`"since": null`, empty `risers`/`fallers`).

## Class big boards (`/api/big-board`)

Hand-ordered per-class rankings live in `training_data/big_boards.json`
(`{"2027": ["Name|Team", ...], ...}`).

- **Read:** `GET /api/big-board?class=2027` → `board` (your ordered rows,
  resolved against the live prospect cache; unresolvable keys are skipped and
  listed in `missing`), `rest` (every other prospect of that class,
  model-ranked by grade/success probability), and `updated_at`.
- **Write:** `POST /api/big-board` with `{"class": 2027, "board": ["Name|Team", ...]}`,
  guarded exactly like `/api/analytics/summary` — supply `ANALYTICS_KEY` via
  the `X-Analytics-Key` header or `?key=`; unset/wrong key returns 404. Keys
  are validated against the prospect cache and the file is written atomically;
  every worker hot-reloads it on mtime, no restart needed.
- **Export:** `GET /api/big-board/export` (same guard) downloads the raw file.
- **Persistence:** the JSON lives on the machine's disk — it **survives
  restarts** but **RESETS on every redeploy** to the copy committed in the
  repo. To make rankings permanent, export the file and commit it as
  `training_data/big_boards.json` before the next deploy.

## Response compression + HTTP caching

- `flask-compress` (optional dep — the app boots without it) gzip/brotli
  compresses responses; the ~1-2MB `/api/prospects` payload shrinks ~10x.
- `/api/prospects`, `/api/hs-prospects`, `/api/movers`, and `/init` send
  `Cache-Control: public, max-age=300, stale-while-revalidate=600` plus a weak
  `ETag` derived from the underlying cache file's mtime + query string, and
  honor `If-None-Match` with `304 Not Modified` — the board changes weekly, so
  clients revalidate instead of redownloading identical megabytes. `/search`
  is uncached (tiny) and `/api/analytics/*` is never cached.

- **Weekly YouTube recap** — after each board refresh, the workflow runs
  `scripts/generate_weekly_recap.py`, which turns `board_movers.json` + board
  history + big boards into `content/recaps/recap_<date>.md` (titles, cold
  open, risers/fallers, disagreements). The bot commit includes `content/recaps/`.

## Market Edge board (`/api/edge`)

Read-only comparison of DraftVision model output vs public Kalshi prices
(no auth, no orders — we never take or facilitate bets). Logic lives in
`dv_edge.py`; `XGBOost.py` only registers the routes.

- **Read:** `GET /api/edge` → `{generated_at, markets, note}`. Relevant open
  markets are discovered from Kalshi's public data API (bounded pagination,
  results cached in-memory ~10 min per worker). Off-season or upstream
  failure degrades to `{"markets": [], "note": "seasonal"}` with HTTP 200 —
  the page renders a friendly empty state, never a 500.
- **Honest mapping:** an `edge` is computed only for top-X draft markets
  (32 ≤ X ≤ 50) whose matched player sits in the model's "Top 50 Pick"
  bucket; all other markets are listed with null model fields. See the
  policy comment at the top of `dv_edge.py` before changing this.
- **Paper ledger:** `GET /api/edge/ledger` serves
  `training_data/edge_ledger.json`. When a matched |edge| ≥ 10 points is
  observed, one row per (ticker, day) is appended (atomic write + mtime
  hot-reload, same pattern as the other caches). Like the big boards, the
  file survives restarts but **resets on redeploy** to the committed copy —
  export and commit it to preserve the track record.

## Fly.io Postgres (current production database)

> Production now runs on Fly.io (app `draftvision`, https://draft.jkrek.com).
> The Railway sections above are kept for historical reference.

A durable Postgres cluster is attached to the app, so **analytics events,
users, and anything written through the DB layer now survive restarts AND
redeploys** (unlike the JSON-file caches, which still reset on deploy).

| | |
|---|---|
| App name | `draftvision-db` (unmanaged Fly Postgres, postgres-flex 18) |
| Region | `ord` (same as the app) |
| Size | 1 node, `shared-cpu-1x`, 256MB RAM, 3GB volume |
| Cost | ~$3-5/mo (cheapest single-node; no HA replica) |
| Attached to | `draftvision` via `fly postgres attach` — injects the `DATABASE_URL` secret |

`XGBOost.py` auto-detects `DATABASE_URL` at startup and switches from SQLite
to Postgres (startup log prints `Using Postgres: ...`; `dv_analytics` logs
`backend postgres`). No code or config changes are needed for this to work —
if the secret is present, Postgres is used.

### Operations

```bash
fly postgres list                          # see the cluster
fly postgres connect -a draftvision-db    # psql shell into the database
fly status -a draftvision-db               # machine health
```

- Note: `fly postgres attach` **stages** the `DATABASE_URL` secret; it is not
  live until the app machines restart with it (`fly secrets deploy -a
  draftvision` if `fly secrets list` shows it as "Staged").
- Single-node means no automatic failover and Fly does not manage/support
  unmanaged Postgres — take occasional snapshots (`fly volumes snapshots
  list`) if the analytics history becomes precious.
- Do **not** detach or destroy `draftvision-db`; it holds the only durable
  copy of analytics/user data.
