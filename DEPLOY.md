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
season via GitHub Actions (`.github/workflows/refresh-board.yml`):

- **Schedule:** Tuesdays 09:00 UTC, August through January (cron `0 9 * 8-12,1 2`).
  It can also be run on demand from the Actions tab (`workflow_dispatch`).
- **What it does:** the job runs `scripts/refresh_board.sh` against the live API
  (repository variable `BOARD_API_URL`, e.g. `https://draft.jkrek.com`). The script
  invokes `build_prospect_cache.py`, which:
  1. rebuilds `training_data/prospect_cache.json` (every FBS player re-graded),
  2. snapshots the slim board to `training_data/board_history/board_<YYYY-MM-DD>.json`,
  3. diffs against the most recent prior snapshot and writes
     `training_data/board_movers.json` (top-15 risers/fallers plus per-player
     deltas, served by `GET /api/movers` and merged into `/api/prospects` rows
     as an optional `trend` field).
- **Publishing:** the workflow commits the refreshed cache, the new history
  snapshot, and `board_movers.json` back to `main`. Railway auto-deploys on
  every push to `main`, and the freshly deployed image ships the new JSON, so
  the site picks the refresh up automatically.
- **No-redeploy path:** the Flask app also hot-reloads these files on mtime —
  each request to `/api/prospects` / `/api/movers` stats the JSON and reloads
  it if it changed on disk. So a cache uploaded manually to a running instance
  (e.g. via the Railway shell) takes effect immediately, no restart needed.
- **Local refresh:** start the API (`python XGBOost.py`), then run
  `scripts/refresh_board.sh` (defaults to `http://localhost:5001`; override
  with `BOARD_API_URL`).
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
