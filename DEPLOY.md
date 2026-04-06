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
| `FRONTEND_ORIGIN` | `https://jkrek.com,https://www.jkrek.com` |
| `CANONICAL_HOST` | `jkrek.com` |
| `FORCE_HTTPS` | `true` |
| `AUTO_SYNC_COLLEGE_PROSPECTS` | `false` (set to `true` on first deploy to seed data) |

`DATABASE_URL` and `PORT` are set automatically by Railway.

## Step 5 — Connect your domain

1. Railway project → **Settings** → **Domains** → **Add Custom Domain**
2. Add both domains:
   - `jkrek.com`
   - `www.jkrek.com`
3. Railway gives you DNS targets to add at your registrar:
   - Add the apex/root record for `@` exactly as Railway shows
   - Add the `www` record exactly as Railway shows (usually a `CNAME` to your Railway hostname)
4. In Railway variables, set:
   - `FRONTEND_ORIGIN=https://jkrek.com,https://www.jkrek.com`
   - `CANONICAL_HOST=jkrek.com`
   - `FORCE_HTTPS=true`
5. Wait for DNS propagation and SSL issuance
6. Verify both URLs redirect correctly:
   - `http://jkrek.com` → `https://jkrek.com`
   - `https://www.jkrek.com` → `https://jkrek.com`

## Step 6 — First deploy

1. Merge any changes → push to `main` → Railway auto-deploys
2. Watch build logs in Railway dashboard
3. Hit `/health` on your domain to confirm the app is running:
   ```
   https://jkrek.com/health
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
