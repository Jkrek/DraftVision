# DraftVision 🏈

**A full-stack machine-learning web app that predicts NFL draft success for college and high-school football prospects.**

Live demo: **[draft.jkrek.com](https://draft.jkrek.com)**  ·  Built by [Jared Krekeler](https://github.com/Jkrek)

> Enter any college or high-school player and DraftVision returns a success probability, projected draft grade, and historical NFL comparisons — powered by a calibrated two-model ensemble trained on real 2010–2020 draft outcomes.

---

## Highlights

- **9,000+ college prospects** ranked and gradeable; **10,000+ high-school recruits** (2020–2027 classes) from the CFBD 247Sports composite.
- **2-model calibrated ensemble** — XGBoost + CatBoost, probabilities averaged then passed through a fitted calibrator; a rule-based scorer is the fallback when model inference is unavailable. Trained and evaluated on real 2010–2020 NFL draft outcomes (nflverse), holdout = 2019–2020 classes.
- **13 engineered features** — a 0–100 position-percentile `production_score`, `games_played`, a position-normalized `combine_speed_score`, a 1–10 `conference_tier`, and 9 position one-hots.
- **Per-player explanations** — `/predict` returns SHAP contributions (XGBoost TreeSHAP) for the individual player, not just global feature importances.
- **Claude Vision mock-draft import** — drop in a PNG of a mock draft (e.g. PFF's image-only export) and `claude-haiku-4-5-20251001` extracts the picks into clean JSON — no brittle parsing.
- **AI mock-draft generators** — college and HS mock drafts that mirror real draft-room logic (per-team positional needs, honored commitments, best-available fallback).
- **Sub-100ms leaderboard** — an offline cache pipeline pre-computes every FBS roster so pages load with zero live API calls.

## Tech Stack

| Layer | Tools |
|------|------|
| Frontend | React (SPA, React Router), Auth0, ag-grid |
| Backend | Python, Flask (serves the built React app as a single artifact) |
| ML | XGBoost, CatBoost, scikit-learn (calibration), Pandas / NumPy |
| Data | ESPN API (live rosters/stats), College Football Data API, Claude Vision API |
| Deploy | Docker → Fly.io |

## How it works

1. **Predict** (`/predict`) — a player name is fuzzy-matched against live ESPN rosters, features are engineered on the fly, and the ensemble returns a success probability + draft grade. Historical NFL comps (hits *and* busts) come from a weighted feature-distance similarity over production, athleticism, conference tier, and accolades — same-position-group only.
2. **Leaderboard** (`/leaderboard`) — `build_prospect_cache.py` walks every FBS roster, scores each player, and bakes results into the Docker image, so the ranked grid of 9,033 prospects loads instantly.
3. **Mock drafts** (`/mock-draft`, `/college-mock-draft`, `/hs-mock-draft`) — import a mock via Claude Vision, or generate one from a needs-based simulation over the real NFL draft order.

Pre-trained models (`.cbm` / `.pkl` / `.json`) are baked into the image — no training at request time.

## Run locally

```bash
# Backend (Flask API on :5001)
pip install -r requirements.txt
python -m src.app        # see "Run Instructions" / DEPLOY.md for the exact entrypoint

# Frontend (React on :3000, proxies to :5001)
npm install
npm start
```

See `DEPLOY.md` for the Docker/Fly.io deployment and `DEMO_GUIDE.md` for a full feature walkthrough.

## About

DraftVision was built as a Computer Science Senior Design project at the University of Cincinnati. It combines applied machine learning (feature engineering, model calibration, ensembling), live data integration, and full-stack web development in a single deployed product.

*Data from ESPN and the College Football Data API. Not affiliated with the NFL, NFLPA, or any team.*

## Auth setup (optional)

1. Create an Auth0 **Single Page Application** in your Auth0 dashboard.
2. Set Allowed Callback URLs, Allowed Logout URLs, and Allowed Web Origins to your site URL (e.g. `http://localhost:3000`).
3. Set `REACT_APP_AUTH0_DOMAIN`, `REACT_APP_AUTH0_CLIENT_ID`, and (optional) `REACT_APP_AUTH0_AUDIENCE` — see `.env.example`.
4. Optional: create an Auth0 API and set its identifier as the audience; the backend then verifies tokens via `AUTH0_DOMAIN` + `AUTH0_AUDIENCE`.
5. With no Auth0 vars set, the app runs fully logged-out — the sign-in button simply hides.
