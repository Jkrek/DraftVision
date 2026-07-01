# DraftVision 🏈

**A full-stack machine-learning web app that predicts NFL draft success for college and high-school football prospects.**

Live demo: **[draft.jkrek.com](https://draft.jkrek.com)**  ·  Built by [Jared Krekeler](https://github.com/Jkrek)

> Enter any college or high-school player and DraftVision returns a success probability, projected draft grade, and historical NFL comparisons — powered by a 3-model ML ensemble over live ESPN and College Football Data.

---

## Highlights

- **9,000+ college prospects** ranked and gradeable; **10,000+ high-school recruits** (2020–2027 classes) from the CFBD 247Sports composite.
- **3-model ML ensemble** — calibrated XGBoost (`CalibratedClassifierCV`) + CatBoost + a rule-based fallback, probability-averaged for a stable success score.
- **16 engineered features** — e.g. a 0–100 position-normalized `production_score`, a 1–10 `conference_tier`, a position-normalized `combine_speed_score`, All-American / award flags, and position one-hots.
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

1. **Predict** (`/predict`) — a player name is fuzzy-matched against live ESPN rosters, features are engineered on the fly, and the ensemble returns a success probability + draft grade. Cosine similarity across the 16-feature vector surfaces historical NFL comps.
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
