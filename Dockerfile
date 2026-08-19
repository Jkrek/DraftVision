# ── Stage 1: Build React ─────────────────────────────────────────────────────
FROM node:18-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY public/ public/
COPY src/ src/
# CRA bakes REACT_APP_* (Auth0 public credentials) from this at build time
COPY .env.production ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
# dv_*.py glob: every backend module ships (adding a module without updating
# this line crashed prod twice — dv_analytics, then dv_edge)
COPY XGBOost.py dv_*.py ./
COPY *.json *.pkl *.cbm ./
COPY models/ ./models/
COPY training_data/ ./training_data/

# Copy React build from stage 1
COPY --from=frontend /app/build ./build

# Fly injects PORT; gunicorn binds to it
ENV FLASK_ENV=production
EXPOSE 8080

CMD gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} --timeout 120 --access-logfile - XGBOost:app
