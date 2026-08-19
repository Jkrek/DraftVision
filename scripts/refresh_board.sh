#!/usr/bin/env bash
# LOCAL convenience wrapper for a prospect-board refresh: rebuild
# prospect_cache.json against a running prediction API, snapshot the board into
# training_data/board_history/, and write training_data/board_movers.json
# (risers/fallers vs the prior snapshot).
#
# NOTE: CI no longer uses this script. The weekly GitHub Actions job
# (.github/workflows/refresh-board.yml) boots the prediction server inside the
# runner and invokes build_prospect_cache.py directly with --workers 6 —
# production is never hit.
#
# Usage:
#   scripts/refresh_board.sh                          # local API (http://localhost:5001)
#   scripts/refresh_board.sh --max-teams 20           # extra args pass through
#   scripts/refresh_board.sh --workers 4 --delay 0    # concurrent grading
#
# Env:
#   BOARD_API_URL  Base URL of the prediction API (default http://localhost:5001)
#   PYTHON_BIN     Python interpreter to use (default: python3)
set -euo pipefail

cd "$(dirname "$0")/.."

API_URL="${BOARD_API_URL:-http://localhost:5001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Refreshing prospect board against ${API_URL}"
exec "$PYTHON_BIN" build_prospect_cache.py --api-url "$API_URL" "$@"
