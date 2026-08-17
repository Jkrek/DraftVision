#!/usr/bin/env bash
# Refresh the prospect board: rebuild prospect_cache.json against a running
# prediction API, snapshot the board into training_data/board_history/, and
# write training_data/board_movers.json (risers/fallers vs the prior snapshot).
#
# Usage:
#   scripts/refresh_board.sh                          # local API (http://localhost:5001)
#   BOARD_API_URL=https://jkrek.com scripts/refresh_board.sh   # production API
#   scripts/refresh_board.sh --max-teams 20           # extra args pass through
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
