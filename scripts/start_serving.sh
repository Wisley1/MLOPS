#!/usr/bin/env bash
# Start ClearML Serving HTTP API locally (clearml-serving >= 1.3 has no "serve" subcommand).
set -euo pipefail

SERVING_ID="${1:-${CLEARML_SERVING_TASK_ID:-}}"
PORT="${CLEARML_SERVING_PORT:-8088}"

if [[ -z "${SERVING_ID}" ]]; then
  echo "Usage: $0 <SERVING_ID>"
  echo "Or:    CLEARML_SERVING_TASK_ID=<id> $0"
  echo ""
  echo "Create a service first:"
  echo "  clearml-serving create --name \"Arxiv Classifier\""
  exit 1
fi

export CLEARML_API_HOST="${CLEARML_API_HOST:-http://localhost:8008}"
export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-http://localhost:8080}"
export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-http://localhost:8081}"
export CLEARML_SERVING_TASK_ID="${SERVING_ID}"
export CLEARML_SERVING_POLL_FREQ="${CLEARML_SERVING_POLL_FREQ:-1}"

echo "Starting ClearML Serving on http://localhost:${PORT}"
echo "SERVING_ID=${SERVING_ID}"

exec uvicorn clearml_serving.serving.main:app --host 0.0.0.0 --port "${PORT}"
