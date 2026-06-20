#!/usr/bin/env bash
# Start ClearML Agent worker listening on the students queue.
set -euo pipefail

QUEUE="${CLEARML_QUEUE:-students}"

# Prefer local ClearML Server unless explicitly overridden
export CLEARML_API_HOST="${CLEARML_API_HOST:-http://localhost:8008}"
export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-http://localhost:8080}"
export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-http://localhost:8081}"

if ! command -v clearml-agent >/dev/null 2>&1; then
  echo "Error: clearml-agent not found."
  echo "Install it with: pip install clearml-agent"
  echo "Or install all deps: pip install -r requirements.txt"
  exit 1
fi

if pgrep -fl "clearml-agent daemon" >/dev/null 2>&1; then
  echo "WARNING: another clearml-agent daemon is already running."
  echo "Stop all agents first: pkill -f 'clearml-agent daemon'"
fi

AGENT_ARGS=(--queue "${QUEUE}" --create-queue)

# macOS has no NVIDIA GPU in Docker — run locally by default.
# Set CLEARML_AGENT_DOCKER=python:3.11-slim-bookworm to force Docker + --cpu-only.
if [[ "$(uname -s)" == "Darwin" && -z "${CLEARML_AGENT_DOCKER:-}" ]]; then
  echo "Starting ClearML Agent on queue: ${QUEUE} (local mode, no Docker)"
else
  DOCKER_IMAGE="${CLEARML_AGENT_DOCKER:-python:3.11-slim-bookworm}"
  echo "Starting ClearML Agent on queue: ${QUEUE}"
  echo "Docker image for tasks: ${DOCKER_IMAGE}"
  AGENT_ARGS+=(--docker "${DOCKER_IMAGE}")
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "macOS detected: adding --cpu-only"
    AGENT_ARGS+=(--cpu-only)
  fi
fi

clearml-agent daemon "${AGENT_ARGS[@]}"
