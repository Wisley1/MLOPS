#!/usr/bin/env bash
# Start ClearML Triton Inference Server container (syncs models from Serving task).
set -euo pipefail

SERVING_ID="${1:-${CLEARML_SERVING_TASK_ID:-}}"
if [[ -z "${SERVING_ID}" ]]; then
  echo "Usage: $0 <SERVING_ID>"
  exit 1
fi

CONF="${HOME}/clearml.conf"
if [[ ! -f "${CONF}" ]]; then
  echo "ERROR: ${CONF} not found. Run clearml-init first."
  exit 1
fi

if [[ -z "${CLEARML_API_ACCESS_KEY:-}" || -z "${CLEARML_API_SECRET_KEY:-}" ]]; then
  CLEARML_API_ACCESS_KEY="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path.home().joinpath("clearml.conf").read_text()
m = re.search(r'access_key\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "")
PY
)"
  CLEARML_API_SECRET_KEY="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path.home().joinpath("clearml.conf").read_text()
m = re.search(r'secret_key\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "")
PY
)"
  export CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY
fi

# Inside the container, localhost → host machine (see --add-host below).
# host.docker.internal breaks fileserver auth for stored model URLs.
export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-http://localhost:8080}"
export CLEARML_API_HOST="${CLEARML_API_HOST:-http://localhost:8008}"
export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-http://localhost:8081}"

if docker image inspect clearml/clearml-serving-triton:latest >/dev/null 2>&1; then
  echo "Triton image already present, skipping pull."
else
  echo "Pulling clearml/clearml-serving-triton (first run may take a while)..."
  docker pull --platform linux/amd64 clearml/clearml-serving-triton:latest
fi

docker rm -f clearml-serving-triton 2>/dev/null || true

docker run -d --name clearml-serving-triton \
  --platform linux/amd64 \
  --add-host=localhost:host-gateway \
  -v "${CONF}:/root/clearml.conf:ro" \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  -p 8001:8001 \
  -e CLEARML_WEB_HOST="${CLEARML_WEB_HOST}" \
  -e CLEARML_API_HOST="${CLEARML_API_HOST}" \
  -e CLEARML_FILES_HOST="${CLEARML_FILES_HOST}" \
  -e CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}" \
  -e CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}" \
  -e CLEARML_SERVING_TASK_ID="${SERVING_ID}" \
  -e CLEARML_EXTRA_PYTHON_PACKAGES="transformers>=4.44.0,<5 onnxruntime>=1.16.0" \
  clearml/clearml-serving-triton:latest

echo "Triton started on gRPC :8001 (SERVING_ID=${SERVING_ID})"
echo "Watch logs: docker logs -f clearml-serving-triton"
