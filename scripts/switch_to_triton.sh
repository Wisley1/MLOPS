#!/usr/bin/env bash
# Switch ClearML Serving to Triton engine and restart HTTP serving.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVING_ID="${1:-${CLEARML_SERVING_TASK_ID:-84a733bf0b2a4d9f83fb94a89f666c4a}}"
PORT="${CLEARML_SERVING_PORT:-8088}"

wait_docker() {
  echo "Waiting for Docker daemon..."
  for i in $(seq 1 30); do
    if docker ps >/dev/null 2>&1; then
      echo "Docker ready (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: Docker daemon not responding. Open Docker Desktop and wait until it is ready."
  exit 1
}

ensure_clearml_server() {
  if curl -sf --max-time 3 http://localhost:8008/debug.ping >/dev/null 2>&1; then
    echo "ClearML API OK"
    return 0
  fi
  echo "Starting ClearML Server..."
  (cd "${ROOT}/docker" && docker compose up -d)
  for i in $(seq 1 30); do
    if curl -sf --max-time 3 http://localhost:8008/debug.ping >/dev/null 2>&1; then
      echo "ClearML API ready (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ClearML API not available on :8008"
  exit 1
}

resolve_triton_model_id() {
  if [[ -n "${CLEARML_TRITON_MODEL_ID:-}" ]]; then
    echo "${CLEARML_TRITON_MODEL_ID}"
    return 0
  fi
  python3 - <<'PY'
from clearml import Model

def is_valid_onnx_model(model: Model) -> bool:
    url = (model.url or "").lower()
    if not url or "uploading_file" in url:
        return False
    try:
        local = model.get_local_copy()
        if not local:
            return False
        from pathlib import Path
        path = Path(local)
        if path.is_dir():
            path = path / "model.onnx"
        return path.exists() and path.stat().st_size > 1024
    except Exception:
        return False

candidates = []
for name in ("arxiv-distilbert-onnx", "arxiv-distilbert"):
    for model in Model.query_models(
        project_name="Arxiv Classification",
        model_name=name,
        only_published=True,
    ) or []:
        if name.endswith("-onnx") or "onnx" in (model.framework or "").lower():
            candidates.append(model)

seen = set()
unique = []
for model in candidates:
    if model.id in seen:
        continue
    seen.add(model.id)
    unique.append(model)

for model in unique:
    if is_valid_onnx_model(model):
        print(model.id)
        raise SystemExit(0)

raise SystemExit(
    "No valid published ONNX model found.\n"
    "After register_model --for-triton, run:\n"
    "  CLEARML_TRITON_MODEL_ID=<onnx_model_id> ./scripts/switch_to_triton.sh"
)
PY
}

cd "${ROOT}"
wait_docker
ensure_clearml_server

MODEL_ID="$(resolve_triton_model_id)"
echo "Using Triton model: ${MODEL_ID}"

echo "=== 1/4 Configure serving for Triton ==="
clearml-serving --id "${SERVING_ID}" model remove --endpoint arxiv_classify 2>/dev/null || true

clearml-serving --id "${SERVING_ID}" model add \
  --engine triton \
  --endpoint "arxiv_classify" \
  --version 1 \
  --model-id "${MODEL_ID}" \
  --preprocess "src/serving/preprocessing.py" \
  --input-name "input_ids" "attention_mask" \
  --input-type int64 int64 \
  --input-size "[-1, -1]" "[-1, -1]" \
  --output-name "logits" \
  --output-type float32 \
  --output-size "[-1, 8]" \
  --aux-config "src/serving/triton_config.pbtxt"

clearml-serving --id "${SERVING_ID}" config --triton-grpc-server 127.0.0.1:8001

# auto-update leaves stale entries without auxiliary_cfg; they override endpoints in Triton sync
python3 - <<PY
from clearml import Task
Task.get_task(task_id="${SERVING_ID}").set_configuration_object(name="model_monitoring_eps", config_dict={})
print("Cleared stale model_monitoring_eps")
PY

echo "=== 2/4 Start Triton container ==="
chmod +x scripts/start_triton.sh
./scripts/start_triton.sh "${SERVING_ID}"

echo "=== 3/4 Restart HTTP serving on :${PORT} ==="
if lsof -ti :"${PORT}" >/dev/null 2>&1; then
  kill "$(lsof -ti :"${PORT}")" 2>/dev/null || true
  sleep 2
fi

export CLEARML_API_HOST="${CLEARML_API_HOST:-http://localhost:8008}"
export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-http://localhost:8080}"
export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-http://localhost:8081}"
export CLEARML_SERVING_TASK_ID="${SERVING_ID}"
export CLEARML_SERVING_POLL_FREQ="${CLEARML_SERVING_POLL_FREQ:-1}"

nohup uvicorn clearml_serving.serving.main:app --host 0.0.0.0 --port "${PORT}" \
  > /tmp/clearml_serving.log 2>&1 &
echo "Serving PID=$! log=/tmp/clearml_serving.log"

echo "=== 4/4 Wait for Triton model sync (90s) ==="
sleep 90

echo "=== Test ==="
./scripts/test_endpoint.sh
