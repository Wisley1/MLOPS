set -euo pipefail

SERVING_URL="${CLEARML_SERVING_URL:-http://localhost:8088}"
ENDPOINT="${CLEARML_SERVING_ENDPOINT:-arxiv_classify}"
VERSION="${CLEARML_SERVING_VERSION:-1}"

echo "=== transformers ==="
curl -s -X POST "${SERVING_URL}/serve/${ENDPOINT}/${VERSION}" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"text": "We propose a novel transformer architecture for long-range sequence modeling."}' \
  | python3 -m json.tool

echo ""
echo "=== medicine ==="
curl -s -X POST "${SERVING_URL}/serve/${ENDPOINT}/${VERSION}" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"text": "Clinical trial of immunotherapy for advanced melanoma patients."}' \
  | python3 -m json.tool

echo ""
echo "=== computer vision ==="
curl -s -X POST "${SERVING_URL}/serve/${ENDPOINT}/${VERSION}" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"text": "Real-time object detection with convolutional neural networks for autonomous driving."}' \
  | python3 -m json.tool
