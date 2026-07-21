#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/artifacts}"

echo "==> Waiting for Ollama at ${OLLAMA_URL} ..."
for _ in $(seq 1 60); do
    if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; then
        echo "==> Ollama is reachable."
        break
    fi
    sleep 2
done
curl -sf "${OLLAMA_URL}/api/tags" > /dev/null || {
    echo "!! Ollama never became reachable at ${OLLAMA_URL}" >&2
    exit 1
}

echo "==> Ensuring ${EMBEDDING_MODEL} is pulled (this is the real readiness gate) ..."
curl -sf -X POST "${OLLAMA_URL}/api/pull" \
    -d "{\"name\": \"${EMBEDDING_MODEL}\", \"stream\": false}" > /dev/null

echo "==> Verifying ${EMBEDDING_MODEL} responds to an embedding request ..."
model_ready=false
for _ in $(seq 1 30); do
    if curl -sf -X POST "${OLLAMA_URL}/api/embeddings" \
        -d "{\"model\": \"${EMBEDDING_MODEL}\", \"prompt\": \"ready check\"}" > /dev/null; then
        echo "==> Model is ready."
        model_ready=true
        break
    fi
    sleep 2
done
if [ "${model_ready}" != true ]; then
    echo "!! ${EMBEDDING_MODEL} never responded to an embedding request" >&2
    exit 1
fi

mkdir -p "${ARTIFACTS_DIR}"

echo "==> Running integration tests + benchmarks ..."
set +e
uv run pytest integration/ -v --benchmark-json="${ARTIFACTS_DIR}/benchmarks.json"
TEST_EXIT_CODE=$?
set -e

if [ -f "${ARTIFACTS_DIR}/benchmarks.json" ]; then
    echo "==> Rendering benchmark report ..."
    uv run python integration/bench_report.py "${ARTIFACTS_DIR}/benchmarks.json" "${ARTIFACTS_DIR}/benchmarks.md"
fi

exit "${TEST_EXIT_CODE}"
