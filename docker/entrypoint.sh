#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/gemma4-27b-awq}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant35}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "=== vLLM-TurboQuant Server ==="
echo "Model        : ${MODEL_PATH}"
echo "KV cache     : ${KV_CACHE_DTYPE}"
echo "GPU mem util : ${GPU_MEMORY_UTIL}"
echo "Max model len: ${MAX_MODEL_LEN}"

# Only pass metadata path if the file actually exists
META_FLAG=""
TURBOQUANT_META="${MODEL_PATH}/turboquant_kv.json"
if [ -f "${TURBOQUANT_META}" ]; then
    META_FLAG="--turboquant-metadata-path ${TURBOQUANT_META}"
fi

# Only pass --api-key if VLLM_API_KEY is set
API_KEY_FLAG=""
if [ -n "${VLLM_API_KEY:-}" ]; then
    API_KEY_FLAG="--api-key ${VLLM_API_KEY}"
fi

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-turboquant \
    ${META_FLAG} \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    ${API_KEY_FLAG} \
    "$@"
