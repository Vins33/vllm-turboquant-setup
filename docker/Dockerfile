# Dockerfile for vLLM + TurboQuantum + Gemma-4 27B
# RTX 5090 Optimized

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.8 \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_PRECOMPILED=0 \
    VLLM_MAIN_CUDA_VERSION=12.8 \
    PYTHONUNBUFFERED=1 \
    PYTHON_VERSION=3.12

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    build-essential \
    git \
    wget \
    curl \
    software-properties-common \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install uv (faster package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy vLLM source code
COPY vllm-turboquant/ ./vllm-turboquant/

# Setup virtual environment and install vLLM
RUN ~/.local/bin/uv venv /app/.venv --python 3.12 && \
    . /app/.venv/bin/activate && \
    ~/.local/bin/uv pip install -r vllm-turboquant/requirements/lint.txt && \
    cd vllm-turboquant && \
    ~/.local/bin/uv pip install -e . && \
    cd /app && \
    ~/.local/bin/uv pip install torch transformers huggingface_hub

# Create directories for models
RUN mkdir -p /models /logs /cache

# Set PATH to use venv
ENV PATH=/app/.venv/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting vLLM Server..."\n\
echo "   Model: Gemma-4 27B"\n\
echo "   Quantization: TurboQuantum"\n\
echo "   GPU: RTX 5090"\n\
export PYTHONUNBUFFERED=1\n\
exec python -m vllm.entrypoints.openai.api_server \\\n\
    --model /models/gemma4-27b-int8 \\\n\
    --quantization turboquant \\\n\
    --kv-cache-dtype turboquant35 \\\n\
    --turboquant-metadata-path /models/gemma4-27b-int8/turboquant_kv.json \\\n\
    --attention-backend TRITON_ATTN \\\n\
    --gpu-memory-utilization 0.85 \\\n\
    --max-model-len 4096 \\\n\
    --tensor-parallel-size 1 \\\n\
    --host 0.0.0.0 \\\n\
    --port 8000 \\\n\
    --dtype float16 \\\n\
    "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
