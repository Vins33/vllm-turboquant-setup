# syntax=docker/dockerfile:1
# Dockerfile — vLLM-TurboQuant + Gemma 4 27B
# Target: NVIDIA RTX 5090 (Blackwell SM120)

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VLLM_TARGET_DEVICE=cuda \
    # Compile kernels for RTX A6000 (SM86), RTX 5090 (SM120) and DGX Spark (SM121)
    TORCH_CUDA_ARCH_LIST="8.6;12.0;12.1" \
    MAX_JOBS=8 \
    PYTHONUNBUFFERED=1

# Ubuntu 24.04 ships Python 3.12 natively — no PPA needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the vLLM-TurboQuant fork at the tested commit
RUN git clone https://github.com/mitkox/vllm-turboquant.git && \
    cd vllm-turboquant && \
    git checkout cee479e

# Apply SM120 patch — add RTX 5090 (12, 0) to the CUDA capability allowlist
# Without this, supports_turboquant_cuda() returns False for SM120
RUN sed -i \
    's/frozenset(((8, 6), (12, 1)))/frozenset(((8, 6), (12, 0), (12, 1)))/' \
    vllm-turboquant/vllm/v1/attention/ops/turboquant_kv_cache.py

# Create venv, install PyTorch (CUDA 12.8 wheel), then vLLM-TurboQuant
RUN ~/.local/bin/uv venv /app/.venv --python 3.12 && \
    . /app/.venv/bin/activate && \
    ~/.local/bin/uv pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    cd vllm-turboquant && \
    ~/.local/bin/uv pip install -e . --no-build-isolation

RUN mkdir -p /models /logs /cache

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Entrypoint reads config from env vars; see entrypoint.sh for all options
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
