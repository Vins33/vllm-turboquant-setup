# vLLM + TurboQuant — Gemma 3 27B AWQ on RTX 5090

**vLLM `v0.20.1rc1.dev126+gc3868bbbe` · TurboQuant `k8v4` KV cache · Gemma 3 27B-IT AWQ INT4 · NVIDIA RTX 5090 32 GB**

> Status: **fully operational** — server running, 97,673 token KV cache, 97% test pass rate over 10 stability runs.

---

## Stack

| Component | Version / Details |
|-----------|-------------------|
| vLLM | `v0.20.1rc1.dev126+gc3868bbbe` (editable install) |
| Model | `gaunernst/gemma-3-27b-it-int4-awq` · AWQ Marlin INT4 · ~17.8 GiB |
| KV cache | `turboquant_k8v4` (q8_0-K + turbo4-V) · 196 bytes/slot |
| KV tokens | **97,673** (32 GB GDDR7, 90% utilization) |
| VRAM saved | ~69% vs BF16 KV cache |
| Concurrency | 2.98× at 32K context |
| Hardware | RTX 5090 · Blackwell SM120 · CUDA 12.8 · WSL2 |
| API | OpenAI-compatible at `http://localhost:8000` |

---

## Quick Start

```bash
# Build image (applies 4 runtime patches automatically)
cd docker
docker compose build

# Start server
docker compose up -d

# Verify
curl http://localhost:8000/health
```

---

## Runtime Patches

vLLM's current main branch has four bugs triggered by the combination of Gemma 3 27B's mixed attention (sliding window + full) and TurboQuant's non-standard slot sizing. All four patches are applied automatically in the Dockerfile.

| Patch | File | Fix |
|-------|------|-----|
| **A** | `kv_cache_interface.py` | Adds `TQSlidingWindowSpec` class with TQ-aware `real_page_size_bytes` |
| **B** | `attention.py` | Returns `TQSlidingWindowSpec` for SWA layers when `kv_cache_dtype` is TurboQuant |
| **C** | `kv_cache_utils.py` | Replaces `max(page_sizes)` with `math.lcm(*page_sizes)` to handle non-divisible page sizes |
| **D** | `single_type_kv_cache_manager.py` | MRO-based lookup so `TQSlidingWindowSpec` inherits `SlidingWindowManager` |

Full technical writeup: [`docs/TURBOQUANT_PATCHES_REPORT.md`](docs/TURBOQUANT_PATCHES_REPORT.md)  
Italian implementation plan: [`docs/PIANO_VLLM_GEMMA_TURBOQUANT.md`](docs/PIANO_VLLM_GEMMA_TURBOQUANT.md)

---

## Repository Structure

```
vllm-turboquant-setup/
├── docker/
│   ├── Dockerfile              Image with 4 runtime patches applied
│   ├── docker-compose.yml      Server + optional Prometheus/Grafana
│   ├── entrypoint.sh           vLLM launch flags
│   └── models/
│       └── gemma-3-27b-it-awq/ Model weights (4 safetensors shards)
├── docs/
│   ├── TURBOQUANT_PATCHES_REPORT.md  ← Patch analysis + test results + charts
│   ├── PIANO_VLLM_GEMMA_TURBOQUANT.md  Italian technical plan
│   ├── SETUP_COMPLETE_GUIDE.md
│   └── README_SETUP.md
├── tests/
│   ├── test_business_agent.py  Business agent test suite (25 tests)
│   ├── report_validation_tests.py  Validates data from LLM inference report
│   ├── visualize_results.py    Generates PNG charts from CSV results
│   ├── pyproject.toml
│   └── results/
│       ├── business_agent_results.csv
│       ├── business_agent_results_stability.csv   (10-run stability)
│       ├── report_validation_results.csv
│       └── fig1…fig9 *.png     Charts
├── config/
│   ├── prometheus.yml
│   └── alert_rules.yml
└── scripts/
    ├── deploy.sh
    └── advanced_benchmark.py
```

---

## Tests

```bash
cd tests

# Install deps
uv pip install requests matplotlib seaborn

# Business agent suite (RAG, agent tool dispatch, multi-turn, structured output, intent)
.venv/bin/python test_business_agent.py

# 10-run stability loop
.venv/bin/python test_business_agent.py --runs 10

# Report data validation (47 tests against arXiv:2504.19874)
.venv/bin/python report_validation_tests.py

# Generate charts from results
uv run .venv/bin/python visualize_results.py
```

**Results:**

| Suite | Result |
|-------|--------|
| Business agent (single run) | 25/25 |
| Business agent (10-run stability) | **97% avg pass rate** |
| Report validation | 43/47 (4 = doc arithmetic errors, not model errors) |

---

## Performance

| Metric | Value |
|--------|-------|
| KV cache tokens | 97,673 |
| VRAM — weights | ~17.8 GiB |
| VRAM — KV cache | ~12.5 GiB (TurboQuant k8v4) |
| VRAM — KV cache (BF16 equiv.) | ~40 GiB |
| PPL degradation (q8_0-K/turbo4-V) | < 2.5% vs BF16 |
| Concurrency @ 32K ctx | 2.98× vs BF16 |
| Agent response latency (median) | ~7–10s |
| Intent classification latency (median) | ~2.3s |

---

## Prerequisites

- NVIDIA RTX 5090 (or any Blackwell/Ada GPU with ≥24 GB VRAM)
- CUDA 12.8, driver 560+
- Docker with NVIDIA Container Toolkit
- ~80 GB disk (model weights + Docker image)

---

## Monitoring

```bash
# Start with Prometheus + Grafana
docker compose --profile monitoring up -d
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```
