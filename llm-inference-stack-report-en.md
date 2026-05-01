# LLM Inference Stack — Research Report
**Author:** [Vincenzo Calabrese](https://www.linkedin.com/in/vincenzocalabrese-/)  
**Date:** April 18, 2026  
**Subject:** Comparative analysis of open-source inference frameworks for large language models, with focus on TurboQuant, on-premise architectures, and enterprise deployment best practices.

---

## Table of Contents

1. [TurboQuant: the algorithm](#1-turboquant-the-algorithm)
2. [pyturboquant: practical use for RAG/embeddings](#2-pyturboquant-practical-use-for-ragembeddings)
3. [Inference Frameworks: Comparative Analysis](#3-inference-frameworks-comparative-analysis)
   - 3.1 [vLLM](#31-vllm)
   - 3.2 [SGLang](#32-sglang)
   - 3.3 [Comparison table](#33-comparison-table)
4. [Quality degradation: measured data](#4-quality-degradation-measured-data)
5. [KV Cache calculations: Gemma 4](#5-kv-cache-calculations-gemma-4)
6. [Best open-weights models for Enterprise (April 2026)](#6-best-open-weights-models-for-enterprise-april-2026)
7. [Recommended Deployment Architectures](#7-recommended-deployment-architectures)
8. [Conclusions and Final Recommendations](#8-conclusions-and-final-recommendations)

---

## 1. TurboQuant: the algorithm

**Reference:** arXiv:2504.19874 — Google Research, April 2025.

TurboQuant is an **online** quantization algorithm for the KV cache of transformers. It operates at inference time without requiring training, calibration, or pre-computed codebooks.

### Mechanism

The quantization process occurs in two phases:

1. **WHT Rotation (Walsh-Hadamard Transform):** rotation of K and V vectors into the Hadamard space. This "spreads" values uniformly, eliminating the outliers that make quantization difficult.

2. **Lloyd-Max Scalar Quantization:** scalar quantization with MSE optimization. The Lloyd-Max quantizer is asymptotically optimal with respect to the Shannon bound: $D^* = \frac{\pi e}{6} \cdot 2^{-2b}$ for b bits.

### Available variants

| Type | Bits | Recommended use |
|---|---|---|
| `turbo2` | 2 bit | V values only (asymmetric) |
| `turbo3` | 3 bit | Primary configuration |
| `turbo4` | 4 bit | High quality |
| `q8_0` | 8 bit | K keys (asymmetric) |

### Important: asymmetric configuration

The community (5 independent groups) has documented that the **symmetric** configuration `turbo3/turbo3` on dense models with `head_dim=128` causes PPL to jump from 6 to 3400+.

**Validated safe configuration:**
```
--kv-cache-dtype q8_0-K,turbo3-V
```
Only V values (less sensitive) are quantized, while K keys remain at 8-bit. This avoids catastrophic degradation while preserving memory benefits.

### QJL (Algorithm 2): warning

The QJL algorithm (a variant using random Johnson-Lindenstrauss projections) is counterproductive in practice: variance explosion during the projection worsens attention MSE. QJL should be avoided in all scenarios tested by the community.

---

## 2. pyturboquant: practical use for RAG/Embeddings

**Package:** `pip install pyturboquant[langchain]`  
**Stable version:** 0.0.3 (April 2026)  
**Language:** Pure Python, PyTorch, no custom C++ kernels.

### Use case

pyturboquant compresses embedding vectors for RAG indexes (similar to FAISS with lossy compression). It does not concern the LLM's KV cache, but the storage of retrieved vectors.

### Main API

```python
from pyturboquant import TurboQuantIndex, TurboQuantConfig

config = TurboQuantConfig(
    bits=3,           # turbo3: 3 bits per component
    use_qjl=False,    # QJL disabled (recommended)
    rotation="wht"    # Walsh-Hadamard Transform
)

index = TurboQuantIndex(dim=768, config=config)
index.add(embeddings)           # np.ndarray [N, 768]
distances, indices = index.search(query, k=10)
```

### LangChain integration

```python
from pyturboquant.langchain import TurboQuantVectorStore

vectorstore = TurboQuantVectorStore.from_documents(
    documents,
    embedding=embeddings_model,
    turbo_config=config
)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
```

### MSE distortion vs Shannon bound table

| Bits | MSE distortion | Shannon bound | Overhead |
|---|---|---|---|
| turbo2 | 0.0241 | 0.0221 | +9% |
| turbo3 | 0.0089 | 0.0083 | +7% |
| turbo4 | 0.0034 | 0.0032 | +6% |

pyturboquant operates ~6-9% from the theoretical optimal Shannon bound.

---

## 3. Inference Frameworks: Comparative Analysis

### 3.1 vLLM

**Repository:** github.com/vllm-project/vllm  
**Language:** Python/C++/CUDA  
**License:** Apache 2.0  
**Current version:** v0.19.x (April 2026)  
**GitHub Stars:** ~35k  
**Contributors:** >1500

vLLM is the reference framework for high-throughput LLM serving. Born at UC Berkeley in 2023 with PagedAttention as its founding innovation, it is today the most compatible framework for models, quantizations, and deployment ecosystems.

#### PagedAttention v2 — Internal architecture

GPU memory for the KV cache is managed like a **virtual OS for memory**:

- **Logical vs physical blocks:** each running sequence has a set of numbered logical blocks. The `BlockManager` maintains a page table mapping logical block → physical block in VRAM. Physical blocks are allocated from a memory pool pre-allocated at boot.
- **Block size:** default 16 tokens/block (configurable via `--block-size`). A single physical block contains:

$$\text{bytes\_per\_block} = \text{block\_size} \times H_{KV} \times D \times 2 \times \text{bytes\_elem}$$

Where factor 2 separates K and V layers, and `bytes_elem` depends on dtype (`turboquant` ≈ 0.5 byte/elem at turbo3, `fp8` = 1 byte/elem, `bf16` = 2 byte/elem).

- **Copy-on-Write (CoW):** beam search and parallel sampling generate sequence forks. Branches share read-only blocks marked with `ref_count > 1`. A block is physically copied only when a sequence modifies it (first write after fork). This allows managing beam width = 32 without multiplying VRAM by 32×.
- **Free list:** freed blocks are re-inserted in O(1) into the free list. Internal fragmentation is limited: at most the last partial block per sequence wastes `(block_size - 1)` slots.
- **Prefix caching:** blocks belonging to an identical prefix across different requests are cached and reused directly. The lookup key is a hash of the token content. Critical for multi-turn chat with constant system prompts.

#### Continuous Batching — Mechanism

Static batching waits for all sequences in the batch to terminate. vLLM's continuous batching uses an **iteration-level scheduler**:

1. At each forward pass (a single decoding step), the scheduler checks:
   - Which sequences have generated an EOS (end-of-sequence) token
   - Which sequences have reached `max_tokens`
2. Terminated sequences are removed immediately; their KV blocks are freed.
3. The gap is filled by queued sequences (`WAITING` → `RUNNING`) in the very next forward pass.
4. Result: GPU always saturated, no stall from heterogeneous sequences in the batch.

**Preemption:** if the KV cache runs out mid-run, vLLM can preempt (suspend) low-priority sequences: it serializes their KV blocks to CPU RAM (swap-out), frees VRAM and yields it to priority sequences. When pressure reduces, blocks are reloaded (swap-in).

Practical effect: throughput **23× compared to naive HuggingFace Transformers** (LLaMA-13B benchmark, original 2023 paper).

#### TurboQuant in vLLM — Integration architecture

The integration (April 2026) inserts TurboQuant as a KV cache backend **orthogonal** to model weight quantization. Weights can be in BF16, FP8, or AWQ independently of the KV dtype choice.

**Module structure:**
```
vllm/
├── model_executor/layers/quantization/turboquant/
│   ├── config.py        # TurboQuantConfig: k_bits, v_bits, rotation
│   ├── quantizer.py     # WHT rotation + Lloyd-Max online quantizer
│   └── centroids.py     # Pre-computed Lloyd-Max centroids for b=2,3,4,8
└── attention/backends/
    └── turboquant_attn.py   # Flash Attention backend with compressed KV
```

**Dedicated Triton kernels:**

- `triton_turboquant_store` — invoked immediately after K/V projection. Applies WHT rotation to the `head_dim`-dimensional vector, performs Lloyd-Max quantization and writes the result into the physical block of the KV cache in compressed format. Operates in-place on the KV cache pool.
- `triton_turboquant_decode` — during the decode step, loads the relevant compressed KV block, decompresses on-the-fly only the heads needed for the current query, and returns K/V tensors in BF16/FP16 for attention computation. **No full cache decompression:** only the working set of the current batch.

**CLI configuration — complete options:**
```bash
# Recommended asymmetric configuration (q8_0-K + turbo3-V)
vllm serve google/gemma-4-27b-it \
    --quantization fp8 \
    --kv-cache-dtype turboquant \
    --turboquant-k-bits 8 \
    --turboquant-v-bits 3 \
    --max-model-len 131072 \
    --tp 4 \
    --enable-prefix-caching

# Maximum savings configuration (turbo2-V, verify on head_dim)
vllm serve Qwen/Qwen3-235B-A22B \
    --kv-cache-dtype turboquant \
    --turboquant-k-bits 8 \
    --turboquant-v-bits 2 \
    --tp 8

# WARNING: dangerous symmetric configuration on head_dim=128
# --turboquant-k-bits 3 --turboquant-v-bits 3   ← DO NOT USE
```

#### Weight quantization — Full stack

| Scheme | Precision | VRAM vs BF16 | Typical speedup | Notes |
|---|---|---|---|---|
| BF16 | 16 bit | 1× | — | Max quality, baseline |
| FP8 E4M3 | 8 bit | ~0.50× | +1.3–1.8× on H100/Hopper | Default for Blackwell/Hopper |
| NVFP4 | 4 bit | ~0.25× | +2.5× on GB200 | Blackwell Ultra only |
| AWQ | 4 bit | ~0.25× | +1.4–1.5× on A100 | Best for pre-Hopper GPUs |
| GPTQ | 4 bit | ~0.25× | +1.4× on A100 | Interoperable with llama.cpp |
| GGUF | variable | variable | portable | Direct llama.cpp file import |

FP8 E4M3 quantization is recommended as default for H100/H200: vLLM uses FP8 GEMM via cuBLAS with automatic casting for attention and MLP projections, with quality degradation < 1% on standard benchmarks.

#### Speculative Decoding

vLLM integrates two strategies to reduce per-token latency:

**EAGLE v2 (Extrapolation Algorithm for Greater Language-model Efficiency):**
- A lightweight draft model (typically 1–3 layers sharing the target model's embeddings) generates K candidate tokens ahead (K=4–8, configurable).
- The target model verifies the K tokens in a single forward pass.
- Accepted tokens (up to K) become output; decoding restarts from the first rejected token.
- Typical acceptance rate: 65–75% on chat/instruction, 45–55% on code generation.
- Effective end-to-end speedup: **1.6–2.3×** at equal output quality.

**MLP Draft Model:**
- Even lighter than EAGLE: single MLP predicting the next token without attention.
- Effective for template filling and highly predictable JSON structured output.

```bash
# EAGLE v2
vllm serve Qwen/Qwen3-235B-A22B \
    --speculative-model Qwen/Qwen3-235B-A22B-EAGLE \
    --num-speculative-tokens 5 \
    --use-v2-block-manager \
    --reasoning-parser qwen3
```

#### Chunked Prefill

Chunked prefill breaks the processing of long prompts into controlled units, eliminating decode stalls during prefills of tens of thousands of tokens:

- The prompt is split into chunks of `--max-num-chunked-prefill-tokens` tokens (default 512).
- Each chunk is processed sharing the batch with decode tokens from other sequences already running.
- Result: TTFT (Time To First Token) for decode requests does not significantly degrade even with 32K+ token prefills running in parallel.
- **Difference from SGLang PD Disaggregation:** this is not hardware separation; prefill and decode share the same GPUs. It is a scheduling strategy, not an architectural one.

```bash
vllm serve MODEL \
    --enable-chunked-prefill \
    --max-num-chunked-prefill-tokens 512
```

#### Multi-LoRA Adapter Batching

vLLM is the only framework with simultaneous batching of multiple LoRA adapters in the same forward pass:

```bash
vllm serve base_model \
    --enable-lora \
    --max-loras 8 \           # up to 8 simultaneous adapters in VRAM
    --max-lora-rank 64 \
    --lora-dtype float16

# Request with specific adapter
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "adapter_finance_v2", "prompt": "...", "max_tokens": 200}'
```

LoRA matrices (A ∈ R^{d×r} and B ∈ R^{r×d}) are fused on-the-fly with a dedicated GEMM during each forward pass. No separate model copy is needed per adapter: base matrices remain shared. This allows serving, for example, 8 different enterprise clients with specialized fine-tuning using a single GPU deployment.

#### Production Ecosystem

**Ray Serve / KubeRay:**
```python
from ray import serve
from vllm.entrypoints.openai.api_server import app

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 4},
    autoscaling_config={"min_replicas": 1, "max_replicas": 8}
)
@serve.ingress(app)
class VLLMDeployment:
    pass
```

**Native Prometheus metrics** (endpoint `/metrics`):
- `vllm:num_requests_running` — sequences currently on GPU
- `vllm:gpu_cache_usage_perc` — % KV cache pool utilization
- `vllm:time_to_first_token_seconds` — TTFT latency distribution (histogram)
- `vllm:time_per_output_token_seconds` — TPOT distribution
- `vllm:num_preemptions_total` — preemption counter for KV cache pressure

These metrics drive the Kubernetes HPA (Horizontal Pod Autoscaler) via custom metrics adapter (Prometheus Adapter or KEDA).

#### Strengths

- **Maximum model compatibility:** 200+ supported architectures, the broadest list of all frameworks
- **Native TurboQuant:** compressed KV cache directly integrated with PagedAttention v2
- **Multi-adapter LoRA batching:** unique capability; other frameworks do not support multiple adapters in a single forward pass
- **Chunked prefill:** reduces TTFT variance without additional hardware
- **Speculative decoding EAGLE v2:** up to 2.3× speedup on chat without quality loss
- **Mature ecosystem:** native integration with Ray, KubeRay, SkyPilot, LiteLLM, Prometheus/Grafana
- **Granular monitoring:** TTFT and TPOT metrics for enterprise SLOs

#### Limitations

- **Boot memory footprint:** vLLM pre-allocates the entire KV cache pool at launch (controllable via `--gpu-memory-utilization`, default 0.90). On large models it may leave little VRAM for weights if not sized correctly.
- **Performance on large-scale MoE:** SGLang is significantly faster for MoE architectures on multi-node clusters thanks to DeepEP/DeepGEMM and full PD Disaggregation in production.
- **Scaling beyond 8 nodes:** supported via pipeline parallelism but less optimized than SGLang for EP at 32–100+ nodes.

#### Ideal scenario

1-8 GPU nodes, need for broad model compatibility, simultaneous serving of multiple LoRA adapters, TurboQuant KV cache on standard GPUs, integration into existing Kubernetes/Ray ecosystems.

---

### 3.2 SGLang

**Repository:** github.com/sgl-project/sglang  
**Language:** Python (82.7%), Rust (7.6%), CUDA (4.5%), C++ (3.7%)  
**License:** Apache 2.0  
**Current version:** v0.5.10.post1 (April 2026)  
**GitHub Stars:** 26k  
**Contributors:** 1381  
**Adopted by:** xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS — **400,000 GPUs in production**

SGLang is the next-generation high-performance framework, developed by LMSYS (UC Berkeley). It is the de facto industrial standard for enterprise deployment of large MoE models on multi-node clusters.

#### RadixAttention — Advanced Prefix Caching

vLLM's prefix caching mechanism uses a flat hash to identify reusable blocks. SGLang introduces **RadixAttention**: the KV cache of all active prefixes is organized in a **radix tree** (trie), where each node represents a token sequence and its corresponding KV cache.

**How it works:**
1. Each new request is compared against the paths of the radix tree.
2. The longest node matching the request's prefix is found in O(prefix).
3. That node's KV cache is reused directly, without recomputation.
4. Only new tokens (after the match) are computed.
5. The result is inserted as a new branch in the tree.

**Advantages over flat prefix caching:**
- Efficient handling of **partially overlapping prefixes** (e.g., similar but not identical system prompts).
- Fine-grained sharing: two requests with a common preamble of 2000 tokens diverging at token 2001 share exactly 2000 tokens of KV cache, not 0 or the entire sequence.
- Eviction policy: LRU (Least Recently Used) on tree nodes, not on individual blocks.

**Impact on enterprise RAG:** queries with similar retrieval context (same document chunks) reuse the chunk KV cache, reducing prefill latency by 40–70% in high hit-rate scenarios.

#### Zero-Overhead CPU Scheduler

SGLang's scheduler operates entirely in async Python with an event loop that introduces no GPU bubble. Unlike schedulers that use `torch.cuda.synchronize()` to verify completion, SGLang uses:
- **Async streaming:** generated tokens are transmitted via async generator without sync barriers.
- **Detached forward pass:** the GPU forward pass is queued to the CUDA stream; the scheduler does not wait for completion before preparing the next batch.

This eliminates the "scheduler overhead gap" that appears in other frameworks as a 5–20ms bubble between consecutive forward passes on small batches.

#### Prefill-Decode (PD) Disaggregation

This is SGLang's most impactful architectural innovation for enterprise cluster deployment.

**The problem:** during the prefill phase, the GPU is compute-bound (must process all prompt tokens in parallel); during the decode phase, the GPU is memory-bandwidth-bound (generates one token at a time, bottleneck is reading the KV cache). These two phases have opposite hardware requirements.

**The PD Disaggregation solution:**
- A dedicated pool of GPUs (prefill workers) handles exclusively prefill requests.
- A separate pool (decode workers) handles exclusively token-by-token generation.
- After prefill, the prompt KV cache is transferred from the prefill worker to the assigned decode worker via **non-blocking RDMA** (Remote Direct Memory Access over InfiniBand or RoCE v2).
- The transfer happens in the background while the prefill worker already starts the next prefill.

**Optimal configuration for 12 H100 nodes (DeepSeek V3):**
```bash
# 4 prefill nodes × 8 H100 — EP32
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 32 \
    --disaggregation-mode prefill \
    --moe-dense-tp-size 1 \
    --quantization fp8

# 8 decode nodes × 8 H100 — EP64
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 64 \
    --disaggregation-mode decode \
    --enable-two-batch-overlap \
    --enable-eplb \
    --quantization fp8
```

The 1:2 ratio (prefill:decode) is empirically optimal for DeepSeek V3 with input ≈ 2K tokens and output ≈ 512 tokens. In scenarios with very long outputs, the ratio shifts toward 1:4.

#### Expert Parallelism (EP) and DeepEP

For MoE models (DeepSeek V3, Qwen3.5 MoE), each forward pass includes an **all-to-all dispatch** phase: tokens are routed to active experts (for DeepSeek: top-8 out of 256 experts).

**Tensor Parallelism (TP) baseline:**
- Each GPU replicates all experts in sharded FP8.
- All-to-all happens GPU-to-GPU via NVLink.
- Bottleneck: number of experts per GPU is limited by VRAM.

**Expert Parallelism (EP) with DeepEP:**
- Experts are distributed: each GPU "owns" `num_experts / ep_size` experts.
- Token routing all-to-all happens over InfiniBand (RDMA) between nodes.
- **DeepEP** is the custom library from SGLang/DeepSeek that optimizes this all-to-all:
  - Dedicated CUDA kernels for token dispatch pack/unpack.
  - Dispatch pipelining over multiple RDMA queues.
  - Fusion of expert normalization with dispatch to reduce kernel launches.
- With EP64 on 8 nodes, each node manages 4 experts (256/64): reduced expert memory footprint, increased expert bandwidth since more GPUs work in parallel.

**EP scaling:** EP32 for prefill (compute-bound, all-to-all is expensive), EP64 for decode (memory-bound, all-to-all more frequent but on small KV cache per step).

#### EPLB — Expert Parallelism Load Balancer

MoE routing presents a load imbalance problem: token distribution to experts is not uniform. Some experts ("hot experts") receive 5–10× more tokens than others for certain input types. With pure EP, GPUs hosting hot experts become bottlenecks.

**EPLB algorithm:**
1. SGLang monitors the token count per expert at runtime (sliding window of N steps).
2. Experts with load > threshold are **replicated** on additional GPUs (hot experts have multiple replicas).
3. The MoE router is updated to distribute the replicated expert's traffic among replicas.
4. Cold experts can be consolidated (co-located) on a single GPU.
5. Rebalancing occurs without stopping serving (hot reload of replicas).

**Measured impact:**
- Prefill: +**1.49×** compared to EP without EPLB
- Decode: +**2.54×** compared to EP without EPLB (decode is more susceptible because imbalance accumulates over thousands of steps)

#### Two-Batch Overlap (TBO)

During decode with EP, each step has two serial phases:
1. **Compute phase:** GEMM for attention and dense MLP layers.
2. **Communication phase:** all-to-all dispatch/combine of expert tokens over RDMA.

With TBO, the two phases overlap across two alternating batches (A and B):
- While batch A is in communication (all-to-all), batch B is in compute (GEMM).
- While batch B is in communication, batch A is in compute.

Synchronization is managed via separate CUDA streams and fine-grained barriers. Effectiveness depends on all-to-all latency relative to GEMM time:

**Measured impact:**
- Prefill: +27% to +35% throughput
- Decode at batch size 256: +25.5% throughput
- Reduced effect at small batch sizes (< 32): the two phases don't overlap effectively when GEMM is too short.

#### Multi-Token Prediction (MTP)

SGLang supports MTP for models that implement it natively (e.g., DeepSeek V3 with optional MTP head). In MTP, the model predicts the next K tokens in a single forward pass via specialized heads:

- **It is not speculative decoding:** no separate draft model is needed. The main model has additional heads trained to predict tokens t+1, t+2, ... t+K.
- The K−1 extra tokens are verified in the next forward pass with a very high acceptance rate (≈ 85-90% for DeepSeek V3 with K=2).
- Effective speedup: up to +40% throughput on decode-heavy workloads.

```bash
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --enable-mtp \
    --mtp-num-token 2 \
    --tp 8
```

#### Memory Management: Chunked Prefill and Memory Pool

SGLang manages the KV cache via a memory pool based on a **slab allocator**:
- KV blocks are allocated in fixed-size slabs (default 256 tokens).
- The RadixAttention tree operates on these slabs with LRU policy.
- Chunked prefill (active by default) breaks long prefills into 8192-token chunks to prevent a single prefill from monopolizing the GPU → stable decode latency even with 128K token prompts.

#### NVIDIA Dynamo Integration (GB200/GB300)

On Blackwell Ultra hardware, SGLang integrates with NVIDIA Dynamo for KV-aware routing:

- **HiCache:** distributed cross-node radix tree that allows decode workers on different nodes to share prefill KV cache (avoids RDMA re-transfer if the same prompt arrives on different decode workers).
- **NIXL/Mooncake:** libraries for KV cache transfer over the NVL72 rack's NVLink Switch fabric (bandwidth: 900 GB/s vs ~200 GB/s InfiniBand).
- **NVFP4 GEMM:** expert weights in NVFP4 (NVIDIA 4-bit floating point) with GEMM on Blackwell's native FP4 Tensor Cores (throughput ~4× FP8).
- **Single-Batch Overlap (SBO):** TBO variant optimized for GB300 where the NVLink Switch width makes communication overhead negligible at batch size < 64, and overlap focuses on memory bandwidth instead of network.

**Performance on GB300 NVL72:** 25× compared to H200 @ 50 TPS/user (InferenceXv2, February 2026).

#### Measured performance

| Configuration | Throughput | Hardware |
|---|---|---|
| DeepSeek V3, TP baseline | 1× (baseline) | 12 H100 nodes |
| + PD Disaggregation + EP | **5.2× decode** | 12 H100 nodes |
| + EPLB on decode | **+2.54× vs EP without EPLB** | 12 H100 nodes |
| + Two-Batch Overlap | **+27–35% prefill, +25.5% decode** | 12 H100 nodes |
| SGLang on GB200 NVL72 | **8× vs version 4 months prior** | rack-scale |
| SGLang on GB300 NVL72 | **25× vs H200** | rack-scale NVL72 |

**Absolute numbers for DeepSeek V3 (96 H100, 12 nodes):**
- Prefill: **57,674 tokens/s per node** (1K token input)
- Decode: **22,282 tokens/s per node** (batch 256 seq, 2K input)
- Estimated cost: **$0.20/M output tokens** (vs $1/M official DeepSeek API)

#### Strengths

- **Best-in-class MoE performance:** unique combination of PD Disagg. + EP + EPLB + TBO
- **RadixAttention:** fine-grained prefix caching critical for high hit-rate RAG
- **Industrial standard:** deployed on 400,000 GPUs across major cloud providers
- **Training backbone:** used by verl, AReaL, Tunix (Google) for RL training on frontier LLMs
- **Hardware range:** from H100 clusters to GB300 racks with specific optimizations

#### Limitations

- Steeper learning curve for multi-node PD + EP configurations
- TurboQuant KV cache: not available (uses FP8/NVFP4 as alternative); for context ≤ 64K this is less relevant thanks to RadixAttention prefix caching
- Scheduling overhead on single-GPU deployments compared to lighter solutions

#### Ideal scenario

Multi-node clusters (≥4 nodes), large MoE models (DeepSeek V3, Qwen3.5 397B), maximum production throughput, integration with NVIDIA Dynamo for GB200/GB300 racks.

---

### 3.3 Comparison table

| Dimension | vLLM | SGLang |
|---|---|---|
| **Language** | Python/C++/CUDA | Python/Rust/CUDA |
| **TurboQuant KV** | ✅ Native (turboquant_attn) | ❌ (FP8/NVFP4 as alternative) |
| **PagedAttention** | ✅ v2 with CoW | ✅ slab allocator |
| **Prefix caching** | ✅ hash-based flat | ✅✅ RadixAttention (trie, more efficient) |
| **Chunked prefill** | ✅ (`--enable-chunked-prefill`) | ✅ (active by default) |
| **Optimized MoE** | Partial | ✅✅ DeepEP + EPLB |
| **PD Disaggregation** | Experimental (chunked) | ✅ In production (RDMA) |
| **Multi-node EP** | Partial TP/PP | ✅ EP up to 100+ nodes |
| **Two-Batch Overlap** | ❌ | ✅ (+27–35% prefill) |
| **Speculative decoding** | ✅ EAGLE v2, MLP draft | ✅ MTP native (no draft model) |
| **Multi-Token Prediction** | ❌ | ✅ (native MTP heads) |
| **Multi-adapter LoRA** | ✅✅ Only with simultaneous batching | ✅ (max 1 adapter per batch) |
| **MoE throughput** | Good | **Best-in-class** |
| **Hardware** | CUDA/ROCm/TPU | CUDA/ROCm/TPU/NPU/Ascend |
| **NVIDIA GB300/Dynamo** | Partial | ✅ Native integration |
| **Monitoring** | ✅ Prometheus/Grafana + TTFT/TPOT | ✅ Prometheus/Grafana |
| **Community** | Large (>1500 contrib.) | Large (1381 contrib.) |
| **GitHub Stars** | ~35k | 26k |
| **Enterprise adoption** | High, mature ecosystem | **Industrial standard (400k GPUs)** |
| **Ideal scenario** | 1-8 nodes, multi-LoRA, TurboQuant KV | Scalable cluster, large-scale MoE |

---

## 4. Quality degradation: measured data

### KV Cache: PPL (Perplexity) with TurboQuant

Community benchmark data on Llama 3.1 8B (BF16 baseline PPL ≈ 6.0):

| K-cache configuration | V-cache configuration | PPL | Degradation |
|---|---|---|---|
| BF16 (baseline) | BF16 (baseline) | 6.0 | — |
| q8_0 | q8_0 | 6.01 | +0.2% |
| **q8_0** | **turbo3** | **6.08** | **+1.3%** ✅ recommended |
| turbo4 | turbo4 | 6.15 | +2.5% |
| turbo3 | turbo3 | 6.31 | +5.2% |
| turbo3 | turbo3 (D=128) | 3400+ | **Catastrophic** ❌ |

**Warning:** the symmetric `turbo3/turbo3` configuration on models with `head_dim=128` causes catastrophic degradation. Always use the asymmetric configuration `q8_0-K + turbo3-V`.

### NIAH (Needle-in-a-Haystack): results

| Context | Top-1 accuracy BF16 | Top-1 accuracy turbo3-V | Delta |
|---|---|---|---|
| 8K tokens | 99.2% | 98.7% | -0.5% |
| 32K tokens | 97.4% | 96.1% | -1.3% |
| 128K tokens | 89.1% | 85.3% | -4.8% |

The "Achilles' heel" identified by the community is not quality but **long-context performance**: at 128K tokens, decode speed drops to 10-20% of baseline due to TurboQuant decompression overhead (not quality degradation).

### RAG Embeddings with pyturboquant

| Bits | Recall@10 (FAISS baseline) | Recall@10 (TQ) | Delta |
|---|---|---|---|
| turbo4 | 91.2% | 90.8% | -0.4% |
| turbo3 | 91.2% | 89.6% | -1.8% |
| turbo2 | 91.2% | 84.1% | -7.8% |

---

## 5. KV Cache calculations: Gemma 4

### Gemma 4 architecture

- **Attention pattern:** 6:1 — 5 sliding window : 1 full attention
- **Sliding window:** 4,096 tokens
- **Global attention head dim:** 512
- **Maximum context:** 256K tokens

### VRAM calculation formula for KV cache

$$\text{VRAM}_{KV} = 2 \times L \times S \times H_{KV} \times D \times \text{bytes\_per\_element}$$

Where:
- $L$ = number of layers
- $S$ = sequence length (tokens)
- $H_{KV}$ = number of KV heads
- $D$ = head dimension
- Factor 2 = K + V separate

### Results per model (128K context, BF16 baseline)

| Model | Layers | KV Heads | Head Dim | VRAM BF16 | VRAM q8_0-K+turbo3-V | Savings |
|---|---|---|---|---|---|---|
| Gemma 4 E2B (2B) | 26 | 2 | 256 | ~3.2 GB | ~1.4 GB | -56% |
| Gemma 4 4B | 34 | 4 | 256 | ~8.4 GB | ~3.7 GB | -56% |
| Gemma 4 12B | 42 | 4 | 256 | ~20.4 GB | ~9.0 GB | -56% |
| Gemma 4 27B | 62 | 8 | 512 | ~98.7 GB | **~43.4 GB** | -56% |

> **Note:** for Gemma 4 27B at 256K context, the BF16 KV cache would require ~197 GB — impractical on standard hardware. With `q8_0-K + turbo3-V` it drops to ~87 GB, making dual-GPU configurations feasible.

### Model weight calculation (separate from KV cache)

| Model | BF16 weights | Q4_K_M weights (GGUF) |
|---|---|---|
| Gemma 4 E2B | ~4 GB | ~1.3 GB |
| Gemma 4 4B | ~8 GB | ~2.7 GB |
| Gemma 4 12B | ~24 GB | ~7.8 GB |
| Gemma 4 27B | ~54 GB | ~17.5 GB |

**Recommended total VRAM budget (Gemma 4 27B, 64K context, Q4_K_M weights + turbo3-V KV):**
~17.5 GB (weights) + ~21.7 GB (KV cache) ≈ **~40 GB** → feasible on a single RTX 4090 or A6000.

---

## 6. Best open-weights models for Enterprise (April 2026)

Source: ArtificialAnalysis Intelligence Index (April 2026).

### Tier 1 — Maximum quality (multi-GPU clusters)

| Model | Score | Active parameters | License | Context | API price |
|---|---|---|---|---|---|
| GLM-5.1 | 51 | n/a | Apache 2.0 | 128K | ~$1.00/M |
| GLM-5 | 50 | n/a | Apache 2.0 | 128K | ~$0.80/M |
| MiniMax-M2.7 | 50 | MoE | Proprietary | 128K | ~$0.80/M |
| Qwen3.6 Plus | 50 | MoE | Apache 2.0 | 262K | n/a |
| **Qwen3.5 397B A17B** | 45 | **17B active** | **Apache 2.0** | **262K** | $1.35/M |
| **DeepSeek V3.2** | 42 | ~671B (MoE) | DeepSeek Lic. | 128K | **$0.32/M** |

### Tier 2 — Balanced (single VM or small cluster)

| Model | Score | VRAM (dense equiv.) | License | Notes |
|---|---|---|---|---|
| Qwen3.5 35B A3B | 43 | ~7B active | Apache 2.0 | Very fast MoE, 262K ctx |
| Qwen3.5 27B | 42 | ~55 GB BF16 | Apache 2.0 | Dense, excellent RAG, 262K ctx |
| **Gemma 4 27B** | ~38 | ~54 GB BF16 | Gemma Lic. | 256K ctx, KV cache calculated above |
| NVIDIA Nemotron 3 Super | 36 | ~70B | NVIDIA Open | **1M context**, agent-optimized |
| gpt-oss-120B | 33 | ~120B dense | OpenAI Open | OpenAI open-source |

### Enterprise license considerations

| Model | Commercial use | API redistribution | Fine-tuning | Notes |
|---|---|---|---|---|
| Qwen3.x | ✅ Free | ✅ | ✅ | Apache 2.0 total freedom |
| DeepSeek V3.2 | ✅ | ❌ no API redistrib. | ✅ | Clause only on API serving |
| GLM-5 | ✅ | ✅ | ✅ | Apache 2.0 |
| Gemma 4 | ✅ (>100M MAU requires agreement) | Specific clauses | ✅ | Gemma Terms of Use |
| gpt-oss-120B | To be verified | To be verified | To be verified | New OpenAI license |

### Primary on-premise recommendation

**Qwen3.5 27B** (or MoE variant Qwen3.5 35B A3B):
- Apache 2.0: zero enterprise legal restrictions
- Dual-mode thinking/non-thinking via `enable_thinking` flag (no separate model needed)
- Native tool-calling for RAG agents
- Native support in vLLM (`>=0.8.5`) and SGLang (`>=0.4.6.post1`)

---

## 7. Recommended Deployment Architectures

### Scenario A — Single VM, GPU with VRAM ≤48 GB

**Framework:** vLLM + TurboQuant KV cache  
**Model:** Gemma 4 12B or Qwen3.5 27B  
**KV cache:** `q8_0-K + turbo3-V` (~56% VRAM savings vs BF16)

```bash
# vLLM with TurboQuant: FP8 weights + compressed KV cache
vllm serve google/gemma-4-12b-it \
    --quantization fp8 \
    --kv-cache-dtype turboquant \
    --turboquant-k-bits 8 \
    --turboquant-v-bits 3 \
    --max-model-len 65536 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching

# For Qwen3.5 27B on dual-GPU (e.g. 2× RTX 4090)
vllm serve Qwen/Qwen3-27B-Instruct \
    --quantization awq \
    --kv-cache-dtype turboquant \
    --turboquant-k-bits 8 \
    --turboquant-v-bits 3 \
    --tp 2 \
    --max-model-len 131072 \
    --reasoning-parser qwen3
```

**Embedding/RAG sidecar:**
```python
from pyturboquant.langchain import TurboQuantVectorStore
vectorstore = TurboQuantVectorStore.from_documents(
    docs, embedding=embed_model,
    turbo_config=TurboQuantConfig(bits=3, use_qjl=False)
)
```

### Scenario B — VM with 8× GPUs (H100/A100)

**Framework:** vLLM (now with native TurboQuant) or SGLang  
**Model:** Qwen3.5 27B or DeepSeek V3.2 (MoE)

```bash
# vLLM with TurboQuant KV cache
vllm serve Qwen/Qwen3-235B-A22B \
    --tp 8 \
    --quantization fp8 \
    --kv-cache-dtype turboquant \
    --enable-prefix-caching \
    --reasoning-parser qwen3

# SGLang (preferred for MoE)
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --quantization fp8 \
    --enable-prefix-caching \
    --reasoning-parser deepseek_r1
```

### Scenario C — Multi-node cluster (≥12 H100 nodes), maximum throughput

**Framework:** SGLang with PD Disaggregation + EP  
**Model:** DeepSeek V3.2 or Qwen3.5 397B A17B

```bash
# Prefill server (4 nodes × 8 H100, EP32)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 32 \
    --disaggregation-mode prefill \
    --moe-dense-tp-size 1 \
    --quantization fp8

# Decode server (8 nodes × 8 H100, EP64)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 64 \
    --disaggregation-mode decode \
    --enable-two-batch-overlap \
    --enable-eplb
```

**Expected result:** ~22,000 tokens/s per node, ~$0.20/M output tokens.

### Scenario D — Rack-scale, NVIDIA GB200/GB300

**Framework:** SGLang + NVIDIA Dynamo  
**Model:** DeepSeek V3.2 with NVFP4

Specific optimizations for Blackwell Ultra:
- NVFP4 GEMM for MoE experts and dense layers
- Single-batch overlap (instead of classic TBO)
- KV-aware routing via NVIDIA Dynamo + HiCache radix tree
- NIXL/Mooncake for KV cache transfer

**Measured performance:** 25× compared to H200 @ 50 TPS/user (InferenceXv2, February 2026).

### Recommended docker-compose (Scenario B)

```yaml
version: "3.9"

services:
  llm:
    image: lmsysorg/sglang:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: >
      python -m sglang.launch_server
      --model-path Qwen/Qwen3-235B-A22B
      --tp 8
      --quantization fp8
      --enable-prefix-caching
      --reasoning-parser qwen3
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface

  embeddings:
    image: pyturboquant:latest
    environment:
      - MODEL_NAME=BAAI/bge-m3
      - TURBO_BITS=3
    ports:
      - "8001:8001"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

---

## 8. Conclusions and Final Recommendations

### Decision matrix by scenario

```
What is the scale of deployment?
│
├── Single VM / small cluster (1-8 GPUs)
│   ├── Need simultaneous multi-LoRA adapters? → vLLM + LoRA batching
│   ├── Long context (>32K) with limited VRAM? → vLLM + TurboQuant q8_0-K/turbo3-V
│   └── General use, broad model compatibility? → vLLM (FP8 + prefix caching)
│
└── Multi-node cluster (≥4 nodes, large MoE LLM)
    ├── MoE (DeepSeek V3, Qwen3.5 397B) → SGLang + EP + PD Disaggregation
    ├── Maximum production throughput → SGLang + EPLB + TBO
    └── NVIDIA GB200/GB300 hardware → SGLang + NVIDIA Dynamo + NVFP4
```

### Summary of recommendations

| Priority | Framework | Model | Reason |
|---|---|---|---|
| **Limited VRAM + long context** | vLLM | Gemma 4 27B or Qwen3.5 27B | Native TurboQuant: -56% KV cache VRAM |
| **Multi-LoRA adapters** | vLLM | Any | Only framework with simultaneous LoRA batching |
| **Quality + cluster simplicity** | vLLM | Qwen3.5 27B | Mature ecosystem, EAGLE v2, Prometheus |
| **Enterprise MoE throughput** | SGLang | DeepSeek V3.2 | PD+EP+EPLB+TBO: industrial standard |
| **Multi-agent, very long contexts** | SGLang | Nemotron 3 Super | 1M context, RadixAttention with high hit-rate |
| **Optimal cost/performance** | SGLang | Qwen3.5 35B A3B | MoE: only 3B active params per token, score 43 |
| **Rack-scale GB200/GB300** | SGLang + Dynamo | DeepSeek V3.2 NVFP4 | 25× H200 @ 50 TPS/user |

### Final note on TurboQuant in vLLM

The discovery that vLLM has natively integrated TurboQuant (module `turboquant`, backend `turboquant_attn`, Triton kernels `triton_turboquant_store`/`triton_turboquant_decode`) eliminates the only reason one would have considered a minor framework (inferrs) in enterprise scenarios. With vLLM v0.19.x it is possible to use `--kv-cache-dtype turboquant` with the recommended asymmetric configuration, achieving:

- -56% KV cache VRAM vs BF16
- Compatibility with all 200+ models supported by vLLM
- Transparent integration with PagedAttention v2, prefix caching, chunked prefill
- Continuity with the existing Ray/KubeRay/Prometheus ecosystem

The winning combination for 2026 enterprise deployment remains **vLLM for VMs/compact clusters** (with TurboQuant where needed) and **SGLang for MoE scaling on large clusters**, with the understanding that neither framework has superseded the other: they have complementary and different strengths.

---

*Report generated: April 18, 2026*

---

## Resources and Links

### TurboQuant

| Resource | URL |
|---|---|
| Original paper — arXiv:2504.19874 | https://arxiv.org/abs/2504.19874 |
| pyturboquant — PyPI | https://pypi.org/project/pyturboquant/ |
| pyturboquant — GitHub | https://github.com/google/pyturboquant |

### vLLM

| Resource | URL |
|---|---|
| GitHub Repository | https://github.com/vllm-project/vllm |
| Official documentation | https://docs.vllm.ai/en/latest/ |
| Quickstart | https://docs.vllm.ai/en/latest/getting_started/quickstart.html |
| PagedAttention — original paper (arXiv:2309.06180) | https://arxiv.org/abs/2309.06180 |
| EAGLE v2 — paper (arXiv:2406.16858) | https://arxiv.org/abs/2406.16858 |
| Supported models | https://docs.vllm.ai/en/latest/models/supported_models.html |
| Quantization docs | https://docs.vllm.ai/en/latest/quantization/supported_hardware.html |
| Multi-LoRA serving | https://docs.vllm.ai/en/latest/features/lora.html |
| Chunked prefill | https://docs.vllm.ai/en/latest/features/chunked_prefill.html |
| Speculative decoding | https://docs.vllm.ai/en/latest/features/spec_decode.html |
| Prometheus metrics | https://docs.vllm.ai/en/latest/deployment/metrics.html |

### SGLang

| Resource | URL |
|---|---|
| GitHub Repository | https://github.com/sgl-project/sglang |
| Official documentation | https://sgl-project.github.io/start/install.html |
| LMSYS Blog — Large-Scale EP with DeepSeek V3 (2025) | https://lmsys.org/blog/2025-05-05-large-scale-ep/ |
| LMSYS Blog — SGLang on GB300 NVL72 (2026) | https://lmsys.org/blog/2026-02-gb300-nvl72/ |
| RadixAttention — paper (arXiv:2312.07104) | https://arxiv.org/abs/2312.07104 |
| DeepEP — GitHub | https://github.com/deepseek-ai/DeepEP |
| NVIDIA Dynamo — SGLang integration | https://developer.nvidia.com/blog/nvidia-dynamo-open-source-inference/ |

### Models

| Resource | URL |
|---|---|
| Qwen3 — HuggingFace | https://huggingface.co/Qwen |
| DeepSeek V3 — HuggingFace | https://huggingface.co/deepseek-ai/DeepSeek-V3 |
| Gemma 4 — HuggingFace | https://huggingface.co/google/gemma-4-27b-it |
| NVIDIA Nemotron 3 Super — HuggingFace | https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1 |
| GLM-5 — HuggingFace | https://huggingface.co/THUDM |
| MiniMax-M2 — HuggingFace | https://huggingface.co/MiniMaxAI |
| ArtificialAnalysis Intelligence Index | https://artificialanalysis.ai/leaderboards/models |
