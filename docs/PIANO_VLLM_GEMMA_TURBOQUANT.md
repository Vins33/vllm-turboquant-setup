# Piano di Implementazione: vLLM + TurboQuant — Gemma 4 27B su RTX 5090

> **Aggiornato**: Aprile 2026 — basato su vLLM-TurboQuant fork (mitkox/vllm-turboquant)

## 📋 Sommario Esecutivo

Guida per servire Gemma 4 27B tramite il fork vLLM-TurboQuant su una NVIDIA RTX 5090.
TurboQuant comprime la **KV cache** (non i pesi) riducendo il consumo di VRAM durante
l'inferenza e abilitando context window più lunghe. I pesi del modello vanno
pre-quantizzati separatamente (AWQ INT4 o FP8) per entrare nei 32 GB della 5090.

> ⚠️ **Nota compatibilità**: il fork dichiara supporto ufficiale per SM86 (RTX A6000)
> e SM121 (GB10/DGX Spark). La RTX 5090 (GB202) usa **SM120**. Le sezioni dedicate
> descrivono come procedere e i possibili workaround.

---

## 🎯 Obiettivi Tecnici

| Aspetto | Target | Note |
|---------|--------|------|
| **Modello** | Google Gemma 4 27B (`-it`) | Multimodale vision+text |
| **Pesi** | AWQ INT4 o FP8 | ~14 GB o ~27 GB — per entrare su 32 GB VRAM |
| **KV Cache** | TurboQuant (`turboquant35`) | Riduce KV cache del 50-65% |
| **Hardware** | NVIDIA RTX 5090 | 32 GB GDDR7, Blackwell SM120 |
| **Framework** | vLLM-TurboQuant fork v0.19+ | Build da sorgente obbligatorio |
| **API** | OpenAI-compatible | `/v1/chat/completions`, `/v1/completions` |

---

## 📊 Analisi Hardware

### RTX 5090 — Specifiche Rilevanti
| Parametro | Valore |
|-----------|--------|
| VRAM | 32 GB GDDR7 |
| Bandwidth | ~1.79 TB/s |
| Compute Capability | **SM 12.0** (Blackwell GB202) |
| TDP | 575 W |
| Driver minimo | 570+ |
| CUDA minimo | 12.8 |

### Memory Budget per Gemma 4 27B (singola GPU)

```
Pesi modello:
  FP16          ~54 GB   ❌ non entra
  FP8           ~27 GB   ⚠️  stretto (restano ~5 GB per KV+overhead)
  AWQ INT4      ~14 GB   ✅ comodo (restano ~15 GB per KV+overhead)

KV Cache con TurboQuant (su pesi AWQ INT4):
  turboquant35  ~2-4 GB per context lungo  ← consigliato
  turboquant25  ~1-2 GB per context lungo  ← max compressione

Layout raccomandato (AWQ INT4 + turboquant35):
  Pesi AWQ:     ~14 GB
  KV Cache TQ:  ~4-6 GB
  Activations:  ~2 GB
  vLLM overhead:~2 GB
  ─────────────────────
  Totale:       ~22-24 GB  ✅ (68-75% dei 32 GB)
```

**Conclusione**: su RTX 5090 serve obbligatoriamente un modello con pesi
quantizzati (AWQ INT4 consigliato). TurboQuant comprime poi ulteriormente la KV cache.

---

## ⚠️ Compatibilità SM120 — Analisi Completa e Patch

### Perché SM121 è supportato ma SM120 no?

Analizzando il codice sorgente del fork, il blocco **non è nel CMakeLists.txt** (che
già include SM120 in `CUDA_SUPPORTED_ARCHS` dal CUDA 12.8+), ma in un **controllo
Python a runtime** in `vllm/v1/attention/ops/turboquant_kv_cache.py`:

```python
# File: vllm/v1/attention/ops/turboquant_kv_cache.py
TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 1)))
#                                                    ^SM86   ^SM121
# RTX 5090 = SM120 = (12, 0) → NON è in questo set → TurboQuant rifiuta
```

**Motivo dell'esclusione**: il progetto è sviluppato dall'autore (mitkox) su macchine
specifiche. Da commit e documenti si evince che ha accesso a:
- RTX A6000 (SM86) — GPU workstation Ampere/Ada
- GB10 / DGX Spark (SM121) — chip Blackwell del Project DIGITS

La RTX 5090 (SM120, Blackwell consumer GB202) **non è mai stata in mano
all'autore** per test, quindi non è stata aggiunta alla allowlist. Il CMakeLists
gestisce SM120 correttamente (vedi la sezione `scaled_mm_c3x_sm120`), ma manca
solo questo controllo Python.

I kernel Triton del fork usano già una configurazione generica per SM121 che è
identicamente valida per SM120 (stesso microarchitecture Blackwell NVB, stessa
struttura warp, same SM compute):

```python
# In get_turboquant_kernel_meta(): SM86 ha config custom, tutti gli altri
# (incluso SM121 e SM120) usano già la stessa config generica Blackwell:
return TurboQuantKernelMeta(
    decode_block_n=8 if head_size >= 256 else 16,
    decode_num_warps=4, update_tile=32, update_num_warps=4,
    postprocess_num_warps=4,
)
```

### Patch: una sola riga Python (nessuna modifica a CMakeLists)

```python
# File: vllm/v1/attention/ops/turboquant_kv_cache.py
# PRIMA:
TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 1)))

# DOPO (aggiungere (12, 0) per SM120 / RTX 5090):
TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 0), (12, 1)))
```

### Opzioni in ordine di priorità

| Opzione | Descrizione | Difficoltà |
|---------|-------------|------------|
| **A — Patch Python** | Modifica 1 riga in `turboquant_kv_cache.py` + test | **Minima** ✅ |
| **B — FP8 fallback** | Usa `--kv-cache-dtype fp8` senza modifiche | Minima |
| **C — Contribuire upstream** | Aprire PR su mitkox/vllm-turboquant | Media |

**Strategia consigliata**: applica la **Patch A** (vedi Fase 2 dedicata).

---

## 🔧 Fasi di Implementazione

### **FASE 1: Setup Ambiente (2-3 ore)**

#### 1.1 Prerequisiti Sistema (Linux)
```bash
# Verificare GPU e driver
nvidia-smi
# Output atteso: RTX 5090, Driver 570+, CUDA 12.8+

# Verificare compute capability
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# Output atteso per RTX 5090: (12, 0)

# Dipendenze di sistema
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv build-essential git
```

#### 1.2 Creare Virtual Environment con uv (consigliato)
```bash
# Installare uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Creare venv con Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

#### 1.3 Configurare Variabili Ambiente
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_MAIN_CUDA_VERSION=12.8
export CUDA_VISIBLE_DEVICES=0  # Singola RTX 5090
```

---

### **FASE 2: Build vLLM-TurboQuant da Sorgente + Patch SM120 (1-2 ore)**

> ⚠️ **Obbligatorio**: non usare wheel PyPI — i kernel TurboQuant richiedono
> compilazione CUDA nativa per l'architettura target.

#### 2.1 Clone Repository
```bash
git clone https://github.com/mitkox/vllm-turboquant.git
cd vllm-turboquant
```

#### 2.2 Patch SM120 — 1 riga Python (da fare PRIMA del build)

> **Perché serve**: il CMakeLists.txt **già include SM120** in `CUDA_SUPPORTED_ARCHS`
> da CUDA 12.8+. Il blocco è solo in Python: una `frozenset` di architetture
> approvate a runtime in `vllm/v1/attention/ops/turboquant_kv_cache.py`.

```bash
# Vedere la riga attuale (solo SM86 e SM121):
grep "TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES" \
    vllm/v1/attention/ops/turboquant_kv_cache.py
# Output:
# TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 1)))

# Applicare la patch (aggiunge SM120 = RTX 5090):
sed -i 's/frozenset(((8, 6), (12, 1)))/frozenset(((8, 6), (12, 0), (12, 1)))/' \
    vllm/v1/attention/ops/turboquant_kv_cache.py

# Verificare la modifica:
grep "TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES" \
    vllm/v1/attention/ops/turboquant_kv_cache.py
# Output atteso:
# TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 0), (12, 1)))
```

In alternativa modifica manuale con qualsiasi editor:

```python
# File: vllm/v1/attention/ops/turboquant_kv_cache.py  (riga ~30)
# PRIMA:
TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 1)))

# DOPO:
TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES = frozenset(((8, 6), (12, 0), (12, 1)))
```

> **Perché è sicuro**: i kernel Triton di TurboQuant usano già la stessa configurazione
> generica per SM121 e qualsiasi altra architettura non-SM86. SM120 (RTX 5090) e
> SM121 (GB10 DGX Spark) condividono la stessa microarchitettura Blackwell NVB e
> istruzione set identico. La differenza è solo commerciale (consumer vs embedded).

#### 2.3 Installare Dipendenze e Build
```bash
uv pip install -r requirements/lint.txt
pre-commit install

# Build (compila kernel CUDA — ~30-45 min)
uv pip install -e .

uv pip install -r requirements/test.txt
```

#### 2.4 Verifica Patch e Compatibilità SM120
```bash
# Versione vLLM
python -c "import vllm; print(vllm.__version__)"

# Verificare che RTX 5090 sia ora accettata
python -c "
from vllm.v1.attention.ops.turboquant_kv_cache import supports_turboquant_cuda
cap = (12, 0)  # RTX 5090 SM120
print('SM120 supportato:', supports_turboquant_cuda(cap))  # deve stampare: True
"

# Test TurboQuant completo su GPU corrente
pytest tests/quantization/test_turboquant.py -v -s -k "turboquant25 or turboquant35"
# PASS → TurboQuant funziona su RTX 5090 ✅
```

---

### **FASE 3: Preparazione Modello Gemma 4 27B (1-2 ore)**

#### 3.1 Scegliere il Formato dei Pesi

> **Gemma 4 27B ha 27B parametri → 54 GB FP16. Non entra su 32 GB.**
> Serve una versione quantizzata dei *pesi* del modello.

| Formato | VRAM pesi | Qualità | Disponibilità HF Hub |
|---------|-----------|---------|---------------------|
| FP16 | ~54 GB | Originale | ✅ `google/gemma-4-27b-it` |
| **FP8** | **~27 GB** | Ottima | ✅ `google/gemma-4-27b-it` (auto FP8) |
| **AWQ INT4** | **~14 GB** | Buona | ✅ Cerca su HF: `gemma-4-27b-it-awq` |
| GPTQ INT4 | ~14 GB | Buona | ✅ Cerca su HF: `gemma-4-27b-it-gptq` |

**Scelta consigliata: AWQ INT4** — lascia ~15 GB per KV cache, activations e overhead.

#### 3.2 Download Modello Quantizzato
```bash
huggingface-cli login  # necessario per modelli Gemma (accettare licenza)

# Opzione A: AWQ INT4 (consigliato — cerca il repo corretto su HF)
huggingface-cli download <username>/gemma-4-27b-it-awq \
    --local-dir ./models/gemma4-27b-awq

# Opzione B: FP8 nativo (se disponibile su HF Hub)
huggingface-cli download google/gemma-4-27b-it \
    --local-dir ./models/gemma4-27b-fp16
# vLLM può caricare FP16 e on-the-fly quantizzare a FP8:
#   --dtype float8_e4m3fn  nell'invocazione serve

# Opzione C: Quantizzare localmente con llmcompressor (richiede GPU/CPU con 64+ GB RAM)
pip install llmcompressor
python - <<'EOF'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-27b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
recipe = QuantizationModifier(targets="Linear", scheme="W4A16")
oneshot(model=model, dataset=ds, recipe=recipe, num_calibration_samples=512)
model.save_pretrained("./models/gemma4-27b-awq", save_compressed=True)
tokenizer.save_pretrained("./models/gemma4-27b-awq")
EOF
```

#### 3.3 Generare Metadati TurboQuant per la KV Cache
```bash
# TurboQuant ha bisogno di statistiche di calibrazione sulla KV cache.
# --target-model  = il modello quantizzato che serviremo
# --calibration-model = modello base FP16/FP8 usato per calibrare le scale

cd vllm-turboquant

.venv/bin/python benchmarks/generate_turboquant_metadata.py \
    --target-model ./models/gemma4-27b-awq \
    --calibration-model google/gemma-4-27b-it \
    --recipe turboquant35 \
    --output ./models/gemma4-27b-awq/turboquant_kv.json

# Per massima compressione KV (context ancora più lungo, qualità leggermente inferiore):
#   --recipe turboquant25
```

---

### **FASE 4: Avvio Server vLLM-TurboQuant (30 min)**

#### 4.1 Serving con TurboQuant (dopo la Patch SM120 della Fase 2)
```bash
export CUDA_VISIBLE_DEVICES=0

.venv/bin/vllm serve ./models/gemma4-27b-awq \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype turboquant35 \
    --enable-turboquant \
    --turboquant-metadata-path ./models/gemma4-27b-awq/turboquant_kv.json \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

#### 4.2 Serving FP8 KV Cache — Fallback (Opzione B, se TurboQuant non gira su SM120)
```bash
# Usa la KV cache FP8 nativa di vLLM — supportata su qualsiasi GPU Blackwell/Ada
export CUDA_VISIBLE_DEVICES=0

.venv/bin/vllm serve ./models/gemma4-27b-awq \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

> **Differenza**: TurboQuant raggiunge 50-65% di riduzione KV con scale per-layer
> calibrate. FP8 nativo è più semplice ma offre ~50% di riduzione con scale fisse.

#### 4.3 Parametri Chiave Spiegati

| Flag | Valore | Perché |
|------|--------|--------|
| `--kv-cache-dtype turboquant35` | TurboQuant INT8 | Miglior balance qualità/compressione |
| `--enable-turboquant` | — | Abilita il path TurboQuant nel fork |
| `--turboquant-metadata-path` | `.json` calibrato | Scale per-layer della KV cache |
| `--attention-backend TRITON_ATTN` | Triton | Richiesto dal fork per TurboQuant |
| `--max-model-len 32768` | 32K token | Regola base a memoria disponibile |
| `--enable-chunked-prefill` | — | Migliora throughput su context lunghi |
| `--enable-prefix-caching` | — | Cache KV per prompt ripetuti |

---

### **FASE 5: Validazione & Testing (1 ora)**

#### 5.1 Test Unità TurboQuant
```bash
cd vllm-turboquant
pytest tests/quantization/test_turboquant.py -v -s -k "turboquant25 or turboquant35"
# PASS → TurboQuant funziona su SM120 della RTX 5090 ✅
# FAIL → verifica che la patch Fase 2.2 sia stata applicata correttamente
#        oppure usa Opzione B (fp8 KV cache fallback)
```

#### 5.2 Test Caricamento e Inferenza
```python
# test_gemma4.py
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/gemma4-27b-awq",
    kv_cache_dtype="turboquant35",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enable_prefix_caching=True,
)

sampling = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(
    ["Spiega l'apprendimento automatico in termini semplici:"],
    sampling
)
print(outputs[0].outputs[0].text)
```

```bash
python test_gemma4.py
```

#### 5.3 Test API OpenAI-Compatible
```bash
# Con il server già avviato (Fase 4)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gemma4-27b-awq",
      "messages": [{"role": "user", "content": "Cos'è il machine learning?"}],
      "max_tokens": 150
    }'
```

#### 5.4 Benchmark Long-Context (Comparativo TurboQuant vs FP8)
```bash
# Script di comparazione del fork (adatta MODEL e MAX_MODEL_LEN)
MODEL=./models/gemma4-27b-awq \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.85 \
bash benchmarks/run_turboquant_gb10_compare.sh
```

---

### **FASE 6: Benchmark Performance (1-2 ore)**

#### 6.1 Script Benchmark Throughput
```python
# benchmark_throughput.py
import time
import torch
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/gemma4-27b-awq",
    kv_cache_dtype="turboquant35",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
)
sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
prompts = ["Explain neural networks:", "What is blockchain?", "How does CUDA work?"] * 4

start = time.time()
outputs = llm.generate(prompts, sampling)
elapsed = time.time() - start
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

print(f"Throughput: {len(prompts)/elapsed:.2f} prompt/sec")
print(f"Token/sec:  {total_tokens/elapsed:.2f}")
print(f"VRAM used:  {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

#### 6.2 Metriche Attese (stima per RTX 5090 + Gemma 4 27B AWQ + turboquant35)
```
Throughput (prompt/sec):   15-30  (batch brevi)
Token generation (tok/s):  500-800
Latenza P50 (100 tok out): 150-300 ms
VRAM totale utilizzata:    ~22-26 GB (69-81%)
KV cache savings vs FP16:  50-65%
Context max supportato:    32K-64K token (con turboquant35)
```

> Dati indicativi — i valori reali dipendono da batch size, lunghezza contesto
> e dall'effettivo support SM120 dei kernel TurboQuant.

---

### **FASE 7: Deploy Produzione (variabile)**

#### 7.1 Dockerfile (Build da Sorgente)
```dockerfile
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv git curl build-essential && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV VLLM_TARGET_DEVICE=cuda
ENV VLLM_USE_PRECOMPILED=0
ENV VLLM_MAIN_CUDA_VERSION=12.8

# Build vLLM-TurboQuant da sorgente
COPY vllm-turboquant/ /workspace/vllm-turboquant/
RUN cd /workspace/vllm-turboquant && \
    uv venv --python 3.12 .venv && \
    .venv/bin/pip install -e .

# Modello e metadati
COPY models/ /workspace/models/

EXPOSE 8000

CMD ["/workspace/vllm-turboquant/.venv/bin/vllm", "serve",
     "/workspace/models/gemma4-27b-awq",
     "--attention-backend", "TRITON_ATTN",
     "--kv-cache-dtype", "turboquant35",
     "--enable-turboquant",
     "--turboquant-metadata-path", "/workspace/models/gemma4-27b-awq/turboquant_kv.json",
     "--gpu-memory-utilization", "0.85",
     "--max-model-len", "32768",
     "--host", "0.0.0.0",
     "--port", "8000"]
```

#### 7.2 Avviare Container
```bash
docker build -t vllm-gemma4-turboquant .
docker run --gpus device=0 -p 8000:8000 \
    --shm-size=16g \
    vllm-gemma4-turboquant
```

#### 7.3 Monitoraggio
```bash
# Metriche vLLM (Prometheus) esposte su /metrics
curl http://localhost:8000/metrics | grep -E "num_requests|gpu_cache|generation_tokens"
```

---

## 📈 Timeline Implementazione

| Fase | Attività | Tempo | Prerequisiti |
|------|----------|-------|--------------|
| 1 | Setup ambiente Linux | 1-2h | Driver 570+, CUDA 12.8 |
| 2 | Build vLLM-TurboQuant | 1-2h | uv, gcc, CUDA devel |
| 3 | Download/quantizzare modello | 1-3h | 50+ GB storage, HF token |
| 4 | Generare metadati TQ + avvio | 1h | Fase 1-3 ✓ |
| 5 | Validazione e test API | 0.5h | Fase 4 ✓ |
| 6 | Benchmark | 1h | Fase 5 ✓ |
| 7 | Docker/deploy | 1h | Fase 6 ✓ |
| **TOTALE** | | **6-10 ore** | |

---

## ⚠️ Considerazioni Critiche

### 1. **Compatibilità SM120 — Il Rischio Principale**
- Il fork supporta esplicitamente SM86 e SM121
- RTX 5090 = SM120 — simile a SM121 ma non identico
- **Cosa fare**: esegui `pytest tests/quantization/test_turboquant.py` dopo il build.
  - PASS → tutto funziona, procedi con TurboQuant
  - FAIL → usa `--kv-cache-dtype fp8` come fallback (comunque efficace)

### 2. **Build Sorgente Obbligatorio**
- ❌ Non usare `pip install vllm` (wheel PyPI senza TurboQuant)
- ✅ `uv pip install -e .` dalla root del fork
- ⏱️ Primo build: 30-45 minuti (compilazione kernel CUDA)

### 3. **Memory Layout Corretto**
```
Gemma 4 27B su RTX 5090 (32 GB):
├── Pesi modello AWQ INT4:  ~14 GB
├── KV Cache turboquant35:  ~4-6 GB (context 32K)
├── Activations:            ~2 GB
├── vLLM overhead:          ~2 GB
└── Totale:                 ~22-24 GB  ✅ (69-75%)

Con pesi FP8 (~27 GB) rimangono solo ~3-4 GB per KV cache
→ context molto limitato (4K-8K token max)
```

### 4. **Recipe TurboQuant — Significato Corretto**
- I recipe agiscono sulla **KV cache**, non sui pesi del modello
- **turboquant35**: KV cache INT8 con scale per-layer calibrate (miglior qualità)
- **turboquant25**: KV cache INT4 (massima compressione, context più lungo)
- **Consigliato per 5090**: `turboquant35` con pesi AWQ INT4

### 5. **Gemma 4 è Multimodale**
- `google/gemma-4-27b-it` supporta input immagine + testo
- vLLM supporta Gemma 4 multimodale con `--limit-mm-per-prompt image=1`
- Per text-only: funziona senza flag aggiuntivi

---

## 🛠️ Troubleshooting

### TurboQuant test fallisce su SM120
```
ERROR: CUDA kernel not supported for compute capability 12.0
```
**Soluzione A — usa fp8 fallback**:
```bash
# Rimuovi --kv-cache-dtype turboquant35 --enable-turboquant
# Sostituisci con:
--kv-cache-dtype fp8 --calculate-kv-scales
```
**Soluzione B — aggiungi SM120 ai target CUDA** (avanzato):
```bash
# Nel CMakeLists.txt del fork, cerca la lista degli arch CUDA
# e aggiungi "12.0" accanto a "12.1"
git diff HEAD CMakeLists.txt  # per vedere la struttura
```

### Out of Memory durante il serving
```
RuntimeError: CUDA out of memory
```
**Soluzione**:
```bash
# Ridurre context
--max-model-len 16384  # o 8192 se persiste

# Ridurre utilizzo GPU
--gpu-memory-utilization 0.80

# Passare a turboquant25 per KV cache più piccola
--kv-cache-dtype turboquant25
```

### turboquant_kv.json non trovato
```
FileNotFoundError: turboquant_kv.json
```
**Soluzione**:
```bash
cd vllm-turboquant
.venv/bin/python benchmarks/generate_turboquant_metadata.py \
    --target-model ./models/gemma4-27b-awq \
    --calibration-model google/gemma-4-27b-it \
    --recipe turboquant35 \
    --output ./models/gemma4-27b-awq/turboquant_kv.json
```

### TRITON_ATTN non disponibile
```
ValueError: Unknown attention backend: TRITON_ATTN
```
**Soluzione**: assicurati di usare il fork (non vLLM vanilla da PyPI) e verifica:
```bash
python -c "from vllm.attention.backends.triton_attn import TritonAttentionBackend; print('OK')"
```

---

## 📚 Risorse Utili

| Risorsa | Link |
|---------|------|
| vLLM-TurboQuant Fork | https://github.com/mitkox/vllm-turboquant |
| TurboQuant KV Cache Docs | https://github.com/mitkox/vllm-turboquant/blob/main/docs/features/quantization/turboquant_a6000.md |
| Quantized KV Cache Docs | https://github.com/mitkox/vllm-turboquant/blob/main/docs/features/quantization/quantized_kvcache.md |
| Benchmark Script | https://github.com/mitkox/vllm-turboquant/blob/main/benchmarks/run_turboquant_gb10_compare.sh |
| Gemma 4 27B su HF Hub | https://huggingface.co/google/gemma-4-27b-it |
| vLLM Docs | https://docs.vllm.ai |
| llmcompressor | https://github.com/vllm-project/llm-compressor |

---

## ✅ Checklist Pre-Deploy

- [ ] NVIDIA Driver 570+ e CUDA 12.8+ installati
- [ ] RTX 5090 riconosciuta con `nvidia-smi` (SM 12.0)
- [ ] Fork clonato e build da sorgente completato (`uv pip install -e .`)
- [ ] Test TurboQuant eseguiti (`pytest tests/quantization/test_turboquant.py`)
- [ ] Modello Gemma 4 27B AWQ INT4 scaricato (o FP8 se preferito)
- [ ] Metadati TurboQuant (`turboquant_kv.json`) generati
- [ ] Server avviato e risponde su `localhost:8000`
- [ ] Test API `curl /v1/chat/completions` passato
- [ ] Benchmark throughput eseguito
- [ ] VRAM utilization < 85%
- [ ] Latenza P50 < 500ms per 100 token

---

## 🎯 Decision Tree Rapido

```
Hai la RTX 5090?
└─ Sì
   ├─ Build vLLM-TurboQuant da sorgente
   ├─ Esegui pytest TurboQuant
   │   ├─ PASS → usa turboquant35 + TRITON_ATTN  ← percorso ottimale
   │   └─ FAIL → usa --kv-cache-dtype fp8         ← fallback affidabile
   └─ Modello AWQ INT4 (~14 GB) o FP8 (~27 GB)?
       ├─ AWQ INT4 → context 32K-64K token possibile
       └─ FP8     → context 4K-8K token (poco spazio per KV)
```

---

*Basato su: mitkox/vllm-turboquant (commit cee479e, aprile 2026)*

