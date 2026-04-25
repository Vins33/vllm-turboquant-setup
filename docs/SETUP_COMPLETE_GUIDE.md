# 📦 Guida Completa: vLLM + TurboQuantum + Gemma-4 27B su RTX 5090

## 📁 File Creati

### 1. **Piano Dettagliato** 
- **File**: `PIANO_VLLM_GEMMA_TURBOQUANT.md`
- **Contenuto**: Documento completo di 400+ linee con:
  - Timeline fase per fase
  - Specifiche hardware
  - Budget memoria
  - Script di installazione dettagliati
  - Benchmark attesi
  - Troubleshooting
  - Checklist pre-deploy
- **Come usare**: Leggi come guida di riferimento durante l'implementazione

---

### 2. **Script di Setup Automatizzato**
- **File**: `setup_vllm_turboquant.py`
- **Funzione**: Automatizza fasi 1-4 del piano (setup ambiente + build vLLM)
- **Utilizzo**:
```bash
# Setup completo
python setup_vllm_turboquant.py

# Continua anche se ci sono errori
python setup_vllm_turboquant.py --continue
```
- **Cosa fa**:
  - ✅ Verifica CUDA 12.8+
  - ✅ Crea virtual environment con uv
  - ✅ Clona vllm-turboquant repo
  - ✅ Build vLLM da sorgente (30-45 min)
  - ✅ Prepara script di test
  - ✅ Crea config files

---

### 3. **Benchmark Avanzato**
- **File**: `advanced_benchmark.py`
- **Funzione**: Suite completa di benchmarking con monitoraggio GPU real-time
- **Utilizzo**:
```bash
# Throughput benchmark (32 prompts, 256 tokens)
python advanced_benchmark.py \
    --model ./models/gemma4-27b-int8 \
    --num-prompts 32 \
    --max-tokens 256

# Latency distribution (20 runs)
python advanced_benchmark.py \
    --model ./models/gemma4-27b-int8 \
    --latency-runs 20 \
    --output benchmark_results.json
```
- **Output**:
  - Throughput (prompt/sec, token/sec)
  - Latency (P50, P95, P99)
  - GPU Memory usage
  - Power draw e temperatura
  - Results salvati in JSON

---

### 4. **Docker Containerization**
- **File**: `Dockerfile`
- **Funzione**: Container ottimizzato con CUDA 12.8 e vLLM
- **Build**:
```bash
docker build -t vllm-gemma4-turboquant:latest -f Dockerfile .
```
- **Vantaggi**:
  - Build riproducibile
  - Kernel CUDA custom compilati
  - Dependency management isolato
  - Facile deployment

---

### 5. **Docker Compose**
- **File**: `docker-compose.yml`
- **Funzione**: Orchestrazione container con opzioni avanzate
- **Utilizzo**:
```bash
# Avvia server
docker compose up -d vllm-gemma4

# Con monitoraggio (Prometheus + Grafana)
docker compose --profile monitoring up -d

# Stop
docker compose down
```
- **Include**:
  - vLLM API server
  - Volume mapping (models, cache, logs)
  - Health checks
  - Prometheus monitoring (opzionale)
  - Grafana dashboards (opzionale)
  - Auto-restart su errori

---

### 6. **Deploy Script**
- **File**: `deploy.sh`
- **Funzione**: Automazione completa build → test → deploy
- **Utilizzo**:
```bash
# Setup completo (recommended)
./deploy.sh full

# Build solo immagine
./deploy.sh build

# Avvia container
./deploy.sh start

# Stop container
./deploy.sh stop

# Visualizza log
./deploy.sh logs

# Esegui benchmark
./deploy.sh benchmark

# Shell nel container
./deploy.sh shell

# Status e info
./deploy.sh status
```
- **Checklist automatico**:
  - ✅ Verifica Docker
  - ✅ Verifica NVIDIA driver
  - ✅ Controlla GPU runtime
  - ✅ Crea directory
  - ✅ Verifica modello
  - ✅ Build immagine (20-30 min)
  - ✅ Start container
  - ✅ Wait health check
  - ✅ Test API
  - ✅ Mostra status

---

### 7. **Configurazione Monitoraggio**
- **File**: `prometheus.yml` + `alert_rules.yml`
- **Funzione**: Monitoraggio metrica in tempo reale
- **Metriche tracciate**:
  - GPU Memory % / GB
  - GPU Temperature
  - GPU Power Draw
  - Throughput (prompt/sec)
  - Latency (P50, P95, P99)
  - Request queue length
  - KV cache compression status
- **Alert configurati**:
  - High GPU memory (>90%)
  - High temperature (>85°C)
  - Low throughput (<5 prompt/sec)
  - High latency (P95 >1s)
  - Service down
  - TurboQuantum issues

---

## 🚀 Quick Start (3 step)

### Step 1: Setup Ambiente (2-3 ore)
```bash
# Esegui script automatizzato
python setup_vllm_turboquant.py

# Attiva environment
source vllm_env/bin/activate
```

### Step 2: Scarica Modello (30 min download)
```bash
# Gemma-4 27B quantizzato
huggingface-cli download google/gemma-4-27b-it \
    --local-dir ./models/gemma4-27b-int8

# Oppure build quantizzato custom:
# python scripts/quantize_gemma.py
```

### Step 3: Deploy Server
```bash
# Opzione A: Native vLLM
./serve_vllm.sh

# Oppure Opzione B: Docker
./deploy.sh full
```

---

## 📊 Metriche Attese

**Hardware**: NVIDIA RTX 5090 (Blackwell SM98, 32GB GDDR7)  
**Modello**: Gemma-4 27B INT8 quantizzato  
**Quantizzazione**: TurboQuantum INT8 (50-65% KV cache compression)

| Metrica | Valore | Note |
|---------|--------|------|
| **Throughput** | 15-25 prompt/sec | Batch size 12 |
| **Token Generation** | 400-600 token/sec | 256 token output |
| **Latency P50** | 200-400ms | Per 100 token |
| **Latency P95** | 400-600ms | Peak performance |
| **Memory Utilizzo** | 18-22 GB | 56-69% della GPU |
| **Power Draw** | 350-450W | Picco under load |
| **Temperature** | 70-80°C | Buon cooling |

---

## 🔧 Troubleshooting

### ❌ CUDA Compute Capability
```bash
# Errore: "Kernel compilation failed for SM98"
export CUDA_ARCH=90
export VLLM_MAIN_CUDA_VERSION=12.8
cd vllm-turboquant && uv pip install -e .
```

### ❌ Out of Memory
```bash
# Riduci max_model_len in serve_vllm.sh:
--max-model-len 2048  # da 4096
```

### ❌ Modello Non Trovato
```bash
# Rigenerato metadati TurboQuantum:
python benchmarks/generate_turboquant_metadata.py \
    --target-model ./models/gemma4-27b-int8 \
    --calibration-model google/gemma-4-27b \
    --recipe turboquant35 \
    --output ./models/gemma4-27b-int8/turboquant_kv.json
```

### ❌ Docker GPU Non Disponibile
```bash
# Installa nvidia-container-runtime:
# https://github.com/NVIDIA/nvidia-container-runtime

# Verifica:
docker run --rm --runtime=nvidia nvidia/cuda:12.8.0-runtime nvidia-smi
```

---

## 📈 Timeline Implementazione

| Fase | Attività | Tempo | Files |
|------|----------|-------|-------|
| 0 | Setup prerequisiti | 30 min | setup_vllm_turboquant.py |
| 1 | Build vLLM | 45 min | Dockerfile, setup script |
| 2 | Download modello | 30 min | - |
| 3 | Config server | 15 min | prometheus.yml, docker-compose.yml |
| 4 | Deploy container | 5 min | deploy.sh |
| 5 | Testing | 30 min | advanced_benchmark.py |
| **TOTALE** | | **3-4 ore** | |

---

## 🎯 Workflow Consigliato

```
1. Leggi PIANO_VLLM_GEMMA_TURBOQUANT.md
   ↓
2. Esegui setup_vllm_turboquant.py
   ↓
3. Scarica modello Gemma-4 27B INT8
   ↓
4. Test native: python scripts/test_inference.py
   ↓
5. Deploy Docker: ./deploy.sh full
   ↓
6. Benchmark: python advanced_benchmark.py
   ↓
7. Monitoring: docker compose up prometheus grafana
   ↓
8. Deploy Produzione: ./deploy.sh start
```

---

## 📚 Risorse Utili

| Risorsa | Link |
|---------|------|
| vLLM Docs | https://docs.vllm.ai |
| vLLM-TurboQuantum | https://github.com/mitkox/vllm-turboquant |
| Gemma-4 Model Card | https://huggingface.co/google/gemma-4-27b |
| NVIDIA CUDA | https://docs.nvidia.com/cuda/ |
| Docker GPU | https://nvidia.github.io/nvidia-container-runtime/ |
| Prometheus Docs | https://prometheus.io/docs/ |

---

## ✅ Pre-Flight Checklist

Prima di andare in produzione:

- [ ] CUDA 12.8+ e driver 560+
- [ ] RTX 5090 riconosciuta (`nvidia-smi`)
- [ ] vLLM build completato senza errori
- [ ] Modello Gemma-4 INT8 scaricato/quantizzato
- [ ] Metadati TurboQuantum generati
- [ ] Test inference locale passato
- [ ] API curl test passato
- [ ] Benchmark throughput eseguito
- [ ] Memory utilizzo < 90%
- [ ] Latency P50 < 500ms
- [ ] Temperature < 80°C
- [ ] Docker image build completato
- [ ] Container health check passed
- [ ] Monitoraggio attivo

---

## 🎉 Prossimi Step

Dopo il deploy iniziale:

1. **Ottimizzazione Performance**
   - Tune `gpu_memory_utilization` (0.8-0.95)
   - Experimenti con `max_model_len`
   - Test diverse `SamplingParams`

2. **Scaling Horizontale**
   - Multiple RTX 5090 con Tensor Parallel
   - Load balancing (nginx, HAProxy)
   - Multi-instance serving

3. **Fine-Tuning Specializzato**
   - Quantizzazione INT4 per modelli più piccoli
   - LoRA adapters per task specifici
   - KV cache prefixing

4. **Production Hardening**
   - Rate limiting e authentication
   - Request queueing e prioritization
   - Circuit breaker patterns
   - Graceful shutdown

---

## 📞 Support

Se incontri problemi:

1. **Logs**: `docker logs vllm-gemma4-server`
2. **Metrics**: Accedi a Prometheus http://localhost:9090
3. **Dashboard**: Grafana http://localhost:3000
4. **Issues**: GitHub https://github.com/mitkox/vllm-turboquant/issues

---

**Ultimo aggiornamento**: Aprile 2026  
**vLLM Version**: 0.19.0+  
**TurboQuantum**: Supportato per Blackwell (SM98)
