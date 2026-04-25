# 📑 INDICE COMPLETO - vLLM + TurboQuantum + Gemma-4 27B

## 📋 Documenti di Pianificazione

### `PIANO_VLLM_GEMMA_TURBOQUANT.md` (⭐ Leggi PRIMA)
**Descrizione**: Piano strategico completo di implementazione  
**Contenuto**: 400+ linee con:
- Sommario esecutivo e obiettivi
- Analisi hardware RTX 5090
- Budget memoria dettagliato
- 7 Fasi di implementazione con script
- Timeline (8-14 ore totali)
- Benchmark attesi
- Troubleshooting comune
- Considerazioni critiche

**📖 Azione**: Leggi come guida di riferimento

---

### `SETUP_COMPLETE_GUIDE.md` (⭐ Riferimento Rapido)
**Descrizione**: Guida operativa di setup completo  
**Contenuto**:
- File creati e loro funzione
- Quick start (3 step)
- Metriche attese
- Troubleshooting
- Timeline fase per fase
- Workflow consigliato
- Pre-flight checklist
- Next steps

**📖 Azione**: Usa durante l'implementazione

---

## 🔧 Script di Automazione

### `setup_vllm_turboquant.py` (Python3)
**Descrizione**: Automatizza fasi 1-4 del setup  
**Funzione**: Setup environment + build vLLM da sorgente  
**Tempo**: ~2 ore (build vLLM: 30-45 min)

**Cosa fa**:
```
✅ Verifica CUDA 12.8+
✅ Crea virtual environment
✅ Clona vllm-turboquant repo
✅ Build da sorgente (kernel CUDA custom)
✅ Crea script di test e config
```

**Utilizzo**:
```bash
python setup_vllm_turboquant.py
# oppure continua se errori:
python setup_vllm_turboquant.py --continue
```

**Output**: 
- `vllm_env/` - Virtual environment completo
- `vllm-turboquant/` - Repository clonato e compilato
- `scripts/` - Test e benchmark script
- `serve_vllm.sh` - Script di avvio

---

### `advanced_benchmark.py` (Python3)
**Descrizione**: Suite avanzata di benchmarking  
**Funzione**: Performance profiling con GPU monitoring real-time  
**Metriche**:
- Throughput (prompt/sec, token/sec)
- Latency distribution (P50, P95, P99)
- GPU memory usage
- Power draw
- Temperature

**Utilizzo**:
```bash
# Throughput benchmark
python advanced_benchmark.py \
    --model ./models/gemma4-27b-int8 \
    --num-prompts 32 \
    --max-tokens 256

# Con latency distribution
python advanced_benchmark.py \
    --model ./models/gemma4-27b-int8 \
    --latency-runs 20 \
    --skip-latency false \
    --output benchmark_results.json
```

**Output**: `benchmark_results.json` con metriche complete

---

### `deploy.sh` (Bash)
**Descrizione**: Orchestrazione deployment Docker  
**Funzione**: Automazione build → test → deploy  

**Comandi**:
```bash
./deploy.sh full         # Setup completo (consigliato)
./deploy.sh build        # Build immagine Docker
./deploy.sh start        # Start container
./deploy.sh stop         # Stop container
./deploy.sh restart      # Restart
./deploy.sh logs         # Visualizza log
./deploy.sh status       # Info status
./deploy.sh shell        # Shell nel container
./deploy.sh benchmark    # Esegui benchmark
./deploy.sh help         # Help
```

**Cosa fa `./deploy.sh full`**:
```
✅ Verifica Docker e NVIDIA
✅ Crea directory (models, cache, logs)
✅ Verifica modello
✅ Build immagine Docker (~20-30 min)
✅ Start container
✅ Aspetta health check
✅ Test API
✅ Mostra status
```

---

## 📦 Container & Deployment

### `Dockerfile`
**Descrizione**: Image Docker ottimizzato per vLLM  
**Base**: `nvidia/cuda:12.8.0-devel-ubuntu22.04`  
**Include**:
- Python 3.12
- CUDA 12.8 + dev tools
- vLLM build da sorgente
- Kernel custom compilati
- Healthcheck integrato

**Build**:
```bash
docker build -t vllm-gemma4-turboquant:latest -f Dockerfile .
```

**Vantaggi**:
- Build riproducibile
- Kernel CUDA ottimizzati
- Dipendenze isolate
- Facile distribuzione

---

### `docker-compose.yml`
**Descrizione**: Orchestrazione multi-container  
**Servizi**:
- `vllm-gemma4` - API server principale
- `prometheus` - Monitoring (profilo: monitoring)
- `grafana` - Dashboard (profilo: monitoring)

**Volume mapping**:
```
./models/          → /models (RO)
./cache/           → /cache
./logs/            → /logs
prometheus.yml     → /etc/prometheus/
```

**Utilizzo**:
```bash
# API server solo
docker compose up -d vllm-gemma4

# Con monitoring (Prometheus + Grafana)
docker compose --profile monitoring up -d

# Stop
docker compose down

# Log
docker compose logs -f vllm-gemma4
```

**Vantaggi**:
- GPU runtime support
- Health checks
- Auto-restart
- Volume management
- Easy scaling

---

## 📊 Monitoraggio

### `prometheus.yml`
**Descrizione**: Configurazione Prometheus  
**Job**:
- vllm-api: `/metrics` endpoint
- prometheus: self-monitoring

**Metrica tracciata**:
- Throughput (prompt/sec)
- Latency (ms)
- GPU memory %
- GPU temperature
- GPU utilization
- Request queue

**Utilizzo**:
```bash
# Avvia Prometheus (via docker-compose)
docker compose --profile monitoring up prometheus

# Accedi a http://localhost:9090
```

---

### `alert_rules.yml`
**Descrizione**: Regole di alert Prometheus  
**Alert configurati**:
- High GPU Memory (>90%)
- High Temperature (>85°C)
- Low Throughput (<5 prompt/sec)
- High Latency (P95 >1s)
- Service Down
- TurboQuantum Issues

**Utilizzo**: Automaticamente caricato da prometheus.yml

---

## 🎯 Configuration & Utils

### `.env`
**Descrizione**: Variabili ambiente CUDA  
**Creato da**: `setup_vllm_turboquant.py`  
**Variabili**:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export CUDA_VISIBLE_DEVICES=0
```

---

## 📂 Directory Structure Finale

```
articologermania/
├── 📄 PIANO_VLLM_GEMMA_TURBOQUANT.md       ⭐ Piano completo
├── 📄 SETUP_COMPLETE_GUIDE.md              ⭐ Guida operativa
├── 📄 INDEX.md                             (questo file)
│
├── 🐍 setup_vllm_turboquant.py             # Setup automation
├── 🐍 advanced_benchmark.py                # Benchmarking
│
├── 🐳 Dockerfile                           # Docker image
├── 🐳 docker-compose.yml                   # Container orchestration
├── 📋 deploy.sh                            # Deployment script
│
├── 📊 prometheus.yml                       # Prometheus config
├── 📊 alert_rules.yml                      # Alert rules
│
├── 📁 vllm_env/                            (creato da setup script)
│   └── bin/activate
├── 📁 vllm-turboquant/                     (clonato da GitHub)
│   ├── vllm/
│   ├── benchmarks/
│   └── requirements/
├── 📁 models/
│   └── gemma4-27b-int8/                    (scarica da HF Hub)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.model
│       └── turboquant_kv.json
├── 📁 cache/                               (HuggingFace cache)
├── 📁 logs/                                (server logs)
└── 📁 scripts/                             (creato da setup)
    ├── test_inference.py
    ├── benchmark.py
    ├── test_api.sh
    └── quantize_gemma.py
```

---

## ⚡ Quick Decision Tree

```
┌─────────────────────────────────────────────────────┐
│ Vuoi deployare vLLM + Gemma-4 + TurboQuantum?       │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
    Nativo             Docker
    (Linux)            (Qualunque)
        │                 │
        │                 └──→ ./deploy.sh full
        │
        └──→ 1. python setup_vllm_turboquant.py
            2. Download model
            3. ./serve_vllm.sh
            4. python scripts/test_inference.py
```

---

## 🚀 Step-by-Step

### Opzione A: Setup Automatico (Consigliato per Docker)
```bash
# Step 1: Clone/Update files
git pull origin main

# Step 2: Deploy completo
./deploy.sh full

# Step 3: Test
docker exec vllm-gemma4-server curl http://localhost:8000/health
```

### Opzione B: Setup Manuale (Consigliato per Linux Nativo)
```bash
# Step 1: Setup
python setup_vllm_turboquant.py

# Step 2: Attiva env
source vllm_env/bin/activate

# Step 3: Download modello
huggingface-cli download google/gemma-4-27b-it \
    --local-dir ./models/gemma4-27b-int8

# Step 4: Start server
./serve_vllm.sh

# Step 5: Test (nuovo terminal)
python scripts/test_inference.py

# Step 6: Benchmark
python advanced_benchmark.py --model ./models/gemma4-27b-int8
```

---

## 📊 File Matrix

| File | Tipo | Dipendenze | Tempo | Output |
|------|------|-----------|-------|--------|
| `setup_vllm_turboquant.py` | Python | CUDA 12.8+ | 2h | venv + vLLM |
| `advanced_benchmark.py` | Python | vLLM | 30min | metrics.json |
| `deploy.sh` | Bash | Docker | 30min | Container |
| `Dockerfile` | Docker | CUDA image | 20-30min | Image |
| `docker-compose.yml` | YAML | Docker | - | Orchestration |
| `prometheus.yml` | YAML | - | - | Config |

---

## ⏱️ Timeline Totale

```
Fase 1: Prerequisiti              5 min    ✓ Verifica
Fase 2: Setup Python              10 min   → setup_vllm_turboquant.py
Fase 3: Build vLLM                45 min   ↓ (venv creato)
Fase 4: Download Modello          30 min   ↓ (Intel/CPU limited)
Fase 5: Server Start              5 min    ↓ (./serve_vllm.sh o docker)
Fase 6: Testing                   15 min   → scripts/test_*.py
Fase 7: Benchmarking              30 min   → advanced_benchmark.py
Fase 8: Deploy Produzione         10 min   ✓ (./deploy.sh full)
────────────────────────────────────
TOTALE                            150 min  (2.5 ore)
```

---

## 🔑 Key Files Summary

| Priority | File | Azione |
|----------|------|--------|
| 🔴 **MUST** | `PIANO_VLLM_GEMMA_TURBOQUANT.md` | **Leggi prima** |
| 🔴 **MUST** | `setup_vllm_turboquant.py` | **Esegui prima** |
| 🟠 SHOULD | `deploy.sh` | Usa per Docker |
| 🟠 SHOULD | `advanced_benchmark.py` | Benchmark |
| 🟡 NICE | `Dockerfile` | Se vuoi custom image |
| 🟡 NICE | `prometheus.yml` | Se vuoi monitoring |

---

## ✅ Validation Checklist

Prima di andare live, verifica:

- [ ] Leggi `PIANO_VLLM_GEMMA_TURBOQUANT.md`
- [ ] Esegui `python setup_vllm_turboquant.py`
- [ ] Scarica modello Gemma-4 INT8
- [ ] Test inference locale
- [ ] Benchmark performance
- [ ] Memory < 90%
- [ ] Latency accettabile
- [ ] Docker image build (se Docker)
- [ ] Health check pass
- [ ] API test curl

---

## 🎯 Success Criteria

✅ Setup completato quando:
- vLLM server avviato su port 8000
- Modello Gemma-4 27B caricato
- TurboQuantum abilitato per KV cache
- Throughput > 10 prompt/sec
- Latency P50 < 500ms
- GPU memory usage < 90%
- Health check responds

---

**Versione**: 1.0  
**Data**: Aprile 2026  
**vLLM**: 0.19.0+  
**GPU**: RTX 5090 (Blackwell SM98)  
**Modello**: Gemma-4 27B  
**Quantizzazione**: TurboQuantum INT8

---

## 📞 Quick Help

```bash
# Mostra questo file
cat INDEX.md

# Leggi il piano
cat PIANO_VLLM_GEMMA_TURBOQUANT.md

# Setup rapido
python setup_vllm_turboquant.py

# Deploy Docker
./deploy.sh full

# View logs
docker logs vllm-gemma4-server

# Benchmark
python advanced_benchmark.py --help

# Health check
curl http://localhost:8000/health
```

**🚀 Ready to deploy? Start with `./deploy.sh full` or `python setup_vllm_turboquant.py`**
