# 🚀 vLLM + TurboQuantum Setup

**Complete setup package for RTX 5090 with Gemma-4 27B**

```
vllm-turboquant-setup/
├── 📄 README.md                    ← You are here
├── 📁 docs/                        Documentation
│   ├── INDEX.md                    File navigation
│   ├── PIANO_VLLM_GEMMA_TURBOQUANT.md   ⭐ Complete plan
│   ├── SETUP_COMPLETE_GUIDE.md     Quick reference
│   └── README_SETUP.md             Overview
├── 📁 scripts/                     Automation scripts
│   ├── setup_vllm_turboquant.py    Setup + build
│   ├── advanced_benchmark.py       Benchmarking
│   ├── deploy.sh                   Docker deployment
│   └── show_setup_summary.sh       Setup summary
├── 📁 docker/                      Container files
│   ├── Dockerfile                  Image definition
│   └── docker-compose.yml          Orchestration
└── 📁 config/                      Configuration
    ├── prometheus.yml              Monitoring
    └── alert_rules.yml             Alerts
```

---

## 🎯 Quick Start

### Option 1: Docker (Recommended)
```bash
cd scripts
chmod +x deploy.sh
./deploy.sh full
```

### Option 2: Native Linux
```bash
cd scripts
python setup_vllm_turboquant.py
source vllm_env/bin/activate
./serve_vllm.sh
```

---

## 📖 Documentation

Start here:
1. **`docs/INDEX.md`** - Navigation hub
2. **`docs/PIANO_VLLM_GEMMA_TURBOQUANT.md`** - Complete technical plan
3. **`docs/SETUP_COMPLETE_GUIDE.md`** - Quick reference

---

## 🚀 Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `setup_vllm_turboquant.py` | Setup environment + build vLLM | 2h |
| `advanced_benchmark.py` | Performance benchmarking | 30min |
| `deploy.sh` | Docker deployment | 20-30min |
| `show_setup_summary.sh` | Display summary | Instant |

---

## 🐳 Docker

Files in `docker/`:
- **Dockerfile** - Optimized CUDA 12.8 image
- **docker-compose.yml** - Full orchestration

Build and deploy:
```bash
cd ../scripts
./deploy.sh full
```

---

## 📊 Configuration

Files in `config/`:
- **prometheus.yml** - Monitoring setup
- **alert_rules.yml** - Alert definitions

Enable monitoring:
```bash
docker compose --profile monitoring up
```

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Throughput | 15-25 prompt/sec |
| Latency P50 | 200-400ms |
| Memory | 18-22 GB |
| Compression | 50-65% KV cache |

---

## ✅ Prerequisites

- CUDA 12.8+
- NVIDIA driver 560+
- RTX 5090
- 80+ GB storage
- Docker (for containerized deployment)

---

## 🔧 File Organization

All files organized by function:
- **docs/** - Read documentation
- **scripts/** - Run automation
- **docker/** - Build containers
- **config/** - Monitoring setup

Each folder is self-contained and clearly labeled.

---

## 📞 Help

Start with: `cat docs/INDEX.md`

For detailed technical info: `cat docs/PIANO_VLLM_GEMMA_TURBOQUANT.md`

---

**Ready to deploy? Start with `cd scripts && ./deploy.sh full` or read the docs!**
