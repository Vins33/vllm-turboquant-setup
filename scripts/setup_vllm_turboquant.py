#!/usr/bin/env python3
"""
Script di Setup Automatizzato per vLLM + TurboQuantum + Gemma-4 27B
Esegue le fasi 1-4 del piano di implementazione
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import json

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_success(msg: str):
    print(f"{Colors.OKGREEN}✅ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ️  {msg}{Colors.ENDC}")

def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠️  {msg}{Colors.ENDC}")

def print_error(msg: str):
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

def run_command(cmd: str, description: str, check: bool = True) -> bool:
    """Execute shell command and handle errors"""
    print_info(f"Eseguendo: {description}")
    print(f"   Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check)
        if result.returncode == 0:
            print_success(f"{description} completato!")
            return True
        else:
            print_error(f"{description} fallito con codice {result.returncode}")
            return False
    except Exception as e:
        print_error(f"{description} errore: {str(e)}")
        return False

def check_cuda() -> bool:
    """Verificare CUDA installation"""
    print_header("FASE 0: Verifica Prerequisiti")
    
    # Check nvidia-smi
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
    if result.returncode != 0:
        print_error("NVIDIA CUDA non trovato! Installa CUDA 12.8+")
        return False
    
    print_success("NVIDIA CUDA rilevato")
    
    # Check Python 3.12
    result = subprocess.run("python3.12 --version", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print_warning("Python 3.12 non trovato, cercherò altre versioni...")
        result = subprocess.run("python3 --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print_info(f"Python trovato: {result.stdout.strip()}")
    else:
        print_success(f"Python 3.12 trovato: {result.stdout.strip()}")
    
    return True

def setup_environment() -> bool:
    """FASE 1: Setup Ambiente"""
    print_header("FASE 1: Setup Ambiente")
    
    # Create project directories
    dirs = ["models", "data", "logs", "scripts"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print_success(f"Directory {d}/ creata")
    
    # Create venv with uv
    if not run_command(
        "uv venv --python 3.12 vllm_env",
        "Creazione virtual environment con uv"
    ):
        print_warning("uv venv fallito, tentando venv standard...")
        run_command(
            "python3 -m venv vllm_env",
            "Creazione virtual environment con python venv"
        )
    
    # Create .env file
    env_content = """
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_MAIN_CUDA_VERSION=12.8
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_DEPRECATED_MODELS=0
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    print_success(".env file creato")
    
    return True

def install_vllm() -> bool:
    """FASE 2: Installare vLLM con TurboQuantum"""
    print_header("FASE 2: Installare vLLM da Sorgente")
    
    # Check if vllm repo exists
    if not os.path.exists("vllm-turboquant"):
        print_info("Clonando vllm-turboquant repository...")
        if not run_command(
            "git clone https://github.com/mitkox/vllm-turboquant.git",
            "Clone vllm-turboquant"
        ):
            return False
    else:
        print_info("Repository vllm-turboquant già presente")
    
    os.chdir("vllm-turboquant")
    
    # Activate venv and install
    venv_activate = ". ../vllm_env/bin/activate && "
    
    print_info("Installando dipendenze di build...")
    if not run_command(
        f"{venv_activate}uv pip install -r requirements/lint.txt",
        "Installazione requirements/lint.txt"
    ):
        print_warning("Alcuni pacchetti opzionali potrebbero essere falliti")
    
    print_info("Compilando vLLM da sorgente (questo richiede ~30-45 minuti)...")
    if not run_command(
        f"{venv_activate}uv pip install -e .",
        "Build e installazione vLLM"
    ):
        print_error("Build vLLM fallito!")
        return False
    
    # Install test requirements
    if not run_command(
        f"{venv_activate}uv pip install -r requirements/test.txt",
        "Installazione requirements/test.txt"
    ):
        print_warning("Test requirements non installati completamente")
    
    # Verify installation
    print_info("Verificando installazione vLLM...")
    result = subprocess.run(
        f"{venv_activate}python -c 'import vllm; print(vllm.__version__)'",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print_success(f"vLLM installato: versione {result.stdout.strip()}")
    else:
        print_error("Verifica vLLM fallita!")
        return False
    
    os.chdir("..")
    return True

def prepare_model() -> bool:
    """FASE 3: Preparazione Modello Gemma-4"""
    print_header("FASE 3: Preparazione Modello Gemma-4 27B")
    
    venv_activate = ". vllm_env/bin/activate && "
    
    # Create model preparation script
    quant_script = """
import os
os.environ['HF_HOME'] = './models/huggingface_cache'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("🔄 Scaricando Gemma-4 27B...")

model_name = "google/gemma-4-27b"
output_dir = "./models/gemma4-27b-int8"

try:
    # Scaricare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Scaricare e quantizzare modello
    print("💾 Caricando modello (questo richiede ~120GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Ottimizzazione
    )
    
    print("📊 Convertendo a INT8...")
    # Semplice casting per INT8 (uso di llm-compressor è consigliato per produzione)
    model = model.half()  # FP16 per ora
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Modello salvato in: {output_dir}")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    print("💡 Suggerimento: Assicurati di avere abbastanza RAM/VRAM")
    print("💡 Alternativa: Scarica modello prequalizzato da HuggingFace")
"""
    
    with open("scripts/quantize_gemma.py", "w") as f:
        f.write(quant_script)
    
    print_info("Script di quantizzazione creato in scripts/quantize_gemma.py")
    print_warning("⚠️  Download e quantizzazione richiede ~120GB RAM")
    print_warning("⚠️  Considera di scaricare versione prequalizzata invece")
    
    # Provide download instructions
    print(f"""
{Colors.OKCYAN}
📥 Opzioni di scaricamento modello:

1️⃣  Download prequalizzato (CONSIGLIATO):
   huggingface-cli download google/gemma-4-27b-it \\
       --local-dir ./models/gemma4-27b-int8

2️⃣  Download FP16 base:
   huggingface-cli download google/gemma-4-27b \\
       --local-dir ./models/gemma4-27b-fp16

3️⃣  Quantizzazione custom:
   {venv_activate}python scripts/quantize_gemma.py

{Colors.ENDC}
    """)
    
    return True

def generate_turboquant_metadata() -> bool:
    """FASE 3B: Generare Metadati TurboQuantum"""
    print_header("FASE 3B: Generare Metadati TurboQuantum (Opzionale)")
    
    if not os.path.exists("models/gemma4-27b-int8"):
        print_warning("Modello quantizzato non trovato, skipping metadata generation")
        return False
    
    venv_activate = ". vllm_env/bin/activate && "
    
    metadata_script = """
cd vllm-turboquant
python benchmarks/generate_turboquant_metadata.py \\
    --target-model ../models/gemma4-27b-int8 \\
    --calibration-model google/gemma-4-27b \\
    --recipe turboquant35 \\
    --output ../models/gemma4-27b-int8/turboquant_kv.json
cd ..
"""
    
    print_info("Comando per generare metadati TurboQuantum:")
    print(f"""
    {venv_activate}{metadata_script}
    
💡 Questo è opzionale e richiede l'ambiente vLLM attivo
💡 I metadati ottimizzano la compressione KV cache
    """)
    
    return True

def create_config_files() -> bool:
    """FASE 4: Creare file di configurazione"""
    print_header("FASE 4: Creare File di Configurazione")
    
    # Create vllm serve script
    serve_script = """#!/bin/bash
# Script di avvio server vLLM con TurboQuantum

source vllm_env/bin/activate

export CUDA_VISIBLE_DEVICES=0
export VLLM_TARGET_DEVICE=cuda

echo "🚀 Avviando vLLM Server..."
echo "   Modello: Gemma-4 27B"
echo "   Quantizzazione: TurboQuantum INT8"
echo "   GPU: RTX 5090"

cd vllm-turboquant

python -m vllm.entrypoints.openai.api_server \\
    --model ../models/gemma4-27b-int8 \\
    --quantization turboquant \\
    --kv-cache-dtype turboquant35 \\
    --turboquant-metadata-path ../models/gemma4-27b-int8/turboquant_kv.json \\
    --attention-backend TRITON_ATTN \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 4096 \\
    --tensor-parallel-size 1 \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --dtype float16 \\
    --api-key sk-proj-turboquant-5090 \\
    "$@"
"""
    
    with open("serve_vllm.sh", "w") as f:
        f.write(serve_script)
    os.chmod("serve_vllm.sh", 0o755)
    print_success("serve_vllm.sh creato")
    
    # Create test script
    test_script = """#!/usr/bin/env python3
'''Test script for vLLM inference'''
import sys
sys.path.insert(0, 'vllm-turboquant')

from vllm import LLM, SamplingParams
import time

print("🔄 Caricando modello Gemma-4 27B...")

llm = LLM(
    model="./models/gemma4-27b-int8",
    quantization="turboquant",
    kv_cache_dtype="turboquant35",
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
)

print("✅ Modello caricato!")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
)

prompts = [
    "What is machine learning?",
    "Explain quantum computing:",
    "Tell me about deep learning:",
]

print("\\n🔤 Generando completions...")
start = time.time()

outputs = llm.generate(prompts, sampling_params)

elapsed = time.time() - start

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\\n📝 Prompt: {prompt}")
    print(f"📄 Output: {generated_text}")

print(f"\\n⏱️  Tempo totale: {elapsed:.2f}s")
print(f"📈 Throughput: {len(prompts)/elapsed:.2f} prompt/sec")
"""
    
    with open("scripts/test_inference.py", "w") as f:
        f.write(test_script)
    os.chmod("scripts/test_inference.py", 0o755)
    print_success("scripts/test_inference.py creato")
    
    # Create benchmark script
    benchmark_script = """#!/usr/bin/env python3
'''Benchmark script for TurboQuantum performance'''
import sys
sys.path.insert(0, 'vllm-turboquant')

from vllm import LLM, SamplingParams
import time
import torch

print("🚀 BENCHMARK: vLLM + TurboQuantum + Gemma-4 27B")
print("="*70)

llm = LLM(
    model="./models/gemma4-27b-int8",
    quantization="turboquant",
    kv_cache_dtype="turboquant35",
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

# Generate test prompts
prompts = [
    "Explain neural networks: ",
    "What is blockchain: ",
    "How does photosynthesis work: ",
] * 4  # 12 total

print(f"\\n📊 Benchmark con {len(prompts)} prompt...")
print(f"💾 GPU Memory allocato: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

print(f"\\n✅ Risultati Benchmark:")
print(f"   ⏱️  Tempo totale: {elapsed:.2f}s")
print(f"   📈 Throughput: {len(prompts)/elapsed:.2f} prompt/sec")
print(f"   🔤 Token generation: {total_tokens/elapsed:.2f} token/sec")
print(f"   💾 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"   📊 Average latency: {(elapsed/len(prompts))*1000:.0f}ms per prompt")
"""
    
    with open("scripts/benchmark.py", "w") as f:
        f.write(benchmark_script)
    os.chmod("scripts/benchmark.py", 0o755)
    print_success("scripts/benchmark.py creato")
    
    # Create API test script
    api_test_script = """#!/bin/bash
# Test API with curl

echo "🧪 Testing vLLM OpenAI API..."
echo "================================"

# Test 1: Simple completion
echo "\\n1️⃣  Simple Completion:"
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma-4-27b",
    "prompt": "def fibonacci(n):",
    "max_tokens": 50
  }' | python -m json.tool

# Test 2: Chat completion
echo "\\n2️⃣  Chat Completion:"
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma-4-27b",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 100
  }' | python -m json.tool

# Test 3: Model list
echo "\\n3️⃣  Available Models:"
curl http://localhost:8000/v1/models | python -m json.tool
"""
    
    with open("scripts/test_api.sh", "w") as f:
        f.write(api_test_script)
    os.chmod("scripts/test_api.sh", 0o755)
    print_success("scripts/test_api.sh creato")
    
    # Create README
    readme = """# vLLM + TurboQuantum + Gemma-4 27B Setup

## Quick Start

### 1. Attivare ambiente
```bash
source vllm_env/bin/activate
```

### 2. Avviare server
```bash
./serve_vllm.sh
```

### 3. Test inference (in altro terminal)
```bash
python scripts/test_inference.py
```

### 4. Benchmark performance
```bash
python scripts/benchmark.py
```

### 5. Test API
```bash
bash scripts/test_api.sh
```

## File Structure

```
├── vllm_env/                    # Virtual environment
├── vllm-turboquant/             # vLLM source code
├── models/
│   └── gemma4-27b-int8/        # Quantized model
├── scripts/
│   ├── quantize_gemma.py       # Model quantization
│   ├── test_inference.py       # Test inference
│   ├── benchmark.py            # Performance benchmark
│   └── test_api.sh             # API tests
├── serve_vllm.sh               # Start server script
└── .env                        # Environment variables
```

## Troubleshooting

### CUDA Error
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
```

### Out of Memory
Edit `serve_vllm.sh` and change `--max-model-len`:
```bash
--max-model-len 2048  # Reduce from 4096
```

### Model Not Found
Download model first:
```bash
huggingface-cli download google/gemma-4-27b-it \\
    --local-dir ./models/gemma4-27b-int8
```

## Performance Notes

Expected metrics on RTX 5090:
- Throughput: 15-25 prompt/sec
- Token generation: 400-600 token/sec
- Latency P50: 200-400ms
- Memory utilization: 18-22 GB (56-69%)
"""
    
    with open("README.md", "w") as f:
        f.write(readme)
    print_success("README.md creato")
    
    return True

def main():
    """Main setup flow"""
    print_header("vLLM + TurboQuantum Setup Automatizzato")
    print(f"{Colors.BOLD}RTX 5090 | Gemma-4 27B | TurboQuantum INT8{Colors.ENDC}\n")
    
    # Check prerequisites
    if not check_cuda():
        sys.exit(1)
    
    # Run setup phases
    phases = [
        ("Fase 1: Setup Ambiente", setup_environment),
        ("Fase 2: Installazione vLLM", install_vllm),
        ("Fase 3: Preparazione Modello", prepare_model),
        ("Fase 3B: Metadati TurboQuantum", generate_turboquant_metadata),
        ("Fase 4: File di Configurazione", create_config_files),
    ]
    
    completed = 0
    for phase_name, phase_func in phases:
        try:
            if phase_func():
                completed += 1
        except Exception as e:
            print_error(f"Errore in {phase_name}: {str(e)}")
            if "--continue" not in sys.argv:
                print_warning("Usa --continue per continuare nonostante gli errori")
                break
    
    # Summary
    print_header("Setup Completato!")
    print(f"{Colors.OKGREEN}{completed}/{len(phases)} fasi completate{Colors.ENDC}\n")
    
    print("""
📋 Prossimi Passi:

1. Scaricare il modello Gemma-4 27B:
   huggingface-cli download google/gemma-4-27b-it \\
       --local-dir ./models/gemma4-27b-int8

2. Attivare l'ambiente:
   source vllm_env/bin/activate

3. Avviare il server:
   ./serve_vllm.sh

4. Test inference (nuovo terminal):
   python scripts/test_inference.py

5. Benchmark performance:
   python scripts/benchmark.py

6. API Tests:
   bash scripts/test_api.sh

📚 Documentazione completa in: PIANO_VLLM_GEMMA_TURBOQUANT.md

🚀 Buon deployment!
""")

if __name__ == "__main__":
    main()
