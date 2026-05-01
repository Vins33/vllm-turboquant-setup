#!/bin/bash
# Build and Deploy Script for vLLM + TurboQuantum
# Automates Docker image creation and container startup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="vllm-gemma4-turboquant"
IMAGE_TAG="latest"
CONTAINER_NAME="vllm-gemma4-server"
MODEL_DIR="./models"
CACHE_DIR="./cache"
LOGS_DIR="./logs"

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker non trovato! Installa Docker desktop."
        exit 1
    fi
    print_success "Docker trovato: $(docker --version)"
}

check_nvidia() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi non trovato! Verifica driver NVIDIA."
        exit 1
    fi
    print_success "NVIDIA driver trovato"
}

check_docker_gpu() {
    if ! docker run --rm --runtime=nvidia nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "Docker GPU runtime potrebbe non essere configurato"
        print_info "Installa nvidia-container-runtime: https://github.com/NVIDIA/nvidia-container-runtime"
    fi
    print_success "Docker GPU runtime disponibile"
}

create_directories() {
    print_header "Creazione Directory"
    
    for dir in $MODEL_DIR $CACHE_DIR $LOGS_DIR; do
        mkdir -p "$dir"
        print_success "Directory creata: $dir"
    done
}

check_model() {
    print_header "Verifica Modello"
    
    if [ ! -f "$MODEL_DIR/gemma4-27b-int8/config.json" ]; then
        print_warning "Modello quantizzato non trovato in: $MODEL_DIR/gemma4-27b-int8"
        print_info "Scarica il modello con:"
        echo "  huggingface-cli download google/gemma-4-27b-it \\"
        echo "      --local-dir $MODEL_DIR/gemma4-27b-int8"
        echo ""
        read -p "Continuo comunque? (s/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Ss]$ ]]; then
            exit 1
        fi
    else
        print_success "Modello trovato"
    fi
}

build_image() {
    print_header "Build Immagine Docker"
    
    if [ ! -d "vllm" ]; then
        print_error "Directory vllm non trovata"
        print_info "Clonando vllm..."
        git clone https://github.com/mitkox/vllm.git
    fi
    
    print_info "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    print_warning "Questo richiede 20-30 minuti la prima volta..."
    
    docker build \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        -f Dockerfile \
        .
    
    print_success "Immagine Docker creata: $IMAGE_NAME:$IMAGE_TAG"
}

start_container() {
    print_header "Avvio Container"
    
    # Check if container already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container $CONTAINER_NAME già in esecuzione"
        read -p "Riavviare? (s/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            docker stop $CONTAINER_NAME
            docker rm $CONTAINER_NAME
        else
            return 0
        fi
    fi
    
    print_info "Avviando container: $CONTAINER_NAME"
    
    docker compose up -d vllm-gemma4
    
    print_success "Container avviato"
}

wait_for_ready() {
    print_header "Attesa Startup Server"
    
    print_info "Attendendo server startup... (max 2 minuti)"
    
    for i in {1..120}; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            print_success "Server pronto!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    print_warning "Timeout! Verifica i log:"
    echo "  docker logs $CONTAINER_NAME"
    return 1
}

test_api() {
    print_header "Test API Server"
    
    print_info "Testing OpenAI-compatible API..."
    
    response=$(curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gemma-4-27b",
            "prompt": "What is machine learning?",
            "max_tokens": 50,
            "temperature": 0.7
        }')
    
    if echo "$response" | grep -q "choices"; then
        print_success "API funziona correttamente!"
        echo "$response" | python -m json.tool | head -20
    else
        print_error "API test fallito!"
        echo "$response"
        return 1
    fi
}

show_status() {
    print_header "Status Container"
    
    docker compose ps
    
    print_info "Logs server (ultimi 50 righe):"
    docker logs --tail 50 $CONTAINER_NAME || true
    
    print_info "GPU utilization:"
    nvidia-smi
}

show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image"
    echo "  start       Start container"
    echo "  stop        Stop container"
    echo "  restart     Restart container"
    echo "  logs        Show container logs"
    echo "  status      Show status"
    echo "  shell       Open shell in container"
    echo "  benchmark   Run benchmark suite"
    echo "  full        Build + Start + Test (complete setup)"
    echo ""
}

# Main
case "${1:-full}" in
    build)
        check_docker
        check_nvidia
        create_directories
        check_model
        build_image
        ;;
    
    start)
        check_docker
        create_directories
        start_container
        wait_for_ready
        test_api
        ;;
    
    stop)
        print_header "Arresto Container"
        docker compose down
        print_success "Container arrestato"
        ;;
    
    restart)
        print_header "Riavvio Container"
        docker compose restart vllm-gemma4
        wait_for_ready
        test_api
        ;;
    
    logs)
        print_header "Container Logs"
        docker compose logs -f vllm-gemma4
        ;;
    
    status)
        show_status
        ;;
    
    shell)
        print_header "Opening Container Shell"
        docker exec -it $CONTAINER_NAME /bin/bash
        ;;
    
    benchmark)
        print_header "Running Benchmark Suite"
        docker exec -it $CONTAINER_NAME python advanced_benchmark.py \
            --model /models/gemma4-27b-int8 \
            --num-prompts 32 \
            --max-tokens 256 \
            --output /logs/benchmark_results.json
        print_success "Benchmark completato. Risultati in: logs/benchmark_results.json"
        ;;
    
    full)
        check_docker
        check_nvidia
        check_docker_gpu
        create_directories
        check_model
        build_image
        start_container
        wait_for_ready
        test_api
        show_status
        
        print_header "Setup Completato! 🎉"
        echo ""
        echo -e "${GREEN}Server disponibile su: http://localhost:8000${NC}"
        echo ""
        echo "📋 Comandi utili:"
        echo "  # Visualizza log in tempo reale"
        echo "  ./deploy.sh logs"
        echo ""
        echo "  # Apri shell nel container"
        echo "  ./deploy.sh shell"
        echo ""
        echo "  # Esegui benchmark"
        echo "  ./deploy.sh benchmark"
        echo ""
        echo "  # Arresta il server"
        echo "  ./deploy.sh stop"
        echo ""
        echo "📚 Test API con curl:"
        echo "  curl http://localhost:8000/v1/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"gemma-4-27b\", \"prompt\": \"Hello\", \"max_tokens\": 50}'"
        echo ""
        ;;
    
    help|-h|--help)
        show_usage
        ;;
    
    *)
        print_error "Comando non riconosciuto: $1"
        show_usage
        exit 1
        ;;
esac
