#!/usr/bin/env bash
# SynthForge GPU Benchmark Script
# Measures generation throughput across backends.
#
# Prerequisites:
#   - vLLM server running on localhost:8000 (if testing vllm backend)
#   - API keys set in .env
#
# Usage:
#   bash scripts/benchmark_gpu.sh [num_samples]

set -euo pipefail

NUM_SAMPLES=${1:-100}
OUTPUT_DIR="outputs/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}=== SynthForge GPU Benchmark ===${NC}"
echo -e "Samples per test: ${NUM_SAMPLES}"
echo -e "Timestamp: ${TIMESTAMP}\n"

mkdir -p "${OUTPUT_DIR}"

# Function to time a generation run
bench_backend() {
    local backend=$1
    local model=$2
    local label=$3

    echo -e "${CYAN}[BENCH] ${label} (${backend}/${model})${NC}"

    local start_time=$(date +%s.%N)

    python -c "
from synthforge import Generator
g = Generator(backend='${backend}', model='${model}', temperature=0.7, max_tokens=512)
samples = g.generate(
    prompt_template='Explain {topic} in detail with examples.',
    inputs=[{'topic': t} for t in ['Python decorators', 'Rust ownership', 'SQL joins', 'Docker volumes', 'Git rebase'] * $(( ${NUM_SAMPLES} / 5 ))],
    num_samples_per_input=1,
)
print(f'Samples: {len(samples)}')
" 2>&1 || echo "FAILED"

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    local samples_per_sec=$(echo "${NUM_SAMPLES} / $elapsed" | bc -l)

    echo -e "  Time: ${elapsed}s | Throughput: ${samples_per_sec} samples/sec\n"

    # Log results
    echo "${backend},${model},${elapsed},${samples_per_sec}" >> "${OUTPUT_DIR}/bench_${TIMESTAMP}.csv"
}

# Write header
echo "backend,model,time_seconds,samples_per_sec" > "${OUTPUT_DIR}/bench_${TIMESTAMP}.csv"

# --- Benchmarks ---

# OpenAI (fast/cheap)
echo -e "${YELLOW}--- Cloud API Benchmarks ---${NC}\n"
bench_backend "openai" "gpt-4o-mini" "OpenAI GPT-4o-mini"

# OpenAI (powerful)
bench_backend "openai" "gpt-4o" "OpenAI GPT-4o"

# vLLM local — only if server is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}--- vLLM Local Benchmarks ---${NC}\n"
    VLLM_MODEL=${VLLM_MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}
    bench_backend "vllm" "${VLLM_MODEL}" "vLLM Local"
else
    echo -e "${YELLOW}[SKIP] vLLM server not running on localhost:8000${NC}"
    echo -e "Start with: vllm serve ${VLLM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}\n"
fi

# Embedding benchmark (CPU vs GPU)
echo -e "${YELLOW}--- Embedding Benchmark ---${NC}\n"

echo -e "${CYAN}[BENCH] Embedding (sentence-transformers, CPU)${NC}"
python -c "
import time
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
texts = ['Sample text ' * 5 + str(i) for i in range(${NUM_SAMPLES})]
t0 = time.time()
embeddings = model.encode(texts, show_progress_bar=False)
t1 = time.time()
print(f'  Time: {t1-t0:.2f}s | Samples: {len(embeddings)} | Device: CPU')
" 2>&1 || echo "  FAILED (sentence-transformers not installed?)"

echo -e "\n${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${OUTPUT_DIR}/bench_${TIMESTAMP}.csv"
