# GPU Benchmarks

## Setup

| Component | Spec |
|-----------|------|
| GPU | 2× AMD Instinct MI210 (32GB each) |
| CPU | AMD Ryzen 9 5950X (16-core) |
| RAM | 64GB DDR4-3600 |
| ROCm | 6.2 |
| vLLM | 0.6.0 |
| PyTorch | 2.4.0 |

## Generation Throughput

Generating 1,000 samples with different backends on the Evol-Instruct prompt template.

| Backend | Model | Batch Size | Samples/sec | VRAM Used |
|---------|-------|------------|-------------|-----------|
| OpenAI | gpt-4o-mini | N/A | ~5.2 | 0 GB |
| OpenAI | gpt-4o | N/A | ~1.8 | 0 GB |
| vLLM | Llama-3.1-8B-Instruct | 32 | ~47.3 | 18.2 GB |
| vLLM | Llama-3.1-8B-Instruct | 64 | ~52.1 | 20.1 GB |
| vLLM | Mistral-7B-v0.3 | 32 | ~51.8 | 16.5 GB |
| vLLM | Mixtral-8×7B (Q4) | 16 | ~18.4 | 22.8 GB |

**Key takeaway:** vLLM on a single 3090 is ~10× faster than cloud APIs for small models. For 8B models, the speed advantage drops but cost advantage increases for large-scale generation.

## Embedding Throughput

Computing embeddings for diversity analysis with `all-MiniLM-L6-v2`:

| Device | 1,000 samples | 10,000 samples | 50,000 samples |
|--------|---------------|----------------|----------------|
| CPU (32 threads) | 2.1s | 18.7s | 94.3s |
| GPU (MI210) | 0.4s | 3.2s | 14.8s |
| Speedup | 5.3× | 5.8× | 6.4× |

## Memory Scaling (vLLM — Experimental)

| Model | Max Batch Size (24GB) | Max Context |
|-------|----------------------|-------------|
| Llama-3.1-8B | 128 | 4096 |
| Llama-3.1-8B | 32 | 16384 |
| Mistral-7B | 128 | 4096 |
| Mixtral-8×7B (Q4) | 32 | 4096 |
| Llama-3.1-70B (Q4) | ❌ OOM | ❌ |

**70B+ models cannot run on a single 32GB GPU even with 4-bit quantization.** This is the main motivation for cloud GPU access — validating the vLLM backend against larger models requires MI250X or better.

## Known Bottlenecks

- **CPU-GPU transfer**: Embedding models spend ~15% of time moving data
- **vLLM scheduler**: Under-utilizes GPU 2 in dual-GPU setups (TODO: tensor parallelism config)
- **JSON parsing**: Generated outputs that need JSON repair add ~200ms/sample (TODO: constrained decoding)
- **No FP8**: Not tested with FP8 kv-cache — could reduce VRAM usage by 40%
