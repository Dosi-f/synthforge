# Quality Metrics

## Current Metrics

### Length Filter

Simplest quality check: removes samples that are too short (likely incomplete) or too long (likely rambling).

| Metric | Description |
|--------|-------------|
| Min chars | Minimum character count (default: 50) |
| Max chars | Maximum character count (default: 10,000) |
| Removed % | Percentage of samples filtered out |

### Diversity Score

Uses cosine similarity between sentence embeddings to measure how "different" samples are from each other.

| Metric | Description |
|--------|-------------|
| Near-duplicate pairs | Number of sample pairs with similarity ≥ threshold |
| Unique ratio | Fraction of samples that aren't flagged as duplicates |
| Mean pairwise similarity | Average similarity across all sample pairs |

**Interpretation:**
- `mean_pairwise_similarity < 0.3`: Very diverse, possibly too scattered
- `0.3 < mean_pairwise_similarity < 0.6`: Healthy diversity
- `mean_pairwise_similarity > 0.6`: Low diversity, many near-duplicates

## Planned Metrics (Not Yet Implemented)

### Reward Model Scoring

Use `RLHFlow/ArmoRM-Llama3-8B` or similar to score outputs across dimensions:
- Helpfulness
- Accuracy
- Coherence
- Safety

**GPU required** — loading an 8B reward model needs ≥ 16GB VRAM.

### Self-BLEU

N-gram based diversity metric. Faster than embedding-based analysis (no GPU needed) but less semantically aware.

### Format Compliance

Validate that outputs match expected JSON schema. Important for structured generation tasks.

### Toxicity / Safety

Run a small classifier to flag potentially harmful content. Models like `unitary/toxic-bert` are lightweight enough for CPU.

## Benchmarking Quality

I'm actively working on benchmarks comparing SynthForge-generated datasets against human-curated ones across fine-tuning runs. Results will be published when I have statistically meaningful data.
