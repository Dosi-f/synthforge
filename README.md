# synthforge

SynthForge is a small synthetic data playground. The goal is not to generate huge datasets, but to test prompt recipes, filter bad samples, and keep track of which generated examples are actually useful for fine-tuning.

## Why I built this

I was trying to create a small instruction dataset for fine-tuning a 3B model. Using raw LLM outputs gave me 10K samples, but maybe 3K were actually usable. The rest were repetitive, off-topic, or just bad. I needed a way to filter and curate, not just generate.

## What it does

1. **Generate**: Run prompt templates against a local LLM
2. **Filter**: Score and filter outputs by quality metrics
3. **Curate**: Keep a log of what was kept and why

## Current state

- Generation: works with vLLM or llama.cpp backends
- Filtering: length-based, keyword-based, perplexity-based
- Curation: JSONL log with filter decisions

## What I'm working on

- Better quality filters (see `docs/quality_filters.md`)
- Prompt recipe testing (see `docs/prompt_recipe_notes.md`)
- Dedup integration with sieve

## Quick start

```bash
pip install -r requirements.txt
python synthforge.py generate --config recipes/alpaca_style.yaml --count 500
python synthforge.py filter --input raw_generations.jsonl --output filtered.jsonl
```

## Examples

- `examples/raw_generations.jsonl` — what raw output looks like
- `examples/filtered_dataset.jsonl` — after filtering
