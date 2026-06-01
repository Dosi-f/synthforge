# Pipeline Configuration Guide

## Overview

SynthForge pipelines are defined in YAML config files. Each config specifies:
- Which backend and model to use
- Prompt template and input variations
- Filters to apply
- Quality and diversity analysis options
- Output format

## Anatomy of a Config

```yaml
pipeline:
  name: "my_pipeline"
  description: "What this pipeline does"

generation:
  backend: "openai"          # openai | anthropic | vllm
  model: "gpt-4o-mini"       # Model name or path
  temperature: 0.7
  max_tokens: 2048
  prompt_template: "..."     # String with {placeholders}
  inputs:                    # List of dicts to fill placeholders
    - variable: "value"
  num_samples_per_input: 5

filters:
  - type: "length"
    min_chars: 100
    max_chars: 10000

quality:
  scoring: "reward_model"    # reward_model | heuristic | none

diversity:
  enabled: true
  model: "all-MiniLM-L6-v2"
  threshold: 0.85

output:
  format: "jsonl"            # jsonl | json | axolotl | llamafactory
  path: "outputs/my_dataset.jsonl"
```

## Prompt Template Variables

Use `{variable_name}` in your template. These get filled from the `inputs` list.

Example:
```yaml
prompt_template: "Generate a question about {topic} at {difficulty} level."
inputs:
  - topic: "Python decorators"
    difficulty: "intermediate"
  - topic: "Rust ownership"
    difficulty: "advanced"
```

## Filter Combinations

Filters run in order. Use `CompositeFilter` when using the Python API, or list them in the config:

```yaml
filters:
  - type: "length"
    min_chars: 100
  - type: "language"
    target: "en"
```

## Diversity Analysis

Uses `sentence-transformers` to embed all outputs and compute pairwise cosine similarity. Samples above `threshold` are flagged as near-duplicates.

**GPU strongly recommended** for datasets > 500 samples.

## Known Issues

- Config validation is minimal — typos in `type` fields silently pass through
- `vllm` backend assumes a running server — no auto-launch
- `reward_model` scoring is not yet implemented

## Examples

See `configs/` directory for working examples:
- `default.yaml` — Basic single-turn generation
- `evol_instruct.yaml` — WizardLM-style instruction evolution
- `persona_driven.yaml` — Multi-turn persona conversations
