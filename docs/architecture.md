# Architecture

## Overview

SynthForge is structured as a pipeline with four main stages:

```
Config → Generate → Filter → Score → Analyze → Export
```

Each stage is independently usable and composable.

## Core Modules

### `generator.py`

Multi-backend LLM generation. Currently supports:
- **OpenAI** — via `openai` Python SDK
- **Anthropic** — via `anthropic` Python SDK
- **vLLM** — via OpenAI-compatible API (EXPERIMENTAL)

Each backend implements the same interface:
```python
def _call_backend(self, prompt: str) -> str
```

### `filters.py`

Quality filters that remove or flag problematic samples. Filters implement:
```python
class BaseFilter:
    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]: ...
    def get_stats(self) -> dict: ...
```

Filters are chainable via `CompositeFilter`.

### `metrics.py`

Quality and diversity analysis. Two main components:
- **DiversityAnalyzer** — Embedding-based semantic dedup
- **QualityScorer** — Reward model scoring (incomplete)

### `exporters.py`

Output format converters. Each exporter implements:
```python
class BaseExporter:
    def export(self, samples: List[GenerationSample], output_path: Path) -> None: ...
```

### `prompts/`

Reusable prompt templates organized by generation strategy:
- `evol_instruct.py` — WizardLM-style instruction evolution
- `persona.py` — Persona-driven conversation generation

## Data Flow

```
Input (config YAML)
  │
  ├─→ Generator.generate()
  │     └─→ [GenerationSample, ...]
  │
  ├─→ FilterPipeline.apply()
  │     └─→ [GenerationSample, ...] (filtered)
  │
  ├─→ QualityScorer.score() [optional]
  │     └─→ quality scores attached to metadata
  │
  ├─→ DiversityAnalyzer.analyze() [optional]
  │     └─→ diversity report
  │
  └─→ Exporter.export()
        └─→ dataset.jsonl / dataset.json / axolotl format
```

## Design Decisions

### Why `GenerationSample` dataclass instead of plain dicts?

Type safety. As filters and metrics get more complex, having a structured type prevents silent bugs from dict key typos. The `metadata` field is intentionally loose (`Dict[str, Any]`) for flexibility.

### Why lazy-load embedding models?

Loading `sentence-transformers` models takes 2-5 seconds and consumes ~500MB RAM. If you're just generating with cloud APIs and not running diversity analysis, there's no reason to pay that cost.

### Why no async yet?

Adding `asyncio` to the generation loop is the next major refactoring target (v0.3.0). The current synchronous design works for batch sizes < 100 but becomes a bottleneck at scale. I'm holding off until the API surface stabilizes.

## GPU Dependency Graph

```
GPU Required:
  ├── vLLM backend (generation)
  ├── DiversityAnalyzer (embedding) — strongly recommended, not strictly required
  └── QualityScorer (reward model) — required for models > 1B parameters

GPU Optional (faster but works on CPU):
  ├── sentence-transformers for diversity analysis
  └── FAISS for large-scale similarity search (future)
```
