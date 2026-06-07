# SynthForge

I needed a dataset to fine-tune a model for Indonesian legal documents. There wasn't one. So I used GPT-4 to generate synthetic examples, and realized the tooling for this is terrible.

SynthForge is what I wish existed: a pipeline for generating synthetic training data using LLMs, with quality filtering and diversity checks built in.

## The idea

```
Prompt templates  ->  LLM generation  ->  Quality filters  ->  Diversity scoring  ->  Export
```

You define what kind of data you want (prompt templates), SynthForge generates it using your LLM of choice, filters out garbage, checks for near-duplicates, and exports in the format your fine-tuning framework expects.

## Supported backends

- OpenAI (gpt-4o, gpt-4o-mini)
- Anthropic (claude-3.5-sonnet)
- Local via vLLM or Ollama
- Any OpenAI-compatible endpoint

## Example

```python
from synthforge import Pipeline, PromptTemplate

template = PromptTemplate(
    system="You are a legal expert.",
    user="Generate a Q&A pair about: {topic}",
    topics=["contract law", "property rights", "criminal procedure"],
)

pipe = Pipeline(
    template=template,
    backend="vllm",  # or "openai", "anthropic"
    model="meta-llama/Llama-3-8B",
    n_samples=1000,
    filters=["length", "language", "safety"],
    diversity_threshold=0.8,
)

dataset = pipe.run()
dataset.to_jsonl("legal_qa.jsonl")
dataset.to_axolotl("legal_qa_axolotl.yaml")
```

## Filters

Built-in filters:
- `length` — min/max token count
- `language` — detect and filter by language
- `safety` — basic content safety check
- `format` — regex/JSON validation
- `dedup` — embedding-based near-duplicate removal

Write your own:
```python
from synthforge.filters import Filter

class NoAllCaps(Filter):
    def check(self, text):
        return text != text.upper()
```

## Diversity scoring

Embeds all generated samples, runs clustering, and reports coverage. If 90% of your samples are saying the same thing, you'll know.

## Export formats

- JSONL (generic)
- Axolotl config + dataset
- LLaMA-Factory format
- HuggingFace `datasets` format

## Install

```bash
pip install synthforge
```

For local inference with vLLM:
```bash
pip install synthforge[vllm]
```

## Status

Works. I've used it to generate ~50k samples for fine-tuning. The quality filtering is the most useful part — raw LLM output is surprisingly noisy.

Apache 2.0 License.
