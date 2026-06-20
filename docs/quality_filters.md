# Quality Filters

## Filters implemented

### Length filter
- Min: 20 characters (removes empty/trivial responses)
- Max: 2000 characters (removes rambling/looping)
- Configurable per template

### Keyword filter
- Remove responses containing: "I cannot", "I'm sorry", "as an AI"
- These indicate the model refused or went into assistant mode
- Sometimes the refusal is correct (e.g., harmful requests) - manual review needed

### Perplexity filter
- Use a small LM to score the output
- Very low perplexity = likely boilerplate/repetitive
- Very high perplexity = likely garbled/nonsensical
- Sweet spot depends on the model and task

### Dedup filter
- Exact match: hash-based
- Near-dedup: embedding similarity > 0.95
- Important because LLMs tend to repeat similar outputs

### Code filter (for code tasks)
- Check if output contains valid Python (ast.parse)
- Check for common patterns: def, class, import, return
- Reject if output is just an explanation with no code

## Filter pipeline

```
raw_generations.jsonl
    ↓ length_filter
    ↓ keyword_filter
    ↓ perplexity_filter
    ↓ dedup_filter
    ↓ code_filter (if code task)
filtered_dataset.jsonl
```

## What I want to add

- LLM-as-judge: use a stronger model to rate quality
- Topic relevance: check if output stays on topic
- Factual accuracy: hard problem, but even basic checks help
