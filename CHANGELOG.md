# Changelog

## 0.3.0 — 2025-06-01
- Persona-driven data generation (define character profiles, generate conversations)
- Evol-Instruct pipeline (self-instruct evolution for harder prompts)
- Quality scoring with pluggable reward models
- Added LLaMA-Factory export format

## 0.2.0 — 2025-05-15
- Anthropic Claude backend support
- Diversity scoring via embedding clustering
- New filters: language detection, content safety
- Fixed: vLLM backend crashing on long prompts (>2048 tokens)

## 0.1.1 — 2025-04-28
- Fixed: JSONL export encoding issues with non-ASCII characters
- Added retry logic for API rate limits
- Better error messages when backend is unreachable

## 0.1.0 — 2025-04-15
- Initial release
- OpenAI and vLLM backends
- Basic prompt templates
- JSONL and Axolotl export
- Length and format filters
