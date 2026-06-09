# Security

## Reporting

Email synthforge@protonmail.com for security issues. I'll respond within 48 hours.

## Scope

SynthForge generates synthetic data using LLM APIs and local models. Security considerations:

- **API keys**: Stored in `.env` or environment variables. `.env.example` shows required vars without real values.
- **Generated content**: SynthForge doesn't filter for harmful content by default. If you're generating data for public-facing applications, enable the `safety` filter.
- **Local inference**: vLLM and Ollama backends run with system-level permissions. Use containerization if running untrusted models.

## Best practices

- Use API keys with minimal permissions (read-only where possible)
- Run generation pipelines in isolated environments
- Review generated datasets before using them for fine-tuning
