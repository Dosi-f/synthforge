# Contributing

If you've ever struggled with building training datasets, you probably have opinions about how this should work. I want to hear them.

## Good first contributions

- New prompt templates for common use cases (code generation, instruction following, domain-specific)
- Additional export formats
- Filter implementations (language-specific, domain-specific)
- Quality scoring improvements

## Setup

```bash
git clone https://github.com/Dosi-f/synthforge.git
pip install -e ".[dev]"
```

## Adding a new filter

```python
# synthforge/filters/my_filter.py
from . import Filter

class MyFilter(Filter):
    def check(self, text: str) -> bool:
        # return True to keep, False to discard
        return True
```

Register it in `synthforge/filters/__init__.py` and add a test.

## Adding a prompt template

Look at `synthforge/prompts/evol_instruct.py` for the pattern. Templates are just classes with a `generate()` method.

## Guidelines

- Test your changes with at least one backend (vLLM or OpenAI)
- Don't break existing export formats
- If you add a new backend, document the env vars needed
