"""
Quality filters for generated datasets.

Filters are designed to be composable — chain them together to build
a quality pipeline.

Available filters:
- LengthFilter: Remove samples that are too short or too long
- LanguageFilter: Detect and filter non-English outputs (TODO)
- ToxicityFilter: Flag toxic/unsafe content (TODO — needs a classifier model)
- FormatFilter: Validate output matches expected JSON structure (TODO)
"""

from abc import ABC, abstractmethod
from typing import List

from synthforge.generator import GenerationSample


class BaseFilter(ABC):
    """Base filter class."""

    @abstractmethod
    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]:
        """Apply the filter and return filtered samples."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Return filter statistics after application."""
        ...


class LengthFilter(BaseFilter):
    """Filter samples based on character or token length.

    Args:
        min_chars: Minimum character count (inclusive).
        max_chars: Maximum character count (inclusive).
        field: Which message field to check. Default: last assistant message.
    """

    def __init__(
        self, min_chars: int = 50, max_chars: int = 10000, field: str = "assistant"
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.field = field
        self._removed: int = 0
        self._total: int = 0

    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]:
        self._total = len(samples)
        filtered = []

        for s in samples:
            # Get last message matching the field
            content = self._extract_content(s)
            if content is None:
                self._removed += 1
                continue

            char_len = len(content)
            if self.min_chars <= char_len <= self.max_chars:
                filtered.append(s)
            else:
                self._removed += 1

        return filtered

    def _extract_content(self, sample: GenerationSample) -> str | None:
        """Extract target content from a sample."""
        if self.field == "assistant":
            for msg in reversed(sample.messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
        elif self.field == "user":
            for msg in sample.messages:
                if msg.get("role") == "user":
                    return msg.get("content", "")
        # TODO: Support other field types
        return None

    def get_stats(self) -> dict:
        return {
            "filter": "LengthFilter",
            "total": self._total,
            "removed": self._removed,
            "kept": self._total - self._removed,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
        }


class LanguageFilter(BaseFilter):
    """
    Filter non-English outputs.

    TODO: Implement using fastText or langdetect.
    Currently a placeholder.
    """

    def __init__(self, target_language: str = "en"):
        self.target_language = target_language
        self._removed: int = 0
        self._total: int = 0

    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]:
        self._total = len(samples)
        # TODO: Implement language detection
        # For now, pass-through
        return samples

    def get_stats(self) -> dict:
        return {
            "filter": "LanguageFilter",
            "total": self._total,
            "removed": self._removed,
            "note": "NOT YET IMPLEMENTED — pass-through only",
        }


class CompositeFilter(BaseFilter):
    """
    Chain multiple filters together.

    Usage:
        pipeline = CompositeFilter([LengthFilter(min_chars=100), LanguageFilter()])
        clean = pipeline.apply(dataset)
    """

    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters

    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]:
        result = samples
        for f in self.filters:
            result = f.apply(result)
        return result

    def get_stats(self) -> dict:
        return {"filter": "CompositeFilter", "sub_filters": [f.get_stats() for f in self.filters]}
