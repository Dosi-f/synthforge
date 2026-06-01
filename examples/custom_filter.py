#!/usr/bin/env python3
"""
Example: Building and using custom filters.

This shows how to create your own filter by subclassing BaseFilter.
The example filter removes samples that don't contain expected keywords
— useful for ensuring generated content stays on-topic.

Usage:
    python examples/custom_filter.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List

from synthforge.filters import BaseFilter
from synthforge.generator import GenerationSample


class KeywordFilter(BaseFilter):
    """
    Custom filter: ensures outputs contain at least one expected keyword.

    This is a simple example — real implementations might use
    NLI models or topic classifiers for this.

    Args:
        keywords: List of keywords to check for (case-insensitive).
        mode: 'any' (at least one keyword) or 'all' (all keywords).
    """

    def __init__(self, keywords: List[str], mode: str = "any"):
        self.keywords = [k.lower() for k in keywords]
        self.mode = mode
        self._removed: int = 0
        self._total: int = 0

    def apply(self, samples: List[GenerationSample]) -> List[GenerationSample]:
        self._total = len(samples)
        filtered = []

        for sample in samples:
            # Extract assistant response
            content = ""
            for msg in reversed(sample.messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    break

            if self._check_keywords(content):
                filtered.append(sample)
            else:
                self._removed += 1

        return filtered

    def _check_keywords(self, text: str) -> bool:
        if self.mode == "any":
            return any(kw in text for kw in self.keywords)
        elif self.mode == "all":
            return all(kw in text for kw in self.keywords)
        return True

    def get_stats(self) -> dict:
        return {
            "filter": "KeywordFilter",
            "keywords": self.keywords,
            "mode": self.mode,
            "total": self._total,
            "removed": self._removed,
            "kept": self._total - self._removed,
        }


def main():
    """Demo the custom filter with some fake samples."""
    print("=" * 60)
    print("SynthForge — Custom Filter Example")
    print("=" * 60)

    # Create some fake samples for demonstration
    samples = [
        GenerationSample(
            messages=[
                {"role": "user", "content": "Explain Python decorators"},
                {"role": "assistant", "content": "A Python decorator is a function that wraps another function..."},
            ],
        ),
        GenerationSample(
            messages=[
                {"role": "user", "content": "What is Rust?"},
                {"role": "assistant", "content": "Rust is a systems programming language focused on safety and performance..."},
            ],
        ),
        GenerationSample(
            messages=[
                {"role": "user", "content": "Tell me about JavaScript promises"},
                {"role": "assistant", "content": "Promises in JavaScript represent the eventual completion of an async operation..."},
            ],
        ),
    ]

    print(f"\nTesting with {len(samples)} samples...")

    # Filter: keep only Python-related content
    python_filter = KeywordFilter(
        keywords=["python", "decorator", "pip", "venv"],
        mode="any",
    )

    filtered = python_filter.apply(samples)
    stats = python_filter.get_stats()

    print(f"\nFilter: {stats['keywords']}")
    print(f"  Mode:   {stats['mode']}")
    print(f"  Total:  {stats['total']}")
    print(f"  Kept:   {stats['kept']}")
    print(f"  Removed:{stats['removed']}")

    print(f"\nKept samples:")
    for i, s in enumerate(filtered):
        content = s.messages[-1]["content"][:100]
        print(f"  [{i}] {content}...")

    print(f"\n{'=' * 60}")
    print("To use this filter in your pipeline:")
    print("  my_filter = KeywordFilter(keywords=['your', 'keywords'])")
    print("  filtered = my_filter.apply(your_dataset)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
