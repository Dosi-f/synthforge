"""Shared test fixtures."""

import pytest
from synthforge.generator import GenerationSample


@pytest.fixture
def sample_conversation() -> GenerationSample:
    """A basic sample for testing filters and exporters."""
    return GenerationSample(
        messages=[
            {"role": "user", "content": "What is Python?"},
            {
                "role": "assistant",
                "content": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            },
        ],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_empty() -> GenerationSample:
    """A sample with empty assistant response."""
    return GenerationSample(
        messages=[
            {"role": "user", "content": "Say nothing."},
            {"role": "assistant", "content": ""},
        ],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_dataset() -> list[GenerationSample]:
    """A small dataset of varied samples."""
    return [
        GenerationSample(
            messages=[
                {"role": "user", "content": "Explain decorators"},
                {
                    "role": "assistant",
                    "content": "Decorators in Python are functions that modify the behavior of other functions. "
                    "They use the @ syntax and are a form of metaprogramming.",
                },
            ],
            metadata={"topic": "python"},
        ),
        GenerationSample(
            messages=[
                {"role": "user", "content": "What is Rust used for?"},
                {
                    "role": "assistant",
                    "content": "Rust is a systems programming language focused on safety, speed, and concurrency. "
                    "It's used for building operating systems, game engines, browsers, and other performance-critical software.",
                },
            ],
            metadata={"topic": "rust"},
        ),
        GenerationSample(
            messages=[
                {"role": "user", "content": "Short answer please."},
                {"role": "assistant", "content": "OK."},
            ],
            metadata={"topic": "test"},
        ),
    ]
