"""Tests for the generator module.

NOTE: These tests require API keys to be set or a vLLM server running.
Tests marked with @pytest.mark.skip will pass without API access.
"""

import os
import pytest
from synthforge.generator import Generator, GenerationSample


class TestGeneratorInit:
    """Test generator initialization."""

    def test_valid_backends(self):
        """All declared backends should be valid."""
        for backend in Generator.SUPPORTED_BACKENDS:
            # Don't actually init clients (needs API keys)
            pass
        assert "openai" in Generator.SUPPORTED_BACKENDS
        assert "anthropic" in Generator.SUPPORTED_BACKENDS
        assert "vllm" in Generator.SUPPORTED_BACKENDS

    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            Generator(backend="invalid_backend")  # type: ignore

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_init(self):
        """Should initialize OpenAI client when API key is available."""
        gen = Generator(backend="openai", model="gpt-4o-mini")
        assert gen.backend == "openai"
        assert gen._client is not None

    def test_anthropic_init_no_key_raises(self):
        """Should raise if no Anthropic API key."""
        if os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY is set")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            Generator(backend="anthropic")


class TestGenerationSample:
    """Test the GenerationSample dataclass."""

    def test_creation(self):
        sample = GenerationSample(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"key": "value"},
        )
        assert len(sample.messages) == 1
        assert sample.metadata["key"] == "value"

    def test_default_metadata(self):
        sample = GenerationSample(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert sample.metadata == {}
