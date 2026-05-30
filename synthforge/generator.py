"""
Core generator module — handles multi-backend LLM generation.

Backends:
- openai: Uses OpenAI-compatible API
- anthropic: Uses Anthropic Messages API
- vllm: Local vLLM server (EXPERIMENTAL)

TODO:
- [ ] Add retry with exponential backoff for API failures
- [ ] Implement streaming via generator interface
- [ ] Add cost tracking per generation run
- [ ] Support async batch generation with asyncio
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import json
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GenerationSample:
    """A single generated sample with metadata."""

    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator:
    """Main generation interface.

    Args:
        backend: Which LLM backend to use.
        model: Model name or path.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per response.
        api_key: API key (reads from env if not provided).
    """

    SUPPORTED_BACKENDS = ("openai", "anthropic", "vllm")

    def __init__(
        self,
        backend: Literal["openai", "anthropic", "vllm"] = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
    ):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Choose from: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if backend == "openai":
            self._client = self._init_openai(api_key)
        elif backend == "anthropic":
            self._client = self._init_anthropic(api_key)
        elif backend == "vllm":
            # EXPERIMENTAL: Assumes a vLLM server is running locally
            self._client = self._init_vllm(api_key)

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set. Pass api_key or set env var.")

        return OpenAI(api_key=key)

    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")

        return Anthropic(api_key=key)

    def _init_vllm(self, api_key: Optional[str]):
        """
        Initialize vLLM client.

        EXPERIMENTAL — assumes vLLM OpenAI-compatible server is running.
        See: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

        TODO: Add health check, auto-start option for local server.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for vLLM backend.")

        vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000/v1")
        return OpenAI(base_url=vllm_host, api_key="not-needed")

    def generate(
        self,
        prompt_template: str,
        inputs: List[Dict[str, str]],
        num_samples_per_input: int = 1,
    ) -> List[GenerationSample]:
        """
        Generate synthetic data using the configured backend.

        Args:
            prompt_template: Template string with {key} placeholders.
            inputs: List of dicts with values to fill template placeholders.
            num_samples_per_input: Number of samples to generate per input variation.

        Returns:
            List of GenerationSample objects.
        """
        results: List[GenerationSample] = []

        for input_dict in inputs:
            prompt = prompt_template.format(**input_dict)

            for _ in range(num_samples_per_input):
                try:
                    response = self._call_backend(prompt)
                    sample = GenerationSample(
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        metadata={
                            "backend": self.backend,
                            "model": self.model,
                            "input_context": input_dict,
                            "temperature": self.temperature,
                        },
                    )
                    results.append(sample)
                except Exception as e:
                    # TODO: Better error handling — retry, collect failures
                    print(f"[WARN] Generation failed for input {input_dict}: {e}")
                    continue

        return results

    def _call_backend(self, prompt: str) -> str:
        """Dispatch to the appropriate backend call."""
        if self.backend == "openai":
            return self._call_openai(prompt)
        elif self.backend == "anthropic":
            return self._call_anthropic(prompt)
        elif self.backend == "vllm":
            return self._call_vllm(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_vllm(self, prompt: str) -> str:
        """
        EXPERIMENTAL: Call local vLLM server.
        Falls back to streaming=False for now; TODO: add streaming support.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_batch(
        self,
        prompt_template: str,
        inputs: List[Dict[str, str]],
        batch_size: int = 8,
    ) -> List[GenerationSample]:
        """
        Generate samples in batches.

        NOTE: Currently runs sequentially; true batching will be added
        with async support. vLLM backend benefits most from this.
        """
        # TODO: Implement async concurrent batching
        # For now, just a convenience wrapper
        results = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            results.extend(
                self.generate(prompt_template, batch_inputs, num_samples_per_input=1)
            )
        return results
