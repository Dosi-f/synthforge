#!/usr/bin/env python3
"""
Basic generation example using the OpenAI backend.

This example generates a small Python Q&A dataset, filters it,
and exports to JSONL. Run it as a quick smoke test.

Usage:
    python examples/basic_generation.py

Requirements:
    - OPENAI_API_KEY set in .env or environment
    - pip install synthforge (or pip install -e .)
"""

import os
import sys

# Add project root to path if running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from synthforge import Generator, JSONLExporter
from synthforge.filters import LengthFilter, CompositeFilter
from synthforge.metrics import DiversityAnalyzer

load_dotenv()


def main():
    print("=" * 60)
    print("SynthForge — Basic Generation Example")
    print("=" * 60)

    # Initialize generator with OpenAI backend
    print("\n[1] Initializing generator...")
    generator = Generator(
        backend="openai",
        model="gpt-4o-mini",  # Cheap, fast, good enough for testing
        temperature=0.7,
        max_tokens=1024,
    )

    # Define a prompt template
    prompt_template = """Generate a Python programming question and a detailed, correct answer.

Topic: {topic}
Difficulty: {difficulty}

Format your response as a JSON object with "question" and "answer" fields."""

    # Define input variations
    inputs = [
        {"topic": "asyncio", "difficulty": "intermediate"},
        {"topic": "decorators", "difficulty": "beginner"},
        {"topic": "type hints", "difficulty": "intermediate"},
        {"topic": "context managers", "difficulty": "intermediate"},
        {"topic": "generators", "difficulty": "beginner"},
        {"topic": "metaclasses", "difficulty": "advanced"},
        {"topic": "dataclasses", "difficulty": "beginner"},
        {"topic": "descriptors", "difficulty": "advanced"},
    ]

    # Generate!
    print(f"\n[2] Generating with {len(inputs)} input variations...")
    samples = generator.generate(
        prompt_template=prompt_template,
        inputs=inputs,
        num_samples_per_input=3,  # 8 inputs × 3 = 24 samples
    )

    print(f"    Generated {len(samples)} raw samples")

    # Apply filters
    print("\n[3] Applying filters...")
    filters = CompositeFilter([
        LengthFilter(min_chars=150, max_chars=5000),
    ])
    filtered = filters.apply(samples)
    stats = filters.get_stats()
    print(f"    {stats['sub_filters'][0]['removed']} samples removed by length filter")
    print(f"    {stats['sub_filters'][0]['kept']} samples kept")

    # Quick diversity check (CPU-only for example, GPU recommended for real use)
    if len(filtered) >= 3:
        print("\n[4] Analyzing diversity (CPU mode)...")
        analyzer = DiversityAnalyzer(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.85,
            device="cpu",  # Force CPU for portability
        )
        diversity = analyzer.analyze(filtered)
        print(f"    Unique ratio: {diversity['unique_ratio']:.2f}")
        print(f"    Mean pairwise similarity: {diversity['mean_pairwise_similarity']:.3f}")
        print(f"    Near-duplicate pairs: {len(diversity['near_duplicate_pairs'])}")

    # Export
    print("\n[5] Exporting...")
    output_path = "outputs/basic_example_output.jsonl"
    JSONLExporter().export(filtered, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("Done! Summary:")
    print(f"  Raw samples:     {len(samples)}")
    print(f"  After filtering: {len(filtered)}")
    print(f"  Output:          {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
