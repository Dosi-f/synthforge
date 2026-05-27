#!/usr/bin/env python3
"""
Evol-Instruct pipeline example.

Generates complex instruction-following examples using the
WizardLM Evol-Instruct approach.

This is a more advanced example showing:
- Custom prompt templates from synthforge.prompts
- Multi-domain generation
- Diversity analysis with GPU (falls back to CPU)

Usage:
    python examples/evol_instruct_pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from synthforge import Generator, JSONLExporter
from synthforge.filters import LengthFilter, CompositeFilter
from synthforge.metrics import DiversityAnalyzer, QualityScorer
from synthforge.prompts import EVOL_INSTRUCT_SYSTEM, EVOL_INSTRUCT_USER
from synthforge.utils import format_duration

load_dotenv()


def main():
    print("=" * 60)
    print("SynthForge — Evol-Instruct Pipeline")
    print("=" * 60)

    start_time = time.time()

    # Use GPT-4o for higher quality instruction evolution
    generator = Generator(
        backend="openai",
        model="gpt-4o",  # Stronger model for complex instructions
        temperature=0.8,  # Higher creativity for diversity
        max_tokens=2048,
    )

    # Build prompt from Evol-Instruct templates
    prompt_template = EVOL_INSTRUCT_SYSTEM + "\n\n" + EVOL_INSTRUCT_USER

    # Multi-domain inputs
    inputs = [
        {"domain": "coding/algorithms", "complexity": "medium", "num_samples": 3, "constraints": "Include edge cases"},
        {"domain": "coding/system design", "complexity": "complex", "num_samples": 2, "constraints": "Include trade-off analysis"},
        {"domain": "writing/technical", "complexity": "medium", "num_samples": 3, "constraints": "Audience is senior engineers"},
        {"domain": "reasoning/logic", "complexity": "complex", "num_samples": 3, "constraints": "Multi-step deduction required"},
        {"domain": "analysis/data", "complexity": "medium", "num_samples": 3, "constraints": "Requires interpreting a described dataset"},
    ]

    print(f"\n[1] Generating across {len(inputs)} domains...")
    all_samples = []

    for inp in inputs:
        print(f"    Domain: {inp['domain']} ({inp['complexity']})")
        samples = generator.generate(
            prompt_template=prompt_template,
            inputs=[inp],
            num_samples_per_input=1,  # Template handles batching internally
        )
        all_samples.extend(samples)
        print(f"      → {len(samples)} samples")

    print(f"\n    Total raw samples: {len(all_samples)}")

    # Filter
    print("\n[2] Filtering...")
    filters = CompositeFilter([
        LengthFilter(min_chars=200, max_chars=8000),
    ])
    filtered = filters.apply(all_samples)
    print(f"    Kept: {len(filtered)} / {len(all_samples)}")

    # Diversity analysis
    if len(filtered) >= 5:
        print("\n[3] Diversity analysis...")
        try:
            analyzer = DiversityAnalyzer(
                model_name="all-MiniLM-L6-v2",
                similarity_threshold=0.85,
                # device="rocm"  # Uncomment if you have GPU
            )
            diversity = analyzer.analyze(filtered)
            print(f"    Unique ratio: {diversity['unique_ratio']:.2f}")
            print(f"    Mean similarity: {diversity['mean_pairwise_similarity']:.3f}")

            if diversity["mean_pairwise_similarity"] > 0.6:
                print("    ⚠ Low diversity detected — try increasing temperature")
            else:
                print("    ✓ Healthy diversity")
        except ImportError:
            print("    [SKIP] sentence-transformers not installed")

    # Quality scoring (placeholder)
    print("\n[4] Quality scoring (placeholder)...")
    scorer = QualityScorer()
    scores = scorer.score(filtered)
    avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0
    print(f"    Average score: {avg_score:.3f} (PLACEHOLDER)")

    # Export
    print("\n[5] Exporting...")
    output_path = "outputs/evol_instruct_example.jsonl"
    JSONLExporter().export(filtered, output_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {format_duration(elapsed)}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
