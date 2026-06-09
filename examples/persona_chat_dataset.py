#!/usr/bin/env python3
"""
Persona-driven conversation generation example.

Generates multi-turn conversations where the assistant maintains
a consistent persona. Useful for fine-tuning chatbots with
specific styles or domain expertise.

Usage:
    python examples/persona_chat_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from synthforge import Generator, JSONLExporter
from synthforge.prompts import PERSONA_SYSTEM, PERSONA_USER

load_dotenv()


def main():
    print("=" * 60)
    print("SynthForge — Persona Chat Dataset")
    print("=" * 60)

    generator = Generator(
        backend="openai",
        model="gpt-4o",  # Better at maintaining persona consistency
        temperature=0.9,  # Higher for creative persona expression
        max_tokens=1500,
    )

    prompt_template = PERSONA_SYSTEM + "\n\n" + PERSONA_USER

    # Define diverse personas
    inputs = [
        {
            "persona_description": "A patient, Socratic math tutor who never gives direct answers. You guide students to discover solutions themselves through questions.",
            "scenario": "A student is struggling with understanding why the derivative of sin(x) is cos(x).",
            "num_turns": 5,
        },
        {
            "persona_description": "A deadpan, sarcastic senior software engineer doing code review. You're not mean, just brutally honest and allergic to over-engineering.",
            "scenario": "Reviewing a pull request where a junior dev wrapped a simple if-else in an AbstractSingletonProxyFactoryBean.",
            "num_turns": 4,
        },
        {
            "persona_description": "An over-enthusiastic historian who relates EVERYTHING back to ancient Rome. You find Roman parallels in the most unlikely topics.",
            "scenario": "Someone asks about modern project management techniques.",
            "num_turns": 4,
        },
        {
            "persona_description": "A laconic cowboy-themed Linux sysadmin. Every answer is short, uses cowboy metaphors, and ends with a practical command.",
            "scenario": "Helping a new team member debug a Docker networking issue.",
            "num_turns": 3,
        },
    ]

    print(f"\n[1] Generating conversations for {len(inputs)} personas...")
    all_samples = []

    for inp in inputs:
        print(f"\n    Persona: {inp['persona_description'][:80]}...")
        # Generate multiple variations per persona
        samples = generator.generate(
            prompt_template=prompt_template,
            inputs=[inp],
            num_samples_per_input=2,
        )
        all_samples.extend(samples)
        print(f"      → {len(samples)} conversation(s)")

    print(f"\n    Total conversations: {len(all_samples)}")

    # Export
    output_path = "outputs/persona_conversations.jsonl"
    JSONLExporter().export(all_samples, output_path)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_samples)} conversations → {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
