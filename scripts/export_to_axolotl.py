#!/usr/bin/env python3
"""
Convenience script: Convert a SynthForge JSONL output to Axolotl format.

Usage:
    python scripts/export_to_axolotl.py \
        --input outputs/my_dataset.jsonl \
        --output outputs/axolotl_ready.json

This is a thin wrapper around synthforge.exporters.AxolotlExporter.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthforge.generator import GenerationSample
from synthforge.exporters import AxolotlExporter


def main():
    parser = argparse.ArgumentParser(
        description="Convert SynthForge JSONL to Axolotl format"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Load JSONL
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(
                GenerationSample(
                    messages=record.get("messages", []),
                    metadata=record.get("metadata", {}),
                )
            )

    print(f"Loaded {len(samples)} samples from {input_path}")

    # Export
    exporter = AxolotlExporter()
    exporter.export(samples, args.output)

    print(f"Ready for: axolotl train --dataset {args.output}")


if __name__ == "__main__":
    main()
