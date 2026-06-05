"""
Export generated datasets to various formats.

Supported formats:
- JSON: Single JSON array
- JSONL: One JSON object per line
- Axolotl: Chat template format (TODO: implement fully)
- LLaMA-Factory: ShareGPT format (TODO)
- HuggingFace datasets: Arrow-based dataset (TODO)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from synthforge.generator import GenerationSample


class BaseExporter(ABC):
    """Base exporter interface."""

    @abstractmethod
    def export(self, samples: List[GenerationSample], output_path: str | Path) -> None:
        """Export samples to the given path."""
        ...


class JSONExporter(BaseExporter):
    """Export as a single JSON array."""

    def export(self, samples: List[GenerationSample], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "messages": s.messages,
                "metadata": s.metadata,
            }
            for s in samples
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[JSONExporter] Exported {len(samples)} samples → {output_path}")


class JSONLExporter(BaseExporter):
    """Export as JSONL (one JSON object per line)."""

    def export(self, samples: List[GenerationSample], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                record = {
                    "messages": s.messages,
                    "metadata": s.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[JSONLExporter] Exported {len(samples)} samples → {output_path}")


class AxolotlExporter(BaseExporter):
    """
    Export in Axolotl chat format.

    Axolotl expects: {"conversations": [{"from": "human", "value": "..."},
                                         {"from": "gpt", "value": "..."}]}

    TODO: Implement role mapping from standard OpenAI format to Axolotl format.
    Currently a rough first pass.
    """

    ROLE_MAP = {
        "system": "system",
        "user": "human",
        "assistant": "gpt",
    }

    def export(self, samples: List[GenerationSample], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: Proper role mapping with validation
        conversations = []
        for s in samples:
            axolotl_conv = []
            for msg in s.messages:
                role = self.ROLE_MAP.get(msg["role"], msg["role"])
                axolotl_conv.append({"from": role, "value": msg["content"]})
            conversations.append({"conversations": axolotl_conv})

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

        print(f"[AxolotlExporter] Exported {len(conversations)} conversations → {output_path}")
