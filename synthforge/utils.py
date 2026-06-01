"""Shared utilities."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml required. Install with: pip install pyyaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_md5(text: str) -> str:
    """Compute MD5 hash of a string — useful for dedup keys."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_output_dir() -> Path:
    """Get the configured output directory."""
    return Path(os.getenv("SYNTHFORGE_OUTPUT_DIR", "./outputs"))


def get_cache_dir() -> Path:
    """Get the configured cache directory."""
    return Path(os.getenv("SYNTHFORGE_CACHE_DIR", "./.cache"))


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
