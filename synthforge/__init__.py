"""
SynthForge: Synthetic data generation toolkit for LLM fine-tuning.
"""

__version__ = "0.2.1"
__author__ = "Dosi-f"

from synthforge.generator import Generator
from synthforge.exporters import JSONExporter, JSONLExporter

__all__ = ["Generator", "JSONExporter", "JSONLExporter"]
