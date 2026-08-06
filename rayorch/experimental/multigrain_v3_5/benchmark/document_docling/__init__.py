"""Docling real-workload adapters for the Multigrain v3.5 runtime."""

from .core_v35 import (
    DoclingTableFormerV1BatchV35Pipeline,
    DoclingTableFormerV2BatchV35Pipeline,
    DoclingTableJobV35Pipeline,
    run_v35,
)

__all__ = [
    "DoclingTableFormerV1BatchV35Pipeline",
    "DoclingTableFormerV2BatchV35Pipeline",
    "DoclingTableJobV35Pipeline",
    "run_v35",
]
