"""Docling real-workload adapters for the Multigrain v3.6 runtime."""

from .core_v36 import (
    DoclingTableFormerV1BatchV36Pipeline,
    DoclingTableFormerV2BatchV36Pipeline,
    DoclingTableJobV36Pipeline,
    run_v36,
)

__all__ = [
    "DoclingTableFormerV1BatchV36Pipeline",
    "DoclingTableFormerV2BatchV36Pipeline",
    "DoclingTableJobV36Pipeline",
    "run_v36",
]
