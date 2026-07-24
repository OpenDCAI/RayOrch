"""Synthetic benchmark utilities for elastic rebatching experiments."""

from .report import BenchmarkReport, summarize_reports, write_reports
from .runner import run_benchmark, run_paired_repetitions
from .workload import (
    ChildWork,
    ParentWork,
    SyntheticWorkload,
    generate_workload,
)

__all__ = [
    "BenchmarkReport",
    "ChildWork",
    "ParentWork",
    "SyntheticWorkload",
    "generate_workload",
    "run_benchmark",
    "run_paired_repetitions",
    "summarize_reports",
    "write_reports",
]
