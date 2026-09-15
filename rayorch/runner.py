"""Convenience execution entry point with an automatically managed lifetime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._execution.executor import Executor
from ._program.plan import CompiledProgram
from .api import Pipeline
from .result import RunResult


def run(
    pipeline: Pipeline | CompiledProgram,
    *source_columns: Sequence[Any],
    microbatch_size: int | None = None,
    max_active_microbatches: int = 1,
    address: str | None = None,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """Execute one finite input with an automatically closed Executor."""

    with Executor(
        pipeline,
        address=address,
        ray_init_kwargs=ray_init_kwargs,
    ) as executor:
        return executor.run(
            *source_columns,
            microbatch_size=microbatch_size,
            max_active_microbatches=max_active_microbatches,
        )


__all__ = ["run"]
