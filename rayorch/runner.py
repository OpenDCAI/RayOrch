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
    input_batch_size: int | None = None,
    max_active_input_batches: int = 1,
    address: str | None = None,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """Execute one finite input with an automatically closed Executor.

    ``input_batch_size`` counts aligned source rows (None uses the full input);
    ``max_active_input_batches`` limits overlapping input batch lifecycles.
    Worker execution microbatch sizes are configured per Call with ``batch_size``.
    """

    with Executor(
        pipeline,
        address=address,
        ray_init_kwargs=ray_init_kwargs,
    ) as executor:
        return executor.run(
            *source_columns,
            input_batch_size=input_batch_size,
            max_active_input_batches=max_active_input_batches,
        )


__all__ = ["run"]
