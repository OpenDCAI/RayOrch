"""Fixed, Ray-free static compiler pipeline.

This is not a pluggable pass manager. The only supported order is:
verify logical -> analyze -> configurable canonicalize -> lower -> verify plan.
Disabling canonicalization retains the same analysis, lowering, and final
verification phases.
"""

from __future__ import annotations

from typing import Mapping

from .._model import CallRef, PoolRef
from .analysis import analyze as _analyze
from .logical import LogicalProgram, UdfSpec
from .lowering import _canonicalize, _lower
from .plan import CompiledProgram
from .verify import verify_logical as _verify_logical
from .verify import verify_runtime_plan as _verify_runtime_plan


def compile_logical(
    logical: LogicalProgram,
    call_options: Mapping[CallRef, tuple[tuple[str, object], ...]],
    call_pools: Mapping[CallRef, PoolRef],
    pool_udfs: Mapping[PoolRef, UdfSpec],
    pool_options: Mapping[PoolRef, tuple[tuple[str, object], ...]],
    *,
    optimize: bool = True,
) -> CompiledProgram:
    """Compile one frozen logical graph through the fixed pipeline."""

    _verify_logical(logical)
    analysis = _analyze(logical)
    canonical = _canonicalize(logical, analysis, enabled=optimize)
    plan, explanation = _lower(
        logical,
        analysis,
        canonical,
        call_options,
        call_pools,
        pool_udfs,
        pool_options,
    )
    _verify_runtime_plan(logical, analysis, plan)
    return CompiledProgram(logical, analysis, plan, explanation)


__all__ = ["compile_logical"]
