"""V3.6 的固定、Ray-free 静态编译流水线。

这不是可插拔 PassManager。编排顺序是唯一合同：
verify logical -> analyze -> optional canonicalize -> lower -> verify plan。
关闭 canonicalization 时仍经过同一 analysis、lowering 和最终 verifier。
"""

from __future__ import annotations

from typing import Mapping

from ..model import CallRef
from .analysis import analyze as _analyze
from .logical import LogicalProgram
from .lowering import _canonicalize, _lower
from .plan import CompiledProgram
from .verify import verify_logical as _verify_logical
from .verify import verify_runtime_plan as _verify_runtime_plan


def compile_logical(
    logical: LogicalProgram,
    call_options: Mapping[CallRef, tuple[tuple[str, object], ...]],
    *,
    optimize: bool = True,
) -> CompiledProgram:
    """Compile one frozen logical graph through the fixed V3.6 pipeline."""

    _verify_logical(logical)
    analysis = _analyze(logical)
    canonical = _canonicalize(logical, analysis, enabled=optimize)
    plan, explanation = _lower(logical, analysis, canonical, call_options)
    _verify_runtime_plan(logical, analysis, plan)
    return CompiledProgram(logical, analysis, plan, explanation)


__all__ = ["compile_logical"]
