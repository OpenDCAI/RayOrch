"""由经过验证的 V3 物理 runtime 支撑的 V3.2 Executor facade。"""

from __future__ import annotations

from typing import Any

from rayorch.experimental.multigrain_v3.executor import (
    Executor as _V3Executor,
    RunResult,
)

from .api import CompiledProgram, Pipeline


class Executor:
    """编译 Port-first graph，然后执行其 V3 物理计划。"""

    def __init__(
        self,
        pipeline: Pipeline | CompiledProgram,
        **options: Any,
    ) -> None:
        self.compiled = (
            pipeline.compile() if isinstance(pipeline, Pipeline) else pipeline
        )
        if not isinstance(self.compiled, CompiledProgram):
            raise TypeError("Executor requires Pipeline or CompiledProgram")
        self._physical = _V3Executor(self.compiled.physical, **options)

    def run(self, *sources: Any) -> RunResult:
        return self._physical.run(*sources)


__all__ = ["Executor", "RunResult"]
