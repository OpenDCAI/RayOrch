"""用于语义测试和 dummy 性能回归的单进程执行器。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .api import Pipeline
from .model import CallRef
from .program import CompiledProgram
from .execution_support import materialize_tree
from .protocol import BlockRef, RowBinding
from .runtime import ArenaEngine
from .worker import LocalWorker


class LocalBlockStore:
    """测试用进程内粗粒度块存储。"""

    def __init__(self) -> None:
        self._next = 0
        self.blocks: dict[BlockRef, tuple[Any, ...]] = {}

    def put(self, values: tuple[Any, ...]) -> BlockRef:
        """把一列值存成不可变粗块并返回块引用。"""

        ref = BlockRef(self._next)
        self._next += 1
        self.blocks[ref] = tuple(values)
        return ref

    def get(self, binding: RowBinding) -> Any:
        """按块引用和行号读取一个业务值。"""

        return self.blocks[binding.block][binding.row]


@dataclass(slots=True)
class CallMetrics:
    """单个 Call 的批处理/RPC 统计。"""

    rpcs: int = 0
    grains: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def average_batch(self) -> float:
        """返回每次 Worker RPC 平均承载的 Grain 数。"""

        return self.grains / self.rpcs if self.rpcs else 0.0


@dataclass(frozen=True, slots=True)
class LocalRunResult:
    """本地执行输出及可审计的 Arena、块和指标。"""

    outputs: object
    elapsed_s: float
    calls: dict[CallRef, CallMetrics]
    arena: ArenaEngine
    store: LocalBlockStore

    @property
    def rpc_count(self) -> int:
        """返回本次运行所有 Call 的 Worker RPC 总数。"""

        return sum(metrics.rpcs for metrics in self.calls.values())


class LocalExecutor:
    """不导入、不初始化 Ray，但执行与 Ray 路径相同的 Worker ABI。"""

    def __init__(self, pipeline: Pipeline | CompiledProgram) -> None:
        self.compiled = pipeline if isinstance(pipeline, CompiledProgram) else pipeline.compile()
        self.program = self.compiled.program
        self.store = LocalBlockStore()
        self.workers = {
            call: LocalWorker(
                spec.kernel.target,
                spec.kernel.init_args,
                spec.kernel.init_kwargs,
            )
            for call, spec in self.program.calls.items()
        }
        self.metrics = {call: CallMetrics() for call in self.program.calls}

    def run(self, *source_columns: Iterable[Any]) -> LocalRunResult:
        """以行对齐 source columns 执行单个 Arena，直到严格完成。"""

        if len(source_columns) != len(self.program.source_ports):
            raise ValueError("source column count does not match Pipeline.forward")
        columns = tuple(tuple(column) for column in source_columns)
        if len({len(column) for column in columns}) != 1:
            raise ValueError("source columns must be row-aligned")

        arena = ArenaEngine(self.program)
        source_bindings = {}
        for port, values in zip(self.program.source_ports, columns):
            block = self.store.put(values)
            source_bindings[port] = tuple(
                RowBinding(block, index) for index in range(len(values))
            )
        arena.admit_sources(source_bindings)
        arena.close_admission()

        started = time.perf_counter()
        while not arena.is_complete():
            ready_calls = arena.ready_calls()
            if not ready_calls:
                raise RuntimeError(self._deadlock_message(arena))
            call = ready_calls[0]
            pool = self.compiled.execution.pools[
                self.compiled.execution.call_to_pool[call]
            ]
            options = dict(pool.options)
            batch_size = int(options.get("batch_size", 1))
            parent_bound = options.get("batch_scope", "elastic") == "parent_bound"
            grains = arena.reserve_batch(
                call,
                max_size=batch_size,
                parent_bound=parent_bound,
            )
            invocations = tuple(
                arena.invocation_plan(grain) for grain in grains
            )
            layouts = self.compiled.execution.output_layouts_by_call[call]
            reports = self.workers[call].execute(
                invocations,
                layouts,
                self.store,
            )
            metrics = self.metrics[call]
            metrics.rpcs += 1
            metrics.grains += len(grains)
            metrics.batch_sizes.append(len(grains))
            for report in reports:
                arena.commit_report(report)

        outputs = materialize_tree(self.program, arena, self.store)
        return LocalRunResult(
            outputs,
            time.perf_counter() - started,
            self.metrics,
            arena,
            self.store,
        )


    @staticmethod
    def _deadlock_message(arena: ArenaEngine) -> str:
        return f"v3.3 local runtime deadlocked: {arena.progress_summary()}"


__all__ = [
    "CallMetrics",
    "LocalBlockStore",
    "LocalExecutor",
    "LocalRunResult",
]
