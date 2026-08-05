"""基于 Ray actor pools 的 v3.3 执行器与多 Arena driver。

逻辑 Program、Arena 状态机和 Worker ABI 均不依赖 Ray；本模块是唯一持有
actor handle 与 ObjectRef 的适配层。不同 Arena 共享 actor capacity，但一个 RPC
不会静默混合多个 Arena 的 Grain，从而保持与设计文档一致的实验口径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .api import Pipeline
from .execution_support import materialize_tree
from .model import CallRef, GrainRef
from .program import CompiledProgram
from .protocol import (
    InvocationPlan,
    OutputLayout,
    RemoteBlockRef,
    RowBinding,
)
from .runtime import ArenaEngine
from .worker import LocalWorker


class RayBlockStore:
    """只负责 ObjectRef 与块内行之间的物理映射，不解释业务值。"""

    def __init__(self, ray_module: Any) -> None:
        self._ray = ray_module
        self._cache: dict[Any, tuple[Any, ...]] = {}

    def begin_batch(self) -> None:
        """清空上一批 payload cache，限制 actor 常驻内存。"""

        self._cache.clear()

    def put(self, values: tuple[Any, ...]) -> RemoteBlockRef:
        """把一列批输出作为一个粗粒度块写入 object store。"""

        return RemoteBlockRef(self._ray.put(tuple(values)))

    def get(self, binding: RowBinding) -> Any:
        """只在 Worker 或最终结果交付时读取一行 payload。"""

        block = binding.block
        if not isinstance(block, RemoteBlockRef):
            raise TypeError(f"RayBlockStore received non-Ray block: {block!r}")
        payload = block.object_ref
        # 同一粗块往往服务一个 batch 的很多 RowBinding；每批只 ray.get 一次。
        if isinstance(payload, self._ray.ObjectRef):
            if payload not in self._cache:
                self._cache[payload] = self._ray.get(payload)
            payload = self._cache[payload]
        return payload[binding.row]


class _RayWorkerActor:
    """Ray 壳层：持久化一个 LocalWorker，并复用完全相同的 Worker ABI。"""

    def __init__(
        self,
        target: Any,
        init_args: tuple[Any, ...],
        init_kwargs: tuple[tuple[str, Any], ...],
    ) -> None:
        import ray

        self._store = RayBlockStore(ray)
        self._worker = LocalWorker(target, init_args, init_kwargs)

    def ready(self) -> bool:
        """在 UDF 构造完成后响应，用作初始 actor pool 启动屏障。"""

        return True

    def execute(
        self,
        invocations: tuple[InvocationPlan, ...],
        layouts: tuple[OutputLayout, ...],
    ):
        """执行一批、一次返回；actor 不读取 Program 或 Arena。"""

        self._store.begin_batch()
        try:
            return self._worker.execute(invocations, layouts, self._store)
        finally:
            # actor 不跨 RPC 持有输入 blocks；输出 blocks 由返回的 RowBinding
            # 进入 driver/Arena 生命周期管理。
            self._store.begin_batch()


@dataclass(slots=True)
class RayCallMetrics:
    """一个 Call 在所有 Arena 上共享的物理执行统计。"""

    actor_starts: int = 0
    rpcs: int = 0
    grains: int = 0
    retries: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def average_batch(self) -> float:
        """每个 RPC 的平均 Grain 数。"""

        return self.grains / self.rpcs if self.rpcs else 0.0


@dataclass(frozen=True, slots=True)
class RayRunResult:
    """有序输出、多 Arena 审计状态和 Ray 调度指标。"""

    outputs: object
    elapsed_s: float
    calls: dict[CallRef, RayCallMetrics]
    arenas: tuple[ArenaEngine, ...]
    max_active_arenas: int
    released_values: int

    @property
    def rpc_count(self) -> int:
        """全部 Call 的实际 actor RPC 数。"""

        return sum(metrics.rpcs for metrics in self.calls.values())

    @property
    def actor_count(self) -> int:
        """本次 ExecutionPool 实际创建的 actor 数。"""

        return sum(metrics.actor_starts for metrics in self.calls.values())


@dataclass(slots=True)
class _ActorSlot:
    """driver 内部的 actor capacity token。"""

    call: CallRef
    handle: Any
    busy: bool = False


@dataclass(slots=True)
class _ArenaSlot:
    """一个 source microbatch 及其唯一 Arena 状态机。"""

    index: int
    arena: ArenaEngine


@dataclass(frozen=True, slots=True)
class _Dispatch:
    """pending ObjectRef 对应的 generation-fenced dispatch lease。"""

    arena_index: int
    actor: _ActorSlot
    grains: tuple[GrainRef, ...]
    max_retries: int


class RayExecutor:
    """Call-only actor pools 与多 Arena overlap 的参考执行器。"""

    _INTERNAL_OPTIONS = {
        "batch_size",
        "batch_scope",
        "replicas",
        "max_retries",
    }

    def __init__(
        self,
        pipeline: Pipeline | CompiledProgram,
        *,
        address: str | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """编译 Pipeline、连接 Ray，并按 Call 创建持久 actor pools。"""

        import ray

        self.ray = ray
        self.compiled = (
            pipeline if isinstance(pipeline, CompiledProgram) else pipeline.compile()
        )
        self.program = self.compiled.program
        if not ray.is_initialized():
            init_kwargs = dict(ray_init_kwargs or {})
            if address is not None:
                init_kwargs["address"] = address
            ray.init(**init_kwargs)

        self.store = RayBlockStore(ray)
        self.metrics = {call: RayCallMetrics() for call in self.program.calls}
        self._actors: dict[CallRef, list[_ActorSlot]] = {}
        for call in self.program.calls:
            self._actors[call] = self._create_pool(call)
        # actor handle 创建是异步的；统一 ready 屏障确保 run() 的计时
        # 不混入 UDF/模型初始化，同时保持所有 pool 并行启动。
        startup_refs = [
            actor.handle.ready.remote()
            for actors in self._actors.values()
            for actor in actors
        ]
        ray.get(startup_refs)

    def run(
        self,
        *source_columns: Iterable[Any],
        arena_size: int | None = None,
        max_in_flight: int = 1,
    ) -> RayRunResult:
        """执行行对齐 sources，并让至多 ``max_in_flight`` 个 Arena 重叠。"""

        columns = self._normalize_sources(source_columns)
        row_count = len(columns[0])
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if arena_size is None:
            arena_size = max(1, row_count)
        if arena_size <= 0:
            raise ValueError("arena_size must be positive")

        slices = [
            tuple(column[start : start + arena_size] for column in columns)
            for start in range(0, row_count, arena_size)
        ]
        if not slices:
            slices = [tuple(() for _ in columns)]

        # actor pool 跨 run 持久化，但调度指标严格按 run 隔离。已有 actor
        # 计为本次使用一次；运行中 replacement 会由 _create_actor 继续累加。
        self.store.begin_batch()
        self.metrics = {
            call: RayCallMetrics(actor_starts=len(self._actors[call]))
            for call in self.program.calls
        }
        all_arenas: list[ArenaEngine | None] = [None] * len(slices)
        active: dict[int, _ArenaSlot] = {}
        completed: dict[int, object] = {}
        pending: dict[Any, _Dispatch] = {}
        next_arena = 0
        high_watermark = 0
        released_values = 0
        started = time.perf_counter()

        while len(completed) < len(slices):
            while next_arena < len(slices) and len(active) < max_in_flight:
                arena = self._admit_arena(slices[next_arena])
                active[next_arena] = _ArenaSlot(next_arena, arena)
                all_arenas[next_arena] = arena
                next_arena += 1
                high_watermark = max(high_watermark, len(active))

            self._dispatch_ready(active, pending)

            # 只有没有 pending lease 的 Arena 才能离开 active 集合。
            pending_arenas = {lease.arena_index for lease in pending.values()}
            for index, slot in tuple(active.items()):
                if index not in pending_arenas and slot.arena.is_complete():
                    completed[index] = materialize_tree(
                        self.program,
                        slot.arena,
                        self.store,
                    )
                    # 最终业务值已经复制到 driver 输出；清空 ValueTable 会释放
                    # page-image 等 ObjectRef，但 Item/Shape/Grain 仍可完整审计。
                    released_values += slot.arena.release_values()
                    del active[index]

            if len(completed) == len(slices):
                break
            if not pending and next_arena < len(slices):
                # 当前 active Arena 已完成，但仍有尚未 admission 的 source slice；
                # 下一 turn 会填充空出的 in-flight credit，这不是 deadlock。
                continue
            if not pending:
                summaries = ", ".join(
                    f"arena[{index}] {slot.arena.progress_summary()}"
                    for index, slot in sorted(active.items())
                )
                raise RuntimeError(f"v3.3 Ray runtime deadlocked: {summaries}")

            ready, _ = self.ray.wait(list(pending), num_returns=1)
            result_ref = ready[0]
            lease = pending.pop(result_ref)
            arena = active[lease.arena_index].arena
            try:
                reports = self.ray.get(result_ref)
            except Exception as error:  # Ray 对用户异常和 actor 异常统一在 get 抛出
                self._handle_failure(arena, lease, error)
            else:
                for report in reports:
                    arena.commit_report(report)
            finally:
                lease.actor.busy = False

        arenas = tuple(arena for arena in all_arenas if arena is not None)
        return RayRunResult(
            self._merge_outputs([completed[index] for index in range(len(slices))]),
            time.perf_counter() - started,
            self.metrics,
            arenas,
            high_watermark,
            released_values,
        )

    def close(self) -> None:
        """显式终止本执行器创建的 actors，但不关闭共享 Ray 集群。"""

        for actors in self._actors.values():
            for actor in actors:
                self.ray.kill(actor.handle, no_restart=True)
        self._actors.clear()

    def __enter__(self):
        """支持用 context manager 约束 ExecutionPool 生命周期。"""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """离开 context 时释放 actors。"""

        self.close()

    def _normalize_sources(
        self,
        source_columns: tuple[Iterable[Any], ...],
    ) -> tuple[tuple[Any, ...], ...]:
        """冻结 sources，并校验列数和行对齐。"""

        if len(source_columns) != len(self.program.source_ports):
            raise ValueError("source column count does not match Pipeline.forward")
        columns = tuple(tuple(column) for column in source_columns)
        if len({len(column) for column in columns}) != 1:
            raise ValueError("source columns must be row-aligned")
        return columns

    def _admit_arena(
        self,
        columns: tuple[tuple[Any, ...], ...],
    ) -> ArenaEngine:
        """把一个 source slice 原子接纳为独立 Arena。"""

        arena = ArenaEngine(self.program)
        bindings = {}
        for port, values in zip(self.program.source_ports, columns):
            block = self.store.put(values)
            bindings[port] = tuple(
                RowBinding(block, row) for row in range(len(values))
            )
        arena.admit_sources(bindings)
        arena.close_admission()
        return arena

    def _dispatch_ready(
        self,
        active: dict[int, _ArenaSlot],
        pending: dict[Any, _Dispatch],
    ) -> None:
        """把 READY Grain 分配给空闲 actor；每个 batch 严格属于一个 Arena。"""

        for call, actors in self._actors.items():
            for actor in actors:
                if actor.busy:
                    continue
                candidate = next(
                    (
                        slot
                        for _, slot in sorted(active.items())
                        if call in slot.arena.ready_calls()
                    ),
                    None,
                )
                if candidate is None:
                    break
                options = self._pool_options(call)
                # 参考调度器采用即时、work-conserving 聚批：只要 actor
                # 空闲就发送当前可见 Grain，不伪装支持定时等待窗口。
                grains = candidate.arena.reserve_batch(
                    call,
                    max_size=int(options.get("batch_size", 1)),
                    parent_bound=options.get("batch_scope", "elastic")
                    == "parent_bound",
                )
                invocations = tuple(
                    candidate.arena.invocation_plan(grain) for grain in grains
                )
                layouts = self.compiled.execution.output_layouts_by_call[call]
                result_ref = actor.handle.execute.remote(invocations, layouts)
                actor.busy = True
                pending[result_ref] = _Dispatch(
                    candidate.index,
                    actor,
                    grains,
                    int(options.get("max_retries", 1)),
                )
                metrics = self.metrics[call]
                metrics.rpcs += 1
                metrics.grains += len(grains)
                metrics.batch_sizes.append(len(grains))

    def _handle_failure(
        self,
        arena: ArenaEngine,
        lease: _Dispatch,
        error: Exception,
    ) -> None:
        """局部 retry 或封闭失败 Grain；不回滚无关 branch/Arena。"""

        metrics = self.metrics[lease.actor.call]
        for grain in lease.grains:
            if arena.infra_failures(grain) < lease.max_retries:
                arena.retry(grain)
                metrics.retries += 1
            else:
                arena.commit_failure(grain, repr(error))

        # actor crash 后原 handle 不可复用；普通 UDF 异常则保留持久实例。
        if isinstance(error, self.ray.exceptions.RayActorError):
            replacement = self._create_actor(lease.actor.call)
            lease.actor.handle = replacement

    def _create_pool(self, call: CallRef) -> list[_ActorSlot]:
        """只为 Call 创建 pool；Port 结构关系没有对应入口。"""

        replicas = int(self._pool_options(call).get("replicas", 1))
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        return [_ActorSlot(call, self._create_actor(call)) for _ in range(replicas)]

    def _create_actor(self, call: CallRef) -> Any:
        """创建或替换一个持久 actor，并记录 startup。"""

        spec = self.program.call(call)
        options = self._pool_options(call)
        actor_options = {
            key: value
            for key, value in options.items()
            if key not in self._INTERNAL_OPTIONS
        }
        actor_class = self.ray.remote(_RayWorkerActor).options(**actor_options)
        handle = actor_class.remote(
            spec.kernel.target,
            spec.kernel.init_args,
            spec.kernel.init_kwargs,
        )
        self.metrics[call].actor_starts += 1
        return handle

    def _pool_options(self, call: CallRef) -> dict[str, Any]:
        """读取一个 Call 的冻结物理配置。"""

        pool = self.compiled.execution.pools[
            self.compiled.execution.call_to_pool[call]
        ]
        return dict(pool.options)

    @classmethod
    def _merge_outputs(cls, outputs: list[object]) -> object:
        """按 Arena admission 顺序拼接同构输出树。"""

        first = outputs[0]
        if isinstance(first, list):
            return [item for output in outputs for item in output]
        if isinstance(first, tuple):
            return tuple(
                cls._merge_outputs([output[index] for output in outputs])
                for index in range(len(first))
            )
        raise RuntimeError("invalid materialized output tree")


__all__ = [
    "RayBlockStore",
    "RayCallMetrics",
    "RayExecutor",
    "RayRunResult",
    "RemoteBlockRef",
]
