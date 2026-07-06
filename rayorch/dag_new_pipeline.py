"""Declarative DAG pipeline with pluggable execution strategies.

Subclass :class:`Pipeline`, attach ``RayModule`` attributes, and
implement ``forward()`` using symbolic ``PipeRef`` wiring.
``compile()`` traces the graph; ``__call__()`` runs batches serially.
Use :class:`DagExecutor` for ``ray.wait``-driven overlapped scheduling.

Example::

    class MyPipe(Pipeline):
        def __init__(self):
            self.a = RayModule(AOp, replicas=2, ...).pre_init()
            self.b = RayModule(BOp, replicas=1).pre_init()
            super().__init__()

        def forward(self, x: PipeRef) -> PipeRef:
            return self.b(self.a(x))

    pipe = MyPipe()
    results = pipe(batches)                              # serial
    results = DagExecutor(max_batches_inflight=4).run(pipe, batches)  # overlapped
    results = pipe.run(batches, executor=DagExecutor())   # convenience
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import inspect
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import ray

from .dispatch_mode import Dispatch
from .container_ops import is_sliceable as _is_sliceable
from .ray_module import RayModule, SOURCE_EXHAUSTED, _SourceExhausted


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipeRef:
    """Symbolic reference to one output slot of a DAG node."""
    node: str
    index: int = 0


@dataclass(frozen=True)
class NodeSpec:
    """Immutable definition of a single compute node in the DAG."""
    name: str
    module: RayModule
    args: Tuple[PipeRef, ...]
    kw_args: Dict[str, PipeRef]
    max_inflight: int = 1
    num_outputs: int = 1


@dataclass
class CompiledGraph:
    """Immutable DAG topology produced by :meth:`Pipeline.compile`."""
    nodes: Dict[str, NodeSpec]
    topo_order: Tuple[str, ...]
    deps: Dict[str, Tuple[str, ...]]
    consumers: Dict[str, Tuple[str, ...]]
    graph_outputs: Tuple[PipeRef, ...]
    input_keys: Tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════
# Graph tracing
# ═══════════════════════════════════════════════════════════════════════════

def _expect_ref(v: Any) -> PipeRef:
    """Validate that *v* is a PipeRef (used during forward tracing)."""
    if isinstance(v, PipeRef):
        return v
    raise TypeError(
        "forward() arguments must be PipeRef values from upstream stages or inputs"
    )


def _validate_ref(
    ref: PipeRef,
    context: str,
    input_keys: set,
    nodes: Dict[str, NodeSpec],
) -> None:
    """Check that *ref* points to a valid node and slot."""
    if ref.index < 0:
        raise ValueError(f"'{context}': negative slot index in {ref}")
    if ref.node in input_keys:
        return
    if ref.node not in nodes:
        raise ValueError(f"'{context}': references unknown node '{ref.node}'")
    if ref.index >= nodes[ref.node].num_outputs:
        raise ValueError(
            f"'{context}': reads slot {ref.index} from '{ref.node}', "
            f"which only has {nodes[ref.node].num_outputs} outputs"
        )


class _GraphTracer:
    """Records node declarations during ``forward()`` and compiles a :class:`CompiledGraph`."""

    def __init__(self) -> None:
        self._tape: List[NodeSpec] = []
        self._name_count: Dict[str, int] = {}

    def _unique_name(self, base: str) -> str:
        n = self._name_count.get(base, 0)
        self._name_count[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    def add_node(
        self,
        base_name: str,
        module: RayModule,
        args: Tuple[PipeRef, ...],
        kw_args: Dict[str, PipeRef],
    ) -> str:
        name = self._unique_name(base_name)
        self._tape.append(NodeSpec(
            name=name,
            module=module,
            args=args,
            kw_args=dict(kw_args),
            max_inflight=module.max_inflight,
            num_outputs=module.num_outputs,
        ))
        return name

    @staticmethod
    def make_refs(node_name: str, num_outputs: int) -> PipeRef | Tuple[PipeRef, ...]:
        if num_outputs <= 1:
            return PipeRef(node_name)
        return tuple(PipeRef(node_name, i) for i in range(num_outputs))

    def build(
        self,
        outputs: PipeRef | Tuple[PipeRef, ...],
        input_keys: Tuple[str, ...],
    ) -> CompiledGraph:
        graph_outputs = (outputs,) if isinstance(outputs, PipeRef) else tuple(outputs)
        input_key_set = set(input_keys)
        by_name = {n.name: n for n in self._tape}
        topo_order = tuple(n.name for n in self._tape)

        deps: Dict[str, Tuple[str, ...]] = {}
        consumers: Dict[str, List[str]] = {n: [] for n in topo_order}

        for node in self._tape:
            all_refs = list(node.args) + list(node.kw_args.values())
            seen: set[str] = set()
            dep_list: List[str] = []
            for ref in all_refs:
                _validate_ref(ref, node.name, input_key_set, by_name)
                if ref.node in input_key_set or ref.node in seen:
                    continue
                seen.add(ref.node)
                dep_list.append(ref.node)
                consumers[ref.node].append(node.name)
            deps[node.name] = tuple(dep_list)

        for ref in graph_outputs:
            _validate_ref(ref, "<graph_output>", input_key_set, by_name)

        return CompiledGraph(
            nodes=by_name,
            topo_order=topo_order,
            deps=deps,
            consumers={k: tuple(v) for k, v in consumers.items()},
            graph_outputs=graph_outputs,
            input_keys=input_keys,
        )


class _TraceProxy:
    """Stands in for a ``RayModule`` during ``forward()`` tracing."""

    def __init__(self, tracer: _GraphTracer, attr_name: str, module: RayModule):
        self._tracer = tracer
        self._attr_name = attr_name
        self._module = module

    def __call__(self, *args: Any, **kwargs: Any) -> PipeRef | Tuple[PipeRef, ...]:
        pipe_args = tuple(_expect_ref(v) for v in args)
        pipe_kw = {k: _expect_ref(v) for k, v in kwargs.items()}
        name = self._tracer.add_node(
            self._attr_name, self._module, pipe_args, pipe_kw,
        )
        return self._tracer.make_refs(name, self._module.num_outputs)


# ═══════════════════════════════════════════════════════════════════════════
# Executors
# ═══════════════════════════════════════════════════════════════════════════

class Executor(ABC):
    """Strategy for running a compiled :class:`Pipeline`."""

    @abstractmethod
    def execute(
        self,
        graph: CompiledGraph,
        columns: Dict[str, Sequence[Any]],
    ) -> List[Any]:
        """Run all batches through *graph* and return results in order."""

    def run(
        self,
        pipeline: Pipeline,
        *inputs: Sequence[Any],
        **named_inputs: Sequence[Any],
    ) -> List[Any]:
        """Compile *pipeline* (if needed), resolve inputs, and execute."""
        return pipeline.run(*inputs, executor=self, **named_inputs)


# ── Shared helpers ────────────────────────────────────────────────────────

def _validate_output(spec: NodeSpec, value: Any) -> None:
    """Check that a node's return value matches its declared ``num_outputs``."""
    if spec.num_outputs == 1:
        if not _is_sliceable(value):
            raise TypeError(
                f"node '{spec.name}' must return a sliceable batch "
                f"(list/tuple/DataFrame/Table/ndarray, got {type(value).__name__})"
            )
        return
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"node '{spec.name}' must return tuple/list of "
            f"{spec.num_outputs} lists (got {type(value).__name__})"
        )
    if len(value) != spec.num_outputs:
        raise ValueError(
            f"node '{spec.name}' declared num_outputs={spec.num_outputs}, "
            f"but returned {len(value)} outputs"
        )
    for i, branch in enumerate(value):
        if not _is_sliceable(branch):
            raise TypeError(
                f"node '{spec.name}' output[{i}] must be a sliceable batch "
                f"(got {type(branch).__name__})"
            )


def _validate_columns(
    graph: CompiledGraph,
    columns: Dict[str, Sequence[Any]],
) -> int:
    """Validate input columns and return batch count."""
    expected = set(graph.input_keys)
    if set(columns.keys()) != expected:
        raise ValueError(
            f"input keys mismatch: expected {graph.input_keys}, "
            f"got {tuple(columns.keys())}"
        )
    sizes = {k: len(v) for k, v in columns.items()}
    if not sizes:
        raise ValueError("input columns cannot be empty")
    unique = set(sizes.values())
    if len(unique) != 1:
        raise ValueError(f"all input columns must have same length, got {sizes}")
    return next(iter(unique))


def _materialize_outputs(
    ctx: Dict[str, tuple],
    graph_outputs: Tuple[PipeRef, ...],
) -> Any:
    """Extract final result from context for one batch."""
    if len(graph_outputs) == 1:
        return ctx[graph_outputs[0].node][graph_outputs[0].index]
    return tuple(ctx[ref.node][ref.index] for ref in graph_outputs)


# ── Sequential executor ──────────────────────────────────────────────────

class SequentialExecutor(Executor):
    """Process batches one-by-one, nodes in topological order.

    No concurrency.  Uses synchronous ``RayModule.__call__`` per node.
    Ideal for debugging, profiling individual ops, and correctness validation.
    """

    def execute(
        self,
        graph: CompiledGraph,
        columns: Dict[str, Sequence[Any]],
    ) -> List[Any]:
        n_batches = _validate_columns(graph, columns)
        results: List[Any] = []

        for bi in range(n_batches):
            ctx: Dict[str, tuple] = {
                key: (columns[key][bi],) for key in graph.input_keys
            }
            for name in graph.topo_order:
                spec = graph.nodes[name]
                args = tuple(ctx[ref.node][ref.index] for ref in spec.args)
                kw = {k: ctx[ref.node][ref.index] for k, ref in spec.kw_args.items()}
                value = spec.module(*args, **kw)
                _validate_output(spec, value)
                ctx[name] = value if spec.num_outputs > 1 else (value,)

            results.append(_materialize_outputs(ctx, graph.graph_outputs))

        return results


# ── DAG executor (ray.wait overlapped scheduler) ─────────────────────────

@dataclass
class _PendingResult:
    refs: List[ray.ObjectRef]
    collect_fn: Optional[Any]


@dataclass
class _NodeStatus:
    phase: str = "waiting"
    pending: Optional[_PendingResult] = None


@dataclass
class _InflightCall:
    batch_idx: int
    node_name: str
    refs: List[ray.ObjectRef]


class _Scheduler:
    """``ray.wait``-driven event loop for :class:`DagExecutor`.

    **流式驱动(不预知 batch 总数)**：input_columns 的每个 value 只需是 *iterable*（list 或任意迭代器/
    生成器）。调度器按 ``max_batches_inflight`` 边拉边 admit——从每个 input_key 的迭代器**同步取一批**
    （所有 key 都还有下一个才 admit 新 batch；任一耗尽即源枯竭）。终止条件是「源枯竭 且 in-flight 清空」，
    **不依赖 len(columns)**。这让 hydp-dataflow 的「有状态流式 reader」(每次 run 吐一 batch、迭代器枯竭即停)
    天然可跑，TB 级/未知长度源不 OOM、不预扫。``_ctx``/``_status`` 按 batch 懒建、完成即释放。

    向后兼容：value 传 list 时 ``iter(list)`` 即可，老行为不变（同步同长、结果按 admit 序返回）。
    """

    def __init__(
        self,
        graph: CompiledGraph,
        input_columns: Dict[str, "Iterable[Any]"],
        max_batches_inflight: int,
    ) -> None:
        self._graph = graph
        self._max_inflight = max(1, int(max_batches_inflight))

        # input_keys 校验(只查键集一致，不再要求等长/已知长度）
        expected = set(graph.input_keys)
        if set(input_columns.keys()) != expected:
            raise ValueError(
                f"input keys mismatch: expected {graph.input_keys}, "
                f"got {tuple(input_columns.keys())}"
            )
        # input_keys 允许为空:纯「自驱动 source」图(每个 root 是有状态流式 reader,自产 batch、
        # 迭代器枯竭时其 run 抛 StopIteration → RunnerActor 转成 SOURCE_EXHAUSTED 哨兵)。此时无 driver
        # 侧输入,batch 由 root source 乐观 admit、靠哨兵终止(见 _on_complete)。约束:空 input_keys 的图
        # **必须**至少有一个终会枯竭的 source root,否则 admit 循环不会停(无外部输入 + 无枯竭信号 = 无限图)。
        # 每个 input_key 一个迭代器（list/生成器/任意 iterable 统一 iter()）
        self._iters: Dict[str, "Iterator[Any]"] = {
            k: iter(v) for k, v in input_columns.items()
        }
        self._source_exhausted = False
        self._void: set[int] = set()                 # 已 admit 但因 source 枯竭而作废、不产出的 batch

        # ── 自驱动图(空 input_keys):source root 是有状态 reader,枯竭信号异步(reader run 回哨兵)──
        # 与 driver 侧 input_keys 不同:那边 admit 前同步 next()、当场知枯竭,不会过度 admit。自驱动这边
        # 枯竭要等 actor 跑完才知,若乐观 admit 满 max_inflight 会在哨兵回来前级联多 spawn 出界的 tick。
        # 故门控:**同一时刻至多 1 批「source 未跑完」的 tick**(1-ahead)。单 actor reader 的 tick 本就在
        # actor 上串行,提前 admit 多个 source tick 不加速读;下游 pipeline overlap 只需「reader[N] 完成即
        # admit N+1、同时 N 的下游在跑」,1-ahead 不损 overlap。代价仅 1 个探测 tick(≈driver 侧 next() 撞
        # StopIteration 那一下)。非自驱动图 self._self_driven=False,门控不触发,老行为逐字节不变。
        # _src_left[bi]=该 batch 尚未完成的 source 节点数;keys 即「source 未跑完」的 batch(门控依据)。
        self._self_driven = not graph.input_keys
        self._source_nodes = tuple(n for n in graph.topo_order if not graph.deps[n])
        self._src_left: Dict[int, int] = {}

        # 按 batch 懒建(dict[bi] 而非 list[N])——不预知总数、完成即可释放
        self._ctx: Dict[int, Dict[str, tuple]] = {}
        self._status: Dict[int, Dict[str, _NodeStatus]] = {}
        self._results: Dict[int, Any] = {}

        self._ready_q: Dict[str, Deque[int]] = {n: deque() for n in graph.topo_order}
        self._node_inflight: Dict[str, int] = {n: 0 for n in graph.topo_order}

        self._live: set[int] = set()
        self._next_batch = 0
        self._batch_inflight: Dict[int, int] = {}    # per-batch 未完成调用数(作废 batch 收尾用)

        self._ref_owner: Dict[ray.ObjectRef, _InflightCall] = {}
        self._outstanding: set[ray.ObjectRef] = set()

        self._output_nodes: frozenset[str] = frozenset(
            ref.node for ref in graph.graph_outputs
        )
        self._input_key_set: frozenset[str] = frozenset(graph.input_keys)

    def run(self) -> List[Any]:
        self._admit_batches()
        self._dispatch()
        while self._outstanding:
            for call in self._drain_completed():
                self._on_complete(call)
            self._admit_batches()
            self._dispatch()
        # 结果按 admit 序(batch idx 升序)还原成 list
        return [self._results[i] for i in sorted(self._results)]

    def _next_input_row(self) -> "Dict[str, Any] | None":
        """从每个 input_key 的迭代器同步取一批;任一耗尽 → 源枯竭,返回 None。"""
        if self._source_exhausted:
            return None
        row: Dict[str, Any] = {}
        for key, it in self._iters.items():
            try:
                row[key] = next(it)
            except StopIteration:
                self._source_exhausted = True
                return None
        return row

    def _drain_completed(self) -> List[_InflightCall]:
        ready, _ = ray.wait(list(self._outstanding), num_returns=1)
        self._outstanding.discard(ready[0])
        if self._outstanding:
            more, _ = ray.wait(
                list(self._outstanding),
                num_returns=len(self._outstanding),
                timeout=0,
            )
            for r in more:
                self._outstanding.discard(r)
            ready.extend(more)

        calls: List[_InflightCall] = []
        for ref in ready:
            call = self._ref_owner.pop(ref, None)
            if call is None:
                continue
            if any(r in self._ref_owner for r in call.refs):
                continue
            calls.append(call)
        return calls

    def _admit_batches(self) -> None:
        # 迭代器驱动:边拉边 admit,拿不到(源枯竭)就停;不依赖 len(columns)。
        while len(self._live) < self._max_inflight:
            # 自驱动图 1-ahead 门控:已有一批 source 未跑完(枯竭未知)→ 先别再 admit,避免哨兵回来
            # 前级联过度 spawn。等那批 source 完成(_on_complete 里清 _src_left)再放行下一批。
            if self._self_driven and self._src_left:
                break
            row = self._next_input_row()
            if row is None:
                break
            bi = self._next_batch
            self._next_batch += 1
            self._live.add(bi)
            self._ctx[bi] = {key: (row[key],) for key in self._graph.input_keys}
            self._status[bi] = {name: _NodeStatus() for name in self._graph.topo_order}
            if self._self_driven:
                self._src_left[bi] = len(self._source_nodes)
            for name in self._graph.topo_order:
                if not self._graph.deps[name]:
                    self._mark_ready(bi, name)

    def _mark_ready(self, bi: int, name: str) -> None:
        st = self._status[bi][name]
        if st.phase != "waiting":
            return
        st.phase = "ready"
        self._ready_q[name].append(bi)

    def _dispatch(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            for name in self._graph.topo_order:
                q = self._ready_q[name]
                cap = self._graph.nodes[name].max_inflight
                while q and self._node_inflight[name] < cap:
                    bi = q.popleft()
                    # batch 可能在入队后被作废(某 source 回哨兵)——跳过:不提交作废 batch 的下游,
                    # 且其 ctx/status 可能已被 _retire_if_drained 清掉,再 _submit 会 KeyError。
                    if bi in self._void or bi not in self._status:
                        continue
                    if self._status[bi][name].phase != "ready":
                        continue
                    self._submit(bi, self._graph.nodes[name])
                    progressed = True

    def _submit(self, bi: int, spec: NodeSpec) -> None:
        ctx = self._ctx[bi]
        args = tuple(ctx[ref.node][ref.index] for ref in spec.args)
        kw = {k: ctx[ref.node][ref.index] for k, ref in spec.kw_args.items()}

        for i, v in enumerate(args):
            if not _is_sliceable(v):
                raise TypeError(
                    f"node '{spec.name}' arg[{i}] must be a sliceable batch "
                    f"(list/tuple/DataFrame/Table/ndarray, got {type(v).__name__})"
                )
        for k, v in kw.items():
            if not _is_sliceable(v):
                raise TypeError(
                    f"node '{spec.name}' kwarg '{k}' must be a sliceable batch "
                    f"(list/tuple/DataFrame/Table/ndarray, got {type(v).__name__})"
                )

        future = spec.module.remote(*args, **kw)
        refs = future.completion_refs()

        self._status[bi][spec.name].phase = "running"
        self._status[bi][spec.name].pending = _PendingResult(
            refs=refs, collect_fn=future.collect_fn,
        )
        self._node_inflight[spec.name] += 1
        self._batch_inflight[bi] = self._batch_inflight.get(bi, 0) + 1

        call = _InflightCall(batch_idx=bi, node_name=spec.name, refs=list(refs))
        for ref in refs:
            self._ref_owner[ref] = call
            self._outstanding.add(ref)

    def _on_complete(self, call: _InflightCall) -> None:
        bi, name = call.batch_idx, call.node_name
        spec = self._graph.nodes[name]
        st = self._status[bi][name]
        pending = st.pending
        assert pending is not None

        if len(call.refs) == 1:
            value = ray.get(call.refs[0])
        else:
            raw = ray.get(call.refs)
            value = pending.collect_fn(spec.module, raw) if pending.collect_fn else raw

        # per-batch / per-node in-flight 记账(哨兵与正常路径都要减,否则收尾判据错)。
        # _submit 保证已先 +1,故 bi 必在字典中——缺失是真 bug,让它 KeyError 暴露而非静默兜底。
        self._node_inflight[name] -= 1
        self._batch_inflight[bi] -= 1
        st.phase = "done"
        st.pending = None

        # 自驱动 1-ahead 门控:source 节点完成(不论出数据还是哨兵)即消 _src_left;本 batch 的 source 全
        # 完成 → 从 _src_left 移除,放行 _admit_batches 下一批。放在两个分支之前,确保哨兵路径也解锁。
        if self._self_driven and bi in self._src_left and not self._graph.deps[name]:
            self._src_left[bi] -= 1
            if self._src_left[bi] <= 0:
                self._src_left.pop(bi, None)

        # ── 流式 source 枯竭:该 source 节点的 run 抛 StopIteration，RunnerActor 已转成哨兵 ──
        # 语义 = 「任一 source 枯竭 → 全图停」(与 driver 侧 _next_input_row 任一迭代器 StopIteration
        # 即停完全对称)。做两件事:① 置枯竭 → _admit_batches 不再 admit 新 batch;② 作废本 batch——
        # 它是「乐观 admit」出来的、上游已无数据,不该产出结果、也不该往下游 dispatch。已在途的兄弟
        # 节点调用完成后同样落进 void 分支被清理,batch 收尾靠 _batch_inflight 归零。
        if isinstance(value, _SourceExhausted):
            self._source_exhausted = True
            self._void.add(bi)
            self._retire_if_drained(bi)
            return

        # 本 batch 已被作废(某 source 先枯竭):不再产出/下推,仅做 in-flight 收尾。
        if bi in self._void:
            self._retire_if_drained(bi)
            return

        _validate_output(spec, value)

        self._ctx[bi][name] = value if spec.num_outputs > 1 else (value,)

        for child in self._graph.consumers[name]:
            if self._all_deps_done(bi, child):
                self._mark_ready(bi, child)

        self._release_upstream(bi, name)

        if self._is_batch_complete(bi):
            self._results[bi] = _materialize_outputs(
                self._ctx[bi], self._graph.graph_outputs,
            )
            self._live.discard(bi)
            # 流式:完成的 batch ctx/status 即刻释放(不全程持有 → TB 级不堆内存)
            self._ctx.pop(bi, None)
            self._status.pop(bi, None)
            self._batch_inflight.pop(bi, None)

    def _retire_if_drained(self, bi: int) -> None:
        """作废 batch 的收尾:等它所有在途调用回来(_batch_inflight 归零)再释放状态。

        不能在遇到哨兵的瞬间就 pop——同一 batch 的兄弟 source/节点可能仍在途(fan-out 并行 admit),
        它们的 ObjectRef 还在 _outstanding 里、完成时要能查到自己的 ctx/status。故按未完成调用计数收尾。
        """
        if self._batch_inflight.get(bi, 0) > 0:
            return
        self._live.discard(bi)
        self._ctx.pop(bi, None)
        self._status.pop(bi, None)
        self._batch_inflight.pop(bi, None)
        self._src_left.pop(bi, None)

    def _release_upstream(self, bi: int, name: str) -> None:
        for dep in self._graph.deps[name]:
            if dep in self._output_nodes or dep in self._input_key_set:
                continue
            if all(self._status[bi][c].phase == "done"
                   for c in self._graph.consumers[dep]):
                self._ctx[bi].pop(dep, None)

    def _all_deps_done(self, bi: int, name: str) -> bool:
        return all(
            self._status[bi][d].phase == "done" for d in self._graph.deps[name]
        )

    def _is_batch_complete(self, bi: int) -> bool:
        return all(
            self._status[bi][ref.node].phase == "done"
            for ref in self._graph.graph_outputs
        )


class DagExecutor(Executor):
    """``ray.wait``-driven overlapped executor with per-node inflight caps."""

    def __init__(self, *, max_batches_inflight: int = 4) -> None:
        self.max_batches_inflight = max(1, int(max_batches_inflight))

    def execute(
        self,
        graph: CompiledGraph,
        columns: Dict[str, "Iterable[Any]"],
    ) -> List[Any]:
        return _Scheduler(graph, columns, self.max_batches_inflight).run()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline (user API)
# ═══════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Declarative DAG pipeline base class.

    Subclass, attach ``RayModule`` attributes, implement ``forward()``
    with ``PipeRef`` wiring, then call the pipeline on batched inputs.

    ``pipe(batches)`` runs serial execution (default).
    ``pipe.run(batches, executor=DagExecutor())`` runs overlapped.
    ``DagExecutor().run(pipe, batches)`` is equivalent.
    """

    def __init__(self) -> None:
        self._compiled: Optional[CompiledGraph] = None
        self._param_names: Tuple[str, ...] = ()
        self._input_keys: Tuple[str, ...] = ()

    def forward(self, x: PipeRef) -> PipeRef | Tuple[PipeRef, ...]:
        raise NotImplementedError

    # ── Compile ───────────────────────────────────────────────────────────

    @staticmethod
    def _input_key(param: str) -> str:
        return f"__input__{param}"

    def _introspect_forward(self) -> Tuple[
        List[PipeRef], Dict[str, PipeRef], Tuple[str, ...], Tuple[str, ...],
    ]:
        sig = inspect.signature(self.forward)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("forward() must declare at least one parameter")

        pos_refs: List[PipeRef] = []
        kw_refs: Dict[str, PipeRef] = {}
        names: List[str] = []
        keys: List[str] = []

        for p in params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    "forward() cannot use *args/**kwargs; use explicit parameters"
                )
            key = self._input_key(p.name)
            ref = PipeRef(key)
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos_refs.append(ref)
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                kw_refs[p.name] = ref
            names.append(p.name)
            keys.append(key)

        return pos_refs, kw_refs, tuple(names), tuple(keys)

    def compile(self) -> Pipeline:
        tracer = _GraphTracer()
        originals: Dict[str, RayModule] = {}
        for attr, val in list(self.__dict__.items()):
            if isinstance(val, RayModule):
                originals[attr] = val
                setattr(self, attr, _TraceProxy(tracer, attr, val))
        if not originals:
            raise ValueError("no RayModule attributes found on pipeline")

        pos_refs, kw_refs, param_names, input_keys = self._introspect_forward()
        try:
            out = self.forward(*pos_refs, **kw_refs)
        finally:
            for attr, mod in originals.items():
                setattr(self, attr, mod)

        self._param_names = param_names
        self._input_keys = input_keys
        self._compiled = tracer.build(out, input_keys)
        return self

    # ── Run ───────────────────────────────────────────────────────────────

    def _resolve_inputs(
        self,
        args: Tuple[Sequence[Any], ...],
        kwargs: Mapping[str, Sequence[Any]],
    ) -> Dict[str, Sequence[Any]]:
        if not self._param_names:
            raise RuntimeError("pipeline is not compiled")
        if args and kwargs:
            raise ValueError("pass inputs positionally or by keyword, not both")

        if kwargs:
            if set(kwargs.keys()) != set(self._param_names):
                raise ValueError(
                    f"keyword mismatch: expected {self._param_names}, "
                    f"got {tuple(kwargs.keys())}"
                )
            return {
                self._input_key(name): kwargs[name]
                for name in self._param_names
            }

        if len(args) != len(self._param_names):
            raise ValueError(
                f"expected {len(self._param_names)} positional inputs, "
                f"got {len(args)}"
            )
        return dict(zip(self._input_keys, args))

    def __call__(
        self, *inputs: Sequence[Any], **named_inputs: Sequence[Any],
    ) -> List[Any]:
        return self.run(*inputs, **named_inputs)

    def run(
        self,
        *inputs: Sequence[Any],
        executor: Executor | None = None,
        **named_inputs: Sequence[Any],
    ) -> List[Any]:
        """Run pipeline.  Defaults to serial; pass ``executor=DagExecutor()`` for overlap."""
        if self._compiled is None:
            self.compile()
        columns = self._resolve_inputs(inputs, named_inputs)
        if executor is None:
            executor = SequentialExecutor()
        return executor.execute(self._compiled, columns)


# Backward-compatible alias
DagPipeline = Pipeline


# ═══════════════════════════════════════════════════════════════════════════
# Inline tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    def _cleanup_pipe_modules(pipe: Pipeline) -> None:
        for value in pipe.__dict__.values():
            if isinstance(value, RayModule):
                for actor in getattr(value, "actors", []):
                    try:
                        ray.kill(actor)
                    except Exception:
                        pass

    def shard_sizes(n: int, replicas: int) -> list[int]:
        base = n // replicas
        rem = n % replicas
        return [base + (1 if i < rem else 0) for i in range(replicas)]

    class ProbeShardSizeOp:
        def run(self, x: list[int]) -> list[int]:
            return [len(x)]

    class ShardProbePipe(Pipeline):
        def __init__(self, replicas: int):
            self.probe = RayModule(
                ProbeShardSizeOp,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef) -> PipeRef:
            return self.probe(x)

    class MultiplyBy2Op:
        def run(self, x: list[int]) -> list[int]:
            return [2 * v for v in x]

    class ShardPostProcessPipe(Pipeline):
        def __init__(self, replicas: int):
            self.mul2 = RayModule(
                MultiplyBy2Op,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            super().__init__()

        def forward(self, x: PipeRef) -> PipeRef:
            return self.mul2(x)

    class SplitParityOp:
        def run(self, x: list[int]) -> tuple[list[int], list[int]]:
            evens = [v for v in x if v % 2 == 0]
            odds = [v for v in x if v % 2 == 1]
            return evens, odds

    class ScaleEvenOp:
        def run(self, x: list[int]) -> list[int]:
            return [2 * v for v in x]

    class ScaleOddOp:
        def run(self, x: list[int]) -> list[int]:
            return [3 * v for v in x]

    class MultiReplicaSplitPipe(Pipeline):
        def __init__(self, replicas: int):
            self.split = RayModule(
                SplitParityOp,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            self.left = RayModule(ScaleEvenOp, replicas=1, max_inflight=2).pre_init()
            self.right = RayModule(ScaleOddOp, replicas=1, max_inflight=2).pre_init()
            super().__init__()

        def forward(self, x: PipeRef) -> Tuple[PipeRef, PipeRef]:
            even_in, odd_in = self.split(x)
            return self.left(even_in), self.right(odd_in)

    ray.init(ignore_reinit_error=True, num_cpus=12)
    replicas = 3
    xs = [list(range(10)), list(range(7)), [11, 12, 13, 14, 15]]
    dag = DagExecutor(max_batches_inflight=4)

    # 1) Serial vs DAG: shard probe
    probe_pipe = ShardProbePipe(replicas=replicas)
    expect_shards = [shard_sizes(len(batch), replicas) for batch in xs]
    serial_out = probe_pipe(xs)
    dag_out = dag.run(probe_pipe, xs)
    print("[shard]  serial:", serial_out == expect_shards, "dag:", dag_out == expect_shards)
    assert serial_out == dag_out == expect_shards
    _cleanup_pipe_modules(probe_pipe)

    # 2) Serial vs DAG: post-process
    post_pipe = ShardPostProcessPipe(replicas=replicas)
    expect_post = [[2 * v for v in batch] for batch in xs]
    serial_out = post_pipe(xs)
    dag_out = dag.run(post_pipe, xs)
    print("[post]   serial:", serial_out == expect_post, "dag:", dag_out == expect_post)
    assert serial_out == dag_out == expect_post
    _cleanup_pipe_modules(post_pipe)

    # 3) Serial vs DAG: multi-output split
    split_pipe = MultiReplicaSplitPipe(replicas=replicas)
    expect_split = [
        ([2 * v for v in b if v % 2 == 0], [3 * v for v in b if v % 2 == 1])
        for b in xs
    ]
    serial_out = split_pipe(xs)
    dag_out = dag.run(split_pipe, xs)
    print("[split]  serial:", serial_out == expect_split, "dag:", dag_out == expect_split)
    assert serial_out == dag_out == expect_split
    _cleanup_pipe_modules(split_pipe)

    # 4) Convenience: pipe.run(executor=...)
    probe2 = ShardProbePipe(replicas=replicas)
    conv_out = probe2.run(xs, executor=dag)
    print("[conv]   match:", conv_out == expect_shards)
    assert conv_out == expect_shards
    _cleanup_pipe_modules(probe2)

    ray.shutdown()
