"""Runtime-aware DAG execution."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Sequence

import ray

from ..dag import CompiledGraph, DagPipeline, NodeSpec, PipeRef
from .core import MicroBatch, QuarantineRecord, RuntimeNodeSpec, RuntimeResult
from .ray_module import RuntimeRayModule


class RuntimeDagExecutor:
    """Run ``RuntimeRayModule`` DAGs over one or more ``MicroBatch`` objects."""

    def __init__(self, *, max_batches_inflight: int = 1) -> None:
        self.max_batches_inflight = max(1, int(max_batches_inflight))

    def run(
        self,
        pipeline: DagPipeline,
        batches: MicroBatch | Sequence[MicroBatch],
    ) -> RuntimeResult | List[RuntimeResult]:
        if pipeline._compiled is None:
            pipeline.compile()
        assert pipeline._compiled is not None
        return self.execute(pipeline._compiled, batches)

    def execute(
        self,
        graph: CompiledGraph,
        batches: MicroBatch | Sequence[MicroBatch],
    ) -> RuntimeResult | List[RuntimeResult]:
        if isinstance(batches, MicroBatch):
            return _RuntimeScheduler(graph, [batches], 1).run()[0]
        return _RuntimeScheduler(
            graph,
            list(batches),
            self.max_batches_inflight,
        ).run()


@dataclass
class _PendingRuntime:
    refs: List[ray.ObjectRef]
    collect_fn: Any


@dataclass
class _RuntimeStatus:
    phase: str = "waiting"
    pending: _PendingRuntime | None = None


@dataclass
class _RuntimeCall:
    batch_idx: int
    node_name: str
    refs: List[ray.ObjectRef]


@dataclass
class _RuntimeAccum:
    quarantined: List[QuarantineRecord] = field(default_factory=list)
    paths: Dict[str, tuple[str, str]] = field(default_factory=dict)
    row_path: Dict[str, str] = field(default_factory=dict)

    def add(self, result: RuntimeResult) -> None:
        self.quarantined.extend(result.quarantined)
        self.paths.update(result.paths)
        self.row_path.update(result.row_path)


class _RuntimeScheduler:
    """Pipeline-level overlap scheduler for runtime DAGs."""

    def __init__(
        self,
        graph: CompiledGraph,
        batches: List[MicroBatch],
        max_batches_inflight: int,
    ) -> None:
        self._graph = graph
        self._batches = batches
        self._max_inflight = max(1, int(max_batches_inflight))
        self._validate_graph()
        for batch in batches:
            _validate_source(graph, batch)

        self._ctx: List[Dict[str, tuple[MicroBatch, ...]]] = [
            {key: (batch,) for key in graph.input_keys} for batch in batches
        ]
        self._status: List[Dict[str, _RuntimeStatus]] = [
            {name: _RuntimeStatus() for name in graph.topo_order}
            for _ in batches
        ]
        self._accum = [_RuntimeAccum() for _ in batches]
        self._results: List[RuntimeResult | None] = [None] * len(batches)

        self._ready_q: Dict[str, Deque[int]] = {n: deque() for n in graph.topo_order}
        self._node_inflight: Dict[str, int] = {n: 0 for n in graph.topo_order}
        self._live: set[int] = set()
        self._next_batch = 0

        self._ref_owner: Dict[ray.ObjectRef, _RuntimeCall] = {}
        self._outstanding: set[ray.ObjectRef] = set()
        self._output_nodes = frozenset(ref.node for ref in graph.graph_outputs)
        self._input_keys = frozenset(graph.input_keys)

    def run(self) -> List[RuntimeResult]:
        self._admit_batches()
        self._dispatch()
        while self._outstanding:
            for call in self._drain_completed():
                self._on_complete(call)
            self._admit_batches()
            self._dispatch()
        return [result for result in self._results if result is not None]

    def _validate_graph(self) -> None:
        for name, spec in self._graph.nodes.items():
            if not isinstance(spec.module, RuntimeRayModule):
                raise TypeError(
                    "RuntimeDagExecutor only supports RuntimeRayModule nodes "
                    f"(got {type(spec.module).__name__} for '{name}')"
                )

    def _admit_batches(self) -> None:
        while (
            self._next_batch < len(self._batches)
            and len(self._live) < self._max_inflight
        ):
            bi = self._next_batch
            self._next_batch += 1
            self._live.add(bi)
            for name in self._graph.topo_order:
                if not self._graph.deps[name]:
                    self._mark_ready(bi, name)

    def _mark_ready(self, bi: int, name: str) -> None:
        status = self._status[bi][name]
        if status.phase != "waiting":
            return
        status.phase = "ready"
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
                    if self._status[bi][name].phase != "ready":
                        continue
                    self._submit(bi, self._graph.nodes[name])
                    progressed = True

    def _submit(self, bi: int, spec: NodeSpec) -> None:
        batch = _node_batch(spec.input_names, spec.args, self._ctx[bi])
        runtime_spec = RuntimeNodeSpec(
            node=spec.name,
            inputs=tuple(spec.input_names),
            outputs=tuple(spec.output_names),
        )
        future = spec.module.remote(batch, runtime_spec)
        refs = future.completion_refs()

        self._status[bi][spec.name].phase = "running"
        self._status[bi][spec.name].pending = _PendingRuntime(
            refs=refs,
            collect_fn=future.collect_fn,
        )
        self._node_inflight[spec.name] += 1

        call = _RuntimeCall(batch_idx=bi, node_name=spec.name, refs=list(refs))
        for ref in refs:
            self._ref_owner[ref] = call
            self._outstanding.add(ref)

    def _drain_completed(self) -> List[_RuntimeCall]:
        ready, _ = ray.wait(list(self._outstanding), num_returns=1)
        self._outstanding.discard(ready[0])
        if self._outstanding:
            more, _ = ray.wait(
                list(self._outstanding),
                num_returns=len(self._outstanding),
                timeout=0,
            )
            for ref in more:
                self._outstanding.discard(ref)
            ready.extend(more)

        calls: List[_RuntimeCall] = []
        for ref in ready:
            call = self._ref_owner.pop(ref, None)
            if call is None:
                continue
            if any(pending_ref in self._ref_owner for pending_ref in call.refs):
                continue
            calls.append(call)
        return calls

    def _on_complete(self, call: _RuntimeCall) -> None:
        bi, name = call.batch_idx, call.node_name
        spec = self._graph.nodes[name]
        pending = self._status[bi][name].pending
        assert pending is not None

        if len(call.refs) == 1:
            result = ray.get(call.refs[0])
        else:
            raw = ray.get(call.refs)
            result = pending.collect_fn(spec.module, raw) if pending.collect_fn else raw
        if not isinstance(result, RuntimeResult):
            raise TypeError(
                f"runtime node '{name}' must return RuntimeResult, "
                f"got {type(result).__name__}"
            )

        self._ctx[bi][name] = tuple(result.batch for _ in range(spec.num_outputs))
        self._accum[bi].add(result)
        self._status[bi][name].phase = "done"
        self._status[bi][name].pending = None
        self._node_inflight[name] -= 1

        for child in self._graph.consumers[name]:
            if self._all_deps_done(bi, child):
                self._mark_ready(bi, child)

        self._release_upstream(bi, name)
        if self._is_batch_complete(bi):
            self._results[bi] = self._materialize_result(bi)
            self._live.discard(bi)

    def _all_deps_done(self, bi: int, name: str) -> bool:
        return all(
            self._status[bi][dep].phase == "done" for dep in self._graph.deps[name]
        )

    def _release_upstream(self, bi: int, name: str) -> None:
        for dep in self._graph.deps[name]:
            if dep in self._output_nodes or dep in self._input_keys:
                continue
            if all(
                self._status[bi][child].phase == "done"
                for child in self._graph.consumers[dep]
            ):
                self._ctx[bi].pop(dep, None)

    def _is_batch_complete(self, bi: int) -> bool:
        return all(
            self._status[bi][ref.node].phase == "done"
            for ref in self._graph.graph_outputs
        )

    def _materialize_result(self, bi: int) -> RuntimeResult:
        final_batch = _final_batch(self._graph, self._ctx[bi])
        accum = self._accum[bi]
        return RuntimeResult(
            batch=final_batch,
            quarantined=accum.quarantined,
            paths=accum.paths,
            row_path=accum.row_path,
        )


def _validate_source(graph: CompiledGraph, batch: MicroBatch) -> None:
    names = tuple(key.removeprefix("__input__") for key in graph.input_keys)
    missing = [name for name in names if name not in batch.columns]
    if missing:
        raise ValueError(f"source MicroBatch missing columns: {missing}")


def _node_batch(
    input_names: tuple[str, ...],
    refs: tuple[PipeRef, ...],
    ctx: Dict[str, tuple[MicroBatch, ...]],
) -> MicroBatch:
    if len(input_names) != len(refs):
        raise ValueError(
            f"runtime node expected {len(input_names)} refs, got {len(refs)}"
        )
    batches = [ctx[ref.node][ref.index] for ref in refs]
    anchor = batches[0]
    columns: Dict[str, List[Any]] = {}
    for name, source in zip(input_names, batches):
        _validate_alignment(anchor, source)
        if name not in source.columns:
            raise KeyError(f"runtime input column '{name}' not found")
        columns[name] = source.columns[name]
    return MicroBatch(columns, list(anchor.row_ids), list(anchor.path_ids))


def _final_batch(
    graph: CompiledGraph,
    ctx: Dict[str, tuple[MicroBatch, ...]],
) -> MicroBatch:
    refs = graph.graph_outputs
    batches = [ctx[ref.node][ref.index] for ref in refs]
    anchor = batches[0]
    columns: Dict[str, List[Any]] = {}

    for ref, source in zip(refs, batches):
        _validate_alignment(anchor, source)
        name = _output_name(graph, ref)
        columns[name] = source.columns[name]
    return MicroBatch(columns, list(anchor.row_ids), list(anchor.path_ids))


def _validate_alignment(left: MicroBatch, right: MicroBatch) -> None:
    if left.row_ids != right.row_ids:
        raise ValueError("runtime DAG joins require aligned row_ids in MVP")
    if left.path_ids != right.path_ids:
        raise ValueError("runtime DAG joins require aligned path_ids in MVP")


def _output_name(graph: CompiledGraph, ref: PipeRef) -> str:
    if ref.node in graph.input_keys:
        return ref.node.removeprefix("__input__")
    return graph.nodes[ref.node].output_names[ref.index]
