"""Runtime-aware DAG execution."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Sequence

import ray

from ..dag import CompiledGraph, DagPipeline, NodeSpec
from ..dag.graph import PipeRef
from .core import (
    LineageNode,
    MicroBatch,
    QuarantineRecord,
    RuntimeNodeSpec,
    RuntimeResult,
    merge_lineage_heads,
)
from .ray_module import RuntimeRayModule

RuntimeInput = (
    MicroBatch
    | Sequence[MicroBatch]
    | Mapping[str, Sequence[Any]]
)


class RuntimeDagExecutor:
    """Run one ``RuntimeRayModule`` DAG over microbatches or column data."""

    def __init__(
        self,
        pipeline: DagPipeline,
        *,
        batch_size: int | None = None,
        max_batches_inflight: int = 1,
        dataset: str = "source",
    ) -> None:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pipeline._compiled is None:
            pipeline.compile()
        assert pipeline._compiled is not None
        self.graph = pipeline._compiled
        self._owned_modules = self._start_modules()
        self.batch_size = batch_size
        self.max_batches_inflight = max(1, int(max_batches_inflight))
        self.dataset = dataset

    def _start_modules(self) -> List[RuntimeRayModule]:
        modules: List[RuntimeRayModule] = []
        seen: set[int] = set()
        for name in self.graph.topo_order:
            module = self.graph.nodes[name].module
            if not isinstance(module, RuntimeRayModule):
                raise TypeError(
                    "RuntimeDagExecutor only supports RuntimeRayModule nodes "
                    f"(got {type(module).__name__} for '{name}')"
                )
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)

        owned: List[RuntimeRayModule] = []
        try:
            for module in modules:
                if not module.is_started:
                    module.start()
                    owned.append(module)
        except Exception:
            for module in owned:
                module.close()
            raise
        return owned

    def close(self) -> None:
        for module in self._owned_modules:
            module.close()
        self._owned_modules = []

    def __enter__(self) -> "RuntimeDagExecutor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def run(
        self,
        data: RuntimeInput | None = None,
        **columns: Sequence[Any],
    ) -> RuntimeResult | List[RuntimeResult]:
        if columns:
            if data is not None:
                raise ValueError("pass either a data mapping or keyword columns, not both")
            data = columns
        if data is None:
            raise ValueError("RuntimeDagExecutor.run() requires input data")

        if isinstance(data, MicroBatch):
            return _RuntimeDagScheduler(self.graph, [data], 1).run()[0]

        if isinstance(data, Mapping):
            batches = _microbatches_from_columns(
                data,
                batch_size=self._require_batch_size(),
                dataset=self.dataset,
            )
            return _RuntimeDagScheduler(
                self.graph,
                batches,
                self.max_batches_inflight,
            ).run()

        return _RuntimeDagScheduler(
            self.graph,
            list(data),
            self.max_batches_inflight,
        ).run()

    def _require_batch_size(self) -> int:
        if self.batch_size is None:
            raise ValueError("batch_size is required when running column data")
        return self.batch_size


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
    paths: Dict[str, LineageNode] = field(default_factory=dict)
    row_path: Dict[str, str] = field(default_factory=dict)

    def add(self, result: RuntimeResult) -> None:
        self.quarantined.extend(result.quarantined)
        self.paths.update(result.paths)
        self.row_path.update(result.row_path)

    def add_paths(self, paths: Mapping[str, LineageNode]) -> None:
        self.paths.update(paths)


class _RuntimeDagScheduler:
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
        batch, join_paths = _node_batch(
            self._graph,
            spec.input_names,
            spec.args,
            spec.kw_args,
            self._ctx[bi],
        )
        self._accum[bi].add_paths(join_paths)
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
        accum = self._accum[bi]
        final_batch, join_paths = _final_batch(self._graph, self._ctx[bi])
        accum.add_paths(join_paths)
        accum.row_path.update(zip(final_batch.row_ids, final_batch.path_ids))
        return RuntimeResult(
            batch=final_batch,
            quarantined=accum.quarantined,
            paths=accum.paths,
            row_path=accum.row_path,
        )


def _microbatches_from_columns(
    columns: Mapping[str, Sequence[Any]],
    *,
    batch_size: int,
    dataset: str,
) -> List[MicroBatch]:
    if not columns:
        raise ValueError("column data must contain at least one column")

    materialized = {name: list(values) for name, values in columns.items()}
    sizes = {len(values) for values in materialized.values()}
    if len(sizes) != 1:
        lengths = {name: len(values) for name, values in materialized.items()}
        raise ValueError(f"column lengths must match: {lengths}")

    total = sizes.pop()
    batches: List[MicroBatch] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        row_ids = [f"{dataset}:{index}" for index in range(start, end)]
        batches.append(
            MicroBatch(
                {name: values[start:end] for name, values in materialized.items()},
                row_ids,
                ["source"] * len(row_ids),
            )
        )
    return batches


def _validate_source(graph: CompiledGraph, batch: MicroBatch) -> None:
    _row_index(batch)
    names = tuple(key.removeprefix("__input__") for key in graph.input_keys)
    missing = [name for name in names if name not in batch.columns]
    if missing:
        raise ValueError(f"source MicroBatch missing columns: {missing}")


def _node_batch(
    graph: CompiledGraph,
    input_names: tuple[str, ...],
    refs: tuple[PipeRef, ...],
    kw_refs: Mapping[str, PipeRef],
    ctx: Dict[str, tuple[MicroBatch, ...]],
) -> tuple[MicroBatch, Dict[str, LineageNode]]:
    ordered_refs = _ordered_input_refs(input_names, refs, kw_refs)
    if len(input_names) != len(ordered_refs):
        raise ValueError(
            f"runtime node expected {len(input_names)} refs, "
            f"got {len(ordered_refs)}"
        )
    fields = [
        (input_name, _output_name(graph, ref), ctx[ref.node][ref.index])
        for input_name, ref in zip(input_names, ordered_refs)
    ]
    return _assemble_aligned_batch(fields)


def _ordered_input_refs(
    input_names: Sequence[str],
    args: Sequence[PipeRef],
    kwargs: Mapping[str, PipeRef],
) -> tuple[PipeRef, ...]:
    """Match traced positional and keyword refs to operator parameter order."""
    positional = iter(args)
    ordered: List[PipeRef] = []
    for name in input_names:
        if name in kwargs:
            ordered.append(kwargs[name])
        else:
            try:
                ordered.append(next(positional))
            except StopIteration as exc:
                raise ValueError(f"runtime input '{name}' has no DAG ref") from exc
    try:
        next(positional)
    except StopIteration:
        return tuple(ordered)
    raise ValueError("runtime node has extra positional DAG refs")


def _final_batch(
    graph: CompiledGraph,
    ctx: Dict[str, tuple[MicroBatch, ...]],
) -> tuple[MicroBatch, Dict[str, LineageNode]]:
    refs = graph.graph_outputs
    fields = [
        (_output_name(graph, ref), _output_name(graph, ref), ctx[ref.node][ref.index])
        for ref in refs
    ]
    return _assemble_aligned_batch(fields)


def _assemble_aligned_batch(
    fields: Sequence[tuple[str, str, MicroBatch]],
) -> tuple[MicroBatch, Dict[str, LineageNode]]:
    """Build one row-aligned batch from required DAG inputs."""
    if not fields:
        raise ValueError("runtime DAG node requires at least one input")

    indexes = [_row_index(batch) for _, _, batch in fields]
    anchor = fields[0][2]
    row_ids = [
        row_id
        for row_id in anchor.row_ids
        if all(row_id in index for index in indexes[1:])
    ]

    columns: Dict[str, List[Any]] = {}
    for (target_name, source_name, batch), index in zip(fields, indexes):
        if source_name not in batch.columns:
            raise KeyError(f"runtime output column '{source_name}' not found")
        columns[target_name] = [
            batch.columns[source_name][index[row_id]]
            for row_id in row_ids
        ]

    path_ids: List[str] = []
    join_paths: Dict[str, LineageNode] = {}
    for row_id in row_ids:
        parents = [
            batch.path_ids[index[row_id]]
            for (_, _, batch), index in zip(fields, indexes)
        ]
        path_id, created = merge_lineage_heads(parents)
        path_ids.append(path_id)
        join_paths.update(created)

    return MicroBatch(columns, row_ids, path_ids), join_paths


def _row_index(batch: MicroBatch) -> Dict[str, int]:
    """Validate row-aligned metadata and index one batch by row identity."""
    size = len(batch.row_ids)
    if len(batch.path_ids) != size:
        raise ValueError("MicroBatch row_ids and path_ids must have equal length")
    for name, values in batch.columns.items():
        if len(values) != size:
            raise ValueError(
                f"MicroBatch column '{name}' length {len(values)} != {size}"
            )

    index = {row_id: position for position, row_id in enumerate(batch.row_ids)}
    if len(index) != size:
        raise ValueError("MicroBatch row_ids must be unique")
    return index


def _output_name(graph: CompiledGraph, ref: PipeRef) -> str:
    if ref.node in graph.input_keys:
        return ref.node.removeprefix("__input__")
    return graph.nodes[ref.node].output_names[ref.index]
