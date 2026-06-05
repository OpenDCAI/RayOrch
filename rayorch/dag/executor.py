"""Execution strategies for compiled DAG graphs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import ray

from .graph import CompiledGraph, NodeSpec, PipeRef


class Executor(ABC):
    """Strategy for running a compiled ``Pipeline``."""

    @abstractmethod
    def execute(
        self,
        graph: CompiledGraph,
        columns: Dict[str, Sequence[Any]],
    ) -> List[Any]:
        """Run all batches through ``graph`` and return results in order."""

    def run(
        self,
        pipeline: Any,
        *inputs: Sequence[Any],
        **named_inputs: Sequence[Any],
    ) -> List[Any]:
        return pipeline.run(*inputs, executor=self, **named_inputs)


def validate_output(spec: NodeSpec, value: Any) -> None:
    """Check that a node's return value matches its declared ``num_outputs``."""
    if spec.num_outputs == 1:
        if not isinstance(value, list):
            raise TypeError(
                f"node '{spec.name}' must return list (got {type(value).__name__})"
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
        if not isinstance(branch, list):
            raise TypeError(
                f"node '{spec.name}' output[{i}] must be list "
                f"(got {type(branch).__name__})"
            )


def validate_columns(
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


def materialize_outputs(
    ctx: Dict[str, tuple],
    graph_outputs: Tuple[PipeRef, ...],
) -> Any:
    """Extract final result from context for one batch."""
    if len(graph_outputs) == 1:
        return ctx[graph_outputs[0].node][graph_outputs[0].index]
    return tuple(ctx[ref.node][ref.index] for ref in graph_outputs)


class SequentialExecutor(Executor):
    """Process batches one-by-one, nodes in topological order."""

    def execute(
        self,
        graph: CompiledGraph,
        columns: Dict[str, Sequence[Any]],
    ) -> List[Any]:
        n_batches = validate_columns(graph, columns)
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
                validate_output(spec, value)
                ctx[name] = value if spec.num_outputs > 1 else (value,)

            results.append(materialize_outputs(ctx, graph.graph_outputs))

        return results


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
    """``ray.wait``-driven event loop for ``DagExecutor``."""

    def __init__(
        self,
        graph: CompiledGraph,
        input_columns: Dict[str, Sequence[Any]],
        max_batches_inflight: int,
    ) -> None:
        self._graph = graph
        self._max_inflight = max(1, int(max_batches_inflight))
        self._n_batches = validate_columns(graph, input_columns)

        self._ctx: List[Dict[str, tuple]] = [
            {key: (input_columns[key][i],) for key in graph.input_keys}
            for i in range(self._n_batches)
        ]
        self._status: List[Dict[str, _NodeStatus]] = [
            {name: _NodeStatus() for name in graph.topo_order}
            for _ in range(self._n_batches)
        ]
        self._results: List[Any] = [None] * self._n_batches

        self._ready_q: Dict[str, Deque[int]] = {n: deque() for n in graph.topo_order}
        self._node_inflight: Dict[str, int] = {n: 0 for n in graph.topo_order}

        self._live: set[int] = set()
        self._next_batch = 0

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
        return self._results

    def _drain_completed(self) -> List[_InflightCall]:
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

        calls: List[_InflightCall] = []
        for ref in ready:
            call = self._ref_owner.pop(ref, None)
            if call is None:
                continue
            if any(pending_ref in self._ref_owner for pending_ref in call.refs):
                continue
            calls.append(call)
        return calls

    def _admit_batches(self) -> None:
        while (
            self._next_batch < self._n_batches
            and len(self._live) < self._max_inflight
        ):
            bi = self._next_batch
            self._next_batch += 1
            self._live.add(bi)
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
                    if self._status[bi][name].phase != "ready":
                        continue
                    self._submit(bi, self._graph.nodes[name])
                    progressed = True

    def _submit(self, bi: int, spec: NodeSpec) -> None:
        ctx = self._ctx[bi]
        args = tuple(ctx[ref.node][ref.index] for ref in spec.args)
        kw = {k: ctx[ref.node][ref.index] for k, ref in spec.kw_args.items()}

        for i, value in enumerate(args):
            if not isinstance(value, list):
                raise TypeError(
                    f"node '{spec.name}' arg[{i}] must be list "
                    f"(got {type(value).__name__})"
                )
        for key, value in kw.items():
            if not isinstance(value, list):
                raise TypeError(
                    f"node '{spec.name}' kwarg '{key}' must be list "
                    f"(got {type(value).__name__})"
                )

        future = spec.module.remote(*args, **kw)
        refs = future.completion_refs()

        self._status[bi][spec.name].phase = "running"
        self._status[bi][spec.name].pending = _PendingResult(
            refs=refs,
            collect_fn=future.collect_fn,
        )
        self._node_inflight[spec.name] += 1

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

        validate_output(spec, value)

        self._ctx[bi][name] = value if spec.num_outputs > 1 else (value,)
        st.phase = "done"
        st.pending = None
        self._node_inflight[name] -= 1

        for child in self._graph.consumers[name]:
            if self._all_deps_done(bi, child):
                self._mark_ready(bi, child)

        self._release_upstream(bi, name)

        if self._is_batch_complete(bi):
            self._results[bi] = materialize_outputs(
                self._ctx[bi],
                self._graph.graph_outputs,
            )
            self._live.discard(bi)

    def _release_upstream(self, bi: int, name: str) -> None:
        for dep in self._graph.deps[name]:
            if dep in self._output_nodes or dep in self._input_key_set:
                continue
            if all(
                self._status[bi][child].phase == "done"
                for child in self._graph.consumers[dep]
            ):
                self._ctx[bi].pop(dep, None)

    def _all_deps_done(self, bi: int, name: str) -> bool:
        return all(
            self._status[bi][dep].phase == "done" for dep in self._graph.deps[name]
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
        columns: Dict[str, Sequence[Any]],
    ) -> List[Any]:
        return _Scheduler(graph, columns, self.max_batches_inflight).run()
