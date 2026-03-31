from __future__ import annotations

"""
Compiled DAG pipeline: subclasses implement ``forward(x: PipeRef)`` using
``RayModule`` attributes. The first call records a static :class:`DagNode` tape;
:class:`DagPipelineExecutor` then runs many inputs with per-node ``max_inflight``
and ``ray.wait``-driven overlap.

Contrast :mod:`overlapped_pipeline`: this path keeps explicit dependency edges and
per-stage backpressure; use it when stages differ in cost, fan-out, or dispatch.

:class:`PipelineExecutor` is a small helper for a **linear** list of stages on top of
:class:`DagPipelineExecutor`.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Sequence, Set, Tuple, Union

import ray

from .ray_module import RayModule

Source = Union[str, Tuple[str, int]]


@dataclass(frozen=True)
class PipeRef:
    """Symbolic handle to a node's output (or ``"input"`` for the batch)."""

    source: Source


@dataclass(frozen=True)
class DagNode:
    name: str
    module: RayModule
    args: Tuple[Source, ...]
    kwargs: Mapping[str, Source]
    max_inflight: int = 1


@dataclass
class _InflightCall:
    node_name: str
    batch_idx: int
    pending: RayModule.RayModuleFuture


def _source_basename(src: Source) -> str:
    return src[0] if isinstance(src, tuple) else src


def _resolve_context(ctx: Dict[str, Any], src: Source) -> Any:
    if isinstance(src, tuple):
        node_name, idx = src
        return ctx[node_name][idx]
    return ctx[src]


def _pipe_arg_to_source(v: Any) -> Source:
    if isinstance(v, PipeRef):
        return v.source
    raise TypeError(
        "DAG forward only accepts PipeRef values from prior stages or the input handle"
    )


def _build_graph(
    nodes: Sequence[DagNode],
) -> Tuple[Dict[str, DagNode], Tuple[str, ...], Dict[str, List[str]], Dict[str, List[str]]]:
    """Validate tape order and return ``by_name``, static ``order``, ``deps``, ``downstream``."""
    if not nodes:
        raise ValueError("nodes cannot be empty")
    order = tuple(n.name for n in nodes)
    if len(set(order)) != len(order):
        raise ValueError("node names must be unique")

    by_name = {n.name: n for n in nodes}
    downstream: Dict[str, List[str]] = {n: [] for n in order}
    deps: Dict[str, List[str]] = {}

    for node in nodes:
        dep_set = {
            _source_basename(s)
            for s in list(node.args) + list(node.kwargs.values())
            if _source_basename(s) != "input"
        }
        dep_list = sorted(dep_set)
        for d in dep_list:
            if d not in by_name:
                raise ValueError(f"node '{node.name}' depends on unknown source '{d}'")
            downstream[d].append(node.name)
        deps[node.name] = dep_list

    return by_name, order, deps, downstream


class _DagScheduler:
    """
    One DAG run: per-batch context, per-node FIFO wait queues, and Ray ref bookkeeping.
    Logic is the same as before; grouping it here keeps :meth:`DagPipelineExecutor.run` short.
    """

    __slots__ = (
        "_by_name",
        "_order",
        "_deps",
        "_downstream",
        "_out_specs",
        "contexts",
        "results",
        "wait_q",
        "inflight",
        "done",
        "submitted",
        "enqueued",
        "ref_owner",
        "outstanding",
    )

    def __init__(
        self,
        by_name: Dict[str, DagNode],
        order: Tuple[str, ...],
        deps: Dict[str, List[str]],
        downstream: Dict[str, List[str]],
        inputs: Sequence[Any],
        out_specs: Tuple[Source, ...],
    ):
        self._by_name = by_name
        self._order = order
        self._deps = deps
        self._downstream = downstream
        self._out_specs = out_specs

        n = len(inputs)
        self.contexts: List[Dict[str, Any]] = [{"input": x} for x in inputs]
        self.results: List[Any] = [None] * n
        self.wait_q: Dict[str, Deque[int]] = {name: deque() for name in order}
        self.inflight: Dict[str, int] = {name: 0 for name in order}
        self.done: List[Set[str]] = [set() for _ in range(n)]
        self.submitted: List[Set[str]] = [set() for _ in range(n)]
        self.enqueued: List[Set[str]] = [set() for _ in range(n)]

        for bi in range(n):
            for name in order:
                if not deps[name]:
                    self.wait_q[name].append(bi)
                    self.enqueued[bi].add(name)

        self.ref_owner: Dict[ray.ObjectRef, _InflightCall] = {}
        self.outstanding: Set[ray.ObjectRef] = set()

    def _ready(self, node_name: str, bi: int) -> bool:
        return all(d in self.done[bi] for d in self._deps[node_name])

    def _submit(self, node_name: str, bi: int) -> None:
        node = self._by_name[node_name]
        ctx = self.contexts[bi]
        args = tuple(_resolve_context(ctx, s) for s in node.args)
        kwargs = {k: _resolve_context(ctx, s) for k, s in node.kwargs.items()}
        pending = node.module.remote(*args, **kwargs)
        self.inflight[node_name] += 1
        self.submitted[bi].add(node_name)
        self.enqueued[bi].discard(node_name)
        state = _InflightCall(node_name=node_name, batch_idx=bi, pending=pending)
        for r in pending.completion_refs():
            self.ref_owner[r] = state
            self.outstanding.add(r)

    def dispatch(self) -> None:
        """Fair scan: dequeue at most ``len(queue)`` times per node per round (probe)."""
        progressed = True
        while progressed:
            progressed = False
            for node_name in self._order:
                cap = self._by_name[node_name].max_inflight
                q = self.wait_q[node_name]
                probe = len(q)
                while probe > 0 and q and self.inflight[node_name] < cap:
                    bi = q.popleft()
                    probe -= 1
                    if self._ready(node_name, bi) and node_name not in self.submitted[bi]:
                        self._submit(node_name, bi)
                        progressed = True
                    else:
                        q.append(bi)

    def _store_final(self, bi: int) -> None:
        ctx = self.contexts[bi]
        specs = self._out_specs
        if all(_source_basename(s) in ctx for s in specs):
            if len(specs) == 1:
                self.results[bi] = _resolve_context(ctx, specs[0])
            else:
                self.results[bi] = tuple(_resolve_context(ctx, s) for s in specs)

    def _schedule_downstream(self, finished_node: str, bi: int) -> None:
        for child in self._downstream[finished_node]:
            if child in self.submitted[bi] or child in self.enqueued[bi]:
                continue
            if self._ready(child, bi):
                self.wait_q[child].append(bi)
                self.enqueued[bi].add(child)

    def run(self) -> List[Any]:
        self.dispatch()
        while self.outstanding:
            done_refs, _ = ray.wait(list(self.outstanding), num_returns=1)
            dr = done_refs[0]
            call = self.ref_owner.pop(dr)
            pend = call.pending
            need = pend.completion_refs()

            if any(r in self.ref_owner for r in need):
                self.outstanding.discard(dr)
                continue

            gathered = pend.gather()
            self.inflight[call.node_name] -= 1
            self.outstanding.difference_update(need)

            bi = call.batch_idx
            self.contexts[bi][call.node_name] = gathered
            self.done[bi].add(call.node_name)
            self._schedule_downstream(call.node_name, bi)
            self._store_final(bi)
            self.dispatch()

        return self.results


class DagPipelineExecutor:
    """Event-loop executor: one ``ray.wait`` completion, then reschedule ready nodes."""

    def __init__(self, nodes: Sequence[DagNode]):
        self._by_name, self._order, self._deps, self._downstream = _build_graph(nodes)
        self._nodes = list(nodes)

    def run(self, inputs: Sequence[Any], outputs: Sequence[Source] | None = None) -> List[Any]:
        if outputs is None:
            sinks = [n for n in self._order if not self._downstream[n]]
            if not sinks:
                raise ValueError("cannot infer output nodes for cyclic graph")
            outputs = tuple(sinks)
        out_specs = tuple(outputs)

        return _DagScheduler(
            self._by_name,
            self._order,
            self._deps,
            self._downstream,
            inputs,
            out_specs,
        ).run()


class DagPipeline:
    """
    Subclass, attach :class:`RayModule` attributes, implement ``forward(x)`` with
    ``PipeRef`` wiring; then ``pipe(inputs)`` runs the compiled DAG.
    """

    def __init__(self, *, stage_options: Mapping[str, Mapping[str, int]] | None = None):
        self._stage_options = dict(stage_options or {})
        self._nodes: List[DagNode] | None = None
        self._outputs: Tuple[Source, ...] | None = None

    def forward(self, x: PipeRef) -> PipeRef | Tuple[PipeRef, ...]:
        raise NotImplementedError("Subclass forward(x: PipeRef) -> PipeRef | tuple")

    def dummy_run(self, x: PipeRef) -> PipeRef | Tuple[PipeRef, ...]:
        return self.forward(x)

    class _StageProxy:
        __slots__ = ("_name", "_out_n", "_tape", "_module", "_cap")

        def __init__(
            self,
            name: str,
            out_n: int,
            tape: List[DagNode],
            module: RayModule,
            max_inflight: int,
        ):
            self._name = name
            self._out_n = out_n
            self._tape = tape
            self._module = module
            self._cap = max_inflight

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self._tape.append(
                DagNode(
                    name=self._name,
                    module=self._module,
                    args=tuple(_pipe_arg_to_source(a) for a in args),
                    kwargs={k: _pipe_arg_to_source(v) for k, v in kwargs.items()},
                    max_inflight=self._cap,
                )
            )
            if self._out_n == 1:
                return PipeRef(self._name)
            return tuple(PipeRef((self._name, i)) for i in range(self._out_n))

    def _swap_proxies(self, tape: List[DagNode]) -> Dict[str, RayModule]:
        originals: Dict[str, RayModule] = {}
        for name, value in list(self.__dict__.items()):
            if not isinstance(value, RayModule):
                continue
            if name in originals:
                raise ValueError(f"duplicated module field '{name}'")
            opts = dict(self._stage_options.get(name, {}))
            cap = max(1, int(opts.get("max_inflight", 1)))
            out_n = max(1, int(opts.get("outputs", 1)))
            originals[name] = value
            setattr(self, name, DagPipeline._StageProxy(name, out_n, tape, value, cap))
        if not originals:
            raise ValueError("No RayModule attributes on pipeline instance")
        return originals

    def _trace_graph(self) -> Tuple[List[DagNode], Tuple[Source, ...]]:
        tape: List[DagNode] = []
        originals = self._swap_proxies(tape)
        try:
            out = self.forward(PipeRef("input"))
        finally:
            for n, mod in originals.items():
                setattr(self, n, mod)

        if isinstance(out, tuple):
            out_specs = tuple(p.source for p in out)
        else:
            out_specs = (out.source,)
        return tape, out_specs

    def compile(self) -> DagPipeline:
        self._nodes, self._outputs = self._trace_graph()
        return self

    def __call__(self, inputs: Sequence[Any]) -> List[Any]:
        if self._nodes is None or self._outputs is None:
            self.compile()
        return DagPipelineExecutor(self._nodes).run(inputs, outputs=self._outputs)

    def run(self, inputs: Sequence[Any]) -> List[Any]:
        return self.__call__(inputs)


class PipelineExecutor:
    """Linear chain of :class:`RayModule` stages; each link has its own ``max_inflight`` cap."""

    def __init__(self, stages: Sequence[RayModule], max_inflight: Sequence[int] | None = None):
        if not stages:
            raise ValueError("stages cannot be empty")
        if max_inflight is not None and len(max_inflight) != len(stages):
            raise ValueError("max_inflight size must equal number of stages")

        nodes: List[DagNode] = []
        for i, stage in enumerate(stages):
            name = f"stage_{i}"
            args: Tuple[str, ...] = ("input",) if i == 0 else (f"stage_{i-1}",)
            lim = 1 if max_inflight is None else max_inflight[i]
            nodes.append(
                DagNode(name=name, module=stage, args=args, kwargs={}, max_inflight=max(1, int(lim)))
            )
        self._dag = DagPipelineExecutor(nodes)
        self._tail = f"stage_{len(stages) - 1}"

    def run(self, inputs: Sequence[Any]) -> List[Any]:
        return self._dag.run(inputs, outputs=(self._tail,))