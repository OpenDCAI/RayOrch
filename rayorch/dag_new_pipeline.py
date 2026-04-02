"""Declarative DAG pipeline with ``ray.wait``-driven scheduling.

Subclass :class:`DagPipeline`, attach ``RayModule`` attributes, and
implement ``forward()`` using symbolic ``PipeRef`` wiring.
``compile()`` traces the graph; ``__call__()`` executes batched inputs
through a scheduler with per-node inflight caps and automatic memory release.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import ray

from .dispatch_mode import Dispatch
from .ray_module import RayModule


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
    """Immutable DAG topology produced by :meth:`DagPipeline.compile`."""
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
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════

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
    """``ray.wait``-driven event loop executing a :class:`CompiledGraph`."""

    def __init__(
        self,
        graph: CompiledGraph,
        input_columns: Dict[str, Sequence[Any]],
        max_batches_inflight: int,
    ) -> None:
        self._graph = graph
        self._max_inflight = max(1, int(max_batches_inflight))
        self._n_batches = self._validate_inputs(input_columns)

        # Per-batch data: values stored as tuple for uniform slot access.
        # Single-output -> (value,), multi-output -> (v0, v1, ...).
        self._ctx: List[Dict[str, tuple]] = [
            {key: (input_columns[key][i],) for key in graph.input_keys}
            for i in range(self._n_batches)
        ]
        self._status: List[Dict[str, _NodeStatus]] = [
            {name: _NodeStatus() for name in graph.topo_order}
            for _ in range(self._n_batches)
        ]
        self._results: List[Any] = [None] * self._n_batches

        # Node scheduling state
        self._ready_q: Dict[str, Deque[int]] = {n: deque() for n in graph.topo_order}
        self._node_inflight: Dict[str, int] = {n: 0 for n in graph.topo_order}

        # Batch admission state
        self._live: set[int] = set()
        self._next_batch = 0

        # Ray ref tracking
        self._ref_owner: Dict[ray.ObjectRef, _InflightCall] = {}
        self._outstanding: set[ray.ObjectRef] = set()

        # Precomputed sets for fast membership tests
        self._output_nodes: frozenset[str] = frozenset(
            ref.node for ref in graph.graph_outputs
        )
        self._input_key_set: frozenset[str] = frozenset(graph.input_keys)

    def _validate_inputs(self, columns: Dict[str, Sequence[Any]]) -> int:
        expected = set(self._graph.input_keys)
        if set(columns.keys()) != expected:
            raise ValueError(
                f"input keys mismatch: expected {self._graph.input_keys}, "
                f"got {tuple(columns.keys())}"
            )
        sizes = {k: len(v) for k, v in columns.items()}
        if not sizes:
            raise ValueError("input columns cannot be empty")
        unique = set(sizes.values())
        if len(unique) != 1:
            raise ValueError(f"all input columns must have same length, got {sizes}")
        return next(iter(unique))

    # ── Main loop ─────────────────────────────────────────────────────────

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
        """Block for at least 1 ref, then greedily collect all other completed refs."""
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

    # ── Batch admission ───────────────────────────────────────────────────

    def _admit_batches(self) -> None:
        while (self._next_batch < self._n_batches
               and len(self._live) < self._max_inflight):
            bi = self._next_batch
            self._next_batch += 1
            self._live.add(bi)
            for name in self._graph.topo_order:
                if not self._graph.deps[name]:
                    self._mark_ready(bi, name)

    # ── Dispatch ──────────────────────────────────────────────────────────

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

        for i, v in enumerate(args):
            if not isinstance(v, list):
                raise TypeError(
                    f"node '{spec.name}' arg[{i}] must be list "
                    f"(got {type(v).__name__})"
                )
        for k, v in kw.items():
            if not isinstance(v, list):
                raise TypeError(
                    f"node '{spec.name}' kwarg '{k}' must be list "
                    f"(got {type(v).__name__})"
                )

        future = spec.module.remote(*args, **kw)
        refs = future.completion_refs()

        self._status[bi][spec.name].phase = "running"
        self._status[bi][spec.name].pending = _PendingResult(
            refs=refs, collect_fn=future.collect_fn,
        )
        self._node_inflight[spec.name] += 1

        call = _InflightCall(batch_idx=bi, node_name=spec.name, refs=list(refs))
        for ref in refs:
            self._ref_owner[ref] = call
            self._outstanding.add(ref)

    # ── Completion ────────────────────────────────────────────────────────

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

        _validate_output(spec, value)

        self._ctx[bi][name] = value if spec.num_outputs > 1 else (value,)
        st.phase = "done"
        st.pending = None
        self._node_inflight[name] -= 1

        for child in self._graph.consumers[name]:
            if self._all_deps_done(bi, child):
                self._mark_ready(bi, child)

        self._release_upstream(bi, name)

        if self._is_batch_complete(bi):
            self._results[bi] = self._materialize(bi)
            self._live.discard(bi)

    def _release_upstream(self, bi: int, name: str) -> None:
        for dep in self._graph.deps[name]:
            if dep in self._output_nodes or dep in self._input_key_set:
                continue
            if all(self._status[bi][c].phase == "done"
                   for c in self._graph.consumers[dep]):
                self._ctx[bi].pop(dep, None)

    # ── Queries ───────────────────────────────────────────────────────────

    def _all_deps_done(self, bi: int, name: str) -> bool:
        return all(
            self._status[bi][d].phase == "done" for d in self._graph.deps[name]
        )

    def _is_batch_complete(self, bi: int) -> bool:
        return all(
            self._status[bi][ref.node].phase == "done"
            for ref in self._graph.graph_outputs
        )

    def _materialize(self, bi: int) -> Any:
        ctx = self._ctx[bi]
        outs = self._graph.graph_outputs
        if len(outs) == 1:
            return ctx[outs[0].node][outs[0].index]
        return tuple(ctx[ref.node][ref.index] for ref in outs)


def _validate_output(spec: NodeSpec, value: Any) -> None:
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


# ═══════════════════════════════════════════════════════════════════════════
# User API
# ═══════════════════════════════════════════════════════════════════════════

def _propagate_forward_sig(cls: type) -> None:
    """Copy ``forward()``'s parameter names onto ``__call__`` / ``run``.

    Each ``PipeRef`` annotation (or unannotated param) is replaced with
    ``Sequence[Any]`` so that IDE tooltips and ``help()`` show the user
    exactly which batched inputs to pass.
    """
    try:
        fwd_sig = inspect.signature(cls.forward)
    except (ValueError, TypeError):
        return

    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name, p in fwd_sig.parameters.items():
        if name == "self":
            continue
        params.append(p.replace(annotation=Sequence[Any]))

    sig = fwd_sig.replace(parameters=params, return_annotation=List[Any])

    base_call = DagPipeline.__call__

    def _typed_call(self, *args, **kwargs):
        return base_call(self, *args, **kwargs)

    _typed_call.__signature__ = sig
    _typed_call.__name__ = "__call__"
    _typed_call.__qualname__ = f"{cls.__qualname__}.__call__"
    _typed_call.__module__ = cls.__module__
    cls.__call__ = _typed_call
    cls.run = _typed_call


class DagPipeline:
    """
    Declarative DAG pipeline base class.

    Subclass, attach ``RayModule`` attributes, implement ``forward()``
    with ``PipeRef`` wiring, then call the pipeline on batched inputs.

    ``__call__`` / ``run`` signatures are automatically derived from
    ``forward()`` via ``__init_subclass__``, so IDE autocompletion shows
    the correct parameter names and ``Sequence[Any]`` types.
    """

    def __init__(self, *, max_batches_inflight: int = 4):
        self.max_batches_inflight = max(1, int(max_batches_inflight))
        self._compiled: Optional[CompiledGraph] = None
        self._param_names: Tuple[str, ...] = ()
        self._input_keys: Tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "forward" in cls.__dict__:
            _propagate_forward_sig(cls)

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

    def compile(self) -> DagPipeline:
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
        if self._compiled is None:
            self.compile()
        columns = self._resolve_inputs(inputs, named_inputs)
        return _Scheduler(
            self._compiled, columns, self.max_batches_inflight,
        ).run()

    run = __call__


# ═══════════════════════════════════════════════════════════════════════════
# Inline tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    def _cleanup_pipe_modules(pipe: DagPipeline) -> None:
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

    class ShardProbePipe(DagPipeline):
        def __init__(self, replicas: int):
            self.probe = RayModule(
                ProbeShardSizeOp,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            super().__init__(max_batches_inflight=4)

        def forward(self, x: PipeRef) -> PipeRef:
            return self.probe(x)

    class MultiplyBy2Op:
        def run(self, x: list[int]) -> list[int]:
            return [2 * v for v in x]

    class ShardPostProcessPipe(DagPipeline):
        def __init__(self, replicas: int):
            self.mul2 = RayModule(
                MultiplyBy2Op,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            super().__init__(max_batches_inflight=4)

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

    class MultiReplicaSplitPipe(DagPipeline):
        def __init__(self, replicas: int):
            self.split = RayModule(
                SplitParityOp,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
                max_inflight=2,
            ).pre_init()
            self.left = RayModule(ScaleEvenOp, replicas=1, max_inflight=2).pre_init()
            self.right = RayModule(ScaleOddOp, replicas=1, max_inflight=2).pre_init()
            super().__init__(max_batches_inflight=4)

        def forward(self, x: PipeRef) -> Tuple[PipeRef, PipeRef]:
            even_in, odd_in = self.split(x)
            return self.left(even_in), self.right(odd_in)

    ray.init(ignore_reinit_error=True, num_cpus=12)
    replicas = 3
    xs = [list(range(10)), list(range(7)), [11, 12, 13, 14, 15]]

    # 1) Validate contiguous shard split + remainder(mod) behavior.
    probe_pipe = ShardProbePipe(replicas=replicas)
    shard_out = probe_pipe(xs)
    expect_shards = [shard_sizes(len(batch), replicas) for batch in xs]
    print("shard inputs :", [len(b) for b in xs])
    print("shard output :", shard_out)
    print("shard expect :", expect_shards)
    print("shard match  :", shard_out == expect_shards)
    if shard_out != expect_shards:
        raise RuntimeError("Shard split/remainder behavior mismatch")
    _cleanup_pipe_modules(probe_pipe)

    # 2) Validate shard-content correctness with an explicit post-process op.
    post_pipe = ShardPostProcessPipe(replicas=replicas)
    post_out = post_pipe(xs)
    expect_post = [[2 * v for v in batch] for batch in xs]
    print("post output  :", post_out)
    print("post expect  :", expect_post)
    print("post match   :", post_out == expect_post)
    if post_out != expect_post:
        raise RuntimeError("Shard post-process value mismatch")
    _cleanup_pipe_modules(post_pipe)

    # 3) Validate content separation and branch correctness for multi-output.
    split_pipe = MultiReplicaSplitPipe(replicas=replicas)
    split_out = split_pipe(xs)
    expect_split = [
        (
            [2 * v for v in batch if v % 2 == 0],
            [3 * v for v in batch if v % 2 == 1],
        )
        for batch in xs
    ]
    print("split output :", split_out)
    print("split expect :", expect_split)
    print("split match  :", split_out == expect_split)
    if split_out != expect_split:
        raise RuntimeError("Multi-output split content mismatch")
    _cleanup_pipe_modules(split_pipe)

    ray.shutdown()
