from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin, get_type_hints

import ray

from .dispatch_mode import Dispatch
from .ray_module import RayModule


Source = Union[str, Tuple[str, int]]


@dataclass(frozen=True)
class PipeRef:
    """Symbolic handle to a node output slot used during DAG tracing."""
    source: Source


@dataclass(frozen=True)
class NodeSpec:
    """Static node definition in the compiled DAG graph."""
    name: str
    module: RayModule
    args: Tuple[Source, ...]
    kwargs: Mapping[str, Source]
    max_inflight: int = 1
    outputs: int = 1


@dataclass
class NodeOutput:
    """Runtime handle for one submitted node invocation (refs + optional multi-replica collect)."""
    refs: List[ray.ObjectRef]
    collect_fn: Optional[Any]

@dataclass
class BatchNodeState:
    """Per-batch state for a node in the scheduler."""
    status: str = "waiting"
    output: Optional[NodeOutput] = None


@dataclass
class InflightReplica:
    """Owner metadata for one in-flight call (possibly multiple refs)."""
    batch_idx: int
    node_name: str
    refs: List[ray.ObjectRef]


@dataclass
class CompiledGraph:
    """Compiled immutable DAG artifacts consumed by the scheduler."""
    nodes: Dict[str, NodeSpec]
    order: Tuple[str, ...]
    deps: Dict[str, Tuple[str, ...]]
    downstream: Dict[str, Tuple[str, ...]]
    outputs: Tuple[Source, ...]
    input_roots: Tuple[str, ...]


class _GraphBuilder:
    """Builds DAG node tape and validates source wiring during compile."""

    def __init__(self, pipe: "DagPipeline"):
        self.pipe = pipe
        self.nodes: List[NodeSpec] = []
        self._name_id: Dict[str, int] = {}

    def new_name(self, base: str) -> str:
        idx = self._name_id.get(base, 0)
        self._name_id[base] = idx + 1
        return base if idx == 0 else f"{base}_{idx}"

    def add_compute(
        self,
        *,
        base_name: str,
        module: RayModule,
        args: Tuple[Source, ...],
        kwargs: Mapping[str, Source],
        max_inflight: int,
        outputs: int,
    ) -> str:
        name = self.new_name(base_name)
        self.nodes.append(
            NodeSpec(
                name=name,
                module=module,
                args=args,
                kwargs=dict(kwargs),
                max_inflight=max(1, int(max_inflight)),
                outputs=max(1, int(outputs)),
            )
        )
        return name

    def output_refs(self, node_name: str, outputs: int) -> PipeRef | Tuple[PipeRef, ...]:
        out_n = max(1, int(outputs))
        if out_n == 1:
            return PipeRef(node_name)
        return tuple(PipeRef((node_name, i)) for i in range(out_n))

    def finish(self, out: PipeRef | Tuple[PipeRef, ...], input_roots: Tuple[str, ...]) -> CompiledGraph:
        if isinstance(out, tuple):
            outputs = tuple(v.source for v in out)
        else:
            outputs = (out.source,)
        input_root_set = set(input_roots)
        by_name = {n.name: n for n in self.nodes}
        order = tuple(n.name for n in self.nodes)
        deps: Dict[str, Tuple[str, ...]] = {}
        downstream: Dict[str, List[str]] = {n: [] for n in order}
        for node in self.nodes:
            dep_names = []
            seen = set()
            for src in list(node.args) + list(node.kwargs.values()):
                base = src[0] if isinstance(src, tuple) else src
                if isinstance(src, tuple):
                    idx = src[1]
                    if idx < 0:
                        raise ValueError(f"node '{node.name}' has negative output index for source '{src}'")
                if base in input_root_set or base in seen:
                    continue
                if base not in by_name:
                    raise ValueError(f"node '{node.name}' depends on unknown source '{base}'")
                if isinstance(src, tuple):
                    idx = src[1]
                    if idx >= by_name[base].outputs:
                        raise ValueError(
                            f"node '{node.name}' reads output {idx} from '{base}', "
                            f"but '{base}' only has {by_name[base].outputs} outputs"
                        )
                seen.add(base)
                dep_names.append(base)
                downstream[base].append(node.name)
            deps[node.name] = tuple(dep_names)
        for src in outputs:
            if isinstance(src, tuple):
                base, idx = src
                if base in input_root_set:
                    if idx < 0:
                        raise ValueError(f"graph output has negative output index for source '{src}'")
                    continue
                if base not in by_name:
                    raise ValueError(f"graph output depends on unknown source '{base}'")
                if idx < 0 or idx >= by_name[base].outputs:
                    raise ValueError(
                        f"graph output reads output {idx} from '{base}', "
                        f"but '{base}' only has {by_name[base].outputs} outputs"
                    )
        return CompiledGraph(
            nodes=by_name,
            order=order,
            deps=deps,
            downstream={k: tuple(v) for k, v in downstream.items()},
            outputs=outputs,
            input_roots=input_roots,
        )


class _StageProxy:
    """Proxy object that records symbolic stage calls instead of running actors."""

    def __init__(self, pipe: "DagPipeline", builder: _GraphBuilder, field_name: str, module: RayModule):
        self.pipe = pipe
        self.builder = builder
        self.field_name = field_name
        self.module = module

    def __call__(self, *args: Any, **kwargs: Any) -> PipeRef | Tuple[PipeRef, ...]:
        src_args = tuple(self.pipe._to_source(v) for v in args)
        src_kwargs = {k: self.pipe._to_source(v) for k, v in kwargs.items()}
        spec = self.pipe._stage_option(self.field_name, self.module)
        compute_name = self.builder.add_compute(
            base_name=self.field_name,
            module=self.module,
            args=src_args,
            kwargs=src_kwargs,
            max_inflight=spec["compute_inflight"],
            outputs=spec["outputs"],
        )
        return self.builder.output_refs(compute_name, spec["outputs"])


class _Scheduler:
    """Event-loop scheduler with per-node inflight caps and per-batch contexts."""

    def __init__(
        self,
        graph: CompiledGraph,
        input_columns: Mapping[str, Sequence[Any]],
        max_batches_inflight: int,
    ):
        self.graph = graph
        self.max_batches_inflight = max(1, int(max_batches_inflight))

        expected_roots = set(self.graph.input_roots)
        if set(input_columns.keys()) != expected_roots:
            raise ValueError(
                f"input roots mismatch: expected {self.graph.input_roots}, "
                f"got {tuple(input_columns.keys())}"
            )
        sizes = {k: len(v) for k, v in input_columns.items()}
        if not sizes:
            raise ValueError("input_columns cannot be empty")
        batch_sizes = set(sizes.values())
        if len(batch_sizes) != 1:
            raise ValueError(f"all input columns must have same batch size, got {sizes}")
        self._n_batches = next(iter(batch_sizes))

        self.contexts: List[Dict[str, Any]] = [
            {root: input_columns[root][i] for root in self.graph.input_roots}
            for i in range(self._n_batches)
        ]
        self.states: List[Dict[str, BatchNodeState]] = [
            {name: BatchNodeState() for name in self.graph.order} for _ in range(self._n_batches)
        ]
        self.results: List[Any] = [None] * self._n_batches

        self.ready_q: Dict[str, Deque[int]] = {name: deque() for name in self.graph.order}
        self.inflight_per_node: Dict[str, int] = {name: 0 for name in self.graph.order}
        self.live_batches: set[int] = set()
        self.next_batch_to_admit = 0

        self.ref_to_call: Dict[ray.ObjectRef, InflightReplica] = {}
        self.outstanding: set[ray.ObjectRef] = set()

    def run(self) -> List[Any]:
        """Drive the DAG until all tracked refs are completed."""
        self._admit_batches()
        self._dispatch_ready()
        while self.outstanding:
            done, _ = ray.wait(list(self.outstanding), num_returns=1)
            ref = done[0]
            call = self.ref_to_call.pop(ref)
            self.outstanding.discard(ref)

            # Multi-replica calls complete only after all refs of that call are done.
            if any(r in self.ref_to_call for r in call.refs):
                continue

            self._finish_node(call.batch_idx, call.node_name, call.refs)
            self._dispatch_ready()
            self._admit_batches()
            self._dispatch_ready()
        return self.results

    def _admit_batches(self) -> None:
        """Admit new batches into the graph under global inflight budget."""
        while self.next_batch_to_admit < self._n_batches and len(self.live_batches) < self.max_batches_inflight:
            bi = self.next_batch_to_admit
            self.next_batch_to_admit += 1
            self.live_batches.add(bi)
            for node_name in self.graph.order:
                if not self.graph.deps[node_name]:
                    self._enqueue_ready(bi, node_name)

    def _enqueue_ready(self, bi: int, node_name: str) -> None:
        state = self.states[bi][node_name]
        if state.status != "waiting":
            return
        state.status = "ready"
        self.ready_q[node_name].append(bi)

    def _dispatch_ready(self) -> None:
        """Submit ready nodes while respecting per-node inflight limits."""
        progressed = True
        while progressed:
            progressed = False
            for node_name in self.graph.order:
                q = self.ready_q[node_name]
                node = self.graph.nodes[node_name]
                while q and self.inflight_per_node[node_name] < node.max_inflight:
                    bi = q.popleft()
                    state = self.states[bi][node_name]
                    if state.status != "ready":
                        continue
                    self._submit_node(bi, node)
                    progressed = True

    def _submit_node(self, bi: int, node: NodeSpec) -> None:
        refs = self._launch_node(bi, node)
        state = self.states[bi][node.name]
        state.status = "running"
        self.inflight_per_node[node.name] += 1
        call = InflightReplica(batch_idx=bi, node_name=node.name, refs=list(refs))
        for ref in refs:
            self.ref_to_call[ref] = call
            self.outstanding.add(ref)

    def _launch_node(self, bi: int, node: NodeSpec) -> List[ray.ObjectRef]:
        ctx = self.contexts[bi]
        args = tuple(self._resolve_submit_arg(ctx, src) for src in node.args)
        kwargs = {k: self._resolve_submit_arg(ctx, src) for k, src in node.kwargs.items()}
        # Pipeline protocol: every edge payload is list-based micro-batch data.
        for i, v in enumerate(args):
            if not isinstance(v, list):
                raise TypeError(
                    f"node '{node.name}' arg[{i}] must be list[...] "
                    f"(got {type(v).__name__})"
                )
        for k, v in kwargs.items():
            if not isinstance(v, list):
                raise TypeError(
                    f"node '{node.name}' kwarg '{k}' must be list[...] "
                    f"(got {type(v).__name__})"
                )
        pending = node.module.remote(*args, **kwargs)
        self.states[bi][node.name].output = NodeOutput(
            refs=pending.completion_refs(),
            collect_fn=pending.collect_fn,
        )
        return pending.completion_refs()

    def _resolve_submit_arg(self, ctx: Dict[str, Any], src: Source) -> Any:
        if isinstance(src, tuple):
            base, idx = src
            val = ctx[base]
            return val[idx]
        return ctx[src]

    def _validate_node_value(self, node: NodeSpec, value: Any) -> None:
        """Enforce list-based output contract for single/multi-output nodes."""
        if node.outputs == 1:
            if not isinstance(value, list):
                raise TypeError(
                    f"node '{node.name}' must return list[...] (got {type(value).__name__})"
                )
            return

        if not isinstance(value, (tuple, list)):
            raise TypeError(
                f"node '{node.name}' must return tuple/list of {node.outputs} list outputs "
                f"(got {type(value).__name__})"
            )
        if len(value) != node.outputs:
            raise ValueError(
                f"node '{node.name}' declared outputs={node.outputs}, "
                f"but runtime returned {len(value)} outputs"
            )
        for i, branch in enumerate(value):
            if not isinstance(branch, list):
                raise TypeError(
                    f"node '{node.name}' output[{i}] must be list[...] "
                    f"(got {type(branch).__name__})"
                )

    def _finish_node(self, bi: int, node_name: str, refs: List[ray.ObjectRef]) -> None:
        """Finalize one completed call, update context, and schedule downstream nodes."""
        node = self.graph.nodes[node_name]
        state = self.states[bi][node_name]
        output = state.output
        if output is None:
            raise RuntimeError(f"node '{node_name}' completed without output handle")

        if len(refs) == 1:
            value = ray.get(refs[0])
        else:
            vals = ray.get(refs)
            value = output.collect_fn(node.module, vals) if output.collect_fn is not None else vals

        self._validate_node_value(node, value)
        self.contexts[bi][node_name] = value
        state.status = "done"
        self.inflight_per_node[node_name] -= 1

        for child in self.graph.downstream[node_name]:
            if self._deps_done(bi, child):
                self._enqueue_ready(bi, child)

        if self._batch_outputs_ready(bi):
            self.results[bi] = self._materialize_outputs(bi)
            self.live_batches.discard(bi)

    def _deps_done(self, bi: int, node_name: str) -> bool:
        for dep in self.graph.deps[node_name]:
            if self.states[bi][dep].status != "done":
                return False
        return True

    def _batch_outputs_ready(self, bi: int) -> bool:
        for src in self.graph.outputs:
            base = src[0] if isinstance(src, tuple) else src
            if self.states[bi][base].status != "done":
                return False
        return True

    def _materialize_outputs(self, bi: int) -> Any:
        ctx = self.contexts[bi]
        if len(self.graph.outputs) == 1:
            return self._resolve_submit_arg(ctx, self.graph.outputs[0])
        return tuple(self._resolve_submit_arg(ctx, src) for src in self.graph.outputs)

class DagPipeline:
    """Declarative DAG pipeline base class for explicit graph-style wiring."""

    def __init__(
        self,
        *,
        max_batches_inflight: int = 4,
        stage_options: Optional[Mapping[str, Mapping[str, int]]] = None,
    ):
        self.max_batches_inflight = max(1, int(max_batches_inflight))
        self._stage_options = dict(stage_options or {})
        self._compiled: Optional[CompiledGraph] = None
        self._input_param_names: Tuple[str, ...] = ()
        self._input_root_names: Tuple[str, ...] = ()

    def forward(self, x: PipeRef) -> PipeRef | Tuple[PipeRef, ...]:
        raise NotImplementedError

    def _infer_outputs_from_module(self, module: RayModule) -> int:
        """Infer logical output arity from the op `run()` return annotation."""
        op_cls = getattr(module, "_op_cls", None)
        if op_cls is None:
            return 1
        run_fn = getattr(op_cls, "run", None)
        if run_fn is None:
            return 1
        try:
            return_hint = get_type_hints(run_fn).get("return")
        except Exception:
            return 1
        if return_hint is None:
            return 1

        origin = get_origin(return_hint)
        if origin in (tuple, Tuple):
            parts = [p for p in get_args(return_hint) if p is not Ellipsis]
            return max(1, len(parts))
        if isinstance(return_hint, type) and hasattr(return_hint, "_fields") and issubclass(return_hint, tuple):
            return max(1, len(return_hint._fields))
        return 1

    def _stage_option(self, name: str, module: RayModule) -> Dict[str, int]:
        opts = dict(self._stage_options.get(name, {}))
        inferred_outputs = self._infer_outputs_from_module(module)
        configured_outputs = opts.get("outputs")
        if configured_outputs is None:
            outputs = inferred_outputs
        else:
            outputs = max(1, int(configured_outputs))
            if outputs != inferred_outputs and inferred_outputs != 1:
                raise ValueError(
                    f"stage '{name}' outputs mismatch: configured outputs={outputs}, "
                    f"but run() return annotation implies outputs={inferred_outputs}"
                )
        return {
            "compute_inflight": max(1, int(opts.get("compute_inflight", opts.get("max_inflight", 1)))),
            "outputs": outputs,
        }

    def _to_source(self, v: Any) -> Source:
        if isinstance(v, PipeRef):
            return v.source
        raise TypeError("forward only accepts PipeRef values returned by upstream stages")

    def _root_name(self, param_name: str) -> str:
        # Internal namespace avoids collision with stage node names.
        return f"__input__{param_name}"

    def _build_forward_inputs(self) -> Tuple[List[PipeRef], Dict[str, PipeRef], Tuple[str, ...], Tuple[str, ...]]:
        sig = inspect.signature(self.forward)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("forward must declare at least one PipeRef input parameter")

        args: List[PipeRef] = []
        kwargs: Dict[str, PipeRef] = {}
        names: List[str] = []
        roots: List[str] = []
        for p in params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError("forward cannot use *args/**kwargs; declare explicit PipeRef parameters")
            name = p.name
            root = self._root_name(name)
            ref = PipeRef(root)
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(ref)
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = ref
            else:
                raise TypeError(f"unsupported forward parameter kind for '{name}': {p.kind}")
            names.append(name)
            roots.append(root)
        return args, kwargs, tuple(names), tuple(roots)

    def compile(self) -> "DagPipeline":
        builder = _GraphBuilder(self)
        originals: Dict[str, RayModule] = {}
        for name, value in list(self.__dict__.items()):
            if isinstance(value, RayModule):
                originals[name] = value
                setattr(self, name, _StageProxy(self, builder, name, value))
        if not originals:
            raise ValueError("No RayModule attributes on pipeline instance")

        f_args, f_kwargs, param_names, root_names = self._build_forward_inputs()
        try:
            out = self.forward(*f_args, **f_kwargs)
        finally:
            for name, value in originals.items():
                setattr(self, name, value)

        self._input_param_names = param_names
        self._input_root_names = root_names
        self._compiled = builder.finish(out, root_names)
        return self

    def _normalize_runtime_inputs(
        self,
        inputs: Tuple[Sequence[Any], ...],
        named_inputs: Mapping[str, Sequence[Any]],
    ) -> Dict[str, Sequence[Any]]:
        if not self._input_param_names or not self._input_root_names:
            raise RuntimeError("pipeline is not compiled with input roots")

        if inputs and named_inputs:
            raise ValueError("Pass runtime inputs either positionally or by keyword, not both")

        if named_inputs:
            expected = set(self._input_param_names)
            got = set(named_inputs.keys())
            if got != expected:
                raise ValueError(
                    f"runtime input names mismatch: expected {self._input_param_names}, got {tuple(named_inputs.keys())}"
                )
            return {
                self._root_name(name): named_inputs[name]
                for name in self._input_param_names
            }

        if len(inputs) != len(self._input_param_names):
            raise ValueError(
                f"runtime positional input count mismatch: expected {len(self._input_param_names)}, got {len(inputs)}"
            )
        return {
            root: col for root, col in zip(self._input_root_names, inputs)
        }

    def __call__(self, *inputs: Sequence[Any], **named_inputs: Sequence[Any]) -> List[Any]:
        if self._compiled is None:
            self.compile()
        columns = self._normalize_runtime_inputs(inputs, named_inputs)
        return _Scheduler(self._compiled, columns, self.max_batches_inflight).run()

    run = __call__


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
            # Return local shard length for this replica.
            return [len(x)]

    class ShardProbePipe(DagPipeline):
        def __init__(self, replicas: int):
            self.probe = RayModule(
                ProbeShardSizeOp,
                replicas=replicas,
                dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            ).pre_init()
            super().__init__(max_batches_inflight=4, stage_options={"probe": {"compute_inflight": 2}})

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
            ).pre_init()
            super().__init__(max_batches_inflight=4, stage_options={"mul2": {"compute_inflight": 2}})

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
            ).pre_init()
            self.left = RayModule(ScaleEvenOp, replicas=1).pre_init()
            self.right = RayModule(ScaleOddOp, replicas=1).pre_init()
            super().__init__(
                max_batches_inflight=4,
                stage_options={
                    # outputs auto inferred from SplitParityOp.run -> tuple[list, list]
                    "split": {"compute_inflight": 2},
                    "left": {"compute_inflight": 2},
                    "right": {"compute_inflight": 2},
                },
            )

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
