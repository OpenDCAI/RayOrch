from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Sequence, Tuple, Union

import ray

from .ray_module import RayModule


Source = Union[str, Tuple[str, int]]


@dataclass(frozen=True)
class DagNode:
    name: str
    module: RayModule
    args: Tuple[Source, ...]
    kwargs: Mapping[str, Source]
    max_inflight: int = 1


@dataclass(frozen=True)
class PipeRef:
    source: Source


@dataclass
class _InFlight:
    node_name: str
    batch_idx: int
    pending: RayModule.PendingResult


def _to_source_name(src: Source) -> str:
    return src[0] if isinstance(src, tuple) else src


def _resolve_source(context: Dict[str, Any], src: Source) -> Any:
    if isinstance(src, tuple):
        node_name, idx = src
        return context[node_name][idx]
    return context[src]


class DagPipelineExecutor:
    """
    Event-loop DAG pipeline executor with per-node in-flight limits.
    """

    def __init__(self, nodes: Sequence[DagNode]):
        if not nodes:
            raise ValueError("nodes cannot be empty")
        node_names = [n.name for n in nodes]
        if len(set(node_names)) != len(node_names):
            raise ValueError("node names must be unique")
        self._nodes = list(nodes)
        self._node_map = {n.name: n for n in nodes}
        self._node_order = node_names
        self._downstream: Dict[str, List[str]] = {name: [] for name in node_names}
        self._deps: Dict[str, List[str]] = {}
        for node in self._nodes:
            deps = sorted(
                {
                    _to_source_name(src)
                    for src in list(node.args) + list(node.kwargs.values())
                    if _to_source_name(src) != "input"
                }
            )
            for d in deps:
                if d not in self._node_map:
                    raise ValueError(f"node '{node.name}' depends on unknown source '{d}'")
                self._downstream[d].append(node.name)
            self._deps[node.name] = deps

    def run(self, inputs: Sequence[Any], outputs: Sequence[Source] | None = None) -> List[Any]:
        if outputs is None:
            sink_nodes = [n for n in self._node_order if not self._downstream[n]]
            if not sink_nodes:
                raise ValueError("cannot infer output nodes for cyclic graph")
            outputs = tuple(sink_nodes)

        resolved_outputs = tuple(outputs)
        results: List[Any] = [None] * len(inputs)
        contexts: List[Dict[str, Any]] = [{"input": x} for x in inputs]
        waiting_queues: Dict[str, Deque[int]] = {name: deque() for name in self._node_order}

        for batch_idx in range(len(inputs)):
            for node_name in self._node_order:
                if not self._deps[node_name]:
                    waiting_queues[node_name].append(batch_idx)
                    # queued_nodes initialized below, filled after declaration

        inflight_counts: Dict[str, int] = {name: 0 for name in self._node_order}
        done_nodes: List[set[str]] = [set() for _ in inputs]
        submitted_nodes: List[set[str]] = [set() for _ in inputs]
        queued_nodes: List[set[str]] = [set() for _ in inputs]
        for batch_idx in range(len(inputs)):
            for node_name in self._node_order:
                if not self._deps[node_name]:
                    queued_nodes[batch_idx].add(node_name)
        ref_to_state: Dict[ray.ObjectRef, _InFlight] = {}
        ready_refs: List[ray.ObjectRef] = []

        def _submit_one(node_name: str, batch_idx: int) -> None:
            node = self._node_map[node_name]
            context = contexts[batch_idx]
            args = tuple(_resolve_source(context, src) for src in node.args)
            kwargs = {k: _resolve_source(context, src) for k, src in node.kwargs.items()}
            pending = node.module.submit(*args, **kwargs)
            inflight_counts[node_name] += 1
            submitted_nodes[batch_idx].add(node_name)
            queued_nodes[batch_idx].discard(node_name)
            refs = pending.refs if isinstance(pending.refs, list) else [pending.refs]
            state = _InFlight(node_name=node_name, batch_idx=batch_idx, pending=pending)
            for r in refs:
                ref_to_state[r] = state
                ready_refs.append(r)

        def _is_node_ready(node_name: str, batch_idx: int) -> bool:
            if node_name in submitted_nodes[batch_idx]:
                return False
            return all(dep in done_nodes[batch_idx] for dep in self._deps[node_name])

        def _try_dispatch() -> None:
            made_progress = True
            while made_progress:
                made_progress = False
                for node_name in self._node_order:
                    node = self._node_map[node_name]
                    probe = len(waiting_queues[node_name])
                    while (
                        probe > 0
                        and waiting_queues[node_name]
                        and inflight_counts[node_name] < node.max_inflight
                    ):
                        batch_idx = waiting_queues[node_name].popleft()
                        probe -= 1
                        if _is_node_ready(node_name, batch_idx):
                            _submit_one(node_name, batch_idx)
                            made_progress = True
                        else:
                            waiting_queues[node_name].append(batch_idx)

        _try_dispatch()

        while ready_refs:
            done, _ = ray.wait(ready_refs, num_returns=1)
            done_ref = done[0]
            state = ref_to_state.pop(done_ref)
            pending_refs = state.pending.refs if isinstance(state.pending.refs, list) else [state.pending.refs]

            if not all(r not in ref_to_state for r in pending_refs):
                ready_refs = [r for r in ready_refs if r != done_ref]
                continue

            node_name = state.node_name
            output = self._node_map[node_name].module.gather(state.pending)
            inflight_counts[node_name] -= 1
            ready_refs = [r for r in ready_refs if r not in pending_refs]
            contexts[state.batch_idx][node_name] = output
            done_nodes[state.batch_idx].add(node_name)

            for child in self._downstream[node_name]:
                if _is_node_ready(child, state.batch_idx) and child not in queued_nodes[state.batch_idx]:
                    waiting_queues[child].append(state.batch_idx)
                    queued_nodes[state.batch_idx].add(child)

            if all(_to_source_name(out) in contexts[state.batch_idx] for out in resolved_outputs):
                if len(resolved_outputs) == 1:
                    results[state.batch_idx] = _resolve_source(contexts[state.batch_idx], resolved_outputs[0])
                else:
                    results[state.batch_idx] = tuple(
                        _resolve_source(contexts[state.batch_idx], out) for out in resolved_outputs
                    )

            _try_dispatch()

        return results


class PipelineExecutor:
    """
    Backward-compatible linear pipeline executor.
    """

    def __init__(self, stages: Sequence[RayModule], max_inflight: Sequence[int] | None = None):
        if not stages:
            raise ValueError("stages cannot be empty")
        if max_inflight is not None and len(max_inflight) != len(stages):
            raise ValueError("max_inflight size must equal number of stages")
        nodes: List[DagNode] = []
        for i, stage in enumerate(stages):
            node_name = f"stage_{i}"
            args: Tuple[Source, ...] = ("input",) if i == 0 else ((f"stage_{i-1}",),)
            if i > 0:
                args = (f"stage_{i-1}",)
            limit = 1 if max_inflight is None else max_inflight[i]
            nodes.append(
                DagNode(
                    name=node_name,
                    module=stage,
                    args=args,
                    kwargs={},
                    max_inflight=max(1, int(limit)),
                )
            )
        self._dag = DagPipelineExecutor(nodes)

    def run(self, inputs: Sequence[Any]) -> List[Any]:
        return self._dag.run(inputs, outputs=("stage_" + str(len(self._dag._node_order) - 1),))


class DummyRunPipeline:
    """
    nn.Module-like pipeline base class.

    Usage:
      class MyPipe(DummyRunPipeline):
          def __init__(self):
              self.pre = RayModule(...)
              self.post = RayModule(...)
              super().__init__(stage_options={"pre": {"max_inflight": 2}})
          def dummy_run(self, x):
              y = self.pre(x)
              return self.post(y)

      pipe = MyPipe()
      outs = pipe.run([mb1, mb2, ...])
    """

    def __init__(self, *, stage_options: Mapping[str, Mapping[str, int]] | None = None):
        self._stage_options = dict(stage_options or {})
        self._compiled_nodes: List[DagNode] | None = None
        self._compiled_outputs: Tuple[Source, ...] | None = None

    def dummy_run(self, x: PipeRef):
        raise NotImplementedError

    class _BoundModuleProxy:
        def __init__(
            self,
            module_name: str,
            outputs: int,
            sink_nodes: List[DagNode],
            module: RayModule,
            max_inflight: int,
        ):
            self._name = module_name
            self._outputs = outputs
            self._sink_nodes = sink_nodes
            self._module = module
            self._max_inflight = max_inflight

        def __call__(self, *args: Any, **kwargs: Any):
            def _to_source(v: Any) -> Source:
                if isinstance(v, PipeRef):
                    return v.source
                raise TypeError("dummy_run only supports PipeRef inputs and outputs from previous RayModule calls")

            args_src = tuple(_to_source(a) for a in args)
            kwargs_src = {k: _to_source(v) for k, v in kwargs.items()}
            self._sink_nodes.append(
                DagNode(
                    name=self._name,
                    module=self._module,
                    args=args_src,
                    kwargs=kwargs_src,
                    max_inflight=self._max_inflight,
                )
            )
            if self._outputs == 1:
                return PipeRef(self._name)
            return tuple(PipeRef((self._name, i)) for i in range(self._outputs))

    def _build(self) -> Tuple[List[DagNode], Tuple[Source, ...]]:
        x = PipeRef("input")
        nodes: List[DagNode] = []
        originals: Dict[str, RayModule] = {}
        for name, value in list(self.__dict__.items()):
            if isinstance(value, RayModule):
                if name in originals:
                    raise ValueError(f"duplicated module name '{name}'")
                opts = dict(self._stage_options.get(name, {}))
                max_inflight = max(1, int(opts.get("max_inflight", 1)))
                outputs = max(1, int(opts.get("outputs", 1)))
                proxy = DummyRunPipeline._BoundModuleProxy(name, outputs, nodes, value, max_inflight)
                originals[name] = value
                setattr(self, name, proxy)

        if not originals:
            raise ValueError("No RayModule fields found on pipeline object")

        try:
            out = self.dummy_run(x)
        finally:
            for name, value in originals.items():
                setattr(self, name, value)

        if isinstance(out, tuple):
            outputs = tuple(v.source for v in out)
        else:
            outputs = (out.source,)
        return nodes, outputs

    def compile(self) -> "DummyRunPipeline":
        self._compiled_nodes, self._compiled_outputs = self._build()
        return self

    def run(self, inputs: Sequence[Any]) -> List[Any]:
        if self._compiled_nodes is None or self._compiled_outputs is None:
            self.compile()
        return DagPipelineExecutor(self._compiled_nodes).run(inputs, outputs=self._compiled_outputs)
