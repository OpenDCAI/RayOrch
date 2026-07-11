"""Ray-backed executor for the experimental multigrain IR (parallelism MVP).

This is the first, deliberately small, Ray *lowering* of the passive
``MultigrainIR``. It proves two things the local ``MultigrainExecutor`` cannot:

1. **Intra-node replica parallelism** -- row-independent nodes (``Map`` /
   ``Filter`` / ``Expand``) are row-sharded across ``PhysicalHints.replicas`` Ray
   tasks and merged back while preserving record identity and lineage.
2. **Pipeline microbatch overlap** -- whole-graph execution is launched per
   microbatch as Ray tasks with a bounded in-flight window.

It reuses the local ``MultigrainExecutor`` node logic inside the Ray tasks, so
the semantics stay identical to local execution; only the scheduling changes.
This executor is intentionally not part of the narrow package ``__init__``:
import it explicitly from ``rayorch.experimental.multigrain.ray_executor``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import ray

from .core import PortBatch, concat
from .executor import MultigrainExecutor
from .graph import IRNode, IRPortRef, MultigrainIR, NodeKind
from .metrics import NodeMetric, RunMetrics

# Node kinds whose rows/parents are independent and therefore safe to row-shard.
# Reduce/Relate need cross-row context, so they run on a single task in the MVP.
_SHARDABLE = frozenset({NodeKind.MAP, NodeKind.FILTER, NodeKind.EXPAND})


class InjectedFault(RuntimeError):
    """Raised inside a Ray task to simulate a task/node crash (fault injection)."""


def _is_injected_fault(exc: BaseException) -> bool:
    """True if ``exc`` (or its Ray-wrapped cause) is an :class:`InjectedFault`."""
    if isinstance(exc, InjectedFault):
        return True
    cause = getattr(exc, "cause", None)
    if isinstance(cause, InjectedFault):
        return True
    # Ray's RayTaskError re-raises as an instanceof the original type when it can;
    # fall back to name matching for the wrapped case.
    return "InjectedFault" in type(exc).__name__ or "InjectedFault" in str(exc)


@dataclass
class FaultSpec:
    """Deterministic fault injection for recovery experiments (M2 metric #3/E3).

    Simulates a *task/node* crash: the selected shards of ``node`` raise on early
    attempts and succeed on retry, so we can measure that recovery recomputes only
    the failed shard's rows (lineage-local) rather than the whole stage.
    """

    node: str
    fail_shards: frozenset[int] = field(default_factory=frozenset)
    fail_until_attempt: int = 1  # fail while attempt < this (1 => only attempt 0)

    def should_fail(self, node_name: str, shard_index: int, attempt: int) -> bool:
        return (
            node_name == self.node
            and shard_index in self.fail_shards
            and attempt < self.fail_until_attempt
        )


@ray.remote
def _run_node_remote(
    node: IRNode,
    inputs: Sequence[PortBatch],
    relation_fn: Callable[[Any], Any] | None,
    fail: bool = False,
) -> tuple[tuple[PortBatch, ...], float]:
    """Execute a single IR node (or one shard of it) inside a *stateless* Ray task.

    Used for cheap, model-free stages (e.g. CPU Expand/Filter). Returns
    ``(outputs, busy_seconds)`` so the driver can compute per-stage idle bubble.
    If ``fail`` is set, raises before doing work to emulate a crash.
    """
    if fail:
        raise InjectedFault(f"injected fault in node '{node.name}'")
    relation_fns = {node.name: relation_fn} if relation_fn is not None else None
    executor = MultigrainExecutor(relation_fns=relation_fns)
    start = time.perf_counter()
    outputs = executor._execute_node(node, tuple(inputs))
    return outputs, time.perf_counter() - start


@ray.remote
class _StageActor:
    """Persistent per-replica worker for a model-holding stage (RayModule-style).

    Mirrors :class:`rayorch.ray_module.RunnerActor`: the operator is instantiated
    exactly once at actor construction (its ``__init__`` loads the model), then
    reused across every shard and every microbatch/chunk routed to this actor.
    The op class + init args come from the node's ``OperatorRecipe``, so the user
    only writes ``__init__`` + ``run`` and declares ``PhysicalHints`` -- no driver
    ever instantiates the model.
    """

    def __init__(
        self,
        node: IRNode,
        relation_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        relation_fns = {node.name: relation_fn} if relation_fn is not None else None
        self._node = node
        self._exec = MultigrainExecutor(relation_fns=relation_fns)
        self._exec.warm(node)  # pay model-load cost now, not on the first shard

    def run_shard(
        self,
        inputs: Sequence[PortBatch],
        fail: bool = False,
    ) -> tuple[tuple[PortBatch, ...], float]:
        if fail:
            raise InjectedFault(f"injected fault in node '{self._node.name}'")
        start = time.perf_counter()
        outputs = self._exec._execute_node(self._node, tuple(inputs))
        return outputs, time.perf_counter() - start

    def ping(self) -> bool:  # readiness barrier: returns once the model is loaded
        return True


@ray.remote
def _run_graph_remote(
    graph: MultigrainIR,
    inputs: Mapping[str, PortBatch],
    relation_fns: Mapping[str, Callable[[Any], Any]] | None,
) -> Any:
    """Execute a whole graph locally inside a Ray task (one microbatch)."""
    return MultigrainExecutor(relation_fns=relation_fns).execute(graph, dict(inputs))


def _contiguous_ranges(total: int, parts: int) -> list[range]:
    if total <= 0:
        return [range(0, 0)]
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    ranges: list[range] = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


class MultigrainRayExecutor:
    """Execute a ``MultigrainIR`` on Ray with replica and microbatch parallelism."""

    def __init__(
        self,
        *,
        relation_fns: Mapping[str, Callable[[Any], Any]] | None = None,
        default_replicas: int = 1,
        shard_planner: Callable[
            [IRNode, Sequence[PortBatch], int], Sequence[Sequence[int]] | None
        ]
        | None = None,
        metrics: RunMetrics | None = None,
        faults: Sequence[FaultSpec] | None = None,
        max_retries: int = 2,
    ) -> None:
        self.relation_fns = dict(relation_fns or {})
        self.default_replicas = max(1, int(default_replicas))
        # Optional policy: given (node, inputs, replicas) -> per-shard row-index
        # lists. Returning None falls back to contiguous ranges. This is where
        # relation-aware / work-aware rebalancing plugs in.
        self.shard_planner = shard_planner
        # Optional instrumentation + fault injection for M2 experiments.
        self.metrics = metrics
        self.faults = list(faults or [])
        self.max_retries = max(0, int(max_retries))
        # Persistent per-node actor pools for model-holding GPU stages. Created
        # lazily on first use and reused across every execute()/microbatch/chunk
        # so the model loads once per replica for the whole run.
        self._pools: dict[str, list[Any]] = {}

    def _fault_for(self, node_name: str) -> FaultSpec | None:
        for spec in self.faults:
            if spec.node == node_name:
                return spec
        return None

    # -- single-graph execution with intra-node replica parallelism ----------
    def execute(
        self,
        graph: MultigrainIR,
        inputs: Mapping[str, PortBatch],
    ) -> PortBatch | tuple[PortBatch, ...]:
        context: dict[IRPortRef, PortBatch] = {}
        for spec in graph.inputs:
            if spec.name not in inputs:
                raise KeyError(f"missing input port '{spec.name}'")
            context[spec.ref] = inputs[spec.name]

        for node in graph.nodes:
            node_inputs = tuple(context[ref] for ref in node.input_refs)
            outputs = self._run_node(node, node_inputs)
            if len(outputs) != len(node.output_refs):
                raise ValueError(
                    f"node {node.name} produced {len(outputs)} outputs, "
                    f"expected {len(node.output_refs)}"
                )
            for ref, batch in zip(node.output_refs, outputs):
                context[ref] = batch

        result = tuple(context[ref] for ref in graph.graph_outputs)
        return result[0] if len(result) == 1 else result

    def _replicas_for(self, node: IRNode) -> int:
        hint = node.physical.replicas if node.physical else 1
        return max(self.default_replicas, int(hint or 1))

    def _num_gpus_for(self, node: IRNode) -> float:
        return float(node.physical.num_gpus_per_replica if node.physical else 0.0)

    def _use_pool(self, node: IRNode) -> bool:
        """Run on a persistent actor pool when it pays off.

        * GPU / model-holding stages: always (never reload the model).
        * CPU stages we parallelize (replicas > 1): also pool them, so heavy
          imports in the op module (e.g. torch/vLLM pulled in transitively) happen
          once per replica instead of on every chunk's stateless task.

        Everything else stays on cheap stateless tasks.
        """
        if node.kind not in _SHARDABLE:
            return False
        if self._num_gpus_for(node) > 0.0:
            return True
        return self._replicas_for(node) > 1

    def _pool_for(self, node: IRNode, replicas: int) -> list[Any]:
        pool = self._pools.get(node.name)
        if pool is not None:
            return pool
        num_gpus = self._num_gpus_for(node)
        relation_fn = self.relation_fns.get(node.name)
        actor_cls = _StageActor.options(num_gpus=num_gpus) if num_gpus else _StageActor
        pool = [actor_cls.remote(node, relation_fn) for _ in range(replicas)]
        # Block until every replica has loaded its model, so the first shard's
        # timing is cold-start free and GPUs are actually reserved.
        ray.get([actor.ping.remote() for actor in pool])
        self._pools[node.name] = pool
        return pool

    def warm_pools(self, graph: MultigrainIR) -> None:
        """Pre-create persistent actor pools (loading their models) before timing.

        Lets a benchmark exclude one-time model-load cost from wall time, matching
        baselines that load the model in their engine constructor.
        """
        for node in graph.nodes:
            if self._use_pool(node):
                self._pool_for(node, self._replicas_for(node))

    def shutdown(self) -> None:
        """Release all persistent actors (and their GPUs)."""
        for pool in self._pools.values():
            for actor in pool:
                ray.kill(actor)
        self._pools.clear()

    def _submit(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        relation_fn: Callable[[Any], Any] | None,
        fail: bool = False,
    ) -> Any:
        num_gpus = self._num_gpus_for(node)
        remote = (
            _run_node_remote.options(num_gpus=num_gpus) if num_gpus else _run_node_remote
        )
        return remote.remote(node, inputs, relation_fn, fail)

    def _run_shards(
        self,
        node: IRNode,
        shard_inputs: list[tuple[PortBatch, ...]],
        submit_shard: Callable[[int, bool], Any],
        fault: FaultSpec | None,
    ) -> tuple[list[tuple[PortBatch, ...]], list[float], int, int]:
        """Run all shards *concurrently*, retrying only the ones that fault.

        Shards are submitted together and awaited together so replicas actually
        overlap; on an injected/task fault we resubmit just the failed shard(s),
        which is why recovery work stays lineage-local (only those shards' rows).
        ``submit_shard(shard_index, fail) -> ObjectRef`` decouples scheduling
        (stateless task vs persistent actor) from the retry loop.
        Returns ``(ordered_outputs, per_shard_busy, retries, recovery_rows)``.
        """
        n = len(shard_inputs)
        outputs: list[tuple[PortBatch, ...] | None] = [None] * n
        busy: list[float] = [0.0] * n
        rows = [len(si[0]) if si else 0 for si in shard_inputs]
        retries = 0
        recovery_rows = 0

        pending = list(range(n))
        attempt = 0
        while pending:
            refs = {
                submit_shard(
                    s,
                    fault.should_fail(node.name, s, attempt) if fault else False,
                ): s
                for s in pending
            }
            failed: list[int] = []
            for ref, shard_index in refs.items():
                try:
                    out, elapsed = ray.get(ref)
                    outputs[shard_index] = out
                    busy[shard_index] = elapsed
                except Exception as exc:  # noqa: BLE001 - inspect for injected fault
                    if not _is_injected_fault(exc):
                        raise
                    failed.append(shard_index)
                    retries += 1
                    # Recovery recomputes only this shard's rows -- lineage-local.
                    recovery_rows += rows[shard_index]
            attempt += 1
            if failed and attempt > self.max_retries:
                raise InjectedFault(
                    f"node '{node.name}' shards {failed} exhausted "
                    f"{self.max_retries} retries"
                )
            pending = failed

        return [out for out in outputs if out is not None], busy, retries, recovery_rows

    def _run_node(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
    ) -> tuple[PortBatch, ...]:
        relation_fn = self.relation_fns.get(node.name)
        fault = self._fault_for(node.name)
        replicas = self._replicas_for(node) if node.kind in _SHARDABLE else 1
        nrows = len(inputs[0]) if inputs else 0

        if replicas <= 1 or nrows <= 1:
            # Even the single-shard path uses the persistent pool for GPU stages
            # so the model is not reloaded per call.
            if self._use_pool(node):
                pool = self._pool_for(node, replicas=1)
                submit = lambda s, fail, _si=[inputs]: pool[0].run_shard.remote(_si[s], fail)  # noqa: E731
            else:
                submit = lambda s, fail, _si=[inputs]: self._submit(  # noqa: E731
                    node, _si[s], relation_fn, fail
                )
            shard_outputs, shard_busy, retries, rec_rows = self._run_shards(
                node, [inputs], submit, fault
            )
            merged = shard_outputs[0]
            self._record(node, inputs, merged, shard_busy, 1, retries, rec_rows)
            return tuple(merged)

        partitions: Sequence[Sequence[int]] | None = None
        if self.shard_planner is not None:
            partitions = self.shard_planner(node, inputs, replicas)
        if partitions is None:
            partitions = [list(rng) for rng in _contiguous_ranges(nrows, replicas)]
        partitions = [list(idx) for idx in partitions if len(idx) > 0]

        shard_inputs = [
            tuple(port.take(idx) for port in inputs) for idx in partitions
        ]
        if self._use_pool(node):
            pool = self._pool_for(node, replicas=replicas)
            submit = lambda s, fail: pool[s].run_shard.remote(shard_inputs[s], fail)  # noqa: E731
        else:
            submit = lambda s, fail: self._submit(  # noqa: E731
                node, shard_inputs[s], relation_fn, fail
            )
        shard_outputs, shard_busy, retries, rec_rows = self._run_shards(
            node, shard_inputs, submit, fault
        )

        merged: list[PortBatch] = []
        for output_index in range(len(node.output_refs)):
            parts = [shard[output_index] for shard in shard_outputs]
            merged.append(concat(parts, name=parts[0].name))
        self._record(
            node, inputs, tuple(merged), shard_busy, len(partitions), retries, rec_rows
        )
        return tuple(merged)

    def _record(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        outputs: Sequence[PortBatch],
        shard_busy: list[float],
        replicas: int,
        retries: int,
        recovery_rows: int,
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.record(
            NodeMetric(
                name=node.name,
                kind=node.kind.value,
                replicas=replicas,
                rows_in=len(inputs[0]) if inputs else 0,
                rows_out=len(outputs[0]) if outputs else 0,
                wall_s=max(shard_busy) if shard_busy else 0.0,
                shard_busy_s=list(shard_busy),
                retries=retries,
                recovery_rows=recovery_rows,
            )
        )

    # -- multi-microbatch execution with bounded overlap ---------------------
    def execute_microbatches(
        self,
        graph: MultigrainIR,
        microbatch_inputs: Sequence[Mapping[str, PortBatch]],
        *,
        max_inflight: int = 1,
    ) -> list[Any]:
        total = len(microbatch_inputs)
        results: list[Any] = [None] * total
        window = max(1, int(max_inflight))
        relation_fns = self.relation_fns or None

        pending: dict[Any, int] = {}
        next_index = 0
        while next_index < total or pending:
            while next_index < total and len(pending) < window:
                ref = _run_graph_remote.remote(
                    graph, microbatch_inputs[next_index], relation_fns
                )
                pending[ref] = next_index
                next_index += 1
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            for ref in done:
                results[pending.pop(ref)] = ray.get(ref)
        return results


def lpt_shard_planner(
    weight_of: Callable[[Any], float],
) -> Callable[[IRNode, Sequence[PortBatch], int], list[list[int]]]:
    """Work-aware shard planner (Longest-Processing-Time greedy bin packing).

    Given a per-row weight (e.g. a page's content length), it balances *total
    work* per shard instead of row count, so imbalanced 1:N fan-outs stop
    creating idle GPUs ("bubbles"). Row identity/ordinals are preserved by the
    executor's ``PortBatch.take``, and any downstream ``Reduce`` restores logical
    order via ordinals, so reordering rows across shards is safe.
    """

    def plan(node: IRNode, inputs: Sequence[PortBatch], replicas: int) -> list[list[int]]:
        base = inputs[0]
        weights = [float(weight_of(value)) for value in base.values]
        order = sorted(range(len(weights)), key=lambda i: -weights[i])
        bins: list[list[int]] = [[] for _ in range(replicas)]
        loads = [0.0] * replicas
        for i in order:
            target = min(range(replicas), key=lambda k: loads[k])
            bins[target].append(i)
            loads[target] += weights[i]
        return bins

    return plan


__all__ = [
    "FaultSpec",
    "InjectedFault",
    "MultigrainRayExecutor",
    "lpt_shard_planner",
]
