"""Ray-backed executor for the experimental multigrain IR.

This is the first, deliberately small, Ray *lowering* of the passive
``MultigrainIR``. It proves two things the local ``MultigrainExecutor`` cannot:

1. **Intra-node replica parallelism** -- row-independent nodes (``Map`` /
   ``Filter`` / ``Expand``) are row-sharded across ``PhysicalHints.replicas`` Ray
   tasks and merged back while preserving record identity and lineage.
2. **Pipeline microbatch overlap** -- a generic driver-side DAG coordinator
   advances bounded microbatches through persistent per-node actor pools.

It reuses the local ``MultigrainExecutor`` node logic inside the Ray tasks, so
the semantics stay identical to local execution; only the scheduling changes.
The backend is available explicitly from ``rayorch.experimental.multigrain.ray``;
the root facade resolves its public symbols lazily for API compatibility.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import ray

from ..data.batch import DeferredRecord, ErrorTrace, NodeExecution, PortBatch, concat
from ..execution.coordinator import ExecutionCoordinator, GraphOutput
from ..execution.local import MultigrainExecutor
from ..execution.metrics import NodeMetric, RunMetrics
from ..ir.capabilities import capabilities_for
from ..ir.model import (
    IRNode,
    IsolationExhaustedAction,
    MultigrainIR,
    NodeKind,
    ShardRecoveryAction,
)

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
    force_inline: bool = False,
) -> tuple[NodeExecution, float]:
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
    result = executor._execute_node_result(
        node,
        tuple(inputs),
        force_inline=force_inline,
    )
    return result, time.perf_counter() - start


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
        force_inline: bool = False,
    ) -> tuple[NodeExecution, float]:
        if fail:
            raise InjectedFault(f"injected fault in node '{self._node.name}'")
        start = time.perf_counter()
        result = self._exec._execute_node_result(
            self._node,
            tuple(inputs),
            force_inline=force_inline,
        )
        return result, time.perf_counter() - start

    def ping(self) -> bool:  # readiness barrier: returns once the model is loaded
        return True


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
        max_retries: int | None = None,
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
        # Compatibility override for existing experiment scripts. New code
        # declares max_shard_retries per node through RecoveryPolicy.
        self.max_retries = (
            None if max_retries is None else max(0, int(max_retries))
        )
        # Persistent per-node actor pools for model-holding GPU stages. Created
        # lazily on first use and reused across every execute()/microbatch/chunk
        # so the model loads once per replica for the whole run.
        self._pools: dict[str, list[Any]] = {}
        self._pool_nodes: dict[str, IRNode] = {}
        self._pool_lock = threading.Lock()

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
        """Execute one input through the same coordinator used for streams."""
        results = list(
            self.execute_stream(
                graph,
                (inputs,),
                max_inflight=1,
                ordered=True,
            )
        )
        return results[0]

    def execute_stream(
        self,
        graph: MultigrainIR,
        microbatch_inputs: Iterable[Mapping[str, PortBatch]],
        *,
        max_inflight: int = 1,
        ordered: bool = True,
    ) -> Iterator[GraphOutput]:
        """Execute a bounded stream through persistent pools and a generic DAG.

        The coordinator schedules nodes when all their input ports are ready, so
        independent branches and different microbatches can overlap.  It never
        inspects workload-specific node names or kinds; sharding, retries, actor
        placement, metrics, and operator semantics stay in ``_run_node``.

        Results preserve input order by default.  Completed-but-unconsumed
        results count against ``max_inflight``, providing end-to-end backpressure
        and bounding payloads retained in Ray's object store.
        """
        unsupported = [
            node.name
            for node in graph.nodes
            if (
                node.recovery.max_record_retries > 0
                and node.kind is not NodeKind.MAP
            )
            or (
                node.recovery.decide_shard(
                    attempt=node.recovery.max_shard_retries
                )
                is ShardRecoveryAction.DEGRADE
                and not capabilities_for(node).row_partitionable
            )
        ]
        if unsupported:
            raise NotImplementedError(
                "unsupported recovery policy/node-kind combination; "
                f"nodes={unsupported}"
            )
        self.warm_pools(graph)
        coordinator = ExecutionCoordinator(
            graph,
            self._run_node,
            max_inflight=max_inflight,
            ordered=ordered,
            drain_node=self._drain_node,
        )
        return coordinator.run(microbatch_inputs)

    def _drain_node(
        self,
        node: IRNode,
        items: tuple[DeferredRecord, ...],
    ) -> NodeExecution:
        """Repack deferred singleton rows and force their record retry inline."""
        if not items:
            return NodeExecution(())
        input_count = len(items[0].inputs)
        combined: list[PortBatch] = []
        for port_index in range(input_count):
            parts = [
                self._rekey_deferred_input(item, port_index)
                for item in items
            ]
            combined.append(concat(parts, name=parts[0].name))
        return self._run_node(node, tuple(combined), force_inline=True)

    @staticmethod
    def _rekey_deferred_input(
        item: DeferredRecord,
        port_index: int,
    ) -> PortBatch:
        source = item.inputs[port_index]
        old_id = source.record_ids[0]
        ancestors = [
            {
                key: item.token if value == old_id else value
                for key, value in source.ancestors[0].items()
            }
        ]
        return PortBatch(
            name=source.name,
            values=list(source.values),
            record_ids=[item.token],
            display_keys=list(source.display_keys),
            ancestors=ancestors,
            ancestor_display=[dict(source.ancestor_display[0])],
            ordinals=[dict(source.ordinals[0])],
            lineage=[tuple(source.lineage[0])],
            relations=[tuple(source.relations[0])] if source.relations else [],
            errors=[],
        )

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
        if not capabilities_for(node).row_partitionable:
            return False
        if self._num_gpus_for(node) > 0.0:
            return True
        return self._replicas_for(node) > 1

    def _pool_for(self, node: IRNode, replicas: int) -> list[Any]:
        # execute_stream may discover independent ready branches concurrently.
        # Serialize pool construction so a node never gets duplicate actor sets.
        with self._pool_lock:
            pool = self._pools.get(node.name)
            if pool is not None:
                if self._pool_nodes[node.name] != node or len(pool) != replicas:
                    raise ValueError(
                        f"actor pool name collision for node '{node.name}'; "
                        "use a separate executor for a different graph/node recipe"
                    )
                return pool
            pool = [self._new_actor(node) for _ in range(replicas)]
            # Readiness barrier: every replica has instantiated its operator.
            ray.get([actor.ping.remote() for actor in pool])
            self._pools[node.name] = pool
            self._pool_nodes[node.name] = node
            return pool

    def _new_actor(self, node: IRNode) -> Any:
        num_gpus = self._num_gpus_for(node)
        relation_fn = self.relation_fns.get(node.name)
        actor_cls = (
            _StageActor.options(num_gpus=num_gpus) if num_gpus else _StageActor
        )
        return actor_cls.remote(node, relation_fn)

    def _replace_actor(
        self,
        node: IRNode,
        replica: int,
        failed_actor: Any,
    ) -> None:
        """Replace one dead pool member and pay model load only for that member."""
        with self._pool_lock:
            pool = self._pools.get(node.name)
            if pool is None or replica >= len(pool):
                return
            # Another concurrent microbatch may already have replaced this slot.
            # Do not kill/reload that new healthy actor a second time.
            if pool[replica] != failed_actor:
                return
            old = pool[replica]
            try:
                ray.kill(old)
            except Exception:  # noqa: BLE001 - actor may already be gone
                pass
            replacement = self._new_actor(node)
            ray.get(replacement.ping.remote())
            pool[replica] = replacement

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
        self._pool_nodes.clear()

    def _submit(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        relation_fn: Callable[[Any], Any] | None,
        fail: bool = False,
        force_inline: bool = False,
    ) -> Any:
        num_gpus = self._num_gpus_for(node)
        remote = (
            _run_node_remote.options(num_gpus=num_gpus) if num_gpus else _run_node_remote
        )
        return remote.remote(node, inputs, relation_fn, fail, force_inline)

    def _run_shards(
        self,
        node: IRNode,
        shard_inputs: list[tuple[PortBatch, ...]],
        submit_shard: Callable[
            [int, tuple[PortBatch, ...], int],
            tuple[Any, int | None, Any | None],
        ],
    ) -> tuple[list[NodeExecution], list[float], int, int]:
        """Run shards, immediately retry opaque faults, then localize if allowed."""
        n = len(shard_inputs)
        outputs: list[NodeExecution | None] = [None] * n
        busy: list[float] = [0.0] * n
        rows = [len(si[0]) if si else 0 for si in shard_inputs]
        retries = 0
        recovery_rows = 0
        pending = list(range(n))
        attempts = [0] * n
        exhausted: list[tuple[int, BaseException]] = []

        while pending:
            refs: dict[Any, tuple[int, int | None, Any | None]] = {}
            for shard_index in pending:
                ref, replica, actor = submit_shard(
                    shard_index,
                    shard_inputs[shard_index],
                    attempts[shard_index],
                )
                refs[ref] = (shard_index, replica, actor)
            retry_next: list[int] = []
            for ref, (shard_index, replica, actor) in refs.items():
                try:
                    out, elapsed = ray.get(ref)
                    outputs[shard_index] = out
                    busy[shard_index] += elapsed
                except Exception as exc:  # noqa: BLE001 - shard policy classifies
                    if replica is not None and self._is_actor_failure(exc):
                        self._replace_actor(node, replica, actor)
                    action = node.recovery.decide_shard(
                        attempt=attempts[shard_index]
                    )
                    if (
                        self.max_retries is not None
                        and attempts[shard_index] < self.max_retries
                    ):
                        action = ShardRecoveryAction.RETRY
                    elif (
                        self.max_retries is not None
                        and attempts[shard_index] >= self.max_retries
                    ):
                        action = (
                            ShardRecoveryAction.DEGRADE
                            if node.recovery.on_shard_exhausted.value == "degrade"
                            else ShardRecoveryAction.ABORT
                        )
                    if action is ShardRecoveryAction.ABORT:
                        raise
                    if action is ShardRecoveryAction.DEGRADE:
                        exhausted.append((shard_index, exc))
                        continue
                    attempts[shard_index] += 1
                    retries += 1
                    recovery_rows += rows[shard_index]
                    retry_next.append(shard_index)
            pending = retry_next

        for shard_index, error in exhausted:
            localized, elapsed, calls, work = self._localize_shard(
                node,
                shard_index,
                shard_inputs[shard_index],
                submit_shard,
                attempts[shard_index] + 1,
                error,
            )
            outputs[shard_index] = localized
            busy[shard_index] += elapsed
            retries += calls
            recovery_rows += work

        return [out for out in outputs if out is not None], busy, retries, recovery_rows

    @staticmethod
    def _is_actor_failure(exc: BaseException) -> bool:
        actor_error = getattr(ray.exceptions, "RayActorError", ())
        if actor_error and isinstance(exc, actor_error):
            return True
        name = type(exc).__name__
        return "ActorDied" in name or "RayActorError" in name

    def _localize_shard(
        self,
        node: IRNode,
        shard_index: int,
        inputs: tuple[PortBatch, ...],
        submit_shard: Callable[
            [int, tuple[PortBatch, ...], int],
            tuple[Any, int | None, Any | None],
        ],
        attempt_base: int,
        root_error: BaseException,
    ) -> tuple[NodeExecution, float, int, int]:
        """Bisect an opaque failed shard under a hard row-work/call budget."""
        original_rows = len(inputs[0]) if inputs else 0
        budget = node.recovery.isolation
        max_work = budget.max_work_factor * original_rows
        calls = 0
        work = 0
        busy = 0.0
        leaves: list[tuple[tuple[int, ...], NodeExecution]] = []
        frontier: list[
            tuple[tuple[int, ...], tuple[PortBatch, ...], BaseException]
        ] = [((), inputs, root_error)]

        while frontier:
            submissions: dict[
                Any,
                tuple[
                    tuple[int, ...],
                    tuple[PortBatch, ...],
                    int | None,
                    Any | None,
                ],
            ] = {}
            next_frontier: list[
                tuple[tuple[int, ...], tuple[PortBatch, ...], BaseException]
            ] = []
            for path, subset, error in frontier:
                size = len(subset[0]) if subset else 0
                if size <= 1:
                    leaves.append(
                        (path, self._quarantine_outputs(node, subset, error))
                    )
                    continue
                midpoint = size // 2
                for branch, indices in enumerate(
                    (range(0, midpoint), range(midpoint, size))
                ):
                    child = tuple(port.take(indices) for port in subset)
                    child_rows = len(child[0])
                    child_path = (*path, branch)
                    if (
                        calls >= budget.max_calls
                        or work + child_rows > max_work
                    ):
                        if (
                            budget.on_exhausted
                            is IsolationExhaustedAction.ABORT
                        ):
                            raise InjectedFault(
                                f"node '{node.name}' adaptive isolation budget "
                                f"exhausted after {calls} calls/{work} rows"
                            ) from error
                        leaves.append(
                            (
                                child_path,
                                self._quarantine_outputs(node, child, error),
                            )
                        )
                        continue
                    ref, replica, actor = submit_shard(
                        shard_index,
                        child,
                        attempt_base + calls,
                    )
                    submissions[ref] = (child_path, child, replica, actor)
                    calls += 1
                    work += child_rows

            for ref, (path, child, replica, actor) in submissions.items():
                try:
                    out, elapsed = ray.get(ref)
                    busy += elapsed
                    leaves.append((path, out))
                except Exception as exc:  # noqa: BLE001 - continue localization
                    if replica is not None and self._is_actor_failure(exc):
                        self._replace_actor(node, replica, actor)
                    next_frontier.append((path, child, exc))
            frontier = next_frontier

        leaves.sort(key=lambda item: item[0])
        return self._merge_leaf_outputs(node, leaves), busy, calls, work

    def _quarantine_outputs(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        error: BaseException,
    ) -> NodeExecution:
        base = inputs[0]
        traces = list(base.errors)
        for index, record_id in enumerate(base.record_ids):
            ancestors = dict(base.ancestors[index])
            ancestors[base.name] = record_id
            traces.append(
                ErrorTrace(
                    source_item=base.display_keys[index],
                    logical_item=base.display_keys[index],
                    failed_op=node.name,
                    grain=base.name,
                    upstream_path=tuple((*base.lineage[index], node.name)),
                    parent=base.display_keys[index],
                    action="quarantined_shard_exhausted",
                    error=str(error),
                    ancestors=ancestors,
                )
            )
        return NodeExecution(
            tuple(
                PortBatch(
                    name=spec.name or spec.ref.port,
                    values=[],
                    record_ids=[],
                    display_keys=[],
                    ancestors=[],
                    ancestor_display=[],
                    ordinals=[],
                    lineage=[],
                    errors=list(traces),
                )
                for spec in node.output_specs
            )
        )

    @staticmethod
    def _merge_leaf_outputs(
        node: IRNode,
        leaves: Sequence[tuple[tuple[int, ...], NodeExecution]],
    ) -> NodeExecution:
        merged: list[PortBatch] = []
        for output_index, spec in enumerate(node.output_specs):
            parts = [result.outputs[output_index] for _, result in leaves]
            merged.append(concat(parts, name=parts[0].name if parts else spec.name))
        deferred = tuple(
            item
            for _, result in leaves
            for item in result.deferred
        )
        return NodeExecution(tuple(merged), deferred)

    def _run_node(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool = False,
    ) -> NodeExecution:
        relation_fn = self.relation_fns.get(node.name)
        fault = self._fault_for(node.name)
        replicas = (
            self._replicas_for(node)
            if capabilities_for(node).row_partitionable
            else 1
        )
        nrows = len(inputs[0]) if inputs else 0

        if replicas <= 1 or nrows <= 1:
            # Even the single-shard path uses the persistent pool for GPU stages
            # so the model is not reloaded per call.
            if self._use_pool(node):
                pool = self._pool_for(node, replicas=1)
                def submit(s, shard, attempt):
                    fail = (
                        fault.should_fail(node.name, s, attempt)
                        if fault
                        else False
                    )
                    actor = pool[0]
                    return actor.run_shard.remote(
                        shard,
                        fail,
                        force_inline,
                    ), 0, actor
            else:
                def submit(s, shard, attempt):
                    fail = (
                        fault.should_fail(node.name, s, attempt)
                        if fault
                        else False
                    )
                    return self._submit(
                        node,
                        shard,
                        relation_fn,
                        fail,
                        force_inline,
                    ), None, None
            shard_outputs, shard_busy, retries, rec_rows = self._run_shards(
                node, [inputs], submit
            )
            result = shard_outputs[0]
            self._record(
                node,
                inputs,
                result.outputs,
                shard_busy,
                1,
                retries,
                rec_rows,
            )
            return result

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
            def submit(s, shard, attempt):
                replica = (s + attempt) % len(pool)
                fail = (
                    fault.should_fail(node.name, s, attempt)
                    if fault
                    else False
                )
                actor = pool[replica]
                return actor.run_shard.remote(
                    shard,
                    fail,
                    force_inline,
                ), replica, actor
        else:
            def submit(s, shard, attempt):
                fail = (
                    fault.should_fail(node.name, s, attempt)
                    if fault
                    else False
                )
                return self._submit(
                    node,
                    shard,
                    relation_fn,
                    fail,
                    force_inline,
                ), None, None
        shard_outputs, shard_busy, retries, rec_rows = self._run_shards(
            node, shard_inputs, submit
        )

        merged: list[PortBatch] = []
        for output_index in range(len(node.output_refs)):
            parts = [result.outputs[output_index] for result in shard_outputs]
            merged.append(concat(parts, name=parts[0].name))
        deferred = tuple(
            item
            for result in shard_outputs
            for item in result.deferred
        )
        self._record(
            node, inputs, tuple(merged), shard_busy, len(partitions), retries, rec_rows
        )
        return NodeExecution(tuple(merged), deferred)

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
        """Compatibility list API; delegates to the persistent streaming path."""
        return list(
            self.execute_stream(
                graph,
                microbatch_inputs,
                max_inflight=max_inflight,
                ordered=True,
            )
        )


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
