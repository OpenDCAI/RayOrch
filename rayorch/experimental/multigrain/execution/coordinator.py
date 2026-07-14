"""Generic driver-side coordinator for streaming passive-IR execution.

The coordinator owns only control flow: microbatch admission, DAG readiness,
bounded in-flight work, and ordered result delivery.  It deliberately knows
nothing about Ray actors, sharding, retries, or user operators.  Those remain
behind the injected ``run_node`` callback, so the same IR scheduler can evolve
without creating a workload-specific execution path.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping

from ..data.batch import DeferredRecord, ErrorTrace, NodeExecution, PortBatch, concat
from ..ir.model import IRNode, IRPortRef, MultigrainIR


GraphOutput = PortBatch | tuple[PortBatch, ...]
RunNode = Callable[[IRNode, tuple[PortBatch, ...]], NodeExecution]
DrainNode = Callable[[IRNode, tuple[DeferredRecord, ...]], NodeExecution]


@dataclass
class _MicrobatchState:
    index: int
    context: dict[IRPortRef, PortBatch]
    pending: dict[str, IRNode]
    running: set[str] = field(default_factory=set)

    @property
    def done(self) -> bool:
        return not self.pending and not self.running


@dataclass
class _DeferredCompletion:
    state: _MicrobatchState
    node: IRNode
    inputs: tuple[PortBatch, ...]
    result: NodeExecution


class ExecutionCoordinator:
    """Schedule arbitrary DAG nodes across bounded in-flight microbatches.

    Each node invocation runs in a driver thread because the Ray-backed
    ``run_node`` callback waits for remote shard results.  Persistent actors and
    payloads still live in Ray; the threads only coordinate ObjectRefs/results.
    Independent branches and different microbatches may progress concurrently.
    """

    def __init__(
        self,
        graph: MultigrainIR,
        run_node: RunNode,
        *,
        max_inflight: int,
        ordered: bool,
        drain_node: DrainNode | None = None,
    ) -> None:
        self.graph = graph
        self.run_node = run_node
        self.max_inflight = max(1, int(max_inflight))
        self.ordered = bool(ordered)
        self.drain_node = drain_node

    def run(
        self,
        microbatch_inputs: Iterable[Mapping[str, PortBatch]],
    ) -> Iterator[GraphOutput]:
        source = enumerate(iter(microbatch_inputs))
        active: dict[int, _MicrobatchState] = {}
        futures: dict[
            Future[NodeExecution],
            tuple[int, IRNode, tuple[PortBatch, ...]],
        ] = {}
        deferred_by_node: dict[str, list[_DeferredCompletion]] = {}
        ready: dict[int, GraphOutput] = {}
        completion_order: list[int] = []
        next_yield = 0
        source_exhausted = False
        max_workers = self.max_inflight * max(1, min(len(self.graph.nodes), 4))

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="multigrain-stage",
        ) as threads:

            def schedule_ready(state: _MicrobatchState) -> None:
                for name, node in list(state.pending.items()):
                    if not all(ref in state.context for ref in node.input_refs):
                        continue
                    inputs = tuple(state.context[ref] for ref in node.input_refs)
                    del state.pending[name]
                    state.running.add(name)
                    future = threads.submit(self.run_node, node, inputs)
                    futures[future] = (state.index, node, inputs)

            def finish_if_done(state: _MicrobatchState) -> None:
                if not state.done:
                    return
                missing = [ref for ref in self.graph.graph_outputs if ref not in state.context]
                if missing:
                    raise RuntimeError(
                        f"microbatch {state.index} finished without graph outputs {missing}"
                    )
                outputs = tuple(state.context[ref] for ref in self.graph.graph_outputs)
                ready[state.index] = outputs[0] if len(outputs) == 1 else outputs
                if not self.ordered:
                    completion_order.append(state.index)
                del active[state.index]

            def admit() -> None:
                nonlocal source_exhausted
                # Completed-but-not-yet-consumed results still count toward the
                # window; otherwise one slow early microbatch could let an
                # unbounded number of later outputs accumulate behind it.
                while (
                    not source_exhausted
                    and len(active) + len(ready) < self.max_inflight
                ):
                    try:
                        index, inputs = next(source)
                    except StopIteration:
                        source_exhausted = True
                        break
                    context: dict[IRPortRef, PortBatch] = {}
                    for spec in self.graph.inputs:
                        if spec.name not in inputs:
                            raise KeyError(
                                f"microbatch {index} missing input port '{spec.name}'"
                            )
                        context[spec.ref] = inputs[spec.name]
                    state = _MicrobatchState(
                        index=index,
                        context=context,
                        pending={node.name: node for node in self.graph.nodes},
                    )
                    active[index] = state
                    schedule_ready(state)
                    finish_if_done(state)

            def finalize_node(
                completion: _DeferredCompletion,
                outputs: tuple[PortBatch, ...],
            ) -> None:
                state = completion.state
                node = completion.node
                for ref, batch in zip(node.output_refs, outputs):
                    state.context[ref] = batch
                state.running.remove(node.name)
                schedule_ready(state)
                finish_if_done(state)

            def terminal_trace(item: DeferredRecord) -> ErrorTrace:
                base = item.inputs[0]
                ancestors = dict(base.ancestors[0])
                ancestors[base.name] = base.record_ids[0]
                return ErrorTrace(
                    source_item=base.display_keys[0],
                    logical_item=base.display_keys[0],
                    failed_op=item.failed_op,
                    grain=base.name,
                    upstream_path=tuple((*base.lineage[0], item.failed_op)),
                    parent=base.display_keys[0],
                    action="quarantined_deferred_exhausted",
                    error=item.error,
                    ancestors=ancestors,
                )

            def merge_completion(
                completion: _DeferredCompletion,
                recovered: NodeExecution,
            ) -> tuple[PortBatch, ...]:
                original_order = completion.inputs[0].record_ids
                merged_outputs: list[PortBatch] = []
                for output_index, healthy in enumerate(completion.result.outputs):
                    recovery_output = recovered.outputs[output_index]
                    token_to_index = {
                        record_id: index
                        for index, record_id in enumerate(
                            recovery_output.record_ids
                        )
                    }
                    pieces = [healthy]
                    terminal: list[ErrorTrace] = []
                    for item in completion.result.deferred:
                        recovered_index = token_to_index.get(item.token)
                        if recovered_index is None:
                            terminal.append(terminal_trace(item))
                            continue
                        piece = item.inputs[0].with_values(
                            [recovery_output.values[recovered_index]],
                            name=healthy.name,
                            op_name=completion.node.name,
                        )
                        piece.errors = []
                        pieces.append(piece)
                    merged = concat(pieces, name=healthy.name)
                    positions = {
                        record_id: index
                        for index, record_id in enumerate(original_order)
                    }
                    order = sorted(
                        range(len(merged)),
                        key=lambda index: positions[merged.record_ids[index]],
                    )
                    merged = merged.take(order)
                    merged.errors.extend(terminal)
                    merged_outputs.append(merged)
                return tuple(merged_outputs)

            def drain_stage(node_name: str) -> None:
                completions = deferred_by_node.pop(node_name)
                if self.drain_node is None:
                    raise NotImplementedError(
                        "deferred records require a stage drain callback"
                    )
                node = completions[0].node
                items = tuple(
                    item
                    for completion in completions
                    for item in completion.result.deferred
                )
                recovered = self.drain_node(node, items)
                for completion in completions:
                    finalize_node(
                        completion,
                        merge_completion(completion, recovered),
                    )

            def drain_ready(*, force: bool = False) -> bool:
                drained = False
                for node_name, completions in list(deferred_by_node.items()):
                    items = [
                        item
                        for completion in completions
                        for item in completion.result.deferred
                    ]
                    rows = len(items)
                    target = min(
                        (max(1, item.target_rows) for item in items),
                        default=1,
                    )
                    node_running = any(
                        node.name == node_name
                        for _, node, _ in futures.values()
                    )
                    if force or rows >= target or not node_running:
                        drain_stage(node_name)
                        drained = True
                return drained

            admit()
            while active or futures or ready or not source_exhausted:
                # Refill before yielding so consumer-side work does not starve
                # already-admissible microbatches.
                admit()

                if self.ordered:
                    if next_yield in ready:
                        result = ready.pop(next_yield)
                        next_yield += 1
                        yield result
                        continue
                elif completion_order:
                    index = completion_order.pop(0)
                    if index in ready:
                        yield ready.pop(index)
                        continue

                if not futures:
                    if deferred_by_node and drain_ready(force=True):
                        continue
                    if active:
                        blocked = {
                            index: list(state.pending)
                            for index, state in active.items()
                        }
                        raise RuntimeError(
                            "execution deadlock: no runnable nodes; "
                            f"pending by microbatch={blocked}"
                        )
                    continue

                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    index, node, inputs = futures.pop(future)
                    state = active[index]
                    result = future.result()
                    if len(result.outputs) != len(node.output_refs):
                        raise ValueError(
                            f"node {node.name} produced "
                            f"{len(result.outputs)} outputs, "
                            f"expected {len(node.output_refs)}"
                        )
                    if result.deferred:
                        deferred_by_node.setdefault(node.name, []).append(
                            _DeferredCompletion(state, node, inputs, result)
                        )
                    else:
                        finalize_node(
                            _DeferredCompletion(state, node, inputs, result),
                            result.outputs,
                        )
                drain_ready(force=source_exhausted)


__all__ = ["ExecutionCoordinator", "GraphOutput"]
