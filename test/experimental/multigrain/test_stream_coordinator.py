from __future__ import annotations

import time

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.execution.coordinator import ExecutionCoordinator
from rayorch.experimental.multigrain.data.batch import (
    DeferredRecord,
    NodeExecution,
    concat,
)


class Identity:
    def run(self, rows):
        return rows


class IdentityPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.identity = mg.Map(Identity)

    def forward(self, rows):
        return self.identity(rows)


def _delay_from_value(_node, inputs):
    batch = inputs[0]
    time.sleep(float(batch.values[0]))
    return NodeExecution((batch,))


def test_completed_outputs_remain_inside_backpressure_window() -> None:
    graph = IdentityPipe().compile()
    admitted = 0

    def inputs():
        nonlocal admitted
        for value in (0.01, 0.01, 0.01, 0.01):
            admitted += 1
            yield {"rows": mg.source([value], name="rows")}

    results = ExecutionCoordinator(
        graph,
        _delay_from_value,
        max_inflight=2,
        ordered=True,
    ).run(inputs())

    first = next(results)
    assert first.values == [0.01]
    assert admitted == 2
    results.close()


def test_unordered_mode_yields_the_first_completed_microbatch() -> None:
    graph = IdentityPipe().compile()
    batches = [
        {"rows": mg.source([0.10], name="rows")},
        {"rows": mg.source([0.01], name="rows")},
    ]

    outputs = list(
        ExecutionCoordinator(
            graph,
            _delay_from_value,
            max_inflight=2,
            ordered=False,
        ).run(batches)
    )

    assert [batch.values for batch in outputs] == [[0.01], [0.10]]


def _run_epoch_case(values, *, max_inflight: int) -> list[int]:
    graph = IdentityPipe().compile()
    drain_sizes: list[int] = []

    def run_node(node, inputs):
        delay, label, target = inputs[0].values[0]
        time.sleep(delay)
        output_name = node.output_specs[0].name or node.name
        empty = inputs[0].take([], name=output_name)
        item = DeferredRecord(
            token=f"token-{label}",
            inputs=inputs,
            failed_op=node.name,
            error="temporary",
            target_rows=target,
        )
        return NodeExecution((empty,), (item,))

    def drain_node(node, items):
        drain_sizes.append(len(items))
        output_name = node.output_specs[0].name or node.name
        parts = []
        for item in items:
            label = item.inputs[0].values[0][1]
            part = item.inputs[0].with_values(
                [f"ok:{label}"],
                name=output_name,
                op_name=node.name,
            )
            part.record_ids = [item.token]
            parts.append(part)
        return NodeExecution((concat(parts, name=parts[0].name),))

    inputs = (
        {"rows": mg.source([value], name="rows")}
        for value in values
    )
    outputs = list(
        ExecutionCoordinator(
            graph,
            run_node,
            max_inflight=max_inflight,
            ordered=True,
            drain_node=drain_node,
        ).run(inputs)
    )
    assert [output.values[0] for output in outputs] == [
        f"ok:{value[1]}" for value in values
    ]
    return drain_sizes


def test_stage_epoch_drains_at_target_row_threshold() -> None:
    sizes = _run_epoch_case(
        [(0.01, "a", 2), (0.03, "b", 2), (0.01, "c", 2)],
        max_inflight=2,
    )
    assert sizes[0] == 2


def test_stage_epoch_drains_when_stage_becomes_quiescent() -> None:
    sizes = _run_epoch_case(
        [(0.01, "a", 10), (0.01, "b", 10)],
        max_inflight=1,
    )
    assert sizes == [1, 1]


def test_stage_epoch_drains_on_input_close_before_other_work_finishes() -> None:
    sizes = _run_epoch_case(
        [(0.01, "a", 10), (0.10, "b", 10)],
        max_inflight=3,
    )
    assert sizes == [1, 1]
