"""Machine-checked evidence for the reordering-invariance theorem.

See ``docs/todos/13-reordering-invariance-theorem.md``. We execute each motif
under *randomly generated legal shard plans* (set partitions of the base rows,
with arbitrary within/among-shard order -- including full shuffles and
adversarial singleton / one-big splits) and assert, against the serial baseline:

* keyed-equality (``≈``, incl. every lineage field) at every intermediate port;
* byte-identical ordered equality at the final ``Reduce`` output.

This mirrors ``MultigrainRayExecutor._run_node`` (partition -> per-shard
``take`` -> node -> ``concat``) but runs locally without Ray, so it is fast and
deterministic in CI. The point is not timing; it is that reordering does not
change results or lineage for the *actual* implementation.
"""
from __future__ import annotations

import random

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.core import concat
from rayorch.experimental.multigrain.executor import MultigrainExecutor
from rayorch.experimental.multigrain.graph import NodeKind

from test.experimental.multigrain.test_dummy_e2e import GovPipe
from test.experimental.multigrain.test_relate_key_join import (
    LinkPipe,
    _fig_source,
)

# Same set as the Ray executor: only these are row-sharded.
_SHARDABLE = frozenset({NodeKind.MAP, NodeKind.FILTER, NodeKind.EXPAND})


def _random_legal_partition(n: int, rng: random.Random) -> list[list[int]]:
    """A random set partition of range(n): disjoint, covering, arbitrary order."""
    if n <= 1:
        return [list(range(n))]
    indices = list(range(n))
    rng.shuffle(indices)  # arbitrary order within/among shards
    k = rng.randint(1, n)  # 1 (one-big) .. n (all singletons)
    bins: list[list[int]] = [[] for _ in range(k)]
    for idx in indices:
        bins[rng.randrange(k)].append(idx)
    return [b for b in bins if b]


def _run(graph, inputs, rng, *, shard: bool, relation_fns=None):
    """Execute the passive IR, optionally random-sharding shardable nodes.

    Returns the full ``IRPortRef -> PortBatch`` context so intermediate ports can
    be compared, not just the final output.
    """
    base = MultigrainExecutor(relation_fns=relation_fns)
    context = {}
    for spec in graph.inputs:
        context[spec.ref] = inputs[spec.name]

    for node in graph.nodes:
        node_inputs = tuple(context[ref] for ref in node.input_refs)
        can_shard = shard and node.kind in _SHARDABLE and node_inputs and len(node_inputs[0]) > 1
        if can_shard:
            parts = _random_legal_partition(len(node_inputs[0]), rng)
            shard_outs = [
                base._execute_node(node, tuple(port.take(idx) for port in node_inputs))
                for idx in parts
            ]
            outs = tuple(
                concat([s[oi] for s in shard_outs], name=shard_outs[0][oi].name)
                for oi in range(len(node.output_refs))
            )
        else:
            outs = base._execute_node(node, node_inputs)
        for ref, batch in zip(node.output_refs, outs):
            context[ref] = batch
    return context


def _keyed(batch):
    """The record-keyed view used to test ``≈`` (identity + all lineage fields)."""
    return {
        rid: (
            batch.values[i],
            tuple(sorted(batch.ancestors[i].items())),
            tuple(sorted(batch.ordinals[i].items())),
            tuple(batch.lineage[i]),
            batch.display_keys[i],
        )
        for i, rid in enumerate(batch.record_ids)
    }


def _assert_keyed_equal_everywhere(phys_ctx, ser_ctx):
    assert phys_ctx.keys() == ser_ctx.keys()
    for ref in ser_ctx:
        assert _keyed(phys_ctx[ref]) == _keyed(ser_ctx[ref]), f"port {ref} not ≈"


def _assert_ordered_equal(phys_batch, ser_batch):
    assert phys_batch.record_ids == ser_batch.record_ids
    assert phys_batch.values == ser_batch.values


# ---------------------------------------------------------------------------
# GovPipe: Expand -> Map -> Filter -> Reduce (anchor = graph input)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_gov_pipeline_is_reordering_invariant(seed: int) -> None:
    rng = random.Random(seed)
    graph = GovPipe().compile()
    docs = mg.source(
        ["a-b-c-d-e", "f-g", "h-i-j-k-l-m-n", "o", "p-q-r"], name="docs"
    )
    inputs = {"docs": docs}

    ser = _run(graph, inputs, rng, shard=False)
    phys = _run(graph, inputs, rng, shard=True)

    # Theorem part 1: keyed/lineage equality at every port, incl. the permuted
    # Expand/Map/Filter ports.
    _assert_keyed_equal_everywhere(phys, ser)

    # Theorem part 2: the Reduce output (anchor = docs input) is byte-identical.
    (out_ref,) = graph.graph_outputs
    _assert_ordered_equal(phys[out_ref], ser[out_ref])


# ---------------------------------------------------------------------------
# LinkPipe: Expand -> Relate(on=) -> Reduce (M:N key-join in the middle)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_key_join_pipeline_is_reordering_invariant(seed: int) -> None:
    rng = random.Random(seed)
    graph = LinkPipe().compile()
    pdfs = mg.source(["a.pdf", "b.pdf"], name="pdfs")
    inputs = {"pdfs": pdfs, "figs": _fig_source()}

    ser = _run(graph, inputs, rng, shard=False)
    phys = _run(graph, inputs, rng, shard=True)

    # With content-addressed key-join ids, even the Relate port is ≈ (Lemma 3a).
    _assert_keyed_equal_everywhere(phys, ser)

    (out_ref,) = graph.graph_outputs
    _assert_ordered_equal(phys[out_ref], ser[out_ref])
