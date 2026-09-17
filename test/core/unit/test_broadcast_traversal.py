"""Broadcast visits related Expansion facts without a per-target waiting index."""

import tracemalloc

import pytest

from rayorch._model import (
    DomainRef,
    EntityRef,
    ExpansionOutcome,
    ItemOutcome,
    ItemRef,
    PortRef,
)
from rayorch._program.logical import DomainSpec
from rayorch._program.plan import BroadcastEffect, RuntimePlan
from rayorch._protocol import BlockRef, RowBinding
from rayorch._runtime.engine import InputBatchEngine
from rayorch._runtime.state import ExpansionRef


class FullScanEngine(InputBatchEngine):
    """Keep the original source-triggered search as an independent oracle."""

    def _broadcast_descendants(self, effect, source_entity):
        for entity in self._entities_by_domain.get(effect.target_domain, ()):
            if self._ancestor_entity(entity, effect.source_domain) == source_entity:
                yield entity


def make_engine(depth=2, rules=1, engine_type=InputBatchEngine):
    domains = tuple(DomainRef(index) for index in range(depth + 1))
    effects = tuple(
        BroadcastEffect(
            PortRef(rules + index), PortRef(index), domains[0], domains[-1], True
        )
        for index in range(rules)
    )
    plan = RuntimePlan(
        calls={},
        domains={
            domain: DomainSpec(domain, domains[index - 1] if index else None)
            for index, domain in enumerate(domains)
        },
        port_domains={
            port: domain
            for effect in effects
            for port, domain in (
                (effect.source_port, domains[0]),
                (effect.target_port, domains[-1]),
            )
        },
        output_tree=tuple(effect.target_port for effect in effects),
        item_effects_by_source={effect.source_port: (effect,) for effect in effects},
        broadcast_effects_by_target_domain={domains[-1]: effects},
    )
    return engine_type(plan), domains, effects


def expand(engine, parent, domain, count, outcome=ExpansionOutcome.SUCCEEDED):
    shape = ExpansionRef(domain, parent)
    children = (
        engine._create_children(shape, count)
        if outcome is ExpansionOutcome.SUCCEEDED
        else None
    )
    engine._publish_expansion(shape, outcome, children=children)
    return () if children is None else children


def publish_source(engine, effect, root, outcome):
    engine._publish_item(
        ItemRef(effect.source_port, root),
        outcome,
        binding=RowBinding(BlockRef(effect.source_port.value), root.value)
        if outcome is ItemOutcome.PRESENT
        else None,
        cause=None if outcome is ItemOutcome.PRESENT else "source issue",
        control=bool(effect.source_port.value)
        if outcome is ItemOutcome.PRESENT
        else None,
    )


@pytest.mark.parametrize(
    "order",
    ["source_first", "target_first", "interleaved", "queued", "queued_source_first"],
)
@pytest.mark.parametrize("outcome", list(ItemOutcome))
def test_broadcast_matches_full_scan_for_arrival_orders_and_outcomes(order, outcome):
    def run(engine_type):
        engine, domains, effects = make_engine(rules=2, engine_type=engine_type)
        roots = tuple(EntityRef(domains[0], index) for index in range(2))
        defer_events = order.startswith("queued")
        for root in roots:
            engine._publish_entity(root)

        def sources():
            for root in reversed(roots):
                publish_source(engine, effects[0], root, outcome)
                publish_source(engine, effects[1], root, ItemOutcome.PRESENT)
            if not defer_events:
                engine.advance()

        if order in {"source_first", "queued_source_first"}:
            sources()
        pages = [expand(engine, root, domains[1], 3) for root in roots]
        if order == "interleaved":
            sources()
        # Create final targets in reverse parent order, unlike DFS traversal.
        # Include empty groups and groups created after the source was consumed.
        for group in reversed(pages):
            expand(engine, group[1], domains[2], 0)
            expand(engine, group[0], domains[2], 2)
        if not defer_events:
            engine.advance()
        if order in {"target_first", "queued"}:
            sources()
        engine.advance()
        for group in pages:
            expand(engine, group[2], domains[2], 1)
        engine.advance()

        for effect in effects:
            for entity in engine.entities(domains[-1]):
                root = engine._ancestor_entity(entity, domains[0])
                target = ItemRef(effect.target_port, entity)
                source = ItemRef(effect.source_port, root)
                assert engine._state.items[target] == engine._state.items[source]
                if engine.item_outcome(source) is ItemOutcome.PRESENT:
                    assert engine.value_binding(target) is engine.value_binding(source)
        for root in roots:
            publish_source(engine, effects[0], root, outcome)
            publish_source(engine, effects[1], root, ItemOutcome.PRESENT)
        assert not engine._fact_queue
        engine.close_admission()
        assert engine.is_complete()
        return engine._state.items, engine._state.values

    assert run(InputBatchEngine) == run(FullScanEngine)


@pytest.mark.parametrize("outcome", list(ExpansionOutcome))
def test_empty_or_failed_expansions_have_no_broadcast_targets(outcome):
    engine, domains, (effect,) = make_engine()
    root = EntityRef(domains[0], 0)
    engine._publish_entity(root)
    pages = expand(engine, root, domains[1], 3)
    for page in pages:
        expand(engine, page, domains[2], 0, outcome)
    engine.advance()
    publish_source(engine, effect, root, ItemOutcome.PRESENT)
    engine.advance()
    assert engine.entities(domains[2]) == ()
    assert list(engine._broadcast_descendants(effect, root)) == []
    engine.close_admission()
    assert engine.is_complete()


class CountingReads(dict):
    reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)


@pytest.mark.parametrize("sparse", [False, True])
def test_source_traversal_only_reads_related_expansions(sparse):
    # One source must not inspect other sources' subtrees, even when they exist.
    for source_count in (1, 8, 32):
        engine, domains, (effect,) = make_engine()
        roots = [EntityRef(domains[0], index) for index in range(source_count)]
        for root in roots:
            engine._publish_entity(root)
            pages = expand(engine, root, domains[1], 16)
            for index, page in enumerate(pages):
                expand(engine, page, domains[2], 2 if not sparse or index == 0 else 0)
        engine.advance()
        engine._state.expansions = CountingReads(engine._state.expansions)
        engine._state.entity_lineage = CountingReads(engine._state.entity_lineage)
        for root in roots:
            before = engine._state.expansions.reads
            lineage_before = engine._state.entity_lineage.reads
            publish_source(engine, effect, root, ItemOutcome.PRESENT)
            engine.advance()
            # Sparse traversal still visits all 16 intermediate pages.
            assert engine._state.expansions.reads - before == 17
            # Two ancestor links plus one publication-index lookup per target,
            # and one lookup for the source publication. A hidden domain scan
            # would add reads proportional to unrelated roots as well.
            assert engine._state.entity_lineage.reads - lineage_before == 1 + 3 * (
                2 if sparse else 32
            )
        assert engine._state.expansions.reads == 17 * source_count
        expected_targets = source_count * (2 if sparse else 32)
        assert (
            sum(item.port == effect.target_port for item in engine._state.items)
            == expected_targets
        )


def test_broadcast_does_not_visit_sibling_domain_branches():
    engine, domains, (effect,) = make_engine()
    root = EntityRef(domains[0], 0)
    engine._publish_entity(root)
    (page,) = expand(engine, root, domains[1], 1)
    (target,) = expand(engine, page, domains[2], 1)
    expand(engine, root, DomainRef(99), 100)
    engine.advance()
    engine._state.expansions = CountingReads(engine._state.expansions)
    assert list(engine._broadcast_descendants(effect, root)) == [target]
    assert engine._state.expansions.reads == 2


def test_traversal_uses_depth_space_without_materializing_wide_groups():
    peaks = []
    for width in (64, 16384):
        engine, domains, (effect,) = make_engine(depth=4)
        root = EntityRef(domains[0], 0)
        engine._publish_entity(root)
        parent = root
        for domain in domains[1:-1]:
            (parent,) = expand(engine, parent, domain, 1)
        expand(engine, parent, domains[-1], width)
        # Measure traversal alone: canonical facts already exist, and published
        # target Items would necessarily consume width-dependent output memory.
        tracemalloc.start()
        try:
            for _ in engine._broadcast_descendants(effect, root):
                pass
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] - peaks[0] < 16 * 1024


def test_deep_broadcast_traversal_does_not_use_python_recursion():
    engine, domains, (effect,) = make_engine(depth=1200)
    root = EntityRef(domains[0], 0)
    engine._publish_entity(root)
    parent = root
    for domain in domains[1:]:
        (parent,) = expand(engine, parent, domain, 1)
    assert list(engine._broadcast_descendants(effect, root)) == [parent]
