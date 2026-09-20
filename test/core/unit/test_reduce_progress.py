"""Incremental Reduce agrees with full scans without repeated sibling reads."""

from itertools import product
from dataclasses import replace

import pytest

from rayorch._model import (
    DomainRef,
    EntityRef,
    ExpansionOutcome,
    ItemOutcome,
    ItemRef,
    PortRef,
)
from rayorch._program.plan import ReduceEffect, RuntimePlan
from rayorch._protocol import BlockRef, RowBinding
from rayorch._runtime.engine import InputBatchEngine
from rayorch._runtime.state import ExpansionRef


def make_group(size, *, shared_port=False):
    parent_domain, child_domain = DomainRef(0), DomainRef(1)
    members, values, output = PortRef(0), PortRef(0 if shared_port else 1), PortRef(2)
    effect = ReduceEffect(output, values, members, child_domain, 0)
    plan = RuntimePlan(
        calls={},
        domains={},
        port_domains={
            members: child_domain,
            values: child_domain,
            output: parent_domain,
        },
        item_effects_by_source={port: (effect,) for port in {members, values}},
        reduce_effects_by_child_domain={child_domain: (effect,)},
    )
    engine = InputBatchEngine(plan)
    parent = EntityRef(parent_domain, 0)
    engine._publish_entity(parent)
    expansion = ExpansionRef(child_domain, parent)
    children = engine._create_children(expansion, size)
    return engine, effect, expansion, children, ItemRef(output, parent)


def publish(engine, item, outcome):
    engine._publish_item(
        item,
        outcome,
        binding=RowBinding(BlockRef("rows"), item.entity.value)
        if outcome is ItemOutcome.PRESENT
        else None,
    )


def scan_result(engine, effect, expansion, children):
    """Original full-scan semantics, independent of the progress index."""
    shape = engine._state.expansions.get(expansion)
    if shape is None:
        return None
    if shape.outcome is not ExpansionOutcome.SUCCEEDED:
        outcome = (
            ItemOutcome.DROPPED
            if shape.outcome is ExpansionOutcome.DROPPED
            else ItemOutcome.SUPPRESSED
        )
        return outcome, expansion
    members = [
        engine._state.items.get(ItemRef(effect.members_port, c)) for c in children
    ]
    for child, member in zip(children, members):
        if member is not None and member.outcome in {
            ItemOutcome.FAILED,
            ItemOutcome.SUPPRESSED,
        }:
            return ItemOutcome.SUPPRESSED, ItemRef(effect.members_port, child)
    if any(member is None for member in members):
        return None
    for child, member in zip(children, members):
        if member.outcome is not ItemOutcome.PRESENT:
            continue
        item = ItemRef(effect.value_port, child)
        value = engine._state.items.get(item)
        if value is None:
            return None
        if value.outcome is not ItemOutcome.PRESENT:
            return ItemOutcome.SUPPRESSED, item
    return ItemOutcome.PRESENT, None


@pytest.mark.parametrize("expansion_first", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_incremental_reduce_matches_full_scan_for_partial_and_terminal_facts(
    expansion_first, batched
):
    # Two siblings, separate membership/value ports, every partial/terminal
    # combination. Reverse publication checks ordinal versus arrival order.
    for outcomes in product((None, *ItemOutcome), repeat=4):
        engine, effect, expansion, children, target = make_group(2)
        expected = None

        def check():
            nonlocal expected
            if expected is None:
                expected = scan_result(engine, effect, expansion, children)
            engine.advance()
            actual = engine._state.items.get(target)
            assert (
                None if actual is None else (actual.outcome, actual.cause)
            ) == expected
            if actual is not None:
                assert target not in engine._reduce_progress
                if actual.outcome is ItemOutcome.PRESENT:
                    assert engine.value_binding(target).flat_items == tuple(
                        ItemRef(effect.value_port, child)
                        for child in children
                        if engine.item_outcome(ItemRef(effect.members_port, child))
                        is ItemOutcome.PRESENT
                    )

        if expansion_first:
            engine._publish_expansion(
                expansion, ExpansionOutcome.SUCCEEDED, children=children
            )
            check()
        facts = [
            ItemRef(port, child)
            for port in (effect.members_port, effect.value_port)
            for child in children
        ]
        for item, outcome in reversed(tuple(zip(facts, outcomes))):
            if outcome is not None:
                publish(engine, item, outcome)
                publish(engine, item, outcome)  # replay must not decrement twice
                if not batched:
                    check()
        if not expansion_first:
            engine._publish_expansion(
                expansion, ExpansionOutcome.SUCCEEDED, children=children
            )
        check()


@pytest.mark.parametrize("outcome", list(ExpansionOutcome))
def test_empty_or_unsuccessful_expansion_releases_progress(outcome):
    engine, effect, expansion, children, target = make_group(0)
    engine._publish_expansion(
        expansion,
        outcome,
        children=children if outcome is ExpansionOutcome.SUCCEEDED else None,
    )
    expected = scan_result(engine, effect, expansion, children)
    engine.advance()
    record = engine._state.items[target]
    assert (record.outcome, record.cause) == expected
    assert not engine._reduce_progress


def test_progress_isolated_by_parent_and_consumer_and_rebuilds_from_facts():
    engine, effect, expansion, children, target = make_group(2)
    other_effect = replace(effect, target_port=PortRef(3))
    effects = (effect, other_effect)
    engine.plan = replace(
        engine.plan,
        item_effects_by_source={
            port: effects for port in (effect.members_port, effect.value_port)
        },
        reduce_effects_by_child_domain={effect.child_domain: effects},
    )
    other_parent = EntityRef(target.entity.domain, 1)
    engine._publish_entity(other_parent)
    other_expansion = ExpansionRef(effect.child_domain, other_parent)
    other_children = engine._create_children(other_expansion, 2)
    for shape, group in ((expansion, children), (other_expansion, other_children)):
        engine._publish_expansion(shape, ExpansionOutcome.SUCCEEDED, children=group)
        for child in group:
            publish(engine, ItemRef(effect.members_port, child), ItemOutcome.PRESENT)
    engine.advance()
    assert len(engine._reduce_progress) == 4
    publish(engine, ItemRef(effect.value_port, children[0]), ItemOutcome.PRESENT)
    publish(engine, ItemRef(effect.value_port, other_children[1]), ItemOutcome.PRESENT)
    engine.advance()
    assert engine._reduce_progress[target].next_value_index == 1
    # Rebuilding one consumer must neither lose facts nor affect its peers.
    del engine._reduce_progress[target]
    publish(engine, ItemRef(effect.value_port, children[1]), ItemOutcome.PRESENT)
    engine.advance()
    assert len(engine._reduce_progress) == 2
    for consumer in effects:
        item = ItemRef(consumer.target_port, target.entity)
        assert engine.value_binding(item).flat_items == tuple(
            ItemRef(effect.value_port, child) for child in children
        )
    publish(engine, ItemRef(effect.value_port, other_children[0]), ItemOutcome.DROPPED)
    engine.advance()
    for consumer in effects:
        item = ItemRef(consumer.target_port, other_parent)
        assert engine.item_outcome(item) is ItemOutcome.SUPPRESSED
        assert engine._state.items[item].cause == ItemRef(
            effect.value_port, other_children[0]
        )
    assert not engine._reduce_progress


class CountingItems(dict):
    """Count canonical fact reads, including scans outside transitions.py."""

    reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)

    def __contains__(self, key):
        self.reads += 1
        return super().__contains__(key)


@pytest.mark.parametrize("shared_port", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_reduce_fact_reads_grow_linearly(shared_port, reverse):
    counts = []
    for size in (32, 64, 128):
        engine, effect, expansion, children, target = make_group(
            size, shared_port=shared_port
        )
        engine._state.items = CountingItems()
        engine._publish_expansion(
            expansion, ExpansionOutcome.SUCCEEDED, children=children
        )
        engine.advance()
        if not shared_port:
            for child in children:
                publish(
                    engine, ItemRef(effect.members_port, child), ItemOutcome.PRESENT
                )
            engine.advance()
        for child in reversed(children) if reverse else children:
            publish(engine, ItemRef(effect.value_port, child), ItemOutcome.PRESENT)
            engine.advance()
        counts.append(engine._state.items.reads)
        assert engine.item_outcome(target) is ItemOutcome.PRESENT
        assert engine.value_binding(target).flat_items == tuple(
            ItemRef(effect.value_port, child) for child in children
        )
        assert not engine._reduce_progress
    assert 2 * (counts[1] - counts[0]) == counts[2] - counts[1]
    assert counts[2] <= 16 * 128 + 10
