"""Extreme-depth Expand/Reduce ordering regressions for Multigrain V3."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class BinaryExpand:
    """Append ordinal 0/1 to every logical path."""

    def run(self, rows):
        return [
            [
                (label, (*path, ordinal))
                for ordinal in range(2)
            ]
            for label, path in rows
        ]


class LeafLabel:
    """Make leaf order human-readable in the final nested result."""

    def run(self, rows):
        return [
            f"{label}:{''.join(map(str, path))}"
            for label, path in rows
        ]


class Gather:
    """Preserve the ordered child group as an immutable tuple."""

    def run(self, groups):
        return [tuple(group) for group in groups]


class FiveLevelPipeline(mg.Pipeline):
    """Five nested fan-outs closed by five corresponding Reduces."""

    def __init__(self):
        self.expands = [
            mg.Expand(BinaryExpand).ray_options(batch_size=16)
            for _ in range(5)
        ]
        self.label = mg.Map(LeafLabel).ray_options(batch_size=64)
        self.reduces = [
            mg.Reduce(Gather).ray_options(batch_size=16)
            for _ in range(5)
        ]

    def forward(self, roots):
        levels = [roots]
        for expand in self.expands:
            levels.append(expand(levels[-1]))

        value = self.label(levels[-1])
        for depth, reduce in zip(range(4, -1, -1), self.reduces):
            value = reduce(anchor=levels[depth], members=value)
        return value


def _expected_tree(label: str, prefix: tuple[int, ...], depth: int):
    if depth == 0:
        return f"{label}:{''.join(map(str, prefix))}"
    return tuple(
        _expected_tree(label, (*prefix, ordinal), depth - 1)
        for ordinal in range(2)
    )


def test_five_nested_expands_reduce_in_order_at_every_scope():
    """Five ordinal domains close inside-out without completion-order leakage."""

    result = mg.Executor(
        FiveLevelPipeline(),
        microbatch_size=1,
        max_inflight_arenas=2,
    ).run(
        [("a", ()), ("b", ())],
    )

    assert result.get() == (
        _expected_tree("a", (), 5),
        _expected_tree("b", (), 5),
    )
    assert result.metrics["active_arenas_high_watermark"] == 2


class DirectToRootPipeline(mg.Pipeline):
    """Attempt to flatten five still-open scopes with one outer Reduce."""

    def __init__(self):
        self.expands = [mg.Expand(BinaryExpand) for _ in range(5)]
        self.label = mg.Map(LeafLabel)
        self.reduce = mg.Reduce(Gather)

    def forward(self, roots):
        value = roots
        for expand in self.expands:
            value = expand(value)
        leaves = self.label(value)
        return self.reduce(anchor=roots, members=leaves)


def test_one_reduce_closes_five_scopes_as_canonical_nested_list():
    """Direct root Reduce receives one list level per traversed Expand."""

    result = mg.Executor(DirectToRootPipeline()).run([("root", ())])
    expected = _lists(_expected_tree("root", (), 5))
    assert result.get() == (tuple(expected),)


def _lists(value):
    if isinstance(value, tuple):
        return [_lists(child) for child in value]
    return value


class UnaryExpand:
    """Create one child so depth can be stressed without exponential payload."""

    def run(self, rows):
        return [[(label, (*path, 0))] for label, path in rows]


class TwentyLevelPipeline(mg.Pipeline):
    """Twenty nested scopes exercise compiler/runtime ancestry iteratively."""

    def __init__(self):
        self.expands = [
            mg.Expand(UnaryExpand).ray_options(batch_size=8)
            for _ in range(20)
        ]
        self.label = mg.Map(LeafLabel).ray_options(batch_size=8)
        self.reduces = [
            mg.Reduce(Gather).ray_options(batch_size=8)
            for _ in range(20)
        ]

    def forward(self, roots):
        levels = [roots]
        for expand in self.expands:
            levels.append(expand(levels[-1]))
        value = self.label(levels[-1])
        for depth, reduce in zip(range(19, -1, -1), self.reduces):
            value = reduce(anchor=levels[depth], members=value)
        return value


def test_twenty_unary_scopes_close_without_recursion_or_state_leak():
    """Depth itself remains bounded by explicit ancestry, not special cases."""

    result = mg.Executor(TwentyLevelPipeline()).run([("deep", ())])
    value = "deep:" + "0" * 20
    for _ in range(20):
        value = (value,)
    assert result.get() == (value,)


class OptionalChildren:
    """Produce zero children for one intermediate parent."""

    def run(self, rows):
        return [
            [] if path == (0,) else [(label, (*path, 0))]
            for label, path in rows
        ]


class ZeroInnerPipeline(mg.Pipeline):
    def __init__(self):
        self.outer = mg.Expand(BinaryExpand)
        self.inner = mg.Expand(OptionalChildren)
        self.label = mg.Map(LeafLabel)
        self.reduce = mg.Reduce(Gather)

    def forward(self, roots):
        level1 = self.outer(roots)
        level2 = self.inner(level1)
        return self.reduce(anchor=roots, members=self.label(level2))


def test_direct_nested_reduce_preserves_intermediate_zero_output_as_empty_list():
    """A surviving parent with N=0 remains an explicit empty nested list."""

    result = mg.Executor(ZeroInnerPipeline()).run([("z", ())])
    assert result.get() == (([], ["z:10"]),)


class KeepSecond:
    def run(self, rows):
        return [path == (1,) for _, path in rows]


class DroppedIntermediatePipeline(mg.Pipeline):
    def __init__(self):
        self.outer = mg.Expand(BinaryExpand)
        self.filter = mg.Filter(KeepSecond)
        self.inner = mg.Expand(UnaryExpand)
        self.label = mg.Map(LeafLabel)
        self.reduce = mg.Reduce(Gather)

    def forward(self, roots):
        level1 = self.outer(roots)
        kept = self.filter(level1)
        level2 = self.inner(kept)
        return self.reduce(anchor=roots, members=self.label(level2))


def test_direct_nested_reduce_omits_dropped_intermediate_node():
    """Dropped intermediate occurrence is omitted, unlike successful N=0."""

    result = mg.Executor(DroppedIntermediatePipeline()).run([("d", ())])
    assert result.get() == ((["d:10"],),)


class DualLeaf:
    def run(self, rows):
        return (
            [f"value:{label}:{''.join(map(str, path))}" for label, path in rows],
            [f"meta:{label}:{''.join(map(str, path))}" for label, path in rows],
        )


class PairTrees:
    def run(self, values, metadata):
        return [(values[index], metadata[index]) for index in range(len(values))]


class MultiGroupNestedPipeline(mg.Pipeline):
    def __init__(self):
        self.outer = mg.Expand(BinaryExpand)
        self.inner = mg.Expand(BinaryExpand)
        self.dual = mg.Map(DualLeaf).ray_options(num_outputs=2)
        self.reduce = mg.Reduce(PairTrees)

    def forward(self, roots):
        level1 = self.outer(roots)
        level2 = self.inner(level1)
        values, metadata = self.dual(level2)
        return self.reduce(
            anchor=roots,
            members=values,
            metadata=metadata,
        )


def test_multiple_nested_groups_share_identical_tree_shape():
    """All GROUP inputs are projected through the same canonical scope tree."""

    result = mg.Executor(MultiGroupNestedPipeline()).run([("m", ())])
    assert result.get() == (
        (
            [["value:m:00", "value:m:01"], ["value:m:10", "value:m:11"]],
            [["meta:m:00", "meta:m:01"], ["meta:m:10", "meta:m:11"]],
        ),
    )


class FailInnerExpand:
    def run(self, rows):
        raise RuntimeError("inner fanout failed")


class FailedNestedPipeline(mg.Pipeline):
    def __init__(self):
        self.outer = mg.Expand(BinaryExpand)
        self.inner = mg.Expand(FailInnerExpand).ray_options(
            recovery="fail_batch"
        )
        self.reduce = mg.Reduce(Gather)

    def forward(self, roots):
        level1 = self.outer(roots)
        leaves = self.inner(level1)
        return self.reduce(anchor=roots, members=leaves)


def test_intermediate_fanout_failure_suppresses_root_nested_reduce():
    """A failed required intermediate fanout fail-closes the root Reduce."""

    result = mg.Executor(FailedNestedPipeline()).run([("f", ())])
    assert result.get() == ()
    assert len(result.failures) == 2
    assert len(result.suppressions) == 1
