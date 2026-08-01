"""Compile-time contracts for the compact V3 General DAG."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3 as mg
from rayorch.experimental.multigrain_v3.dag import InputMode, Primitive


class Unary:
    def run(self, rows):
        return list(rows)


class Fanout:
    def run(self, rows):
        return [[row, row] for row in rows]


class Join:
    def run(self, left, right):
        return [(a, b) for a, b in zip(left, right)]


class Assemble:
    def run(self, members):
        return [tuple(group) for group in members]


class GeneralDag(mg.Pipeline):
    def __init__(self):
        self.left = mg.Map(Unary).ray_options(num_outputs=2)
        self.join = mg.Map(Join)

    def forward(self, rows):
        left, right = self.left(rows)
        return self.join(left, right)


def test_general_dag_is_explicit_and_port_metadata_is_derived():
    """A diamond/multi-output graph compiles without a parallel PortSpec."""

    compiled = GeneralDag().compile()
    assert tuple(stage.kind for stage in compiled.dag.stages) == (
        Primitive.SOURCE,
        Primitive.MAP,
        Primitive.MAP,
    )
    assert compiled.dag.stage(1).output_count == 2
    assert compiled.dag.stage(1).output_ports() == (
        mg.PortId(1, 0),
        mg.PortId(1, 1),
    )
    assert len(compiled.dag.consumers(mg.PortId(1, 0))) == 1
    assert len(compiled.dag.consumers(mg.PortId(1, 1))) == 1


class FilterTuple(mg.Pipeline):
    def __init__(self):
        self.filter = mg.Filter(Unary)

    def forward(self, left, right):
        return self.filter(left, right)


def test_filter_output_ports_are_fixed_to_required_inputs():
    """Filter is tuple-preserving: one output port per required input."""

    compiled = FilterTuple().compile()
    stage = compiled.dag.stage(2)
    assert stage.kind is Primitive.FILTER
    assert stage.output_count == 2
    assert all(spec.mode is InputMode.ONE for spec in stage.inputs)


class BadFilter(mg.Pipeline):
    def __init__(self):
        self.filter = mg.Filter(Unary)

    def forward(self, left, right):
        return self.filter(left, mg.optional(right))


def test_filter_rejects_optional_inputs():
    """Optional tuples must be normalized by Map before Filter."""

    with pytest.raises(mg.CompileError, match="Filter only accepts"):
        BadFilter().compile()


class Nested(mg.Pipeline):
    def __init__(self):
        self.pages = mg.Expand(Fanout)
        self.regions = mg.Expand(Fanout)
        self.region_map = mg.Map(Unary)
        self.inner = mg.Reduce(Assemble)
        self.page_map = mg.Map(Unary)
        self.outer = mg.Reduce(Assemble)

    def forward(self, documents):
        pages = self.pages(documents)
        regions = self.regions(pages)
        processed = self.region_map(regions)
        page_results = self.inner(anchor=pages, members=processed)
        completed = self.page_map(page_results)
        return self.outer(anchor=documents, members=completed)


def test_nested_reduce_closes_one_expand_scope_at_a_time():
    """Inner Reduce restores Page scope before the outer PDF Reduce."""

    compiled = Nested().compile()
    inner = compiled.dag.stage(4)
    outer = compiled.dag.stage(6)
    assert inner.reduce.scope_path == (2,)
    assert outer.reduce.scope_path == (1,)


class CrossTwoExpands(mg.Pipeline):
    def __init__(self):
        self.left = mg.Expand(Fanout)
        self.right = mg.Expand(Fanout)
        self.join = mg.Map(Join)

    def forward(self, rows):
        return self.join(self.left(rows), self.right(rows))


def test_independent_expands_cannot_implicitly_zip_by_length():
    """Equal runtime lengths do not create an alignment domain."""

    with pytest.raises(mg.CompileError, match="different scopes"):
        CrossTwoExpands().compile()


class MultiExpandSingleReduce(mg.Pipeline):
    def __init__(self):
        self.pages = mg.Expand(Fanout)
        self.regions = mg.Expand(Fanout)
        self.reduce = mg.Reduce(Assemble)

    def forward(self, documents):
        pages = self.pages(documents)
        regions = self.regions(pages)
        return self.reduce(anchor=documents, members=regions)


def test_multi_expand_single_reduce_compiles_to_canonical_scope_path():
    """Cross-level Reduce derives one nested-list level per Expand."""

    compiled = MultiExpandSingleReduce().compile()
    reduce = compiled.dag.stages[-1]
    assert reduce.reduce.scope_path == (1, 2)
