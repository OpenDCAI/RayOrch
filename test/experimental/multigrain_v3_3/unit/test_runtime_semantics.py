"""Event-driven Port semantics, lineage, groups, and failure boundaries."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_3 as mg
from rayorch.experimental.multigrain_v3_3.model import (
    GrainOutcome,
    GrainPhase,
    ItemOutcome,
    ItemRef,
    ShapeState,
)
from rayorch.experimental.multigrain_v3_3.program import ExpandOrigin, GroupOrigin
from rayorch.experimental.multigrain_v3_3.protocol import (
    BlockRef,
    CallReport,
    ExpandedRows,
    OutputReport,
    RowBinding,
)
from rayorch.experimental.multigrain_v3_3.runtime import (
    ArenaEngine,
    CommitError,
    GroupBinding,
    ShapeKey,
)


class U:
    pass


def row(block: int, index: int = 0) -> RowBinding:
    return RowBinding(BlockRef(block), index)


def only_port(program, origin_type):
    return next(
        ref for ref, spec in program.ports.items() if isinstance(spec.origin, origin_type)
    )


class ExpandReducePipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.render = mg.RayModule(U)
        self.assemble = mg.RayModule(U)

    def forward(self, documents):
        page_groups = self.render(documents)
        pages = mg.functional.expand(page_groups)
        reconstructed = mg.functional.reduce(pages)
        return self.assemble(documents, reconstructed)


def _start_one(pipeline):
    program = pipeline.compile().program
    arena = ArenaEngine(program)
    root = arena.admit_sources({program.source_ports[0]: (row(0),)})[0]
    grain = arena.reserve_ready()
    return program, arena, root, grain


def _expanded_ports(program):
    return tuple(
        ref
        for ref, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )


def test_expand_reduce_builds_shape_lineage_and_virtual_group():
    program, arena, root, render_grain = _start_one(ExpandReducePipeline())
    render_output = program.outputs_by_call[render_grain.call][0]
    pages = _expanded_ports(program)[0]

    arena.commit_success(
        CallReport(
            render_grain,
            0,
            (
                OutputReport(
                    render_output,
                    expansions=(
                        ExpandedRows(pages, (row(1, 0), row(1, 1), row(1, 2))),
                    ),
                ),
            ),
        )
    )

    children = arena.entities(program.port(pages).domain)
    assert len(children) == 3
    assert [arena.state.entity_lineage[item].ordinal for item in children] == [0, 1, 2]
    shape = arena.state.shapes[ShapeKey(program.port(pages).domain, root)]
    assert shape.state is ShapeState.SUCCEEDED
    assert shape.cardinality == 3

    grouped_port = only_port(program, GroupOrigin)
    grouped_item = ItemRef(grouped_port, root)
    assert arena.state.items[grouped_item].outcome is ItemOutcome.PRESENT
    binding = arena.state.values[grouped_item]
    assert isinstance(binding, GroupBinding)
    assert binding.shape.offsets_by_level == ((0, 3),)
    assert tuple(item.entity for item in binding.flat_items) == children

    assemble = arena.reserve_ready()
    assert assemble.entity == root
    assert arena.state.grains[assemble].phase is GrainPhase.IN_FLIGHT


def test_empty_expand_is_successful_empty_group_not_drop():
    program, arena, root, render_grain = _start_one(ExpandReducePipeline())
    render_output = program.outputs_by_call[render_grain.call][0]
    pages = _expanded_ports(program)[0]
    arena.commit_success(
        CallReport(
            render_grain,
            0,
            (
                OutputReport(
                    render_output,
                    expansions=(ExpandedRows(pages, ()),),
                ),
            ),
        )
    )

    shape = arena.state.shapes[ShapeKey(program.port(pages).domain, root)]
    assert shape.state is ShapeState.SUCCEEDED
    assert shape.cardinality == 0
    grouped = ItemRef(only_port(program, GroupOrigin), root)
    assert arena.state.items[grouped].outcome is ItemOutcome.PRESENT
    binding = arena.state.values[grouped]
    assert isinstance(binding, GroupBinding)
    assert binding.shape.offsets_by_level == ((0, 0),)
    assert binding.flat_items == ()
    assert arena.reserve_ready().entity == root


def test_filter_drives_group_membership_without_changing_domain():
    class FilterPipeline(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U)
            self.mask = mg.RayModule(U)
            self.sink = mg.RayModule(U)

        def forward(self, documents):
            groups = self.render(documents)
            pages = mg.functional.expand(groups)
            masks = self.mask(pages)
            selected = mg.functional.filter(pages, masks)
            selected_groups = mg.functional.reduce(selected)
            return self.sink(documents, selected_groups)

    program, arena, root, render_grain = _start_one(FilterPipeline())
    render_output = program.outputs_by_call[render_grain.call][0]
    pages = _expanded_ports(program)[0]
    arena.commit_success(
        CallReport(
            render_grain,
            0,
            (
                OutputReport(
                    render_output,
                    expansions=(
                        ExpandedRows(pages, (row(10, 0), row(10, 1), row(10, 2))),
                    ),
                ),
            ),
        )
    )

    mask_results = (True, False, True)
    for index, expected in enumerate(mask_results):
        grain = arena.reserve_ready()
        output = program.outputs_by_call[grain.call][0]
        arena.commit_success(
            CallReport(
                grain,
                0,
                (OutputReport(output, row(20, index), control=expected),),
            )
        )

    filtered_port = next(
        ref
        for ref, spec in program.ports.items()
        if spec.origin.__class__.__name__ == "FilterOrigin"
    )
    children = arena.entities(program.port(pages).domain)
    assert [
        arena.state.items[ItemRef(filtered_port, child)].outcome for child in children
    ] == [ItemOutcome.PRESENT, ItemOutcome.DROPPED, ItemOutcome.PRESENT]

    grouped = ItemRef(only_port(program, GroupOrigin), root)
    binding = arena.state.values[grouped]
    assert isinstance(binding, GroupBinding)
    assert binding.shape.offsets_by_level == ((0, 2),)
    assert [item.entity for item in binding.flat_items] == [children[0], children[2]]
    assert arena.reserve_ready().entity == root


def test_broadcast_resolves_when_parent_and_children_arrive_in_one_commit():
    class BroadcastPipeline(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)
            self.page_call = mg.RayModule(U)

        def forward(self, documents):
            groups, metadata = self.render(documents)
            pages = mg.functional.expand(groups)
            page_metadata = mg.functional.broadcast(metadata, like=pages)
            return self.page_call(pages, page_metadata)

    program, arena, _root, render_grain = _start_one(BroadcastPipeline())
    groups, metadata = program.outputs_by_call[render_grain.call]
    pages = _expanded_ports(program)[0]
    arena.commit_success(
        CallReport(
            render_grain,
            0,
            (
                OutputReport(
                    groups,
                    expansions=(ExpandedRows(pages, (row(30, 0), row(30, 1))),),
                ),
                OutputReport(metadata, row(31, 0)),
            ),
        )
    )

    first = arena.reserve_ready()
    second = arena.reserve_ready()
    assert {first.entity, second.entity} == set(arena.entities(program.port(pages).domain))
    for grain in (first, second):
        inputs = arena.state.grains[grain].inputs
        assert arena.state.values[inputs[1]] == row(31, 0)


def test_failed_before_cardinality_creates_no_children_and_suppresses_reduce():
    program, arena, root, render_grain = _start_one(ExpandReducePipeline())
    pages = _expanded_ports(program)[0]
    arena.commit_failure(render_grain, "decode failed")

    assert arena.entities(program.port(pages).domain) == ()
    shape = arena.state.shapes[ShapeKey(program.port(pages).domain, root)]
    assert shape.state is ShapeState.FAILED
    assert shape.cardinality is None
    grouped = ItemRef(only_port(program, GroupOrigin), root)
    assert arena.state.items[grouped].outcome is ItemOutcome.SUPPRESSED

    assemble_call = max(program.calls)
    assemble_grain = next(
        grain for grain in arena.state.grains if grain.call == assemble_call
    )
    assert arena.state.grains[assemble_grain].outcome is GrainOutcome.SUPPRESSED
    assert arena.ready_count == 0


def test_aligned_mismatch_fails_preflight_without_partial_publication():
    class Aligned(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)
            self.merge = mg.RayModule(U)

        def forward(self, documents):
            left_groups, right_groups = self.render(documents)
            left, right = mg.functional.expand_aligned(left_groups, right_groups)
            return self.merge(left, right)

    program, arena, root, render_grain = _start_one(Aligned())
    left_group, right_group = program.outputs_by_call[render_grain.call]
    left, right = _expanded_ports(program)
    with pytest.raises(CommitError, match="cardinality mismatch"):
        arena.commit_success(
            CallReport(
                render_grain,
                0,
                (
                    OutputReport(
                        left_group,
                        expansions=(ExpandedRows(left, (row(40, 0), row(40, 1))),),
                    ),
                    OutputReport(
                        right_group,
                        expansions=(
                            ExpandedRows(
                                right,
                                (row(41, 0), row(41, 1), row(41, 2)),
                            ),
                        ),
                    ),
                ),
            )
        )

    assert arena.state.grains[render_grain].phase is GrainPhase.IN_FLIGHT
    assert ItemRef(left_group, root) not in arena.state.items
    assert arena.state.shapes == {}
    assert arena.entities(program.port(left).domain) == ()


def test_nested_reduce_preserves_intermediate_empty_list():
    class Nested(mg.Pipeline):
        def __init__(self) -> None:
            self.render_pages = mg.RayModule(U)
            self.render_regions = mg.RayModule(U)
            self.sink = mg.RayModule(U)

        def forward(self, documents):
            page_groups = self.render_pages(documents)
            pages = mg.functional.expand(page_groups)
            region_groups = self.render_regions(pages)
            regions = mg.functional.expand(region_groups)
            regions_by_page = mg.functional.reduce(regions)
            regions_by_document = mg.functional.reduce(regions_by_page)
            return self.sink(documents, regions_by_document)

    program, arena, root, page_grain = _start_one(Nested())
    page_group = program.outputs_by_call[page_grain.call][0]
    page_port, region_port = _expanded_ports(program)
    arena.commit_success(
        CallReport(
            page_grain,
            0,
            (
                OutputReport(
                    page_group,
                    expansions=(ExpandedRows(page_port, (row(50, 0), row(50, 1))),),
                ),
            ),
        )
    )

    region_grains = (arena.reserve_ready(), arena.reserve_ready())
    for index, grain in enumerate(region_grains):
        output = program.outputs_by_call[grain.call][0]
        rows = () if index == 0 else (row(51, 0), row(51, 1))
        arena.commit_success(
            CallReport(
                grain,
                0,
                (
                    OutputReport(
                        output,
                        expansions=(ExpandedRows(region_port, rows),),
                    ),
                ),
            )
        )

    root_groups = [
        ref
        for ref, spec in program.ports.items()
        if isinstance(spec.origin, GroupOrigin) and spec.domain == root.domain
    ]
    assert len(root_groups) == 1
    binding = arena.state.values[ItemRef(root_groups[0], root)]
    assert isinstance(binding, GroupBinding)
    assert binding.shape.depth == 2
    assert binding.shape.offsets_by_level == ((0, 2), (0, 0, 2))
    assert len(binding.flat_items) == 2
    assert arena.reserve_ready().entity == root


def test_retry_changes_generation_not_grain_identity_and_fences_stale_report():
    program, arena, _root, grain = _start_one(ExpandReducePipeline())
    output = program.outputs_by_call[grain.call][0]
    pages = _expanded_ports(program)[0]
    arena.retry(grain)
    assert arena.reserve_ready() == grain
    stale = CallReport(
        grain,
        0,
        (OutputReport(output, expansions=(ExpandedRows(pages, ()),)),),
    )
    with pytest.raises(CommitError, match="stale generation"):
        arena.commit_success(stale)
    arena.commit_success(
        CallReport(
            grain,
            1,
            (OutputReport(output, expansions=(ExpandedRows(pages, ()),)),),
        )
    )
    assert arena.state.grains[grain].outcome is GrainOutcome.SUCCESS
