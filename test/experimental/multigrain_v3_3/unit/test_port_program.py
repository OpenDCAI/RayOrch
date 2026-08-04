"""Port/Domain behavior matrix for the v3.3 static Program."""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_3 as mg
from rayorch.experimental.multigrain_v3_3.program import (
    BroadcastOrigin,
    ExpandOrigin,
    FilterOrigin,
    GroupOrigin,
)


class U:
    pass


class PdfPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.render = mg.RayModule(U, num_outputs=2)
        self.ocr = mg.RayModule(U)
        self.layout = mg.RayModule(U)
        self.assemble = mg.RayModule(U)

    def forward(self, pdfs):
        page_groups, metadata = self.render(pdfs)
        pages = mg.functional.expand(page_groups)
        ocr = self.ocr(pages)
        layout = self.layout(pages)
        ocr_groups, layout_groups = mg.functional.reduce_aligned(
            ocr,
            layout,
            members=ocr,
        )
        return self.assemble(
            metadata,
            ocr_groups,
            layout_groups,
            page_groups,
        )


def test_partial_expand_and_branching_have_only_call_execution_nodes():
    compiled = PdfPipeline().compile()
    program = compiled.program

    assert len(program.calls) == 4
    assert len(compiled.execution.pools) == 4
    assert len(program.domains) == 2

    call_outputs = [
        tuple(program.outputs_by_call[call]) for call in sorted(program.calls)
    ]
    page_groups, metadata = call_outputs[0]
    pages = next(
        ref
        for ref, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    )
    assert program.port(page_groups).domain == program.port(metadata).domain
    assert program.port(pages).domain != program.port(page_groups).domain

    page_consumers = program.consumers_by_port[pages]
    call_consumers = [edge for edge in page_consumers if hasattr(edge, "call")]
    assert len(call_consumers) == 2

    groups = [
        spec
        for spec in program.ports.values()
        if isinstance(spec.origin, GroupOrigin)
    ]
    assert len(groups) == 2
    assert all(group.domain == program.port(page_groups).domain for group in groups)
    assert all(group.origin.members_port == call_outputs[1][0] for group in groups)


def test_same_expand_expression_is_interned():
    class Repeated(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U)
            self.merge = mg.RayModule(U)

        def forward(self, rows):
            groups = self.render(rows)
            left = mg.functional.expand(groups)
            right = mg.functional.expand(groups)
            return self.merge(left, right)

    program = Repeated().compile().program
    expanded = [
        port
        for port, spec in program.ports.items()
        if isinstance(spec.origin, ExpandOrigin)
    ]
    assert len(expanded) == 1
    merge = program.call(max(program.calls))
    assert merge.inputs[0].port == merge.inputs[1].port == expanded[0]


def test_independent_expands_do_not_align_by_equal_shape():
    class Independent(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)
            self.merge = mg.RayModule(U)

        def forward(self, rows):
            left_groups, right_groups = self.render(rows)
            left = mg.functional.expand(left_groups)
            right = mg.functional.expand(right_groups)
            return self.merge(left, right)

    with pytest.raises(mg.CompileError, match="different Domains"):
        Independent().compile()


def test_expand_aligned_shares_domain_and_shape_reporters():
    class Aligned(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)
            self.merge = mg.RayModule(U)

        def forward(self, rows):
            left_groups, right_groups = self.render(rows)
            left, right = mg.functional.expand_aligned(
                left_groups,
                right_groups,
            )
            return self.merge(left, right)

    program = Aligned().compile().program
    expanded = [
        spec for spec in program.ports.values() if isinstance(spec.origin, ExpandOrigin)
    ]
    assert len(expanded) == 2
    assert expanded[0].domain == expanded[1].domain
    reporters = program.shape_reporters_by_domain[expanded[0].domain]
    assert reporters == (
        expanded[0].origin.group_port,
        expanded[1].origin.group_port,
    )


def test_broadcast_requires_ancestor_and_reuses_same_domain_port():
    class Broadcast(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)
            self.consume = mg.RayModule(U)

        def forward(self, rows):
            groups, metadata = self.render(rows)
            pages = mg.functional.expand(groups)
            page_metadata = mg.functional.broadcast(metadata, like=pages)
            same = mg.functional.broadcast(page_metadata, like=pages)
            assert same is page_metadata
            return self.consume(pages, page_metadata)

    program = Broadcast().compile().program
    broadcasts = [
        spec
        for spec in program.ports.values()
        if isinstance(spec.origin, BroadcastOrigin)
    ]
    assert len(broadcasts) == 1

    class Invalid(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U, num_outputs=2)

        def forward(self, rows):
            left, right = self.render(rows)
            left_child = mg.functional.expand(left)
            right_child = mg.functional.expand(right)
            return mg.functional.broadcast(left_child, like=right_child)

    with pytest.raises(mg.CompileError, match="ancestor"):
        Invalid().compile()


def test_filter_keeps_domain_and_optional_is_only_an_input_policy():
    class Filtered(mg.Pipeline):
        def __init__(self) -> None:
            self.render = mg.RayModule(U)
            self.mask = mg.RayModule(U)
            self.side = mg.RayModule(U)
            self.merge = mg.RayModule(U)

        def forward(self, rows):
            groups = self.render(rows)
            pages = mg.functional.expand(groups)
            mask = self.mask(pages)
            selected = mg.functional.filter(pages, mask)
            side = self.side(pages)
            return self.merge(selected, aux=mg.functional.optional(side))

    program = Filtered().compile().program
    filtered = next(
        spec
        for spec in program.ports.values()
        if isinstance(spec.origin, FilterOrigin)
    )
    assert filtered.domain == program.port(filtered.origin.source_port).domain
    merge = program.call(max(program.calls))
    assert merge.driving_input == 0
    assert merge.inputs[0].mode is mg.InputMode.REQUIRED
    assert merge.inputs[1].mode is mg.InputMode.OPTIONAL


def test_nested_expand_reduce_returns_one_parent_level_at_a_time():
    class Nested(mg.Pipeline):
        def __init__(self) -> None:
            self.pages = mg.RayModule(U)
            self.regions = mg.RayModule(U)

        def forward(self, docs):
            page_groups = self.pages(docs)
            pages = mg.functional.expand(page_groups)
            region_groups = self.regions(pages)
            regions = mg.functional.expand(region_groups)
            regions_by_page = mg.functional.reduce(regions)
            regions_by_doc = mg.functional.reduce(regions_by_page)
            return regions_by_doc

    program = Nested().compile().program
    domains = sorted(program.domains)
    assert len(domains) == 3
    root, page, region = domains
    assert program.domain(page).parent == root
    assert program.domain(region).parent == page
    output = program.output_tree
    assert program.port(output).domain == root


def test_optional_driving_input_is_rejected():
    class Invalid(mg.Pipeline):
        def __init__(self) -> None:
            self.call = mg.RayModule(U)

        def forward(self, rows):
            return self.call(mg.functional.optional(rows))

    with pytest.raises(mg.CompileError, match="driving input cannot be optional"):
        Invalid().compile()


def test_plain_function_adapter_is_still_a_call():
    @mg.function
    def transform(value):
        return value

    class FunctionPipeline(mg.Pipeline):
        def forward(self, rows):
            return transform(rows)

    compiled = FunctionPipeline().compile()
    assert len(compiled.program.calls) == 1
    assert len(compiled.execution.pools) == 1
