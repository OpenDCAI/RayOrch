"""Port/domain 不变量与 correctness-first 物理 lowering。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_2 as mg
from rayorch.experimental.multigrain_v3.dag import InputMode, Primitive


class U:
    pass


class PartialExpandPipeline(mg.Pipeline):
    def __init__(self):
        self.render = mg.RayModule(U).ray_options(num_outputs=2)
        self.ocr = mg.RayModule(U)
        self.layout = mg.RayModule(U)
        self.assemble = mg.RayModule(U)

    def forward(self, pdfs):
        page_groups, metadata = self.render(pdfs)
        pages = mg.functional.expand(page_groups)
        ocr = self.ocr(pages)
        layout = self.layout(pages)
        content_groups, layout_groups = mg.functional.reduce_aligned(ocr, layout)
        return self.assemble(
            metadata,
            content_groups,
            layout_groups,
            page_groups,
        )


def test_port_expand_preserves_parent_port_and_branches_by_domain():
    compiled = PartialExpandPipeline().compile()
    dag = compiled.logical

    render = dag.nodes[1]
    expand = dag.nodes[2]
    ocr = dag.nodes[3]
    layout = dag.nodes[4]
    reduce = dag.nodes[5]
    assemble = dag.nodes[6]

    page_groups, metadata = render.outputs
    pages = expand.outputs[0]
    assert expand.inputs == (page_groups,)
    assert pages != page_groups
    assert dag.port(page_groups).domain == dag.port(metadata).domain
    assert dag.port(pages).domain != dag.port(page_groups).domain
    assert ocr.inputs[0].port == pages
    assert layout.inputs[0].port == pages

    assert dag.port(reduce.outputs[0]).domain == dag.port(page_groups).domain
    assert dag.port(reduce.outputs[1]).domain == dag.port(page_groups).domain
    assert isinstance(dag.port(reduce.outputs[0]).layout, mg.GroupValue)
    assert tuple(item.port for item in assemble.inputs) == (
        metadata,
        reduce.outputs[0],
        reduce.outputs[1],
        page_groups,
    )


def test_lowering_keeps_partial_expand_as_an_explicit_stage():
    physical = PartialExpandPipeline().compile().physical.dag

    assert tuple(stage.kind for stage in physical.stages) == (
        Primitive.SOURCE,
        Primitive.MAP,
        Primitive.EXPAND,
        Primitive.MAP,
        Primitive.MAP,
        Primitive.REDUCE,
        Primitive.MAP,
    )
    render = physical.stage(1)
    expand = physical.stage(2)
    reduce = physical.stage(5)
    assemble = physical.stage(6)
    assert expand.inputs[0].port == render.output_ports()[0]
    assert tuple(spec.mode for spec in reduce.inputs) == (
        InputMode.ANCHOR,
        InputMode.GROUP,
        InputMode.GROUP,
    )
    assert reduce.inputs[0].port == render.output_ports()[0]
    assert assemble.inputs[-1].port == render.output_ports()[0]
    assert assemble.inputs[0].port == render.output_ports()[1]


def test_independent_expands_do_not_implicitly_align():
    class Independent(mg.Pipeline):
        def __init__(self):
            self.render = mg.RayModule(U).ray_options(num_outputs=2)
            self.join = mg.RayModule(U)

        def forward(self, rows):
            left_groups, right_groups = self.render(rows)
            left = mg.functional.expand(left_groups)
            right = mg.functional.expand(right_groups)
            return self.join(left, right)

    with pytest.raises(mg.LogicalCompileError, match="different entity domains"):
        Independent().compile()


def test_expand_aligned_creates_one_shared_child_domain():
    class Aligned(mg.Pipeline):
        def __init__(self):
            self.render = mg.RayModule(U).ray_options(num_outputs=2)
            self.join = mg.RayModule(U)

        def forward(self, rows):
            left_groups, right_groups = self.render(rows)
            left, right = mg.functional.expand_aligned(
                left_groups,
                right_groups,
            )
            return self.join(left, right)

    compiled = Aligned().compile()
    expand = compiled.logical.nodes[2]
    left, right = expand.outputs
    assert compiled.logical.port(left).domain == compiled.logical.port(right).domain
    physical_expand = compiled.physical.dag.stage(2)
    assert physical_expand.kind is Primitive.EXPAND
    assert physical_expand.output_count == 2
    assert len(physical_expand.inputs) == 2
