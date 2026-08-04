"""V3.1 authoring 降低到未改动的 V3 编译结构。"""

from __future__ import annotations

import pytest

import rayorch.experimental.multigrain_v3_1 as mg
from rayorch.experimental.multigrain_v3.api import CompiledPipeline, Port as V3Port
from rayorch.experimental.multigrain_v3.dag import (
    CompiledDAG,
    InputMode,
    Primitive,
    StageSpec,
)


class Render:
    def run(self, pdfs):
        return [[f"{pdf}:0", f"{pdf}:1"] for pdf in pdfs]


class Ocr:
    def run(self, pages):
        return [page.upper() for page in pages]


class Assemble:
    def run(self, pdfs, content_groups, page_groups):
        return list(zip(pdfs, content_groups, page_groups))


class FunctionalPipeline(mg.Pipeline):
    def __init__(self):
        self.render = mg.RayModule(Render).ray_options(batch_size=2)
        self.ocr = mg.RayModule(Ocr).ray_options(batch_size=8)
        self.assemble = mg.RayModule(Assemble).ray_options(batch_size=2)

    def forward(self, pdfs):
        page_groups = self.render(pdfs)
        pages = mg.functional.expand(page_groups)
        contents = self.ocr(pages)
        content_groups = mg.functional.reduce(contents)
        return self.assemble(pdfs, content_groups, page_groups)


def test_functional_api_lowers_without_adapter_stages():
    compiled = FunctionalPipeline().compile()

    assert isinstance(compiled, CompiledPipeline)
    assert isinstance(compiled.dag, CompiledDAG)
    assert all(isinstance(port, V3Port) for port in compiled.source_ports)
    assert all(isinstance(port, V3Port) for port in compiled.outputs)
    assert all(isinstance(stage, StageSpec) for stage in compiled.dag.stages)
    assert tuple(stage.kind for stage in compiled.dag.stages) == (
        Primitive.SOURCE,
        Primitive.EXPAND,
        Primitive.MAP,
        Primitive.REDUCE,
    )

    reduce_stage = compiled.dag.stages[-1]
    assert tuple(spec.mode for spec in reduce_stage.inputs) == (
        InputMode.ANCHOR,
        InputMode.ONE,
        InputMode.GROUP,
        InputMode.GROUP,
    )
    assert tuple(spec.name for spec in reduce_stage.inputs) == (
        "__anchor__",
        "arg_0",
        "arg_1",
        "arg_2",
    )
    assert reduce_stage.inputs[0].port == compiled.dag.source_ports[0]
    assert reduce_stage.inputs[1].port == compiled.dag.source_ports[0]
    assert reduce_stage.reduce is not None
    assert reduce_stage.reduce.members_input == 2
    assert reduce_stage.reduce.scope_path == (1,)


def test_lazy_group_cannot_escape_as_physical_pipeline_output():
    class BadOutput(mg.Pipeline):
        def __init__(self):
            self.render = mg.RayModule(Render)

        def forward(self, pdfs):
            page_groups = self.render(pdfs)
            pages = mg.functional.expand(page_groups)
            return mg.functional.reduce(pages)

    with pytest.raises(mg.CompileError, match="lazy group"):
        BadOutput().compile()


def test_expand_must_happen_before_group_consumers():
    class LateExpand(mg.Pipeline):
        def __init__(self):
            self.render = mg.RayModule(Render)
            self.consume = mg.RayModule(Ocr)

        def forward(self, pdfs):
            page_groups = self.render(pdfs)
            self.consume(page_groups)
            return mg.functional.expand(page_groups)

    with pytest.raises(mg.CompileError, match="precede all consumers"):
        LateExpand().compile()
