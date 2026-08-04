"""现有 Docling/Video/Nested 实验拓扑在 v3.3 Program 中的可表达性回归。"""

from __future__ import annotations

import rayorch.experimental.multigrain_v3_3 as mg
from rayorch.experimental.multigrain_v3_3.program import GroupOrigin


class U:
    """只用于符号编译、不执行的占位 kernel。"""


class DoclingCoreTopology(mg.Pipeline):
    """PDF→Page 后 layout/OCR/table 多分支，再 ordered Reduce。"""

    def __init__(self) -> None:
        self.parse = mg.RayModule(U)
        self.layout = mg.RayModule(U)
        self.ocr = mg.RayModule(U)
        self.table = mg.RayModule(U)
        self.merge = mg.RayModule(U)
        self.finish = mg.RayModule(U)

    def forward(self, documents):
        """模拟 Docling core-stage disaggregation 的 Domain 关系。"""

        pages = mg.functional.expand(self.parse(documents))
        layout = self.layout(pages)
        ocr = self.ocr(pages)
        tables = self.table(pages)
        merged = self.merge(layout, ocr, tables)
        page_groups = mg.functional.reduce(merged)
        return self.finish(documents, page_groups)


class VideoMultimodalTopology(mg.Pipeline):
    """Video 的 Audio/Frame 独立 child Domains，分别 Reduce 后汇合。"""

    def __init__(self) -> None:
        self.audio_chunks = mg.RayModule(U)
        self.frames = mg.RayModule(U)
        self.whisper = mg.RayModule(U)
        self.vision = mg.RayModule(U)
        self.merge = mg.RayModule(U)

    def forward(self, videos):
        """独立 fan-out 不隐式 zip，只在 Video root Domain 显式 merge。"""

        audio = mg.functional.expand(self.audio_chunks(videos))
        frames = mg.functional.expand(self.frames(videos))
        transcripts = mg.functional.reduce(self.whisper(audio))
        visual = mg.functional.reduce(self.vision(frames))
        return self.merge(videos, transcripts, visual)


class NestedRegionTopology(mg.Pipeline):
    """Document→Page→Region、Filter 和两级 Reduce。"""

    def __init__(self) -> None:
        self.pages = mg.RayModule(U)
        self.regions = mg.RayModule(U)
        self.kind_mask = mg.RayModule(U)
        self.ocr = mg.RayModule(U)
        self.finish = mg.RayModule(U)

    def forward(self, documents):
        """模拟实验矩阵中的嵌套文档解析 case。"""

        pages = mg.functional.expand(self.pages(documents))
        regions = mg.functional.expand(self.regions(pages))
        selected = mg.functional.filter(regions, self.kind_mask(regions))
        contents = self.ocr(selected)
        contents_by_page = mg.functional.reduce(contents, members=selected)
        contents_by_document = mg.functional.reduce(contents_by_page)
        return self.finish(documents, contents_by_document)


def test_docling_core_topology_has_one_child_domain_and_call_only_pools():
    compiled = DoclingCoreTopology().compile()

    assert len(compiled.program.domains) == 2
    assert len(compiled.program.calls) == 6
    assert len(compiled.execution.pools) == 6
    assert sum(
        isinstance(port.origin, GroupOrigin)
        for port in compiled.program.ports.values()
    ) == 1


def test_video_multimodal_topology_keeps_independent_child_domains_explicit():
    compiled = VideoMultimodalTopology().compile()

    assert len(compiled.program.domains) == 3
    assert len(compiled.program.calls) == 5
    merge = compiled.program.call(max(compiled.program.calls))
    assert all(
        compiled.program.port(input_.port).domain
        == compiled.program.port(compiled.program.source_ports[0]).domain
        for input_ in merge.inputs
    )


def test_nested_region_topology_has_three_domains_and_two_level_group():
    compiled = NestedRegionTopology().compile()

    assert len(compiled.program.domains) == 3
    output_call = compiled.program.call(max(compiled.program.calls))
    grouped = output_call.inputs[1].port
    origin = compiled.program.port(grouped).origin
    assert isinstance(origin, GroupOrigin)
    assert isinstance(
        compiled.program.port(origin.value_port).origin,
        GroupOrigin,
    )
