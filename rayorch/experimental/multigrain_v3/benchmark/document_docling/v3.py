"""通过 V3 执行 PDF→Pages→Docling→Document prototype。"""

from __future__ import annotations

from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from ...executor import Executor, RunResult
from .workload import (
    assemble_document,
    create_converter,
    parse_pages,
    render_pdf,
)


class RenderPages:
    """Expand UDF：把 PDF 动态展开为 PNG pages。"""

    def __init__(self, scale: float = 1.5) -> None:
        """保存 PDFium render scale。"""

        self.scale = scale

    def run(self, paths: list[str]) -> list[list[Any]]:
        """逐 PDF 返回按 ordinal 排列的 page records。"""

        return [render_pdf(path, scale=self.scale) for path in paths]


class ParsePages:
    """Map UDF：每个 actor 持有一个 persistent Docling converter。"""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        do_ocr: bool = True,
        do_table_structure: bool = True,
    ) -> None:
        """初始化 Docling page-image pipeline；模型只加载一次。"""

        self.converter = create_converter(
            input_format="image",
            device=device,
            num_threads=num_threads,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

    def run(self, pages: list[Any]) -> list[Any]:
        """通过 convert_all 处理一个跨 PDF page batch。"""

        return parse_pages(self.converter, pages)


class AssembleDocuments:
    """Reduce UDF：恢复 ordered pages 并输出文档 Markdown。"""

    def run(self, groups: list[list[Any]]) -> list[dict[str, Any]]:
        """逐文档调用共用 assembly。"""

        return [assemble_document(group) for group in groups]


class DoclingV3Pipeline(Pipeline):
    """页面级 Docling V3 pipeline。"""

    def __init__(
        self,
        *,
        scale: float = 1.5,
        page_replicas: int = 1,
        page_batch_size: int = 1,
        batch_scope: str = "elastic",
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        """配置 render、Docling page actors 与 ordered Reduce。"""

        self.render = (
            Expand(RenderPages)
            .pre_init(scale=scale)
            .ray_options(batch_size=1, replicas=1, num_cpus=1)
        )
        self.parse = (
            Map(ParsePages)
            .pre_init(device=device, num_threads=num_threads)
            .ray_options(
                batch_size=page_batch_size,
                batch_scope=batch_scope,
                replicas=page_replicas,
                num_cpus=max(1, num_threads),
            )
        )
        self.assemble = Reduce(AssembleDocuments).ray_options(
            batch_size=2,
            replicas=1,
            num_cpus=1,
        )

    def forward(self, documents):
        """声明 PDF page fan-out、跨文档 page packing 与 ordered fan-in。"""

        pages = self.render(documents)
        parsed = self.parse(pages)
        return self.assemble(anchor=documents, members=parsed)


def run_v3(
    paths: list[str],
    *,
    microbatch_size: int = 1,
    max_inflight_arenas: int = 2,
    **pipeline_options: Any,
) -> RunResult:
    """运行 Docling V3 prototype；调用方负责 Ray 与 Docling runtime_env。"""

    return Executor(
        DoclingV3Pipeline(**pipeline_options),
        microbatch_size=microbatch_size,
        max_inflight_arenas=max_inflight_arenas,
    ).run(paths)
