"""Docling page workload 的框架无关快速测试。"""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.workload import (
    DoclingPage,
    DoclingPageResult,
    assemble_document,
    parse_pages,
)


def test_assemble_document_restores_page_order() -> None:
    """Docling page completion order 不应改变最终 Markdown 顺序。"""

    output = assemble_document(
        [
            DoclingPageResult(2, "page-2"),
            DoclingPageResult(0, "page-0"),
            DoclingPageResult(1, "page-1"),
        ]
    )

    assert output["page_ordinals"] == (0, 1, 2)
    assert output["markdown"].index("page-0") < output["markdown"].index(
        "page-1"
    )
    assert output["markdown"].index("page-1") < output["markdown"].index(
        "page-2"
    )


def test_assemble_document_rejects_missing_page() -> None:
    """缺页不能被静默视为成功文档。"""

    with pytest.raises(ValueError, match="ordinals"):
        assemble_document(
            [
                DoclingPageResult(0, "page-0"),
                DoclingPageResult(2, "page-2"),
            ]
        )


def test_parse_pages_preserves_batch_cardinality_and_ordinals(
    monkeypatch,
) -> None:
    """Docling convert_all completion 应与输入 page ordinal 逐项绑定。"""

    class FakeDocument:
        """提供 Docling document 的最小 Markdown API。"""

        def __init__(self, value: str) -> None:
            """保存测试文本。"""

            self.value = value

        def export_to_markdown(self) -> str:
            """返回测试文本。"""

            return self.value

    class FakeResult:
        """模拟 ConversionResult。"""

        def __init__(self, value: str) -> None:
            """创建 fake document。"""

            self.document = FakeDocument(value)

    class FakeConverter:
        """记录 convert_all 输入并返回相同 cardinality。"""

        def convert_all(self, streams):
            """按输入文件名生成 fake results。"""

            return [FakeResult(stream.name) for stream in streams]

    class FakeStream:
        """避免快速测试依赖 Docling 安装。"""

        def __init__(self, *, name, stream) -> None:
            """保存文件名与字节流。"""

            self.name = name
            self.stream = stream

    import sys
    import types

    io_module = types.ModuleType("docling_core.types.io")
    io_module.DocumentStream = FakeStream
    monkeypatch.setitem(sys.modules, "docling_core.types.io", io_module)

    outputs = parse_pages(
        FakeConverter(),
        [
            DoclingPage("doc.pdf", 0, b"a"),
            DoclingPage("doc.pdf", 1, b"b"),
        ],
    )

    assert [output.page_ordinal for output in outputs] == [0, 1]
    assert outputs[0].markdown.endswith("p0.png")
    assert outputs[1].markdown.endswith("p1.png")
