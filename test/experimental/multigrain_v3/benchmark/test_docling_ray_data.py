"""Docling Ray Data baseline 的显式 lineage 快速测试。"""

from __future__ import annotations

import numpy as np

from rayorch.experimental.multigrain_v3.benchmark.document_docling.ray_data import (
    AssembleDocumentGroups,
)


def test_ray_data_docling_group_restores_page_order() -> None:
    """Ray Data shuffle 后必须按 page ordinal 恢复 Markdown。"""

    output = AssembleDocumentGroups()(
        {
            "document_id": np.asarray([5, 5, 5]),
            "page_ordinal": np.asarray([2, 0, 1]),
            "markdown": np.asarray(["page-2", "page-0", "page-1"]),
        }
    )

    markdown = str(output["markdown"][0])
    assert markdown.index("page-0") < markdown.index("page-1")
    assert markdown.index("page-1") < markdown.index("page-2")
    assert int(output["pages"][0]) == 3
