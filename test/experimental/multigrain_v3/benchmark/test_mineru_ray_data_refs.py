"""Ray Data reference-only regroup baseline 的快速测试。"""

from __future__ import annotations

import numpy as np
import pytest

from rayorch.experimental.multigrain_v3.benchmark import (
    mineru_ray_data_refs,
)


def test_reference_group_sorts_and_deduplicates_blocks(
    monkeypatch,
    tmp_path,
) -> None:
    """同一 block 的多个 rows 只能 ray.get 一次，并按 ordinal 组装。"""

    seen: dict[str, object] = {"gets": 0}
    block_a = (0, 10)
    block_b = (1, 11)
    blocks = {
        block_a: (
            ({"page_id": 0}, "ocr-0"),
            ({"page_id": 2}, "ocr-2"),
        ),
        block_b: (({"page_id": 1}, "ocr-1"),),
    }

    class FakeAssembler:
        """记录 reference baseline 恢复出的 ordered values。"""

        def __init__(self, output_dir: str, parse_method: str) -> None:
            """接受 production 构造签名。"""

        def run(self, contents, pages, stems):
            """保存输入并返回摘要。"""

            seen["contents"] = contents
            seen["pages"] = pages
            return [{"pdf": stems[0], "pages": len(pages[0])}]

    class FakeRay:
        """提供计数型 ray.get。"""

        @staticmethod
        def get(ref):
            """解析 fake remote return 或 block ref。"""

            seen["gets"] = int(seen["gets"]) + 1
            return ref

    class FakeRemote:
        """模拟 actor method `.remote()`。"""

        def __init__(self, fn):
            """保存 callable。"""

            self.fn = fn

        def remote(self, *args):
            """同步调用。"""

            return self.fn(*args)

    class FakeStore:
        """模拟 payload store。"""

        def __init__(self, slot):
            """保存 store slot。"""

            self.slot = slot
            self.acquire = FakeRemote(
                lambda block_id: blocks[(slot, block_id)]
            )
            self.release = FakeRemote(lambda block_id: None)

    monkeypatch.setattr(
        mineru_ray_data_refs,
        "MinerUAssembleDoc",
        FakeAssembler,
    )
    monkeypatch.setitem(__import__("sys").modules, "ray", FakeRay)
    assembler = mineru_ray_data_refs.RayDataAssembleReferenceGroups(
        str(tmp_path),
        (FakeStore(0), FakeStore(1)),
    )

    output = assembler(
        {
            "parent_id": np.asarray([4, 4, 4]),
            "page_ordinal": np.asarray([2, 0, 1]),
            "pdf_path": np.asarray(["doc.pdf"] * 3),
            "store_slot": np.asarray([0, 0, 1]),
            "block_id": np.asarray([10, 10, 11]),
            "payload_row": np.asarray([1, 0, 0]),
        }
    )

    # 两个 block 各 acquire/get 一次，最后一次批量 get 等待 release。
    assert seen["gets"] == 5
    assert seen["contents"] == [["ocr-0", "ocr-1", "ocr-2"]]
    assert [page["page_id"] for page in seen["pages"][0]] == [0, 1, 2]
    assert int(output["payload_blocks"][0]) == 2


def test_reference_group_rejects_missing_ordinal(
    monkeypatch,
    tmp_path,
) -> None:
    """reference-only 优化不能削弱 ordered Reduce correctness。"""

    class UnusedAssembler:
        """invalid manifest 不应进入业务 assembler。"""

        def __init__(self, output_dir: str, parse_method: str) -> None:
            """接受构造参数。"""

    monkeypatch.setattr(
        mineru_ray_data_refs,
        "MinerUAssembleDoc",
        UnusedAssembler,
    )
    assembler = mineru_ray_data_refs.RayDataAssembleReferenceGroups(
        str(tmp_path),
        (),
    )

    with pytest.raises(ValueError, match="ordinals"):
        assembler(
            {
                "parent_id": np.asarray([1, 1]),
                "page_ordinal": np.asarray([0, 2]),
                "pdf_path": np.asarray(["doc.pdf", "doc.pdf"]),
                "store_slot": np.asarray([0, 0]),
                "block_id": np.asarray([1, 2]),
                "payload_row": np.asarray([0, 0]),
            }
        )
