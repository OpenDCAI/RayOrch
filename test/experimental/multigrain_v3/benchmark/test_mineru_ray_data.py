"""裸 Ray Data MinerU baseline 中显式 lineage 与 ordered regroup 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from rayorch.experimental.multigrain_v3.benchmark import mineru_ray_data


def test_runtime_env_disables_vllm_usage_cpu_probe() -> None:
    env = mineru_ray_data._runtime_env("/tmp/flash")["env_vars"]
    assert env["VLLM_NO_USAGE_STATS"] == "1"
    assert env["DO_NOT_TRACK"] == "1"
    assert env["LOGURU_LEVEL"] == "WARNING"
    assert env["VLLM_LOGGING_LEVEL"] == "WARNING"


def test_rows_from_batch_preserves_object_columns() -> None:
    """numpy object columns 转 row 时不能丢失 parent、ordinal 或 payload。"""

    page0 = {"value": "p0"}
    page1 = {"value": "p1"}
    rows = mineru_ray_data._rows_from_batch(
        {
            "parent_id": np.asarray([3, 3]),
            "page_ordinal": np.asarray([0, 1]),
            "page": np.asarray([page0, page1], dtype=object),
        }
    )

    assert [int(row["parent_id"]) for row in rows] == [3, 3]
    assert [int(row["page_ordinal"]) for row in rows] == [0, 1]
    assert [row["page"] for row in rows] == [page0, page1]


def test_page_from_columnar_row_restores_mineru_record() -> None:
    """Arrow tensor/scalar columns 应能无损恢复 MinerU assemble 所需字段。"""

    image = np.zeros((12, 18, 3), dtype=np.uint8)
    page = mineru_ray_data._page_from_row(
        {
            "pdf_path": "doc.pdf",
            "page_id": 2,
            "image_rgb": image,
            "scale": 2.5,
            "page_width": 100,
            "page_height": 200,
            "pdf_len": 7,
        }
    )

    assert page["pdf_path"] == "doc.pdf"
    assert page["page_id"] == 2
    assert page["img_pil"].size == (18, 12)
    assert page["scale"] == 2.5
    assert page["pdf_len"] == 7


def test_jpeg_shuffle_payload_restores_page_without_raw_rgb() -> None:
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    encoded = mineru_ray_data._encode_page_jpeg(image)
    assert len(encoded) < image.nbytes
    page = mineru_ray_data._page_from_row(
        {
            "pdf_path": "doc.pdf",
            "page_id": 0,
            "image_jpeg": encoded,
            "scale": 2.0,
            "page_width": 96,
            "page_height": 64,
            "pdf_len": 1,
        }
    )
    assert page["img_pil"].size == (96, 64)
    assert page["img_pil"].mode == "RGB"


def test_real_ocr_output_uses_large_binary_shuffle_columns() -> None:
    import pyarrow as pa

    class FakeOcr:
        def run(self, pages):
            return [{"page": page["page_id"]} for page in pages]

    worker = object.__new__(mineru_ray_data.RayDataOcrPages)
    worker.ocr = FakeOcr()
    image = np.full((32, 48, 3), 128, dtype=np.uint8)
    output = worker(
        {
            "parent_id": np.asarray([1, 1], dtype=np.int64),
            "page_ordinal": np.asarray([0, 1], dtype=np.int64),
            "pdf_path": np.asarray(["doc.pdf", "doc.pdf"], dtype=object),
            "page_id": np.asarray([0, 1], dtype=np.int64),
            "image_rgb": np.asarray([image, image]),
            "scale": np.asarray([2.0, 2.0]),
            "page_width": np.asarray([48, 48]),
            "page_height": np.asarray([32, 32]),
            "pdf_len": np.asarray([2, 2]),
            "render_failed": np.asarray([False, False]),
        }
    )
    assert isinstance(output, pa.Table)
    assert output.schema.field("image_jpeg").type == pa.large_binary()
    assert output.schema.field("content_pickle").type == pa.large_binary()
    assert "image_rgb" not in output.column_names
    rows = output.to_pylist()
    assert [mineru_ray_data.pickle.loads(row["content_pickle"])["page"] for row in rows] == [0, 1]


def test_assemble_sorts_page_ordinals_before_reduce(monkeypatch, tmp_path) -> None:
    """Ray Data shuffle 后 group 行序不稳定，assembler 必须按 ordinal 恢复顺序。"""

    seen: dict[str, object] = {}

    class FakeAssembler:
        """记录收到的 group 顺序，避免单元测试加载真实 MinerU。"""

        def __init__(self, output_dir: str, parse_method: str) -> None:
            """记录构造参数。"""

            seen["init"] = (output_dir, parse_method)

        def run(self, contents, pages, stems):
            """返回可断言的轻量 document 摘要。"""

            seen["contents"] = contents
            seen["pages"] = pages
            seen["stems"] = stems
            return [{"pdf": stems[0], "pages": len(pages[0])}]

    monkeypatch.setattr(mineru_ray_data, "MinerUAssembleDoc", FakeAssembler)
    assembler = mineru_ray_data.RayDataAssemblePdf(str(tmp_path))
    output = assembler(
        {
            "parent_id": np.asarray([7, 7, 7]),
            "page_ordinal": np.asarray([2, 0, 1]),
            "pdf_path": np.asarray(
                ["/data/doc.pdf", "/data/doc.pdf", "/data/doc.pdf"],
                dtype=object,
            ),
            "page_id": np.asarray([2, 0, 1]),
            "image_rgb": np.asarray(
                [
                    np.zeros((12, 18, 3), dtype=np.uint8),
                    np.zeros((12, 18, 3), dtype=np.uint8),
                    np.zeros((12, 18, 3), dtype=np.uint8),
                ]
            ),
            "scale": np.asarray([2.0, 2.0, 2.0]),
            "page_width": np.asarray([100, 100, 100]),
            "page_height": np.asarray([200, 200, 200]),
            "pdf_len": np.asarray([3, 3, 3]),
            "content": np.asarray(["ocr-2", "ocr-0", "ocr-1"], dtype=object),
        }
    )

    assert seen["contents"] == [["ocr-0", "ocr-1", "ocr-2"]]
    assert [page["page_id"] for page in seen["pages"][0]] == [0, 1, 2]
    assert seen["stems"] == ["doc"]
    assert int(output["parent_id"][0]) == 7
    assert output["output"][0] == {"pdf": "doc", "pages": 3}


@pytest.mark.parametrize(
    "ordinals",
    (
        [0, 2],
        [0, 0],
        [1, 2],
    ),
)
def test_assemble_rejects_missing_or_duplicate_ordinals(
    monkeypatch,
    tmp_path,
    ordinals: list[int],
) -> None:
    """缺页或重复页不能被静默组装成看似成功的 PDF。"""

    class UnusedAssembler:
        """若 ordinal validator 正确，业务 assembler 不会被调用。"""

        def __init__(self, output_dir: str, parse_method: str) -> None:
            """接受 production 构造签名。"""

        def run(self, contents, pages, stems):
            """标记测试失败，因为 invalid group 不应进入业务逻辑。"""

            raise AssertionError("invalid ordinal group reached assembler")

    monkeypatch.setattr(mineru_ray_data, "MinerUAssembleDoc", UnusedAssembler)
    assembler = mineru_ray_data.RayDataAssemblePdf(str(tmp_path))
    size = len(ordinals)

    with pytest.raises(ValueError, match="ordinals"):
        assembler(
            {
                "parent_id": np.asarray([2] * size),
                "page_ordinal": np.asarray(ordinals),
                "pdf_path": np.asarray(["/data/doc.pdf"] * size, dtype=object),
                "page": np.asarray(list(range(size)), dtype=object),
                "content": np.asarray(list(range(size)), dtype=object),
            }
        )


def test_smoke_ocr_preserves_lineage_and_emits_stable_page_ids() -> None:
    """CPU smoke actor 应覆盖真实 OCR actor 的 batch/lineage ABI。"""

    output = mineru_ray_data.RayDataSmokeOcrPages()(
        {
            "parent_id": np.asarray([1, 2]),
            "page_ordinal": np.asarray([3, 0]),
            "pdf_path": np.asarray(["a.pdf", "b.pdf"], dtype=object),
            "page_id": np.asarray([3, 0]),
            "image_rgb": np.asarray(
                [
                    np.zeros((200, 100, 3), dtype=np.uint8),
                    np.zeros((200, 100, 3), dtype=np.uint8),
                ],
            ),
        }
    )

    assert output["parent_id"].tolist() == [1, 2]
    assert output["page_ordinal"].tolist() == [3, 0]
    assert output["content_page_id"].tolist() == [3, 0]


def test_smoke_assemble_returns_ordered_page_ids() -> None:
    """CPU smoke Reduce 应显式证明 shuffle 后的 group 被恢复成 ordinal 顺序。"""

    output = mineru_ray_data.RayDataSmokeAssemblePdf()(
        {
            "parent_id": np.asarray([4, 4, 4]),
            "page_ordinal": np.asarray([1, 2, 0]),
            "pdf_path": np.asarray(["doc.pdf"] * 3, dtype=object),
            "content_page_id": np.asarray([1, 2, 0]),
        }
    )

    assert output["output"][0]["page_ids"] == (0, 1, 2)
