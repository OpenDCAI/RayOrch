"""裸 Ray Data 的 page-level Docling baseline。"""

from __future__ import annotations

from typing import Any

from .workload import (
    DoclingPage,
    DoclingPageResult,
    assemble_document,
    create_converter,
    parse_pages,
    render_pdf,
)


def _rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """把 Ray Data numpy batch 转为普通 rows。"""

    if not batch:
        return []
    size = len(next(iter(batch.values())))
    return [
        {name: values[index] for name, values in batch.items()}
        for index in range(size)
    ]


class RenderPdfPages:
    """Ray Data flat_map actor：显式附加 document/page lineage。"""

    def __init__(self, scale: float = 1.5) -> None:
        """保存 PDF render scale。"""

        self.scale = scale

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """把单个 PDF 展开为 PNG byte rows。"""

        return [
            {
                "document_id": int(row["document_id"]),
                "pdf_path": str(row["pdf_path"]),
                "page_ordinal": page.page_ordinal,
                "png_bytes": page.png_bytes,
            }
            for page in render_pdf(
                str(row["pdf_path"]),
                scale=self.scale,
            )
        ]


class ParsePageBatches:
    """Ray Data actor：持有一个 persistent Docling image converter。"""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        do_ocr: bool = True,
        do_table_structure: bool = True,
    ) -> None:
        """按与 V3 相同的配置初始化 Docling。"""

        self.converter = create_converter(
            input_format="image",
            device=device,
            num_threads=num_threads,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """调用共用 Docling convert_all，并保留显式 lineage。"""

        import numpy as np

        rows = _rows(batch)
        results = parse_pages(
            self.converter,
            [
                DoclingPage(
                    pdf_path=str(row["pdf_path"]),
                    page_ordinal=int(row["page_ordinal"]),
                    png_bytes=bytes(row["png_bytes"]),
                )
                for row in rows
            ],
        )
        return {
            "document_id": batch["document_id"],
            "page_ordinal": batch["page_ordinal"],
            "markdown": np.asarray(
                [result.markdown for result in results],
            ),
        }


class AssembleDocumentGroups:
    """Ray Data map_groups actor：按 page ordinal 恢复文档。"""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """校验并输出 Arrow-friendly document summary columns。"""

        import numpy as np

        rows = _rows(batch)
        output = assemble_document(
            [
                DoclingPageResult(
                    page_ordinal=int(row["page_ordinal"]),
                    markdown=str(row["markdown"]),
                )
                for row in rows
            ]
        )
        return {
            "document_id": np.asarray(
                [int(rows[0]["document_id"])],
                dtype="int64",
            ),
            "pages": np.asarray([int(output["pages"])], dtype="int64"),
            "markdown": np.asarray([str(output["markdown"])]),
            "markdown_chars": np.asarray(
                [int(output["markdown_chars"])],
                dtype="int64",
            ),
        }


def build_dataset(
    paths: list[str],
    *,
    scale: float = 1.5,
    render_replicas: int = 1,
    page_replicas: int = 1,
    page_batch_size: int = 4,
    device: str = "cpu",
    num_threads: int = 4,
):
    """构造与 V3 相同业务阶段的 lazy Ray Data DAG。"""

    import ray

    source = ray.data.from_items(
        [
            {"document_id": index, "pdf_path": path}
            for index, path in enumerate(paths)
        ],
        override_num_blocks=max(
            1,
            min(len(paths), max(render_replicas, page_replicas)),
        ),
    )
    pages = source.flat_map(
        RenderPdfPages,
        concurrency=render_replicas,
        num_cpus=1,
        fn_constructor_kwargs={"scale": scale},
    )
    parsed = pages.map_batches(
        ParsePageBatches,
        batch_size=page_batch_size,
        batch_format="numpy",
        concurrency=page_replicas,
        num_cpus=max(1, num_threads),
        fn_constructor_kwargs={
            "device": device,
            "num_threads": num_threads,
        },
    )
    return parsed.groupby(
        "document_id",
        num_partitions=max(1, min(len(paths), render_replicas)),
    ).map_groups(
        AssembleDocumentGroups,
        batch_format="numpy",
        concurrency=render_replicas,
        num_cpus=1,
    )


def run_ray_data(
    paths: list[str],
    **options: Any,
) -> tuple[dict[str, Any], ...]:
    """执行 Ray Data baseline 并按 document_id 恢复 source order。"""

    rows = build_dataset(paths, **options).take_all()
    rows.sort(key=lambda row: int(row["document_id"]))
    return tuple(
        {
            "pages": int(row["pages"]),
            "page_ordinals": tuple(range(int(row["pages"]))),
            "markdown": str(row["markdown"]),
            "markdown_chars": int(row["markdown_chars"]),
        }
        for row in rows
    )
