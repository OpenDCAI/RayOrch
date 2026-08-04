"""不占 GPU 的真实 PDF/render/ObjectRef/Reduce 物理链 smoke。

该入口复用 MinerU 的真实 PDF renderer，但用轻量 PageSignature 替代 VLM。它不能替代
4-PDF MinerU correctness gate，只用于在 GPU 被其他实验占用时提前验证 PIL payload、
粗块 cache、多 Arena 和 ordered Reduce。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from ...multigrain_v3.benchmark.mineru import (
    DEFAULT_FLASH_REPO,
    MinerUPdfToPages,
    PdfMetadata,
    _runtime_env,
)
from .. import functional as F
from ..api import Pipeline, RayModule
from ..ray_executor import RayExecutor


class PageSignature:
    """读取真实 Page record，但只输出轻量、确定性签名。"""

    def run(self, pages):
        """保留 pdf stem、page ordinal 和 image size。"""

        return [
            {
                "pdf": Path(page["pdf_path"]).stem,
                "page_id": int(page["page_id"]),
                "image_size": tuple(page["img_pil"].size),
            }
            for page in pages
        ]


class AssembleSignatures:
    """把 ordered page signatures 汇总回 PDF。"""

    def run(self, stems, groups):
        """输出 page count 和连续 ordinal，便于严格检查。"""

        return [
            {
                "pdf": stem,
                "pages": len(group),
                "ordinals": [item["page_id"] for item in group],
            }
            for stem, group in zip(stems, groups)
        ]


class MinerUCpuSmokePipeline(Pipeline):
    """真实 Render→Expand→PageSignature→Reduce 的 CPU-only Pipeline。"""

    def __init__(self, runtime_env) -> None:
        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(replicas=2, batch_size=1, num_cpus=1, runtime_env=runtime_env)
        )
        self.signature = RayModule(PageSignature).ray_options(
            replicas=2,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
        )
        self.metadata = RayModule(PdfMetadata).ray_options(
            replicas=1,
            batch_size=16,
            num_cpus=1,
            runtime_env=runtime_env,
        )
        self.assemble = RayModule(AssembleSignatures).ray_options(
            replicas=1,
            batch_size=8,
            num_cpus=1,
            runtime_env=runtime_env,
        )

    def forward(self, pdfs):
        """显式声明真实 page fan-out 与 ordered parent reduce。"""

        pages = F.expand(self.render(pdfs))
        groups = F.reduce(self.signature(pages))
        stems = self.metadata(pdfs)
        return self.assemble(stems, groups)


def run_smoke(*, flash_repo: str, limit: int = 4) -> dict:
    """运行 CPU smoke，并校验每个 PDF 的 ordinal 连续性。"""

    flash_repo = os.path.abspath(flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[:limit]
    if len(pdfs) != limit:
        raise FileNotFoundError(f"expected {limit} PDFs under {flash_repo}")
    runtime_env = _runtime_env(flash_repo)
    with RayExecutor(
        MinerUCpuSmokePipeline(runtime_env),
        address="auto",
        ray_init_kwargs={"runtime_env": runtime_env},
    ) as executor:
        result = executor.run(pdfs, arena_size=2, max_in_flight=2)

    for output in result.outputs:
        if output["ordinals"] != list(range(output["pages"])):
            raise AssertionError(f"non-contiguous page order: {output['pdf']}")
    return {
        "engine": "multigrain_v3_3_cpu_smoke",
        "pdfs": len(result.outputs),
        "pages": sum(output["pages"] for output in result.outputs),
        "rpc_count": result.rpc_count,
        "actor_count": result.actor_count,
        "active_arenas_high_watermark": result.max_active_arenas,
        "elapsed_s": result.elapsed_s,
        "outputs": result.outputs,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(flash_repo=args.flash_repo, limit=args.limit), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())


__all__ = ["MinerUCpuSmokePipeline", "run_smoke"]
