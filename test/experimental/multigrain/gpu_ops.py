"""GPU dummy operators simulating a MinerU-style parse pipeline.

Shape: document -> (Expand) variable-length pages -> (Map, GPU) OCR/layout whose
cost scales with page content length -> (Reduce) assemble back to the document.

The GPU Map deliberately does *variable* work per page (a page's ``work`` = number
of square matmuls) so that an imbalanced 1:N fan-out produces uneven GPU load.
That is exactly the situation the relation-aware rebalancing is meant to fix.

Kept in an importable module (not a pytest file) so Ray workers can reconstruct
the operators from the ``OperatorRecipe.cls_ref`` stored in the IR.
"""
from __future__ import annotations

UNIT_MATMUL_N = 4096  # one "work unit" == one N x N matmul on the GPU

# Per-worker cached operands so wall-time scales with the *number of matmuls*
# (i.e. page content length) instead of per-call allocation overhead.
_OPERAND_CACHE: dict = {}


def gpu_matmul_work(reps: int, *, n: int = UNIT_MATMUL_N) -> float:
    """Run ``reps`` chained square matmuls on the GPU; return a checksum.

    Operands are allocated once per (device, n) and reused, so total time is
    proportional to ``reps`` -- that is what makes a variable-length page cost a
    variable amount of GPU time.
    """
    import torch

    key = (torch.cuda.current_device(), n)
    operands = _OPERAND_CACHE.get(key)
    if operands is None:
        operands = (
            torch.randn(n, n, device="cuda"),
            torch.randn(n, n, device="cuda"),
        )
        _OPERAND_CACHE[key] = operands
    acc, b = operands
    for _ in range(max(0, int(reps))):
        acc = acc @ b  # chained so iterations cannot be elided; shape stays n x n
    torch.cuda.synchronize()
    return float(acc.float().mean().item())


class PdfToPages:
    """Expand: one document dict -> a group of variable-length page dicts.

    Each input doc is ``{"name": str, "page_works": [int, ...]}``. Each page dict
    is ``{"doc": name, "page": index, "work": units}``.
    """

    def run(self, docs: list[dict]) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for doc in docs:
            groups.append(
                [
                    {"doc": doc["name"], "page": index, "work": int(work)}
                    for index, work in enumerate(doc["page_works"])
                ]
            )
        return groups


class OcrGpu:
    """Map (GPU): per page, do ``work`` matmuls, return a text string."""

    def run(self, pages: list[dict]) -> list[str]:
        texts: list[str] = []
        for page in pages:
            gpu_matmul_work(page["work"])
            texts.append(f"text[{page['doc']}#p{page['page']}#w{page['work']}]")
        return texts


class AssemblePages:
    """Reduce: pages grouped back to their document."""

    def run(self, docs: list[dict], grouped_texts: list[list[str]]) -> list[str]:
        return [
            f"{doc['name']}|pages={len(texts)}"
            for doc, texts in zip(docs, grouped_texts)
        ]


def page_work(page: dict) -> float:
    """Weight function for work-aware sharding: a page's GPU cost in units."""
    return float(page["work"])


def _build_ocr_actor():
    """Return a GPU-pinned, long-lived actor class for clean per-GPU timing.

    Defined lazily so importing this module does not require Ray. Each actor pays
    torch/cuda init once and then stays warm, giving cold-start-free measurements
    (unlike short-lived tasks) -- which is also how a real MinerU stage runs.
    """
    import time

    import ray

    @ray.remote(num_gpus=1)
    class GpuOcrActor:
        def warmup(self) -> int:
            gpu_matmul_work(30)
            return 1

        def ocr_timed(self, pages: list[dict]) -> tuple[float, int, int]:
            start = time.time()
            for page in pages:
                gpu_matmul_work(page["work"])
            return time.time() - start, len(pages), sum(p["work"] for p in pages)

    return GpuOcrActor
