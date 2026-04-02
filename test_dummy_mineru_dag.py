from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List

import ray

from rayorch import DagNewPipeline, Dispatch, RayModule


def chunked(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def flatten(nested: List[List[str]]) -> List[str]:
    out: List[str] = []
    for x in nested:
        out.extend(x)
    return out


class JitterSleepMixin:
    """Per-actor deterministic jittered sleep for timeline visualization."""

    def __init__(self, base_s: float, jitter_s: float = 0.25):
        self.base_s = float(base_s)
        self.jitter_s = float(jitter_s)
        self._rng = random.Random(os.getpid())
        self._calls = 0

    def _sleep_once(self, scale: float = 1.0) -> None:
        self._calls += 1
        delta = self._rng.uniform(-self.jitter_s, self.jitter_s)
        delay = max(0.02, self.base_s * scale + delta)
        time.sleep(delay)


class DummyBlock(dict):
    """ContentBlock-like dict subclass to mimic MinerU block payloads."""

    def __init__(self, type: str, bbox: list[float], content: str | None = None):
        super().__init__()
        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = None
        self["content"] = content

    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str) -> None:
        self["type"] = value

    @property
    def content(self) -> str | None:
        return self.get("content")

    @content.setter
    def content(self, value: str | None) -> None:
        self["content"] = value


class Pdf2ImageDummyOp(JitterSleepMixin):
    def __init__(self):
        super().__init__(base_s=0.15, jitter_s=0.05)

    def run(self, pdf_path_list: List[str]) -> List[List[dict]]:
        self._sleep_once(scale=max(1.0, 0.2 * len(pdf_path_list)))
        out: List[List[dict]] = []
        for pdf in pdf_path_list:
            page_n = 6 + (abs(hash(pdf)) % 12)
            pages = []
            for i in range(page_n):
                pages.append(
                    {
                        "pdf_path": pdf,
                        "page_id": i,
                        "page_width": 1000,
                        "page_height": 1400,
                        "img_pil": f"dummy_img<{pdf}:{i}>",
                    }
                )
            out.append(pages)
        return out


class LayoutDetectionDummyOp(JitterSleepMixin):
    def __init__(self):
        super().__init__(base_s=0.8, jitter_s=0.35)
        self._rng2 = random.Random(os.getpid() ^ 0xABCDEF)

    def run(self, image_dict_list: List[List[dict]]) -> List[List[List[DummyBlock]]]:
        # Emulate medium-heavy model stage with per-pdf/page variation.
        self._sleep_once(scale=max(1.0, 0.4 * len(image_dict_list)))
        output: List[List[List[DummyBlock]]] = []
        for pdf_pages in image_dict_list:
            blocks_list = []
            for page in pdf_pages:
                n_blocks = 3 + self._rng2.randint(0, 5)
                page_blocks = []
                for bi in range(n_blocks):
                    page_blocks.append(
                        DummyBlock(
                            type="text",
                            bbox=[10.0 + bi, 20.0 + bi, 300.0 + bi, 100.0 + bi],
                            content=f"layout<{page['page_id']}>#{bi}",
                        )
                    )
                blocks_list.append(page_blocks)
            output.append(blocks_list)
        return output


class OCRDummyOp(JitterSleepMixin):
    def __init__(self):
        super().__init__(base_s=1.1, jitter_s=0.45)

    def run(
        self,
        image_dict_list: List[List[dict]],
        blocks_list_per_pdf: List[List[List[DummyBlock]]],
    ) -> List[List[List[dict]]]:
        # Emulate heavy OCR stage.
        self._sleep_once(scale=max(1.0, 0.5 * len(image_dict_list)))
        out: List[List[List[dict]]] = []
        for pdf_pages, per_pdf_blocks in zip(image_dict_list, blocks_list_per_pdf):
            pdf_out = []
            for page, page_blocks in zip(pdf_pages, per_pdf_blocks):
                page_out = []
                for b in page_blocks:
                    page_out.append(
                        {
                            "type": b.type,
                            "bbox": b["bbox"],
                            "angle": b.get("angle"),
                            "content": f"ocr<{page['page_id']}>:{b.content}",
                        }
                    )
                pdf_out.append(page_out)
            out.append(pdf_out)
        return out


class Convert2MDDummyOp(JitterSleepMixin):
    def __init__(self):
        super().__init__(base_s=0.2, jitter_s=0.08)

    def run(self, model_results: List[List[List[dict]]], images: List[List[dict]]) -> List[str]:
        self._sleep_once(scale=max(1.0, 0.2 * len(images)))
        outs: List[str] = []
        for idx, image_pages in enumerate(images):
            pdf_path = image_pages[0]["pdf_path"]
            name = os.path.splitext(os.path.basename(pdf_path))[0]
            n_pages = len(model_results[idx])
            n_blocks = sum(len(p) for p in model_results[idx])
            outs.append(f"{name}.md (pages={n_pages}, blocks={n_blocks})")
        return outs


class DummyMineruDagPipeline(DagNewPipeline):
    """Same DAG shape as MinerU: pdf2img -> layout -> ocr -> img2md."""

    def __init__(
        self,
        *,
        replicas: int,
        max_batches_inflight: int,
    ):
        self.pdf2img = RayModule(Pdf2ImageDummyOp, replicas=1, num_gpus_per_replica=0.0).pre_init()
        self.layout = RayModule(
            LayoutDetectionDummyOp,
            replicas=replicas,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=max(1, replicas),
        ).pre_init()
        self.ocr = RayModule(
            OCRDummyOp,
            replicas=replicas,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            max_inflight=max(1, replicas),
        ).pre_init()
        self.img2md = RayModule(
            Convert2MDDummyOp,
            replicas=1,
            num_gpus_per_replica=0.0,
            dispatch_mode=Dispatch.BROADCAST,
        ).pre_init()

        super().__init__(max_batches_inflight=max(1, int(max_batches_inflight)))

    def forward(self, x):
        images = self.pdf2img(x)
        layout_items = self.layout(images)
        ocr_results = self.ocr(images, layout_items)
        return self.img2md(model_results=ocr_results, images=images)


def parse_args():
    p = argparse.ArgumentParser(description="Dummy MinerU-like DAG timeline probe.")
    p.add_argument("--num-pdfs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--inflight", type=int, default=4, help="max_batches_inflight")
    p.add_argument("--timeline", type=str, default="test_dummy_mineru_dag_timeline.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    pdfs = [f"/dummy/path/pdf_{i:04d}.pdf" for i in range(args.num_pdfs)]
    batches = chunked(pdfs, max(1, int(args.batch_size)))

    pipe = DummyMineruDagPipeline(
        replicas=max(1, int(args.replicas)),
        max_batches_inflight=max(1, int(args.inflight)),
    )
    t0 = time.perf_counter()
    per_batch = pipe(batches)
    elapsed = time.perf_counter() - t0
    all_out = flatten(per_batch)

    print(
        "done:",
        {
            "num_pdfs": len(pdfs),
            "num_batches": len(batches),
            "replicas": args.replicas,
            "inflight": args.inflight,
            "elapsed_sec": round(elapsed, 2),
            "outputs": len(all_out),
        },
    )
    ray.timeline(args.timeline)
    print(f"timeline: {args.timeline}")


if __name__ == "__main__":
    main()

