"""Run a 512-PDF Flash-MinerU-style runtime demo with bad-row isolation."""
from __future__ import annotations

import argparse
import time
from collections import Counter

import ray

from rayorch import (
    BadRecordError,
    DagPipeline,
    RuntimeDagExecutor,
    RuntimeRayModule,
)


class Pdf2ImageDummy:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def run(self, pdf, meta):
        images = []
        for index, (path, item) in enumerate(zip(pdf, meta)):
            time.sleep(self.delay)
            if item["fault"] == "pdf":
                raise BadRecordError("PDF decode failed", index=index)
            item["pages"] = 2
            images.append([f"image:{path}:0", f"image:{path}:1"])
        return images, meta


class LayoutDummy:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def run(self, images, meta):
        layouts = []
        for pages, item in zip(images, meta):
            time.sleep(self.delay)
            item["layout_blocks"] = len(pages) * 4
            layouts.append([f"layout:{item['name']}:{page}" for page in range(len(pages))])
        return layouts, meta


class OcrDummy:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def run(self, images, layouts, meta):
        texts = []
        for pages, page_layouts, item in zip(images, layouts, meta):
            time.sleep(self.delay)
            if item["fault"] == "ocr":
                # A normal exception has no row index, so Runtime uses split-and-retry.
                raise RuntimeError("OCR model rejected this document")
            item["ocr_pages"] = len(page_layouts)
            texts.append([
                f"text:{image}:{layout}"
                for image, layout in zip(pages, page_layouts)
            ])
        return texts, meta


class MarkdownDummy:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def run(self, texts, meta):
        markdown = []
        for pages, item in zip(texts, meta):
            time.sleep(self.delay)
            markdown.append(
                f"{item['name']}.md pages={len(pages)} "
                f"blocks={item['layout_blocks']}"
            )
        return markdown


class FlashMineruDummyPipeline(DagPipeline):
    def __init__(
        self,
        delay: float,
        *,
        pdf_replicas: int,
        layout_replicas: int,
        ocr_replicas: int,
        output_replicas: int,
        stage_inflight: int,
    ) -> None:
        self.pdf2image = RuntimeRayModule(
            Pdf2ImageDummy,
            replicas=pdf_replicas,
            max_inflight=stage_inflight,
            num_outputs=2,
        ).pre_init(delay)
        self.layout = RuntimeRayModule(
            LayoutDummy,
            replicas=layout_replicas,
            max_inflight=stage_inflight,
            num_outputs=2,
        ).pre_init(delay)
        self.ocr = RuntimeRayModule(
            OcrDummy,
            replicas=ocr_replicas,
            max_inflight=stage_inflight,
            num_outputs=2,
        ).pre_init(delay)
        self.markdown = RuntimeRayModule(
            MarkdownDummy,
            replicas=output_replicas,
            max_inflight=stage_inflight,
            num_outputs=1,
        ).pre_init(delay)
        super().__init__()

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> list[str]:
        images, meta = self.pdf2image(pdf, meta)
        layouts, meta = self.layout(images, meta)
        texts, meta = self.ocr(images, layouts, meta)
        markdown = self.markdown(texts=texts, meta=meta)
        return markdown


def print_compiled_graph(executor: RuntimeDagExecutor) -> None:
    print("\n[compiled DAG]")
    graph = executor.graph
    for name in graph.topo_order:
        spec = graph.nodes[name]
        positional = [f"{ref.node}[{ref.index}]" for ref in spec.args]
        keywords = {
            key: f"{ref.node}[{ref.index}]"
            for key, ref in spec.kw_args.items()
        }
        print(
            f"  {name}: "
            f"parameters={list(spec.input_names)} "
            f"args={positional} "
            f"kwargs={keywords} "
            f"outputs={list(spec.output_names)} "
            f"replicas={spec.module._replicas} "
            f"max_inflight={spec.max_inflight}"
        )
    print(
        "  graph_outputs=",
        [f"{ref.node}[{ref.index}]" for ref in graph.graph_outputs],
    )


def fault_indices(count: int) -> tuple[set[int], set[int]]:
    """Choose deterministic PDF/OCR faults while remaining useful for small runs."""
    if count >= 512:
        return {37, 190, 511}, {74, 201, 405}
    candidates = list(range(count))
    pdf = {index for index in (1, count // 3) if index in candidates}
    ocr = {
        index
        for index in (max(0, count // 2), count - 1)
        if index in candidates and index not in pdf
    }
    return pdf, ocr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--stage-inflight", type=int, default=4)
    parser.add_argument("--pdf-replicas", type=int, default=2)
    parser.add_argument("--layout-replicas", type=int, default=4)
    parser.add_argument("--ocr-replicas", type=int, default=4)
    parser.add_argument("--output-replicas", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.005)
    args = parser.parse_args()

    pdf_faults, ocr_faults = fault_indices(args.pdfs)
    pdfs = [f"paper-{index:04d}.pdf" for index in range(args.pdfs)]
    meta = []
    for index in range(args.pdfs):
        fault = "pdf" if index in pdf_faults else "ocr" if index in ocr_faults else None
        meta.append({"name": f"paper-{index:04d}", "fault": fault})

    ray.init(ignore_reinit_error=True, num_cpus=16)
    try:
        pipeline = FlashMineruDummyPipeline(
            args.delay,
            pdf_replicas=args.pdf_replicas,
            layout_replicas=args.layout_replicas,
            ocr_replicas=args.ocr_replicas,
            output_replicas=args.output_replicas,
            stage_inflight=args.stage_inflight,
        )
        print(
            "simulated topology: "
            f"pdf={args.pdf_replicas}, "
            f"layout_gpu={args.layout_replicas}, "
            f"ocr_gpu={args.ocr_replicas}, "
            f"output={args.output_replicas}, "
            f"pipeline_inflight={args.max_inflight}, "
            f"stage_inflight={args.stage_inflight}"
        )
        print(
            f"injected faults: pdf={sorted(pdf_faults)} "
            f"ocr={sorted(ocr_faults)}"
        )

        with RuntimeDagExecutor(
            pipeline,
            batch_size=args.batch_size,
            max_batches_inflight=args.max_inflight,
            dataset="flash-mineru",
        ) as executor:
            print_compiled_graph(executor)

            started = time.perf_counter()
            results = executor.run(pdf=pdfs, meta=meta)
            elapsed = time.perf_counter() - started

            print(
                f"\nprocessed={args.pdfs} batches={len(results)} "
                f"batch_size={args.batch_size} elapsed={elapsed:.2f}s"
            )

            all_errors = []
            for batch_index, result in enumerate(results):
                start = batch_index * args.batch_size
                end = min(start + args.batch_size, args.pdfs)
                print(
                    f"\n[batch {batch_index:02d}] input_rows={start}:{end} "
                    f"healthy={len(result.batch)} "
                    f"errors={len(result.quarantined)}"
                )
                for error in result.quarantined:
                    path = result.trace(error.path_id)
                    item_meta = error.values.get("meta", {})
                    document = error.values.get("pdf") or item_meta.get("name")
                    print(
                        f"  row={error.row_id} document={document} "
                        f"failed_op={error.op} "
                        f"successful_path={path} "
                        f"error={error.error!r}"
                    )
                all_errors.extend(result.quarantined)

            healthy = sum(len(result.batch) for result in results)
            by_op = Counter(error.op for error in all_errors)
            expected_errors = len(pdf_faults) + len(ocr_faults)
            print("\n[summary]")
            print(f"  healthy={healthy}")
            print(f"  quarantined={len(all_errors)}")
            print(f"  errors_by_op={dict(by_op)}")
            print(f"  accounted_for={healthy + len(all_errors)}/{args.pdfs}")

            assert healthy + len(all_errors) == args.pdfs
            assert len(all_errors) == expected_errors
            assert by_op == {
                "pdf2image": len(pdf_faults),
                "ocr": len(ocr_faults),
            }
            print("  validation=PASS")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
