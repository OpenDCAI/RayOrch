"""通过 V3 public Pipeline API 运行真实 Flash-MinerU 回归。

本模块是手工 4×H20 benchmark 入口，只负责构造真实 UDF、采集 observation 数据和写
复现实验产物，不参与 V3 runtime correctness。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..api import Expand, Map, Pipeline, Reduce
from ..executor import Executor


DEFAULT_FLASH_REPO = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/workspace/Flash-mineru"
)
DEFAULT_MODEL = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/model/MinerU2.5-2509-1.2B"
)
V25_ELASTIC_MEDIAN_S = 581.509


class MinerUPdfToPages:
    """在 CPU actor 中把每个 PDF 渲染为 first-class page records。"""

    def __init__(self, dpi: int = 200) -> None:
        """保存 PDF 渲染分辨率。"""

        self.dpi = dpi

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        """逐 PDF 读取字节并返回按 page ordinal 排列的页面记录。"""

        from flash_mineru.mineru_core.utils.pdf_image_tools import (
            load_images_from_pdf,
        )

        groups = []
        for path in pdf_paths:
            with open(path, "rb") as handle:
                pdf_bytes = handle.read()
            images, pdf_doc = load_images_from_pdf(pdf_bytes, dpi=self.dpi)
            pages = []
            for page_id, image in enumerate(images):
                width, height = map(int, pdf_doc[page_id].get_size())
                pages.append(
                    {
                        "pdf_path": path,
                        "page_id": page_id,
                        "img_pil": image["img_pil"],
                        "scale": image.get("scale"),
                        "page_width": width,
                        "page_height": height,
                        "pdf_len": len(images),
                    }
                )
            pdf_doc.close()
            groups.append(pages)
        return groups


class MinerUVlmOcrPage:
    """在单个 GPU persistent actor 中运行真实 MinerU vLLM 页面抽取。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
    ) -> None:
        """加载 vLLM 模型并创建 MinerUClient；每个 actor 只初始化一次。"""

        from mineru_vl_utils import MinerUClient
        from vllm import LLM

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self.llm,
        )

    def run(self, pages: list[dict[str, Any]]) -> list[Any]:
        """对一个 logical page batch 执行两阶段 VLM extraction。"""

        return list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )


class PdfMetadata:
    """生成 Reduce UDF 所需的轻量 parent context，避免传输 PDF anchor payload。"""

    def run(self, paths: list[str]) -> list[str]:
        """把 PDF path 转为用于输出目录命名的 stem。"""

        return [Path(path).stem for path in paths]


class MinerUAssembleDoc:
    """在不接收 PDF anchor payload 的情况下组装有序页面结果。"""

    def __init__(self, output_dir: str, parse_method: str = "vlm") -> None:
        """保存输出根目录和 MinerU parse method。"""

        self.output_dir = output_dir
        self.parse_method = parse_method

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        """把 ordered OCR/page GROUPs 写成 Markdown、layout JSON 和摘要。"""

        from flash_mineru.mineru_core.data.data_reader_writer import (
            FileBasedDataWriter,
        )
        from flash_mineru.mineru_core.engine.model_output_to_middle_json import (
            result_to_middle_json,
        )
        from flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent import (
            union_make as vlm_union_make,
        )
        from flash_mineru.mineru_core.utils.enum_class import MakeMode

        outputs = []
        for contents, pages, stem in zip(
            grouped_contents,
            grouped_pages,
            stems,
        ):
            markdown_dir = Path(self.output_dir) / stem / self.parse_method
            image_dir = markdown_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_writer = FileBasedDataWriter(str(image_dir))
            markdown_writer = FileBasedDataWriter(str(markdown_dir))
            middle = result_to_middle_json(
                list(contents),
                list(pages),
                image_writer,
            )
            markdown = vlm_union_make(
                middle["pdf_info"],
                MakeMode.MM_MD,
                "images",
            )
            markdown_writer.write_string(f"{stem}.md", markdown)
            (markdown_dir / "layout.json").write_text(
                json.dumps(middle, indent=2),
                encoding="utf-8",
            )
            outputs.append(
                {
                    "pdf": stem,
                    "md_path": str(
                        (markdown_dir / f"{stem}.md").resolve()
                    ),
                    "chars": len(markdown),
                    "pages": len(pages),
                }
            )
        return outputs


class MinerUV3Pipeline(Pipeline):
    """用于 V3 性能回归的真实 PDF→Page→OCR→Document DAG。"""
    def __init__(
        self,
        *,
        output_dir: str,
        mode: str,
        model: str,
        replicas: int,
        batch_size: int,
        max_batch_wait_ms: float,
        gpu_memory_utilization: float,
        render_replicas: int,
        reduce_replicas: int,
        runtime_env: dict[str, Any],
    ) -> None:
        """按 benchmark 参数配置 render、metadata、OCR 和 assemble Stages。"""

        scope = "elastic" if mode == "elastic" else "parent_bound"
        self.render = (
            Expand(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(
                replicas=render_replicas,
                batch_size=1,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.metadata = Map(PdfMetadata).ray_options(
            replicas=1,
            batch_size=32,
            num_cpus=1,
            runtime_env=runtime_env,
        )
        self.ocr = (
            Map(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                max_batch_wait_ms=max_batch_wait_ms,
                batch_scope=scope,
                num_gpus=1.0,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )
        self.assemble = (
            Reduce(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                replicas=reduce_replicas,
                batch_size=4,
                num_cpus=1,
                runtime_env=runtime_env,
            )
        )

    def forward(self, pdfs):
        """声明 semantic-only PDF anchor 与 page-level elastic OCR DAG。"""

        pages = self.render(pdfs)
        contents = self.ocr(pages)
        metadata = self.metadata(pdfs)
        return self.assemble(
            anchor=pdfs,
            members=contents,
            pages=pages,
            context=metadata,
        )


@dataclass(frozen=True, slots=True)
class GpuSample:
    """只用于观测的 GPU utilization 与 memory sample。"""
    monotonic_s: float
    utilization: tuple[int | None, ...]
    memory_used: tuple[int, ...]


class ResourceSampler:
    """后台采集 driver RSS/GPU 指标，不拥有任何调度 authority。"""
    def __init__(self, interval_s: float) -> None:
        """初始化采样间隔、driver process 和后台线程。"""

        import psutil

        self.interval_s = interval_s
        self.process = psutil.Process()
        self.driver_start = int(self.process.memory_info().rss)
        self.driver_peak = self.driver_start
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """启动 daemon observation thread。"""

        self._thread.start()

    def stop(self) -> tuple[int, int, tuple[GpuSample, ...]]:
        """停止采样并返回 driver RSS 起点/峰值和 GPU samples。"""

        self._stop.set()
        self._thread.join(timeout=max(2, self.interval_s * 2))
        return (
            self.driver_start,
            self.driver_peak,
            tuple(self.samples),
        )

    def _run(self) -> None:
        """按固定间隔采集 driver RSS 与 GPU 状态，失败时跳过本次样本。"""

        while not self._stop.wait(self.interval_s):
            try:
                self.driver_peak = max(
                    self.driver_peak,
                    int(self.process.memory_info().rss),
                )
                self.samples.append(_gpu_sample())
            except Exception:
                continue


def _gpu_sample() -> GpuSample:
    """best-effort 采集所有可见 GPU 的 utilization 和 used memory。"""

    try:
        import pynvml

        pynvml.nvmlInit()
        utilization = []
        memory = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                utilization.append(
                    int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                )
            except Exception:
                utilization.append(None)
            memory.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        pynvml.nvmlShutdown()
        return GpuSample(time.monotonic(), tuple(utilization), tuple(memory))
    except Exception:
        return GpuSample(time.monotonic(), (), ())


def _runtime_env(flash_repo: str) -> dict[str, Any]:
    """构造让 Ray actors 能导入 Flash-MinerU 与当前仓库的 runtime_env。"""

    current = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path for path in (flash_repo, os.getcwd(), current) if path
    )
    return {"env_vars": {"PYTHONPATH": pythonpath}}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """运行一次真实 MinerU benchmark，并写 timeline/GPU/result artifacts。"""

    import ray

    flash_repo = os.path.abspath(args.flash_repo)
    pdfs = sorted(glob.glob(os.path.join(flash_repo, "*.pdf")))[: args.limit]
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found under {flash_repo}")
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=args.num_cpus,
            num_gpus=args.replicas,
            object_store_memory=int(args.object_store_gb * 1024**3),
            include_dashboard=False,
        )

    pipeline = MinerUV3Pipeline(
        output_dir=os.path.abspath(args.output_dir),
        mode=args.mode,
        model=args.model,
        replicas=args.replicas,
        batch_size=args.batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        gpu_memory_utilization=args.gpu_memory_utilization,
        render_replicas=args.render_replicas,
        reduce_replicas=args.reduce_replicas,
        runtime_env=_runtime_env(flash_repo),
    )
    ocr_stage = next(
        stage.id
        for stage in pipeline.compile().dag.stages
        if stage.udf is not None and stage.udf.target is MinerUVlmOcrPage
    )
    sampler = ResourceSampler(args.rss_interval_s)
    sampler.start()
    started = time.perf_counter()
    try:
        result = Executor(
            pipeline,
            microbatch_size=args.microbatch_size,
            max_inflight_arenas=args.max_inflight_arenas,
        ).run(pdfs)
    finally:
        driver_start, driver_peak, gpu_samples = sampler.stop()
    outputs = result.get()
    end_to_end = time.perf_counter() - started
    pages = sum(int(output["pages"]) for output in outputs)
    ocr_events = [
        event
        for event in result.timeline
        if event.stage == ocr_stage
        and event.worker_started_at is not None
        and event.worker_finished_at is not None
    ]
    busy = sum(
        event.worker_finished_at - event.worker_started_at
        for event in ocr_events
    )
    if ocr_events:
        start = min(event.worker_started_at for event in ocr_events)
        stop = max(event.worker_finished_at for event in ocr_events)
        capacity = max(stop - start, 1e-9) * args.replicas
        bubble = max(0.0, 1.0 - busy / capacity)
    else:
        bubble = 0.0
    gpu_peak = tuple(
        max((sample.memory_used[index] for sample in gpu_samples), default=0)
        for index in range(args.replicas)
    )
    payload = {
        "engine": "multigrain_v3",
        "mode": args.mode,
        "n_pdf": len(pdfs),
        "pages": pages,
        "docs": len(outputs),
        "batch_size": args.batch_size,
        "replicas": args.replicas,
        "microbatch_size": args.microbatch_size,
        "max_inflight_arenas": args.max_inflight_arenas,
        "startup_s": round(result.metrics["startup_time_s"], 3),
        "measured_wall_s": round(result.metrics["measured_wall_time_s"], 3),
        "end_to_end_wall_s": round(end_to_end, 3),
        "pages_per_s": round(
            pages / result.metrics["measured_wall_time_s"],
            4,
        ),
        "rpc_count": result.metrics["rpc_count"],
        "grains_per_rpc": result.metrics["grains_per_rpc"],
        "batch_fill_ratio": result.metrics["batch_fill_ratio"],
        "tail_or_recovery_rpc_fraction": result.metrics[
            "tail_or_recovery_rpc_fraction"
        ],
        "active_arenas_high_watermark": result.metrics[
            "active_arenas_high_watermark"
        ],
        "ocr_bubble_ratio": round(bubble, 4),
        "driver_rss_start": driver_start,
        "driver_rss_peak": driver_peak,
        "gpu_memory_peak": gpu_peak,
        "speedup_vs_v25_elastic": (
            V25_ELASTIC_MEDIAN_S / result.metrics["measured_wall_time_s"]
            if len(pdfs) == 368
            else None
        ),
        "output_dir": os.path.abspath(args.output_dir),
    }
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dispatch_timeline.jsonl").open("w") as handle:
        for event in result.timeline:
            handle.write(json.dumps(asdict(event)) + "\n")
    with (root / "gpu_samples.jsonl").open("w") as handle:
        for sample in gpu_samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    with Path(args.result_jsonl).open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造 V3 MinerU 手工 benchmark CLI 参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("elastic", "parent_bound"), default="elastic")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-inflight-arenas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batch-wait-ms", type=float, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--render-replicas", type=int, default=4)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=32)
    parser.add_argument("--object-store-gb", type=float, default=100)
    parser.add_argument("--rss-interval-s", type=float, default=1)
    parser.add_argument("--flash-repo", default=DEFAULT_FLASH_REPO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析 CLI、执行 benchmark，并把摘要 JSON 打印到 stdout。"""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
