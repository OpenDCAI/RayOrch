"""MinerU UDF definitions shared by every benchmark adapter.

Heavy optional packages stay behind method calls, so importing the benchmark
registry does not import Flash-MinerU, vLLM, Pillow, or NVML.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FLASH_REPO = os.environ.get("RAYORCH_MINERU_REPO", "./Flash-mineru")
DEFAULT_MODEL = os.environ.get(
    "RAYORCH_MINERU_MODEL",
    "./models/MinerU2.5-2509-1.2B",
)


class MinerUPdfToPages:
    """在 CPU actor 中把每个 PDF 渲染为 first-class page records。"""

    def __init__(self, dpi: int = 200) -> None:
        """保存 PDF 渲染分辨率。"""

        self.dpi = dpi

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        """逐 PDF 读取字节并返回按 page ordinal 排列的页面记录。"""

        from flash_mineru.mineru_core.utils.pdf_image_tools import (  # pyright: ignore[reportMissingImports]
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

        from mineru_vl_utils import MinerUClient  # pyright: ignore[reportMissingImports]
        from vllm import LLM  # pyright: ignore[reportMissingImports]

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

        from flash_mineru.mineru_core.data.data_reader_writer import (  # pyright: ignore[reportMissingImports]
            FileBasedDataWriter,
        )
        from flash_mineru.mineru_core.engine.model_output_to_middle_json import (  # pyright: ignore[reportMissingImports]
            result_to_middle_json,
        )
        from flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent import (  # pyright: ignore[reportMissingImports]
            union_make as vlm_union_make,
        )
        from flash_mineru.mineru_core.utils.enum_class import MakeMode  # pyright: ignore[reportMissingImports]

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

        import psutil  # pyright: ignore[reportMissingModuleSource]

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
        import pynvml  # pyright: ignore[reportMissingImports]

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

    if os.environ.get("RAYORCH_PACKAGED_RUNTIME") == "1":
        return {}
    current = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        path for path in (flash_repo, os.getcwd(), current) if path
    )
    return {"env_vars": {"PYTHONPATH": pythonpath}}


__all__ = [
    "DEFAULT_FLASH_REPO",
    "DEFAULT_MODEL",
    "GpuSample",
    "MinerUAssembleDoc",
    "MinerUPdfToPages",
    "MinerUVlmOcrPage",
    "PdfMetadata",
    "ResourceSampler",
    "_runtime_env",
]
