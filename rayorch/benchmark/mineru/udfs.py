"""MinerU UDFs used by the released RayOrch regression.

Heavy optional packages stay behind method calls, so importing the benchmark
registry does not import Flash-MinerU, vLLM, Pillow, or NVML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = os.environ.get(
    "RAYORCH_MINERU_MODEL",
    "./models/MinerU2.5-2509-1.2B",
)


class MinerUPdfToPages:
    """Render each PDF into ordered, first-class page records on a CPU actor."""

    def __init__(self, dpi: int = 200) -> None:
        """Store the PDF rendering resolution."""

        self.dpi = dpi

    def run(self, pdf_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Read each PDF and return page records in document order."""

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
    """Run MinerU's vLLM page extraction in one persistent GPU actor."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        gpu_memory_utilization: float = 0.8,
    ) -> None:
        """Load the model and create one MinerU client per actor."""

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
        """Execute MinerU's two-step extraction for one logical page batch."""

        return list(
            self.client.batch_two_step_extract(
                images=[page["img_pil"] for page in pages]
            )
        )


class PdfMetadata:
    """Create lightweight parent context without forwarding PDF payloads."""

    def run(self, paths: list[str]) -> list[str]:
        """Convert PDF paths to stems used for output directory names."""

        return [Path(path).stem for path in paths]


class MinerUAssembleDoc:
    """Assemble ordered page results without receiving the source PDF bytes."""

    def __init__(self, output_dir: str, parse_method: str = "vlm") -> None:
        """Store the output root and MinerU parse method."""

        self.output_dir = output_dir
        self.parse_method = parse_method

    def run(
        self,
        grouped_contents: list[list[Any]],
        grouped_pages: list[list[dict[str, Any]]],
        stems: list[str],
    ) -> list[dict[str, Any]]:
        """Write ordered OCR/page groups as Markdown, layout JSON, and metadata."""

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
            strict=True,
        ):
            if len(contents) != len(pages):
                raise ValueError("MinerU content and page groups must align")
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

__all__ = [
    "DEFAULT_MODEL",
    "MinerUAssembleDoc",
    "MinerUPdfToPages",
    "MinerUVlmOcrPage",
    "PdfMetadata",
]
