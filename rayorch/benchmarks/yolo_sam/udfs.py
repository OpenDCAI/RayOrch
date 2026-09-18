"""Business UDFs for the YOLO -> SAM image workload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LoadImages:
    def run(self, paths: list[str]) -> tuple[list[Any], list[dict[str, Any]]]:
        import cv2  # pyright: ignore[reportMissingImports]

        images = []
        metadata = []
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"cannot read image: {path}")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            metadata.append({"input_path": path})
        return images, metadata


class DetectObjects:
    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        confidence: float = 0.25,
    ) -> None:
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

        self.model = YOLO(model)
        self.device = device
        self.confidence = confidence

    def run(
        self,
        images: list[Any],
        metadata: list[dict[str, Any]],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        import cv2  # pyright: ignore[reportMissingImports]

        results = self.model(images, device=self.device, verbose=False)
        drawn_images = []
        output_metadata = []
        for image, result, item in zip(images, results, metadata, strict=True):
            drawn = image.copy()
            detections = 0
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, confidence in zip(boxes, confidences, strict=True):
                    if confidence < self.confidence:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    detections += 1
            drawn_images.append(drawn)
            output_metadata.append({**item, "yolo_boxes": detections})
        return drawn_images, output_metadata


class SegmentImages:
    def __init__(
        self,
        checkpoint: str,
        *,
        model_type: str = "vit_b",
        device: str = "cuda",
    ) -> None:
        from segment_anything import (  # pyright: ignore[reportMissingImports]
            SamAutomaticMaskGenerator,
            sam_model_registry,
        )

        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device=device)
        self.generator = SamAutomaticMaskGenerator(model)

    def run(
        self,
        images: list[Any],
        metadata: list[dict[str, Any]],
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        masks = []
        output_metadata = []
        for image, item in zip(images, metadata, strict=True):
            image_masks = self.generator.generate(image)
            masks.append(image_masks)
            output_metadata.append({**item, "sam_masks": len(image_masks)})
        return masks, output_metadata


class RenderMasks:
    def __init__(self, *, alpha: float = 0.35, seed: int = 1234) -> None:
        self.alpha = alpha
        self.seed = seed

    def run(
        self,
        images: list[Any],
        mask_groups: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        import numpy as np  # pyright: ignore[reportMissingImports]

        overlays = []
        output_metadata = []
        for index, (image, masks, item) in enumerate(
            zip(images, mask_groups, metadata, strict=True)
        ):
            random = np.random.default_rng(self.seed + index)
            overlay = image.astype(np.float32).copy()
            for annotation in masks:
                segmentation = annotation.get("segmentation")
                if segmentation is None:
                    continue
                color = random.integers(0, 256, size=(3,)).astype(np.float32)
                selection = segmentation.astype(bool)
                overlay[selection] = (
                    overlay[selection] * (1.0 - self.alpha)
                    + color * self.alpha
                )
            overlays.append(np.clip(overlay, 0, 255).astype(np.uint8))
            output_metadata.append({**item, "overlay_alpha": self.alpha})
        return overlays, output_metadata


class SaveImages:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)

    def run(
        self,
        images: list[Any],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        import cv2  # pyright: ignore[reportMissingImports]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for image, item in zip(images, metadata, strict=True):
            source = Path(item["input_path"])
            destination = self.output_dir / f"{source.stem}.overlay.jpg"
            ok = cv2.imwrite(
                str(destination),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            if not ok:
                raise RuntimeError(f"cannot save image: {destination}")
            outputs.append({**item, "output_path": str(destination.resolve())})
        return outputs


class SaveMetadata:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)

    def run(self, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for item in metadata:
            source = Path(item["input_path"])
            destination = self.output_dir / f"{source.stem}.json"
            destination.write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return metadata


__all__ = [
    "DetectObjects",
    "LoadImages",
    "RenderMasks",
    "SaveImages",
    "SaveMetadata",
    "SegmentImages",
]
