import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _same_len(a: List[Any], b: List[Any]) -> None:
    if len(a) != len(b):
        raise ValueError("batch/meta length mismatch")


class ImageLoadOp:
    """paths -> (images_rgb, meta_list)"""

    def run(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        import cv2

        images: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(f"imread failed: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            meta.append({"input_path": p})
        return images, meta


class YOLODrawOp:
    """(images, meta) -> (images_drawn, meta)"""

    def __init__(self, weight_path: str, device: str = "cuda", conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weight_path)
        self.model.to(device)
        self.device = device
        self.conf = conf

    def run(
        self,
        images: List[np.ndarray],
        meta: List[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        import cv2

        _same_len(images, meta)
        results = self.model(images, device=self.device)

        out_images: List[np.ndarray] = []
        out_meta: List[Dict[str, Any]] = []

        for img, r, m in zip(images, results, meta):
            drawn = img.copy()
            kept = 0

            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, cf in zip(boxes, confs):
                    if cf < self.conf:
                        continue
                    kept += 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mm = dict(m)
            mm["yolo_pid"] = os.getpid()
            mm["yolo_boxes"] = kept
            out_images.append(drawn)
            out_meta.append(mm)

        return out_images, out_meta


class SAMOp:
    """(images, meta) -> (masks_list, meta)"""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.gen = SamAutomaticMaskGenerator(sam)

    def run(
        self,
        images: List[np.ndarray],
        meta: List[Dict[str, Any]],
    ) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        _same_len(images, meta)

        masks_list: List[List[Dict[str, Any]]] = []
        out_meta: List[Dict[str, Any]] = []

        for img, m in zip(images, meta):
            masks = self.gen.generate(img)
            mm = dict(m)
            mm["sam_pid"] = os.getpid()
            mm["sam_masks"] = len(masks)
            masks_list.append(masks)
            out_meta.append(mm)

        return masks_list, out_meta


class OutputSAMMasksOp:
    """(images, masks, meta) -> (overlay_images, meta)"""

    def __init__(self, alpha: float = 0.35, seed: int = 1234):
        self.alpha = alpha
        self.seed = seed

    def run(
        self,
        images: List[np.ndarray],
        masks_list: List[List[Dict[str, Any]]],
        meta: List[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        _same_len(images, meta)
        _same_len(images, masks_list)

        overlays: List[np.ndarray] = []
        out_meta: List[Dict[str, Any]] = []

        for i, (img, masks, m) in enumerate(zip(images, masks_list, meta)):
            rng = np.random.default_rng(self.seed + i)
            out = img.astype(np.float32).copy()

            for ann in masks:
                seg = ann.get("segmentation")
                if seg is None:
                    continue
                seg = seg.astype(bool)
                color = rng.integers(0, 256, size=(3,), dtype=np.int32).astype(np.float32)
                out[seg] = out[seg] * (1.0 - self.alpha) + color * self.alpha

            overlays.append(np.clip(out, 0, 255).astype(np.uint8))

            mm = dict(m)
            mm["overlay_alpha"] = self.alpha
            out_meta.append(mm)

        return overlays, out_meta


class ImageSaveOP:
    """(images, meta, save_paths|None) -> (saved_paths, meta)"""

    def __init__(self, save_dir: str = "outputs", prefix: str = "out", ext: str = "jpg"):
        self.save_dir = save_dir
        self.prefix = prefix
        self.ext = ext

    def run(
        self,
        images: List[np.ndarray],
        meta: List[Dict[str, Any]],
        save_paths: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        import cv2

        _same_len(images, meta)
        os.makedirs(self.save_dir, exist_ok=True)

        if save_paths is None:
            save_paths = [
                os.path.join(self.save_dir, f"{self.prefix}_{i}.{self.ext}") for i in range(len(images))
            ]
        _same_len(images, save_paths)

        out_paths: List[str] = []
        out_meta: List[Dict[str, Any]] = []

        for img, p, m in zip(images, save_paths, meta):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            ok = cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"save failed: {p}")

            mm = dict(m)
            mm["saved_path"] = p
            out_paths.append(p)
            out_meta.append(mm)

        return out_paths, out_meta


class JsonSaveOP:
    """meta -> json_path"""

    def run(self, meta: List[Dict[str, Any]], json_path: str) -> str:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        safe: List[Dict[str, Any]] = []
        for m in meta:
            d: Dict[str, Any] = {}
            for k, v in m.items():
                if isinstance(v, np.ndarray):
                    continue
                try:
                    json.dumps(v)
                    d[k] = v
                except TypeError:
                    d[k] = str(v)
            safe.append(d)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)

        return json_path
