import os
import time
from typing import List, Optional, Tuple, Dict, Any

import ray
from rayorch import RayModule, Dispatch  # 你现有库的接口
from rayorch import EnvRegistry

from ops import (
    ImageLoadOp,
    YOLODrawOp,
    SAMOp,
    OutputSAMMasksOp,
    ImageSaveOP,
    JsonSaveOP,
)
from utils import download_sample_images

def abspaths(xs: List[str]) -> List[str]:
    return [os.path.abspath(x) for x in xs]

class YoloSamPipeline:
    """
    类 nn.Module 风格：
      pipe = YoloSamPipeline(...)
      saved_paths, meta, meta_json = pipe(image_paths)

    env 用来让 YOLO 和 SAM 跑在不同 runtime_env（例如不同 .venv / conda env）。
    """

    def __init__(
        self,
        *,
        yolo_weight: str,
        sam_checkpoint: str,
        yolo_env: Optional[str] = "yolo_mxc",
        sam_env: Optional[str] = "sam_mxc",
        device: str = "cuda",
        yolo_replicas: int = 2,
        sam_replicas: int = 2,
        save_dir: str = "outputs_min",
        meta_json_path: str = "outputs_min/meta.json",
        alpha: float = 0.35,
        seed: int = 1234,
    ):
        self.save_dir = save_dir
        self.meta_json_path = meta_json_path

        # 1) loader: 单 actor
        self.loader = RayModule(
            ImageLoadOp,
            env=yolo_env,  # 这里用哪个 env 都行，主要保证 cv2/numpy 在里面
            replicas=1,
            num_gpus_per_replica=0.0,
        ).pre_init()

        # 2) yolo: 多 replica (dispatch all_sliced_to_all)
        self.yolo = RayModule(
            YOLODrawOp,
            env=yolo_env,
            replicas=yolo_replicas,
            num_gpus_per_replica=1.0 if device.startswith("cuda") else 0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
        ).pre_init(weight_path=yolo_weight, device=device, conf=0.25)

        # 3) sam: 多 replica
        self.sam = RayModule(
            SAMOp,
            env=sam_env,
            replicas=sam_replicas,
            num_gpus_per_replica=1.0 if device.startswith("cuda") else 0.0,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
        ).pre_init(checkpoint_path=sam_checkpoint, model_type="vit_b", device=device)

        # 4) render: 单 actor
        self.render = RayModule(
            OutputSAMMasksOp,
            env=sam_env,
            replicas=1,
            num_gpus_per_replica=0.0,
        ).pre_init(alpha=alpha, seed=seed)

        # 5) save: 单 actor
        self.save = RayModule(
            ImageSaveOP,
            env=yolo_env,
            replicas=1,
            num_gpus_per_replica=0.0,
        ).pre_init(save_dir=save_dir, prefix="overlay", ext="jpg")

        # 6) json: 单 actor
        self.json_save = RayModule(
            JsonSaveOP,
            env=yolo_env,
            replicas=1,
            num_gpus_per_replica=0.0,
        ).pre_init()

    def __call__(
        self,
        image_paths: List[str],
        save_paths: Optional[List[str]] = None,
        meta_json_path: Optional[str] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], str]:
        images, meta = self.loader(image_paths)
        images_yolo, meta = self.yolo(images, meta)
        masks, meta = self.sam(images_yolo, meta)
        overlay, meta = self.render(images_yolo, masks, meta)
        saved_paths_out, meta = self.save(overlay, meta, save_paths)
        out_json = self.json_save(meta, meta_json_path or self.meta_json_path)
        return saved_paths_out, meta, out_json


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "./"})

    # 1) 自动下载 5 张样例图到 ./cache
    image_paths = download_sample_images("./cache")

    from prepare_ckpt import main as prepare_ckpt_main
    prepare_ckpt_main()  # 自动下载 YOLO 和 SAM 的权重到 ./cache/ckpt
    if not os.path.exists("./cache/ckpt/yolo26n.pt") or not os.path.exists("./cache/ckpt/sam_vit_b_01ec64.pth"):
        raise RuntimeError("Checkpoint files not found. Please check the output of prepare_ckpt_main().")


    # 2) 后续照旧
    yolo_weight = os.path.abspath("./cache/ckpt/yolo26n.pt")
    sam_ckpt = os.path.abspath("./cache/ckpt/sam_vit_b_01ec64.pth")

    out_dir = os.path.abspath("./outputs_min")
    meta_json = os.path.join(out_dir, "meta.json")
    
    EnvRegistry.register(name="yolo", env_input="yolo_mxc")
    EnvRegistry.register(name="sam", env_input="sam_mxc")

    pipe = YoloSamPipeline(
        yolo_weight=yolo_weight,
        sam_checkpoint=sam_ckpt,
        yolo_env="yolo",
        sam_env="sam",
        device="cuda",
        yolo_replicas=2,
        sam_replicas=2,
        save_dir=out_dir,
        meta_json_path=meta_json,
        alpha=0.35,
        seed=1234,
    )

    saved_paths, meta, meta_json_out = pipe(image_paths)
    print("saved_paths:", saved_paths[:3], "... total =", len(saved_paths))
    print("meta_json:", meta_json_out)

    time.sleep(1)
    ray.shutdown()
