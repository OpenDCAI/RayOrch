import time
import ray
from typing import List, Tuple
import os
import time
from datetime import datetime
from rayorch import RayModule
from ops import vLLMServingOP


def to_list(x):
    return x if isinstance(x, list) else [str(x)]


class DualVLLMPipeline_DevNsysProfile:
    """
    Minimal demo:
      vLLM(model_A) -> vLLM(model_B)

    两个独立 vLLM Serving，各自加载不同模型。
    开启 dev_mode 后，会使用 NVTX 进行 profiling。
    """

    def __init__(
        self,
        *,
        model_a_path: str,
        model_b_path: str,
        vllm_env: str,
        tp_size: int = 1,
        gpu_mem_util: float = 0.9,
        dev_mode: bool = False,
    ):
        # vLLM A
        self.vllm_a = RayModule(
            vLLMServingOP,
            env=vllm_env,
            replicas=1,
            num_gpus_per_replica=1.0,
            dev_mode=dev_mode,
            init_tag_fn=lambda args, kwargs, rid: f"vLLM_A | rid={rid}",
            run_tag_fn=lambda args, kwargs, rid: f"vLLM_A | rid={rid}"
        ).pre_init(
            hf_model_name_or_path=model_a_path,
            vllm_tensor_parallel_size=tp_size,
            vllm_gpu_memory_utilization=gpu_mem_util,
        )

        # vLLM B
        self.vllm_b = RayModule(
            vLLMServingOP,
            env=vllm_env,
            replicas=1,
            num_gpus_per_replica=1.0,
            dev_mode=dev_mode,
            init_tag_fn=lambda args, kwargs, rid: f"vLLM_B | rid={rid}",
            run_tag_fn=lambda args, kwargs, rid: f"vLLM_B | rid={rid}"
        ).pre_init(
            hf_model_name_or_path=model_b_path,
            vllm_tensor_parallel_size=tp_size,
            vllm_gpu_memory_utilization=gpu_mem_util,
        )

        print("[DRIVER] dual vLLM actors ready", flush=True)

    def __call__(self, prompt: str) -> Tuple[List[str], List[str]]:
        transcript = []

        # Step 1: model A
        transcript.append("=== Input to Model A ===\n" + prompt)
        out_a = self.vllm_a(to_list(prompt))
        transcript.append("=== Output from Model A ===\n" + "\n".join(out_a))

        # Step 2: model B
        prompt_b = f"Refine and improve the following answer:\n\n{chr(10).join(out_a)}"
        transcript.append("=== Input to Model B ===\n" + prompt_b)

        out_b = self.vllm_b(to_list(prompt_b))
        transcript.append("=== Output from Model B ===\n" + "\n".join(out_b))

        return out_b, transcript


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    VLLM_ENV = None

    MODEL_A = "/home/dataset-local/models/Qwen3-0.6B"
    MODEL_B = "/home/dataset-local/models/Qwen3-0.6B"

    pipe = DualVLLMPipeline_DevNsysProfile(
        model_a_path=MODEL_A,
        model_b_path=MODEL_B,
        vllm_env=VLLM_ENV,
        tp_size=1,
        gpu_mem_util=0.9,
        dev_mode=True,
    )

    final_out, transcript = pipe(
        "Explain why runtime environment isolation matters in distributed systems."
    )

    print("\n===== FINAL OUTPUT =====\n")
    print("\n".join(final_out))

    print("\n===== TRANSCRIPT =====\n")
    for t in transcript:
        print(t)
        print()


    # 保存日志到 ./cache
    os.makedirs("./cache", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join("./cache", f"dual_vllm_dev_nsys_profile_{ts}.txt"))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("========== FINAL OUTPUT ==========\n\n")
        f.write("\n".join(final_out))
        f.write("\n\n========== TRANSCRIPT ==========\n\n")
        f.write("\n\n".join(transcript))
        f.write("\n")

    print("\nLogs saved to:", log_path)

    time.sleep(1)
    ray.shutdown()
