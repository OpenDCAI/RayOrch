import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

import ray
from rayorch import RayModule
from rayorch import EnvRegistry

from ops import SGLangServingOP, vLLMServingOP
from pingpong_prompts import build_prompts


def to_list(x) -> List[str]:
    return x if isinstance(x, list) else [str(x)]


class PingPongServingPipeline:
    """
    显式写清楚每一轮：
      SGLang -> vLLM -> SGLang -> vLLM ...
    不做抽象，方便当 demo/样例阅读。
    """

    def __init__(
        self,
        *,
        model_path: str,
        sglang_env: str,
        vllm_env: str,
        sgl_tp: int = 1,
        vllm_tp: int = 1,
        sgl_mem: float = 0.90,
        vllm_mem: float = 0.90,
    ):
        self.sgl = RayModule(
            SGLangServingOP,
            env=sglang_env,
            replicas=1,
            num_gpus_per_replica=1.0,
        ).pre_init(
            hf_model_name_or_path=model_path,
            sgl_tp_size=sgl_tp,
            sgl_dp_size=1,
            sgl_mem_fraction_static=sgl_mem,
        )

        self.vllm = RayModule(
            vLLMServingOP,
            env=vllm_env,
            replicas=1,
            num_gpus_per_replica=1.0,
        ).pre_init(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=vllm_tp,
            vllm_gpu_memory_utilization=vllm_mem,
        )

        print("[DRIVER] actors created (pre_init done)", flush=True)

    def __call__(self, user_prompt: str) -> Tuple[List[str], List[str]]:
        prompts = build_prompts(user_prompt)
        transcript: List[str] = []

        # -------- Seed --------
        seed = prompts["seed"]
        transcript.append("=== Seed Prompt ===\n" + seed)

        # -------- R1: SGLang --------
        r1_in = prompts["r1_en_summary"]
        transcript.append("=== R1 Input (to SGLang) ===\n" + r1_in)

        r1_out = self.sgl(to_list(r1_in))  # list[str]
        transcript.append("=== R1 Output (from SGLang) ===\n" + "\n".join(r1_out))

        # -------- R2: vLLM --------
        r2_in = prompts["r2_zh_title_tpl"].format(prev="\n".join(r1_out))
        transcript.append("=== R2 Input (to vLLM) ===\n" + r2_in)

        r2_out = self.vllm(to_list(r2_in))
        transcript.append("=== R2 Output (from vLLM) ===\n" + "\n".join(r2_out))

        # -------- R3: SGLang --------
        r3_in = prompts["r3_ja_polite_tpl"].format(prev="\n".join(r2_out))
        transcript.append("=== R3 Input (to SGLang) ===\n" + r3_in)

        r3_out = self.sgl(to_list(r3_in))
        transcript.append("=== R3 Output (from SGLang) ===\n" + "\n".join(r3_out))

        # -------- R4: vLLM --------
        r4_in = prompts["r4_es_tweet_tpl"].format(prev="\n".join(r3_out))
        transcript.append("=== R4 Input (to vLLM) ===\n" + r4_in)

        r4_out = self.vllm(to_list(r4_in))
        transcript.append("=== R4 Output (from vLLM) ===\n" + "\n".join(r4_out))

        # -------- R5: SGLang --------
        r5_in = prompts["r5_en_actions_tpl"].format(prev="\n".join(r4_out))
        transcript.append("=== R5 Input (to SGLang) ===\n" + r5_in)

        r5_out = self.sgl(to_list(r5_in))
        transcript.append("=== R5 Output (from SGLang) ===\n" + "\n".join(r5_out))

        # -------- R6: vLLM --------
        r6_in = prompts["r6_zh_spoken_tpl"].format(prev="\n".join(r5_out))
        transcript.append("=== R6 Input (to vLLM) ===\n" + r6_in)

        r6_out = self.vllm(to_list(r6_in))
        transcript.append("=== R6 Output (from vLLM) ===\n" + "\n".join(r6_out))

        # -------- R7: SGLang --------
        r7_in = prompts["r7_yaml_tpl"].format(prev="\n".join(r6_out))
        transcript.append("=== R7 Input (to SGLang) ===\n" + r7_in)

        r7_out = self.sgl(to_list(r7_in))
        transcript.append("=== R7 Output (from SGLang) ===\n" + "\n".join(r7_out))

        # -------- R8: vLLM (FINAL) --------
        r8_in = prompts["r8_final_tpl"].format(prev="\n".join(r7_out))
        transcript.append("=== R8 Input (to vLLM) ===\n" + r8_in)

        final_out = self.vllm(to_list(r8_in))
        transcript.append("=== R8 Output (FINAL, from vLLM) ===\n" + "\n".join(final_out))

        return final_out, transcript


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    SGLANG_ENV = "df_sglang_mxc"
    VLLM_ENV = "df_vllm_mxc"
    MODEL_PATH = "/home/dataset-local/models/Qwen2.5-7B"
    
    EnvRegistry.register(name = "sglang", env_input=SGLANG_ENV)
    EnvRegistry.register(name = "vllm", env_input=VLLM_ENV)

    pipe = PingPongServingPipeline(
        model_path=MODEL_PATH,
        sglang_env='sglang',
        vllm_env='vllm',
        sgl_tp=1,
        vllm_tp=1,
        sgl_mem=0.90,
        vllm_mem=0.90,
    )

    final_out, transcript = pipe(
        "We want a minimal RayOrch demo that shows two different serving backends calling each other."
    )

    print("\n================ FINAL OUTPUT ================\n")
    print("\n".join(final_out))

    print("\n================ TRANSCRIPT (ALL STEPS) ================\n")
    for t in transcript:
        print(t)
        print()

    # 保存日志到 ./cache
    os.makedirs("./cache", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join("./cache", f"pingpong_demo_{ts}.txt"))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("========== FINAL OUTPUT ==========\n\n")
        f.write("\n".join(final_out))
        f.write("\n\n========== TRANSCRIPT ==========\n\n")
        f.write("\n\n".join(transcript))
        f.write("\n")

    print("\nLogs saved to:", log_path)

    time.sleep(1)
    ray.shutdown()
