import os
from dataflow.serving import LocalModelLLMServing_sglang, LocalModelLLMServing_vllm
import os, time, sys
class SGLangServingOP():
    def __init__(self, 
                 hf_model_name_or_path,
                 sgl_tp_size=4,
                 sgl_dp_size=1,
                 sgl_mem_fraction_static: float = 0.9,
                 ):
        self.hf_model_name_or_path = hf_model_name_or_path
        self.sgl_tp_size = sgl_tp_size
        self.sgl_dp_size = sgl_dp_size
        self.sgl_mem_fraction_static = sgl_mem_fraction_static

        print(f"[INIT-START] pid={os.getpid()} time={time.time():.3f} exe={sys.executable}", flush=True)

        self.llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path=self.hf_model_name_or_path,
            sgl_tp_size=self.sgl_tp_size,
            sgl_dp_size=self.sgl_dp_size,
            sgl_mem_fraction_static=self.sgl_mem_fraction_static
        )
        self.llm_serving.start_serving() # Important !!!
        print(f"[INIT-DONE ] pid={os.getpid()} time={time.time():.3f}", flush=True)

    def run(self, input_prompts):
        res = self.llm_serving.generate_from_input(input_prompts)
        # add a tag before each output to indicate which operator it comes from
        tagged_res = [f"SGLangOP: {output}" for output in res]
        return tagged_res
    

class vLLMServingOP():
    def __init__(
            self,
            hf_model_name_or_path,
            vllm_tensor_parallel_size=4,
            vllm_gpu_memory_utilization=0.9,
    ):
        print(f"[INIT-START] pid={os.getpid()} time={time.time():.3f} exe={sys.executable}", flush=True)
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path=hf_model_name_or_path,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        self.llm_serving.start_serving() # Important !!!
        print(f"[INIT-DONE ] pid={os.getpid()} time={time.time():.3f}", flush=True)

    def run(self, input_prompts):
        res = self.llm_serving.generate_from_input(input_prompts)
        # add a tag before each output to indicate which operator it comes from
        tagged_res = [f"vLLMOP: {output}" for output in res]
        return tagged_res