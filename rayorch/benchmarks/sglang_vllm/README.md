# SGLang → vLLM Cross-Environment Benchmark

## Why this case exists

This is the smallest built-in example of one RayOrch Pipeline whose persistent
stages run in different Conda environments. SGLang and vLLM can keep their own
dependency stacks; RayOrch passes ordinary values between their actors.

There is no environment registry. Each model stage uses Ray's native
`runtime_env={"conda": ...}` option.

## Topology

```text
input prompt
    │
    ▼
SGLang actor ────────► draft
runtime_env: sglang_env
    │
    ▼
BuildHandoffPrompt
    │
    ▼
vLLM actor ──────────► final answer
runtime_env: vllm_env
    │
    ▼
BuildResult ─────────► prompt + draft + final answer
```

The two model actors are persistent and reserve their tensor-parallel GPU
counts concurrently.

## Prepare the environments

Create two Conda environments on every eligible Ray node:

```text
rayorch-sglang: Python + Ray + RayOrch source/package + SGLang
rayorch-vllm:   Python + Ray + RayOrch source/package + vLLM
```

Use compatible Python and Ray versions in the driver and both actor
environments. The model and input paths must be visible from every node.
`env.json` intentionally does not install either backend into the Ray Job
driver environment.

## Run

`input_path` may be a text file with one prompt per line, or JSONL containing
strings or objects with a `prompt` field.

```python
from rayorch.benchmark import SglangVllmBench

bench = SglangVllmBench(
    input_path="/shared/prompts.jsonl",
    output_dir="/shared/sglang-vllm-output",
    model="/shared/models/Qwen3-0.6B",
    sglang_env="rayorch-sglang",
    vllm_env="rayorch-vllm",
    batch_size=8,
    input_limit=100,
)

report = bench.run(ray_address="auto")
```

The environment names are normal Benchmark parameters and therefore also
survive `benchmark_config()`, Ray Job submission, and experiment cloning.
Advanced actor options can still replace any stage default:

```python
stage_options={
    "sglang": {
        "runtime_env": {"conda": "custom-sglang"},
        "resources": {"sglang_node": 0.001},
    },
    "vllm": {
        "runtime_env": {"conda": "custom-vllm"},
        "resources": {"vllm_node": 0.001},
    },
}
```

For a source checkout:

```python
from rayorch.benchmark import LocalSource

run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(
        project_root="/path/to/RayOrch",
        install_dependencies=False,
    ),
)
report = run.wait(timeout_s=3600)
```

`install_dependencies=False` is intentional here: SGLang and vLLM are already
owned by their respective actor environments.

## Results

The report is written under:

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

Each output is:

```python
{
    "prompt": "...",
    "sglang": "...",
    "vllm": "...",
}
```

The repository's automated end-to-end test replaces both model engines with
deterministic UDFs while preserving this four-stage graph. The separate
cross-Conda integration test executes a real RayOrch actor in a configured
second environment without requiring a model or GPU.
