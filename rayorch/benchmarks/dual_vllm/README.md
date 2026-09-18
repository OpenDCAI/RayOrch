# Dual-vLLM Benchmark

## Why this case exists

This ports the early RayOrch dual-vLLM demonstration to the current Pipeline
and Benchmark APIs. It shows how two independently loaded model actors can form
one explicit, inspectable inference workflow.

## Topology

```text
input prompt
    │
    ▼
vLLM Model A ──► first answer
    │
    ▼
BuildRefinementPrompt
    │
    ▼
vLLM Model B ──► refined answer
    │
    ▼
BuildResult ───► prompt + both answers
```

Both model actors are persistent. With `tensor_parallel_size=N`, each actor
requests `N` GPUs, so the default two-model Pipeline reserves `2 × N` GPUs.
They are separate actors even when `model_a` and `model_b` point to the same
model directory.

## Input

`input_path` may be:

- a UTF-8 text file containing one non-empty prompt per line; or
- a JSONL file whose rows are strings or objects with a `prompt` field.

## Run

```python
from rayorch.benchmark import DualVllmBench

bench = DualVllmBench(
    input_path="/shared/prompts.jsonl",
    output_dir="/shared/dual-vllm-output",
    model_a="/shared/models/model-a",
    model_b="/shared/models/model-b",
    input_limit=100,
    tensor_parallel_size=1,
    batch_size=8,
    max_tokens=128,
)

report = bench.run(ray_address="auto")
```

To submit from a source checkout:

```python
from rayorch.benchmark import LocalSource

run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(project_root="/path/to/RayOrch"),
)
report = run.wait(timeout_s=3600)
```

Per-stage scheduling can be changed without editing the Pipeline:

```python
stage_options={
    "model_a": {
        "resources": {"model_a_node": 0.001},
        "runtime_env": {"conda": "vllm-a"},
    },
    "model_b": {
        "resources": {"model_b_node": 0.001},
        "runtime_env": {"conda": "vllm-b"},
    },
}
```

The driver environment does not need vLLM when both model stages select
prepared Conda environments. Each actor environment must contain compatible
Python, Ray, RayOrch, vLLM, and CUDA dependencies.

## Results

The report is written to:

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

Each output contains:

```python
{
    "prompt": "...",
    "model_a": "...",
    "model_b": "...",
}
```

`report.metrics` also contains prompt count and generated character counts,
alongside the common RayOrch actor, RPC, batching, timing, and profile metrics.

The automated repository test replaces vLLM with deterministic UDFs while
retaining this exact four-stage graph. A real run requires vLLM, compatible
CUDA libraries, enough GPUs for both persistent engines, and model directories
visible from every eligible cluster node.
