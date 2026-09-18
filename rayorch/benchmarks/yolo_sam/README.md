# YOLO → SAM Benchmark

## Why this case exists

This ports RayOrch's first public example into the current Benchmark layout.
It demonstrates a multi-model image workflow whose expensive YOLO and SAM
models live in separate persistent Ray actor pools.

## Topology

```text
image path
    │
    ▼
LoadImages ──► RGB image + metadata
    │
    ▼
YOLO ────────► detected image + box count
    │
    ▼
SAM ─────────► masks + metadata
    │
    ▼
RenderMasks ─► overlay image
    │
    ▼
SaveImages ───► output record
    │
    ▼
SaveMetadata ─► per-image JSON + final output
```

Each stage receives and returns aligned batches. `yolo_replicas` and
`sam_replicas` create persistent one-GPU actors by default. Therefore two YOLO
replicas plus two SAM replicas reserve four GPUs concurrently.

## Run

Install the dependencies in `env.json`, and prepare the model files and input
images on paths visible to every Ray node:

```python
from rayorch.benchmark import YoloSamBench

bench = YoloSamBench(
    input_path="/shared/images",
    output_dir="/shared/yolo-sam-output",
    yolo_model="/shared/models/yolo.pt",
    sam_checkpoint="/shared/models/sam_vit_b.pth",
    input_limit=100,
    yolo_replicas=2,
    sam_replicas=2,
    batch_size=4,
)

report = bench.run(ray_address="auto")
```

For a CPU smoke run, use `device="cpu"` and normally set both replica counts to
one. For Ray Jobs:

```python
from rayorch.benchmark import LocalSource

run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(project_root="/path/to/RayOrch"),
)
report = run.wait(timeout_s=3600)
```

Advanced actor settings can be overridden without editing the Pipeline:

```python
stage_options={
    "yolo": {
        "num_gpus": 0.5,
        "runtime_env": {"conda": "yolo-environment"},
    },
    "sam": {
        "num_gpus": 0.5,
        "runtime_env": {"conda": "sam-environment"},
    },
}
```

Heavy imports happen inside the stage actors, so the driver environment does
not need Ultralytics or Segment Anything when those stages use separate Conda
environments. Each environment must contain compatible Python, Ray, RayOrch,
and that stage's dependencies.

## Results

Outputs are written under `output_dir`:

```text
<image-stem>.overlay.jpg
<image-stem>.json
.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

`report.metrics` includes input/output counts, execution metrics, image count,
YOLO detection count, and SAM mask count. `report.outputs` contains one final
saved-image record per input.

The automated repository test replaces heavy YOLO/SAM implementations with
deterministic UDFs while retaining this exact Pipeline graph. A real run
requires `ultralytics`, `segment-anything`, compatible Torch/CUDA libraries,
both checkpoints, and images.
