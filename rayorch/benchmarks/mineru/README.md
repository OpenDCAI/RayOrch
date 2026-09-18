# MinerU PDF Benchmark

## Why this case exists

This is RayOrch's included real-model workload. It demonstrates dynamic page
fan-out, batched GPU inference, ordered document reconstruction, multi-node
actor placement, and a complete Benchmark report.

## Topology

```text
PDF path
   ├──────────────► PdfMetadata ──────────────────────────┐
   │                                                      │
   ▼                                                      │
Render PDF ─► Page ─► MinerU vLLM OCR ─► ordered reduce ──┤
                                                          ▼
                                                   Assemble document
                                                          │
                                                          ▼
                                                  Markdown + JSON
```

One PDF expands into a Page domain. OCR runs across a configurable GPU actor
pool. Page outputs are reduced in page order and joined with lightweight PDF
metadata at the original document level.

## Run

```python
from rayorch.benchmark import MinerUBench

bench = MinerUBench(
    input_path="/shared/pdfs",
    output_dir="/shared/mineru-output",
    model="/shared/models/MinerU2.5",
    input_limit=100,
    num_gpus=8,
    batch_size=64,
    input_batch_size=24,
    max_active_input_batches=3,
)

report = bench.run(ray_address="auto")
```

`input_path` may be one PDF or a directory. For a development Ray Job:

```python
from rayorch.benchmark import LocalSource

run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(
        project_root="/path/to/RayOrch",
        modules=("/path/to/Flash-mineru/flash_mineru",),
    ),
)
report = run.wait(timeout_s=3600)
```

Use `stage_options` for uncommon resource experiments:

```python
stage_options={
    "render": {"replicas": 4, "num_cpus": 2},
    "ocr": {"batch_size": 32},
    "assemble": {"replicas": 2},
}
```

## Results

MinerU writes Markdown, layout JSON, and extracted images under `output_dir`.
The Benchmark artifacts are:

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

`report.metrics` includes document/page throughput and common actor, RPC,
batching, timing, and profile data. `report.outputs` contains the output path
and page count for every processed PDF.

All input, model, output, and report paths must be visible at the same location
on every eligible cluster node.
