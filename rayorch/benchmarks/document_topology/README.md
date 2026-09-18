# Nested Document Topology Benchmark

## Why this case exists

This dependency-free case preserves the shape of the historical Docling
workflow without importing Docling or model dependencies. It is the smallest
runnable example of two nested fan-outs, empty child groups, and ordered
reductions.

## Topology

```text
Document
   │
   ▼
Parse ─► Page ─► Layout ─► OCR ─► Postprocess
                              │
                              ▼
                         TableJob ─► TableCore
                              │
                              ▼
                         reduce to Page
                              │
                              ▼
                       reduce to Document
```

The generated input deliberately includes pages with zero table jobs, showing
that empty nested groups still reduce correctly.

## Run

```python
from rayorch.benchmark import DocumentTopologyBench

report = DocumentTopologyBench(
    output_dir="./results",
    document_count=8,
    pages_per_document=6,
    tables_per_page=2,
    workers=4,
    batch_size=8,
    input_batch_size=1,
    max_active_input_batches=4,
).run()
```

It has no optional model or dataset dependencies and can also be submitted:

```python
run = DocumentTopologyBench(output_dir="/shared/results").submit(
    "http://ray-head:8265"
)
report = run.wait(timeout_s=300)
```

RayOrch must already be installed on the cluster, or source can be supplied
with `LocalSource`.

## Results

Each output contains the document name, ordered page IDs, and table IDs for
every page. `report.metrics` adds total `pages` and `tables` to the standard
RayOrch execution metrics.

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

This is a topology and scheduler example, not a production Docling adapter.
