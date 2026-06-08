# Runtime MVP

The Runtime MVP adds row-level fault isolation and lineage to RayOrch DAG
pipelines while keeping the regular `DagPipeline` compiler reusable.

## Core Types

- `MicroBatch`: row-aligned data columns plus internal `row_ids` and `path_ids`.
- `RuntimeRayModule`: a `RayModule` whose replicas execute an operator with
  row-level isolation.
- `RuntimeDagExecutor`: runs one or more `MicroBatch` objects through a compiled
  DAG with pipeline-level overlap.
- `RuntimeResult`: final healthy rows, quarantine records, and lineage deltas.

## Define Operators

Operators use the existing RayOrch convention: a class with a `run()` method.
Inputs and outputs are batch columns represented as Python lists.

```python
from rayorch.runtime import BadRecordError


class Pdf2Image:
    def run(self, pdf, meta):
        images = []
        for index, (path, item) in enumerate(zip(pdf, meta)):
            if path.endswith(".broken.pdf"):
                raise BadRecordError("cannot decode PDF", index=index)
            images.append([f"image:{path}:0", f"image:{path}:1"])
            item["pages"] = 2
        return images, meta


class ToMarkdown:
    def run(self, images, meta):
        return [
            f"{item['name']}.md pages={len(pages)}"
            for pages, item in zip(images, meta)
        ]
```

`BadRecordError(index=...)` identifies one bad row directly. Other exceptions
are isolated with split-and-retry until the failing row is found.

## Define A Runtime DAG

```python
from rayorch import DagPipeline, PipeRef, RuntimeRayModule


class PdfPipeline(DagPipeline):
    def __init__(self):
        self.pdf2image = RuntimeRayModule(
            Pdf2Image,
            replicas=2,
            max_inflight=2,
            num_outputs=2,
        ).pre_init()
        self.markdown = RuntimeRayModule(
            ToMarkdown,
            replicas=2,
            max_inflight=2,
            num_outputs=1,
        ).pre_init()
        super().__init__()

    def forward(self, pdf: PipeRef, meta: PipeRef):
        images, meta = self.pdf2image(pdf, meta)
        markdown = self.markdown(images, meta)
        return markdown
```

`DagPipeline.compile()` traces `forward()` and binds runtime ports from the
operator signature and assignment names. Port names are local to each DAG node.

For multi-output operators, explicitly setting `num_outputs` is currently the
most reviewable API even though direct tuple assignments can also be inferred.

## Run One MicroBatch

```python
from rayorch import RuntimeDagExecutor
from rayorch.runtime import MicroBatch


source = MicroBatch.source(
    {
        "pdf": ["paper-a.pdf", "paper-b.pdf"],
        "meta": [{"name": "paper-a"}, {"name": "paper-b"}],
    },
    dataset="papers",
)

result = RuntimeDagExecutor().run(PdfPipeline(), source)

print(result.batch.columns["markdown"])
print(result.quarantined)
```

## Pipeline-Level Overlap

Pass multiple source microbatches and set `max_batches_inflight`:

```python
batches = [
    MicroBatch.source(
        {
            "pdf": [f"paper-{batch_index}-{i}.pdf" for i in range(8)],
            "meta": [
                {"name": f"paper-{batch_index}-{i}"}
                for i in range(8)
            ],
        },
        dataset=f"papers-{batch_index}",
    )
    for batch_index in range(4)
]

results = RuntimeDagExecutor(max_batches_inflight=4).run(
    PdfPipeline(),
    batches,
)
```

There are two independent concurrency controls:

- `RuntimeRayModule(replicas=N)`: actor replicas inside one stage.
- `RuntimeDagExecutor(max_batches_inflight=N)`: microbatches concurrently moving
  through the whole pipeline.

`RuntimeRayModule(max_inflight=N)` limits concurrent submissions to that DAG
node.

## Inspect Faults And Lineage

```python
for record in result.quarantined:
    print(record.row_id, record.op, record.error, record.values)

for row_id, path_id in result.row_path.items():
    print(row_id, path_id)
```

A quarantine record contains the failed row, failed operator, previous path, and
error. Healthy rows continue through downstream stages.

## Current MVP Limits

- Runtime DAG nodes must be `RuntimeRayModule` instances.
- Multi-parent joins require matching `row_ids` and `path_ids`.
- Lineage is returned in memory; persistent sinks are future work.
- Operator/actor retries and process recovery are not yet production-grade.
- The executor currently targets row-aligned batch transformations.

## Tests

Default runtime tests exclude long benchmarks:

```bash
pytest test/test_runtime_correctness.py test/test_runtime_overlap.py test/test_runtime_edge_semantics.py
```

Run the Flash-MinerU-like 40-item benchmark explicitly:

```bash
pytest --runslow -s test/test_runtime_benchmark_dummy.py
```
