# Runtime MVP

The Runtime MVP adds row-level fault isolation and lineage to RayOrch DAG
pipelines while keeping the regular `DagPipeline` compiler reusable.

The lifecycle and product boundary between `RayModule` and `RuntimeRayModule`
is defined in [`runtime_module_lifecycle.md`](runtime_module_lifecycle.md).

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
from rayorch import DagPipeline, RuntimeRayModule


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

    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> list[str]:
        images, meta = self.pdf2image(pdf, meta)
        markdown = self.markdown(images, meta)
        return markdown
```

For `RuntimeRayModule`, `.pre_init(...)` only stores the user operator's
constructor arguments. It does not initialize Ray or create actors. The normal
lifecycle is:

```text
RuntimeRayModule(...).pre_init(...)  configure
RuntimeDagExecutor(pipeline, ...)    compile and start replicas
executor.run(...)                    execute
executor.close()                     stop replicas started by the executor
```

Use `module.start()` directly only when testing or running a standalone
`RuntimeRayModule`. Repeated `start()` calls are idempotent.

`DagPipeline.compile()` traces `forward()` and binds runtime ports from the
operator signature and assignment names. Port names are local to each DAG node.
The application-level annotations describe actual column values; internal
symbolic references are not exposed in the user API.

For multi-output operators, annotate the operator's `run()` return type. The
compiler uses that annotation to infer output count and port types, and checks
it against the assignment in `forward()`. Use `num_outputs` only as a fallback
for unannotated operators.

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

result = RuntimeDagExecutor(PdfPipeline()).run(source)

print(result.batch.columns["markdown"])
print(result.quarantined)
```

## Run Column Data

For normal dataset-style usage, pass aligned input columns and let the executor
split them into microbatches:

```python
pdfs = [f"paper-{i}.pdf" for i in range(32)]
meta = [{"name": f"paper-{i}"} for i in range(32)]

results = RuntimeDagExecutor(
    PdfPipeline(),
    batch_size=8,
    max_batches_inflight=4,
    dataset="papers",
).run(pdf=pdfs, meta=meta)
```

`results` contains one `RuntimeResult` per generated microbatch. Row ids remain
global within the dataset, for example `papers:0`, `papers:1`, ...

## Pipeline-Level Overlap

There are two independent concurrency controls:

- `RuntimeRayModule(replicas=N)`: actor replicas inside one stage.
- `RuntimeDagExecutor(..., max_batches_inflight=N)`: microbatches concurrently
  moving through the whole pipeline.

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
- Multi-parent joins align required branches by `row_id` and preserve diverged
  paths through internal multi-parent lineage nodes.
- Direct-return node calls currently use generated output names unless the
  result is assigned to a readable local variable; deterministic direct-return
  naming is tracked in the same TODO.
- The current record unit is one document. Pages and blocks remain nested
  values; explicit filter/flat-map/dedup/merge cardinality changes are deferred
  to the
  [`multi-grain port design`](todos/09-multi-grain-port-cardinality-api.md).
  Advanced command-style emission is tracked separately in
  [`todos/07-runtime-emit-api.md`](todos/07-runtime-emit-api.md).
- Lineage is returned in memory; persistent sinks are future work.
- Operator/actor retries and process recovery are not yet production-grade.
- The executor currently targets row-aligned batch transformations.

## Tests

Fast compile and local semantics tests do not start Ray:

```bash
pytest test/runtime/unit
```

Runtime integration tests share one Ray cluster per test module:

```bash
pytest test/runtime/integration
```

Run the Flash-MinerU-like 40-item benchmark explicitly:

```bash
pytest --runslow -s test/runtime/performance
```
