# DAG Pipeline Architecture

Date: 2026-06-05

The DAG implementation now lives under `rayorch/dag/`.  The old
`rayorch.dag_new_pipeline` module is kept as a compatibility re-export.

## Infrastructure Layers

RayOrch organizes system concerns into three planes so that each layer can
evolve or be replaced independently.

### Control Plane

The control plane governs what should run, when it should run, and what the
outcome was.

- **PostgreSQL / MySQL** – persistent source of truth for DAG definitions, job
  and stage state, block metadata, image references, OCR result indexes, and
  audit logs.  The database is the record that survives process restarts.
- **MetadataStoreProvider** – abstraction over the relational store (see
  [`provider_abstractions.md`](provider_abstractions.md)).  The MVP ships an
  in-memory stub; production swaps in a SQL backend.

### Execution Plane

The execution plane owns scheduling, dispatch, and in-flight state.

- **Ray** – distributed actor scheduling, task dispatch, and in-flight state
  tracking.  Ray is the source of truth for *whether a task is currently
  running*, but not for durable job history.
- **RuntimeDagExecutor / RayModule / RuntimeRayModule** – executor objects that
  translate compiled DAG nodes into Ray actor calls, enforce inflight limits,
  and collect outputs, quarantine records, and lineage deltas.

### Storage Plane

The storage plane holds raw inputs and large intermediate artifacts.

- **S3 / MinIO** – object store for raw PDF/docx files, extracted images, and
  OCR input/output intermediates.
- **ArtifactStoreProvider** – abstraction over the object store (see
  [`provider_abstractions.md`](provider_abstractions.md)).  References to
  stored objects are small strings that travel through the DAG; the heavy bytes
  live in the object store.

### Observability (future)

- **Prometheus / Grafana** – metrics collection and dashboards via
  **MetricsProvider**.
- **Kafka** – event streaming to offload high-volume write pressure from the
  RDBMS via **EventBusProvider**.

Both are abstracted as pluggable providers so that early phases can use no-op
or in-memory stubs without changing operator or executor code.  See
[`provider_abstractions.md`](provider_abstractions.md) for the full interface
catalogue.

## File Layout

```text
rayorch/dag/
  graph.py      # internal symbolic refs, NodeSpec, CompiledGraph
  compiler.py   # forward() AST hints + dynamic trace
  executor.py   # Executor, SequentialExecutor, DagExecutor
  pipeline.py   # user-facing Pipeline / DagPipeline
```

## Boundary

`graph.py` only describes the DAG.

`compiler.py` owns the PyTorch-like dynamic tracing:

- replace `RayModule` attributes with `TraceProxy`;
- derive source schemas from the application-level `forward()` annotations;
- run `forward()` with internal symbolic inputs;
- build `NodeSpec` records and validate references;
- notify optional module hooks after graph validation.

`executor.py` owns runtime scheduling and does not inspect `Pipeline`.

`pipeline.py` is the public subclassing API and input resolver.

## Execution Ownership

`Pipeline.forward()` is the single user-authored graph definition. The default
`run()` and `__call__()` paths execute it eagerly without compiling:

```python
result_a = pipeline.run(input_batch)
result_b = pipeline(input_batch)
```

Both entry points expose the runtime signature of the subclass's `forward()`
for inspection and interactive help.

Executors bind one pipeline during construction and compile that same
`forward()` definition for scheduled multi-batch execution:

```python
serial = SequentialExecutor(pipeline)
overlap = DagExecutor(pipeline, max_batches_inflight=4)

serial_result = serial.run(input_batches)
overlap_result = overlap.run(input_batches)
```

No executor is accepted as a `Pipeline.run()` argument. This keeps eager
execution simple and execution policy explicit.

`RuntimeRayModule` pipelines require `RuntimeDagExecutor` because their
execution contract needs compiled port specs and `MicroBatch` lineage metadata.
Calling `pipeline.run(...)` or `pipeline(...)` on such a pipeline raises an
explicit error instead of attempting partial eager execution.

Runtime module construction is resource-free. `RuntimeRayModule.pre_init()`
stores user-op constructor arguments, while `RuntimeDagExecutor` starts each
unique runtime module after compilation. This keeps Pipeline construction,
signature inspection, and compilation free of Ray actor side effects.

## Public Type Model

`PipeRef` is an internal compiler object and is not part of the public API.
Users annotate `forward()` with the values that operators receive at runtime:

```python
class PdfPipeline(DagPipeline):
    def forward(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> list[str]:
        images, meta = self.pdf2img(pdf, meta)
        return self.convert(images, meta)
```

The compiler combines:

- source and output annotations from `forward()`;
- parameter and return annotations from each operator's `run()`;
- the symbolic trace produced by executing `forward()` at compile time.

Missing source annotations can be inferred from the first annotated operator
that consumes the source. Generic operator annotations such as
`run(values: list[T]) -> list[T]` bind `T` from the upstream port and propagate
the resolved type downstream.

Type mismatches fail during `compile()` with the node, input port, expected
type, actual type, and upstream source in the error message. Unannotated
operators fall back to `Any`.

## Symbolic Restrictions

Pipeline values are symbolic while `forward()` is compiled. Topology-building
calls are supported, but data-dependent Python operations are not:

```python
if len(pdf) > 10:  # compile-time error
    ...
```

Boolean conversion, `len()`, iteration, and indexing produce an explicit error
that explains the symbolic tracing boundary. Dynamic data-dependent behavior
should live inside an operator.

## Naming Hints

Dynamic tracing is the source of truth for topology.  AST parsing is only a
small naming helper:

```python
images, meta = self.pdf2img(pdf, meta)
```

This lets compile assign node-local labels:

```text
inputs  = ("pdf", "meta")
outputs = ("images", "meta")
```

Nested calls are handled in Python evaluation order, so hints stay aligned with
the actual trace.

## Compile Hook

The DAG core is runtime-agnostic.  It does not import `rayorch.runtime`.

Modules that need compile-time metadata can implement:

```python
def on_compile_node(self, spec: NodeSpec) -> None:
    ...
```

`RuntimeRayModule` uses this hook to bind its `RuntimeNodeSpec` after the graph
has been validated.
