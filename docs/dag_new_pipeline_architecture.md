# DAG Pipeline Architecture

Date: 2026-06-05

The DAG implementation now lives under `rayorch/dag/`.  The old
`rayorch.dag_new_pipeline` module is kept as a compatibility re-export.

## File Layout

```text
rayorch/dag/
  graph.py      # PipeRef, NodeSpec, CompiledGraph
  compiler.py   # forward() AST hints + dynamic trace
  executor.py   # Executor, SequentialExecutor, DagExecutor
  pipeline.py   # user-facing Pipeline / DagPipeline
```

## Boundary

`graph.py` only describes the DAG.

`compiler.py` owns the PyTorch-like dynamic tracing:

- replace `RayModule` attributes with `TraceProxy`;
- run `forward()` with symbolic `PipeRef` inputs;
- build `NodeSpec` records and validate references;
- notify optional module hooks after graph validation.

`executor.py` owns runtime scheduling and does not inspect `Pipeline`.

`pipeline.py` is the public subclassing API and input resolver.

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
