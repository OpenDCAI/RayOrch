# DSL Compiler

Date: 2026-06-12

RayOrch pipelines are defined in a constrained subset of standard Python.  The
compiler translates that DSL into a backend-agnostic JSON IR, then dispatches
to one or more backend code generators.

## Motivation

The existing `DagPipeline` tracing approach (see
[`dag_new_pipeline_architecture.md`](dag_new_pipeline_architecture.md)) is
already backend-agnostic at the Python level.  Surfacing an explicit IR layer:

- makes it possible to target backends other than Ray (e.g. `TorchBackend`);
- enables offline validation, serialization, and versioning of pipeline
  definitions;
- provides a stable contract between front-end tooling and execution engines.

## Compilation Stages

```text
Python DSL source
      │
      ▼ (1) AST parse  ── Python `ast` standard library
Python AST
      │
      ▼ (2) Semantic analysis  ── type inference, symbolic tracing
Typed DAG graph
      │
      ▼ (3) IR emit
JSON IR
      │
      ├─▶ (4a) RayBackend  ── RuntimeDagExecutor + Ray actors
      └─▶ (4b) TorchBackend  ── torch.fx / torch.compile graph
```

## Python DSL

The DSL is valid Python that `python -c` can parse.  The compiler enforces
additional restrictions at the semantic analysis stage.

### Allowed constructs inside `forward()`

```python
class MyPipeline(DagPipeline):
    def forward(self, pdf: list[str], meta: list[dict]) -> list[str]:
        images, meta = self.pdf2img(pdf, meta)      # operator call
        text = self.ocr(images)                      # operator call
        return self.render(text, meta)               # operator call
```

- Module attribute calls (`self.op(...)`)
- Simple assignment (`x = ...`, `a, b = ...`)
- Pass-through of earlier names as arguments
- Return of one name or one operator call

### Prohibited constructs inside `forward()`

```python
if len(pdf) > 10:     # data-dependent branch – compile-time error
    ...

for item in pdf:      # iteration – compile-time error
    ...
```

Data-dependent control flow and mutation must live inside operator `run()`
methods, not in the pipeline definition.  The compiler raises a
`SymbolicTraceError` that names the offending construct and the line number.

## JSON IR Schema

The IR is a JSON object with the following top-level keys.

```json
{
  "version": "1",
  "name": "MyPipeline",
  "inputs": [
    {"name": "pdf",  "type": "list[str]"},
    {"name": "meta", "type": "list[dict]"}
  ],
  "outputs": ["render.out0"],
  "nodes": [
    {
      "id":      "pdf2img",
      "op":      "MyPipeline.pdf2img",
      "inputs":  ["$pdf", "$meta"],
      "outputs": ["images", "meta"],
      "config":  {"replicas": 2, "max_inflight": 4}
    },
    {
      "id":      "ocr",
      "op":      "MyPipeline.ocr",
      "inputs":  ["images"],
      "outputs": ["text"],
      "config":  {"replicas": 4, "max_inflight": 4}
    },
    {
      "id":      "render",
      "op":      "MyPipeline.render",
      "inputs":  ["text", "meta"],
      "outputs": ["render.out0"],
      "config":  {"replicas": 2, "max_inflight": 2}
    }
  ]
}
```

### Field reference syntax

Source inputs use `$name`.  Node outputs use `node_id.port_name` or the local
alias assigned in `forward()`.  The IR uses fully-qualified references; local
aliases from `forward()` are resolved during semantic analysis.

### Node `config`

`config` carries backend-agnostic resource hints.  Each backend interprets
only the keys it understands and ignores the rest.  Common keys:

| Key | Type | Meaning |
|---|---|---|
| `replicas` | int | Desired parallelism |
| `max_inflight` | int | Max concurrent submissions per node |
| `device` | str | `"cpu"` / `"cuda"` / `"cuda:0"` |
| `memory_gb` | float | Suggested memory reservation |

## Emitting the IR

```python
from rayorch.compiler import emit_ir

pipeline = MyPipeline()
ir = emit_ir(pipeline)          # returns dict
import json
print(json.dumps(ir, indent=2))
```

`emit_ir` performs AST parse + semantic analysis + IR emission in one call.
Each stage is also accessible independently for tooling:

```python
from rayorch.compiler import parse_dsl, analyze, to_ir

ast_tree    = parse_dsl(pipeline)
typed_graph = analyze(ast_tree, pipeline)
ir          = to_ir(typed_graph)
```

## Backends

### RayBackend

The default backend.  Translates the IR into `RuntimeRayModule` instances,
wires a compiled `DagPipeline`, and returns a `RuntimeDagExecutor` ready to
run.

```python
from rayorch.compiler.backends import RayBackend

executor = RayBackend(
    providers={...},
    batch_size=8,
    max_batches_inflight=4,
).build(ir)

results = executor.run(pdf=pdfs, meta=metas)
```

`RayBackend.build(ir)` is equivalent to constructing the pipeline by hand.  It
exists so that a serialized IR can be loaded and executed without the original
Python source.

### TorchBackend (future)

Targets `torch.fx` and `torch.compile` for GPU-only pipelines where operators
are pure tensor functions.

```python
from rayorch.compiler.backends import TorchBackend

graph_module = TorchBackend().build(ir)
output = graph_module(pdf_tensor, meta_tensor)
```

`TorchBackend` maps each IR node to a `torch.nn.Module` and uses
`torch.fx.Graph` for the execution schedule.

### Adding a New Backend

A backend is any callable that accepts the IR dict and returns an executor or
runnable:

```python
class MyBackend:
    def build(self, ir: dict) -> Any:
        ...
```

Backends are registered by name for use from the CLI or config files:

```python
from rayorch.compiler.registry import register_backend

register_backend("mybackend", MyBackend)
```

## IR Stability

The IR schema is versioned (`"version": "1"`).  Breaking changes increment the
major version.  Additive changes (new optional node keys, new config keys) are
non-breaking.  Backends must ignore unknown keys to remain forward-compatible.

Serialized IR files can be checked into source control, diffed, and used to
reproduce a pipeline exactly, independent of the Python class hierarchy that
generated them.
