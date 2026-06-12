# DSL Compiler

Date: 2026-06-12

> **Status: Design draft — not yet implemented.**  Module paths, exception
> names, and API signatures described here do not exist in the current
> codebase.  This document records the target architecture for Phase 2 of the
> [roadmap](roadmap.md).

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
      ▼ (1) AST parse  ── ASTVersionProvider (wraps Python `ast` std lib)
Python AST
      │
      ▼ (2) Semantic analysis  ── type inference, symbolic tracing
Typed DAG graph (with inferred / annotated types per port)
      │
      ▼ (3) IR emit
JSON IR (includes type metadata per input/output port)
      │
      ├─▶ (4a) RayBackend  ── RuntimeDagExecutor + Ray actors
      └─▶ (4b) TorchBackend  ── torch.fx / torch.compile graph
```

## Python AST Version Adaptation

Python's `ast` module changes between minor releases: new node types are added
(e.g. `ast.TypeAlias` in 3.12), existing nodes gain or lose fields, and the
concrete grammar for certain constructs differs.  The compiler shields the rest
of the pipeline from these differences through a thin **`ASTVersionProvider`**
abstraction.

```python
class ASTVersionProvider:
    """Wraps the `ast` standard library for a specific CPython minor version."""

    @property
    def version(self) -> tuple[int, int]:
        """Return the (major, minor) Python version this provider targets."""
        ...

    def parse(self, source: str, filename: str = "<string>") -> ast.Module:
        """
        Parse `source` into an AST, applying any version-specific pre-processing.
        Raises `DSLSyntaxError` on parse failure.
        """
        ...

    def get_function_body(self, func_def: ast.FunctionDef) -> list[ast.stmt]:
        """
        Return the statement list for `func_def`, normalising version-specific
        differences in how the body is represented.
        """
        ...

    def get_annotation(self, node: ast.arg | ast.AnnAssign) -> str | None:
        """
        Extract a type annotation string from an argument or annotated assignment,
        handling differences in how annotations are stored across versions.
        """
        ...
```

Concrete providers are registered by Python version and selected automatically
at runtime:

```python
from rayorch.compiler.ast_version import get_ast_provider

provider = get_ast_provider()          # picks Py310ASTProvider, Py311ASTProvider, …
ast_tree = provider.parse(source, filename="pipeline.py")
```

Bundled providers:

| Class | Target Python | Notes |
|---|---|---|
| `Py310ASTProvider` | 3.10.x | Baseline |
| `Py311ASTProvider` | 3.11.x | Handles `ast.TryStar` (PEP 654) |
| `Py312ASTProvider` | 3.12.x | Handles `ast.TypeAlias` (PEP 695) |

Adding support for a new Python release requires only a new `ASTVersionProvider`
subclass; the rest of the compiler pipeline is unchanged.

## Python DSL

The DSL is **strictly a subset of standard Python syntax**.  Every valid DSL
file is also a syntactically valid Python file; the compiler only adds semantic
restrictions on top of the standard grammar.  This means:

- `python -m py_compile pipeline.py` must succeed.
- Any Python 3.10+ interpreter can import the file; the compiler is not
  required to evaluate it.
- DSL-specific constraints are enforced at semantic analysis (stage 2), not at
  the syntax level, so error messages always reference the offending Python
  construct.

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

The IR is a JSON object with the following top-level keys.  Type annotations
from the DSL source are extracted during semantic analysis and recorded in the
IR so that downstream backends and tooling can use them for optimization,
validation, and code generation without re-parsing the source.

```json
{
  "version": "1",
  "name": "MyPipeline",
  "inputs": [
    {"name": "pdf",  "type": "list[str]"},
    {"name": "meta", "type": "list[dict]"}
  ],
  "outputs": [
    {"ref": "render.out0", "type": "list[str]"}
  ],
  "nodes": [
    {
      "id":      "pdf2img",
      "op":      "MyPipeline.pdf2img",
      "inputs":  [
        {"ref": "$pdf",  "type": "list[str]"},
        {"ref": "$meta", "type": "list[dict]"}
      ],
      "outputs": [
        {"name": "images", "type": "list[list[str]]"},
        {"name": "meta",   "type": "list[dict]"}
      ],
      "config":  {"replicas": 2, "max_inflight": 4}
    },
    {
      "id":      "ocr",
      "op":      "MyPipeline.ocr",
      "inputs":  [
        {"ref": "images", "type": "list[list[str]]"}
      ],
      "outputs": [
        {"name": "text", "type": "list[str]"}
      ],
      "config":  {"replicas": 4, "max_inflight": 4}
    },
    {
      "id":      "render",
      "op":      "MyPipeline.render",
      "inputs":  [
        {"ref": "text", "type": "list[str]"},
        {"ref": "meta", "type": "list[dict]"}
      ],
      "outputs": [
        {"name": "render.out0", "type": "list[str]"}
      ],
      "config":  {"replicas": 2, "max_inflight": 2}
    }
  ]
}
```

### Type information in the IR

Type strings use PEP 484 annotation syntax (`list[str]`, `dict[str, Any]`,
`tuple[int, ...]`) as they appear in the source.  The semantic analysis stage
resolves types through the following priority order:

1. **Explicit annotation** — the `run()` method signature of the concrete
   operator class (most authoritative).
2. **Inferred from `forward()` annotations** — input/output names that appear
   in `forward()` carry over their annotated types.
3. **`"unknown"` sentinel** — when no annotation is found, the type is recorded
   as the string `"unknown"`.  Backends treat `"unknown"` as opaque and skip
   any type-dependent optimisation.

Type information is recorded per-port so that backends can:

- validate that connected ports carry compatible types at build time;
- select device placement (e.g. push `"torch.Tensor"` ports to GPU);
- generate typed stubs for IDE support.

Type inference is **advisory**: the runtime does not enforce types at execution
time.  Production correctness still depends on operator implementations.

### Field reference syntax

The IR distinguishes **declarations** (`"name"`) from **references** (`"ref"`):

- **`"name"`** — declares a new port.  Used in top-level `inputs` (pipeline
  input port definitions) and in node `outputs` (ports produced by a node).
- **`"ref"`** — references an existing port.  Used in top-level `outputs`
  (pointing at node output ports that become pipeline outputs) and in node
  `inputs` (consuming outputs of upstream nodes or pipeline inputs).

Source inputs are referenced with a `$` prefix (`$pdf`).  Node outputs are
referenced as `node_id.port_name` or the local alias assigned in `forward()`;
all aliases are resolved to fully-qualified references during semantic analysis.

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
from rayorch.compiler.backends.registry import register_backend

register_backend("mybackend", MyBackend)
```

## IR Stability

The IR schema is versioned (`"version": "1"`).  Breaking changes increment the
major version.  Additive changes (new optional node keys, new config keys) are
non-breaking.  Backends must ignore unknown keys to remain forward-compatible.

Serialized IR files can be checked into source control, diffed, and used to
reproduce a pipeline exactly, independent of the Python class hierarchy that
generated them.
