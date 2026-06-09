# RayOrch

Lightweight orchestration utilities for building asynchronous Ray pipelines with
`RayModule`, overlapped microbatch execution, and DAG-style scheduling.

## Install

```bash
pip install rayorch
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Core Concepts

- `RayModule`: wraps an operator class into Ray actors with optional replica
  dispatch and collect.
- `OverlappedPipeline`: graphless microbatch overlap with backpressure.
- `DagPipeline` / `DagExecutor`: symbolic multi-input/multi-output DAG
  scheduling.
- `RuntimeRayModule` / `RuntimeDagExecutor`: row-level fault isolation,
  lineage, Ray replicas, and overlapped microbatch execution.

Runtime MVP usage and current limitations are documented in
[`docs/runtime_mvp.md`](docs/runtime_mvp.md).
The lifecycle and feature boundary between lightweight `RayModule` and
executor-managed `RuntimeRayModule` is documented in
[`docs/runtime_module_lifecycle.md`](docs/runtime_module_lifecycle.md).

## Minimal Example

```python
from rayorch import OverlappedPipeline, RayModule


class AddOne:
    def run(self, x):
        return x + 1


class Pipe(OverlappedPipeline):
    def __init__(self):
        self.a = RayModule(AddOne, replicas=1).pre_init()
        self.b = RayModule(AddOne, replicas=1).pre_init()
        super().__init__(max_inflight=4)

    def forward(self, x):
        return self.b(self.a(x))


pipe = Pipe()
print(pipe([1, 2, 3]))  # [3, 4, 5]
```

## Runtime Tests

Fast compile and local semantics tests do not start Ray:

```bash
pytest test/runtime/unit
```

Runtime integration tests share one local Ray cluster per test module:

```bash
pytest test/runtime/integration
```

Long performance regressions are skipped by default. Run them explicitly with:

```bash
pytest --runslow -s test/runtime/performance
```

Run the 512-PDF Flash-MinerU-style fault-isolation demo. Its default topology
simulates four layout GPU workers, four OCR GPU workers, and four microbatches
in flight:

```bash
python examples/runtime_flash_mineru_512.py
```

Replica counts, batch size, inflight limits, and dummy inference delay are
available as command-line options:

```bash
python examples/runtime_flash_mineru_512.py --help
```

## License

Apache-2.0. See `LICENSE`.
