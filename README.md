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

## License

Apache-2.0. See `LICENSE`.
