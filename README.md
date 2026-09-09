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
- `DagPipeline` / `DagPipelineExecutor`: explicit dependency DAG scheduling.
- `rayorch.multigrain`: cardinality-aware map/expand/reduce runtime promoted from
  the v3.6 experiments.

## Lazy benchmark plugins

Benchmark workloads live below `rayorch.benchmark` and are imported only when
used. Each UDF group owns one Ray `runtime_env` file, while its business UDFs,
framework adapters, validation, and runner remain separate. For example,
`import rayorch` does not import Ray, vLLM, Flash-MinerU, or Daft; accessing
`rayorch.benchmark.mineru.pipeline` loads only the requested MinerU surface.

The MinerU Ray Jobs launcher builds small source wheels in a temporary staging
directory and submits them with the group's shared environment:

```bash
rayorch-mineru-job \
  --address http://RAY_DASHBOARD:8265 \
  --flash-repo /path/to/Flash-mineru \
  -- \
  --model /path/to/MinerU-model \
  --input-manifest /path/to/pdfs.json \
  --replicas 4 \
  --output-dir /shared/output \
  --artifact-dir /shared/artifacts \
  --result-jsonl /shared/results.jsonl
```

See [benchmark plugin layout](docs/benchmark_plugins.md) for extension and
runtime-environment details.

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
