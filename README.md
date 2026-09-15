# RayOrch

RayOrch is a cardinality-aware dataflow runtime for Ray. It models map,
one-to-many expansion, filtering, broadcast, and ordered reduction while
dispatching each downstream grain as soon as its own dependencies are ready.

## Install

```bash
pip install rayorch
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Core Concepts

- `Pipeline` traces a declarative graph without constructing UDF instances.
- `RayModule` defines one persistent, batched Ray actor pool.
- `F.expand`, `F.filter`, `F.broadcast`, and `F.reduce` express cardinality and
  lineage without creating structural actors.
- `Executor` runs overlapping microbatches through completion-driven per-Call
  READY queues. A downstream stage can start before its upstream stage drains.
- Recovery, failure attribution, and output reconstruction operate on stable
  logical Grain and Entity identities.

Compiler, scheduler, and Ray actor internals live in private modules. Application
and benchmark code should import only from `rayorch` and `rayorch.benchmark`.

## Lazy benchmark plugins

Benchmark workloads live below `rayorch.benchmark` and are imported only when
used. Each UDF group owns one Ray `runtime_env` file, while its business UDFs,
pipeline, registration metadata, and launcher remain separate. For example,
`import rayorch` does not import Ray, vLLM, or Flash-MinerU; importing the
MinerU pipeline still does not load those heavy runtime packages.

The MinerU Ray Jobs launcher builds small source wheels in a temporary staging
directory and submits them with the group's shared environment:

```bash
rayorch-mineru-job \
  --address http://RAY_DASHBOARD:8265 \
  --flash-repo /path/to/Flash-mineru \
  -- \
  --model /path/to/MinerU-model \
  --input-manifest /path/to/pdfs.json \
  --golden-corpus-sha256 EXPECTED_DIGEST \
  --replicas 4 \
  --output-dir /shared/output \
  --artifact-dir /shared/artifacts \
  --result-jsonl /shared/results.jsonl
```

The historical 368-document performance gate is evaluated only when the
observed input digest and the expected 7,072-page shape both match. Other runs
still report timing and batching metrics, but are not labeled comparable.

See [benchmark plugin layout](docs/benchmark_plugins.md) for extension and
runtime-environment details.

## Minimal Example

```python
import rayorch as ro


class AddOne:
    def run(self, values):
        return [value + 1 for value in values]


class Pipe(ro.Pipeline):
    def __init__(self):
        self.a = ro.RayModule(AddOne).ray_options(replicas=1, batch_size=8)
        self.b = ro.RayModule(AddOne).ray_options(replicas=1, batch_size=8)

    def forward(self, values):
        return self.b(self.a(values))


result = ro.run(
    Pipe(),
    [1, 2, 3],
    microbatch_size=2,
    max_active_microbatches=2,
)

print(result.outputs)  # [3, 4, 5]
```

Use `Executor` directly when several runs should reuse the same persistent actor
pools. A UDF may return `RecordFailure` or `GroupFailure`, and receives `MISSING`
for a missing input declared with `F.optional`; these values are also available
from the package root.

## License

Apache-2.0. See `LICENSE`.
