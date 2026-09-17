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
- `RayModule` declares a UDF and its configuration. Each use in the graph creates
  a Call with its own persistent actor pool.
- `F.expand`, `F.filter`, `F.broadcast`, and `F.reduce` express cardinality and
  lineage without creating structural actors.
- `Executor` runs overlapping input batches through completion-driven per-Call
  READY queues. A downstream stage can start before its upstream stage drains.
- Recovery, failure attribution, and output reconstruction operate on stable
  logical Grain and Entity identities.

Compiler, scheduler, and Ray actor internals live in private modules. Application
and benchmark code should import only from `rayorch` and `rayorch.benchmark`.

`forward()` builds a static graph using symbolic Ports. A Port has no Python
truth value: `if port`, `bool(port)`, and other truth checks raise `TypeError`.
Use `F.filter` for data filtering or put per-item conditions inside a UDF.
Ordinary configuration booleans may still choose graph branches in `forward()`.

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
    input_batch_size=2,
    max_active_input_batches=2,
)

print(result.outputs)  # [3, 4, 5]
```

`input_batch_size` counts aligned source rows; `max_active_input_batches` limits
how many input batches can overlap. Each input batch owns its derived work and
state. A Call groups its ready Grains into an `ExecutionMicrobatch` for one Worker
RPC, bounded by `ray_options(batch_size=...)`. It never mixes input batches.

Use `Executor` directly when several runs should reuse the same persistent actor
pools. A UDF may return `RecordFailure` or `GroupFailure`, and receives `MISSING`
for a missing input declared with `F.optional`; these values are also available
from the package root.

Final outputs keep successful business values unchanged, including `None` and
empty lists. A value that is dropped, failed, or suppressed is represented by
`OutputIssue(outcome, cause)`. Use the public `ItemOutcome` enum for decisions;
`cause` is optional text for diagnostics:

```python
for output in result.outputs:
    if isinstance(output, ro.OutputIssue):
        print(output.outcome.name, output.cause)
    else:
        consume(output)
```

`RecordFailure` and `GroupFailure` remain UDF return signals. `OutputIssue` is
reserved for final framework results, not ordinary business output values.
See the [result contract](docs/api_migration.md#output-results) for all outcomes
and cause rules.

Execution metrics come from the driver's existing counters. `run()` does not
issue diagnostic RPCs or call UDF audit methods after processing completes.
Use Ray Dashboard for process/resource observation; per-Worker snapshots are
not part of `RunResult`.

See the [preview API migration guide](docs/api_migration.md) for the explicit
batch, retry-limit, and metric names introduced during the pre-release review.

## License

Apache-2.0. See `LICENSE`.
