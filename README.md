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

## Lazy Benchmarks

Benchmarks are registered lazily and expose a typed Python API. Importing
`rayorch` or `rayorch.benchmark` does not import Ray, vLLM, Flash-MinerU, or
other workload dependencies. Constructing a Benchmark only stores validated
configuration; dependencies and models are loaded when `run()` actually starts.

```python
from rayorch.benchmark import MinerUBench

bench = MinerUBench(
    input_path="/shared/data/pdfs",
    input_limit=368,
    model="/shared/models/MinerU2.5",
    output_dir="/shared/output",
    num_gpus=8,
    batch_size=64,
    input_batch_size=24,
    max_active_input_batches=3,
    # Optional advanced overrides for individual Pipeline stages:
    stage_options={
        "ocr": {"batch_size": 32},
        "assemble": {"replicas": 4},
    },
)

report = bench.run(ray_address="auto")
report.print_summary()
```

The report contains input/output counts, wall time, throughput, per-UDF
actor/RPC/Grain/batching metrics, driver RSS, and best-effort GPUs visible to
the driver. Artifacts default to
`OUTPUT_DIR/.rayorch-benchmark/RUN_ID/`.

Use `submit()` to run the same configuration as a Ray Job:

```python
from rayorch.benchmark import LocalSource

# Stable cluster image: no source argument is needed.
run = bench.submit("http://RAY_DASHBOARD:8265")

# Development: Ray uploads source and installs the workload dependencies.
run = bench.submit(
    "http://RAY_DASHBOARD:8265",
    source=LocalSource(
        project_root="/path/to/RayOrch",
        modules=("/path/to/Flash-mineru/flash_mineru",),
    ),
)

report = run.wait()
```

Input, model, output, and artifact paths used by a Ray Job must be visible from
the cluster. Wheel construction and a workload-specific CLI are not part of the
Benchmark contract.

Framework code is under `rayorch/benchmark/`; built-in workloads that can be
copied as examples are under `rayorch/benchmarks/`. See
[benchmark framework and workloads](docs/benchmarks.md) for the short
authoring flow, multi-node contract, submission, and runtime-environment details.

Three dependency-free reference Benchmarks make the historical graph test
cases directly runnable:

```python
from rayorch.benchmark import (
    DocumentTopologyBench,
    VideoCaptionTopologyBench,
    VideoMultimodalTopologyBench,
)

report = VideoCaptionTopologyBench(
    output_dir="./results",
    video_count=8,
    frames_per_video=16,
    workers=4,
).run()
```

They demonstrate nested fan-out/reduction, video frame fan-out, and sibling
audio/vision relations. Their UDFs are synthetic; MinerU remains the included
real-model Benchmark. The original YOLO -> SAM and dual-vLLM examples are also
available as `YoloSamBench` and `DualVllmBench`. `SglangVllmBench` is the
minimal cross-environment example: its SGLang and vLLM stages use separate
Ray-native Conda `runtime_env` settings. Every built-in Benchmark directory
contains a dedicated README with its topology, setup, run command, resource
requirements, and result format.

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
pools. A UDF may return `RecordFailure` or `GroupFailure`; both are available
from the package root.

Calls execute only when every input Item is `PRESENT`. A filtered (`DROPPED`)
input propagates `DROPPED` outputs without invoking the UDF; `None` remains an
ordinary business value.

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
