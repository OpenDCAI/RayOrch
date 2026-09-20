# Preview API migration

The pre-release naming cleanup distinguishes input batches from Worker execution
microbatches and makes retry limits and metric units explicit.
Update the following keyword arguments, attributes, and serialized field names;
the old names are not aliases. Execution behavior and counter timing are unchanged.

## Batch names

| Previous name | Current name |
| --- | --- |
| `run` / `Executor.run`: `microbatch_size` | `input_batch_size` |
| `run` / `Executor.run`: `max_active_microbatches` | `max_active_input_batches` |
| `RunResult.microbatches` | `RunResult.input_batches` |
| `RunResult.peak_active_microbatches` | `RunResult.peak_active_input_batches` |
| `rayorch.result.MicrobatchMetrics` | `rayorch.result.InputBatchMetrics` |
| MinerU CLI `--microbatch-size` | `--input-batch-size` |
| MinerU CLI `--max-active-microbatches` | `--max-active-input-batches` |
| MinerU JSON `microbatch_size` | `input_batch_size` |
| MinerU JSON `max_active_microbatches` | `max_active_input_batches` |
| MinerU JSON `active_arenas_high_watermark` | `peak_active_input_batches` |

An InputBatch owns a slice of aligned source rows and all work derived from them.
`input_batch_size` counts these rows; `None` uses the full input.
`max_active_input_batches` limits overlapping input batch lifecycles.
An ExecutionMicrobatch groups Grains of one Call within one InputBatch for a
Worker RPC. The existing `ray_options(batch_size=...)` bounds that group and
keeps its name. A group need not fill this limit.

Internally, `MicrobatchEngine` becomes `InputBatchEngine` and `DispatchBatch`
becomes `ExecutionMicrobatch`. InputBatch is a lifecycle concept, not a new
wrapper class. Module paths and scheduling behavior are unchanged.

## Retry limits and metrics

| Previous name | Current name |
| --- | --- |
| `RecoveryPolicy.retry_batch(attempts=n)` | `RecoveryPolicy.retry_batch(max_retries=n)` |
| `RecoveryPolicy.retry_tail(attempts=n)` | `RecoveryPolicy.retry_tail(max_retries=n)` |
| `RecoveryPolicy.udf_attempts` | `RecoveryPolicy.max_udf_retries` |
| `CallMetrics.grains` | `CallMetrics.grain_dispatches` |
| `CallMetrics.retries` | `CallMetrics.grain_requeues` |
| `WorkerSnapshot` / `CallMetrics.worker_snapshots` | Removed; see Worker observation below |
| MinerU JSON `ocr_grains` | `ocr_grain_dispatches` |
| MinerU JSON `ocr_retries` | `ocr_grain_requeues` |

`max_retries=2` allows two extra UDF retries after the initial execution. Both
retry constructors require a positive limit; use `RecoveryPolicy.abort()` for
no UDF retries. Infrastructure retries have their own unchanged `infra_retries`
budget. With `isolate_tail()`, the single whole-batch retry precedes recursive
isolation; it does not cap the number of split-batch executions.

`grain_dispatches` includes repeated dispatches of the same logical Grain.
`grain_requeues` counts Grains requeued for recovery, even if the next dispatch
has not happened yet. `InputBatchMetrics.grain_count` remains the logical Grain
count. `average_batch` retains its existing dispatch-based calculation.

For runtime maintainers, `DispatchState.is_idle` is now `queues_empty`: empty
ready and recovery queues can coexist with in-flight Grains. Executor counter
fields use the same units as their corresponding public metrics.

`pre_init()`, `Item`, `Grain`, their identity references, and failure names remain
unchanged.

## Required Call inputs

The pre-release `F.optional(port)` and `MISSING` input sentinel have been removed.
They had no benchmark or production adapter using them and added a second Call
input policy throughout the compiler, runtime state machine, and Worker ABI.

All Call inputs now follow one rule:

- all inputs are `PRESENT`: execute the Grain;
- any input is `DROPPED`: do not execute it and publish `DROPPED` outputs;
- any input is `FAILED` or `SUPPRESSED`: do not execute it and publish
  `SUPPRESSED` outputs.

`None` remains an ordinary business value and is not treated as missing. Sparse
branch combination is not a structural primitive in this release; applications
can keep a tagged optional value inside one ordinary business Item when needed.

## Output results

Non-present final outputs now return `rayorch.OutputIssue` instead of a bare
private enum. `rayorch.ItemOutcome` exposes the existing four terminal states;
no new runtime states or recovery actions are introduced.

| Outcome | Meaning | Final representation |
| --- | --- | --- |
| `PRESENT` | A value exists, including a valid `None` or empty list | The original business value |
| `DROPPED` | Filtered out, or downstream of an input that was filtered | `OutputIssue(ItemOutcome.DROPPED)` |
| `FAILED` | A failed value, including one inherited through a transparent view | `OutputIssue(ItemOutcome.FAILED, cause)` |
| `SUPPRESSED` | A result unavailable or discarded because of another failure or dependency outcome | `OutputIssue(ItemOutcome.SUPPRESSED, cause)` |

`OutputIssue` is a frozen two-field record: `outcome: ItemOutcome` and
`cause: str | None = None`. It rejects PRESENT, non-enum outcomes, and non-text
causes. A suppressed output does not imply that remote computation never began.

Replace checks such as `output is ItemOutcome.FAILED` with:

```python
if isinstance(output, ro.OutputIssue) and output.outcome is ro.ItemOutcome.FAILED:
    print(output.cause)
```

Cause rules are deliberately small:

- Ordinary drops have no cause text.
- Business failure strings are preserved; other supplied causes are converted
  to text. A missing cause remains `None`, and an empty string remains empty.
- A UDF exception isolated into a record failure uses `error_type: message`.
- Propagation follows the existing selected cause to its source. It does not
  aggregate all failures or expose internal identity/protocol objects.
- If a business cause cannot be converted to text, its class name is retained.

Text is diagnostic, not an error code or a reconstruction of the original cause
object. Tracebacks remain in the existing execution-error path. Final issue
records retain no Engine references and remain usable after executor cleanup.

Reduce still omits dropped members and retains its existing failure propagation
rules. Outputs describe final positions, not a complete failure log. A whole-run
abort still raises an exception rather than returning partial issue records.
The existing public `CompileError`, `ExecutionError`, `RecordFailure`, and
`GroupFailure` retain their roles.

`OutputIssue` is a reserved result marker. Supplying one as an ordinary output
value (including a structural group's leaf value) raises `ExecutionError` during
materialization. Opaque business containers are not recursively inspected.

## Symbolic Port truth checks

Ports now reject Python truth conversion with `TypeError`. Previously, expressions
such as `if port:` silently treated the symbolic object as true during tracing,
regardless of the eventual input values. The same restriction applies to
`bool(port)`, `not port`, and `and`/`or` when they evaluate a Port's truth value,
including inside helper functions.

Use `F.filter(values, masks)` for data filtering; place per-item business
conditions inside a UDF. Ordinary configuration booleans can still choose static
graph branches. This guard does not inspect source syntax or implement dynamic
control flow. Object-identity checks such as `port is not None` do not convert a
Port to bool and do not test its eventual business value.

## Worker observation

Automatic end-of-run Worker observation has been removed. `WorkerSnapshot` and
`CallMetrics.worker_snapshots` no longer exist; this also removes the interim
`lifetime_batches_received` name introduced earlier in the preview review.
No empty compatibility field or alias is retained.

`run()` returns after output processing and freezing driver-owned metrics, without
issuing observe RPCs or invoking `batch_audit()` / probing `last_*_audit` fields.
Worker-side lifetime counters, PID/RSS snapshots, and custom audit dictionaries
are no longer collected. Ray Dashboard is the chosen process/resource observation
tool; application-specific diagnostics belong to the application or benchmark.

Existing RPC, Grain dispatch/requeue, execution batch-size, actor-instance,
input-batch, and timing metrics retain their definitions. RPC dispatch counts
are not a substitute for the removed count of batches actually received by a
Worker. Scheduling, recovery, and final output semantics are unchanged.

## Benchmark API

The pre-release `rayorch-mineru-job` launcher and
`rayorch.benchmark.mineru.job` module have been removed. They inferred a source
checkout from an installed package and always built transport wheels, so the
installed CLI contract was not reliable.

Benchmark framework code now lives in `rayorch.benchmark`; built-in workload
implementations live separately in `rayorch.benchmarks`. The supported public
import remains unchanged:

Use the lazy Python API instead:

```python
from rayorch.benchmark import MinerUBench

bench = MinerUBench(
    input_path="/shared/pdfs",
    input_limit=368,
    model="/shared/models/mineru",
    output_dir="/shared/results",
    num_gpus=8,
    batch_size=64,
)

report = bench.run(ray_address="auto")
```

There is no replacement workload-specific CLI. Python is the single public
entrypoint, avoiding a duplicate parameter surface.

Remote submission uses Ray Jobs through `MinerUBench.submit()`:

- omit `source` when RayOrch and workload dependencies are already installed;
- pass `source=LocalSource(project_root, modules=...)` for Ray-managed source
  upload and dependency installation;
- set `install_dependencies=False` only for an already prepared environment.

Wheel construction is no longer required, and no source root is inferred from
`site-packages`. Input, model, output, and artifact paths used remotely must be
visible from the cluster.
