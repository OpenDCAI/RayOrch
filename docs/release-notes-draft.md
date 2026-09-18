# Release notes — draft

Unreleased. Updated: 2026-09-18. Release version and date are not yet assigned.
Completed changes below refer to the working tree; planned changes are explicitly
marked and must be verified before being announced as shipped.

## Completed changes

- Clarified InputBatch and ExecutionMicrobatch terminology, retry limits and
  metric units. See the [API migration guide](api_migration.md) for breaking names.
- Fixed infrastructure recovery when every Grain in a batch is suppressed.
- Preserved nested group depth through Filter and Broadcast, including empty
  groups, and used the compiled shape contract consistently during Reduce.
- Added explicit `OutputIssue` results for non-present final outputs and rejected
  Python truth-value checks on symbolic Ports.
- Removed automatic Worker observation and UDF audit-field probing. Driver-owned
  execution metrics remain; process/resource observation uses Ray Dashboard.
- Replaced repeated Reduce membership scans with a member count and a forward-only
  value cursor. Per-group work is O(E + N + L), with O(1) additional persistent
  state per unfinished group: E dependency deliveries, N children and L output
  layout/leaf references. Existing facts and output construction still consume
  space proportional to their size. This is an operation-count improvement, not
  a measured end-to-end throughput claim.
- Replaced source-triggered Broadcast scans of the entire target Domain with
  traversal of the source's existing Expansion subtree, without a waiting table.
- Consolidated UDF recovery in the Engine: partition the failed batch once,
  evaluate the existing pure policy, then apply the result. Executor retains
  failure classification, error wrapping, actor/RPC ownership and metrics.
  Retry budgets, batch boundaries and isolation policies are unchanged; no
  additional RPC, queue or persistent cache is introduced by this refactor.
- Removed the unused pre-release `F.optional` / `MISSING` input mechanism.
  Calls now have one input rule: all inputs must be present to execute; dropped
  inputs propagate dropped outputs, while failed or suppressed inputs propagate
  suppression. This also removes the optional-mode wrapper and Worker sentinel
  branch rather than retaining compatibility aliases.
- Fixed Executor deadlock detection when active InputBatch capacity is full and
  later source slices have not entered. Remaining input now postpones the
  deadlock check only when the next loop can actually admit another batch;
  cleanup-only semantic progress and pending RPC behavior are unchanged.
- Replaced the MinerU wheel-building Job launcher with a lazy `MinerUBench`
  Python API. Input count, GPU count, batching, model, and paths are explicit
  configuration; `run()` returns a persisted `BenchmarkReport` with scheduling,
  throughput, RSS, and driver-visible GPU metrics. `submit()` uses Ray Jobs with
  either the installed cluster environment or explicit `LocalSource` upload, so
  wheel construction is unnecessary and source roots are never inferred from
  an installed package.
- Separated reusable Benchmark infrastructure (`rayorch.benchmark`) from
  built-in workload implementations (`rayorch.benchmarks`). The public import
  remains `from rayorch.benchmark import MinerUBench`.
- Reduced the built-in workload contract to four meaningful files:
  `udfs.py`, `pipeline.py`, `env.json`, and a thin `benchmark.py`.
  Workload-specific plugin, CLI, runner, validation, profile, and submission
  files are not part of the authoring model.
- Promoted the historical Docling-shaped, video-caption, and dual-modality
  graph regressions into three dependency-free reference Benchmarks under
  `rayorch.benchmarks`. They use deterministic synthetic UDFs and the same
  four-file workload layout, so they are runnable examples rather than hidden
  test-only Pipelines. They do not claim to ship the old production model
  adapters.
- Ported the original YOLO -> SAM and dual-vLLM examples to the current
  declarative Pipeline and Benchmark APIs. The old `EnvRegistry`, eager actor
  construction, custom dispatch, downloader, and workload-owned Ray lifecycle
  are not restored. Every built-in case now includes a dedicated README.
- Made per-stage execution environments an explicit public feature through
  Ray's native `RayModule.ray_options(runtime_env=...)` contract. Added a
  minimal SGLang -> vLLM Benchmark whose model actors run in separate Conda
  environments, plus an opt-in cross-Conda integration test. Heavy dependency
  validation now happens inside the configured actor instead of incorrectly
  requiring model libraries in the driver environment.

## Recovery overhead

Recovery-entry consolidation reduces repeated driver-side classification and
temporary allocations. For K Grains it remains O(K) time and O(K) worst-case
temporary space; it does not eliminate driver bottlenecks or guarantee lower RSS.
Binary failure isolation may still create small batches and additional execution
RPCs, and normal dispatch may send a partially filled batch when an actor is idle.
Rebatching, recovery-queue indexing and distributed scheduling are separate changes.
Controlled dispatch tests preserve task membership/order, generations, batch sizes
and RPC counts; real Ray tests also check retry/isolation RPC counts and batch sizes.

## Broadcast: memory and traversal tradeoff

**Status: implemented and regression-tested in the working tree; unreleased.**
When a source Item arrives, Broadcast traverses existing `ExpansionRecord.children`
along the relevant Domain path, visiting descendants of that source. Newly
created targets continue to use the existing Entity-event path to read available
source facts. Values, outcomes and causes retain their existing semantics.
Targets released by one source event are visited in depth-first child-ordinal
order, rather than target-Domain creation order. The READY queue remains FIFO;
the grouping/order of work released together may consequently change.

### Keep the kernel small

Do not add a per-target waiting table or automatic strategy switching for this
release. There is currently no evidence that deep, sparse workloads justify
maintaining another runtime mechanism. Revisit this decision when users report
a concrete workload where Broadcast traversal is a bottleneck; use the reported
reproduction and time/memory measurements to guide any implementation.

Existing Expansion facts already record child relationships. Reusing them avoids
a second persistent collection of waiting relationships, whose size could grow
with target count, Broadcast count and active InputBatch count. Slow source
results can retain those relationships for the lifetime of the pending work.

### Cost and known limitation

For one Broadcast rule in one InputBatch, let S be the source count, T the target
count, H the number of Domain levels crossed, and V the total nodes visited by
source-triggered traversal, including intermediate nodes.

- Total work: O((S + T) × H + V). The Domain parent path is read in O(H) per
  source event, avoiding an additional persistent path cache. For one-level
  Broadcast, total work is O(S + T).
- Additional traversal space: O(H), using a depth-first stack of
  iterators, without collecting all descendants or pushing all siblings at once.
- Deep, wide expansions with very few final targets may require visiting many
  intermediate nodes. This approach is not guaranteed to outperform a waiting
  table or the current target scan for every data distribution.
- O(H) describes auxiliary traversal space only. Entity/Item/Expansion facts,
  pending work, payloads and outputs retain their existing memory costs.
  `input_batch_size` bounds source rows, not expanded descendants;
  `max_active_input_batches` bounds overlapping batches, not bytes.

Validation covers five arrival orders, all Item outcomes, empty/failed expansions,
multiple sources and rules, unrelated Domain branches, and 1,200-level traversal
without Python recursion. Dense/sparse operation counts and a traversal-only
allocation check cover the time/space tradeoff. The complete Ray regression suite
passed (206 tests); detailed measurements are in QA-08 of the release review.
These checks do not establish end-to-end throughput or total driver memory limits.
Sum work across Broadcast rules and account for all active batches when evaluating
memory.

## Release preparation

Implementation decisions and validation records are maintained in the
[release review](release-review_2026-09-16_105922.md). Resolve pending statuses and
assign the release version/date before publishing these notes.
