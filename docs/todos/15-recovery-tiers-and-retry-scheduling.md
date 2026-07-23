# Recovery Tiers & Retry Scheduling (design spec)

Status: **retry granularity finalized; Map record retry, immediate shard retry,
budgeted adaptive localization, actor replacement, and bounded stage-epoch
deferred drain implemented. Other operator record units remain incremental.**
Operationalizes the "adaptive isolation rather than unconditional binary
splitting" bullet of [`08-lineage-guided-partial-replay.md`](08-lineage-guided-partial-replay.md).
Reordering-invariance guarantees referenced here are proved in
[`13-reordering-invariance-theorem.md`](13-reordering-invariance-theorem.md).

This doc records the **full option space** (chosen and not), the **locked
decisions**, the **abstraction** (so all options have a stable interface home),
and an **incremental implementation plan**. Nothing here is a fly-line: every
tier funnels into the *same* quarantine → (optional drain) → `missing_child`
cascade that already exists.

---

## 1. Locked decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | `retry_timing` a first-class knob? | **Yes** | The *when/where* of retry is a real axis, orthogonal to *how many times*. |
| D2 | `placement` / `batching` first-class? | **No — implied by `deferred`** | Avoid knob explosion (YAGNI). Deferred naturally re-places and batches. |
| D3 | Default granularity for `deferred` drain | **`stage_global` (logical target)** | Only stage-global reaps "re-place to dodge node-local transient faults + LPT-repack the recovery batch". The current executor only owns one submitted chunk at a time, so the implementable scope is initially **chunk-global** until the execution model is unified (§6). |
| D4 | Phasing | **Interface-first, implement/test incrementally** | Define the whole `RecoveryPolicy` now so the IR is stable; ship behaviors in phases. If any abstraction turns hard / fly-line-y, stop and discuss before coding. |
| D5 | Does `retry_timing` delay opaque shard retries? | **No** | Other shards already progress independently. An opaque failed invocation is immediately retried on a healthy replica; delaying it only increases tail latency. |
| D6 | Shard-exhausted localization | **Budgeted adaptive split** | Sparse poison is localized precisely; dense/systemic failure has a hard work/call bound. Whole-shard quarantine is not the default. |
| D7 | Localization budget exhaustion | **Fail-closed the smallest unresolved subsets** | Never emit incomplete results; retain every sibling subset that already succeeded. |
| D8 | Meaning of `stage_global` | **Bounded stage epoch** | A node-owned buffer spans active microbatches, not the entire dataset. Shard-size threshold, stage quiescence, or input close triggers drain. |

---

## 2. The tier ladder (a single lightest→heaviest staircase)

```
① record transient       → retry/defer logical record   max_record_retries + retry_timing
② record deterministic    → isolate logical record       BadRecordError(retryable=False)
③ opaque shard fault      → immediate healthy-replica retry  max_shard_retries
④ shard exhausted         → budgeted adaptive split      on_shard_exhausted="degrade"
⑤ budget exhausted        → quarantine unresolved leaves IsolationBudget
⑥ shard exhausted         → abort the job                on_shard_exhausted="abort"
⑦ missing consumer        → suppress / fail-open         missing_child (orthogonal)
```

`retry_timing` applies to attributable record recovery (①) and localization
work admitted to the recovery pool (④), not ordinary opaque shard retry (③).
An inline record retry runs now; a deferred retry quarantines now, keeps healthy
work flowing, then re-runs the set as a recovery batch.

The logical record unit follows the operator contract:

| operator | attributable retry/isolation unit |
|---|---|
| Map / Filter | one aligned input row |
| Expand | one parent row and all children emitted from it |
| Reduce | one anchor and its grouped descendants |
| Relate | one output relation/evidence group |

All output ports of one logical invocation recover atomically.

`recovery` (how hard to *produce* a record) and `missing_child` (what a
*consumer* does when a record is missing) are **orthogonal** and must stay so.

---

## 3. Full option space (every dimension, chosen or not)

Retry has four sub-dimensions. We first-class only two; the rest are recorded
here so the design is complete and the interface can grow without churn.

| Dimension | Options | Status | Notes |
|---|---|---|---|
| **count** | `0` (isolate now) … `k` | first-class (`max_record_retries`, `max_shard_retries`) | 0 record-retries = today's behavior. |
| **timing** | `inline`, `deferred` | **first-class (`retry_timing`)** [D1/D5] | Applies to attributable records/localization, not ordinary shard retry. |
| **placement** | `same_replica`, `any_replica` | scheduler-owned [D2/D5] | Timing and placement are separate. A dead actor always requires replacement; no user knob. |
| **batching** | `one_by_one`, `as_batch` | **implied by `deferred`** [D2] | deferred drain re-runs the quarantined set as one (LPT-repackable) batch. |
| **drain scope** | `shard_local`, `stage_global` | first-class (`drain_scope`), default **`stage_global`** [D3/D8] | stage-global means a bounded node-owned epoch, never an unbounded whole-dataset barrier. |
| **shard-exhausted action** | `abort`, `degrade` | first-class (`on_shard_exhausted`), default `abort` | `degrade` = tier ④. |
| **degrade localization** | coarse quarantine, singleton scan, adaptive split | adaptive split fixed initially [D6] | Not another public strategy knob; alternatives remain experiment baselines. |
| **localization budget** | work factor + call count + exhausted action | nested first-class `IsolationBudget` | Defaults: 3× localization row-work, 64 calls, then quarantine unresolved leaves. |

---

## 4. The abstraction: one policy, two sites, one sink

### 4.1 `RecoveryPolicy` (declarative node policy)

Lives on `NodeSpec.recovery` as the sole retry authority. It is threaded from the
primitive into the passive graph; the executor does not override it. Serializable
and passive.

```python
@dataclass(frozen=True)
class IsolationBudget:
    max_work_factor: float = 3.0
    max_calls: int = 64
    on_exhausted: str = "quarantine"

@dataclass(frozen=True)
class RecoveryPolicy:
    # -- record level (enforced in the op wrapper) --
    max_record_retries: int = 0            # 0 = isolate immediately (current)
    retry_timing: str = "inline"           # "inline" | "deferred"
    # -- shard level (enforced in the executor scheduler) --
    max_shard_retries: int = 2
    on_shard_exhausted: str = "abort"      # "abort" | "degrade"
    isolation: IsolationBudget = IsolationBudget()
    # -- deferred drain (only when retry_timing == "deferred") --
    drain_scope: str = "stage_global"      # "shard_local" | "stage_global"
    # placement & batching are IMPLIED by "deferred" (see D2), not fields.
```

All fields exist from day one (stable IR). Behaviors ship in phases (§7); an
executor that meets an unimplemented combination raises `NotImplementedError`
with a clear message rather than silently mis-behaving.

### 4.2 Fault classification lives on the exception (not scattered ifs)

- `BadRecordError(index=i, retryable: bool = False)` — record-attributable.
  `retryable=False` (default) = deterministic poison ⇒ retrying is pointless
  (value-purity), isolate. `retryable=True` = transient ⇒ eligible for ①.
- Any other exception ⇒ non-attributable ⇒ shard-level (③/④/⑤).

Decision logic is centralized, but should remain **scope-local** rather than
becoming one god function:

- `policy.decide_record(fault, attempt)` → `{RETRY_RECORD, ISOLATE_RECORD}`;
- `policy.decide_shard(fault, attempt)` → `{RETRY_SHARD, DEGRADE_SHARD, ABORT}`.

The two functions share one passive policy object but expose only the actions
their mechanism site can actually perform. Neither site hardcodes a retry count
or fallback branch.

### 4.3 Two mechanism sites (physically unavoidable, decision centralized)

- **Op-wrapper site** (shared isolation helper used by eligible wrappers):
  record-level. Only a wrapper understands the operator's logical record unit
  and can re-invoke the UDF on that subset. Handles ①/② and inline retry.
- **Scheduler site** (`multigrain.ray.executor._run_shards` / stage loop): shard-level.
  Only the driver can resubmit/split a shard and select a healthy actor. Handles
  ③–⑥ and the deferred **drain**.

### 4.4 The shared sink is DATA, not a process

The runtime "sink" is the passive `PortBatch.errors: list[ErrorTrace]` channel:
error traces ride *inside* the same `PortBatch` as values, are copied by
`take`/`with_values`, merged across shards by `concat`, and read by `Reduce`.
**No collector actor, no queue, no central bottleneck** — this preserves the
no-center / no-fly-line property and rides Ray's object store for free.

The **deferred drain pool** needs more than a description (it must retain the
failed *input*, to re-run it). It is node-owned driver metadata coordinated by
the generic executor/coordinator, drained through the existing persistent actor
pool. In Ray the retained payloads remain ObjectRefs, so the driver does not copy
large values.
**Never introduce a dedicated sink/collector actor** — that would be a bottleneck
and a fly-line.

### 4.5 Unified data path (every tier ends here)

```
run stage → healthy results ─────────────────────────────────────┐
          ↘ faults → quarantine (record row / degraded shard's rows)│
                        │  drain policy (inline? deferred? scope? k×)│
                        ▼                                           │
                   recovery drain  (re-run, may re-place; batched)   │
                        │ recovered → merge back by identity/lineage ┤
                        │ still-failed → ErrorTrace ─────────────────┘
                                                                    ▼
                                          missing_child cascade (final fate)
```

A new tier is a new *entry point* into this path, not a new path. Degrade first
splits and salvages successful subsets; only unresolved leaves enter quarantine.
Deferred drain retries that set elsewhere; cascade suppresses documents whose
required leaves remain unavailable.

---

## 5. Semantics & guarantees

- **Reordering-invariance holds for deferred retry.** Re-running a row later /
  on another replica and re-attaching its result by record identity is, under
  UDF value-purity, just another legal physical reordering ⇒ output and lineage
  are equivalent to the serial baseline (doc 13). Deferred retry needs **no new
  proof**; it is an instance of the existing theorem.
- **Degrade preserves lineage-driven downstream.** A degraded shard synthesizes
  one `ErrorTrace` per unresolved row *with `ancestors`*. Successful sibling
  subsets retain normal outputs, so the cascade affects only documents actually
  represented by unresolved leaves.
- **Localization cost is bounded.** Ordinary shard retries are excluded from the
  isolation budget. Before every split invocation, charge one call and its input
  row count. Stop before exceeding either `max_calls` or
  `max_work_factor × original_shard_rows`.
- **`abort` semantics unchanged.** `on_shard_exhausted="abort"` still raises;
  in a chunked driver, already-written outputs of prior chunks persist (that is
  not a global rollback — there is no job-level checkpoint yet; see §8).

---

## 6. Cost / complexity budget (honest)

- inline retry (①) is wrapper-local. Adaptive degrade (④) adds a bounded split
  worklist in the shard scheduler but no new process or side channel. It removes
  hardcoded retry decisions and makes dense-failure cost explicit.
- deferred drain (stage-global): this is where the cost is. It needs
  (a) a bounded node-owned quarantine buffer on the driver, and (b) a **drain
  dependency**:
  the `Reduce` `fail_closed` decision must wait for the drain to finish, else it
  could suppress a document that was about to be recovered. The barrier moves the
  affected downstream work becomes ready only after drain. The epoch drains when
  it reaches a normal shard target, healthy work is quiescent, or input closes;
  it never waits for an unbounded stream.

### 6.1 Execution-model prerequisite (initial convergence landed)

The previous runtime had three execution paths: node-by-node `execute`,
stateless whole-graph `execute_microbatches` (which lost model/pool reuse), and a
MinerU benchmark that manually called `_pool_for`/`run_shard`. This was an
architectural blocker for deferred recovery.

The recovery-tiers branch now has one initial public path:

- `ExecutionCoordinator` admits bounded microbatches, schedules arbitrary IR
  nodes when their input refs are ready, supports independent branches/fan-in,
  and yields ordered or completion-order results;
- `execute`, `execute_microbatches`, and `execute_stream` all delegate to it;
- node execution still reuses `_run_node`, so persistent actor pools, sharding,
  retry accounting, lineage, and metrics have one implementation;
- completed-but-unconsumed outputs count toward `max_inflight` (bounded
  backpressure);
- `Flash-mineru/mg_bridge/run_bench.py` now calls only the public
  `execute_stream`; it no longer imports private pools, actor methods, concat, or
  writes its own render/OCR/assemble scheduler.

Verified so far: 214 default tests + 16 Ray parallelism/recovery tests; sparse
opaque poison salvages healthy siblings, dense failure stops exactly at budget,
dead actors are replaced one replica at a time, and all three epoch triggers are
covered. A real 4-PDF/48-page/4-GPU MinerU smoke produced 4/4 markdown in 17.34 s
(minimum token Jaccard 0.9953 vs baseline). A real poison-page run executed 47
healthy page OCRs, quarantined one page, suppressed exactly its document, and
produced 3/3 logically consistent healthy markdowns (minimum Jaccard 0.9916).
The 368-PDF performance regression remains separate.

Remaining before **true stage-global deferred**: the coordinator must own the
node recovery epoch across microbatches rather than letting each `_run_node`
finish recovery independently. The epoch is bounded by a shard-size target and
the coordinator's admission window. Also measure per-node Ray-task overhead for
cheap CPU nodes and add per-stage queue limits if `max_inflight` alone is
insufficient.

### 6.2 Other scaling bottlenecks outside retry policy

1. **Per-record lineage allocation.** `PortBatch` currently carries several
   Python objects per row (`record_ids`, `ancestors`, `ancestor_display`,
   `ordinals`, `lineage`, `relations`), and `take`/`with_values` copy many of
   them at each stage. This is acceptable for thousands of pages but can become
   a GC/memory wall at millions of records. Required optimization: share 1:1
   lineage paths and use compact/columnar ancestor and ordinal storage. The
   coordinator-owned UUID, `BatchArena`, hash-consed path table, relation
   encodings, and migration order are specified in
   [`16-batch-arena-and-compact-lineage.md`](16-batch-arena-and-compact-lineage.md).
2. **Driver-side merge/control.** `concat`, stage coordination, and some Reduce
   work are driver-side. Unifying the scheduler must preserve bounded buffering
   and avoid turning the driver into a payload-copy bottleneck.
3. **In-memory `Relate` join.** The current key join builds all role indexes in
   one process and materializes per-key Cartesian products. It represents M:N
   correctly but is not a distributed large-join implementation; partitioned
   join/spill is future work.

---

## 7. Interface-first, incremental plan (per D4)

- **Interface (done):** `RecoveryPolicy` lives directly on `NodeSpec`; every
  user primitive accepts `recovery=`; it serializes through the passive
  `ExecutionGraph` and survives executor wrapper reconstruction.
  `BadRecordError(..., retryable=)` and scope-local `decide_record` /
  `decide_shard` are landed. Non-default behavior raises a clear
  `NotImplementedError`; defaults reproduce today's semantics.
- **Phase 1 (done for Map/shardable nodes):** inline Map record retry +
  budgeted adaptive shard localization, actor rotation/replacement, sparse and
  dense poison accounting. Expand/Filter/Reduce/Relate attributable units are
  explicitly rejected until their wrapper-specific contracts land.
- **Phase 1.5 (execution unification; initial implementation done):** public
  `execute_stream` + generic DAG coordinator + persistent-pool reuse are landed;
  `run_bench.py` no longer uses private scheduling APIs. Remaining gate: rerun
  368 PDFs and preserve the measured ~585 s performance/backpressure.
- **Phase 2 (done for Map):** node-owned recovery buffer + readiness dependency
  + re-placement/repacked recovery batch. Target rows, stage quiescence, and
  input close are tested. Deferred inputs use invocation-unique temporary tokens
  while combined, then restore original identity before downstream execution;
  no collector actor or workload-specific path exists.
- **Phase 3 (future / optional):** precise degrade localization (§8), job-level
  checkpoint/resume, durable quarantine backend (doc 05).

Guiding rule (D4): if at any phase the clean abstraction can't hold (would scatter
policy logic or add a side-channel), pause and revisit the design rather than
wiring around it.

---

## 8. Open risks / to-revisit

1. **Dense opaque failures.** Adaptive split is efficient for sparse poison:
   one bad row costs less than `2n` localization row-work. If failures occupy
   much of the shard, unrestricted splitting approaches `O(n log n)` row-work
   and nearly `2n` calls. `IsolationBudget` caps this; unresolved leaves are
   conservatively quarantined. A future strict-salvage mode may scan to
   singleton at intentionally unbounded cost.
2. **No job-level checkpoint.** `abort` restarts the run; chunked drivers keep
   already-written outputs but there is no resume-from-progress. Phase 3.
3. **Epoch drain vs. throughput.** Larger epochs improve recovery batching but
   delay affected records. Initial trigger size derives from normal shard size,
   not a new public knob; measure latency/goodput before exposing tuning.
4. **Single-index error reporting.** `BadRecordError(index=i)` can identify only
   one bad row per invocation, so many bad rows cause repeated peel-and-rerun.
   Preserve `index` for compatibility, but leave room for
   `BadRecordError(indices=[...])` so a UDF that can identify multiple failures
   can isolate them in one pass.
5. **Value-purity is an explicit scope boundary.** Reordering and deferred retry
   do not cover globally stateful/sessionized operators, global sorting,
   cross-record deduplication, iterative/cyclic dataflow, or unsafe external
   side effects. These are non-goals unless a future materialization/commit
   contract gives them explicit semantics.

---

## 9. Test plan (dummy-first)

- Reuse `test/experimental/multigrain/lineage_ops.py` poison hooks + the real
  `mg_bridge/mineru_poison_ops.py`.
- **Local tests**: assert ① a transient (`retryable=True`) row recovers within
  `k` and is NOT quarantined; ② deterministic stays isolated. Local execution
  has no shard and cannot validate ④/deferred scheduling.
- **Ray tests (required for scheduler tiers)**: assert ④ a degraded shard yields
  per-row quarantine traces and the `missing_child` cascade suppresses exactly
  the affected documents (others intact).
- Phase 2 Ray tests: assert deferred stage-/chunk-global drain keeps healthy throughput (no
  head-of-line stall), re-places onto a different replica, and produces output +
  lineage identical to the inline path (reordering-invariance property test).
