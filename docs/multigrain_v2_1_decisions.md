# Multigrain V2.1 Architecture Decisions

Status: historical discussion checkpoint.

The authoritative Chinese V2.2 development documents are now:

- `multigrain_v2_2_architecture.md`;
- `multigrain_v2_2_implementation_plan.md`;
- `multigrain_v2_2_paper_and_experiments.md`.

This file preserves the reasoning trail only. Its unresolved list and any
earlier contract are non-normative.

This document records the decisions confirmed after the first V2 design review.
Where it conflicts with `multigrain_v2_architecture.md` or
`multigrain_v2_implementation_plan.md`, the confirmed V2.1 decision in this
document takes precedence. It is not yet the final implementation
specification: unresolved items are listed at the end and must be closed before
switching to implementation.

## 1. Design objective

V2.1 continues to use a clean-break package:

```text
rayorch/experimental/multigrain_v2/
```

The V1 package remains unchanged as an archive and experiment baseline:

```text
rayorch/experimental/multigrain/
```

The design priorities are:

- one authority for identity, provenance, grouping, and failure semantics;
- no compatibility layer between V1 PortBatch and V2 objects;
- no duplicate Local/Ray engines;
- no speculative plugin, storage, lease, or serialization frameworks;
- ordinary users write UDFs inside the five semantic primitives;
- Ray actor batches are physical performance containers, not semantic scopes.

## 2. One internal CompiledGraph, not a public IR stack

A static graph is required so the engine can validate the DAG, create one
replica pool per node, and reuse it across microbatches. V2.1 does not keep
separate authored, verified, and compiled graph object families.

```python
@dataclass(frozen=True, slots=True)
class CompiledGraph:
    inputs: tuple[CompiledInput, ...]
    nodes: tuple[CompiledNode, ...]       # topological order
    outputs: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class CompiledNode:
    id: NodeId
    operation: Operation
    input_ports: tuple[PortId, ...]
    output_ports: tuple[CompiledPort, ...]
    replicas: int
    batch_size: int
    recovery: RecoveryPolicy


@dataclass(frozen=True, slots=True)
class CompiledPort:
    id: PortId
    domain: DomainId
    grain: str
    relation: OutputRelation
```

Pipeline tracing uses an ephemeral GraphBuilder:

```text
Pipeline DSL
  -> temporary GraphBuilder
  -> lower authoring macros
  -> validate
  -> assign NodeId/PortId/DomainId
  -> one immutable CompiledGraph
```

The builder is discarded after compilation. Graph internals are not a public
advanced API, and the module should be named `graph`/`compile` rather than
presented as a standalone IR language.

## 3. Five semantic primitives and Select

The semantic core remains:

```text
Map / Filter / Expand / Reduce / Relate
```

`Select` is retained because annotate-then-filter is common in ML curation:

```text
input
  -> Map(mask + annotations)
  -> Filter the input and annotations by mask
```

Select is only an authoring macro. It must completely lower before graph
validation and creates no SelectOp, Select handler, capability, provenance
table, or recovery path.

## 4. Source identity and admission order

Compiled input groups retain one key mode:

```python
PositionKeys | ProvidedKeys
```

PositionKeys:

- identity is derived from the caller's admission logical ordinal;
- all ports in the same input group must have equal length and ordinal
  alignment;
- changing the caller's logical order intentionally changes identity;
- Ray shard/replica/RPC/completion order does not.

ProvidedKeys:

- keys must be canonical-encodable and unique within a port;
- ports in the same input group must have exactly the same key set;
- admission aligns and orders them by canonical encoded key bytes;
- identity comes from the provided key, never the input position.

Different input groups do not align merely because their positions match.

## 5. WorkUnit coordinates

`InvocationRef` already owns batch and node:

```python
@dataclass(frozen=True, slots=True)
class InvocationRef:
    batch: BatchId
    node: NodeId
```

Work coordinates therefore store only the minimal semantic discriminator:

```python
WorkCoordinate = (
    RowCoordinate(entity: EntityId)
    | FiberCoordinate(anchor: EntityId)
    | JoinKeyCoordinate(key: CanonicalValue)
    | WholeInvocationCoordinate()
)


@dataclass(frozen=True, slots=True)
class WorkUnitRef:
    invocation: InvocationRef
    coordinate: WorkCoordinate
```

Confirmed invariants:

- node metadata determines the primary/anchor port and DomainId;
- Row/Fiber coordinates do not repeat BatchId, DomainId, or PortId;
- JoinKey uses canonical semantic key bytes, never physical partition ID;
- equality is structural;
- WorkUnit identity is independent of shard, actor, RPC batch, and retry.

There is no opaque WorkUnit ID or WorkUnit registry.

## 6. Batched value-column UDF ABI

Ray calls UDFs with batches of value columns. UDFs never receive RowRef,
IdentityKey, WorkUnitRef, or provenance.

Map:

```python
run(col_a, col_b, ...) -> output_column(s)
```

Every output has the input batch length and shares one EntityId vector.

Filter predicate:

```python
run(target_a, target_b, ...) -> bool_mask
```

Filter by an existing mask:

```python
filter = Filter.by_mask()
filtered_a, filtered_b = filter(target_a, target_b, mask=mask_port)
```

Filter rules:

- all positional inputs are filtered target ports;
- one mask is applied to every target;
- outputs have the same number/order as targets;
- every output relation is SubsetOf its corresponding target;
- target ports must be aligned by IdentityKey;
- mask is a consumed control port and is not an output;
- false is a successful zero-cardinality result.

Expand:

```python
run(columns...) -> nested output column(s)
```

The outer dimension is the parent batch. For each parent, all output ports must
have equal child count and the same child EntityId vector.

Reduce:

```python
run(
    anchor_values,
    grouped_a,
    grouped_b,
    ...,
) -> output column(s)
```

The outer dimension is the batched anchor/fiber count. Every output aligns
one-to-one with anchors.

Key Relate:

- the engine forms unique ordered parent tuples;
- UDF role columns and output columns align with those tuples;
- all outputs share the same tuple set and EntityId vector.

Custom Relate:

- receives independently sized role columns;
- returns parent indexes plus aligned output columns;
- duplicate ordered parent tuples are contract violations.

One additional hard contract is sufficient:

> An RPC batch is a performance container. A WorkUnit's business output must
> not depend on other WorkUnits placed in the same actor batch.

No deterministic/nondeterministic/batch-sensitive class hierarchy or public
exact flag is introduced.

## 7. Diamond merge and ordering

Diamond/fan-in alignment is based on IdentityKey, not physical row position.

For aligned ports A and B:

```python
A.domain == B.domain
set(A.entities) == set(B.entities)
```

The engine gathers all input columns in one authoritative ordered
`WorkUnitRef` sequence. It must never directly zip physical PortData columns.

Example:

```text
A physical order: item-1, item-2, item-3
B physical order: item-3, item-1, item-2

actor batch order: item-2, item-1
A values: A[item-2], A[item-1]
B values: B[item-2], B[item-1]
```

Each UDF position therefore still represents the same item. Output identity is
derived from the primary input, while parent-edge projection retains every
aligned branch representation.

Ordering is relation-specific rather than a mandatory global total order:

- Position source has admission ordinal;
- SameAs/SubsetOf preserve available source order evidence;
- Expand children carry sibling ordinals;
- Reduce can recover ownership order for ordered descendants;
- Relate is an unordered parent-tuple set by default;
- ByRole groups are unordered unless a future explicit order key is introduced.

The semantic guarantee is:

- identity-keyed relation equivalence for every port;
- ordered equality only when explicit order evidence exists.

Physical RowId/column order is not used for branch merge or cross-execution
semantic comparison. Trace comparison resolves RowRefs to identities and typed
edges first.

## 8. Sparse failure propagation

Fail-fast remains the safe default. Fail-closed is explicit.

Only a UDF's explicit `BadRecordError` may become a sparse direct data failure.
Generic UDF exceptions, backend failures, and contract violations never
silently become missing data.

Final sparse failure:

```python
@dataclass(frozen=True, slots=True)
class TerminalFailure:
    work_unit: WorkUnitRef
    outcome: PermanentlyMissing | Suppressed
    code: FailureCode
    message: str
    causes: tuple[WorkUnitRef, ...]
```

Invariants:

- PermanentlyMissing is a direct explicit bad-record result and has no causes;
- Suppressed has one or more upstream causes;
- one WorkUnit has at most one TerminalFailure;
- missing data without a TerminalFailure explanation is a contract violation.

Propagation:

- Map/Filter align by EntityId and suppress only the affected downstream row;
- Expand records the failed parent Row WorkUnit and never invents unknown
  child rows;
- a legal empty Expand has no TerminalFailure;
- Reduce suppresses an anchor fiber when required upstream members failed;
- an upstream Relate input whose join key cannot be obtained conservatively
  suppresses the whole Relate invocation;
- once key partitions are known, a direct relation failure may be localized to
  its JoinKey WorkUnit;
- custom Relate conservatively suppresses the whole invocation.

Fail-fast aborts before sparse propagation. In fail-closed mode:

- explicit BadRecordError may produce PermanentlyMissing;
- downstream causal absence becomes Suppressed;
- generic UDF failure after retry/isolation aborts;
- infrastructure retry exhaustion aborts;
- driver contract violation aborts immediately and is never retried.

## 9. Actor batch wrapper and stale-result fencing

V2.1 uses one token for an entire all-or-none actor RPC. It does not carry a
generation per WorkUnit.

```python
@dataclass(frozen=True, slots=True)
class BatchAttempt:
    token: BatchAttemptToken
    node: NodeId
    work_units: tuple[WorkUnitRef, ...]
    payload: ShardPayload
    deadline_ns: int | None


@dataclass(frozen=True, slots=True)
class RawBatchManifest:
    token: BatchAttemptToken
    value_refs: tuple[ObjectRef, ...]
    offsets_by_work_unit: tuple[int, ...]
    entities: tuple[EntityId, ...]
    evidence: RawEvidence
```

Coordinator state:

```text
READY
IN_FLIGHT(batch token)
ACCEPTED
TERMINAL
```

Rules:

- a WorkUnit belongs to at most one active batch token;
- v2.1 does not run concurrent hedged duplicates;
- WorkerFailure/InfrastructureFailure has no trusted partial outputs;
- isolation creates new batch tokens for retried/split WorkUnits;
- timeout/cancel first marks the token stale;
- a late stale manifest is discarded as a whole;
- a batch is accepted only after every output, offset, cardinality, and evidence
  check succeeds;
- task tokens and retry counters never enter semantic identity/provenance.

## 10. Ray-only execution

Ray is a mandatory dependency. V2.1 has no LocalExecutor and no multi-backend
plugin abstraction.

Pure compiler/runtime functions are unit-tested directly. DSL end-to-end tests
always use a single-node or cluster Ray executor.

There is one public executor:

```python
with mg.Executor() as executor:
    result = executor.run(pipeline, inputs)

with mg.Executor() as executor:
    for result in executor.run_stream(
        pipeline,
        microbatches,
        max_inflight=8,
    ):
        consume(result)
```

- Pipeline only authors/compiles a graph;
- Executor owns actor/session/cancellation lifecycle;
- RunResult is detached and has no close requirement;
- v2.1 initially exposes a synchronous iterator API only.

Failure behavior:

- fail-fast errors are raised by run or stream iteration;
- fail-closed completion returns `RunResult.failures`.

## 11. RayModule-style NodeRuntime and P2P handles

The current preferred execution topology follows RayOrch's working pattern:

```text
Driver coordinator
  -> one driver-side NodeRuntime per CompiledNode
       -> a tuple of persistent replica actor handles
```

NodeRuntime is an ordinary driver Python object, not a manager actor.

Payload flow:

1. an upstream replica writes one actor-batch payload to Ray object storage;
2. it returns a small RawBatchManifest containing ObjectRefs and semantic
   metadata;
3. the driver gets only that small manifest;
4. the driver aligns/sorts handles and metadata by identity/provenance;
5. downstream replicas receive ObjectRefs;
6. Ray transfers payloads between object store/raylets and downstream actors;
7. the driver never `ray.get`s large payload values.

Accepted payload refs are retained by ordinary coordinator/RunArena Python
references. Unaccepted/stale batch refs are dropped. After all graph consumers
finish, refs are released. Final refs move into detached RunResult and are
released when the result is discarded.

No custom ownership transaction, ArenaLease, chunk-lease registry, or manager
actor is introduced.

Driver scheduling uses one synchronous `ray.wait` state machine:

```text
submit nonblocking actor calls
ray.wait for ready small manifests
ray.get only ready small manifests
update state and dispatch more work
```

Python asyncio and an async public API are not introduced. Ray actor execution
and P2P object movement remain asynchronous.

The first implementation should reuse per-node replica-pool patterns from the
existing RayModule/RuntimeDagExecutor where they do not import V1 semantic
metadata.

## 12. Initial batching boundary

The earlier proposal for a cross-microbatch round-robin dynamic StageBatcher
was not confirmed.

The current preferred simpler direction is:

- each NodeRuntime owns persistent replicas;
- one node invocation is divided into actor batches using semantic WorkUnits
  and `batch_size`;
- replicas consume those batches as they become idle;
- multiple microbatches overlap at the pipeline/node level through
  `max_inflight`;
- downstream receives ObjectRef handles after the upstream node invocation
  settles and publishes;
- cross-microbatch co-batching into one actor RPC may be added later if
  experiments show under-filled GPU batches.

This keeps the public and semantic data model compatible with a future dynamic
queue without requiring it in the first rewrite.

## 13. Memory and data-lifetime boundary

`max_inflight` is a count bound, not a byte-memory guarantee.

V2.1 initially guarantees:

- at most max_inflight active/ready microbatches;
- target WorkUnit count per actor batch;
- no driver gathering of large payloads;
- ordinary Ray ObjectRef lifetime and release;
- an optional backend hard safety limit may reject an oversized returned
  batch.

It does not initially provide:

- byte-perfect admission control;
- max queued/staged/object-store byte contracts;
- a custom spill/ownership subsystem;
- per-primitive cardinality budget APIs.

Full byte budgets and distributed exchange/spill are paper-scale data-plane
work informed by measurement, not semantic-core prerequisites.

## 14. No durable graph/trace file format initially

V2.1 does not initially provide:

- graph.save/load;
- durable trace save/load;
- checkpoint/resume;
- a public pickle artifact format.

Ray's trusted same-version Python transport is sufficient for normal
execution.

When experiment events/metrics/provenance are explicitly exported, each export
receives its own `rayorch.multigrain_v2.*` schema namespace and version. That
does not imply a durable pipeline graph format.

## 15. Observability default

Normal execution writes no files:

- metrics may be reduced in memory;
- standard logging emits compact run/status summaries;
- errors project sparse WorkUnit cause traces;
- complete events/provenance are explicit experiment/debug outputs;
- source payloads are not silently checkpointed.

Run closure ends active-run replay availability. A later rerun requires source
inputs to be supplied again.

## 16. Paper claim after V2.1 simplification

The defensible mechanism story is currently:

> Typed relation semantics derive row, fiber, join-key, and whole-invocation
> WorkUnits that drive actor-batch formation, failure isolation, replay closure,
> identity alignment, and atomic node publication.

The first implementation does not claim:

- fragment-level downstream streaming;
- post-publication selective downstream replay;
- cross-microbatch co-batching inside one actor RPC;
- byte-bounded execution;
- driver/node failure recovery;
- payload equality for nondeterministic UDFs.

`max_inflight` supplies inter-microbatch pipeline overlap. Cross-microbatch
dynamic batching remains a later optimization/ablation candidate.

## 17. Confirmed decisions checklist

- [x] Minimal structural WorkUnit coordinates.
- [x] PositionKeys and ProvidedKeys admission rules.
- [x] Select retained only as an authoring macro.
- [x] Batched value-column UDF ABI.
- [x] Filter maps N aligned target ports to N subset outputs.
- [x] Conservative sparse failure propagation.
- [x] Diamond merge uses explicit IdentityKey join.
- [x] Ordering is relation-specific, not a global semantic total order.
- [x] Actor RPC uses one all-or-none batch token.
- [x] Only explicit BadRecordError may become fail-closed missing data.
- [x] RPC batch peers are not part of WorkUnit semantics.
- [x] Ray ObjectRef/Python references provide the initial data lifetime.
- [x] max_inflight is count-bounded, not byte-bounded.
- [x] Pipeline is passive; Executor is the only execution entry.
- [x] Ray is mandatory; there is no LocalExecutor.
- [x] No durable graph/trace file format initially.
- [x] One internal CompiledGraph replaces a public/multi-stage IR stack.
- [x] Driver uses synchronous ray.wait, not asyncio.

## 18. Unresolved QA before implementation

The following items were not yet confirmed:

1. Long-tail/isolation details:
   - finite retry count;
   - binary split depth;
   - optional batch timeout;
   - ray.cancel versus actor kill/replacement;
   - retry queue priority.
2. Final confirmation of the NodeRuntime first implementation:
   - per-invocation actor batching;
   - no cross-microbatch dynamic batching initially.
3. Resource startup policy:
   - eager all-node replica creation versus stage/residency-group creation;
   - failure rollback.
4. Exact public `RecoveryPolicy`, `RunResult`, `ResultPort`, and projected error
   fields.
5. Whether primitive/backend plugin registries are explicitly forbidden in the
   contributor contract.
6. Objective test thresholds for fairness, timeout, isolation work
   amplification, and long-running memory behavior.
7. Updated implementation phase order and paper experiments after removing
   LocalExecutor and initial cross-microbatch StageBatcher.

After these QA items are closed, this checkpoint must be folded into the
architecture, implementation, and paper documents. Only then should the user
be asked for confirmation to switch to Agent mode.
