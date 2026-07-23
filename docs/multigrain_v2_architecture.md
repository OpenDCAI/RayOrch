# Multigrain V2 Architecture

Status: superseded historical V2 design.

The authoritative development architecture is now
`multigrain_v2_2_architecture.md`. This file is retained only to explain the
earlier, more complex design and must not be used as an implementation
contract.

This document supersedes the earlier `PortBatch` convergence proposals. V2 is a
separate experimental implementation; it does not preserve V1 runtime objects,
pickle artifacts, placeholder recovery behavior, or internal extension points.

## 1. Goals

- Give identity, provenance, grouping, ordering, and failure localization one
  semantic authority.
- Compile the five cardinality-changing primitives into explicit semantic work
  units.
- Batch compatible work units into efficient Local/Ray calls without making RPC
  boundaries part of logical semantics.
- Retry or isolate failed work units before atomically publishing a node result.
- Keep the beginner API focused on primitives, replicas, batch size, and
  pipeline inflight.

V2 does not initially provide fragment-level downstream streaming,
post-publication repair, checkpoint/resume, durable provenance, or arbitrary
effectful UDF semantics.

## 2. Package boundary

```text
rayorch/experimental/multigrain_v2/
├── api.py
├── authoring.py
├── errors.py
├── ir/
│   ├── model.py
│   ├── compile.py
│   └── serde.py
├── runtime/
│   ├── model.py
│   ├── provenance.py
│   └── materialize.py
├── execution/
│   ├── protocol.py
│   ├── coordinator.py
│   ├── local.py
│   ├── events.py
│   └── metrics.py
└── backends/ray/
    ├── executor.py
    └── worker.py
```

Dependency direction:

```text
authoring -> IR
runtime   -> IR
execution -> IR + runtime
Ray       -> execution protocol
metrics   -> execution events
```

IR/runtime/execution must not import Ray. Ray workers produce raw values and
evidence; they never create identity or provenance.

## 3. Public API

The beginner facade remains small:

```python
from rayorch.experimental import multigrain_v2 as mg

self.ocr = mg.Map(
    Ocr,
    replicas=4,
    batch_size=8,
    gpus_per_replica=1,
)

results = pipeline.run_stream(
    inputs,
    backend=ray_backend,
    max_inflight=8,
)
```

- `replicas` is per graph node and limits concurrent prepared execution slots.
- `batch_size` targets the number of compatible work units in one UDF/RPC call.
- `max_inflight` belongs to one stream run and bounds active microbatches.
- Increasing inflight does not create replicas; increasing replicas does not
  enlarge the admission window.

Queue sizes, permits, flush timers, placement groups, arena objects, provenance
tables, and task layouts are internal.

## 4. Compiled identity model

There is no `IdentityFamilyId` and no persistent `DomainPlan`.

```python
@dataclass(frozen=True, slots=True)
class CompiledInputGroup:
    id: IdentityGroupId
    domain: DomainId
    key_mode: PositionKeys | ProvidedKeys


@dataclass(frozen=True, slots=True)
class CompiledPort:
    id: PortId
    grain: str
    domain: DomainId
    relation: OutputRelation
```

The compiler assigns domains once:

- Inputs in the same identity group share a domain and key mode.
- `SameAs` and `SubsetOf` inherit the source domain.
- `AggregateOf` inherits the anchor domain.
- All aligned outputs of one `Expand` share one fresh derived domain.
- All outputs of one `Relate` share one fresh relation domain.
- Independent derivation nodes never accidentally share a domain.
- Grain names are labels, not identity namespaces.

Runtime references:

```python
@dataclass(frozen=True, slots=True)
class RowRef:
    batch: BatchId
    port: PortId
    row: RowId


@dataclass(frozen=True, slots=True)
class IdentityKey:
    batch: BatchId
    domain: DomainId
    entity: EntityId
```

`RowRef` identifies a concrete representation. `IdentityKey` identifies a
logical entity. Same-identity rows on different ports have different RowRefs.

## 5. PortData and RunArena

Batch and domain are stored once by the arena and compiled port. A port stores
only the varying entity column:

```python
@dataclass(frozen=True, slots=True)
class PortData:
    port: PortId
    entities: tuple[EntityId, ...]
    values: ImmutableValueColumn
    provenance: ProvenanceTable
```

`RowId` is the canonical base-column index assigned after a node invocation has
settled. Physical shard offsets and completion order never enter RowId.

Each active microbatch owns one `RunArena` containing:

- admitted input PortData;
- atomically published node bundles;
- rebuildable identity/representation/query indexes;
- staged raw results needed until the current invocation settles.

There is no separate provenance, diagnostic, disposition, or lease registry.
Indexes are caches, not semantic facts.

Port validation must enforce:

- value/entity/provenance lengths are equal;
- EntityIds are unique within a port;
- every referenced RowRef is valid and belongs to the same BatchId;
- provenance table type matches the compiled relation;
- selectors and roles match compiled order;
- canonical ordering is independent of Python container and RPC completion
  order.

## 6. Typed provenance

Each PortData directly owns exactly one immutable typed table:

```python
ProvenanceTable = (
    SourceRows
    | AliasRows
    | ChildRows
    | AggregateRows
    | RelatedRows
)
```

- `SourceRows`: admitted entities.
- `AliasRows`: exact source RowRefs for Map/Filter representations.
- `ChildRows`: one parent RowRef and child ordinal per Expand row.
- `AggregateRows`: anchor plus canonical selector-member CSR.
- `RelatedRows`: one direct parent column per declared role.

Semantically identical multi-output ports may share one immutable table object.
Transitive paths and ancestor dictionaries are never copied into every row.

`runtime/provenance.py` exposes one structural projection:

```python
parent_edges(row: RowRef) -> tuple[ParentEdge, ...]
```

Edge kinds are representation, ownership, anchor, member, role, and side-input.
Sealed query policies implement:

- ancestor resolution;
- dependency trace;
- direct role lookup;
- aggregate member lookup.

Ancestor resolution preserves `NoAncestor`, `UniqueAncestor`, and
`ConflictingAncestors`. No consumer may silently choose one identity from a
conflict.

## 7. Primitive contracts

### Map

- Inputs are aligned by IdentityKey, not physical offsets.
- Input 0 determines output identity.
- All aligned inputs remain visible in representation/dependency trace.
- Each successful work unit produces exactly one row per output port.

### Filter and Select

- A boolean decision is aligned with the input entity.
- Survivors preserve identity.
- A false decision is a successful zero-cardinality result, not a failure.

### Expand

- One work unit is one parent entity.
- The UDF returns one child list per parent and per output.
- For one parent, all outputs have equal child counts.
- Aligned outputs share the compiled derived domain and identical EntityId
  vectors.
- Child identity uses parent identity and child ordinal; output slot is not
  part of identity.

### Reduce

Grouping produces an ephemeral readiness result:

```python
ReadyGroup(anchor, members) | SuppressedGroup(anchor, causes)
```

- A complete empty group is valid.
- A group made incomplete by recorded upstream failure is suppressed.
- Missing data without an upstream terminal failure is a contract violation.
- Each ready anchor produces exactly one row per output port and reuses anchor
  identity.
- Suppressed anchors produce no placeholder payload.

### Relate

- Built-in key relation work units are complete join-key partitions.
- Custom relation defaults to whole-invocation work.
- Each output has exactly one parent per declared role.
- The relation is a set of ordered parent tuples.
- Duplicate parent tuples are contract violations; V2 has no stable key,
  discriminator, or parallel relation edges.
- Relation identity is derived from the ordered role-parent identities.

## 8. Invocation and work-unit coordinates

```python
@dataclass(frozen=True, slots=True)
class InvocationRef:
    batch: BatchId
    node: NodeId


WorkCoordinate = (
    RowCoordinate
    | FiberCoordinate
    | RelationPartitionCoordinate
    | WholeInvocationCoordinate
)


@dataclass(frozen=True, slots=True)
class WorkUnitRef:
    invocation: InvocationRef
    coordinate: WorkCoordinate
```

There is no opaque WorkUnit ID or WorkUnit registry.

One physical shard may contain many WorkUnitRefs. One WorkUnitRef may be
attempted multiple times. Shard/attempt identifiers are operational only and
never enter identity or provenance.

## 9. Task layout

Task payloads normalize values separately from their access shape:

```python
@dataclass(frozen=True, slots=True)
class ShardPayload:
    slices: tuple[PortSlice, ...]
    layout: WorkLayout


WorkLayout = RowAligned | FiberGrouped | RelationClosure
```

- RowAligned references aligned input slices.
- FiberGrouped references anchors plus selector CSR indexes.
- RelationClosure references independently sized role slices and either a
  complete key partition or whole invocation.

Backends transport layouts but do not reinterpret them.

## 10. Stage scheduling and batched RPC

V2 uses a node-invocation publication barrier. Work units execute independently
inside a node, but downstream nodes read only the final atomic bundle.

```text
WorkUnits
  -> per-node fair ready queue
  -> StageBatcher
  -> idle prepared replica
  -> one batched RPC
  -> raw result staging
  -> all WorkUnits settled
  -> canonicalize
  -> atomic node publication
```

`StageBatcher` flushes when one of these limits is reached:

- target work-unit count;
- estimated input bytes;
- maximum wait of the oldest ready unit.

Compatible work units may be co-batched across admitted microbatches of the
same graph node. Logical readiness never implies one RPC per WorkUnit.

Ray returns one `RawShardBatch` with columnar outputs, per-work-unit offsets,
and compact failure metadata. It must not return one object per row/work unit.
The driver should retain remote value-column handles instead of gathering and
repacking large payloads.

## 11. Replica ownership and resources

Each graph node has one `StageRuntime`:

```text
StageRuntime
  compiled node
  fair ready queue
  StageBatcher
  idle prepared replicas
```

- Checking out an idle replica is the sole capacity authority.
- One prepared replica executes at most one batched RPC at a time.
- Actor mailbox backlog is not accepted as scheduler state.
- Success, failure, cancellation, and replacement release a slot exactly once.
- Retry backoff does not occupy a replica.

Ray resource setup is transactional:

1. Build the complete CPU/GPU/memory/custom-resource plan.
2. Atomically reserve resources, for example with a placement group.
3. Prepare all node replicas inside the reservation.
4. Roll back the whole reservation if preparation fails.

## 12. Attempts, replay, and long tails

Attempt outcomes are deliberately small:

```python
Success(raw_batch)
WorkerFailure(error)
InfrastructureFailure(error)
```

A successfully returned raw result is still checked by the driver materializer.
Length, role, closure, identity, or cardinality violations are terminal
`ContractViolation`s and are never retried as actor faults.

V2 replay exists only before node publication:

```python
replay_closure(work_unit: WorkUnitRef) -> ShardPayload
```

- Row work reuses all aligned inputs.
- Fiber work reuses the complete anchor/member closure.
- Key relation work reuses the complete role partition.
- Custom relation work reuses the whole invocation.

Each WorkUnit has an ephemeral first-accepted fence. Late results are ignored.
Worker/infrastructure failures are retried within finite budgets and may be
split to isolate a failing unit. A still-running long tail requires a real
timeout, cancellation/kill, stale-result fence, and actor replacement; a retry
counter alone is not a bound.

Published bundles are immutable. Recovery after publication, checkpoint
recovery, and downstream patching are outside V2's initial fault model.

## 13. Terminal failure and atomic publication

Final failure information uses WorkUnitRef as its only locator:

```python
@dataclass(frozen=True, slots=True)
class TerminalFailure:
    work_unit: WorkUnitRef
    outcome: PermanentlyMissing | Suppressed
    code: FailureCode
    message: str
    causes: tuple[WorkUnitRef, ...]
```

There is at most one TerminalFailure per WorkUnitRef. Causes reference unique
upstream terminal failures and therefore follow graph topology.

```python
@dataclass(frozen=True, slots=True)
class OutputBundle:
    invocation: InvocationRef
    ports: tuple[PortData, ...]
    failures: tuple[TerminalFailure, ...]
```

The bundle contains every compiled output port in compiled order, including
empty ports. Successful zero-cardinality Filter/Expand/Relate work has no
failure record. Failed units produce sparse outputs plus terminal failures.

`RunArena.publish(bundle)` validates and installs all ports/failures under one
lock. An invocation publishes at most once. There is no bundle ID,
candidate/manifest pair, bundle state machine, semantic digest, or mutable
post-publication disposition.

## 14. Nondeterministic UDFs

V2 does not add an `exact=True/False` beginner option.

All automatically replayed UDFs must be free of external side effects. Within
an active run, V2 guarantees that the original WorkUnit closure remains
available for another attempt and that only one accepted result is published.

Payload/cardinality equality across attempts or runs is only claimed for
deterministic, locality-admissible UDFs. For VLMs and other nondeterministic
operators, V2 guarantees atomic structural validity and rerunnability, not
equal output.

A true cross-row semantic dependency is not a row-local Map and must use an
appropriate fiber/relation/whole-invocation contract.

## 15. Result and observability lifecycle

RunArena is retained only while its microbatch is active. On terminal success:

1. Final ports are converted into detached result columns.
2. Reachable terminal-failure cause traces are projected.
3. A detached `RunResult` is returned.
4. Intermediate arena data is released.

RunResult does not expose live RowRefs or lazy full-provenance queries. There is
no ArenaLease, close/context-manager protocol, or descriptor reacquire API.

Normal execution writes no files by default:

- metrics are reduced in memory;
- standard logging emits only small run/status summaries and sampled errors;
- complete events/provenance are exported only through explicit
  experiment/debug configuration;
- payload and durable source checkpoints are never silently written.

## 16. Required invariants

1. Domain assignment is compiled once and read uniformly by runtime/execution.
2. Every PortData has equal value/entity/provenance cardinality.
3. Every RowRef resolves within the current arena.
4. SameAs/SubsetOf/AggregateOf preserve source/source/anchor identity.
5. Expand outputs share domain and EntityId vectors.
6. Relate parent tuples are unique.
7. Physical shard, replica, attempt, and completion order never enter identity.
8. Work layouts are validated against operation capability.
9. A stage has at most `replicas` active RPCs.
10. A run has at most `max_inflight` active/ready microbatches.
11. Every WorkUnit is either successfully staged or has one terminal failure
    before publication.
12. Every invocation publishes at most one complete OutputBundle.
13. Downstream execution observes only published bundles.
14. Backend code never constructs identity or provenance.

## 17. Intentional V1 breaks

- No PortBatch parallel metadata columns.
- No runtime IdentityDomain object or name-derived implicit independent roots.
- No IdentityFamilyId.
- No copied transitive lineage/ancestor/ordinal dictionaries.
- No Relate stable key or same-parent parallel edges.
- No fail-closed placeholder payload.
- No mutable deferred-token rekeying.
- No actor-mailbox capacity model.
- No post-publication result mutation.
- No default artifact write or lazy full-lineage retention.

Characterization tests protect only explicitly retained user semantics. New
normative tests define all intentional breaks.
