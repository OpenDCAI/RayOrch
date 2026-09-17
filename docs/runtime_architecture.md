# Runtime architecture

RayOrch exposes one small API from the package root. Its implementation has
three private layers:

```text
_program/    symbolic analysis, lowering, and verification
_runtime/    input batch state, dispatch indexes, and result materialization
_execution/  Ray actors, RPC lifecycle, and worker ABI
```

The package root exports authoring (`Pipeline`, `RayModule`, `Port`, `F.*`),
execution (`Executor`, `run`, `RunResult`), and recovery (`RecoveryPolicy`).
Internal identity and worker-protocol types stay private so their representation
can evolve without expanding the compatibility surface. The UDF-facing
`MISSING`, `RecordFailure`, and `GroupFailure` values remain available from the
package root; failure values can also be imported from `rayorch.failures`.

Structural primitives describe relationships and never create actors. Calling a
`RayModule` while tracing creates a static Call; runtime Entities give rise to
its logical Grains. Each InputBatch owns its source rows and all derived work,
managed by an `InputBatchEngine`. The runtime uses immutable coordinates,
generation-fenced reports, atomic multi-output commit, and an input batch-local
parent barrier for `GroupFailure`.

Each Call owns one completion-ordered READY queue. Committing an upstream Grain
immediately publishes its facts and may make downstream Grains dispatchable;
execution never waits for an entire stage or input domain to drain. READY
Grains from different parent entities may share an `ExecutionMicrobatch`: one
Call's Grains within one InputBatch, grouped for a Worker RPC and bounded by
the Call's `batch_size`. `input_batch_size` instead counts source rows, while
`max_active_input_batches` limits overlapping input batch lifecycles. Parent
identity remains part of runtime state only for lineage and `GroupFailure`
isolation, not as a batching policy.

## Group shape

Compiler analysis records each Port's structural group depth; it does not infer
the contents of arbitrary Python values. Source rows and ordinary UDF outputs
have depth zero. A UDF output used by `F.expand` is represented as one group of
child rows, so that output has depth one and its expanded rows have depth zero.
The current API permits Expand only on UDF outputs.

`F.reduce` adds one group layer. `F.filter` and `F.broadcast` preserve their
source value's depth. Lowering puts the input depth in `ReduceEffect.value_depth`;
the runtime uses this contract for both empty and nonempty groups.
`NestedGroupLayout` records the actual boundaries at each level and validates
nested child depths. No shared mutable stack or separate shape object is needed.

## Recovery ownership

Executor classifies Worker failures, wraps execution errors, owns actor/RPC
lifecycles, and updates dispatch counters. For an opaque UDF failure it passes
the batch, immutable RecoveryPolicy and cause to `InputBatchEngine.apply_udf_recovery`.
The Engine partitions live/barriered Grains once, calls the policy's pure
`decide_udf`, and applies the result through DispatchState and fact publication.

The entry returns the requeued Grain count, or `None` to abort without changing
state, matching the infrastructure-recovery return convention. An entirely
barriered batch is sealed and published without consulting the UDF policy.
Recovery only updates local queues and facts; Executor's existing dispatch loop
sends any later RPC. Retry budgets, binary isolation and queue priorities remain
unchanged. For K Grains, classification remains O(K) time and O(K) worst-case
temporary space; the change removes repeated work without a persistent cache.

## Incremental Reduce

Each unfinished Reduce output has one engine-owned `_ReduceProgress`: a pending
member count, the lowest failed member ordinal, and a forward-only value cursor.
It is a rebuildable index over canonical Item/Expansion facts, not another source
of outcomes. The engine initializes it once from existing facts, updates membership
at first Item publication through the compiled dependency index, and removes it
when the Reduce becomes terminal. Replayed facts never decrement the count twice.

Publication updates the summary without evaluating Reduce. The existing fact
queue still triggers evaluation, so all facts committed in a Worker batch are
visible before choosing a cause. `reduce_transition` retains the precedence:
Expansion, member failure, pending members, then survivor values in ordinal order.
The engine supplies a lazy iterator beginning at the first unchecked value; the
transition stops at a missing or unsuccessful value. Dropped members need no value.
Successful prefixes are never rescanned, and the final group is built once.

For N children, E dependency deliveries and L output layout/leaf references,
total work per Reduce group is O(E + N + L), including initialization and output
construction. One event may advance many cursor positions; this is an amortized
bound, not constant worst-case event latency. The extra persistent index is O(1)
per pending group, O(G) for G groups. Existing facts, output storage and final
construction still require O(N + L) space. Consumer fanout contributes to E.

## Broadcast traversal

A source Item event walks only that source Entity's existing Expansion subtree,
following the Domain parent path to the target Domain. A depth-first iterator
stack holds one position per level and reuses `ExpansionRecord.children`; there
is no per-target waiting table or persistent traversal index. Entity creation
events continue to publish Broadcast targets whose source facts already exist.
Worker commits publish complete Expansion and lineage facts before events run,
so these two paths cover either arrival order without missed wakeups.

For one rule in one InputBatch, let S be the source count, T the target count,
H the Domain distance and V the nodes visited by source-triggered traversal.
The Domain path is read per source event. Total work is O((S + T) × H + V),
with O(H) auxiliary traversal space; one-level Broadcast takes O(S + T) work.
Deep, wide, sparse trees can require many intermediate-node visits. Facts and
published outputs still need their existing storage, and the event queue can
grow with the number of targets released together.

One source event visits targets in depth-first child-ordinal order, replacing
target-Domain creation order for that event. READY insertion remains FIFO;
simultaneously released work may have different execution microbatch boundaries.
Waiting tables or automatic strategy selection remain deferred until concrete
user reports justify the additional mechanism. See the
[release notes draft](release-notes-draft.md) for the tradeoff and validation.

## Final output issues

`ItemOutcome` is public; Grain phases and internal coordinates remain private.
Materialization returns business values for PRESENT Items and frozen
`OutputIssue(outcome, cause)` records for the other terminal outcomes.
The engine's read-only `item_cause()` follows the existing selected Item/Expansion
provenance. Materialization converts that cause into optional text without
changing stored facts or retaining protocol objects in the public result.
This boundary adds no scheduling transitions or recovery actions.

## Execution metrics

Executor freezes its existing driver-owned counters when a run completes.
There is no end-of-run observation RPC or UDF audit hook. Worker owns the UDF
instance and batch value processing, without probing business-specific diagnostic
fields. Process/resource observation is left to Ray Dashboard, and benchmark
sampling remains outside the core execution path.
