# Multigrain V2 Implementation Plan

Status: superseded historical V2 roadmap.

Use `multigrain_v2_2_implementation_plan.md` for implementation. This file
retains the earlier roadmap only as design history.

Normative semantics are defined in
[`multigrain_v2_architecture.md`](multigrain_v2_architecture.md). Research claims
and experiment gates are defined in
[`multigrain_v2_paper_and_experiments.md`](multigrain_v2_paper_and_experiments.md).

## 1. Clean-break policy

Create:

```text
rayorch/experimental/multigrain_v2/
test/experimental/multigrain_v2/
```

Do not mutate V1 while establishing V2 semantics. Do not add compatibility
adapters between `PortBatch` and V2 runtime objects.

V1 may be reused only as:

- workload/operator fixtures;
- user-visible behavioral examples explicitly listed as retained;
- baseline implementation for experiments;
- a source of counterexamples and regression tests.

V1 internal metadata layouts, executor private methods, retry tokens, pickle
artifacts, and placeholder behavior are not V2 interfaces.

## 2. Phase 0: freeze contracts

Before executor code:

- create V2 package/test skeletons;
- freeze the beginner facade and exception names;
- encode intentional semantic breaks as tests;
- add retained-semantics characterization tests for the five primitives;
- define canonical ID/value codec fixtures;
- define WorkUnit and OutputBundle examples in tests.

Required break tests:

- no Relate parallel parent tuples;
- no fail-closed placeholder;
- no output-slot contribution to Expand identity;
- no copied lineage dictionaries;
- no post-publication mutation;
- no implicit file artifacts.

Exit gate: tests describe every retained or intentionally changed V1 behavior.

## 3. Phase 1: passive IR and compiler

Implement:

- NodeId/PortId/DomainId/IdentityGroupId;
- GraphInput, NodeSpec, OutputSpec, relations, and operations;
- CompiledInputGroup and CompiledPort;
- deterministic domain propagation;
- sealed operation capabilities;
- structural graph validation;
- graph v1 serialization.

Compiler checks:

- source identity groups and key modes;
- relation/operation compatibility;
- multi-output Expand shared domain;
- selector/role validity;
- Relate parent-tuple-set contract;
- backend capability compatibility.

Exit gate: graph compilation and serialization need no runtime/Ray imports.

## 4. Phase 2: semantic runtime and Local vertical slice

Implement:

- RunArena;
- RowRef, IdentityKey, EntityId codec;
- PortData and direct typed provenance tables;
- parent-edge projection and sealed queries;
- InvocationRef and WorkUnitRef;
- OutputBundle and TerminalFailure;
- atomic arena publication.

Complete one end-to-end Source → Map Local slice:

```text
admit inputs
  -> compile Row work
  -> execute Local batch
  -> materialize EntityIds/provenance
  -> canonicalize RowIds
  -> publish bundle
  -> detach RunResult
```

Exit gate:

- physical input order does not affect identity/order;
- bundle publication is all-or-none;
- detached results contain no live arena references;
- empty and failed outputs are distinguishable.

## 5. Phase 3: complete Local semantics

Add primitives in this order:

1. Filter/Select;
2. Expand;
3. Reduce;
4. Relate.

For each primitive, implement together:

- WorkUnit construction;
- ShardPayload layout;
- raw result validation;
- EntityId derivation;
- typed provenance materialization;
- canonical ordering;
- replay closure;
- terminal-failure behavior;
- property/reference-model tests.

Required cases:

- multi-input Map diamonds;
- zero-output Filter;
- multi-output Expand and nested ordinal ordering;
- zero-child Expand;
- empty, ready, and suppressed Reduce groups;
- nested Expand → Reduce;
- M:N key Relate with repeated join keys;
- duplicate Relate parent-tuple rejection;
- custom whole-invocation Relate;
- multiple upstream failure causes.

Exit gate: all Local primitive/reference tests pass without Ray installed.

## 6. Phase 4: coordinator and batched scheduling

Implement:

- AdmissionWindow(max_inflight);
- one StageRuntime per compiled node;
- fair cross-microbatch ready queues;
- StageBatcher with unit/byte/wait bounds;
- idle replica checkout;
- ShardTask/RawShardBatch protocol;
- first-accepted WorkUnit fences;
- retry/split/isolation budgets;
- invocation settlement and canonical publication.

Use Local prepared replicas first so scheduler tests do not require Ray.

Required tests:

- active RPCs never exceed replicas;
- active/ready microbatches never exceed max_inflight;
- compatible work from multiple microbatches co-batches;
- ordered result delivery preserves backpressure;
- every replica slot is released exactly once;
- batched failure requeues only unaccepted WorkUnits;
- long-tail work does not starve unrelated microbatches;
- bundle publication waits for all success/terminal outcomes.

Exit gate: Local scheduler behavior is deterministic under randomized completion
order.

## 7. Phase 5: Ray backend

Implement Ray only through the shared execution protocol:

- transactional resource planning/reservation;
- per-node prepared actor replicas;
- one active batched RPC per replica;
- compact RawShardBatch manifests;
- remote immutable value-column handles;
- timeout/cancel/kill/actor replacement;
- stale-result rejection;
- backend conformance events and metrics.

Do not:

- queue hidden work in actor mailboxes;
- return one ObjectRef per WorkUnit;
- gather large payload columns into the driver;
- call Local executor private methods;
- reconstruct identity/provenance in workers.

Required fault tests:

- actor death before/after worker execution;
- transport failure;
- timeout and replacement;
- late result after cancellation;
- batch split isolation;
- nondeterministic first-accepted publication;
- multi-output atomicity.

Exit gate: Local/Ray agree on values, entities, normalized provenance,
TerminalFailures, and canonical order for deterministic fixtures.

## 8. Phase 6: distributed data plane and exchange

This phase is mandatory for paper-scale claims even though logical publication
remains node-level atomic.

Implement:

- chunked immutable value columns;
- partitioned provenance columns/CSR;
- object-store/spill ownership;
- compact arena and bundle manifests;
- byte-level backpressure;
- distributed ByAnchor exchange;
- distributed ByJoinKey exchange;
- partition-completeness validation;
- M:N cardinality guards.

The driver holds manifests and compact semantic metadata, not payload copies.
Physical chunks are not public fragment-streaming semantics.

Exit gate:

- Reduce/Relate execute without driver payload gather;
- staged data survives actor replacement;
- memory remains bounded in long steady-state runs;
- bundle publication atomically exposes complete manifests.

## 9. Phase 7: observability and artifacts

Keep normal execution no-file by default.

Implement:

- streaming in-memory metric reduction;
- lightweight standard logging;
- sampled terminal-error rendering;
- explicit experiment event sink;
- explicit anonymized provenance/trace export;
- graph/event/trace schema namespaces;
- actual GPU/network/object-store/process-memory sampling.

Exit gate: metrics names correspond to measured quantities and experiment
tracing can be disabled without changing execution semantics.

## 10. Phase 8: experiments

Implement the workloads, baselines, ablations, and fault injection specified in
`multigrain_v2_paper_and_experiments.md`.

Run in increasing scale:

1. deterministic local correctness;
2. single-node Ray functional/fault tests;
3. single-node long steady-state memory tests;
4. multi-node synthetic scaling;
5. document and multimodal workloads;
6. full baseline/ablation matrix.

Do not treat an earlier V1 performance result as V2 mechanism evidence.

## 11. Global acceptance gates

V2 is functionally ready when:

- every semantic fact has one owner;
- no V1 runtime object crosses the V2 package boundary;
- all five primitives pass Local/Ray conformance;
- retries never change semantic coordinates;
- node outputs are canonical and atomically published;
- nondeterministic tests assert structure/completeness rather than equality;
- no default run creates files;
- no large payload is gathered by the driver.

V2 is paper-ready only after it also satisfies the scale, baseline, fault, and
artifact gates in the paper document.
