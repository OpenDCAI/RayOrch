# Multigrain V2 Paper Positioning and Experiments

Status: superseded historical V2 research plan.

Use `multigrain_v2_2_paper_and_experiments.md` for the current research claims
and evaluation contract. This file is retained as design history.

The architecture rewrite is a prerequisite, not a paper contribution by
itself. This document defines the mechanism story that V2 must implement and
measure.

## 1. Proposed thesis

> Typed relation semantics compile cardinality-changing ML dataflows into
> semantically closed work units, enabling cross-microbatch accelerator
> batching and bounded dependency-closed recovery before atomic node
> publication.

The scope is finite, closed-microbatch DAGs built from Map, Filter, Expand,
Reduce, and Relate.

“Dependency-closed” means relative to the declared operation contract:

- row-local operations replay aligned row inputs;
- Reduce replays a complete anchor fiber;
- key Relate replays a complete join-key partition;
- opaque/global kernels replay a whole invocation.

## 2. Defensible contributions

### C1. Relation-to-work-unit compilation

Static relation contracts determine row, parent, fiber, key-partition, and
whole-invocation work boundaries. The same boundaries are used by execution,
grouping, retry, failure localization, and canonical publication.

Required evidence:

- a formal work-unit/closure definition for every primitive;
- compiler/validator rules;
- property tests over random legal DAGs, partitions, retries, and physical
  reorderings;
- a conditional compilation-soundness argument.

### C2. Relation-aware StageBatcher

The scheduler co-batches compatible semantic work units from multiple admitted
microbatches while preserving:

- per-node replica capacity;
- run-level admission backpressure;
- operation closure requirements;
- canonical node outputs;
- atomic multi-output publication.

RPC batching, persistent actors, or max inflight alone are engineering
features. The research claim requires showing that relation-derived work
boundaries make batching safer or more effective than fixed row/shard/global
batch boundaries.

### C3. Bounded pre-publication isolation

On worker or actor failure, V2:

1. retains accepted healthy work;
2. reconstructs the failed semantic closure;
3. retries or splits within explicit budgets;
4. replaces unhealthy replicas when needed;
5. publishes only after every work unit succeeds or reaches a terminal
   fail-closed outcome.

The comparison target is whole-shard or whole-microbatch retry under identical
output-completeness policy.

### C4. Canonical atomic semantics

For deterministic, locality-admissible UDFs, legal batching, replica assignment,
completion order, and retry do not change canonical logical output.

For nondeterministic UDFs, V2 claims only:

- original-closure rerunnability during the active run;
- one first-accepted result;
- valid identity/provenance/cardinality structure;
- atomic publication.

Payload equality is not claimed.

## 3. Claims explicitly excluded

V2 must not claim:

- fragment-level inter-stage streaming;
- elimination of node-invocation barriers;
- post-publication selective downstream replay;
- checkpoint/resume or driver-failure recovery;
- end-to-end exactly-once side effects;
- deterministic payload equality for VLMs or other nondeterministic UDFs;
- scalable execution of every custom Relate or non-combinable Reduce;
- durable provenance after a normal RunResult is detached;
- hot-fiber splitting before that mechanism is implemented and evaluated.

Atomic publication deliberately prevents downstream consumers from observing
unsettled node outputs. A correction discovered after publication requires a
new run; V2 does not patch an existing run.

## 4. Fault model

Initial supported faults:

- UDF exception;
- explicit bad record;
- Ray actor death;
- RPC/transport failure;
- bounded straggler timeout followed by cancellation/actor replacement.

Initial exclusions:

- driver death;
- cluster-node loss that destroys published object-store data;
- source loss;
- external-service side effects;
- durable restart from checkpoint.

Actor death recovery is valid only if unaccepted work is requeued and published
data is owned outside the failed actor. The paper must state where staged and
published values live and when they are released.

## 5. Required distributed implementation

Node-level atomic publication does not imply driver-side payload gathering.
For credible multi-node evaluation, V2 needs:

- chunked/partitioned immutable value columns;
- compact driver manifests rather than Python payload copies;
- byte-level queue/backpressure accounting;
- spill or object-store residency for large ports;
- distributed `ByAnchor` routing for Reduce;
- distributed `ByJoinKey` exchange for Relate;
- complete partition manifests;
- bounded M:N cardinality safeguards;
- timeout, kill, actor replacement, and stale-result fencing.

Custom global Relate and indivisible hot Reduce fibers may remain explicit
barriers. Their limits must be measured rather than hidden.

## 6. Workloads

Use at least two public, structurally different workloads:

### W1. Document parsing

```text
PDF
  -> Expand pages
  -> GPU layout/OCR Map
  -> optional block/evidence Relate
  -> Reduce document
```

This workload supplies nested 1:N/N:1 structure, image/token skew, GPU batching,
bad-page isolation, and document-level fail-closed behavior.

### W2. Multimodal or video curation

```text
video/image collection
  -> Expand frames/clips
  -> VLM embedding/caption Map
  -> key/content Relate
  -> Reduce asset summary
```

This workload supplies different fanout, M:N density, model behavior, and
nondeterministic output.

### W3. Synthetic relation workload

Sweep:

- Expand fanout;
- nested depth;
- fiber size and key skew;
- M:N relation density;
- row/fiber/partition service-time tails;
- failure rate and locality;
- payload size;
- nondeterministic cardinality;
- input microbatch size and inflight.

## 7. Baselines

Required tuned baselines:

- naive Ray actors with fixed batching/sharding;
- Ray Data configured for the same models and resources;
- Trident or the strongest relevant cardinality-changing ML dataflow system;
- whole-shard/microbatch retry using the same failure and completeness policy.

Spark should be included only if accelerator UDF execution and fault semantics
can be configured fairly.

All systems must use:

- identical models and model residency assumptions;
- comparable batch-size tuning budgets;
- identical input/output completeness policy;
- the same injected failures;
- the same warmup and measurement window.

A baseline that aborts while V2 suppresses a bad record is not a valid recovery
comparison.

## 8. Ablations

At minimum:

- relation-derived work units vs row-only units;
- relation-derived units vs whole invocation;
- no cross-microbatch batching;
- fixed RPC batches vs StageBatcher;
- no isolation;
- whole-shard retry;
- binary split isolation without semantic closure;
- no provenance capture;
- one vs multiple inflight microbatches;
- replica count and target batch size;
- node-barrier wait contribution;
- Reduce fiber skew and Relate key skew;
- deterministic and nondeterministic UDFs;
- failure rate/type/locality;
- allocator/actor-recycle mitigation for long steady-state Ray runs.

Post-publication downstream replay is not an ablation because it is not a V2
claim.

## 9. Metrics

Performance:

- observed wall time and goodput;
- items/rows/fibers/relations per second;
- p50/p95/p99 end-to-end and stage latency;
- real GPU utilization;
- batch fill and batch-size distribution;
- RPC count and bytes;
- queue wait, service time, canonicalization time, and publication wait;
- network and object-store traffic;
- peak driver/actor/object-store memory.

Recovery:

- accepted useful work;
- retried and redundant work;
- isolation depth and calls;
- timeout/cancellation/actor-replacement cost;
- failed WorkUnits and suppressed downstream outputs;
- completeness under equal fault policy.

Provenance:

- bytes per row and relation edge;
- identity/provenance materialization time;
- query/grouping overhead;
- port cardinality and M:N density.

Ray memory behavior:

- long-running actor RSS;
- anonymous mappings;
- object-store bytes;
- serialization allocation rate;
- actor recycle frequency;
- OOM incidence under identical steady-state load.

Proxy metrics must not be labeled as GPU utilization, wall time, or memory.

## 10. Correctness evidence

Deterministic tests must compare:

- values;
- EntityIds/IdentityKeys;
- normalized typed provenance;
- TerminalFailures;
- canonical output order.

Perturb:

- shard plans;
- batch composition;
- replica assignment;
- completion order;
- actor loss;
- bounded retry/split paths;
- Python hash seed and process boundary.

Nondeterministic workloads compare:

- structural validity;
- output completeness;
- valid relation/identity invariants;
- baseline-vs-baseline payload/cardinality variance.

They do not assert equal payload.

## 11. Scale and artifact gates

Target evaluation:

- 4–8 nodes;
- preferably 32 or more accelerators;
- repeated steady-state runs;
- strong/weak scaling and confidence intervals.

Public artifact:

- public data manifests/download scripts;
- pinned containers, dependencies, and models;
- cluster launch/configuration;
- baseline configurations and tuning records;
- raw anonymous execution traces for experiment runs;
- plotting and correctness-comparison scripts;
- fault definitions;
- smoke and full-scale reproduction guides.

Core results must not depend on private CEPH paths.

## 12. Rejection risks

The paper is not ready if:

- gains come mainly from persistent actors, ordinary batching, or inflight;
- WorkUnitRef is only a datatype and does not drive batching/recovery;
- Reduce/Relate execute by driver gather at claimed scale;
- isolation is compared against a baseline with different output semantics;
- only one workload or one node is evaluated;
- node-barrier and hot-key costs are hidden;
- VLM nondeterminism is presented as deterministic equivalence;
- Ray memory claims use object-store size as a proxy for process RSS;
- competitors are simulated rather than implemented;
- the artifact cannot reproduce the main result.

## 13. Go/no-go standard

The V2 architecture is sufficient to start implementation.

The system becomes a credible SDS submission only after demonstrating:

1. relation contracts produce materially better work boundaries;
2. those boundaries improve both batch formation and failure isolation;
3. canonical atomic semantics survive legal physical perturbations;
4. the mechanism scales through a real distributed data plane;
5. improvements persist against tuned baselines under equal semantics.
