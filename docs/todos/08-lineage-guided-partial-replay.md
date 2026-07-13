# TODO: Lineage-Guided Partial Replay

Status: research direction after the current Runtime MVP.

## Motivation

A feature-rich framework is not automatically a research contribution.
GPU orchestration, DAG execution, lineage, retry, and Ray integration are useful,
but each mechanism already exists in related systems.

The research question should be:

> How can a heterogeneous GPU pipeline use fine-grained lineage, operator cost,
> and failure history to choose an isolation and recovery scope that preserves
> goodput under bad records and system failures?

This turns orchestration, lineage, and recovery into one mechanism rather than
three independent features.

## Proposed Positioning

Working title:

> A Lineage-Guided Fault-Isolated Runtime for Heterogeneous AI Data Pipelines

The target workload is a multi-stage AI data pipeline containing CPU, GPU,
VLM, and LLM operators with microbatch overlap.

## Core Mechanism

Lineage must participate in recovery decisions instead of serving only as a
post-execution reporting facility.

For a bad record:

```text
one PDF fails in OCR
  -> identify the affected record and dependency branches
  -> reuse successful PDF decode and layout outputs
  -> replay only the necessary OCR sub-batch and downstream subgraph
  -> allow unrelated microbatches and stages to continue
```

For a system failure:

```text
GPU actor or node fails
  -> inspect committed outputs and lineage
  -> identify records whose computation was not committed
  -> restart from the nearest recoverable stage
  -> avoid restarting the entire pipeline
```

## Candidate Contributions

### Heterogeneous AI DAG Execution

Pipeline microbatches across CPU, GPU, VLM, and LLM operators while respecting
per-stage resources, replica counts, and inflight limits.

### Hybrid-Granularity Lineage

- Compress record-preserving `1:1` execution as shared path lineage.
- Materialize parent edges only when `1:N`, `N:1`, or `N:M` operators change
  record identity.
- Support low-cost impact analysis without requiring full cell lineage.

The first bullet is a **required optimization, not current behavior**. The MVP
currently stores multiple Python metadata objects per row and copies lineage /
ancestor / ordinal structures through `take` and `with_values`. This is adequate
for the 7k-page MinerU experiment but must be replaced with shared 1:1 paths and
compact/columnar ancestry before claiming million-record scalability.

### Lineage-Guided Fault Isolation And Partial Replay

Use lineage, operator cost, failure type, and prior failures to choose between:

- direct bad-record removal;
- recursive batch isolation;
- replay from a materialized intermediate;
- actor-level retry;
- subgraph replay;
- abort for unsafe side-effect operators.

The policy should address bad input, Python exceptions, GPU OOM, actor crashes,
and node loss without unnecessarily replaying healthy work.

## Current Foundation

The Runtime MVP already provides:

- Ray Actor replicas and resource configuration;
- the primitives needed for pipeline-level microbatch overlap;
- record identity;
- multi-parent DAG lineage;
- bad-record quarantine;
- continued execution of healthy records;
- Flash-MinerU-compatible DAG topology.

Current lineage is primarily used for reporting and error attribution.

Important execution caveat: the public multigrain executor currently provides
node-by-node sharding and persistent pools, but the high-performance
render/OCR/assemble overlap used by the MinerU benchmark is orchestrated in
`Flash-mineru/mg_bridge/run_bench.py` through private pool/shard APIs. The graph
itself is framework-native (`MinerUReal` compiles
`Pipeline → Expand → Map → Reduce` into passive IR); the benchmark-specific
physical scheduler is not yet behind one public executor API. Unifying that
scheduler is a prerequisite for a truthful stage-global deferred-replay barrier.

## Missing Research Loop

The paper-level system still needs:

```text
lineage
  -> determine precisely affected computation
  -> locate reusable committed intermediates
  -> replay only affected records and DAG nodes
  -> measure the recovered GPU goodput
```

This requires:

- intermediate materialization and commit policy;
- deterministic, nondeterministic, and side-effect operator declarations;
- actor and node failure detection;
- replay planning;
- `1:N` and `N:M` lineage through the buffered emit API;
- persistent lineage and execution metadata;
- adaptive isolation rather than unconditional binary splitting.

## Scope Boundaries

The correctness argument assumes UDF value-purity: a record's output is
independent of internal IDs, global position, execution time, and replica.
Consequently, the current model deliberately does not define semantics for
globally stateful/sessionized operators, global sorting, cross-record
deduplication, iterative/cyclic dataflow, or unsafe external side effects.
These require explicit state/materialization/commit contracts rather than being
silently treated as reorderable maps.

`Relate` represents M:N evidence and lineage, but its current key-join execution
is an in-process in-memory hash join with per-key Cartesian expansion. Large
distributed M:N joins require partitioning, spill/backpressure, and distributed
materialization; representation completeness should not be confused with
execution scalability.

## Evaluation Requirements

Use Flash-MinerU plus at least one LLM or multimodal data-governance workload.
Evaluate on an 8-32 GPU cluster where possible.

Compare against:

- plain Ray pipelines;
- Ray Data or Ray Data LLM;
- whole-microbatch retry;
- per-record execution;
- fixed binary split isolation;
- execution without lineage.

Measure:

- failure-free throughput and GPU utilization;
- successful records per unit of GPU time (`goodput`);
- wasted GPU work under different failure positions and rates;
- replay scope and recovery latency;
- lineage runtime, memory, and storage overhead;
- lineage query latency;
- scalability across stages, replicas, and GPUs.

The intended result is:

- near-baseline performance without failures;
- higher goodput than coarse-grained retry under failures;
- substantially smaller replay scope;
- low and predictable lineage overhead.

## Acceptance Criterion

This direction becomes a credible VLDB Research Track contribution when the
system demonstrates that lineage-guided recovery is both:

1. materially more precise than batch or stage replay; and
2. measurably better for end-to-end GPU goodput on real heterogeneous AI
   pipelines.

Until then, the Runtime remains a strong engineering framework and experimental
platform, but the research claim is not yet complete.
