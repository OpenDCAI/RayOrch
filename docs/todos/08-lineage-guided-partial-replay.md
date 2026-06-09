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
- pipeline-level microbatch overlap;
- record identity;
- multi-parent DAG lineage;
- bad-record quarantine;
- continued execution of healthy records;
- Flash-MinerU-compatible DAG topology.

Current lineage is primarily used for reporting and error attribution.

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
