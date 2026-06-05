# Iteration Log: Runtime, RayModule, and Stage Supervisor

Date: 2026-06-05

## Context

RayOrch already has a working performance path:

```text
DagExecutor(driver)
  -> RayModule(driver-side physical module)
      -> RunnerActor replicas
```

This path has been validated on Flash-MinerU-like multi-stage pipelines and can reach the expected pipeline-parallel lower bound when replicas and inflight batches are configured well.

The new runtime work adds record-level semantics:

```text
MicroBatch
  columns
  row_ids
  path_ids

Runtime
  rowwise fault isolation
  quarantine
  path lineage
```

The open design question is whether every RayModule should get a dedicated supervisor process or Ray actor to manage macro-stage dispatch, queueing, retry, and health.

## Decision For Now

Do not add a per-RayModule supervisor by default in the MVP.

Keep the current direct-dispatch shape:

```text
DagExecutor(driver)
  -> RayModule(driver object)
      -> RunnerActor replica
          -> Runtime instance
```

Each replica actor should own its own Runtime instance. Record-level retry belongs inside the worker actor because the user op instance, model weights, GPU context, and local buffers already live there.

## Why Not Supervisor Yet

Adding a supervisor gives useful control points, but it also adds cost:

- one more Ray RPC hop per stage call;
- another failure domain;
- more complicated debugging;
- possible supervisor bottlenecks;
- heavier architecture before the runtime semantics are stable.

The current driver-side DagExecutor is not yet the bottleneck for the known Flash-MinerU-like pipeline. We should preserve the proven path while integrating runtime semantics.

## Retry Boundaries

Record-level retry:

```text
Worker Runtime
  BadRecordError / ordinary Python exception
  split-and-retry
  quarantine singleton bad rows
```

Actor/task-level retry:

```text
RayModule or future StageSupervisor
  RayActorError
  worker crash
  node loss
  severe GPU failure
  adaptive smaller-batch retry
```

DAG-level scheduling:

```text
DagExecutor
  dependency readiness
  graph-level inflight
  downstream release
```

## Future StageSupervisor Trigger

Introduce an optional StageSupervisor when we need several of these:

- stage-local queues and backpressure;
- actor health tracking and replica replacement;
- adaptive batch sizing after OOM;
- online metrics and dashboard control;
- long-running production stage state;
- driver becoming a scheduling bottleneck;
- complex stage-level retry policy.

The future shape can be:

```text
DagExecutor(driver)
  -> StageSupervisor actor
      -> RunnerActor replicas
          -> Runtime instance
```

This should be optional, likely as a `SupervisedRayModule` or `RuntimeStage`, not a mandatory replacement for RayModule.

## Near-Term Path

1. Keep RayModule as the physical actor/dispatch/collect layer.
2. Put one Runtime instance inside each worker replica actor.
3. Add MicroBatch-aware dispatch and RuntimeResult-aware collect.
4. Let DagExecutor aggregate runtime outputs, quarantines, and lineage deltas.
5. Keep the StageSupervisor boundary in mind, but do not implement it until actor-level retry or adaptive batching requires it.
