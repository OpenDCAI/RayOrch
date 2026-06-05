# TODO: Optional Stage Supervisor

## Goal

Preserve a future path for per-stage supervisor actors without forcing them into the MVP.

## Current MVP Shape

```text
DagExecutor(driver)
  -> RayModule(driver object)
      -> RunnerActor replicas
```

This is good enough while the driver is not a bottleneck and retry logic is simple.

## Future Shape

```text
DagExecutor(driver)
  -> StageSupervisor actor
      -> RunnerActor replicas
```

## Supervisor Responsibilities

A supervisor should handle stage-level concerns:

- local stage queue;
- replica health;
- actor/task retry;
- adaptive batch sizing;
- OOM fallback;
- metrics;
- backpressure.

It should not handle record-level split-and-retry. That belongs inside each worker Runtime.

## Trigger

Implement only when the runtime needs actor health tracking, adaptive batching, or stage-local scheduling beyond what DagExecutor can cleanly manage.
