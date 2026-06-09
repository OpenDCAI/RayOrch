# TODO: Runtime and RayModule Integration

Status: core adapter implemented; lifecycle direction recorded in
[`../runtime_module_lifecycle.md`](../runtime_module_lifecycle.md).

## Goal

Make the runtime semantics usable through the RayModule ecosystem without turning RayModule into a lineage-heavy abstraction.

## Current Direction

Keep `RayModule` as the lightweight eager product:

```text
RayModule
  actor replicas
  dispatch
  collect
  max_inflight
```

Let `RuntimeRayModule` reuse selected low-level machinery without inheriting
the eager lifecycle:

```text
RuntimeRayModule
  wraps user op
  owns runtime spec
  shares low-level actor machinery
  defers physical startup to Runtime Executor
```

## Key Design

Each Ray actor replica should hold:

```text
self.op       # user op instance
self.runtime  # rowwise runtime instance
```

Actor run flow:

```text
receive MicroBatch shard
runtime.run_rowwise(op.run, shard)
return RuntimeResult
```

## Acceptance Test

A RayModule-wrapped runtime op should:

- receive a MicroBatch through Ray;
- isolate a bad row inside the actor;
- return healthy MicroBatch output;
- return quarantine records;
- preserve row IDs for healthy rows;
- advance path IDs for healthy rows.
