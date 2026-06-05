# TODO: Runtime and RayModule Integration

## Goal

Make the runtime semantics usable through the RayModule ecosystem without turning RayModule into a lineage-heavy abstraction.

## Preferred Direction

Keep RayModule as the physical layer:

```text
RayModule
  actor replicas
  dispatch
  collect
  max_inflight
```

Add a runtime-aware wrapper or adapter:

```text
RuntimeModule / RuntimeRayModule
  wraps user op
  owns runtime spec
  internally uses RayModule
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
