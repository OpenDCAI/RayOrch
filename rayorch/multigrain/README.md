# MultiGrain runtime

This package is intentionally self-contained and has three layers:

```text
program/    symbolic authoring, analysis, lowering, and verification
runtime/    semantic microbatch state and dispatch indexes
execution/  Ray actors, RPC lifecycle, worker ABI, and result materialization
```

The package root exports the small authoring surface (`Pipeline`, `RayModule`,
`Port`, `F.*`), execution (`Executor`, `RunResult`), and explicit failure
values (`RecordFailure`, `GroupFailure`). `protocol.py` and `recovery.py` are
Ray-free contracts shared across the layers.

Structural primitives describe relationships and never create actors. Only a
`RayModule` call creates a worker grain. The runtime uses immutable coordinates,
generation-fenced reports, atomic multi-output commit, and a microbatch-local
parent barrier for `GroupFailure`.

Each Call owns one completion-ordered READY queue. Committing an upstream Grain
immediately publishes its facts and may make downstream Grains dispatchable;
execution never waits for an entire stage or input domain to drain. READY
Grains from different parent entities may share a bounded worker batch. Parent
identity remains part of runtime state only for lineage and `GroupFailure`
isolation, not as a batching policy.
