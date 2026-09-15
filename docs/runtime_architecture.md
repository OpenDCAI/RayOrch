# Runtime architecture

RayOrch exposes one small API from the package root. Its implementation has
three private layers:

```text
_program/    symbolic analysis, lowering, and verification
_runtime/    semantic microbatch state and dispatch indexes
_execution/  Ray actors, RPC lifecycle, worker ABI, and result materialization
```

The package root exports authoring (`Pipeline`, `RayModule`, `Port`, `F.*`),
execution (`Executor`, `run`, `RunResult`), and recovery (`RecoveryPolicy`).
Internal identity and worker-protocol types stay private so their representation
can evolve without expanding the compatibility surface. The UDF-facing
`MISSING`, `RecordFailure`, and `GroupFailure` values remain available from the
package root; failure values can also be imported from `rayorch.failures`.

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
