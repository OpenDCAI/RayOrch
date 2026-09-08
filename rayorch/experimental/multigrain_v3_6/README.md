# MultiGrain v3.6 implementation

This directory is the supplemental implementation evaluated in the paper. It
is intentionally self-contained and has three layers:

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

For the reviewer-facing explanation, start with
[`docs/multigrain_v3_6_getting_started.md`](../../../docs/multigrain_v3_6_getting_started.md).
The v3--v3.5 directories are historical experiments and are not used for the
reported results.
