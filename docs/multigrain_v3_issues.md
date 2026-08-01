# Multigrain V3 implementation issues

This file records concrete implementation questions discovered while building
`rayorch.experimental.multigrain_v3`. Resolved design decisions belong in
`docs/multigrain_v3_prototype_1.md`; this file should stay short.

## Open

- None currently.

## Resolved

- V3 uses `CompiledDAG` consistently; no parallel `CompiledProgram` type.
- Reduce anchors are semantic-only and never enter actor RPC payloads.
- Optional normal absence is explicit through `optional(port)` and `MISSING`.
- Public primitives retain RayModule-style `.pre_init(...).ray_options(...)`.
- Run execution must retain bounded multi-Arena in-flight overlap over shared
  persistent Stage actor pools; a single-Arena sequential executor is not an
  acceptable V3 milestone.
- Cross-block Reduce groups use multi-block row selectors; the Driver never
  resolves business payloads to compact a group before RPC.
- `RunResult.timeline` records arena/stage/dispatch/worker timing for overlap
  verification.
- `parent_bound` groups child Grains by the nearest `EntityLineage` parent;
  source-scope Grains group by their own EntityId.
- Stable RPC DTOs live in `protocol.py`; `execution.py` and `worker.py` do not
  import Arena internals.
- Arena keeps coarse blocks until delivery in Prototype 1, but enforces a
  `max_blocks` hard limit; early release remains benchmark-driven future work.
- The Arena implementation is a package: `state.py` contains passive records
  and queues, `reduce.py` owns the pure hierarchical ReduceAccumulator, and
  `engine.py` contains the sole Arena state machine. No mixin or second
  runtime authority was introduced.
- Hierarchical Reduce derives a canonical `scope_path` from anchor/member
  scopes. One Expand equals one nested-list level; no extra API keyword is
  required and implicit flattening remains unsupported.
- Deep-scope regressions cover five binary levels, twenty unary levels,
  intermediate N=0, intermediate drop, multiple aligned GROUPs, and
  intermediate fanout failure containment.
