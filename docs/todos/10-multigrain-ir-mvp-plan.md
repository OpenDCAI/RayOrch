# TODO: Multigrain IR MVP Implementation Plan

Status: implementation plan for the experimental `rayorch.experimental.multigrain`
prototype.

This document turns the multi-grain API design in
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md)
into an implementation checklist. The immediate goal is not full Ray execution.
The goal is a semantically complete, passive, optimizer-friendly IR that can
represent all MVP features before we build more runtime machinery.

## Design Target

The prototype should follow the same architecture lesson as HYDP DataFlow:

```text
authoring API
  -> passive IR
  -> validation / analysis / transform passes
  -> executor or backend lowering
```

For RayOrch multi-grain dataflow, the IR must additionally represent:

- per-port grain;
- parent relations across grains;
- child ordinals and display identities;
- materialization and recovery annotations;
- physical hints for Ray lowering;
- optimizer-inserted nodes such as rebatch or materialize.

The high-level API can grow convenience wrappers such as `Select`, but the IR
should remain a small relation-aware algebra.

The MVP should explicitly treat user-friendly APIs as front-end sugar over the
same canonical IR, not as separate execution models:

```text
torch-like module calls
wrapper declarations
forward helpers
plain return shapes
adapter hooks
optional return protocols
        -> MultigrainIR
```

This keeps the frontend open enough to borrow good ideas from Torch-like DAGs,
functional helpers, pure-UDF adapters, and relation protocols, while preventing
the representation from fragmenting. The compiler should normalize equivalent
meaning into the same `Map / Expand / Filter / Reduce / Relate / Project` IR
shape before verification and optimization.

## Design Principles

1. **Port first**: lineage, rebatching, reduce grouping, and recovery all attach
   to ports, not only to nodes.
2. **Passive IR**: no live operator instances, Ray handles, runtime actors, or
   backend datasets in the IR.
3. **Recipe nodes**: nodes carry importable operator recipes, constructor args,
   logical contracts, properties, and physical hints.
4. **Relation explicit**: every cross-grain fan-in must have an explicit
   relation contract such as `group_by` or `Relate`.
5. **Physical fusion preserves logical lineage**: optimization may fuse execution
   stages, but trace/recovery must still see the logical operators.
6. **Materialization is an annotation**: checkpoint/replay/debug boundaries are
   IR metadata or optimizer-inserted nodes, not executor-local hacks.

## Passive IR Decision: Trace Token vs. IR Ref

The single most important structural rule: **the trace-time symbolic token must
not enter the IR.** The token can carry a live builder for authoring ergonomics,
but the compiled IR only ever stores passive references/specs. This keeps
`MultigrainIR` picklable, engine-agnostic, and safe to hand to passes, remote
lowering, and offline tooling.

### What tracing needs vs. what the IR needs

- `forward()` is traced with torch-like ordinary calls (`images = self.expand(pdfs)`).
  For that, the value flowing through `forward()` (`SymbolicPort`) needs a handle
  back to the graph builder (`tracer`) so each wrapper call can register a node.
- The IR, by contrast, needs only passive data: `IRPortRef` (node/port/index)
  and `IRPortSpec` (ref + grain + display/ordinal metadata). No tracer, no live
  operator, no Ray handle.

### How other frameworks draw this line

| framework | trace token | token carries builder? | stored in IR | IR passive? |
|---|---|---|---|---|
| PyTorch FX | `Proxy` | yes (via `Tracer`) | `Node` (name/op/args) | yes |
| JAX | `Tracer` | yes (trace context) | `Var`/atom in `Jaxpr` | yes |
| TVM Relay / MLIR | builder `Expr`/`Value` | builder separate | `Expr`/`Value` ref | yes |
| ONNX | none (direct build) | n/a | `NodeProto`/`ValueInfo` | yes |
| HYDP-dataflow | `PortRef` | no (pure dataclass) | `PortRef` | yes |
| RayOrch (before) | `SymbolicPort` | **yes (`tracer`)** | **`SymbolicPort`** | **no** |
| RayOrch (now) | `SymbolicPort` | yes (`tracer`) | `IRPortRef`/`IRPortSpec` | yes |

Two proven paradigms both yield a passive IR:

- **A (FX / JAX):** the token carries the builder, but the IR stores a *different*
  passive ref type. Two types: trace token + IR ref.
- **B (HYDP-dataflow):** the token *is* the passive ref; the builder lives in an
  external `contextvar` active tracer. One type end-to-end.

The earlier RayOrch MVP was an unclean hybrid: it used paradigm A's builder-
carrying token but then stored that same token in the IR, so the IR transitively
held a live `tracer` and was not passive.

### Decision (paradigm A / FX-JAX aligned)

We keep the builder-carrying `SymbolicPort` for trace-time ergonomics and make
the IR store passive `IRPortSpec`/`IRPortRef`:

- `SymbolicPort` is trace-time only. It holds `tracer` so `self.op(x)` can
  register nodes during `forward()`. It never appears inside `MultigrainIR`.
- `GraphTracer.add_node()` returns `SymbolicPort` to `forward()` but records
  `IRNode(input_specs=..., output_specs=...)` using passive `IRPortSpec`.
- `GraphTracer.build()` converts graph inputs to `IRPortSpec` and graph outputs
  to `IRPortRef`.
- `IRNode.input_refs` / `output_refs` / `inputs` / `outputs` are convenience
  properties derived from the passive specs (used by passes and display).

Why paradigm A rather than paradigm B (HYDP's single ref): RayOrch wrappers
already call `port.tracer.add_node(...)` directly, which is explicit and needs no
global/`contextvar` state and supports concurrent/nested traces without
interference. Aligning to FX/JAX gets a passive IR with minimal churn; collapsing
to a single ref type (paradigm B) is a possible future refinement, not a
requirement.

Result: a compiled `MultigrainIR` now pickles cleanly and contains no
`SymbolicPort` (verified against the traced PDF pipeline), which is exactly the
"passive IR" property the passes and future Ray lowering depend on.

## Proposed Data Model

### References and Port Specs

```python
@dataclass(frozen=True)
class IRPortRef:
    node: str
    port: str = "out"
    index: int = 0


@dataclass(frozen=True)
class IRPortSpec:
    ref: IRPortRef
    grain: str
    name: str | None = None            # logical label; passive, no tracer
    payload: PayloadKind = PayloadKind.OBJECT
    display_key: DisplayKeySpec | None = None
    ordinal_key: OrdinalKeySpec | None = None
    materialize: MaterializePolicy = MaterializePolicy.NEVER
    # node/index/port convenience properties delegate to `ref`
```

`IRPortSpec` is the passive port carrier that actually lives in the IR (the
FX-`Node`/JAX-`Var` analogue). The trace-time `SymbolicPort` is a separate type
that additionally holds the live `tracer` and never enters the IR.

`display_key` supports user-facing trace output such as
`document=a.pdf/page=17`. `ordinal_key` supports deterministic reduce ordering
after physical rebatching.

### Nodes

```python
@dataclass(frozen=True)
class IRNode:
    name: str
    kind: NodeKind
    input_specs: tuple[IRPortSpec, ...]   # passive; no SymbolicPort
    output_specs: tuple[IRPortSpec, ...]  # passive; no SymbolicPort
    contract: CardinalityContract
    op: OperatorRecipe
    properties: OperatorProperties
    physical: PhysicalHints
    # input_refs / output_refs / inputs / outputs are derived properties
```

`NodeKind` starts with:

```text
MAP
EXPAND
FILTER
REDUCE
RELATE
PROJECT
REBATCH
MATERIALIZE
```

`PROJECT`, `REBATCH`, and `MATERIALIZE` may be inserted by optimizer passes.

### Cardinality and Relations

```python
@dataclass(frozen=True)
class CardinalityContract:
    kind: CardinalityKind
    input_grains: tuple[str, ...]
    output_grains: tuple[str, ...]
    relations: tuple[RelationSpec, ...]


@dataclass(frozen=True)
class RelationSpec:
    output: IRPortRef
    relation: RelationKind
    parents: tuple[IRPortRef, ...]
    parent_input: int | None = None
    anchor: IRPortRef | None = None
    ordinal: OrdinalPolicy = OrdinalPolicy.PRESERVE
    missing: MissingChildPolicy = MissingChildPolicy.FAIL_OPEN
```

`RelationKind` covers:

```text
PRESERVE   1:1
EXPAND     1:N
FILTER     1:0/1
REDUCE     N:1
RELATE     M:N
GROUPED    helper relation for reduce input grouping
```

### Operator Recipe and Properties

```python
@dataclass(frozen=True)
class OperatorRecipe:
    cls_ref: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorProperties:
    deterministic: bool = True
    side_effect: bool = False
    idempotent: bool = True
    retryable: bool = True
    expensive: bool = False
    gpu_heavy: bool = False
    stateful: bool = False
```

These properties are mostly annotations in the MVP, but they are required for
future partial replay and materialization planning.

### Physical Hints

```python
@dataclass(frozen=True)
class PhysicalHints:
    replicas: int = 1
    num_gpus_per_replica: float = 0.0
    max_inflight: int = 1
    batch_size: int | None = None
    prefer_rebatch: bool = False
    engine: str | None = None
```

The logical IR stores hints, not concrete Ray actors or resource allocation.

### Materialization

```python
@dataclass(frozen=True)
class MaterializationSpec:
    port: IRPortRef
    policy: MaterializePolicy
    reason: MaterializeReason
    storage: StorageSpec | None = None
```

Policies:

```text
NEVER
DEBUG_ONLY
ON_FAILURE
ALWAYS
CHECKPOINT
```

Reasons:

```text
TRACE
REPLAY_BOUNDARY
EXPENSIVE_OP
NONDETERMINISTIC
SIDE_EFFECT
USER_REQUEST
```

### Top-Level IR

```python
@dataclass(frozen=True)
class MultigrainIR:
    name: str
    inputs: tuple[IRPortSpec, ...]
    nodes: Mapping[str, IRNode]
    topo_order: tuple[str, ...]
    deps: Mapping[str, tuple[str, ...]]
    consumers: Mapping[str, tuple[str, ...]]
    graph_outputs: tuple[IRPortRef, ...]
    materialization: tuple[MaterializationSpec, ...] = ()
```

The IR should provide:

```python
ir.describe()
ir.to_dict()
ir.to_mermaid()
```

Like HYDP DataFlow, the IR should be serializable and reconstructable without
holding live operator instances.

## Completeness Checklist

Use this matrix to check whether the IR can represent a feature without special
cases:

| Axis | Required Coverage |
| --- | --- |
| Topology | inputs, outputs, fanout, fanin, deps, consumers, topo order |
| Ports | named inputs, named outputs, multi-output, per-port grain |
| Grain | document, page, block, chunk, sample, arbitrary user labels |
| Relation | preserve, expand, filter, reduce, relate, grouped helper |
| Execution | resource hints, stateful, gpu-heavy, backend binding |
| Optimization | fusion, rebatching, reduce group planning, materialization |
| Recovery | deterministic, side-effect, retryable, checkpoint policy |
| Observability | display identity, child ordinal, trace, drop, quarantine |
| Serialization | operator reference, args, kwargs, provenance, no live object |
| Contract Checks | static declaration, dynamic evidence, runtime shape/relation validation |

## MVP Review Against Frontend-Sugar Target

Current coverage:

| Frontend mechanism | MVP status | Canonical IR target |
| --- | --- | --- |
| Torch-like module call | Implemented in `Pipeline.compile()` tracing | `MultigrainIR` nodes and ports |
| Wrapper declarations | Implemented for `Map`, `Expand`, `Filter`, `Reduce`, `Relate`, `Select` | Node kind, operator recipe, cardinality contract |
| Forward helper | `group_by()` implemented | `REDUCE` with anchor and descendants |
| Plain return shape | Implemented for nested `Expand`, `Filter` masks, `Select` annotations | `EXPAND`, `FILTER`, `MAP+FILTER+PROJECT` |
| Adapter hook | `Relate` now has three tiers: declarative `on={role: field}` key-join, by-ref `relation_adapter="pkg:fn"` (dotted path, resolved at execute), and live `relation_fn`; all execute locally and via the passive IR. `mask_fn` / `key_fn` for `Select` still not implemented. See [`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md) | Existing relation specs plus runtime evidence checks; `on` / `relation_adapter` stored as pure data in the recipe provenance |
| Return protocol | Not implemented yet | Escape hatch for advanced `RELATE` |
| Relation-aware display | Basic `describe()`, `to_dict()`, `to_mermaid()` implemented | IR inspection and debugging |
| Validation | Basic `VerifyPass` implemented | Reject ambiguous or non-canonical relation use |
| Local IR execution | `MultigrainExecutor` implemented for `Map`, `Expand`, `Filter`, `Reduce`, `Relate(relation_fn)`, the `Select` lowering (`Map + SelectFilter + Project`, mask position carried in the recipe provenance), plus `Rebatch` / `Materialize` pass-through | Execute canonical, lowered, and transformed IR without a live Pipeline object; also validated on pickled-then-reloaded IR |
| IR passes | `RelationSummaryPass`, `RebatchCandidatePass`, `PlanReduceGroupsPass`, `MarkMapFilterFusionCandidatesPass`, and `InsertRebatchAfterExpandPass` implemented | Analysis and transform passes that read/write canonical IR |
| Package structure | User-facing `__init__` is narrowed; operator wrappers split by primitive family | Keep open-source API small while preserving internal IR/pass modules |

Gaps to close before calling the MVP semantically complete:

- Adapter hooks for `Relate` are done (three tiers: `on=` key-join,
  `relation_adapter` by-ref, `relation_fn` live). Remaining: `mask_fn` / `key_fn`
  for `Select`-like APIs, and outer/left-join semantics for `on=` (currently
  inner equi-join only).
- Decide whether the current `PortBatch.relations` sidecar is enough for the
  local MVP or should become a dedicated relation-aware batch/container.
- Add canonicalization tests showing that different frontend sugars with the
  same meaning produce equivalent normalized IR.
- Extend optimizer tests beyond the implemented IR-only pass tests.
- Extend local IR executor coverage for future return protocols and non-trivial
  physical behavior beyond `REBATCH` / `MATERIALIZE` pass-through.
- Clarify which APIs are stable surface and which are escape hatches.

MVP acceptance should be judged by this rule:

```text
Every accepted user API must either lower to the small canonical IR directly,
or be rejected with a readable diagnostic before execution.
```

## Initial Passes

### IR Validation

```text
ValidateCrossGrainFanIn
  reject ordinary Map calls that mix grains without group_by or Relate

ValidateExpandParent
  require multi-input Expand to declare parent_input

ValidateReduceGroupBy
  require Reduce inputs to include one anchor and descendants of that anchor

ValidatePortRelations
  check every output port has exactly one relation spec
```

### Runtime Contract Checks

Some relation facts are not statically knowable because they depend on UDF
outputs. The executor or wrapper adapter must validate them at invocation time:

```text
CheckMapPreserve
  output length must match aligned input length

CheckExpandEvidence
  nested group count must match parent rows
  multi-output expanded groups must share relation lengths unless explicitly
  declared as separate relations

CheckFilterEvidence
  mask length must match input rows
  mask values must be bool
  kept records preserve identity

CheckReduceEvidence
  descendants must have ancestry to the anchor
  child ordering must use ordinal or stable key
  missing-child policy must be explicit

CheckRelateEvidence
  parent refs must be invocation-local
  roles must match declared inputs
  relation multiplicity and ordering must be explicit
```

These checks are not optimizer passes. They are runtime assertions that the
opaque UDF's observed relation effect satisfies the IR contract. Failures should
be reported as contract violations, not as business quarantine records.

### Analysis

```text
PortProvenance
  compute upstream port ancestry for each port

RelationLineage
  compute logical relation chain used by trace and partial replay

MaterializationCandidates
  mark ports that may be useful replay/debug boundaries
```

Implemented MVP analysis passes:

```text
RelationSummaryPass
  summarize node kinds, relation kinds, expand outputs, and reduce groups

RebatchCandidatePass
  find Expand outputs whose physical hints request child-grain rebatching

PlanReduceGroupsPass
  create minimal reduce grouping plans from REDUCE contracts

MarkMapFilterFusionCandidatesPass
  detect canonical Map -> Filter patterns for physical Select fusion
```

### Transforms

```text
InsertRebatchAfterExpand
  insert child-grain REBATCH before GPU-heavy Map stages

FuseMapFilter
  keep logical Map+Filter but create a fused physical Select candidate

FuseAdjacentMaps
  fuse cheap same-grain Maps when safe

PlanReduceGroups
  decide child order, missing-child policy, and grouping materialization

PlaceMaterializationBoundaries
  add materialization annotations for expensive or unsafe replay regions
```

Implemented MVP transform pass:

```text
InsertRebatchAfterExpandPass
  insert logical REBATCH nodes after Expand outputs and rewrite downstream
  input refs in the IR
```

## Implementation Checklist

### 1. Upgrade `ir/model.py` Data Model

- Add `IRPortRef`, `IRPortSpec`, `IRNode`, `RelationSpec`,
  `CardinalityContract`, `OperatorRecipe`, `OperatorProperties`,
  `PhysicalHints`, `MaterializationSpec`, and `MultigrainIR`.
- Keep backward-compatible aliases if helpful:
  `CompiledGraph = MultigrainIR`, `NodeSpec = IRNode`.
- Preserve current test ergonomics while adding richer fields.

### 2. Update `GraphTracer`

- Build `MultigrainIR` from `Pipeline.forward()`.
- Populate port specs with grain and relation metadata.
- Populate `deps`, `consumers`, and `topo_order`.
- Store operator recipe instead of live operator when possible.

### 3. Add IR Display and Serialization

- Implement `describe()`.
- Implement `to_dict()` with no live object references.
- Implement `to_mermaid()` for graph debugging.

### 4. Add IR Verification

- Start with pure functions or small classes.
- Cover cross-grain Map, Expand parent, Reduce group_by, and output relation
  consistency.

### 5. Add Pass Skeleton

- Add minimal `AnalysisPass`, `VerifyPass`, `TransformPass`, `LintPass`.
- Add a small `PassManager` after the IR stabilizes.
- Do not overbuild caching/invalidations until a transform needs them.

### 6. Implement Local IR Executor

- Implemented `MultigrainExecutor` for local execution from `MultigrainIR`.
- Executes `Map`, `Expand`, `Filter`, `Reduce`, `Relate` with `relation_fn`, the
  `Select` lowering (synthetic `SelectFilter` reads the mask column position from
  `op.provenance["mask_input"]` and drops it), and `Project` / `Rebatch` /
  `Materialize` pass-through.
- Stores node outputs in a context keyed by `IRPortRef`.
- Keeps this local-only until semantics are stable.
- Routes UDF results through eager wrapper contract checks before materializing
  output ports.
- Validated end-to-end with fresh dummy operators over every motif, including
  execution of a pickled-then-reloaded IR and a rebatch-transformed IR, which
  confirms the passive IR alone (no live `Pipeline`) can drive execution.

### 7. Implement Filter and Select

- Implement `Filter` as mask-only `1:0/1`.
- Model `Select` as high-level API over `Map + Filter + Project`.
- Preserve drop metadata separately from quarantine.

### 8. Implement First Optimizer Passes

- Insert within-microbatch rebatch after `Expand` before GPU-heavy `Map`.
- Add a physical-fusion annotation for `Map + Filter`.
- Plan reduce grouping and ordering.

### 9. Materialization and Recovery Annotations

- Add annotation-only support first.
- Do not implement real replay until materialization boundaries are observable in
  tests.

### 10. Test Matrix

Add tests for:

- linear Map;
- fanout;
- same-grain fanin;
- invalid cross-grain fanin;
- multi-output Expand with shared relation;
- Reduce with group_by;
- Filter and drop metadata;
- Select lowering;
- rebatch insertion;
- materialization annotations;
- `to_dict()` contains no live operator instances.

## Ray Execution MVP (`MultigrainRayExecutor`)

A first, deliberately small Ray lowering exists in
`rayorch/experimental/multigrain/ray/executor.py`. It reuses the local
`MultigrainExecutor` node logic *inside* Ray tasks, so semantics stay identical
to local execution and only scheduling changes. It validates that the passive IR
plus `PhysicalHints` can drive real parallelism:

- **Intra-node replica parallelism**: row-independent nodes (`Map`, `Filter`,
  `Expand`) are row-sharded across `PhysicalHints.replicas` Ray tasks and merged
  back with `core.concat`, preserving record identity and lineage. `Reduce` and
  `Relate` run on a single task in the MVP (they need cross-row context).
- **Pipeline microbatch overlap**: `execute_microbatches(...)` launches
  whole-graph execution per microbatch as Ray tasks with a bounded in-flight
  window (`max_inflight`).

Measured on CPU + per-row sleep dummy ops (8 rows / 4 microbatches):

| knob | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Map replicas (8 rows) | 2.06s | 1.10s | 0.70s | 0.42s |
| microbatch overlap (4 mb) | 3.22s | 1.61s | 0.81s | - |

Covered by `test/experimental/multigrain/test_ray_parallelism.py` (marked
`slow`; needs `--runslow` + the `ray_cluster` fixture). The executor is not part
of the narrow package `__init__`; import it from
`rayorch.experimental.multigrain.ray`.

MVP limits still open: `batch_size` hints are not yet consumed; `Reduce`/`Relate`
are not sharded; there is no cross-node streaming scheduler (microbatches overlap
at whole-graph granularity, not per-stage).

### GPU load balancing on imbalanced 1:N fan-out

`num_gpus_per_replica` is now consumed: sharded node tasks are submitted with
`.options(num_gpus=...)`, so each replica is pinned to a GPU. The executor also
takes an optional `shard_planner(node, inputs, replicas) -> per-shard row-index
lists`; returning `None` falls back to contiguous ranges. `lpt_shard_planner(
weight_of)` implements Longest-Processing-Time greedy bin packing so shards
balance *total work* (e.g. page content length) rather than row count. Row
identity/ordinals survive via `PortBatch.take`, and any downstream `Reduce`
restores logical order via ordinals, so cross-shard reordering is safe.

Validated with a real GPU MinerU simulation on 4×H20
(`test/experimental/multigrain/{gpu_ops.py,bench_gpu_mineru.py}`, run under the
`torch-base` conda env): document -> Expand to variable-length pages -> GPU Map
(per-page cost = content length, real `torch` matmuls, ~4.5 ms/unit, perfectly
linear) -> Reduce back to document. 12 docs / 67 pages / 964 GPU units, skewed so
a few documents dominate:

| shard planner | per-GPU makespan | idle bubble | e2e pipeline |
|---|---|---|---|
| contiguous (equal row count) | 1.52s (units 275/334/121/234) | ~28% | 1.53s |
| LPT (work-balanced)          | 1.10s (units ~241 each)      | ~0%  | 1.11s |

Contiguous row-sharding leaves the light-shard GPU idle (~28% bubble) because the
1:N fan-out is imbalanced; work-aware LPT rebalancing removes the bubble (~1.38x
here). This is the "1:N reordering to eliminate bubbles" mechanism made concrete
on real GPUs. (Clean per-GPU numbers use a warmed, GPU-pinned actor pool to avoid
Ray worker cold-start; the actors are released before the executor's GPU tasks run
so they don't hold the GPUs.)

#### Long-tail acceleration vs. scheduling theory

Sharding an imbalanced fan-out across `R` GPUs is makespan minimization on
identical machines. For per-page weights `w_i`: `OPT >= max(sum(w)/R, max(w_i))`
(lower bound), `makespan_LPT <= (4/3 - 1/(3R))*OPT` (Graham 1969), and
`efficiency = sum(w)/(R*makespan) = 1 - idle_bubble`. Naive contiguous (equal row
count) sharding ignores `w_i`, so under a long tail a few heavy pages collide on
one GPU and the bubble -- hence the achievable speedup -- grows with tail
heaviness. `test/experimental/multigrain/bench_gpu_longtail.py` sweeps a Pareto
tail on 4×H20 (32 pages, real `torch` matmuls):

| Pareto alpha | tail p50/p99/max | measured speedup | theory speedup | LPT efficiency | 4/3 bound |
|---|---|---|---|---|---|
| 2.5 (light) | 4/10/10 | 1.12x | 1.12x | 100% | ok |
| 1.7 | 5/13/13 | 1.16x | 1.16x | 98% | ok |
| 1.2 (heavy) | 6/39/39 | **1.46x** | 1.46x | 99% | ok |

Measured GPU makespan tracks the analytic prediction to two decimals, LPT stays
under its 4/3 bound and runs at ~99% efficiency, and the speedup rises
monotonically as the tail gets heavier -- exactly what the theory says. On a full
`Expand -> GPU OCR -> Reduce` pipeline with two dominant heavy documents (16 docs
/ 65 pages), the executor goes **1.56s -> 1.02s (1.53x)** with identical results.

#### Lineage is invariant to the reordering

The LPT rebalancing *permutes rows across shards*, so it must not corrupt
lineage. `test/experimental/multigrain/test_lineage_under_parallelism.py`
(CPU dummy ops, `slow` + `ray_cluster`) treats the single-task local executor as
ground truth and asserts the parallel + reordered run is byte-for-byte equal:

- **identity/ordinal**: `Reduce` regroups by ancestor id and re-sorts by ordinal,
  so every document's pages come back in original order despite the scatter;
- **fault isolation**: a `BadRecordError(index=...)` raised inside whatever shard
  the bad page landed in is quarantined with the *same* `ErrorTrace`
  (`source_item`, `logical_item` `d03/page=2`, `upstream_path
  (SplitPages, EmbedPage)`), and every healthy page survives.

This closes the loop: the same lineage metadata that powers automatic error
tracing is exactly what makes the performance reordering safe.

#### Complex multi-stage load: bubble taxonomy and what we can kill

A single Map stage only exposes one bubble source. To confirm bubble elimination
under a realistic load, `test/experimental/multigrain/bench_gpu_complex.py` runs
`doc -Expand-> pages -Expand-> blocks -Filter-> dense -Map(GPU OCR)-> -Reduce-> doc`,
stacking three bubble sources on the wide GPU stage at once, and prints the
honest residuals. The full taxonomy:

| # | Bubble source | Trigger | Mechanism | Status |
|---|---|---|---|---|
| 1 | intra-stage load imbalance | long-tail per-item work | LPT re-shard by work | killed (~99%) |
| 2 | compounded fan-out | Expand->Expand long tail | LPT at the wide stage | killed |
| 3 | filter-induced skew | data-dependent drop | passive IR re-partitions each node's *own* input | killed |
| 4 | inter-stage barrier | stage N+1 waits for all of stage N | whole-graph microbatch overlap | partial |
| 5 | reduce fan-in skew | one anchor gets a huge group | Reduce is single-task in MVP | residual |
| 6 | atomic giant item | one row's work ~= sum(w)/R | none (a row cannot be split) | residual (OPT-bound) |

Because the IR is passive, every node re-partitions the data it actually
receives, so sources #1-#3 are absorbed at the OCR stage even though they arise
upstream. Measured on 4×H20 (8 docs / 63 raw blocks -> 44 survive a `work>=4`
filter, two-level long-tail fan-out, per-block work long-tailed):

- OCR stage: contiguous makespan 1.38s (efficiency 84%, **16% bubble**) ->
  LPT 1.15s (**efficiency 100%, 0% bubble**), exactly the OPT lower bound
  (`max(sum(w)/R, max w)` = 1.15s) and inside the 4/3 bound;
- end-to-end pipeline 1.39s -> 1.16s (1.19x) with byte-identical results.

The residuals are stated, not hidden: #5 (the heaviest document was ~0.95-1.5x of
`W/R` across seeds) would idle GPUs if the Reduce itself were GPU-heavy, because
Reduce is not sharded yet; #6 is fundamental (the largest single block was only
~0.1-0.2x of `W/R` here, so it was not binding, but a monster item would cap the
speedup at `OPT`). The scheduling guarantee (LPT meets Graham's bound and >=95%
efficiency on a long tail, and beats contiguous, including after a filter) is
locked into CI deterministically by
`test/experimental/multigrain/test_load_balancing.py` (no GPU, no timing).

## MVP Boundaries

Do not implement yet:

- Ray lowering of `Reduce`/`Relate` sharding (Map/Filter/Expand done);
- `batch_size`-driven rebatching (GPU placement via `num_gpus_per_replica` done);
- a per-stage streaming scheduler (vs whole-graph microbatch overlap);
- persistent lineage sink;
- global shuffle or distributed join;
- full `Relate`/emit semantics;
- actual partial replay;
- adaptive runtime re-optimization.

These depend on the IR being complete enough to express them first.

