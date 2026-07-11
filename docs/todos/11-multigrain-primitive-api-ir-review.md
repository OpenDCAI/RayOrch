# Multigrain Primitive API and IR Review

This note reviews the experimental multigrain programming model from the
primitive level. The key question is not whether the system can trace a DAG.
The key question is whether each user-facing primitive has one clear semantic
meaning, lowers to one canonical IR shape, and leaves enough room for future
lineage, recovery, scheduling, and optimization.

## Review Standard

Each primitive should pass the same checks:

- User API: the common case should read like an ordinary Torch-style module call.
- Semantic contract: the cardinality and relation meaning should be explicit.
- Canonical IR: equivalent user syntax should lower to one normalized IR shape.
- Validation: illegal or ambiguous cases should be rejected early with readable
  diagnostics.
- Runtime check: UDF outputs or adapter outputs should be checked against the
  declared relation contract when the relation effect is dynamic.
- Runtime value: the IR should expose enough metadata for lineage trace,
  row-level recovery, rebatching, materialization, and Ray lowering.
- Extension point: advanced behavior should extend the primitive through explicit
  metadata or a more general primitive, not through hidden conventions.
- Relation evidence: the primitive must obtain enough static and dynamic
  relation information for a unique IR, preferably without coupling the UDF to
  framework APIs.

The API and IR have different jobs. The API should be small and ergonomic. The
IR should be complete, strict, and optimizer-friendly.

The review assumes one frontend architecture: RayOrch native Torch-like DAG calls
are the main authoring surface, while wrappers, helpers, adapters, return shapes,
and return protocols are all syntax sugar for the same canonical IR. A primitive
is only acceptable if it can lower to the small IR algebra without adding a
frontend-specific special case.

## Relation Evidence Review

This prototype uses a different relation boundary from HYDP-dataflow, but that
does not mean UDFs should eagerly import RayOrch APIs. HYDP-style operators can
often stay almost pure UDFs because the framework can derive enough execution
meaning from table ports, column keys, and engine metadata. Multigrain object
pipelines cannot always do that: `Expand`, `Filter`, and `Relate` create
fine-grained record relations that are not recoverable from plain values alone.

The rule for each primitive is to prefer the least intrusive relation evidence
source that is sufficient:

```text
Preferred evidence sources:
  wrapper declaration
  forward helper
  business values
  nested groups
  boolean masks
  adapter hooks such as relation_fn / mask_fn / key_fn

Escape-hatch evidence:
  local parent references
  role names

Forbidden evidence:
  global record IDs
  lineage heads
  PortBatch construction
  quarantine records
  replay handles
  Ray/runtime internals
```

This turns the primitive review into a sharper question: does the API ask users
for the minimum semantic information needed to build a unique relation-aware IR?
If yes, the evidence source is part of the model. If no, the primitive is
leaking framework internals and should be redesigned.

## Primitive Set

The minimal logical set remains:

```text
Map       1:1      preserve aligned record identity
Expand    1:N      one parent produces zero or more children
Filter    1:0/1    intentionally drop records while preserving kept identities
Reduce    N:1      group descendants by an anchor and return anchor-grain rows
Relate    M:N      explicitly emit arbitrary local parent-child relations
```

Graph helpers are syntax-level relation expressions, not full user operators:

```text
group_by(anchor, *descendants)
zip_by_identity(*same_grain_ports)
```

`Select` should be treated as a friendly high-level API that lowers into
`Map + Filter + Project` or into one fused physical operator. It should not add a
new logical relation kind unless it eventually proves to be semantically distinct
from annotation plus filtering.

## Map

User shape:

```python
self.ocr = orch.Map(OCR)

texts = self.ocr(images)
texts, scores = self.ocr(images, layouts)
```

Semantic contract:

- All input ports must be at the same grain.
- Inputs align by logical record identity, not by physical row position.
- Every output preserves the same record identity and grain as the aligned input.
- Multiple outputs represent multiple columns over the same logical rows.

Canonical IR:

```text
IRNode.kind = MAP
contract.kind = MAP
relation = PRESERVE for every output
input_grains = (G, G, ...)
output_grains = (G, G, ...)
```

Why the API is elegant:

- Ordinary call syntax matches user intuition.
- Same-grain fan-in is implicit because it is the common case.
- Multi-output is natural Python unpacking.

Required validation:

- Reject cross-grain inputs unless a relation helper or relation primitive is
  used.
- Reject same-grain fan-in if identity domains are incompatible.
- After physical reordering, require identity alignment before invoking user
  code.

Corner cases:

- Multi-output `Map` must preserve one shared identity relation across all
  outputs.
- A `Map` that changes row count is incorrectly declared and should be rejected
  at runtime or by shape metadata.
- A `Map` with side effects should be marked in `OperatorProperties` so recovery
  does not blindly retry it.

Current prototype status:

- Implemented for eager local execution and symbolic tracing.
- IR records `PRESERVE` relation, operator recipe, properties, and physical
  hints.
- Verify pass checks same-grain fan-in.

## Expand

User shape:

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0, child_label="page")

pages, page_meta = self.pdf_to_pages(documents, meta)
```

Semantic contract:

- One declared parent input owns the generated child identity.
- Each parent row produces a child group of length zero or more.
- Every child has a stable parent relation and an ordinal or stable child key.
- Multiple outputs from one `Expand` share the same child relation by default.

Canonical IR:

```text
IRNode.kind = EXPAND
contract.kind = EXPAND
relation = EXPAND for every output
parent_input = declared parent index
ordinal = CHILD_INDEX or explicit stable key
physical.prefer_rebatch = true by default
```

Why the API is elegant:

- Cardinality is declared once in `__init__`.
- The forward path remains an ordinary call.
- `parent=...` is explicit only when needed and prevents hidden broadcast rules.

Required validation:

- `parent` must be in range.
- Multi-output `Expand` must either share one relation or explicitly declare
  multiple relations.
- If more than one input can plausibly own the child identity, use `Relate`
  instead of guessing.

Corner cases:

- `parent=1` with a side input must produce the same canonical IR as
  `parent=0`, except for the parent index.
- Independent expansions such as `pages` and `figures` should be separate nodes
  or a future multi-relation `Relate`.
- Empty child groups must still be visible to `Reduce` completion and replay
  logic.

Current prototype status:

- Implemented for eager local execution and symbolic tracing.
- Multi-output shared relation is enforced in eager mode.
- IR records `EXPAND`, `parent_input`, output ports, and rebatching hint.

## Filter

Preferred user shape:

```python
self.keep_good_pages = orch.Filter(ValidatePages)

good_pages, good_meta = self.keep_good_pages(pages, page_meta)
```

Semantic contract:

- Inputs must be same-grain and identity-aligned.
- Kept records preserve identity.
- Dropped records are business drops, not quarantine records.
- The selection mask is lineage-relevant metadata even if it is not returned to
  the user as a normal output.

Canonical IR:

```text
IRNode.kind = FILTER
contract.kind = FILTER
relation = FILTER for kept outputs
input_grains = (G, G, ...)
output_grains = (G, G, ...)
selection = mask sidecar or explicit predicate output
```

Why the API is tricky:

- In real data pipelines, filtering often also computes a score, reason, or
  annotation.
- A pure `Filter` can be elegant for simple validation but awkward when the
  predicate is useful downstream.

Recommended API split:

```python
self.score = orch.Map(ScorePages)
self.keep = orch.Filter(KeepHighQuality)

scores = self.score(pages)
good_pages = self.keep(pages, scores)
```

and a convenience API:

```python
self.select_good_pages = orch.Select(ScoreAndKeep)

good_pages, scores = self.select_good_pages(pages)
```

Lowering:

```text
Select
  -> Map(annotation / score)
  -> Filter(mask)
  -> Project(kept annotations)
```

or, after optimization:

```text
Map + Filter + Project
  -> one fused physical Select stage
```

Required validation:

- Dropped rows must not be reported as errors.
- Quarantine remains reserved for unexpected failures.
- Downstream `group_by` must distinguish "parent had no kept children" from
  "parent was never processed".

Current prototype status:

- `orch.Filter` is implemented for eager local execution and symbolic tracing.
- `orch.Select` is implemented as a high-level API that lowers to
  `Map + Filter + Project` in symbolic IR.
- Tests cover kept-identity preservation, business drops vs. quarantine, and
  canonical lowering.

## Reduce

User shape:

```python
self.assemble = orch.Reduce(AssembleDocument)

markdown = self.assemble(orch.group_by(documents, texts, page_meta))
```

Semantic contract:

- The first grouped argument is the anchor.
- Descendant ports are grouped by their ancestor relation to the anchor.
- Output normally returns to the anchor grain.
- Missing descendants follow an explicit missing-child policy.

Canonical IR:

```text
IRNode.kind = REDUCE
contract.kind = REDUCE
relation = REDUCE for every output
anchor = input_refs[0]
grouped = true
parent_input = 0
missing = fail_open / fail_closed / partial / retry_first
```

Why the API is elegant:

- `group_by(anchor, descendants...)` makes cross-grain fan-in visible exactly
  where it matters.
- The user does not manipulate row IDs or parent-child maps.
- The operation reads like a normal module call once the grouping expression is
  supplied.

Required validation:

- Reject ungrouped `Reduce` unless a carefully designed shorthand is enabled.
- Verify every descendant has a known ancestry path to the anchor.
- Verify output grain is the anchor grain unless explicitly declared otherwise.

Corner cases:

- Fail-open assembly is useful for partial documents, but the policy must be in
  IR so recovery and evaluation understand it.
- Reordered children must be sorted by ordinal or stable key before user reduce
  code sees them.
- Multi-output `Reduce` should share the same anchor relation unless declared
  otherwise.

Current prototype status:

- Implemented with explicit `group_by`.
- Prototype rejects direct `self.reduce(documents)` misuse with a readable error.
- Verify pass checks grouped reduce and anchor contract.

## Relate

Potential user shape:

```python
self.cluster = orch.Relate(ClusterChunks)

clusters = self.cluster(chunks)
```

or for multiple inputs:

```python
self.match = orch.Relate(MatchImagesAndCaptions)

pairs = self.match(images, captions)
```

Semantic contract:

- The user operator emits new records plus explicit local parent references.
- Outputs can relate to zero, one, or multiple parents from one or more input
  ports.
- Cardinality is arbitrary within an invocation: M:N, dedup, clustering, merge,
  pair generation, or graph construction.

Canonical IR:

```text
IRNode.kind = RELATE
contract.kind = RELATE
relation = RELATE
parents = explicit input refs
relation_schema = parent ports, local ids, optional role names
```

Why the API is the hardest:

- It is the escape hatch for everything that is not clean 1:1, 1:N, or N:1.
- Too much convenience here can hide expensive joins or ambiguous global
  matching.

Recommended constraint:

- MVP `Relate` should only express invocation-local relations.
- Global joins, shuffles, and cross-batch matching should be separate physical
  planning features with explicit partitioning or key semantics.

Required validation:

- Every emitted parent reference must point to a row visible in the invocation.
- Relation roles should be named when multiple parent ports have the same grain.
- Downstream `Reduce` must know whether the relation is ordered, unordered, or
  many-parent.

Current prototype status:

- `orch.Relate` is implemented for symbolic tracing.
- IR records `RELATE`, output grain, input refs, and optional relation roles.
- Eager and passive-IR execution support three relation-evidence tiers
  (see [`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md)):
  1. declarative `on={role: field}` key-join (inner equi-join; pure data in
     provenance; the common cross-branch case, no relation code in `forward`);
  2. by-ref `relation_adapter="pkg.mod:fn"` (dotted path, resolved at execute,
     serializable) for arbitrary non-equi relations;
  3. live `relation_fn` for local eager use (not serialized into the IR).
- The local MVP stores invocation-local parent refs in `PortBatch.relations`;
  `on=` key-join additionally merges both branches' ancestry so a downstream
  `Reduce` can regroup by an ancestor only one branch carried. Runtime checks
  reject out-of-range/undeclared parent refs. Distributed/global join and
  outer-join semantics remain future work.

## group_by Helper

User shape:

```python
markdown = self.assemble(orch.group_by(documents, texts, page_meta))
```

Semantic contract:

- This is not a data operator.
- It creates a relation expression: descendants should be routed by ancestor
  relation to the anchor.
- It must lower into `REDUCE` input routing metadata, not a separate physical
  stage unless a pass decides to materialize groups.

Canonical IR:

```text
REDUCE node:
  inputs = (anchor, descendants...)
  grouped = true
  anchor = input_refs[0]
```

Why the API is elegant:

- It only appears at the call site where implicit cross-grain fan-in would be
  dangerous.
- It avoids exposing internal IDs to users.

Required validation:

- Descendants must have lineage to the anchor.
- Multiple ancestry paths should be rejected or require an explicit path selector.

Current prototype status:

- Implemented in eager local execution and symbolic tracing.

## zip_by_identity Helper

Potential user shape:

```python
texts = self.ocr(orch.zip_by_identity(images, layouts))
```

or more likely implicit:

```python
texts = self.ocr(images, layouts)
```

Semantic contract:

- Ports are at the same grain and share the same logical identity domain.
- Physical row order may differ; alignment should happen by record identity.

Canonical IR:

```text
MAP node:
  inputs = same-grain ports
  relation = PRESERVE
  alignment = identity
```

Recommendation:

- Keep it implicit for common same-grain `Map` fan-in.
- Add explicit `zip_by_identity()` later only for diagnostics, disambiguation,
  or user-controlled alignment policy.

## Canonicalization Rules

The compiler should normalize equivalent user expressions into one IR shape:

- `Map` same-grain fan-in always becomes one `MAP` node with `PRESERVE`
  relations and identity alignment.
- `Expand` always records exactly one parent input per simple expanded relation.
- Multi-output `Expand` shares one relation by default.
- `Reduce(group_by(anchor, ...))` always becomes one `REDUCE` node whose anchor is
  `input_refs[0]`.
- `Select` lowers to `Map + Filter + Project` before optimization.
- Optimizer fusion changes physical nodes or physical hints, not the logical
  lineage contract.
- Materialization is an annotation on ports or logical boundaries, not a new
  user-visible data dependency.

These rules make the IR unique enough for testing: if two APIs mean the same
thing, their normalized `MultigrainIR.to_dict()` should match except for names
and source provenance.

## Priority Corner Case Tests

Already covered in the prototype:

- `Map` rejects cross-grain fan-in.
- `Expand` supports non-zero `parent`.
- Multi-output `Expand` records shared relation.
- `Reduce` requires `group_by`.
- IR exposes deps, consumers, relation contracts, physical hints, and
  materialization annotations.

Now covered (were previously "next"):

- `Filter` preserves kept identity and separates business drops from quarantine
  (`test_upper_primitives.py`).
- `Select` lowers into canonical `Map + Filter + Project` (`test_dummy_e2e.py`,
  `test_upper_primitives.py`).
- `Relate` emits explicit local parent references, plus `on=` key-join and
  by-ref adapter (`test_relate_key_join.py`, `test_upper_primitives.py`).
- Lineage is invariant under LPT cross-shard reordering, incl. fault isolation
  (`test_lineage_under_parallelism.py`).

Still to add:

- `Reduce` validates descendant ancestry to the anchor (explicit negative test).
- Same-grain fan-in aligns by identity after physical reorder (explicit test).
- Empty child groups and all-filtered parents remain visible to downstream
  completion/recovery.
- Multiple ancestry paths require explicit path selection.
- Canonicalization equivalence: different frontend sugars with the same meaning
  produce equal normalized `to_dict()`.

## Current Verdict

The current torch-like wrapper API is still the right main path:

```python
images = self.pdf_to_images(pdfs)
layouts = self.layout(images)
texts = self.ocr(images, layouts)
markdown = self.assemble(orch.group_by(pdfs, texts))
```

It is user-friendly because ordinary transformations look ordinary, and relation
helpers appear only where the graph would otherwise be ambiguous. The IR is also
moving in the right direction: it already records grains, relation kinds, parent
inputs, output specs, operator recipes, properties, physical hints, and
materialization.

The remaining design risk is concentrated in two areas:

- `Filter/Select`: the API must avoid forcing users to split natural
  score-then-filter logic into awkward boilerplate while still lowering to
  canonical relation-aware IR.
- `Relate`: the API must be powerful enough for M:N data governance without
  becoming an implicit join/shuffle system.

Those two primitives should drive the next prototype iteration.
