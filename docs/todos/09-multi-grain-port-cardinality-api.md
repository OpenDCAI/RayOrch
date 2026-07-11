# TODO: Multi-Grain Port and Cardinality API

Status: experimental MVP in progress under `rayorch.experimental.multigrain`.

## Goal

Support record-preserving, expanding, filtering, reducing, and general
many-to-many operators without exposing record IDs or lineage internals to
users.

The central abstraction is:

> Every output port owns its own record batch and cardinality relation.

A node may therefore return ports at different grains:

```text
port "pages"         -> page records
port "document_meta" -> document records
```

This replaces the current assumption that every output port of one node shares
the same `MicroBatch`.

## Frontend Sugar Target

The top-level authoring model should remain the RayOrch native Torch-like DAG:

```text
__init__   declares logical operator wrappers
forward    calls wrappers like ordinary modules
compiler   traces calls into relation-aware IR
```

Other API forms are not competing pipeline paradigms. They are complementary
front-end sugars that provide relation evidence for the same IR:

```text
wrapper declaration          self.op = orch.Expand(Op, parent=0)
forward relation helper      self.reduce(orch.group_by(anchor, children))
plain return shape           nested groups, masks, annotations
adapter hook                 relation_fn / mask_fn / key_fn for pure UDFs
return protocol              escape hatch for advanced relation-heavy operators
```

All of these should canonicalize into a small relation-aware IR algebra:

```text
Map / Expand / Filter / Reduce / Relate / Project
```

The design target is therefore:

> keep user-facing APIs flexible and ergonomic, but keep the IR small, unique,
> serializable, and optimizer-friendly.

## Design Principles

- User operators remain ordinary Python classes with a normal `run()` method.
- `Pipeline.forward()` should read like a Torch module graph: operators are
  module attributes and are called directly.
- Module wrappers such as `orch.Map`, `orch.Expand`, `orch.Filter`,
  `orch.Reduce`, and `orch.Relate` declare logical cardinality contracts.
- Direct calls to the underlying user operator outside Runtime return ordinary
  usable business values.
- Type annotations describe business data types, not execution semantics.
- Small return helpers carry dynamic relation data for the current invocation
  when static contracts are not enough.
- Runtime-only wrappers never escape the actor adapter into downstream user
  operators.
- Users never construct or observe global record IDs, lineage heads, or parent
  edges.
- One actor invocation still produces one Ray RPC result.

## Relation Contract Boundary: How This Differs From HYDP-dataflow

This design is intentionally not the same separation as HYDP-dataflow.

HYDP-dataflow optimizes for backend decoupling:

```text
business UDF
  -> receives neutral table/object batches
  -> returns neutral table/object batches
  -> framework owns ports, columns, execution backend, and most metadata
```

The important HYDP boundary is whether an operator touches backend-native
handlers such as `ray.data.Dataset`, Spark DataFrame, or disk handles. If it
does not, the UDF remains portable and mostly framework-agnostic. The IR can
infer or store port shape, column bindings, resource hints, and engine lowering
outside the UDF body.

RayOrch multigrain has a different hard problem. It is not enough to know that a
node has input and output ports. The runtime must also know the fine-grained
record relation created by the UDF:

```text
document -> pages
page -> kept page or business drop
pages -> document
image + caption -> matched pair
```

That relation cannot always be recovered from ordinary Python values after the
fact. A flat list does not say which parent produced each child. A filtered list
does not say whether missing rows are business drops or failures. A pair list
does not say which image and caption rows produced each pair.

The important conclusion is subtle:

> RayOrch multigrain does not have to give up framework-independent UDFs, but it
> cannot support fully relation-oblivious UDF contracts for all primitives.

The runtime needs dynamic relation evidence. That evidence can live in three
places:

```text
1. Wrapper declaration in __init__
   static contract: Map / Expand(parent=0) / Filter / Reduce / Relate roles

2. Relation expression in forward()
   routing contract: group_by(anchor, descendants), future zip_by_identity(...)

3. Ordinary UDF return shape or an external adapter
   dynamic evidence: nested groups, masks, local parent indexes, relation_fn(...)
```

These mechanisms are complementary front ends to the same IR. They should not
become separate pipeline paradigms. The top-level authoring style remains the
RayOrch native Torch-like DAG:

```text
__init__   declares logical operator wrappers
forward    calls wrappers like ordinary modules
compiler   traces calls into relation-aware IR
```

Inside that DAG, the framework should keep the relation-evidence surface as
small as possible:

```text
static wrapper contract      Map / Expand / Filter / Reduce / Relate
forward relation helper      group_by(...), future zip_by_identity(...)
plain return shape           nested groups, masks, annotations
adapter hook                 relation_fn / mask_fn / key_fn for pure UDFs
return protocol              escape hatch for advanced Relate-like cases
```

All of them must canonicalize into the same small IR algebra:

```text
Map / Expand / Filter / Reduce / Relate / Project
```

The API decision is therefore not "which style replaces the others." It is:
prefer the least intrusive front-end mechanism that can provide enough relation
evidence for a unique IR.

Therefore the strongest desirable boundary is not "the UDF must import or know
RayOrch". It is:

```text
UDF may remain framework-independent.
The pipeline must still provide enough static and dynamic relation information
to build a unique relation-aware IR.
```

The desired boundary is:

```text
User UDF:
  returns ordinary business values, or ordinary values with interpretable shape

Logical wrapper:
  orch.Map / Expand / Filter / Reduce / Relate declares static relation contract

Optional adapter:
  extracts dynamic relation evidence when plain return values are ambiguous

Runtime:
  creates internal record ids, parent edges, ordinals, lineage, replay metadata
```

This is a relation contract, not an internal metadata contract. Users may return
ordinary values, nested groups, masks, grouped inputs, or local relation
descriptors. Alternatively, the wrapper may receive an adapter such as
`relation_fn=` to interpret a completely framework-independent UDF output. Users
should not construct `PortBatch`, global record IDs, lineage deltas, quarantine
records, or replay handles.

The practical rule is:

```text
HYDP-dataflow:
  keep UDFs decoupled from execution backend and framework metadata

RayOrch multigrain:
  keep UDFs decoupled from internal ids and, when possible, from RayOrch APIs;
  require the pipeline contract to expose cardinality/relation semantics when
  values alone are ambiguous
```

Examples:

```python
class PdfToPages:
    def run(self, pdfs):
        # Nested lists are business values plus local 1:N shape.
        return [[page0, page1], [page0]]

self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

```python
class KeepPages:
    def run(self, pages, scores):
        # A bool mask is business validation semantics, not a failure signal.
        return [score >= 0.8 for score in scores]

self.keep_pages = orch.Filter(KeepPages)
```

```python
class MatchImagesAndCaptions:
    def run(self, images, captions):
        # This can remain a plain business return.
        return pairs

self.match = orch.Relate(
    MatchImagesAndCaptions,
    roles=("image", "caption"),
    relation_fn=lambda pairs: [
        (pair, {"image": pair.image_index, "caption": pair.caption_index})
        for pair in pairs
    ],
)
```

This controlled relation contract is central to the research value of the
system. The IR is not just a typed port DAG. It is a relation-aware pipeline
contract that makes lineage trace, row-level isolation, partial replay,
relation-aware rebatching, and M:N data governance optimizable without asking
users to manipulate internal runtime metadata.

## UDF Logic vs. IR Contract

The IR should not try to encode or understand all logic inside a user UDF. It is
not a Python AST, a model graph, or a business-rule engine. It should encode the
relation effect that the framework needs for correctness, tracing, recovery, and
optimization.

```text
IR must know:
  grain
  cardinality effect
  parent / anchor / relation roles
  child ordinal or stable key
  business drop vs. failure quarantine
  deterministic / side-effect / retryable properties

IR may ignore:
  OCR model internals
  PDF parsing details
  layout heuristics
  scoring formula
  matching model internals
  arbitrary business logic that does not affect relation semantics
```

This is the same separation as a type system or database logical plan: the IR
records the semantic contract needed by the optimizer and runtime, while the UDF
remains the implementation of business logic.

The implementation should use a three-layer contract model:

```text
1. Static declaration
   What the operator promises at compile time.

2. Dynamic evidence
   What the actual invocation returns or what an adapter extracts.

3. Runtime contract check
   Assertions that dynamic evidence satisfies the static contract.
```

Examples:

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

Static declaration:

```text
EXPAND, parent input = 0
```

Dynamic evidence:

```text
UDF returns nested groups, one group per parent row
```

Runtime checks:

```text
group count equals parent row count
multi-output groups share lengths unless declared otherwise
children receive parent relation and child ordinal
```

```python
self.keep_pages = orch.Filter(KeepPages)
```

Static declaration:

```text
FILTER, same-grain inputs, kept rows preserve identity
```

Dynamic evidence:

```text
UDF or adapter returns a boolean mask
```

Runtime checks:

```text
mask length equals input row count
mask values are bools
dropped rows are business drops, not quarantine records
kept rows preserve identity
```

```python
self.match = orch.Relate(
    MatchImagesAndCaptions,
    roles=("image", "caption"),
    relation_fn=extract_pair_parents,
)
```

Static declaration:

```text
RELATE, roles = image/caption, output grain = pair
```

Dynamic evidence:

```text
adapter extracts invocation-local parent references
```

Runtime checks:

```text
every parent reference points to an input row visible in this invocation
role names match declared input roles
relation ordering / multiplicity policy is explicit
no global runtime record ids are accepted from user code
```

This gives the system a local solution when UDF internals cannot be statically
bound into the IR: keep the UDF logic opaque, but check its observable relation
effect at runtime. A contract violation should be reported as a framework error
with the node name, expected contract, observed shape, and user-facing item
context when available.

## Preferred Programming Model: Torch-Like Calls + Internal Relations

The preferred surface API is a Torch-like module style:

```python
class MineruPipeline(orch.Pipeline):
    def __init__(self):
        self.pdf_to_images = orch.Expand(PdfToImages, parent=0)
        self.layout = orch.Map(Layout)
        self.ocr = orch.Map(OCR)
        self.assemble = orch.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        markdown = self.assemble(orch.group_by(pdfs, texts, page_meta))
        return markdown
```

`forward()` stays business-shaped: the user calls operators, not methods on
ports. The logical wrapper around each operator tells the compiler what
cardinality relation to expect. The internal runtime normalizes every call into
port-local batches plus a relation algebra:

```text
Port
  -> logical grain
  -> PortBatch at execution time
  -> parent relation to one or more input ports
  -> lineage heads for recovery and attribution
```

For the common PDF case:

```text
pdfs                 document grain
  -- Expand -->      images/page_meta at page grain, parent = pdfs
  -- Map -->         layouts/texts preserve page grain
  -- group_by -->    group page descendants by ancestor pdf
  -- Reduce -->      markdown returns to document grain
```

The user sees ordinary Python module calls. The system sees:

```text
Expand + Map + Map + GroupByAncestor + Reduce
```

This keeps the front end ergonomic while giving the runtime a formal relation
model for lineage, fault isolation, and future partial replay.

## DAG IR and Optimizer Passes

The Torch-like API should compile into an explicit operator DAG before physical
execution. This DAG is the main IR boundary:

```text
user forward()
  -> traced logical DAG
  -> relation-aware IR
  -> optimizer passes
  -> physical execution plan
```

The user-facing API may stay high-level and ergonomic, but the IR should remain
orthogonal and easy to optimize. For example, a future convenience API such as
`Select` can be represented as lower-level logical operations:

```text
Select(quality_then_keep)
  -> Map(score / annotation)
  -> Filter(mask)
  -> Project(kept annotations)
```

The runtime may still choose to execute this as one fused actor call. The
important separation is:

```text
logical IR:    precise, analyzable, relation-preserving
physical plan: fused, reordered, resource-aware
```

### IR Node Contract

Each compiled node should carry at least:

```text
node name
operator kind: Map / Expand / Filter / Reduce / Relate / helper
input ports
output ports
input and output grains
cardinality contract
parent input, for Expand
grouping anchor and descendants, for Reduce
logical output names
physical hints: replicas, resources, max_inflight, environment
```

This makes the DAG more than an execution order. It becomes the shared
representation for validation, lineage generation, scheduling, and lowering to
Ray actors.

### Initial Optimizer Passes

The first optimizer should be simple, explicit passes rather than a broad cost
based optimizer:

```text
ValidateCrossGrainFanIn
  reject ordinary Map calls that mix document/page/block grains without group_by

InsertRebatchAfterExpand
  insert child-grain rebatching before GPU-heavy Map stages

FuseMapFilter
  turn Map(score) + Filter(mask) into one physical Select stage

FuseAdjacentMaps
  fuse cheap consecutive Map stages at the same grain when safe

PlanReduceGroups
  choose how group_by(anchor, descendants) materializes groups and orders children

PlaceMaterializationBoundaries
  decide which ports need persistence for tracing, replay, or expensive recompute
```

Optimization should preserve logical lineage. A pass may change physical batch
layout or fuse operators, but it must not erase the logical operator names and
relations needed for trace views and partial replay.

### Design Rule

The API should be allowed to grow convenience wrappers, but the IR should stay a
small relation-aware algebra:

```text
API: friendly shortcuts for users
IR:  Map / Expand / Filter / Reduce / Relate plus explicit grouping metadata
```

This keeps future features such as fused `Select`, relation-aware rebatching,
and Ray actor lowering from turning into one-off special cases.

## Minimal Logical Primitives

The smallest useful primitive set is:

```text
Map       1:1      preserve the input record identity
Expand    1:N      one parent record produces zero or more child records
Filter    1:0/1    intentionally drop records while preserving kept identities
Reduce    N:1      group descendants by an anchor grain and produce anchor rows
Relate    M:N      explicitly attach outputs to one or more local parents
```

`flat_map` is `Expand`; `select` is `Filter`; `aggregate` is `Reduce`;
deduplication, clustering, merge, and multimodal pairing are `Relate` once their
input records are co-located in one invocation.

Two helper expressions are part of the graph syntax but are not operator
primitives:

```python
orch.group_by(anchor, *descendants)
orch.zip_by_identity(*ports)
```

`group_by()` makes cross-grain fan-in explicit. `zip_by_identity()` is usually
implicit when same-grain ports flow into a `Map`, but a named helper is useful
for diagnostics and cases where implicit alignment would be ambiguous.

## Minimal Graph Motifs

Most multimodal data-governance DAGs should be expressible as compositions of a
small number of motifs:

### Linear `Map`

```python
b = self.op(a)
```

`a` and `b` share the same grain and record identities.

### Fanout

```python
b = self.op1(a)
c = self.op2(a)
```

Several branches consume the same parent port.

### Same-Grain Fan-In

```python
d = self.op(b, c)
```

For a `Map`, all input ports must align by record identity. Otherwise the
compiler rejects the call.

### Expand

```python
children = self.expand(parent)
```

Each parent row produces a child group. The runtime stores a parent relation
instead of exposing child IDs to users.

### Filter

```python
kept = self.filter(records)
```

Kept rows preserve identity. Dropped rows are normal business drops, not
quarantine records.

### Anchor Reduce

```python
out = self.assemble(orch.group_by(parent, children))
```

Descendants are grouped by their ancestor in `parent`. The output normally
preserves the anchor grain.

### General Relate

```python
out = self.cluster(records)
```

`self.cluster = orch.Relate(...)` expects the user operator to return or emit
an explicit invocation-local parent relation.

These motifs cover the common shape:

```text
document -> pages/images/chunks/blocks
         -> enrich / score / OCR / VLM / filter
         -> group back to document or sample
```

## Runtime-Aware Return Helpers

Cardinality helpers detect whether they run inside a Runtime invocation, using
an invocation-local context such as `contextvars.ContextVar`.

Outside Runtime:

```python
orch.expanded(groups)
```

returns the ordinary flattened values.

Inside Runtime, the same call temporarily returns an internal sidecar:

```text
ExpandedPort(values, offsets)
```

The actor adapter immediately converts that sidecar into a port-local record
batch. It is never passed to another user operator.

This permits natural standalone debugging:

```python
pages, meta = Pdf2Pages().run(documents, meta)
```

and richer Runtime interpretation without maintaining two implementations of
the operator.

## Static and Dynamic Contracts

The compiler cannot discover `orch.expanded(...)` by executing `run()` because
compilation traces `Pipeline.forward()` with symbolic references. Real user
code may decode PDFs, iterate over concrete values, call native libraries, or
launch GPU inference.

Static module wrappers therefore describe the logical call:

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

Runtime helpers describe concrete relation data for the current invocation:

```python
class PdfToPages:
    def run(self, documents):
        page_groups = [decode(document) for document in documents]
        return orch.expanded(page_groups)
```

The two signals serve different purposes:

```text
orch.Expand(PdfToPages, parent=0)
  -> compile-time schema: output port is expanding from input port 0

orch.expanded(page_groups)
  -> runtime data: concrete group sizes and parent mapping
```

Runtime must cross-check them:

- a declared expanded port must return an expanded sidecar, or an equivalent
  value shape that the wrapper can normalize unambiguously;
- an undeclared port must not return an expanded sidecar;
- all relation metadata must match actual output lengths;
- multi-input expanding calls must declare which input port is the parent.

The wrapper does not change the underlying user's `run()` method. Standalone
debugging can still instantiate and call the operator directly.

## Proposed User API

### Record-Preserving `1:1`

Use `orch.Map`:

```python
self.ocr = orch.Map(OCR)

texts, confidences = self.ocr(pages, layouts)
```

Each output port preserves the aligned input record identity. Multiple outputs
may hold different columns but remain at the same grain.

### Expanding `1:N`

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)

pages, page_meta = self.pdf_to_pages(documents, meta)
```

The ports have different grains:

```text
output 0 -> flattened page records with document parents
output 1 -> page metadata at the same page grain, if the wrapper declares
            shared output relation
```

Multiple outputs from one `Expand` call should share one expanded relation by
default. This supports common pairs such as `(pages, page_meta)`. Independent
expansions should be written as separate logical nodes unless explicitly
declared:

```python
pages = self.extract_pages(documents)
figures = self.extract_figures(documents)
```

This avoids silently mixing unrelated offsets in one node.

### Filtering `1:0/1`

Filtering should return the actual filtered business values during standalone
debugging while carrying only a lightweight selection sidecar in Runtime:

```python
self.validate_pages = orch.Filter(ValidatePages)

valid_pages, valid_meta = self.validate_pages(pages, page_meta)
```

Outside Runtime this behaves like:

```python
filtered_pages, filtered_meta = ValidatePages().run(pages, meta)
```

Inside Runtime, the selection bitmap preserves record identity for kept rows
and creates lightweight drop metadata for rejected rows. Dropped payloads are
not returned through Ray RPC.

All ports selected by one helper share the same output record set. A separate
preserved-grain output must be returned outside that helper:

```python
return (*orch.selected(keep, pages, page_meta), document_stats)
```

### Reducing `N:1`

Reduce is different because grouping must happen before user code runs. Use
`orch.group_by()` at the call site when the input relation is cross-grain:

```python
self.assemble = orch.Reduce(AssembleDocument)

markdown = self.assemble(orch.group_by(documents, texts, captions))
```

The first argument to `group_by()` is the anchor grain:

```text
group_by(anchor, descendants...)
```

At runtime:

- anchor-grain inputs remain aligned `1:1`;
- descendant ports are independently grouped by their ancestry to the anchor;
- unrelated inputs are rejected;
- the output preserves the anchor record identity unless explicitly declared
  otherwise.

For simple cases the wrapper may allow a shorthand:

```python
markdown = self.assemble(documents, page_texts, figure_captions)
```

but only when `orch.Reduce(..., anchor=0)` is declared and all non-anchor inputs
are descendants of that anchor. Otherwise `group_by()` is required.

Standalone debugging remains ordinary Python: the caller supplies already
grouped descendant columns.

### General `M:N`

Arbitrary batch-local transformations require an explicit relation:

```python
self.cluster = orch.Relate(ClusterChunks)

clusters = self.cluster(chunks)
```

Inside the user operator, `parent_groups` uses invocation-local input positions,
never global IDs:

```python
[
    [0, 3],
    [1, 2, 4],
]
```

For multiple input ports, parent groups are keyed by input parameter:

```python
return orch.related(
    samples,
    parents={
        "chunks": chunk_groups,
        "figures": figure_groups,
    },
)
```

Command-style `ctx.emit()` may remain an advanced syntax for dynamic control
flow, but it must finalize into the same batch relation and must not issue
per-record RPCs.

## Unified Physical Relation

Every output port is normalized to:

```text
PortBatch
  columns or value
  record_ids
  lineage_heads
  parent relation
```

The parent relation can use a CSR-like representation:

```text
parent_offsets
parent_indices
parent_ports
```

All cardinalities become special cases:

```text
1:1  -> output i has parent input i
1:N  -> several outputs share one input parent
1:0  -> no output is produced for a rejected input
N:1  -> one output has several input parents
M:N  -> arbitrary output-to-input parent sets
```

## First-Class Runtime Benefits From the API

The API is not only a nicer way to spell cardinality. Once every port carries a
grain and relation contract, the runtime can use the same metadata as an
execution control plane.

Two benefits should be first-class system mechanisms:

1. automatic lineage trace views for debugging, observability, and user-facing
   error reports;
2. relation-aware physical rebatching/reordering after `1:N` expansion to reduce
   GPU bubbles.

### Automatic Lineage Trace View

The runtime should turn internal lineage and parent relations into a readable
diagnostic view:

```text
ErrorTrace
  source item: pdf_path=paper-a.pdf
  logical item: page=17
  failed op: OCR
  grain: page
  upstream path: PdfToImages -> Layout -> OCR
  parent document: paper-a.pdf
  downstream impact: Assemble(markdown for paper-a.pdf)
  action: quarantined / retried / skipped / partial replay
```

Users should not inspect global `record_id`, `path_id`, or parent edge tables.
Instead, the runtime should derive display identities from source values,
logical port names, and local ordinals:

```text
document=paper-a.pdf
document=paper-a.pdf/page=17
document=paper-a.pdf/page=17/block=3
```

This requires each relation-changing operator to preserve enough metadata for
human-readable trace reconstruction:

- source label columns or display keys;
- child ordinal within its parent group, unless the operator provides a stronger
  business key such as page number;
- failed operator name and logical port;
- current grain of the failed invocation;
- upstream parent chain and downstream impacted anchors.

The intended user experience is:

```python
result.errors.to_table()
result.trace(error)
result.trace_item(document="paper-a.pdf", page=17)
```

The exact API can change, but the capability should be part of the design:

```text
lineage relation -> item-level diagnosis -> actionable recovery explanation
```

### Relation-Aware Rebatching and Reordering

`1:N` expansion often creates highly imbalanced child counts:

```text
pdf0 -> 2 pages
pdf1 -> 80 pages
pdf2 -> 1 page
pdf3 -> 40 pages
```

If downstream GPU stages keep the original document-shaped batch layout, fast
replicas finish early while others process long documents. The `Expand`
relation lets the runtime flatten child-grain records, form dense child batches,
and later group them back by ancestor:

```text
documents
  -> Expand(document -> pages)
  -> flatten/rebatch/reorder page records
  -> dense GPU Map stages over pages
  -> group_by(document, page_outputs)
  -> Reduce back to document grain
```

The logical DAG is unchanged. Only the physical batch layout changes:

```text
logical relation: document -> page -> OCR text
physical layout:  page batches balanced across GPU replicas
restore:          group_by(document, OCR text) using parent relation
```

This is the main performance payoff of making `1:N` explicit. Parent relations
are not just lineage metadata; they permit the runtime to remove bubbles while
still reconstructing the original document grouping.

Required invariants:

- rebatching must preserve child ordinals or stable child keys so `Reduce` can
  restore deterministic order;
- same-grain fan-in after reordering must align by identity, not by accidental
  physical position;
- `group_by(anchor, descendants)` must tolerate missing, filtered, quarantined,
  or retried children according to the reduce policy;
- relation metadata must move with records through dispatch and replica collect;
- bounded reorder scopes must define when an anchor's children are complete.

Useful reorder scopes:

```text
within one microbatch     simplest and safest
across K microbatches     better load balance with bounded buffering
dataset/global            requires shuffle, barriers, or stateful routing
```

The MVP should start with within-microbatch or bounded K-microbatch reordering.
Dataset-wide routing is a separate physical execution problem.

## Required Type and API Changes

### `MicroBatch`

Current `MicroBatch` represents all output columns of a node and assumes one
shared row set. It should either:

1. become the port-local batch type; or
2. be renamed internally to `PortBatch`.

The important invariant is:

> All values inside one port batch share one record set and grain.

Different output ports may have different lengths, identities, and lineage.

### `RuntimeResult`

Current:

```python
RuntimeResult(batch=MicroBatch(...))
```

Direction:

```python
RuntimeResult(ports=(PortBatch(...), PortBatch(...)), ...)
```

Compatibility may provide:

```python
result.batch
```

only when a result has one port, or when all final ports share one record set
and can safely form a column batch. Ambiguous mixed-grain results must require
explicit port access instead of silently aligning incompatible records.

### Executor Context

The executor currently stores one `MicroBatch` and repeats it for every output
slot:

```text
ctx[node] = tuple(result.batch for each output)
```

It must instead store the actual port batches:

```text
ctx[node] = result.ports
```

`PipeRef(node, index)` already identifies an output port, so the compiled DAG
reference model does not need a fundamental redesign.

### `NodeSpec` and `RuntimeNodeSpec`

Each output needs a static cardinality contract:

```text
PRESERVE
EXPAND
FILTER
RELATE
```

Reduce is primarily an input routing contract:

```text
kind=REDUCE
anchor_input=0, if shorthand calls are allowed
grouped_inputs=(1, 2, ...), if the call site used group_by()
```

The compiler obtains these contracts from logical module wrappers such as
`orch.Map`, `orch.Expand`, `orch.Filter`, `orch.Reduce`, and `orch.Relate`, then
attaches them to the node specification.

### Compiler and AST Naming

The existing AST assignment:

```python
pages, document_meta = self.pdf2pages(documents, meta)
```

continues to provide output names. Type annotations continue to provide
business types.

Cardinality metadata is separate:

```text
AST assignment       -> port names
type annotations     -> business value types
module wrapper       -> static cardinality contracts
call-site helper     -> grouping or explicit alignment intent
runtime helper       -> concrete parent mapping for this invocation
```

No `Grouped[T]` execution type or user-visible `PipeRef` is required.

### Logical Wrappers and `RuntimeRayModule`

The public `RuntimeRayModule` constructor currently focuses on physical
execution:

```python
RuntimeRayModule(
    Op,
    replicas=...,
    num_gpus_per_replica=...,
    max_inflight=...,
)
```

The new logical wrappers should not force users to repeat cardinality in
`RuntimeRayModule`. There are two plausible integration shapes:

```python
self.ocr = orch.Map(OCR, replicas=4, num_gpus_per_replica=1)
```

or:

```python
self.ocr = orch.Map(RuntimeRayModule(OCR, replicas=4, num_gpus_per_replica=1))
```

The first is more ergonomic; the second keeps physical execution explicit and
closer to the current implementation. This is an open API boundary decision.
In both cases, the compiled node must carry both:

```text
logical contract: Map / Expand / Filter / Reduce / Relate
physical contract: replicas, resources, max_inflight, environment
```

The actor adapter must normalize every returned port independently.

### Dispatch and Replica Collection

Input dispatch remains row-based for ordinary and expanded port batches.

Replica result collection must merge matching output ports independently:

```text
replica 0 port 0 + replica 1 port 0 -> merged port 0
replica 0 port 1 + replica 1 port 1 -> merged port 1
```

It must not assume that different ports from the same replica have equal row
counts.

### Fan-In

Ordinary `1:1` fan-in remains strict record-identity alignment.

Different-grain inputs may meet only through an operator whose contract
explains the relationship:

- `orch.group_by(anchor, *descendants)` groups descendants to an anchor;
- `orch.Relate(...)` provides an explicit relation;
- a future key-based join/shuffle routes independent datasets.

A normal operator receiving page and document ports without such a contract
must fail clearly rather than broadcast implicitly.

### Corner Cases and Required Checks

#### Multi-Input `Expand`

For:

```python
crops = self.crop(images, layouts)
```

`self.crop = orch.Expand(Crop)` is ambiguous. The wrapper must declare which
input owns the child records:

```python
self.crop = orch.Expand(Crop, parent=0)
```

If multiple inputs can serve as parents, the operator is no longer a simple
`Expand`; it should be modeled as `Relate`.

#### Multi-Output `Expand`

For:

```python
images, page_meta = self.pdf_to_images(pdfs)
```

the default should be one shared expanded relation for both outputs. This means
`images[i]` and `page_meta[i]` describe the same page record. If one operator
returns pages and figures with independent offsets, it should either be split
into two logical nodes or declared as an advanced multi-relation `Relate`.

The shared relation must include a child ordinal or stable child key. Downstream
rebatching may change physical order, but `group_by(pdfs, images)` must be able
to reconstruct page order for document-level reduce.

#### Cross-Grain `Map`

For:

```python
out = self.op(documents, pages)
```

if `self.op = orch.Map(Op)`, the compiler must reject the call unless the input
ports are aligned by identity. It must not implicitly broadcast document rows to
page rows.

After relation-aware rebatching, even same-grain inputs may arrive in different
physical orders. `Map` fan-in must therefore align by record identity; physical
position is only an optimization when identity order is already known to match.

#### Reduce Shorthand

For:

```python
markdown = self.assemble(documents, texts)
```

the shorthand is allowed only if `self.assemble = orch.Reduce(Assemble,
anchor=0)` and all non-anchor inputs are descendants of the anchor. Otherwise,
users should write:

```python
markdown = self.assemble(orch.group_by(documents, texts))
```

When descendants have been physically reordered across microbatches, the reduce
runtime must know when an anchor is complete. Within-microbatch reduce can use
the local expand relation. Bounded or global reordering needs watermarks,
end-of-parent markers, or another completion protocol.

Reduce policy should also define how missing children are handled:

```text
fail-open    assemble with missing/error metadata
fail-closed  quarantine the whole anchor
retry-first  run child-grain partial replay before reduce
partial      emit output plus structured warnings
```

#### Filter vs Quarantine

Filtering is intentional data curation. Quarantine is execution failure. They
may both remove rows from the healthy path, but they need separate metadata and
recovery policies.

#### Whole-Node Commit

One actor invocation returns one result. A node with multiple output ports should
commit all its port relations together. Per-port partial commits require a
separate future commit protocol.

### Drops and Quarantine

Normal filtering and execution failure remain distinct:

```text
drop       -> intentional removal with lightweight metadata
quarantine -> execution failure for one record
```

Both should default to references and metadata rather than embedding large
image, tensor, or document payloads in RPC responses.

### Failure and Retry Semantics

`BadRecordError(index=...)` remains local to the records presented to the
current operator invocation:

```text
document-grain op -> index identifies a document
page-grain op     -> index identifies a page
reduce op         -> index identifies an anchor group
```

A reduce failure therefore quarantines or retries the affected anchor group by
default. It does not implicitly identify one child inside that group. Precise
child isolation should normally happen in an upstream child-grain operator.
A future advanced error may carry an invocation-local child reference, but it
must not expose global IDs.

Split-and-retry must preserve the current invocation grain:

- a page batch is split by pages;
- an anchor-based reduce batch is split by complete anchor groups;
- an expanding operator is retried by source input records, not by partially
  produced child values.

An expanding operator that wants to preserve partial successful children from
a failed source record needs an explicit `related()` or buffered emit protocol.
The basic `expanded()` contract is atomic per source input record.

System-level retry must replay the complete set of output ports from one node
invocation. A successful page port and a failed document port from the same
attempt must not be committed independently unless a future commit protocol
explicitly allows per-port commits.

### Lineage

Path lineage remains useful for record-preserving operator history, but
cardinality changes additionally require explicit record-parent edges.

The internal model therefore separates:

```text
execution path lineage
record parent relation
```

Both are hidden from the normal user API and can later drive impact analysis
and partial replay.

## What Does Not Change

- User operators still expose `run()`.
- Operators can still be instantiated and debugged without Ray.
- `Pipeline.forward()` remains a normal business DAG.
- `PipeRef(node, index)` remains the internal output-port reference.
- Existing `1:1` operators require no relation helper beyond the logical
  `Map` wrapper.
- Type annotations remain normal application types.
- Ray RPC remains one call in and one result out per actor invocation.
- Runtime physical settings remain on `RuntimeRayModule`.
- Existing `BadRecordError(index=...)` remains invocation-local, although the
  represented grain may now be document, page, block, or reduce anchor.

## Explicitly Separate Problems

This API describes relations among records already present in one actor
invocation. It does not itself implement:

- cross-microbatch group-by;
- global deduplication;
- distributed joins;
- repartition or shuffle;
- windows, barriers, or dataset-wide state.

Those features require routing and recovery semantics above the port relation
model. Once records are co-located, their local output relation can still use
the same `related()` representation.

## Prototype Questions Not Yet Settled

The current proposal is intentionally focused on the local multi-grain API. The
following design questions still need explicit decisions before implementation:

### Wrapper and Physical Runtime Boundary

Should users write:

```python
self.ocr = orch.Map(OCR, replicas=4)
```

or compose logical and physical layers explicitly:

```python
self.ocr = orch.Map(RuntimeRayModule(OCR, replicas=4))
```

The first is friendlier. The second makes the current `RuntimeRayModule`
lifecycle easier to preserve. The decision affects API stability, docs, and how
physical settings are surfaced.

### Operator Body Return Shape

For `Expand`, should user operators return ordinary nested groups:

```python
return page_groups, page_meta_groups
```

and let `orch.Expand` normalize them, or should they always use:

```python
return orch.expanded(page_groups, page_meta_groups)
```

The former is cleaner for users. The latter is easier to validate and supports
standalone debugging with fewer hidden conventions.

### Shared vs Independent Multi-Output Relations

The proposal defaults multi-output `Expand` to a shared relation. We still need
a syntax for rare independent multi-output expansions, or a policy that such
operators must be split into multiple logical nodes.

### Reduce API Surface

`orch.group_by(anchor, *descendants)` is explicit and safe. `Reduce(anchor=0)`
shorthand is concise. The docs should decide which one is the recommended path
and which one is merely allowed.

### Relate Ergonomics

`Relate` is the completeness escape hatch for `M:N`, but writing parent index
lists is low level. We still need a user-friendly pattern for common cases:

- local deduplication;
- clustering;
- merge by local key;
- multimodal sample pairing.

`ctx.emit()` may be the advanced form, but common `Relate` workloads should not
force users into record-by-record command code.

### Global Routing Boundary

The relation algebra can represent local parent edges after records are
co-located. It does not decide how global deduplication, joins, shuffle,
windows, or stateful indexes route records. Those operators need separate
physical execution semantics.

### Replay and Determinism Contracts

For lineage-guided partial replay, each operator eventually needs declarations
for determinism, side effects, materialization safety, and replay policy. This
document only defines the relation API that makes replay scopes computable.

## Initial Acceptance Workload

Use a multi-grain Flash-MinerU-shaped pipeline:

```python
class MineruPipeline(orch.Pipeline):
    def __init__(self):
        self.pdf_to_images = orch.Expand(PdfToImages, parent=0)
        self.layout = orch.Map(Layout)
        self.ocr = orch.Map(OCR)
        self.assemble = orch.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        return self.assemble(orch.group_by(pdfs, texts, page_meta))
```

Validate:

- `images` and `page_meta` leave one node at page grain with shared parent
  relation to `pdfs`;
- child ordinals or stable child keys survive dispatch, rebatching, and
  collection;
- direct operator calls return ordinary lists;
- the compiler knows wrapper-declared static port contracts;
- Runtime validates concrete dynamic relation metadata;
- page failures do not quarantine the whole document unless the reduce policy
  chooses to fail closed;
- failure traces identify the source document, child item, failed operator,
  upstream path, and impacted anchor;
- page-grain rebatching reduces idle time for imbalanced PDF page counts;
- reduce groups only healthy page descendants to each document anchor;
- empty groups remain visible to the reduce operator;
- different output ports are merged correctly across replicas;
- cross-grain `Map` calls fail unless explicitly grouped or related;
- same-grain fan-in after rebatching aligns by record identity;
- no per-record Ray RPC occurs;
- lineage can trace markdown to participating pages and source documents.

## Implementation Order

1. Introduce logical module wrappers (`Map`, `Expand`, `Filter`, `Reduce`,
   `Relate`) and compile-time cardinality contracts without changing execution.
2. Add symbolic relation helpers (`group_by`, optional `zip_by_identity`) and
   compiler checks for illegal cross-grain fan-in.
3. Add invocation-local return helpers and standalone tests for `Expand`,
   `Filter`, and `Relate`.
4. Convert executor storage from repeated node batches to real per-port batches.
5. Implement preserved and expanded mixed-grain outputs with shared relation
   multi-output support.
6. Add display identity metadata and an automatic item-level trace view for
   quarantine and runtime errors.
7. Implement lightweight filtering and drop metadata.
8. Implement identity-based same-grain fan-in after physical reordering.
9. Implement within-microbatch relation-aware rebatching for `1:N` outputs.
10. Implement anchor-based reduce routing through `group_by`, including child
    ordering and missing-child policy.
11. Implement bounded K-microbatch rebatching only after an anchor-completion
    protocol is defined.
12. Implement general batch-local `related()` parent maps and/or buffered emit.
13. Connect parent relations to partial replay and persistent lineage.
