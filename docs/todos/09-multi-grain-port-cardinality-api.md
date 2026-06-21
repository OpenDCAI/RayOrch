# TODO: Multi-Grain Port and Cardinality API

Status: accepted design direction; implementation not started.

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

## Design Principles

- User operators remain ordinary Python classes with a normal `run()` method.
- Direct calls outside Runtime return ordinary usable business values.
- Type annotations describe business data types, not execution semantics.
- Decorators declare static port cardinality contracts for compilation.
- Small return helpers carry dynamic relation data for the current invocation.
- Runtime-only wrappers never escape the actor adapter into downstream user
  operators.
- Users never construct or observe global record IDs, lineage heads, or parent
  edges.
- One actor invocation still produces one Ray RPC result.

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

Static decorators therefore describe which ports may change grain:

```python
@orch.expand(outputs=0)
class Pdf2Pages:
    def run(self, documents, meta):
        page_groups = [decode(document) for document in documents]
        return orch.expanded(page_groups), meta
```

The two signals serve different purposes:

```text
@orch.expand(outputs=0)
  -> compile-time schema: output port 0 is expanding

orch.expanded(page_groups)
  -> runtime data: concrete group sizes and parent mapping
```

Runtime must cross-check them:

- a declared expanded port must return an expanded sidecar;
- an undeclared port must not return an expanded sidecar;
- all relation metadata must match actual output lengths.

The decorator does not wrap or replace `run()`.

## Proposed User API

### Record-Preserving `1:1`

No decorator or helper is required:

```python
class OCR:
    def run(self, pages, layouts):
        return texts, confidences
```

Each output port preserves the aligned input record identity. Multiple outputs
may hold different columns but remain at the same grain.

### Expanding `1:N`

```python
@orch.expand(outputs=0)
class Pdf2Pages:
    def run(self, documents, meta):
        page_groups = [decode(document) for document in documents]
        return orch.expanded(page_groups), meta
```

The ports have different grains:

```text
output 0 -> flattened page records with document parents
output 1 -> preserved document records
```

Multiple expanded ports are allowed:

```python
@orch.expand(outputs=(0, 1))
class ParseDocument:
    def run(self, documents):
        return (
            orch.expanded(page_groups),
            orch.expanded(figure_groups),
            document_meta,
        )
```

Each expanded port has an independent relation. Page and figure counts do not
need to match.

### Filtering `1:0/1`

Filtering should return the actual filtered business values during standalone
debugging while carrying only a lightweight selection sidecar in Runtime:

```python
@orch.filter(outputs=(0, 1))
class ValidatePages:
    def run(self, pages, meta):
        keep = [is_valid(page) for page in pages]
        return orch.selected(keep, pages, meta)
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

Reduce is different because grouping must happen before user code runs.

```python
@orch.reduce
class AssembleDocument:
    def run(self, documents, texts, captions):
        return [
            assemble(document, text_group, caption_group)
            for document, text_group, caption_group
            in zip(documents, texts, captions)
        ]
```

The first argument is the anchor grain:

```text
reduce(anchor, descendants...)
```

At runtime:

- anchor-grain inputs remain aligned `1:1`;
- descendant ports are independently grouped by their ancestry to the anchor;
- unrelated inputs are rejected;
- the output preserves the anchor record identity unless explicitly declared
  otherwise.

This distinguishes multiple expansion sources without scopes or IDs:

```python
return self.assemble(documents, page_texts, figure_captions)
```

Page texts and figure captions are grouped independently for each document.

Standalone debugging remains ordinary Python: the caller supplies already
grouped descendant columns.

### General `N:M`

Arbitrary batch-local transformations require an explicit relation:

```python
@orch.relate(outputs=0)
class ClusterChunks:
    def run(self, chunks):
        outputs, parent_groups = cluster(chunks)
        return orch.related(outputs, parents=parent_groups)
```

`parent_groups` uses invocation-local input positions, never global IDs:

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
N:M  -> arbitrary output-to-input parent sets
```

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
anchor_input=0
grouped_inputs=(1, 2, ...)
```

The compiler obtains these contracts from decorators and attaches them to the
node specification.

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
decorator            -> static cardinality contracts
runtime helper       -> concrete parent mapping
```

No `Grouped[T]` execution type or user-visible `PipeRef` is required.

### `RuntimeRayModule`

The public constructor remains focused on physical execution:

```python
RuntimeRayModule(
    Op,
    replicas=...,
    num_gpus_per_replica=...,
    max_inflight=...,
)
```

Users should not repeat cardinality configuration in `RuntimeRayModule`.
It reads the logical contract from the operator class and receives compiled
port specifications from the DAG compiler.

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

- `@orch.reduce` groups descendants to an anchor;
- `@orch.relate` provides an explicit relation;
- a future key-based join/shuffle routes independent datasets.

A normal operator receiving page and document ports without such a contract
must fail clearly rather than broadcast implicitly.

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
- Existing `1:1` operators require no decorator or helper.
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

## Initial Acceptance Workload

Use a multi-grain Flash-MinerU-shaped pipeline:

```python
@orch.expand(outputs=0)
class Pdf2Pages:
    def run(self, documents, meta):
        return orch.expanded(page_groups), meta


class OCR:
    def run(self, pages):
        return texts


@orch.reduce
class Assemble:
    def run(self, document_meta, text_groups):
        return markdown
```

Validate:

- `pages` and `document_meta` leave one node at different grains;
- direct operator calls return ordinary lists;
- the compiler knows the static port contracts;
- Runtime validates the dynamic expanded sidecar;
- page failures do not quarantine the whole document;
- reduce groups only healthy page descendants to each document anchor;
- empty groups remain visible to the reduce operator;
- different output ports are merged correctly across replicas;
- no per-record Ray RPC occurs;
- lineage can trace markdown to participating pages and source documents.

## Implementation Order

1. Introduce static cardinality contracts and decorators without changing
   execution.
2. Add invocation-local return helpers and standalone tests.
3. Convert executor storage from repeated node batches to real per-port batches.
4. Implement preserved and expanded mixed-grain outputs.
5. Implement lightweight filtering and drop metadata.
6. Implement anchor-based reduce routing.
7. Implement general batch-local `related()` parent maps.
8. Connect parent relations to partial replay and persistent lineage.
