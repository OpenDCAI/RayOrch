# TODO: Runtime DAG Fan-In and Output Naming

Status: document-level fan-in implemented; deterministic direct-return naming
remains open.

## Scope Decision

The immediate target is record-preserving execution:

```text
one healthy input document -> one healthy output document
one failed input document  -> one quarantine record
```

This is record-level `1:1` with exceptional `1:0`. It does not mean a node may
have only one DAG input or one output column. A node may consume several
branches and return several columns as long as all columns represent the same
aligned document records.

Pages, blocks, images, and text spans remain nested values inside a document
record for this phase:

```python
list[document]
list[list[page]]
list[list[block]]
```

Page-level scheduling and arbitrary record cardinality changes are deferred to
the multi-grain port API tracked in
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md).
Advanced command-style emission remains tracked separately in
[`07-runtime-emit-api.md`](07-runtime-emit-api.md).

## Problem 1: Fan-In After Path Divergence

A realistic Flash-MinerU OCR stage may consume both the original images and
layout results:

```python
images, meta = self.pdf2image(pdf, meta)
layouts, meta = self.layout(images, meta)
texts, meta = self.ocr(images, layouts, meta)
```

The Runtime aligns required branches by `row_id`. Different `path_ids` are
expected and produce an internal multi-parent join lineage node.

## Implemented Semantics

A fan-in should:

- align required inputs by `row_id`;
- pass only rows present in every required input branch;
- preserve every parent lineage path used by the node;
- create one output path that references the complete parent set;
- keep quarantine records traceable through the relevant input branches;
- distinguish an already quarantined row from a healthy aligned row;
- reject duplicate or otherwise ambiguous row identities explicitly.

Lineage changed from a single-parent edge:

```text
path_id -> parent_path_id + op
```

to a multi-parent edge:

```text
path_id -> parent_path_ids + op
```

The current `path_id` remains one opaque string per record. At fan-in, the
Runtime creates an internal join lineage node with all distinct upstream path
heads as parents. The user op continues to receive only normal batch columns.

This representation intentionally prepares for future emit outputs without
adding record-cardinality APIs to the current MVP.

## Remaining: Direct Return Output Names

The compiler can infer a readable output name from:

```python
markdown = self.markdown(texts, meta)
return markdown
```

But a direct return:

```python
return self.markdown(texts, meta)
```

falls back to a generated name such as `markdown.out0`. Runtime callers then
cannot reliably assume:

```python
result.batch.columns["markdown"]
```

This behavior is valid internally but surprising in the public API.

## Desired Direction

Define a deterministic naming rule for direct-return calls. Candidates:

- use the DAG attribute/node name for a single output: `markdown`;
- retain `node.out0` only for unnamed multi-output calls;
- allow an explicit compile-time override when semantic names matter.

The rule must distinguish node identity from variable identity and remain
stable when the same `RuntimeRayModule` instance is reused at multiple DAG
nodes.

## Acceptance Tests

- A Runtime OCR node can consume aligned `images`, `layouts`, and `meta` from
  different paths without losing lineage.
- A row already absent from one required branch does not enter the fan-in node.
- A bad OCR row reports both the failed node and its complete upstream history.
- Required branches use their healthy `row_id` intersection in stable order.
- Duplicate `row_ids` fail before actor execution.
- `return self.markdown(...)` produces a documented, deterministic final column
  name.
- Assigned and direct-return forms compile to equivalent topology and types.
- The API tour can use the realistic Flash-MinerU fan-in without a workaround.
