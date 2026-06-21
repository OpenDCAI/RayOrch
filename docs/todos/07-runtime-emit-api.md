# TODO: Batch Emit API for Record Cardinality Changes

Status: future design direction. Do not block the current document-level
record-preserving Runtime MVP.

The default cardinality and mixed-grain port API is now refined in
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md).
This document remains focused on the advanced command-style `ctx.emit()` path
and its shared buffered execution protocol.

## Goal

Support filter, flat-map, deduplication, merge, clustering, and general `N:M`
parent relationships while preserving microbatch execution and one Ray result
per actor call.

The design principle is:

```text
batch execution
record-level semantics
local buffered emit
single-result Ray RPC
```

## Execution Model

```text
actor receives MicroBatch
  -> create RuntimeEmitContext and local EmitBuffer
  -> user op processes the microbatch
  -> ctx.emit()/ctx.drop() append local metadata
  -> ctx.finalize() builds one ExecResult
  -> actor returns once
  -> scheduler advances downstream nodes
```

`ctx.emit()` must not perform RPC, object-store writes, or immediate downstream
dispatch.

## Advanced User API

```python
@orch.emit_op(outputs=("doc",), record_local=False)
class DedupOp:
    def __call__(self, docs):
        ctx = orch.ctx()
        seen = set()

        for src, doc in ctx.inputs(docs):
            key = normalize(doc)
            if key in seen:
                ctx.drop(parent=src, reason="duplicate")
            else:
                seen.add(key)
                ctx.emit(parent=src, values={"doc": doc})
```

The `src` value is an opaque invocation-local input reference. Users should not
construct record IDs or lineage nodes directly.

## Buffered Helpers

The single-record API remains readable:

```python
ctx.emit(parent=src, values={"doc": doc})
ctx.drop(parent=src, reason="duplicate")
```

Batch helpers should avoid Python-level append overhead for common patterns:

```python
ctx.emit_batch(values=..., parents=...)
ctx.emit_filter(values=..., parents=..., mask=...)
ctx.emit_flatmap(values=..., parents=..., offsets=...)
```

All helpers append to the same actor-local buffer and finalize once.

## Runtime Result Direction

The finalized result should contain:

```text
healthy output MicroBatch
multi-parent lineage delta
normal drop records
quarantine records
```

Normal filtering and execution failure must remain distinct:

```text
emit       -> produced record
drop       -> intentionally filtered record
exception  -> quarantined record
```

## Scheduling Boundary

Emit describes output records and parent relationships within one actor call.
It does not decide which records reach that call.

Dataset-wide deduplication, group-by, clustering, or joins may additionally
require:

```text
partitioning
shuffle
stateful actors
barriers or windows
```

Those scheduling primitives should be designed separately from `EmitBuffer`.

## Preparation Required Now

The current `1:1` Runtime should avoid assumptions that would block emit:

- lineage nodes may have multiple parents;
- lineage storage should be append-only and mergeable;
- user-visible paths remain opaque IDs;
- Runtime result metadata should tolerate future drop and emission deltas;
- fan-in should align records by identity rather than require identical paths;
- nested page/block values must not be confused with independent Runtime rows.

Do not expose `record_id` or `alignment_id` to users. Static cardinality
decorators, runtime-aware return helpers, and per-port batches are specified in
the multi-grain port design. Shuffle APIs remain deferred until a real
dataset-global workload defines their semantics.

## Initial Acceptance Workload

When implemented, validate with a batch-local dedup operator:

- one microbatch enters one actor call;
- duplicate records are reported as normal drops;
- unique records retain exact parent lineage;
- no per-record Ray RPC occurs;
- one finalized result returns from the actor.
