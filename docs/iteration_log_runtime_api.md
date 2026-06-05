# Iteration Log: Runtime API and Mutation Tracking

Date: 2026-06-05

## Decision

Remove `mutates` from the MVP runtime path.

The MVP runtime tracks operator-level path lineage:

```text
row_id -> path_id
path_id -> parent_path_id + op
quarantine -> row_id + failed_op + previous_path + error
```

It does not track object-level mutation lineage or field-level diffs.

## Rationale

For record-level fault isolation, we only need to answer:

- which row failed;
- which op failed;
- which operators the row successfully passed before failure;
- whether healthy rows can continue.

This does not require knowing whether a Python object was modified in place.

If a user returns a value from an op, that value is part of the downstream dataflow:

```python
return images, meta
```

Whether `meta` was copied, rebuilt, or mutated in place is not needed for MVP path lineage.

## Future Scope

Mutation tracking may return later as an advanced feature for:

- fanout alias checks;
- copy-vs-inplace planning;
- field-level debugging;
- storage/materialization decisions.

If reintroduced, it should be optional and probably declared at the op/spec level. It should not be required for normal runtime execution.

## DAG Compile Inference

`RuntimeRayModule` now supports deferred specs:

```python
RuntimeRayModule(Pdf2ImgOp, replicas=4)
```

`DagPipeline.compile()` uses two lightweight sources:

- dynamic tracing remains the source of truth for DAG topology;
- `op.run` signature binding provides input names when possible;
- a small AST prepass reads direct assignments such as
  `images, meta = self.pdf2img(pdf, meta)` to provide output names.

These names are node-local labels. Reusing `a` in two unrelated assignments does
not merge their lineage or imply they are the same variable globally.

If no readable assignment exists, the compiler falls back to names like
`pdf2img.out0`. Explicit or annotated output arity is validated against simple
AST assignments, so mismatches fail at compile time rather than during a long
run.

## Runtime DAG Execution

`RuntimeDagExecutor` is the first runtime-aware DAG executor.

It accepts one source `MicroBatch` or a sequence of source `MicroBatch`
objects, executes compiled `RuntimeRayModule` nodes, and returns merged
`RuntimeResult` objects:

- final healthy `MicroBatch`;
- all quarantined records;
- lineage path deltas from every stage.

This keeps the generic DAG compiler and executor runtime-agnostic while letting
runtime stages use the `MicroBatch -> RuntimeResult` contract.

Current MVP scope:

- supports Ray actor replicas inside each `RuntimeRayModule`;
- supports pipeline-level overlap across multiple source microbatches with
  `max_batches_inflight`;
- supports Flash-MinerU-like linear DAGs with row-aligned outputs;
- requires aligned row IDs/path IDs for multi-parent joins;
- keeps generic `DagExecutor` and runtime execution separate.

Dummy timing on the local Ray conda environment:

```text
3-stage sleep DAG, 4 source microbatches, 0.35s/stage
runtime sequential: 4.509s
runtime overlap:    2.124s
generic DagExecutor overlap: 2.137s
```

The runtime overlap path is therefore in the same rough timing band as the
generic DAG scheduler for this dummy workload.

Flash-MinerU-like dummy timing:

```text
5 source microbatches x 8 docs = 40 docs
pdf2image replicas=2, 0.05s/doc
layout replicas=4, 2.0s/doc
ocr replicas=4, 2.0s/doc
output replicas=2, 0.05s/doc

generic DagExecutor: 25.007s
RuntimeDagExecutor: 24.458s
```

This benchmark runs 40 layout dummy inferences and 40 OCR dummy inferences, each
sleeping 2 seconds, and shows the runtime executor staying within the same
timing band as the generic DAG executor.

## Fault Isolation Under Overlap

`RuntimeDagExecutor` has a multi-batch inflight fault-isolation test:

```text
4 source microbatches
max_batches_inflight=4
pdf2img replicas=2
layout replicas=4
ocr replicas=4
output replicas=2
```

The test injects:

- one bad row in `pdf2img` for batch 1;
- one bad row in `ocr` for batch 2;
- one bad row in `pdf2img` and one bad row in `ocr` for batch 3.

Expected behavior is verified:

- healthy rows continue to final output;
- each batch receives only its own quarantine records;
- `pdf2img` failures have source paths;
- `ocr` failures trace back through `pdf2img -> layout`;
- other inflight microbatches are unaffected.
