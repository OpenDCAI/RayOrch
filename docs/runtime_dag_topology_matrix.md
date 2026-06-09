# Runtime DAG Topology Matrix

This matrix defines the document-level `1:1` lineage and error semantics covered
by `test/runtime/integration/test_dag_topologies.py`.

All tests share one session-scoped Ray cluster. Operators use one replica and no
artificial delay so the suite primarily measures semantics rather than startup
or throughput.

## Lineage Rules

- A normal node adds its operator name to each healthy row path.
- Fan-out copies the current path head into each branch.
- Fan-in aligns required branches by `row_id`.
- Diverged paths produce one internal multi-parent join node.
- Shared ancestors appear once when a path is traced.
- A quarantined row is absent from that branch's healthy output.
- Required fan-in uses the healthy row intersection in first-input order.
- The failed operator is stored on `QuarantineRecord.op`; its `path_id` points
  to the complete successful history before the failure.

## Covered Topologies

### Three-Way Fan-In

```text
              -> left ----\
source -> base -> middle ---+-> merge
              -> right ----/
```

Checks multi-parent lineage and shared-ancestor de-duplication.

### Nested Diamonds

```text
              -> left --\              -> tail_left --\
source -> base           -> merge ---------------------> final
              -> right -/              -> tail_right -/
```

Checks that lineage remains complete through repeated fan-out/fan-in.

### Source Rejoin

```text
source --------------------\
source -> derived ----------+-> merge
```

Checks that a source path and a derived path may be required by one node.

### Same-Node Multi-Output

```text
source -> split -> output 0 --\
                -> output 1 ---+-> merge
```

Both outputs share one lineage head, so Runtime must not invent a false branch.

### Common-Ancestor Failure

```text
source -> clean -> left --\
                -> right --+-> merge
```

A failure in `clean` is quarantined once before fan-out.

### Independent Branch Failures

```text
source -> left filter --\
source -> right filter --+-> merge
```

Different failed rows are removed independently. If the same row fails on both
branches, both error events remain observable.

### Empty Required Branch

```text
source -> reject all --\
source -> healthy ------+-> merge(empty)
```

The final result is an empty, schema-preserving batch with upstream quarantine
records intact.

### Failure At Fan-In

```text
source -> left --\
source -> right --+-> merge(fails)
```

The error path contains both successful branch histories, while the error
record identifies `merge` as the failing operator.

### Failure After Fan-In

```text
source -> left --\
source -> right --+-> merge -> sink(fails)
```

The merged multi-parent history survives into downstream quarantine records.

### Multiple Graph Outputs

```text
source -> left  -> graph output
source -> right -> graph output
```

The final `RuntimeResult` remains row-aligned and receives a combined lineage
head even without an explicit merge operator.

## Explicitly Deferred

These require the future emit and routing APIs rather than implicit `1:1`
alignment:

- document-to-page record expansion;
- normal filtering distinct from quarantine;
- group/reduce and many-to-one output identity;
- arbitrary `N:M` parent maps;
- shuffle, partitioning, and dataset-wide state;
- optional fan-in inputs and outer-join semantics.

Operator keyword arguments are an API concern rather than a topology shape.
Their compile-time and Ray execution contracts are covered separately by
`test_keyword_compile.py` and `test_keyword_arguments.py`.
