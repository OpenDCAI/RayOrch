# Expand 混合输出与 identity forest

状态：**relation vocabulary 已预留；marker、forest verifier 与 runtime 整体 deferred**。

本文记录未来 `mg.out.same` / `mg.out.children` 的边界。当前代码不能使用这些 marker；
现有多输出 Expand 仍是 shared-child cohort。

## 1. 问题

默认多输出 Expand：

```python
pages, page_metadata = mg.Expand(ParsePages, num_outputs=2)(documents)
```

两个 outputs 当前都：

- 使用同一个 `ChildrenOf(document)`；
- 每个 document 的 group length 相同；
- 共享 child identity、ancestry 和 ordinal；
- 只是同一批 child records 的不同 value columns。

不可拆分的底层调用有时同时产生：

```text
pages             children(document)
page_metadata     same(page)
document_metadata same(document)
images            children(document)
blocks            children(page)
```

把它们强制视为一个 cohort 会产生错误 identity；无条件拆 UDF 又可能重复 decode/model
工作。

## 2. 已实现的类型基础

relation algebra 已分别定义必要的 per-output identity 类型：

```python
OutputSpec(..., relation=SameAs(source))
OutputSpec(..., relation=ChildrenOf(parent))
```

但当前类型名为 `ExecutionGraph`，`verify_graph()` 因而只接受 runtime 真正可执行的
shared-`ChildrenOf` Expand。`ExpandOp + SameAs`、不同 parent 或不同 child grain 都在
验证期拒绝，不产生“验证通过、执行失败”的半成品 graph。未来实现 marker 时，source
顺序、grain、无环 forest 与 runtime materialization 必须一起落地。

## 3. Future API

未来只考虑两个小 output marker：

```python
mg.out.same(values, as_=source)
mg.out.children(groups, of=source, grain=..., key=None)
```

- `same` lower 为 `SameAs(source_ref)`；
- `children` lower 为 `ChildrenOf(source_ref)`；marker 的 `grain=` 写入
  `OutputSpec.grain`，identity 由 source identity 与 ordinal 派生；
- 未标注 output 保持默认 shared-child Expand 行为。

它们只描述 output identity source，不引入万能 Transform/emit DSL。

## 4. Identity forest

允许的 edge 只有：

```text
same      1:1，复用 source identity
children  1:N，从 source 派生 child identity
```

组合可表示：

```text
document
├── same → document_metadata
└── children → page
    ├── same → page_metadata
    └── children → block
        └── same → block_score
```

约束：

1. 每条 output record 有且仅有一个直接 identity source；
2. source 是 invocation input 或同 return bundle 的前序 output；
3. graph 无环，无 forward reference；
4. relation 在 invocation 内确定；
5. cardinality 是 1:1 或 1:N。

## 5. 大 primitive 边界

大 primitive 决定 invocation scope：

- `Map`：aligned row-local 1:1；
- `Filter`：aligned row-local 0:1；
- `Expand`：parent-local 1:N；
- `Reduce`：group-complete N:1；
- `Relate`：cross-role M:N。

`Select` 是 `MapOp → FilterByMaskOp` authoring macro；FilterByMask 消费且排除 mask
port，只输出 filtered inputs 和 annotations。

`mg.out.*` 仅解决 Expand 不可拆 invocation 的 output identity forest，不表达：

- 0:1 selection 或 routing；
- N:1 aggregation；
- M:N / multiple logical parents；
- global shuffle/dedup/rank；
- window/streaming；
- external asynchronous workflow。

## 6. Target examples（尚不可运行）

以下代码是 future API design，不是当前示例。

### Parent metadata

```python
pages = mg.out.children(parsed.pages, of=documents, grain="page")
document_metadata = mg.out.same(parsed.metadata, as_=documents)
return pages, document_metadata
```

### Sibling-aligned metadata

```python
pages = mg.out.children(page_groups, of=documents, grain="page")
page_metadata = mg.out.same(page_meta_groups, as_=pages)
return pages, page_metadata
```

### Independent cohorts

```python
pages = mg.out.children(page_groups, of=documents, grain="page")
images = mg.out.children(image_groups, of=documents, grain="image")
return pages, images
```

### Nested fanout

```python
pages = mg.out.children(page_groups, of=documents, grain="page")
blocks = mg.out.children(block_groups, of=pages, grain="block")
scores = mg.out.same(score_groups, as_=blocks)
return pages, blocks, scores
```

## 7. Deferred implementation work

### Authoring/tracing

- define typed `mg.out.same` / `mg.out.children` return markers；
- perform restricted return-bundle analysis without executing the UDF；
- resolve marker sources to `GraphInputRef` / `NodeOutputRef`；
- emit per-output `SameAs` / `ChildrenOf` relations；
- preserve unmarked shared-child compatibility。

### Runtime

- typed wrappers carry values plus invocation-local source evidence；
- materialize outputs in source dependency order；
- `same` validates shape and reuses source metadata；
- `children` validates one group per source record and derives child metadata；
- tuple materialization fails atomically；
- eager, Local and Ray share one implementation；
- no expansion of record-level recovery scope。

Current `verify_graph` and `ExpandHandler` both accept only one shared
`ChildrenOf` relation.

## 8. Reordering obligations

Before runtime support can be called complete:

- parent-row sharding must remain exact-cover validated；
- `same` must preserve source keyed records；
- `children` identity must derive from source identity plus stable child
  ordinal/key, never emission order；
- all output shape checks must be shard-order independent；
- every materialized output must satisfy `PortBatch.name == OutputSpec.grain`；
- the reordering theorem and property tests must cover mixed-output forests。

## 9. Historical rationale

Earlier design discussions considered a generic Transform, arbitrary emit schema,
extra subset/route markers, and execution-stage materialization. They were rejected
to keep the public model small. The retained decision is: five big primitives,
Select as a macro, and only two future Expand output markers.

## 10. Action

Do not implement marker/runtime work without a concrete workload requiring one
indivisible heterogeneous return bundle. Until then:

- use normal shared-child multi-output Expand when group shapes match；
- split operations when outputs have different invocation requirements；
- use Filter, Reduce or Relate for their own relation families；
- describe mixed forests as a fully deferred feature, not a currently valid
  `ExecutionGraph`。
