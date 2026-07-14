# Primitives：大原语、Select macro 与 output relation

## 1. 共同 UDF 契约

compiled graph 使用可导入 class：

```python
class NormalizePage:
    def __init__(self, size):
        self.size = size

    def run(self, pages):
        return [normalize(page, self.size) for page in pages]


op = mg.Map(NormalizePage, size=1024)
```

构造参数保存进 `OperatorFactorySpec`，实例在 executor/actor 内延迟创建。已构造 instance
仅用于 eager。

UDF 只接收 values，不接收 `record_id`、display key、ancestor、ordinal、lineage、
shard index 或全局 row index。行局部 UDF 必须只依赖对应 row values；batch adapter
必须满足本文和重排定理中的置换等变契约。

## 2. Core

大 primitive 是：

| Primitive | invocation scope | output relation |
|---|---|---|
| Map | aligned row-local 1:1 | `SameAs` |
| Filter | aligned row-local 0:1 | `SubsetOf` |
| Expand | parent-local 1:N | `ChildrenOf` |
| Reduce | group-complete N:1 | `AggregateOf` |
| Relate | cross-role M:N | `RelatedFrom` |

`Select` 是 authoring macro，不是新的 core relation。

## 3. Map

多 input Map 先按 record identity 对齐。UDF 返回 `num_outputs` 个等长 lists；每个 output：

- 保留 base record identity、display、ancestor 和 ordinal；
- 稳定合并 aligned branches 的 lineage 和 relation evidence；
- 追加当前 operation 名。

```python
self.detect = mg.Map(Detect, num_outputs=2)
boxes, scores = self.detect(images)
```

两个 outputs 是同一 records 的不同 value columns，graph 中都是
`SameAs(images_ref)`。

Map 是当前 attributable record recovery 支持最完整的 primitive。`BadRecordError.index`
是本次 invocation 的局部下标。

## 4. Filter

```python
class KeepReadable:
    def run(self, pages):
        return [page.quality >= 0.5 for page in pages]
```

mask 必须是与 input rows 等长的 strict bool list。多个 inputs 按 identity 对齐后使用
同一个 mask，每个 output 分别声明 `SubsetOf(对应输入)`。kept rows 的 identity 和
metadata 不变；业务过滤不是错误。

## 5. Select

Select 计算一个 mask 和若干 annotation columns：

```python
class ScoreAndKeep:
    def run(self, pages):
        scores = [score(page) for page in pages]
        return [value >= 0.5 for value in scores], scores


self.select = mg.Select(ScoreAndKeep, num_annotations=1)
kept_pages, kept_scores = self.select(pages)
```

compiled lowering 是：

```text
MapOp(mask + annotations)
  → FilterByMaskOp(mask_input=...)
```

Map outputs 使用 `SameAs`；FilterByMask 对原 inputs 和 annotations 产生 `SubsetOf`
outputs，并排除 mask port 本身。如果 inputs 数为 `I`、annotations 数为 `A`，UDF
outputs 为 `1 + A`，Select outputs 为 `I + A`。

## 6. Expand

```python
class SplitPages:
    def run(self, documents):
        return [split_document(document) for document in documents]


self.expand = mg.Expand(
    SplitPages,
    parent=0,
    child_label="page",
)
```

outer list 与 parent rows 等长，每项是 child list/tuple。每个 child：

- identity 由 operation、parent ID 和 child index 派生；
- ancestry 加入 direct parent；
- ordinals 加入 `child_label → child index` 对应层级；
- lineage 追加 Expand。

当前多输出 Expand 是 shared-child cohort：

```python
pages, metadata = mg.Expand(Parse, num_outputs=2)(documents)
```

所有 outputs 都声明相同 `ChildrenOf(parent, label)`，且每个 parent 在所有 outputs
上的 child 数必须相同，因此同位置 values 共享 child identity。`label` 命名 child
grain/display path；identity 由 parent identity 与 ordinal 派生。

### Mixed-output future

新的 relation algebra 和 verifier 已允许 Expand output 使用 `ChildrenOf` 或 `SameAs`，
并允许后一个 output 引用同 node 的更早 output，从而表示 identity forest。

但 `mg.out.same` / `mg.out.children` marker、trace-time marker analysis 和 runtime
materialization 都 **deferred/not implemented**。不要在当前代码中使用这些 API，也
不要把 IR 可表示性误写成 runtime 支持。

## 7. Reduce

```python
self.assemble = mg.Reduce(
    Assemble,
    missing_child=mg.IncompleteGroupPolicy.FAIL_CLOSED,
)

return self.assemble(mg.group_by(documents, texts, pages))
```

`group_by(anchor, *descendants)` 是声明。graph output relation 为：

```python
AggregateOf(
    anchor=document_ref,
    members=(texts_ref, pages_ref),
    incomplete=IncompleteGroupPolicy.FAIL_CLOSED,
)
```

runtime 根据 `ancestors[anchor.name]` regroup，并按从 anchor 层开始的完整 ordinal path
排序，最后以 record ID 作稳定 tie-break。UDF 收到 anchor values 和每个 descendant 的
grouped value lists，output 与 anchor 等长并复用 anchor identity。

`FAIL_OPEN` 使用存活 descendants；`FAIL_CLOSED` 不对 poisoned anchor 调 UDF，而是输出
incomplete placeholder 和 `suppressed_incomplete` error。

## 8. Relate

Relate 当前单输出、whole-batch。graph output 使用：

```python
RelatedFrom(
    roles=(
        RoleSource("image", images_ref),
        RoleSource("caption", captions_ref),
    )
)
```

### Key join

```python
self.link = mg.Relate(
    BuildPair,
    roles=("image", "caption"),
    on={"image": "doc_id", "caption": "doc_id"},
    output_grain="visual_pair",
)
```

它是 inner equi-join；重复 keys 产生完整 Cartesian product。compiled `on=` extractor
必须是 field name。

### Importable adapter

```python
self.link = mg.Relate(
    DiscoverRelations,
    roles=("image", "caption"),
    relation_adapter="my_project.relations:link_visual_refs",
)
```

adapter 返回：

```python
[(output_value, {"image": 3, "caption": 7})]
```

同一 parent tuple 多输出时加入 deterministic stable key。local indexes 仅在 invocation
内有效，框架立即解析为 `ParentRef` 和 content-addressed identity。

adapter 必须置换等变：任意 input row permutation 经 local-index rebinding 后，应产生
相同的 `(value, parent tuple, stable_key)` keyed set。不得把 local position 当业务证据。

`relation_fn` 仅允许 eager。compiled Relate 不接受 live `relation_fn` 或 executor-side
injection；必须使用 `on=`
生成 `KeyJoinSpec`，或使用 dotted `relation_adapter=` 生成 `RelationAdapterSpec`。

## 9. Worker 与 recovery 声明

所有大 primitive 接受：

```python
workers=mg.WorkerPoolSpec(replicas=4, gpus_per_worker=1.0)
recovery=mg.RecoveryPolicy(...)
```

worker pool 是 Ray placement 声明；recovery 是 retry/isolation policy。unsupported
operation/policy 组合显式失败，不会静默降级。

## 10. Runtime grain invariant

primitive output 进入 executor context 前必须满足 `PortBatch.name ==
OutputSpec.grain`：Map/Filter 保留 source grain，Expand 使用 child `label`，Reduce 返回
anchor grain，Relate 使用 `output_grain`。
