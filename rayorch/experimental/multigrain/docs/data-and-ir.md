# Data 与 IR：运行时记录和关系感知 ExecutionGraph

## 1. 运行时数据：PortBatch

一个 `PortBatch` 表示一个逻辑 port 上、同一 grain 的记录：

```python
PortBatch(
    name=...,
    values=...,
    record_ids=...,
    display_keys=...,
    ancestors=...,
    ancestor_display=...,
    ordinals=...,
    lineage=...,
    relations=...,
    errors=...,
)
```

除 batch-level `name` 和 `errors` 外，metadata 列与 `values` 等长。第 `i` 个位置的字段
共同描述同一记录。

| 字段 | 含义 |
|---|---|
| `values[i]` | UDF 可见的业务值 |
| `record_ids[i]` | 重排不改变的框架 identity |
| `display_keys[i]` | 仅用于 trace/debug 的可读名称 |
| `ancestors[i]` | ancestor port/grain 到 record ID，用于 Reduce 和故障归因 |
| `ancestor_display[i]` | ancestry 的可读投影 |
| `ordinals[i]` | 每层 child index，用于恢复逻辑顺序 |
| `lineage[i]` | operator 路径；fan-in 时稳定 union |
| `relations[i]` | M:N output 的 `ParentRef` tuple |
| `errors` | batch 携带的 `ErrorTrace` |

UDF 不接收这些 metadata。`record_id` 是正确性 identity，`display_key` 不能用于 join。

## 2. Trace-time token

`TracePort(ref, grain, tracer)` 位于 `multigrain/tracing.py`，只在
`Pipeline.forward()` tracing 中存在。最终 graph 只保存被动 ref：

```python
GraphInputRef(name="documents")
NodeOutputRef(node="OcrPage", output="out")
```

`PortRef` 是两者的 union。graph input 与 node output 不再用伪 node 名或 numeric output
index 混在一个类型中。

## 3. Per-output relation algebra

每个 `OutputSpec` 恰好携带一个 relation object：

```python
SameAs(source)
SubsetOf(source)
ChildrenOf(parent, label)
AggregateOf(anchor, members, incomplete)
RelatedFrom(roles=(RoleSource(role, source), ...))
```

### `SameAs`

输出复用 source 的 identity 和 grain。用于 Map outputs，以及未来 Expand
mixed-output forest 中的 aligned metadata。

### `SubsetOf`

输出是 source identity 的 0:1 子集，grain 不变。用于 Filter 和 Select lowering 的
`FilterByMaskOp`。

### `ChildrenOf`

输出为一个直接 parent 创建 children。`label` 命名 child grain 和 display path，不能为空；
child identity 由 parent identity 与 ordinal 派生，而不是由 label 本身充当 key。默认
多输出 Expand 的 outputs 使用相同的 `ChildrenOf` 并共享 child identity。

### `AggregateOf`

```python
AggregateOf(
    anchor=anchor_ref,
    members=(descendant_ref, ...),
    incomplete=IncompleteGroupPolicy.FAIL_OPEN,
)
```

Reduce 的 input 0 必须是 anchor，其余 inputs 必须按顺序等于 members；output grain 必须
等于 anchor grain。`FAIL_CLOSED` 会抑制丢失 descendant 的 anchor。

### `RelatedFrom`

`RoleSource` 把唯一、非空 role 名绑定到 node input。role sources 必须按 node input 顺序
完整覆盖。runtime `ParentRef` 保存每条实际 output 的 role-parent evidence。

## 4. Typed operations

graph 不保存通用 kind enum。`NodeSpec.operation` 是以下 typed operation 之一：

```python
MapOp(factory)
FilterOp(factory)
ExpandOp(factory)
ReduceOp(factory)
RelateOp(factory, matcher)
FilterByMaskOp(mask_input)
```

前五个对应大 primitive。`FilterByMaskOp` 只用于 Select lowering；它消费 mask input，
但 outputs 排除 mask port 本身。

用户 operator 的被动工厂是：

```python
OperatorFactorySpec(
    import_path="package.module.OcrPage",
    args=(),
    kwargs={"model": "model/path"},
)
```

compiled graph 不保存 live instance。compiled `RelateOp.matcher` 必须是由 `on=`
产生的 `KeyJoinSpec`，或由 dotted `relation_adapter=` 产生的
`RelationAdapterSpec`。`relation_fn` 仍是 eager-only，不能注入 compiled executor。

## 5. Graph dataclasses

```python
GraphInputSpec(ref=GraphInputRef(...), grain="documents")

OutputSpec(
    ref=NodeOutputRef("SplitPages", "out"),
    grain="page",
    relation=ChildrenOf(GraphInputRef("documents"), label="page"),
)

NodeSpec(
    name="SplitPages",
    inputs=(GraphInputRef("documents"),),
    outputs=(...,),
    operation=ExpandOp(...),
    workers=WorkerPoolSpec(...),
    recovery=RecoveryPolicy(...),
)

ExecutionGraph(
    name="PipelineName",
    inputs=(...,),
    nodes=(...,),
    outputs=(NodeOutputRef(...),),
)
```

`ExecutionGraph.nodes` 是 tracing 产生的拓扑顺序。`dependencies`、`consumers`、
`describe()`、`to_mermaid()` 和 `to_dict()` 都从 immutable graph 重算，不写回第二份
状态。

### Runtime grain invariant

executor 对每个 node output 强制：

```text
PortBatch.name == OutputSpec.grain
```

Map/Filter 保留 source grain；Expand 的 output grain 是 `ChildrenOf.label`；Reduce 回到
anchor grain；Relate 使用声明的 `output_grain`。这让静态 grain 不只是展示 metadata，
也成为 runtime boundary check。

## 6. Worker 与 recovery policy

```python
WorkerPoolSpec(replicas=4, gpus_per_worker=1.0)
```

它只描述 Ray worker 数和每 worker GPU allocation，不表达 batch size、engine 或其他
未实现的物理 hint。

`RecoveryPolicy` 保存 record/shard retry 与 isolation budget：

```python
RecoveryPolicy(
    max_record_retries=2,
    retry_timing=RetryTiming.INLINE,
    max_shard_retries=2,
    on_shard_exhausted=ShardExhaustedAction.DEGRADE,
    isolation=IsolationBudget(max_work_factor=3.0, max_calls=64),
)
```

不是所有 operation/policy 组合都实现；executor 在运行前显式拒绝 unsupported 组合。

## 7. Mandatory verification

`verify_graph(graph)` 是唯一公共验证入口。它在三个边界强制执行：

1. `GraphTracer.build()` / `Pipeline.compile()`；
2. `MultigrainExecutor.execute()`；
3. `MultigrainRayExecutor.execute_stream()`。

它验证：

- graph input 与 node/output 名唯一；
- refs 已由 graph input 或前序 node 产生；
- graph outputs 存在；
- operation 与 output relation 类型匹配；
- `SameAs` / `SubsetOf` grain 与 source 相同；
- `ChildrenOf.label` 非空且等于 output grain；
- Map outputs 精确 `SameAs(input 0)`，Filter/FilterByMask outputs 精确覆盖其数据 inputs；
- Reduce anchor/members 精确匹配 inputs；
- Related roles 唯一并按序覆盖 inputs；
- Select mask index 合法；
- node-local output relation 只能引用 invocation input 或更早 output，禁止 self/forward
  source。

因此 mixed-output identity forest 已可由 IR 表示并由 verifier 检查拓扑和 grain，但
runtime marker/materialization 仍未实现。

## 8. Derived capability

当前没有持久化 capability record，唯一查询是：

```python
is_row_partitionable(node)
```

| Operation | row partitionable |
|---|---:|
| Map / Filter / FilterByMask | yes |
| Expand | yes |
| Reduce | no |
| Relate | no |

这里 Expand 可分片指按完整 parent rows 分片，不是拆开单个 parent 的 child group。
Reduce 的 complete-group 需求属于 `AggregateOf` execution semantics，没有单独的
capability 查询。

## 9. Exact shard-plan validation

`validate_shard_plan(partitions, row_count)` 把 planner 输出规范化后检查：

- index 范围为 `[0, row_count)`；
- index 不重复；
- `0..row_count-1` 每个 index 恰好出现一次。

这把 reordering theorem 的“legal set partition”前提变成运行时检查，而不是 planner
约定。

## 10. 当前与 deferred

当前图中没有 pass manager、optimizer-inserted operation、graph-level rebatch 或
materialize node。`data.rebatch()` 仅对一个 `PortBatch` 做数据级重组。

未来 `mg.out.same` / `mg.out.children` 只用于 Expand mixed outputs。IR/verifier 已能表达
identity forest；authoring markers、restricted analysis 和 Local/Ray output materializer
仍 deferred。
