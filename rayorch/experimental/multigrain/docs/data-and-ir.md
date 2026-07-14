# Data 与 IR：字段、关系和 Capability

本文定义 multigrain 的术语和字段语义。修改这里列出的 dataclass、enum、relation
推导或 verifier 时，应同步更新本文。

## 1. 运行时数据：PortBatch

一个 `PortBatch` 表示一个逻辑 port 上、同一 grain 的一组记录：

```python
@dataclass
class PortBatch:
    name: str
    values: list[Any]
    record_ids: list[str]
    display_keys: list[str]
    ancestors: list[dict[str, str]]
    ancestor_display: list[dict[str, str]]
    ordinals: list[dict[str, int]]
    lineage: list[tuple[str, ...]]
    relations: list[tuple[ParentRef, ...]]
    errors: list[ErrorTrace]
```

除 `name` 和 batch-level `errors` 外，其余非空列必须与 `values` 等长。第 `i` 个位置的
所有字段共同描述一条记录。

### 1.1 字段定义

| 字段 | 作用 | 稳定性/约束 | UDF 可见性 |
|---|---|---|---|
| `name` | 当前 port/grain 的运行时名称 | batch 内唯一语义名称 | 不直接传入 |
| `values[i]` | 用户业务值 | 可由 UDF 改变 | 可见 |
| `record_ids[i]` | 框架内部逻辑 identity | 重排不改变；同 port 内应唯一 | 不可见 |
| `display_keys[i]` | 人可读 identity | 用于 trace/debug，不应用于 join 正确性 | 不可见 |
| `ancestors[i]` | `ancestor_grain → record_id` | 用于 Reduce regroup 和故障归因 | 不可见 |
| `ancestor_display[i]` | `ancestor_grain → display_key` | 仅用于可读 trace | 不可见 |
| `ordinals[i]` | `parent_grain → child_index` | 按层插入，用于重排后恢复顺序 | 不可见 |
| `lineage[i]` | 经过的 operator 名称路径 | Map/Filter 等追加，diamond 时稳定 union | 不可见 |
| `relations[i]` | M:N 输出的 role parent refs | Relate 输出通常非空 | 不可见 |
| `errors` | batch 携带的隔离/级联错误 | 下游传播；不是逐行平行列 | 不可见 |

### 1.2 为什么 identity 与 display 分开

`record_id` 是机器语义。例如：

```text
SplitPages:documents:0:2
```

`display_key` 是用户语义。例如：

```text
paper.pdf/page=2
```

显示名称可能重复或变化，不能承担 join/recovery identity。内部 ID 不应暴露给 UDF，
否则 UDF 可能依赖物理顺序或内部实现，破坏重排不变性。

### 1.3 ParentRef

M:N relation 无法只用单一 ancestor 表达，因此使用：

```python
@dataclass(frozen=True)
class ParentRef:
    role: str
    port: str
    record_id: str
    display_key: str
```

例如 image-caption pair：

```text
relations = (
  ParentRef(role="image", port="images", record_id="images:4", ...),
  ParentRef(role="caption", port="captions", record_id="captions:7", ...),
)
```

### 1.4 ErrorTrace

`ErrorTrace` 是对一个失败逻辑项的用户可读解释：

- `source_item`：源输入；
- `logical_item`：失败记录；
- `failed_op`：失败 operator；
- `grain`：失败所在 grain；
- `upstream_path`：操作路径；
- `parent`：可读父项；
- `action`：如 `quarantined`、`suppressed_incomplete`；
- `error`：原始错误；
- `ancestors`：用于下游把失败映射回 anchor。

`Reduce(FAIL_CLOSED)` 正是通过 `ErrorTrace.ancestors[anchor.name]` 找到需要抑制的输出。

## 2. Trace-time token 与 passive port

### 2.1 SymbolicPort

`SymbolicPort` 只在 `Pipeline.forward()` tracing 中存在，包含 live `GraphTracer`。
它不能进入最终 IR。

### 2.2 IRPortRef

```python
IRPortRef(node="OcrPage", port="out", index=0)
```

它是 node output 的稳定地址。输入 port 的 node 名以 `__input__` 开头。

### 2.3 IRPortSpec

`IRPortSpec` 在 ref 之外保存：

- `grain`：port 上记录的逻辑粒度，是 verifier 判断 preserve/expand/reduce 合法性的核心
  类型信息；
- `name`：用户可读的 port 名；与 `IRPortRef` 的机器地址分离，便于 trace 和错误展示；
- `payload`：`OBJECT / TABLE / TENSOR`，描述 value 的物理表示类别，为未来 engine lowering
  选择 transport 或 batch format 留出入口；
- `display_key` specification：声明如何生成用户可读 key，本身不保存运行时 key 值；
- `ordinal_key` specification：声明 child order 的所属 parent grain 与 key，供重排后恢复
  逻辑顺序；
- `materialize` policy：声明该 port 是否需要 debug、failure 或 checkpoint 物化，不直接
  持有 storage handle。

这些都是被动描述，不持有 builder、UDF 或 runtime value。

## 3. IRNode

一个 `IRNode` 包含：

```python
IRNode(
    name=...,
    kind=...,
    input_specs=...,
    output_specs=...,
    contract=CardinalityContract(...),
    op=OperatorRecipe(...),
    properties=OperatorProperties(...),
    physical=PhysicalHints(...),
    recovery=RecoveryPolicy(...),
    parent_input=...,
    grouped=...,
)
```

### 3.1 NodeKind

用户 primitive：

- `MAP`：保持 grain 与 identity，只转换每条记录的业务值；
- `FILTER`：保持 grain，从 identity-aligned records 中选择子集；
- `EXPAND`：把一个 parent record 展开为零到多个 child records；
- `REDUCE`：把 descendants 按 anchor regroup，返回 anchor grain；
- `RELATE`：根据 role evidence 组合多个 ports，产生一般 M:N relation rows。

内部 node：

- `PROJECT`：只投影/透传指定 ports；主要承接 Select lowering 的最终输出整理；
- `REBATCH`：声明物理重组 batch、不改变逻辑 relation；当前 handler 仍是 identity
  pass-through；
- `MATERIALIZE`：声明 trace/replay/checkpoint 边界；当前 IR 已表达意图，但 storage
  backend 尚未接入。

`Select` 没有独立 NodeKind，它 lower 成 `MAP → FILTER → PROJECT`。

### 3.2 OperatorRecipe

```python
OperatorRecipe(
    cls_ref="package.module.OcrPage",
    args=(),
    kwargs={"model": "model/path"},
    provenance={},
)
```

- `cls_ref`：必须指向可导入 operator class；executor 在目标 process/actor 内据此加载
  class，避免在 driver 序列化模型实例；
- `args/kwargs`：原样保存用户传给 operator constructor 的参数，让每个 replica 可以
  独立重建等价 UDF；
- `provenance`：保存不属于通用 IR 字段的 primitive-specific passive metadata，例如
  Expand 的 `child_label`、Reduce 的 `missing_child`、Relate 的 `on`，供 handler 精确
  恢复 wrapper 声明。

live instance 只允许 eager；compiled graph 不序列化实例状态。

### 3.3 OperatorProperties

| 字段 | 当前含义 |
|---|---|
| `deterministic` | 相同值输入是否产生相同值输出 |
| `side_effect` | 是否具有外部副作用 |
| `idempotent` | 重复执行是否安全 |
| `retryable` | operator 是否允许恢复系统重试 |
| `expensive` | 是否值得考虑物化/缓存 |
| `gpu_heavy` | 是否是 GPU 重阶段 |
| `stateful` | operator 是否跨调用持有语义状态 |

这些字段是逻辑/执行约束声明。当前 capability 推导尚未完整消费所有字段，因此不能仅凭
`row_partitionable=True` 推断任意 stateful UDF 都安全。

### 3.4 PhysicalHints

| 字段 | 含义 |
|---|---|
| `replicas` | 期望并行副本数 |
| `num_gpus_per_replica` | 每个副本的 GPU 数 |
| `max_inflight` | operator 级并发提示 |
| `batch_size` | 期望物理 batch size |
| `prefer_rebatch` | 是否倾向在此输出后插入 rebatch |
| `engine` | 可选执行引擎提示 |

这些是 hint，不应改变逻辑结果、identity 或 lineage。

## 4. CardinalityContract 与 RelationSpec

`CardinalityContract` 是 node 级契约：

```python
@dataclass(frozen=True)
class CardinalityContract:
    kind: NodeKind
    input_grains: tuple[str, ...]
    output_grains: tuple[str, ...]
    relations: tuple[RelationSpec, ...]
```

每个 output 对应一个 `RelationSpec`：

```python
@dataclass(frozen=True)
class RelationSpec:
    output: IRPortRef
    relation: RelationKind
    parents: tuple[IRPortRef, ...]
    roles: tuple[str, ...] = ()
    parent_input: int | None = None
    anchor: IRPortRef | None = None
    ordinal: OrdinalPolicy = PRESERVE
    missing: MissingChildPolicy = FAIL_OPEN
```

### 4.1 RelationSpec 字段

| 字段 | 精确定义 |
|---|---|
| `output` | 本 relation 描述的 node output |
| `relation` | 输出与输入的关系 family |
| `parents` | 参与产生该 output 的所有 input refs，顺序与 node inputs 一致 |
| `roles` | Relate 中每个 input 的角色名 |
| `parent_input` | Expand 的主要 parent 在 inputs 中的索引 |
| `anchor` | Reduce 返回的目标 grain/input |
| `ordinal` | 输出是否保留 ordinal，或由 child index 新建 ordinal |
| `missing` | Reduce 对缺失 descendant 的语义 |

### 4.2 RelationKind

| RelationKind | 逻辑含义 |
|---|---|
| `PRESERVE` | output 与 input records 保持 identity |
| `FILTER` | output 是同 identity input 的子集 |
| `EXPAND` | 每个 output child 属于一个 parent |
| `REDUCE` | descendants 按 anchor regroup，output 回到 anchor grain |
| `RELATE` | output 同时引用多个 role parents |
| `GROUPED` | 表达显式 grouped relation 的保留枚举值；当前 primitive tracing 不直接产生 |

### 4.3 OrdinalPolicy

- `PRESERVE`：保持输入 ordinal；
- `CHILD_INDEX`：Expand 为每个 parent 下的 child 追加 index；
- `NONE`：不声明顺序关系。

运行时 nested Expand 使用 insertion-ordered ordinal dict 表示完整层级路径。Reduce 从
anchor grain 对应位置开始排序，并用 record ID 作为稳定最终 tie-break。

### 4.4 MissingChildPolicy

- `FAIL_OPEN`：从存活 children 继续 Reduce，可能得到部分结果；
- `FAIL_CLOSED`：不对受污染 anchor 调用 UDF，生成 suppressed placeholder 和 error；
- `RETRY_FIRST`、`PARTIAL`：已进入被动接口，但当前执行语义没有覆盖所有组合；
  unsupported 组合必须显式拒绝，不能静默降级。

## 5. Primitive → RelationSpec

| Primitive/internal node | relation | parent_input | anchor | ordinal | missing |
|---|---|---:|---|---|---|
| Map | `PRESERVE` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Filter | `FILTER` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Select map | `PRESERVE` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Select filter | `FILTER` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Project | `PRESERVE` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Expand | `EXPAND` | 用户指定索引 | `None` | `CHILD_INDEX` | `FAIL_OPEN` |
| Reduce | `REDUCE` | `0` | `inputs[0]` | `PRESERVE` | 用户策略 |
| Relate | `RELATE` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |
| Rebatch/Materialize | `PRESERVE` | `None` | `None` | `PRESERVE` | `FAIL_OPEN` |

## 6. Capability 是派生执行视图

```python
@dataclass(frozen=True)
class PrimitiveCapabilities:
    identity_alignment: bool
    group_completion: bool
    row_partitionable: bool
    relation_evidence: RelationEvidenceFamily
```

### 6.1 字段定义

| Capability | 含义 |
|---|---|
| `identity_alignment` | 多输入是否可按相同 record ID 逐行对齐 |
| `group_completion` | 执行前是否必须收齐 anchor 对应的完整 descendants |
| `row_partitionable` | 是否允许把输入记录分成独立 row shards |
| `relation_evidence` | output builder 需要哪类关系证据 |

Evidence family：

- `ALIGNED`：同 identity 输入；
- `PARENT`：一个明确 parent 和 child ordinal；
- `ANCHOR`：anchor 与完整 descendant groups；
- `ROLE`：多个 role parent refs；
- `INTERNAL`：没有已知用户 relation family。

### 6.2 完整推导矩阵

| Relation set | identity_alignment | row_partitionable | group_completion | evidence |
|---|---:|---:|---:|---|
| `{PRESERVE}` | true | true | false | `ALIGNED` |
| `{FILTER}` | true | true | false | `ALIGNED` |
| `{PRESERVE, FILTER}` | true | true | false | `ALIGNED` |
| `{EXPAND}` | false | true | false | `PARENT` |
| `{REDUCE}` | false | false | true | `ANCHOR` |
| `{RELATE}` | false | false | false | `ROLE` |
| 空或混合未知 family | false | false | false | `INTERNAL` |

以 Expand 为例：

```text
RelationSpec:
  relation = EXPAND
  parent_input = 0
  ordinal = CHILD_INDEX

Derived capability:
  identity_alignment = false
  row_partitionable = true
  group_completion = false
  relation_evidence = PARENT
```

这里 `row_partitionable=true` 表示可以按完整 parent rows 分片，不表示可以把一个 parent
产生的 child group 从中间拆开。

### 6.3 为什么 Capability 不写入 IR

`RelationSpec` 是语义事实；Capability 是可重复计算的执行结论。如果两者同时序列化，
可能出现矛盾：

```text
relation = REDUCE
row_partitionable = true
```

因此执行器调用：

```python
caps = capabilities_for(node)
```

而不是信任第二份持久字段。未来可以在 verified execution plan 中缓存结果，但不应把它
变成原始 IR 的第二事实来源。

### 6.4 推导前提与当前限制

Capability 安全性还依赖：

1. IR 已通过 structural verification；
2. UDF 满足 value-purity/row-independence 契约；
3. shard planner 保持 aligned inputs 同分片；
4. Expand 按 parent row 分片；
5. stateful/side-effect properties 没有否定重试和重排安全性。

当前 `capabilities_for()` 主要从 relation family 推导。后续收紧时应组合
`RelationSpec + OperatorProperties + verified invariants`，而不是把结果写回 IR。

## 7. Structural verification

`VerifyPass` 首先检查通用结构：

- node name 唯一；
- input ref 已由 graph input 或前序 node 产生；
- graph output 存在；
- contract kind 与 node kind 一致；
- contract grains 与 input/output specs 一致；
- 每个 output 恰有一个 relation；
- relation outputs 与 node outputs 顺序一致；
- relation parents 与 node inputs 顺序一致；
- 一个 node 不混用多个 relation family。

再检查 family-specific 约束：

- aligned family 输入 grain 必须相同，输出 grain 必须保持；
- Expand 必须有合法且统一的 `parent_input`；
- Reduce 必须显式 grouped、anchor 为 input 0、输出回到 anchor grain；
- Relate roles 数必须等于 input 数。

新增 handler 前必须先让 IR 满足这些结构不变量。

## 8. RecoveryPolicy 也是被动 IR

```python
RecoveryPolicy(
    max_record_retries=0,
    retry_timing=RetryTiming.INLINE,
    max_shard_retries=2,
    on_shard_exhausted=ShardExhaustedAction.ABORT,
    isolation=IsolationBudget(
        max_work_factor=3.0,
        max_calls=64,
        on_exhausted=IsolationExhaustedAction.QUARANTINE,
    ),
    drain_scope=DrainScope.STAGE_GLOBAL,
)
```

声明进入 IR，但并非每种 primitive/policy 组合都已实现。executor 必须在运行前明确拒绝
unsupported 组合，不能自动改成更弱语义。

## 9. MultigrainIR

最终图包含：

- named graph inputs；
- topologically ordered immutable nodes；
- graph output refs；
- materialization specs。

它提供：

- `topo_order`：按 graph dependencies 返回稳定拓扑 node 序列，供 pass 和 executor 遍历；
- `deps`：`node → direct upstream nodes`，用于 readiness 与结构分析；
- `consumers`：`output ref → downstream nodes`，用于优化 pass 和 fan-out 分析；
- `describe()`：输出面向开发者的节点、grain、relation 与 hint 摘要；
- `to_mermaid()`：生成 DAG 可视化文本，帮助检查 branch、fan-in 和 lowering 结果；
- `to_dict()`：把 passive dataclass/enum 递归转换为可序列化结构，供持久化和测试比较。

`MultigrainIR` 是逻辑计划，不是 execution plan。actor pool、handler instance、
runtime capability cache 和 `PortBatch` context 均属于执行层。
