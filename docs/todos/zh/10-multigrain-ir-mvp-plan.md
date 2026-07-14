# TODO：多粒度 IR MVP 实施计划

> 历史方案（已被取代）。当前实现已收敛为更薄的 `ExecutionGraph`，
> 下文所述 `ir/model.py` 与 pass 系统已删除。现行契约请从
> [`rayorch/experimental/multigrain/README.md`](../../../rayorch/experimental/multigrain/README.md)
> 开始阅读。

状态：实验`rayorch.experimental.multigrain`的实施计划
原型。

本文档将多粒度 API 设计转变为
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md)
纳入实施清单。近期目标不是完整的 Ray 执行。
目标是语义完整、被动、优化器友好的 IR，它可以
在我们构建更多运行时机器之前，先展示所有 MVP 功能。

## 设计目标

原型应该遵循与 HYDP DataFlow 相同的架构课程：

```text
authoring API
  -> passive IR
  -> validation / analysis / transform passes
  -> executor or backend lowering
```

对于 RayOrch 多粒度数据流，IR 还必须表示：

- 每个端口谷物；
- 跨谷物的亲子关系；
- 子序数和显示身份；
- 物化和恢复注释；
- 射线降低的物理提示；
- 优化器插入的节点，例如 rebatch 或 Materialize。

高级 API 可以增长方便的包装器，例如`Select`，但是 IR
应该仍然是一个小的关系感知代数。

MVP 应该明确地将用户友好的 API 视为前端糖衣
相同的规范 IR，而不是单独的执行模型：

```text
torch-like module calls
wrapper declarations
forward helpers
plain return shapes
adapter hooks
optional return protocols
        -> MultigrainIR
```

这使得前端足够开放，可以从类似 Torch 的 DAG 中借鉴好的想法，
功能助手、纯 UDF 适配器和关系协议，同时防止
来自碎片的表示。编译器应该规范化等效项
含义与`Map / Expand / Filter / Reduce / Relate / Project`IR相同
验证和优化之前的形状。

## 设计原则

1. **端口优先**：沿袭、重新分批、减少分组和恢复所有附加
   到端口，而不仅仅是节点。
2. **被动红外**：没有实时操作员实例、光线手柄、运行时演员或
   IR 中的后端数据集。
3. **配方节点**：节点携带可导入的运算符配方、构造函数参数、
   逻辑契约、属性和物理提示。
4. **关系显式**：每个跨粒度扇入都必须有一个显式关系
   关系合约，例如`group_by`或`Relate`。
5. **物理融合保留逻辑血统**：优化可能会融合执行
   阶段，但跟踪/恢复仍然必须看到逻辑运算符。
6. **具体化是一个注释**：检查点/重播/调试边界是
   IR 元数据或优化器插入的节点，而不是执行器本地黑客。

## 被动 IR 决策：Trace Token 与 IR Ref

最重要的一个结构规则：**跟踪时间符号标记必须
不进入 IR。** 令牌可以携带用于创作人体工程学的实时构建器，
但编译后的 IR 仅存储被动参考/规格。这保持
`MultigrainIR`可腌制、与引擎无关、可安全地进行远程传递
降低和离线工具。

### 追踪需求与 IR 需求

- `forward()`通过类似火炬的普通调用（`images = self.expand(pdfs)`）进行跟踪。
  为此，流经`forward()`(`SymbolicPort`) 的值需要一个句柄
  返回到图形构建器（`tracer`），以便每个包装器调用都可以注册一个节点。
- 相比之下，IR 仅需要被动数据：`IRPortRef`（节点/端口/索引）
  和`IRPortSpec`（参考+颗粒+显示/序数元数据）。没有追踪器，没有直播
  操作员，无射线手柄。

### 其他框架如何划定这条线

| 框架 | 追踪代币 | 代币携带构建者？  | 存储在 IR 中 | IR 无源？  “管道”
|---|---|---|---|---|
| PyTorch FX |`Proxy`| 是（通过`Tracer`） |`Node`（名称/操作/参数） | 是|
| JAX |`Tracer`| yes（跟踪上下文） |`Var`/atom in`Jaxpr`| yes |
| TVM 继电器 / MLIR | 构建器`Expr`/`Value`| 独立构建器 |`Expr`/`Value`参考 | 是 |
| ONNX | 无（直接构建） | 不适用 |`NodeProto`/`ValueInfo`| 是 |
| HYDP-数据流 |`PortRef`| 否（纯数据类） |`PortRef`| 是 |
| RayOrch（之前） |`SymbolicPort`| **是 (`tracer`)** | **`SymbolicPort`** | **否** |
| RayOrch（现在） |`SymbolicPort`| 是 (`tracer`) |`IRPortRef`/`IRPortSpec`| 是 |

两种经过验证的范例都产生被动 IR：

- **A (FX / JAX):** 令牌携带构建器，但 IR 存储*不同*
  被动引用类型。两种类型：跟踪令牌 + IR 参考。
- **B（HYDP-数据流）：**令牌*是*被动引用；建筑商住在一个
  外部`contextvar`主动跟踪器。一种端到端的类型。

早期的 RayOrch MVP 是一个不干净的混合体：它使用范例 A 的构建器 -
携带令牌，但随后将相同的令牌存储在 IR 中，因此 IR 传递
举办了现场`tracer`，并不被动。

### 决策（范例 A / FX-JAX 一致）

我们保留了构建器携带的`SymbolicPort`，以实现跟踪时间人体工程学，并制作
IR 存储被动`IRPortSpec`/`IRPortRef`：

- `SymbolicPort`仅是跟踪时间。它容纳了`tracer`，所以`self.op(x)`可以
  在`forward()`期间注册节点。它从未出现在`MultigrainIR`内部。
- `GraphTracer.add_node()`将`SymbolicPort`返回到`forward()`但记录
  `IRNode(input_specs=..., output_specs=...)`使用被动`IRPortSpec`。
- `GraphTracer.build()`将图形输入转换为`IRPortSpec`并进行图形输出
  到`IRPortRef`。
- `IRNode.input_refs`/`output_refs`/`inputs`/`outputs`方便
  源自被动规格的属性（由通道和显示使用）。

为什么采用范式 A 而不是范式 B（HYDP 的单一参考）：RayOrch 包装器
已经直接调用了`port.tracer.add_node(...)`，这是显式的，不需要
全局/`contextvar`状态并支持并发/嵌套跟踪，无需
干涉。与 FX/JAX 保持一致可以获得被动 IR，且流失率最小；崩溃
单一引用类型（范式 B）是未来可能的改进，而不是
要求。

结果：编译后的`MultigrainIR`现在可以干净地腌制并且不包含
`SymbolicPort`（根据跟踪的 PDF 管道进行验证），这正是
通过和未来射线降低取决于“被动红外”属性。

## 提议的数据模型

### 参考和端口规格

```python
@dataclass(frozen=True)
class IRPortRef:
    node: str
    port: str = "out"
    index: int = 0


@dataclass(frozen=True)
class IRPortSpec:
    ref: IRPortRef
    grain: str
    name: str | None = None            # logical label; passive, no tracer
    payload: PayloadKind = PayloadKind.OBJECT
    display_key: DisplayKeySpec | None = None
    ordinal_key: OrdinalKeySpec | None = None
    materialize: MaterializePolicy = MaterializePolicy.NEVER
    # node/index/port convenience properties delegate to `ref`
```

`IRPortSpec`是实际上存在于 IR 中的无源端口载体（
FX-`Node`/JAX-`Var`模拟）。跟踪时间`SymbolicPort`是一个单独的类型
它还保留实时的`tracer`并且永远不会进入IR。

`display_key`支持面向用户的跟踪输出，例如
`document=a.pdf/page=17`。`ordinal_key`支持确定性归约排序
物理重新配料后。

### 节点

```python
@dataclass(frozen=True)
class IRNode:
    name: str
    kind: NodeKind
    input_specs: tuple[IRPortSpec, ...]   # passive; no SymbolicPort
    output_specs: tuple[IRPortSpec, ...]  # passive; no SymbolicPort
    contract: CardinalityContract
    op: OperatorRecipe
    properties: OperatorProperties
    physical: PhysicalHints
    # input_refs / output_refs / inputs / outputs are derived properties
```

`NodeKind`开头为：

```text
MAP
EXPAND
FILTER
REDUCE
RELATE
PROJECT
REBATCH
MATERIALIZE
```

`PROJECT`、`REBATCH`和`MATERIALIZE`可以通过优化器通道插入。

### 基数和关系

```python
@dataclass(frozen=True)
class CardinalityContract:
    kind: CardinalityKind
    input_grains: tuple[str, ...]
    output_grains: tuple[str, ...]
    relations: tuple[RelationSpec, ...]


@dataclass(frozen=True)
class RelationSpec:
    output: IRPortRef
    relation: RelationKind
    parents: tuple[IRPortRef, ...]
    parent_input: int | None = None
    anchor: IRPortRef | None = None
    ordinal: OrdinalPolicy = OrdinalPolicy.PRESERVE
    missing: MissingChildPolicy = MissingChildPolicy.FAIL_OPEN
```

`RelationKind`涵盖：

```text
PRESERVE   1:1
EXPAND     1:N
FILTER     1:0/1
REDUCE     N:1
RELATE     M:N
GROUPED    helper relation for reduce input grouping
```

### 操作员配方和属性

```python
@dataclass(frozen=True)
class OperatorRecipe:
    cls_ref: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorProperties:
    deterministic: bool = True
    side_effect: bool = False
    idempotent: bool = True
    retryable: bool = True
    expensive: bool = False
    gpu_heavy: bool = False
    stateful: bool = False
```

这些属性大部分是 MVP 中的注释，但它们是必需的
未来的部分重播和实现计划。

### 物理提示

```python
@dataclass(frozen=True)
class PhysicalHints:
    replicas: int = 1
    num_gpus_per_replica: float = 0.0
    max_inflight: int = 1
    batch_size: int | None = None
    prefer_rebatch: bool = False
    engine: str | None = None
```

逻辑 IR 存储提示，而不是具体的 Ray 参与者或资源分配。

### 物化

```python
@dataclass(frozen=True)
class MaterializationSpec:
    port: IRPortRef
    policy: MaterializePolicy
    reason: MaterializeReason
    storage: StorageSpec | None = None
```

政策：

```text
NEVER
DEBUG_ONLY
ON_FAILURE
ALWAYS
CHECKPOINT
```

理由：

```text
TRACE
REPLAY_BOUNDARY
EXPENSIVE_OP
NONDETERMINISTIC
SIDE_EFFECT
USER_REQUEST
```

### 顶级投资者关系

```python
@dataclass(frozen=True)
class MultigrainIR:
    name: str
    inputs: tuple[IRPortSpec, ...]
    nodes: Mapping[str, IRNode]
    topo_order: tuple[str, ...]
    deps: Mapping[str, tuple[str, ...]]
    consumers: Mapping[str, tuple[str, ...]]
    graph_outputs: tuple[IRPortRef, ...]
    materialization: tuple[MaterializationSpec, ...] = ()
```

IR 应提供：

```python
ir.describe()
ir.to_dict()
ir.to_mermaid()
```

与 HYDP DataFlow 一样，IR 应该是可序列化和可重构的，无需
保存实时操作员实例。

## 完整性检查表

使用这个矩阵来检查IR是否可以在没有特殊的情况下代表一个特征
案例：

| 轴 | 所需覆盖范围 |
| --- | --- |
| 拓扑 | 输入、输出、扇出、扇入、依赖、消费者、拓扑顺序 |
| 端口 | 命名输入、命名输出、多输出、每端口颗粒 |
| Grain | 文档、页面、块、块、样本、任意用户标签 |
| 关系 | 保留、扩展、过滤、减少、关联、分组助手 |
| 执行 | 资源提示、有状态、gpu 重、后端绑定 |
| 优化 | 融合、重新分批、减少组计划、物化 |
| 恢复 | 确定性、副作用、可重试、检查点策略 |
| 可观察性 | 显示身份、子序号、跟踪、删除、隔离 |
| 序列化 | 运算符引用、args、kwargs、出处、无活动对象 |
| 合约检查 | 静态声明、动态证据、运行时形状/关系验证 |

## 针对前端 Sugar 目标的 MVP 审查

当前覆盖范围：

| 前端机制 | MVP 状态 | 规范 IR 目标 |
| --- | --- | --- |
| 类似 Torch 的模块调用 | 在`Pipeline.compile()`跟踪中实现 |`MultigrainIR`节点和端口 |
| 包装器声明 | 为`Map`、`Expand`、`Filter`、`Reduce`、`Relate`、`Select`| 实现 节点类型、操作符配方、基数合约⟪管道⟫
| 前向助手|`group_by()`实现了|`REDUCE`，带有锚点和后代|
| 普通返回形状 | 实现嵌套`Expand`、`Filter`掩码、`Select`注释 |`EXPAND`、`FILTER`、`MAP+FILTER+PROJECT`|
| 适配器挂钩 |`Relate`现在具有三层：声明式`on={role: field}`key-join、by-ref`relation_adapter="pkg:fn"`（点路径，在执行时解析）和实时`relation_fn`；全部通过被动 IR 在本地执行。`Select`的`mask_fn`/`key_fn`仍未实现。请参阅[`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md)|现有关系规范以及运行时证据检查；`on`/`relation_adapter`作为纯数据存储在配方出处 |
| 返回协议 | 尚未实现 | 高级逃生舱口`RELATE`|
| 关系感知显示 | 基本`describe()`、`to_dict()`、`to_mermaid()`实现 | IR 检查和调试 |
| 验证 | 基本`VerifyPass`已实现 | 拒绝不明确或非规范关系的使用 |
| 本地 IR 执行 |`MultigrainExecutor`为`Map`、`Expand`、`Filter`、`Reduce`、`Relate(relation_fn)`实现，`Select`降低（`Map + SelectFilter + Project`，配方来源中携带的掩码位置），加上`Rebatch`/`Materialize`直通 | 在没有实时 Pipeline 对象的情况下执行规范、降低和转换的 IR；还在腌制后重新加载的 IR | 上进行了验证
| IR 通道 |`RelationSummaryPass`、`RebatchCandidatePass`、`PlanReduceGroupsPass`、`MarkMapFilterFusionCandidatesPass`和`InsertRebatchAfterExpandPass`实现 | 读取/写入规范 IR 的分析和转换通道 |
| 封装结构 | 面向用户`__init__`缩小；按原始家族划分的运算符包装器| 保持开源 API 较小，同时保留内部 IR/pass 模块|

在调用 MVP 语义上完成之前需要弥补的差距：

- `Relate`的适配器挂钩已完成（三层：`on=`键连接，
  `relation_adapter`参考，`relation_fn`现场）。剩余：`mask_fn`/`key_fn`
  对于`Select`类似的API，以及`on=`的外/左连接语义（目前
  仅内部等值连接）。
- 决定当前的`PortBatch.relations`sidecar 是否足以满足
  本地 MVP 或者应该成为专用的关系感知批处理/容器。
- 添加规范化测试，显示不同的前端糖与
  相同的含义产生等效的归一化 IR。
- 将优化器测试扩展到已实施的仅 IR 通过测试之外。
- 扩展本地 IR 执行器的覆盖范围，以应对未来的返回协议和重要任务
  超出`REBATCH`/`MATERIALIZE`传递的物理行为。
- 明确哪些 API 是稳定表面，哪些是逃生舱口。

MVP 的接受度应该按照以下规则来判断：

```text
Every accepted user API must either lower to the small canonical IR directly,
or be rejected with a readable diagnostic before execution.
```

## 初始通过

### 红外验证

```text
ValidateCrossGrainFanIn
  reject ordinary Map calls that mix grains without group_by or Relate

ValidateExpandParent
  require multi-input Expand to declare parent_input

ValidateReduceGroupBy
  require Reduce inputs to include one anchor and descendants of that anchor

ValidatePortRelations
  check every output port has exactly one relation spec
```

### 运行时合约检查

一些关系事实不是静态可知的，因为它们依赖于 UDF
输出。执行器或包装适配器必须在调用时验证它们：

```text
CheckMapPreserve
  output length must match aligned input length

CheckExpandEvidence
  nested group count must match parent rows
  multi-output expanded groups must share relation lengths unless explicitly
  declared as separate relations

CheckFilterEvidence
  mask length must match input rows
  mask values must be bool
  kept records preserve identity

CheckReduceEvidence
  descendants must have ancestry to the anchor
  child ordering must use ordinal or stable key
  missing-child policy must be explicit

CheckRelateEvidence
  parent refs must be invocation-local
  roles must match declared inputs
  relation multiplicity and ordering must be explicit
```

这些检查不是优化器通过。它们是运行时断言
不透明 UDF 观察到的关系效应满足 IR 契约。失败应该
被报告为合同违规行为，而不是作为商业检疫记录。

### 分析

```text
PortProvenance
  compute upstream port ancestry for each port

RelationLineage
  compute logical relation chain used by trace and partial replay

MaterializationCandidates
  mark ports that may be useful replay/debug boundaries
```

实施的 MVP 分析过程：

```text
RelationSummaryPass
  summarize node kinds, relation kinds, expand outputs, and reduce groups

RebatchCandidatePass
  find Expand outputs whose physical hints request child-grain rebatching

PlanReduceGroupsPass
  create minimal reduce grouping plans from REDUCE contracts

MarkMapFilterFusionCandidatesPass
  detect canonical Map -> Filter patterns for physical Select fusion
```

### 变换

```text
InsertRebatchAfterExpand
  insert child-grain REBATCH before GPU-heavy Map stages

FuseMapFilter
  keep logical Map+Filter but create a fused physical Select candidate

FuseAdjacentMaps
  fuse cheap same-grain Maps when safe

PlanReduceGroups
  decide child order, missing-child policy, and grouping materialization

PlaceMaterializationBoundaries
  add materialization annotations for expensive or unsafe replay regions
```

实现了 MVP 转换过程：

```text
InsertRebatchAfterExpandPass
  insert logical REBATCH nodes after Expand outputs and rewrite downstream
  input refs in the IR
```

## 实施清单

### 1. 升级 `ir/model.py` 数据模型

- 添加`IRPortRef`、`IRPortSpec`、`IRNode`、`RelationSpec`、
  `CardinalityContract`,`OperatorRecipe`,`OperatorProperties`,
  `PhysicalHints`、`MaterializationSpec`和`MultigrainIR`。
- 如果有帮助，请保留向后兼容的别名：
  `CompiledGraph = MultigrainIR`，`NodeSpec = IRNode`。
- 保留当前的测试人体工程学，同时添加更丰富的字段。

### 2.更新`GraphTracer`

- 从`Pipeline.forward()`构建`MultigrainIR`。
- 使用粒度和关系元数据填充端口规范。
- 填充`deps`、`consumers`和`topo_order`。
- 尽可能存储操作员配方而不是现场操作员。

### 3.添加红外显示和序列化

- 实施`describe()`。
- 实现没有活动对象引用的`to_dict()`。
- 实现`to_mermaid()`进行图形调试。

### 4.添加IR验证

- 从纯函数或小类开始。
- 覆盖跨粒Map、Expandparent、Reducegroup_by、输出关系
  一致性。

### 5.添加通道骨架

- 添加最小的`AnalysisPass`、`VerifyPass`、`TransformPass`、`LintPass`。
- IR稳定后添加一个小的`PassManager`。
- 在转换需要它们之前，不要过度构建缓存/失效。

### 6. 实现本地IR执行器

- 实现了`MultigrainExecutor`以从`MultigrainIR`本地执行。
- 执行`Map`、`Expand`、`Filter`、`Reduce`、`Relate`和`relation_fn`，
  `Select`降低（合成`SelectFilter`从
  `op.provenance["mask_input"]`并丢弃它），以及`Project`/`Rebatch`/
  `Materialize`直通。
- 将节点输出存储在由`IRPortRef`键控的上下文中。
- 保持此仅本地状态，直到语义稳定。
- 在实现之前通过急切的包装器契约检查路由 UDF 结果
  输出端口。
- 使用新的虚拟操作符对每个主题进行端到端验证，包括
  执行 pickled-then-reloaded IR 和 rebatch-transformed IR，其中
  确认仅被动 IR（无实时`Pipeline`）可以驱动执行。

### 7. 实现过滤和选择

- 将`Filter`实现为仅掩模`1:0/1`。
- 模型`Select`作为`Map + Filter + Project`的高级API。
- 将放置元数据与隔离区分开保存。

### 8. 实施第一轮优化器传递

- 在 GPU 密集型`Map`之前的`Expand`之后插入微批次内重新批次。
- 为`Map + Filter`添加物理融合注释。
- 计划减少分组和排序。

### 9. 物化和恢复注释

- 首先添加仅注释支持。
- 在可观察到物化边界之前，不要实施真正的重放
  测试。

### 10. 测试矩阵

添加测试：

- 线性映射；
- 扇出；
- 同粒扇子；
- 无效的横纹扇形；
- 具有共享关系的多输出扩展；
- 用group_by减少；
- 过滤和删除元数据；
- 选择下降；
- 重新批量插入；
- 物化注释；
- `to_dict()`不包含任何实时操作员实例。

## 射线执行 MVP (`MultigrainRayExecutor`)

第一个、故意的小射线降低存在于
`rayorch/experimental/multigrain/ray/executor.py`。它重用了本地
`MultigrainExecutor`节点逻辑 *内部* Ray 任务，因此语义保持相同
本地执行并且仅调度更改。它验证了被动红外
再加上`PhysicalHints`可以驱动真正的并行性：

- **节点内副本并行性**：行独立节点（`Map`，`Filter`，
  `Expand`）在`PhysicalHints.replicas`Ray 任务之间进行行分片并合并
  返回`core.concat`，保留记录身份和血统。`Reduce`和
  `Relate`在 MVP 中的单个任务上运行（它们需要跨行上下文）。
- **管道微批次重叠**：`execute_microbatches(...)`推出
  每个微批次的全图执行作为具有有限飞行中的 Ray 任务
  窗口（`max_inflight`）。

在 CPU + 每行睡眠虚拟操作（8 行/4 微批次）上测量：

| 旋钮 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 映射副本（8 行） | 2.06s | 1.10s | 0.70s | 0.42s |
| 微批次重叠 (4 mb) | 3.22s | 1.61s | 0.81s | - |

由`test/experimental/multigrain/test_ray_parallelism.py`覆盖（标记为
`slow`;需要`--runslow`+`ray_cluster`灯具）。执行人不属于
窄封装`__init__`；将其导入自
`rayorch.experimental.multigrain.ray`。

MVP 限制仍然开放：`batch_size`提示尚未消耗；`Reduce`/`Relate`
没有被分片；没有跨节点流调度程序（微批次重叠
以整个图的粒度，而不是每个阶段）。

### 不平衡 1:N 扇出上的 GPU 负载平衡

`num_gpus_per_replica`现已被消耗：分片节点任务通过
`.options(num_gpus=...)`，因此每个副本都固定到一个 GPU。执行人还
采用可选的`shard_planner（节点，输入，副本） - >每个分片行索引
列出`; returning `无` falls back to contiguous ranges. `lpt_shard_planner(
weight_of)` 实现了最长处理时间的贪婪装箱，因此分片
平衡*总工作量*（例如页面内容长度）而不是行数。排
身份/序数通过`PortBatch.take`和任何下游`Reduce`生存
通过序数恢复逻辑顺序，因此跨分片重新排序是安全的。

在 4×H20 上使用真实 GPU MinerU 模拟进行验证
（`test/experimental/multigrain/{gpu_ops.py,bench_gpu_mineru.py}`，在
`torch-base`conda env): 文档 -> 扩展到可变长度页面 -> GPU Map
（每页成本 = 内容长度，真实的`torch`matmuls，~4.5 ms/单位，完美
线性）-> 缩减回文档。 12 个文档/67 页/964 个 GPU 单元，倾斜如此
一些文件占主导地位：

| 分片规划器 | 每个 GPU 的完工时间 | 空闲气泡 | e2e 管道 |
|---|---|---|---|
| 连续（行数相等） | 1.52s（单位 275/334/121/234） | ~28% | 1.53s |
| LPT（工作平衡） | 1.10s（每个单位 ~241） | ~0% | 1.11s |

连续行分片使轻分片 GPU 空闲（约 28% 的泡沫），因为
1：N扇出不平衡；工作感知 LPT 重新平衡消除了泡沫（~1.38x
这里）。这就是具体化的“1：N重排序消除气泡”机制
在真实的 GPU 上。 （干净的每 GPU 数字使用预热的、GPU 固定的 actor 池来避免
Ray 工作者冷启动； Actor 在执行器的 GPU 任务运行之前被释放
所以他们没有 GPU。）

#### 长尾加速与调度理论

将不平衡的扇出分片到`R`GPU 上就是最小化完工时间
相同的机器。对于每页权重`w_i`:`OPT >= max(sum(w)/R, max(w_i))`
（下界），`makespan_LPT <= (4/3 - 1/(3R))*OPT`（Graham 1969），以及
`efficiency = sum(w)/(R*makespan) = 1 - idle_bubble`。朴素连续（等行
count) 分片会忽略`w_i`，因此在长尾下，一些重页面会发生冲突
一个 GPU 和气泡（因此可实现的加速比）随尾部增长
沉重。`test/experimental/multigrain/bench_gpu_longtail.py`横扫帕累托
4×H20 上的尾部（32 页，真实的`torch`matmuls）：

| Pareto alpha | tail p50/p99/max | 测量加速比 | 理论加速比 | LPT 效率 | 4/3 边界 |
|---|---|---|---|---|---|
| 2.5（浅） | 4/10/10 | 1.12x | 1.12x | 100% | 好的 |
⟪管道⟫ 1.7 ⟪管道⟫ 5/13/13 ⟪管道⟫ 1.16x ⟪管道⟫ 1.16x ⟪管道⟫ 98% ⟪管道⟫ 好的⟪管道⟫
| 1.2（重） | 6/39/39 | **1.46x** | 1.46x | 99% | 好的 |

测量的 GPU 完工时间跟踪分析预测精确到小数点后两位，LPT 保持不变
在其 4/3 界限下并以约 99% 的效率运行，并且加速比上升
随着尾巴变重，单调地变化——正如理论所说。上一个完整的
`Expand -> GPU OCR -> Reduce`具有两个主要重文档的管道（16 个文档
/ 65 页），执行器执行 **1.56s -> 1.02s (1.53x)** 并获得相同的结果。

#### 谱系对于重新排序是不变的

LPT 重新平衡*跨分片*排列行，因此它不能损坏
血统。`test/experimental/multigrain/test_lineage_under_parallelism.py`
（CPU 虚拟操作，`slow`+`ray_cluster`）将单任务本地执行器视为
基本事实并断言并行+重新排序的运行是逐字节相等的：

- **身份/序数**：`Reduce`按祖先ID重新分组并按序数重新排序，
  因此，尽管分散，但每个文档的页面都会按原始顺序返回；
- **故障隔离**：在任何分片内引发的`BadRecordError(index=...)`
  登陆的坏页面被隔离为*相同*`ErrorTrace`
  (`source_item`,`logical_item``d03/page=2`, `upstream_path
  (SplitPages, EmbedPage)`)，并且每个健康的页面都会存活下来。

这就结束了循环：相同的沿袭元数据支持自动错误
跟踪正是使性能重新排序安全的原因。

#### 复杂的多级负载：泡沫分类法以及我们可以消灭的东西

单个地图阶段仅暴露一个气泡源。确认气泡消除
在实际负载下，`test/experimental/multigrain/bench_gpu_complex.py`运行
`doc -Expand-> pages -Expand-> blocks -Filter-> dense -Map(GPU OCR)-> -Reduce-> doc`，
同时在宽 GPU 平台上堆叠三个气泡源，并打印
诚实的残差。完整的分类：

| # | 气泡源 | 触发器 | 机制 | 状态 |
|---|---|---|---|---|
| 1 | 阶段内负载不平衡 | 每个项目的长尾工作 | LPT 按工作重新分片 | 被杀死 (~99%) |
| 2 | 复合扇出 | 扩展->扩展长尾 | 宽阶段的 LPT | 被杀死 |
| 3 | 滤波器引起的偏差 | 数据相关丢弃 | 被动 IR 重新分区每个节点的*自己的* 输入 | 已杀死 |
| 4 | 阶段间屏障 | 阶段 N+1 等待所有阶段 N | 全图微批次重叠 | 部分 |
| 5 | 减少扇入偏差 | 一个主播得到一个巨大的组 |Reduce 是 MVP 中的单任务 | 残差 |
| 6 | 原子巨项 | 一行的工作 ~= sum(w)/R | 无（一行不能拆分） | 残差（OPT 绑定） |

由于 IR 是被动的，因此每个节点实际上都会重新分区数据
接收，因此源#1-#3 在 OCR 阶段被吸收，即使它们出现
上游。在 4×H20 上测量（8 个文档/63 个原始块 -> 44 个在`work>=4`中幸存）
滤波器，两级长尾扇出，每块工作长尾）：

- OCR 阶段：连续完工时间 1.38 秒（效率 84%，**16% 气泡**）->
  LPT 1.15s（**效率100%，0%气泡**），正是OPT下限
  (`max(sum(w)/R, max w)`= 1.15s) 且在 4/3 范围内；
- 端到端管道 1.39s -> 1.16s (1.19x)，字节相同的结果。

残差已说明，未隐藏：#5（最重的文档约为 0.95-1.5 倍）
如果Reduce本身是GPU密集型的，`W/R`跨种子）将使GPU闲置，因为
Reduce还没有分片； #6 是基本的（最大的单个块只是
~0.1-0.2x 的`W/R`这里，所以它没有约束力，但是怪物物品会限制
加速`OPT`）。调度保证（LPT满足Graham的界限并且>=95%
长尾效率，并且击败连续的，包括过滤器之后）是
确定性地锁定到 CI
`test/experimental/multigrain/test_load_balancing.py`（无 GPU，无计时）。

## MVP 边界

还没有实施：

- `Reduce`/`Relate`分片的光线降低（完成映射/过滤/扩展）；
- `batch_size`驱动的重新批处理（通过`num_gpus_per_replica`完成GPU放置）；
- 每阶段流调度程序（与全图微批次重叠）；
- 持久的谱系水槽；
- 全局随机播放或分布式连接；
- 完整的`Relate`/发出语义；
- 实际部分重播；
- 自适应运行时重新优化。

这些取决于 IR 是否足够完整以首先表达它们。

