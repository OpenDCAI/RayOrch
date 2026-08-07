# RayOrch Multi-Grain v3.4：Port / Domain 驱动的运行时

> 状态：**权威设计规范（implementation contract）**。
>
> v3.4 是独立新内核，不通过 lowering 复用 v3/v3.2 executor。v3 仅作为 table-driven runtime、lineage、GroupShape、atomic commit、recovery 与实验行为的参考。

## 0.1 v3.4 相对 v3.3 的精简

v3.4 不改变 Port、Domain、Grain、Item、Shape 与 Arena 语义，只收敛执行层：

- 全部完整 Pipeline 统一经过 Ray；Ray 是正常多进程执行环境，不是可选 backend；
- 只公开一个 Executor，不再存在 LocalExecutor、RayExecutor 或 backend 选择；
- Worker ABI 与 Arena 状态机仍可无 Ray 单测，但不组成第二套端到端 runner；
- dummy、nested、failure、packing 与 recovery 回归全部走真实 actor、RPC 和 ObjectRef；
- 不引入 ExecutionBackend、ReferenceRunner 或插件接口来补偿被删除的本地路径。

v3.3 的真实 workload 性能数据只作为历史基线，不能直接标记为 v3.4 结果。

## 0. 核心结论

v3.4 将计算和结构关系彻底分开：

1. `RayModule` Call 是唯一计算边界，也是唯一可能产生 actor、RPC 和 Grain 的对象。
2. Port 表示每个 Entity 上的逻辑值；Domain 表示 Entity 的坐标空间。
3. `expand/reduce/broadcast/filter` 是 Port 与 Domain 的结构关系，不是计算算子。
4. 逻辑结构可以在物理执行时融合进 producer/consumer worker，但逻辑关系不能因此消失。
5. Runtime 只有一个语义写入口；所有 Ray completion 必须先形成 CommitDelta，再原子提交。

```python
class PdfPipeline(rayorch.Pipeline):
    def __init__(self):
        self.render = rayorch.RayModule(RenderPages)
        self.ocr = rayorch.RayModule(OcrPage)
        self.assemble = rayorch.RayModule(AssemblePdf)

    def forward(self, pdfs):
        page_groups = self.render(pdfs)
        pages = F.expand(page_groups)

        contents = self.ocr(pages)
        content_groups = F.reduce(contents)

        return self.assemble(pdfs, content_groups, page_groups)
```

这段 Pipeline 只有三个 Call 和三个 actor pools。`F.expand`、`F.reduce` 均为零 actor、零 RPC、零 Grain。

## 1. 抽象层次

论文和代码采用同一套四层模型：

```text
Programming Model
    RayModule + symbolic Port API

Logical Data Model
    Program / Call / Port / Domain / Entity / Item / Grain

Incremental Runtime
    semantic facts + derived progress + atomic event transitions

Physical Execution
    actor pools + batches + ObjectRefs + worker layout plans
```

### 1.1 Programming Model

回答“用户如何声明计算和结构关系”。这一层不出现 actor handle、ObjectRef、Arena 或 runtime table。

### 1.2 Logical Data Model

回答：

- 哪个 Call 执行 UDF；
- 哪个 Port 传递逻辑值；
- Port 属于哪个 Domain；
- parent/child Entity 如何对应；
- Item 和 Grain 的稳定 identity 是什么。

### 1.3 Incremental Runtime

回答：

- 哪些 Item、Shape、Grain 已达到终态；
- 哪个 Call invocation 已经 ready；
- 哪个 group 已经完整；
- 失败、drop 和 suppression 如何传播。

### 1.4 Physical Execution

回答 batching、actor placement、block layout、ObjectRef ownership、retry 和 worker ABI。物理层不得反向定义逻辑语义。

## 2. 操作与行为矩阵

| API | Call/Grain | actor/RPC | Domain | Entity 关系 | Value 行为 |
|---|---:|---:|---|---|---|
| `RayModule(...)` | 有 | 有 | 不变 | 同 Entity 上 N 输入→K 输出 | 执行 UDF |
| `F.expand(p)` | 无 | 无 | parent→child | 1:M | producer flatten / structural projection |
| `F.reduce(p)` | 无 | 无 | child→parent | M:1 | 虚拟有序 group |
| `F.broadcast(p, like=x)` | 无 | 无 | ancestor→descendant | 沿既有 lineage 投影 | 复用 value binding |
| `F.filter(p, mask)` | 无 | 无 | 不变 | 每个 Entity 1:0/1 | alias 或 DROPPED |
| `F.optional(p)` | 无 | 无 | 不变 | 不改变 membership | Call edge 上 DROPPED→MISSING |

### 2.1 Call 保持 execution domain

一个 Call 的所有 Port outputs 与 Call execution domain 相同。即使 UDF 返回 `list[Page]`，原始 output 仍是：

```text
Port[PDF, list[Page]]
```

只有 `F.expand()` 创建 Page Domain。

### 2.2 Expand 是 1:M relation

`F.expand(group_port)`：

- 创建一个 child Domain；
- 为每个 parent Entity 建立独立 Shape；
- Shape 成功后创建有序 child Entities；
- 不运行用户 UDF。

### 2.3 Reduce 是 lineage gather，不是 fold

`F.reduce()` 只沿已知 Domain ancestry 恢复有序结构，不接受 reducer UDF，不执行 sum/aggregate。业务聚合必须由后续 RayModule 完成。

### 2.4 Broadcast 不创建 Entity

Broadcast 只能从 ancestor Domain 投影到已存在的 descendant Domain。它不能代替 Expand，也不能 child→parent；后者必须 Reduce。

### 2.5 Filter 保持 Domain

Filter 不创建 subset Domain。源 Port 和 filtered Port 共享 Entity coordinate；filtered Item 为 PRESENT 或 DROPPED。这样不同分支可分别消费 filtered 和 unfiltered Port。

### 2.6 Optional 是 input edge policy

Optional 不创建 PortValue。它只改变某个 Call input 对正常 DROPPED 的解释；FAILED/SUPPRESSED 永远不能被 optional 吞掉。

## 3. 第一版 scope

### 3.1 支持

- 一般静态 DAG，多输入、多输出、分支、汇合；
- 动态一层和多层 fan-out/fan-in；
- ordered group、nested group、N=0；
- Filter、optional、broadcast；
- 多个 aligned expanded/grouped Ports；
- 持久 actor、elastic/parent-bound batching；
- coarse ObjectRef blocks；
- 多 Arena overlap；
- 记录级失败、generation fencing、局部 replay；
- pipeline output materialization 和 lineage explain。

### 3.2 不支持

- 任意 key group-by/shuffle/join；
- timestamp/window join；
- 仅因长度相等而 zip 两个独立 Domain；
- 保留原 Python collection object identity；
- partial multi-output commit；
- 无界 fan-out/group。

## 4. 稳定引用

逻辑引用：

```text
CallRef
PortRef
DomainRef
EntityRef

ItemRef  = (PortRef, EntityRef)
GrainRef = (CallRef, EntityRef)
```

物理引用：

```text
PoolRef
BlockRef
_Dispatch (Executor-private)
generation
```

逻辑 identity 禁止包含：

```text
actor / replica / batch / dispatch / generation / ObjectRef / completion order
```

### 4.1 EntityRef

根 Entity identity 来自 `(run, source bundle, source coordinate)`。child Entity identity 由以下信息稳定生成：

```text
H(child_domain, parent_entity, ordinal)
```

实现可以使用 compact integer/u128，但必须通过 EntityLineage O(1) 找回 parent 和 ordinal。

### 4.2 GrainRef

```text
GrainRef = (CallRef, execution EntityRef)
```

输入 bindings、GroupShape 和 outcomes 不进入 Grain identity。静态 Program 中一个 Call 对同一 Entity 至多执行一次逻辑 Grain。

## 5. 静态 Program

```text
Program
├── calls:   CallRef -> CallSpec
├── ports:   PortRef -> PortSpec
├── domains: DomainRef -> DomainSpec
├── source_ports
├── output_tree
├── control_ports: frozenset[PortRef]
└── derived indexes
    ├── consumers_by_port
    ├── outputs_by_call
    ├── views_by_source
    └── shape_reporters_by_domain
```

Program 在 compile 后不可变，不包含 runtime state 或 Ray 对象。

`control_ports` 是 Filter 控制值的静态需求闭包。编译器从每个
`FilterOrigin.mask_port` 出发，沿 `BroadcastOrigin.source_port` 和
`ExpandOrigin.group_port` 反向传播；它只说明哪些 Port 必须额外产生小型
control manifest，不会把 bool payload 搬进 Program。

### 5.1 CallSpec

```text
CallSpec
├── ref
├── kernel_ref
├── execution_domain
├── inputs: tuple[InputSpec, ...]
└── driving_input: int
```

```text
InputSpec
├── name / position
├── port
└── mode: REQUIRED | OPTIONAL
```

Call outputs 不重复保存在 CallSpec；由 `PortOrigin.CallOutput` 推导。

#### 输入终态分类

| receipt | driving REQUIRED | side REQUIRED | OPTIONAL |
|---|---|---|---|
| PRESENT | value | value | value |
| DROPPED | outputs DROPPED | outputs SUPPRESSED | worker receives MISSING |
| FAILED | SUPPRESSED | SUPPRESSED | SUPPRESSED |
| SUPPRESSED | SUPPRESSED | SUPPRESSED | SUPPRESSED |
| absent | wait | wait | wait |

`driving_input` 默认第一个业务 Port，可显式指定。常量参数不能成为 driving input。

### 5.2 PortSpec

```text
PortSpec
├── ref
├── domain
└── origin: PortOrigin
```

`PortOrigin` 是封闭联合类型：

```text
Source(source_index)

CallOutput(call_ref, output_index)

ExpandView(group_port)

GroupView(
    value_port,
    members_port,
)

BroadcastView(source_port)

FilterView(source_port, mask_port)
```

它描述 provenance，不保存业务 payload、consumer 或 execution policy。所有 view 的目标 Domain 只由所属 `PortSpec.domain` 决定，禁止在 PortOrigin 中保存第二份 target Domain。

### 5.3 DomainSpec

v3.4 第一版规定 Domain graph 是 rooted forest：每个非根 Domain 恰有一个 parent。

```text
DomainSpec
├── ref
├── parent: DomainRef | None
├── max_fanout
└── debug_name
```

在该约束下，不需要独立 `ExpansionRef`。动态 Shape 的 key 为：

```text
ShapeKey = (child DomainRef, parent EntityRef)
```

如果未来允许一个 Domain 有多个 incoming relation，再新增 RelationRef；第一版不预留空泛抽象。

### 5.4 ExecutionPlan

部署信息与 Program 分离：

```text
ExecutionPlan
├── pools: PoolRef -> PoolSpec
├── call_to_pool: CallRef -> PoolRef
└── call_policies: CallRef -> Batch/Resource/RecoveryPolicy
```

第一版默认一个 CallRef 对一个 PoolRef，但这不是逻辑不变量。结构 Port 永远不进入 call_to_pool。

## 6. Functional API 规则

### 6.1 expand

```python
child = F.expand(group)
children = F.expand_aligned(group_a, group_b)
```

规则：

- 输入 group Ports 必须同 Domain；
- `expand_aligned` 的 outputs 共享 child Domain；
- 每个 reporter 对每个 parent 必须给出相同 cardinality；
- 相同 structural signature 在同一 Program 中 intern；
- 两个不同 group Ports 的独立 `expand()` 永不因 cardinality 相等而对齐。
- 第一版 aligned group Ports 必须来自同一个 producer Call；跨 Call shape barrier 留作显式后续能力。

重复调用完全相同的 `F.expand(p)` 返回同一 logical relation。需要不同 relation 时必须通过显式 API 创建，不依赖 Python 调用次数制造 identity。

### 6.2 reduce

```python
group = F.reduce(values)
group = F.reduce(values, members=selected)

groups = F.reduce_aligned(
    values_a,
    values_b,
    members=selected,
)
```

规则：

- values 和 members 必须位于同一 child Domain；
- 默认 `members=values`；
- reduce 一次只回到直接 parent Domain；
- 连续 reduce 可由 compiler 融合为同一 hierarchical GroupShape；
- members PRESENT 保留 ordinal；DROPPED 删除 ordinal；FAILED/SUPPRESSED 使 group SUPPRESSED；
- aligned values 在 surviving ordinals 上必须满足对应 InputPolicy。

### 6.3 broadcast

```python
page_meta = F.broadcast(pdf_meta, like=pages)
```

规则：

- source domain 必须是 target domain 的 ancestor；
- 广播可跨多层 ancestry；
- runtime 解析到 ancestor Item 的现有 ValueBinding；
- 不复制 payload，不创建 actor，不创建新的 Entity。

### 6.4 filter

```python
keep = self.quality(pages)
good_pages = F.filter(pages, keep)
```

规则：

- source 和 mask 同 Domain；
- mask 必须为每 Entity 的 bool contract；
- true 发布 PRESENT 并复用 source binding；
- false 发布 DROPPED；
- mask failure 不能当作 false；
- filter 不改变其他 Port 的 membership。

### 6.5 optional

```python
result = self.merge(primary, aux=F.optional(aux))
```

规则：

- optional 只允许 Call input；
- DROPPED 转换成稳定 singleton `MISSING`；
- Python `None` 是合法业务值；
- UDF 返回 MISSING 是 contract error。

## 7. 编译与规范化

编译流程：

```text
Pipeline.forward(symbolic Ports)
→ trace Calls and Port views
→ normalize/intern Port expressions
→ infer/validate Domains
→ derive indexes
→ validate contracts and reachability
→ freeze Program
→ build independent ExecutionPlan
```

### 7.1 必须验证

- Call 的所有输入在应用显式 view 后属于 execution domain；
- unrelated Domains 不能进入同一 Call；
- Broadcast source 是 target ancestor；
- Reduce target 是 source ancestor，第一版 API 一次一层；
- Filter source/mask 同 Domain；
- Group members/value 同 Domain；
- Domain graph 无环且每个 child 只有一个 parent；
- 每个 child Domain 至少有一个 shape reporter；
- aligned reporter 集固定；
- Call output index 唯一连续；
- pipeline output tree 中所有 Port 可达；
- Port graph 无环；
- literal/default/MISSING 规则合法。

### 7.2 可做的 Port 优化

- identical structural view interning；
- repeated broadcast collapse 到最终 ancestor mapping；
- consecutive GroupView scope-path fusion；
- `reduce(expand(p))` 在无 filter/selection 变化时复用 canonical group realization；
- dead Call/Port elimination；
- 一个 group output 被多个 expand/view 使用时只 physical flatten 一次；
- derived indexes 使用 compact input slot，不复制 InputSpec。

所有优化必须保持 PortRef/DomainRef/ItemRef 和 terminal outcomes 可解释。

## 8. Runtime State

### 8.1 Semantic facts

```text
GrainTable:      GrainRef -> GrainRecord
ItemTable:       ItemRef -> ItemRecord
ShapeTable:      ShapeKey -> ShapeRecord
EntityLineage:   EntityRef -> EntityOrigin
```

### 8.2 Derived progress

```text
PendingInvocation: GrainRef -> input slots
GroupAccumulator:  grouped ItemRef -> progress
ReceiptQueue
ReadyQueue[CallRef]
```

Derived progress 可以缓存，但必须能从 Program 和 semantic facts 重建。

### 8.3 Value realization

```text
ValueTable: ItemRef -> ValueBinding

ValueBinding =
    RowBinding(block_ref, row)
  | GroupBinding(group_shape, flat_item_refs)

BlockRef(handle: opaque physical block handle)
```

ValueTable 与 ItemTable 分离。v3.4 不维护第二张 BlockTable；BlockRef 只透明包装物理块句柄，Ray 下为 ObjectRef，Worker ABI 单测中可为内存编号。物理块生命周期或 replay 不改写 Item terminal semantics。

### 8.4 Execution state

```text
DispatchLease
generation
PendingRPC
PoolCapacity
RecoveryCounters
```

这些不参与 Item/Grain identity。

## 9. Runtime Records

### 9.1 GrainRecord

```text
GrainRecord
├── ref
├── inputs: tuple[InputBinding, ...]
├── phase: READY | IN_FLIGHT | SEALED
├── outcome: SUCCESS | FAILED | SUPPRESSED | None
├── generation
├── active_attempt
└── infra_failures
```

### 9.2 ItemRecord

```text
ItemRecord
├── outcome: PRESENT | DROPPED | FAILED | SUPPRESSED
├── cause: CauseRef | None
└── control: bool | None
```

不存在 ItemRecord 表示未 terminal，不保存 PENDING。PRESENT Item 必须在同一
commit 中获得合法 ValueBinding。`control` 只允许出现在 PRESENT Item 上，
当前唯一用途是保存 Filter 所需的 bool 语义副本；它不是业务 payload，也不是
ArenaEngine 的隐藏缓存。重复 publication 必须同时匹配 outcome、cause、control
与 ValueBinding。

### 9.3 ShapeRecord

```text
ShapeRecord
├── state: OPEN | SUCCEEDED | DROPPED | FAILED
├── cardinality: int | None
├── expected_reporters
├── received_reports
└── cause
```

```text
SUCCEEDED(0)  合法空 group
DROPPED       parent group 正常被过滤
FAILED        无法建立合法 child structure
```

Shape 未 seal 前不得发布 aligned child Entities。

### 9.4 EntityOrigin

```text
EntityOrigin
├── domain
├── parent_entity
├── ordinal
└── shape_key
```

### 9.5 GroupShape

继承 v3 的 canonical representation：

```text
GroupShape(offsets_by_level)
+ flat ordered ItemRefs
```

它能表示一层和任意有限深度 nested list，支持 intermediate N=0。GroupBinding 在 ValueTable 中只构造一次，可被多个 Call consumers 共享。

## 10. Lineage

Lineage 分为四条正交链：

### 10.1 Port provenance

```text
PortRef -> PortOrigin -> upstream PortRefs
```

### 10.2 Entity ancestry

```text
child EntityRef -> EntityOrigin -> parent EntityRef -> root
```

### 10.3 Compute lineage

```text
GrainRef -> InputBindings -> CallOutput Ports
```

### 10.4 Failure lineage

```text
ItemRecord.cause / GrainRecord.outcome / ShapeRecord.cause
```

Producer 不必在每个 derived Item 中重复保存。可由 PortOrigin 和 EntityLineage 求得：

```text
ItemRef(expanded_port, page_entity)
→ ExpandView(group_port)
→ parent PDF entity
→ ItemRef(group_port, pdf_entity)
→ CallOutput(render_call)
→ GrainRef(render_call, pdf_entity)
```

实现可缓存该查询，但 authority 仍是 Program + EntityLineage。

## 11. Event-driven Arena

Arena 是单 microbatch 的唯一语义状态机。公开接口：

```text
admit_sources()
advance()
reserve_dispatch()
commit()
handle_failure()
next_deadline()
is_complete()
finish()
```

事件链：

```text
Source admission / BatchReport
→ preflight + CommitDelta
→ Item/Shape/Grain/Value facts
→ terminal receipts
→ consumers_by_port
→ PendingInvocation / GroupAccumulator / structural views
→ READY or terminal Grain
→ ReadyQueue
→ DispatchIntent
```

Runtime 不扫描整张 DAG 求 fixed point。

## 12. Structural Port 状态迁移

### 12.1 Expand

producer worker 按 OutputLayoutPlan：

1. 对每个 parent UDF output 验证 finite collection contract；
2. flatten 到 coarse child block；
3. 返回 cardinality 和 row ranges；
4. runtime 收集该 child Domain 的所有 reporter；
5. cardinality 一致后 seal Shape；
6. 创建 child EntityLineage；
7. 发布 Expanded child Items 和 ValueBindings。

原 grouped output 可用同一 child rows 构造 GroupBinding，不保存第二份原始 Python list。

多个 aligned reporters 在全部到齐前，其 output blocks 是 provisional execution state；不能提前驱动下游。

### 12.2 Reduce

GroupAccumulator 等待：

- scope path 上所有必要 Shape terminal；
- members Port 对所有 potential leaves terminal；
- surviving leaves 上所有 value Ports terminal。

Finalize：

1. members PRESENT ordinal 进入 canonical path；
2. members DROPPED ordinal 被删除；
3. members FAILED/SUPPRESSED 导致 group SUPPRESSED；
4. aligned values 按 surviving path 投影；
5. 构造 GroupShape + flat ItemRefs；
6. 在 parent Domain 发布 grouped Item PRESENT。

Reduce 不创建 Grain。后续 consumer Call 对 parent Entity 创建 Grain。

### 12.3 Broadcast

对 target Entity 沿 EntityLineage 找到 source Domain 对应 ancestor Entity：

- ancestor Item PRESENT：目标 Item PRESENT，复用相同 RowBinding/GroupBinding；
- ancestor Item DROPPED：目标 Item DROPPED；
- ancestor FAILED/SUPPRESSED：按原 cause 传播。

不创建 AliasBinding 链。

### 12.4 Filter

等待 source 与 mask 同 Entity terminal：

- source PRESENT + mask true：PRESENT，复用 source binding；
- source PRESENT + mask false：DROPPED；
- source DROPPED：DROPPED；
- source/mask FAILED 或 SUPPRESSED：对应失败传播；
- 非 bool mask：contract failure。

#### control manifest 的完整流动

用户 UDF 的 `list[bool]` 是 Worker batching ABI；逻辑模型仍是每个 Entity 一个
mask Item：

```text
compile: Filter.mask_port
    ↓ 反向计算 Program.control_ports
Worker/source admission
    ├── scalar bool → OutputReport.control
    └── expanded list[bool] → ExpandedRows.controls（与 rows 等长）
    ↓ Arena commit
ItemRecord.control
    ↓ broadcast 按 Item 复制；expand 已按 row 拆到 child Item
FilterOrigin
    ├── True  → alias source ValueBinding，发布 PRESENT
    └── False → 发布 DROPPED
```

因此恢复时只需 Program、ItemTable 和其他语义事实即可重新驱动 Filter；不需要
读取 Ray ObjectRef 中的 bool payload，也不存在 `_mask_values` 侧表。source
Port 被用于 mask 时，Executor 从对应 source 列生成逐行 control manifest；
类型不是严格 `bool` 或与行数不对齐都会在 admission/commit 前失败。

## 13. Map、Reduce 与多层级

RayModule Call 是 generalized entity-local map：多个同 Domain inputs 对一个 Entity 形成一个 Grain。

```text
Document Domain
    ↓ F.expand
Page Domain
    ↓ F.expand
Region Domain
```

```python
region_groups = F.reduce(regions)          # Region -> Page
document_groups = F.reduce(region_groups) # Page -> Document
```

逻辑上逐层明确；compiler 可把连续 GroupView 融合成：

```text
scope_path=(PageDomain, RegionDomain)
GroupShape(offsets_by_level)
flat Region ItemRefs
```

worker 一次重建 `list[list[Region]]`。

多层 corner cases：

- intermediate `N=0` 保留空 list；
- intermediate DROPPED 删除对应 subtree；
- failed-before-shape 不创建 child，ancestor group SUPPRESSED；
- known-N child failure 保留 coordinate/cause；
- hard limit 在创建部分 metadata 前原子检查。

## 14. 多输入、多输出、多分支

### 14.1 同 Port 多分支

```python
pages = F.expand(page_groups)
text = self.ocr(pages)
layout = self.layout(pages)
```

两条 Call 分支共享 Page EntityRef 和 source ItemRef，各自维护 Grain state。一个分支失败不改写 source，也不取消另一个分支。

### 14.2 同 Domain 汇合

```python
merged = self.merge(text, layout)
```

每个 Page Entity 形成 `GrainRef(merge, page_entity)`，无需 zip。

### 14.3 不同 Domain 汇合

必须显式：

```python
page_meta = F.broadcast(pdf_meta, like=pages)
merged = self.merge(pages, page_meta)
```

或：

```python
text_groups = F.reduce(text)
result = self.assemble(pdf_meta, text_groups)
```

没有显式 relation 时编译失败。

### 14.4 多输出

一个 Call 的 outputs 是独立 Ports，可拥有不同 consumers 和 structural views。任一 output contract 校验失败时，整个 Grain 失败，所有 outputs 均不可部分可见。

需要 partial success 的业务必须拆成多个 Calls。

### 14.5 Aligned groups

```python
images, boxes = F.expand_aligned(image_groups, box_groups)

image_groups2, box_groups2 = F.reduce_aligned(
    images,
    boxes,
    members=selected_images,
)
```

共享 Domain/Shape 和 canonical surviving paths，但每个 Port 保留独立 ItemRef/ValueBinding。

## 15. Worker ABI

Execution 只传稳定 DTO：

```text
BatchCall
├── call_ref / generation
├── grains
├── input_blocks
├── input_plans
└── output_layout_plans

BatchReport
├── grain results
├── output blocks
├── shape reports
├── scalar controls: OutputReport.control
├── expanded controls: ExpandedRows.controls
├── per-record failures
└── metrics
```

InputPlan：

```text
ScalarTake(block, row)
GroupTake(block selectors, GroupShape)
MissingTake
```

OutputLayoutPlan：

```text
OutputLayout
├── port
├── expanded_ports
└── control_ports
```

`OutputLayout` 是 compiler 根据 Expand 与 control demand 生成的物理计划，不进入
CallSpec；RayModule 不知道下游存在 Expand 或 Filter。Worker 只按布局校验 bool、
拆分行并填写 manifest。

Worker 不导入 Program、Arena、Driver；Execution 不解释 PortOrigin。

## 16. Atomic Commit

所有 source admission 和 BatchReport 均通过相同 commit discipline：

```text
1. generation fencing
2. validate report token/shape/cardinality/contracts
3. precompute CommitDelta
4. check hard limits and conflicting publications
5. publish blocks/values/grains/items/shapes/entities
6. enqueue receipts
```

任意 preflight 失败：

```text
不发布部分 output
不创建部分 child Entity
不暴露 provisional blocks
按 contract policy abort Arena 或 seal failed Grain
```

## 17. Failure、Drop、Optional 与 Recovery

### 17.1 终态区别

```text
PRESENT      有合法 value realization
DROPPED      正常 membership 删除
FAILED       自身计算/contract 失败
SUPPRESSED   因 required dependency 未执行
```

### 17.2 fan-out 前失败

cardinality unknown：

- Shape FAILED(cardinality=None)；
- 不创建伪 child；
- child-domain Calls 不产生 Grain；
- 依赖 group 的 parent Call SUPPRESSED；
- 无关分支继续。

### 17.3 retry

- retry 改变 attempt/generation，不改变 GrainRef/ItemRef/EntityRef；
- 旧 generation completion 被忽略；
- mixed stale/current report 是 contract abort；
- ObjectRef 丢失后的已提交值 replay 是后续能力；当前 v3.4 只重放尚未 commit 的 actor crash RPC；
- replay 一个 multi-output Grain 时所有 outputs 一起重新生成。

## 18. Batching、Arena 与背压

Batch policy 只改变 ready Grains 的物理 packing：

```text
elastic       同 Call 下可跨 parent packing
parent_bound  按最近 lineage parent 限制 batch
```

第一版保持 v3 的清晰边界：elastic 在单 Arena 内跨 parent；actor capacity 跨 Arena 共享。跨 Arena 合批是独立后续 feature，不能静默改变论文实验口径。

必须有：

- `max_active_grains_per_arena`；
- `max_fanout_per_parent`；
- `max_group_slots_per_arena`；
- `max_blocks_per_arena`；
- per-Call ready/in-flight limits；
- max retries；
- output backpressure；
- source admission high watermark。

## 19. Arena 完成条件

必须同时满足：

```text
source admission closed
receipt queue empty
no unclassified PendingInvocation
no OPEN Shape needed by reachable outputs
no OPEN GroupAccumulator
all ReadyQueues empty
no IN_FLIGHT Grain or DispatchLease
Execution has no pending RPC for Arena
all final output candidates terminal
```

不能仅因暂时没有 RPC 或 ready item 就结束 Arena。

## 20. 组件与依赖

v3.4 的源码目录与概念层次一一对应：

    multigrain_v3_4/
    ├── model.py              # 稳定引用与 outcome；无 Ray
    ├── program.py            # immutable Program 与 ExecutionPlan
    ├── api.py                # Pipeline/RayModule symbolic tracing
    ├── functional.py         # Port 结构关系 API
    ├── protocol.py           # Arena/Executor/Worker 共用 DTO；无 Ray
    ├── runtime/state.py      # 被动语义表
    ├── runtime/engine.py     # Arena 唯一语义写入口
    ├── worker.py             # value-only、列式 Worker ABI
    ├── materialize.py        # 只经 Arena 公开查询物化最终输出
    ├── executor.py           # 唯一 public Executor 与私有 Ray driver/transport
    └── benchmark/

这里刻意不为测试创建 LocalExecutor 或 Backend 层。executor.py 内部的私有对象只表达真实 Ray 物理事实：_RayBlockStore 管 ObjectRef 块，_RayWorkerActor 持有一个 Worker，_ActorSlot 表示 capacity，_Dispatch 表示 generation-fenced pending RPC；Executor 负责 actor pool 生命周期和多 Arena event loop。

依赖方向：

    api/functional -> program/model
    runtime        -> program/model/protocol
    worker         -> model/protocol
    materialize    -> program/runtime/protocol
    executor       -> api/program/runtime/protocol/worker/Ray
    benchmark      -> public API/Executor

关键边界：

- 只有 ArenaEngine 写 Item、Shape、Grain 与 lineage 语义表；
- 只有 Executor 持有 actor handle、ObjectRef、pending RPC 与 capacity；
- Worker 不读取 Program、Arena 或 Driver，只消费 InvocationPlan；
- benchmark 与结果物化不读取 Arena 私有 table；
- metrics 不参与 correctness；结构 Port 不创建 actor、RPC 或 Grain；
- 不使用 actor-to-actor RPC、callback 修改 Arena、全局 mutable registry 或第二套执行状态机。

## 21. Port 行为回归矩阵

### 21.1 Compile-time

- Call 单/多输入输出；
- partial output expand；
- same Port multi-branch；
- same-Domain merge；
- unrelated-Domain rejection；
- explicit broadcast/reduce；
- identical view interning；
- independent expand rejection；
- aligned expand/reduce；
- nested Domain forest；
- invalid filter mask domain；
- invalid reduce ancestry；
- optional only on Call edge；
- import/dependency boundaries。

### 21.2 Runtime semantics

- fanout N=0/1/N；
- all filtered / partial filtered；
- group members vs aligned values；
- multiple group consumers；
- nested 5 binary levels；
- nested 20 unary levels；
- intermediate N=0/drop/failure；
- failed-before-cardinality；
- known-N child failure；
- aligned cardinality mismatch；
- branch failure isolation；
- optional DROPPED vs FAILED；
- atomic multi-output commit；
- hard-limit atomicity；
- stale generation；
- Arena completion audit。

### 21.3 Worker ABI

- scalar/group/missing input reconstruction；
- multi-block groups；
- expand flatten once；
- multiple expanded consumers reuse rows；
- per-record failure isolation；
- driver never reads payload；
- no structural actor/RPC；
- coarse block count and memory bounds。

### 21.4 唯一 Executor 集成

- persistent actor initialized once；
- actor pools only for Calls；
- multi-Arena overlap；
- elastic/parent-bound packing；
- actor crash/replacement；
- generation fencing；
- 逐 Grain replay；
- ordered output delivery。

## 22. Dummy 性能回归

Dummy benchmark 必须使用真实 Python payload/block 和 deterministic CPU work，不用 `sleep` 作为正式回归。

拓扑：

```text
Document
→ Render variable Pages
→ CPU-heavy Page Transform
→ optional Filter
→ ordered Page group
→ Assemble Document
```

配置至少覆盖：

- 1,000+ parents；
- long-tail fanout；
- batch cap 16/32/64；
- parent-bound vs elastic；
- in-flight Arena 1/2/3；
- nested variant Document→Page→Region；
- deterministic bad leaf/retry variant。

指标：

```text
wall/items_per_s
Call RPC count
items_per_RPC/fill
driver/worker RSS
live blocks
shape/group counts
structural actor/RPC = 0
output parity
```

## 23. 现有实验迁移与 gate

### 23.1 MinerU

```python
page_groups = render(pdfs)
pages = F.expand(page_groups)
contents = ocr(pages)
stems = metadata_call(pdfs)

content_groups, ordered_page_groups = F.reduce_aligned(
    contents,
    pages,
    members=contents,
)

return assemble(
    content_groups,
    ordered_page_groups,
    stems,
)
```

必须依次完成：

```text
4 PDF correctness smoke
48 PDF feasibility/memory
368 PDF / 7,072 pages clean elastic regression
需要报告策略 speedup 时，再做 clean parent-bound/elastic paired regression
```

对比保持：v3.4 elastic、v3.4 parent-bound、v3.3、现有 v3、Ray Data、Native。相同
source/model/batch cap/GPU/correctness gate。所有性能运行还必须满足 clean preflight：
启动前无 foreign GPU process、无其他活跃 Ray session；否则只保留 correctness/packing
证据，wall-time 样本标记为 invalid。

### 23.2 Docling core-stage

一层 Page Domain、多输入 page Calls 和 parent group，完整可表达。现有 12 docs/43 pages 四臂 runner 作为 smoke；368 文档矩阵目前是计划，不算历史完成项。

### 23.3 Video A/B/C

- A/B：标准 Video→Frame→Video；
- C：Audio/Frame 独立 child Domains，各自 Reduce 后在 Video Domain merge；
- child-level timestamp/window join 不在第一版 scope。

### 23.4 Nested document/failure

现有文档中 Page→Region 和 deterministic bad leaf 主要是计划，不是已完成真实实验。它们必须先成为 v3.4 conformance/integration tests，再升级为真实 workload。

## 24. 验收顺序

```text
Phase 1  Program/Port/Domain pure-Python tests
Phase 2  Arena semantic transition tests
Phase 3  Worker ABI 与 Arena 直接语义测试
Phase 4  唯一 Executor 的 single-Arena Ray 集成
Phase 5  multi-Arena/recovery/dummy performance regression
Phase 6  MinerU 4 PDF
Phase 7  MinerU 48 PDF
Phase 8  MinerU 368 PDF paired regression
```

每个 Phase 必须保留前一层测试，不允许以端到端通过替代语义层测试。

### 24.1 v3.4 结构验收结果（更新于 2026-08-04）

- v3.4 全量回归为 53 passed，4 warnings，57.87s，全部完整 Pipeline 测试均经过真实 Ray actor、RPC 与 ObjectRef；
- parent-bound/elastic、multi-Arena、nested、bad leaf、multi-output atomicity 和 actor retry 均已通过；
- 顶层只导出 Executor，不再导出 LocalExecutor 或 RayExecutor；
- 生产执行器由 v3.3 的 171 行 local 路径加 482 行 Ray 路径，收敛为 v3.4 单一 485 行 Executor；
- Worker 仍可用进程内 MemoryBlockStore 做 ABI 单测，但该对象只存在于测试文件，不属于系统组件；
- protocol 只保留 BlockRef(handle)，不再为测试与 Ray 分裂两种块引用；
- dummy 的 1,000-parent batching matrix 已在 Ray 路径通过，parent-bound 与 elastic 输出摘要一致；
- 本轮只完成架构与 correctness 回归，尚未运行 v3.4 MinerU 4、48 或 368 PDF 性能实验；
- v3.3 的 clean 368 elastic、parent-bound 与污染诊断继续作为历史基线，不能改写为 v3.4 实验结果；
- v3.4 真实 workload 的数值必须写入独立的 v3_4_regression_matrix.md。

计时口径不变：startup_s 包含 actor/UDF/模型构造，measured_wall_s 只包含 run。Executor 采用即时、work-conserving 聚批，不暴露尚未实现的定时等待参数。性能 gate 仍必须记录 GPU/Ray preflight；资源隔离失败时 measured_wall_s 不具备可比性。

## 25. 最终不变量

1. Call 是唯一计算边界。
2. Port view 不创建 Grain、actor 或 RPC。
3. 一个 Call 在一个 Entity 上至多一个 Grain。
4. Item terminal 只发布一次；冲突重复发布是 contract error。
5. Filter 保持 Domain，只改变 membership。
6. Expand 是创建 child Domain 的唯一方式。
7. Reduce 只能沿 Domain ancestry。
8. 独立 Domains 不自动对齐。
9. Group membership 必须有明确 members Port。
10. Multi-output commit 原子。
11. MISSING 只存在于 worker input。
12. Shape seal 前不发布 aligned child Entities。
13. ItemTable 与 ValueTable 分离。
14. logical identity 不含物理执行信息。
15. Runtime commit 是语义事实的唯一写入口。
16. progress cache 可重建。
17. driver 不读取业务 payload。
18. Program/runtime/execution/worker 依赖只能自顶向下。

开发者的完整心智模型应保持为：

```text
Call          决定做什么
Port          决定传什么
Domain        决定在哪些 Entity 上
Shape         决定 parent-child 数量
Item          记录逻辑值终态
Grain         记录一次 UDF 调用
ValueBinding  决定如何实现该值
ExecutionPlan 决定在哪里、如何物理执行
```

这份模型是 v3.4 实现、测试和论文描述的唯一架构依据。
