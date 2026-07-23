# Multigrain V2.1 架构设计

状态：已被 `multigrain_v2_2_architecture.md` 取代的历史开发架构。

本文档仅保留 V2.1 设计演进记录，不再指导实现。若与 V2.2 冲突，必须以
`multigrain_v2_2_architecture.md` 为唯一 authority。

## 1. 核心目标与范围

Multigrain V2.1 将类型化关系语义编译为稳定的语义 WorkUnit。同一组 WorkUnit
统一驱动：

- 多输入 identity 对齐；
- actor batch 形成；
- identity 与 provenance 物化；
- 错误定位和二分隔离；
- replay closure；
- node invocation 原子发布。

首版语言限定为有限、单 microbatch 内闭合的 DAG：

```text
Map / Filter / Expand / Reduce / Relate
```

`Select` 只是 authoring macro，完整降低为 `Map + Filter`。

五类原语覆盖的直接父关系和基数形态为：

```text
Map      1 -> 1，复用 identity
Filter   1 -> 0/1，复用 identity
Expand   1 -> N，派生 child identity
Reduce   N -> 1，复用 anchor identity
Relate   ordered parent tuple -> relation identity
```

首版不覆盖：

- 带 optional parent 的 Union/outer relation；
- 跨 microbatch window、watermark 和 state；
- 具有业务含义的全局 sort/rank；
- recursion/fixpoint；
- checkpoint/resume；
- node 发布后的增量修补。

## 2. 与原版 RayOrch 的关系

V2.1 有意复用原版 RayOrch 已验证的控制面形态：

```text
Pipeline.forward tracing
  -> 静态 compiled DAG
  -> 每 node 常驻 actor replicas
  -> 有界 microbatch admission
  -> driver 同步 ray.wait
  -> completion-driven pipeline overlap
```

可借鉴的原版实现：

- `rayorch/dag/pipeline.py`：`forward()` 符号 tracing；
- `rayorch/dag/compiler.py`：参数绑定、节点命名、拓扑校验；
- `rayorch/dag/executor.py`：admission、ready queue、`ray.wait`、释放；
- `rayorch/runtime/executor.py`：Executor 生命周期和启动失败回滚；
- `rayorch/ray_module.py`：每 actor 常驻一个 UDF/model 实例；
- `rayorch/env_registry.py`：Ray runtime environment。

但不能复用原版执行合同：

- 原版一次 logical call 会 fanout 到全部 replicas；
- 原版 node `max_inflight` 会把多个 logical calls 排进 actor mailbox；
- 原版 collector 会把完整 payload `ray.get` 回 driver；
- 原版 dispatch 假定等长 Python list columns；
- 原版 runtime 在 actor 内生成 row lineage。

V2.1 的三个并发参数严格区分：

```text
replicas       = 一个 node 的常驻 actor 并行容量
batch_size     = 一次 actor RPC 目标包含的 WorkUnit 数
max_inflight   = 一次 stream run 中活跃 microbatch 数
```

一个 replica 同时至多执行一个 actor batch。actor mailbox 不属于容量模型。

因此，V2.1 不是重新发明 Ray orchestration，而是在原版最小控制骨架上替换：

```text
node scheduling semantics
payload protocol
identity/provenance runtime
failure/recovery contract
```

论文贡献也不应包含 persistent actor、`ray.wait` 或普通 DAG overlap。

## 3. 包结构与依赖方向

新实现位于：

```text
rayorch/experimental/multigrain_v2/
├── __init__.py
├── api.py
├── authoring.py
├── graph.py
├── identity.py
├── data.py
├── provenance.py
├── work.py
├── kernel.py
├── kernels/
│   ├── __init__.py
│   ├── map_filter.py
│   ├── expand_reduce.py
│   └── relate.py
├── execution/
│   ├── arena.py
│   ├── materialize.py
│   ├── node_runtime.py
│   ├── executor.py
│   └── metrics.py
├── ray/
│   ├── actor.py
│   └── resources.py
└── errors.py
```

这是源码职责划分，不要求每个小类型都独占一个文件。相关的小对象应放在一起，避免
Java 式文件爆炸。

依赖方向：

```text
identity <- graph
identity + graph <- data + provenance + work
graph + data + work <- kernel + builtin kernels
所有语义模块 <- execution
kernel + work <- Ray actor adapter
execution + Ray resources <- public Executor
```

首版没有：

- backend abstraction；
- LocalExecutor；
- manager actor；
- storage plugin；
- lease manager；
- durable graph serializer；
- 通用 event bus。

Ray 是强制依赖。纯函数直接单测不等于再造一个 Local execution engine。

## 4. 用户侧 Module 设计

### 4.1 保留原版 RayModule 的优点

原版 RayModule 最值得保留的模式是：

- 用户提供 Python UDF class；
- driver 只保存构造参数，不加载模型；
- 每个 prepared actor 构造并常驻一个 UDF/model 实例；
- module 声明 replicas 和 GPU 资源。

以下原版 API 不进入 V2.1：

- `module.remote()`；
- `module.gather()`；
- public actor handle list；
- dispatch/collect function；
- per-node `max_inflight`；
- eager direct execution。

源码级原型：

```python
@dataclass(frozen=True, slots=True)
class OperatorFactory:
    op_cls: type
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...]

    def build(self) -> object:
        return self.op_cls(*self.args, **dict(self.kwargs))


@dataclass(frozen=True, slots=True)
class ReplicaResources:
    cpus: float = 1
    gpus: float = 0
    custom: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    replicas: int
    batch_size: int
    resources: ReplicaResources
    runtime_env: str | None
    max_retries: int
    timeout_s: float | None


class PrimitiveModule:
    operation: Operation
    factory: OperatorFactory
    config: ModuleConfig
    num_outputs: int

    def __call__(self, *ports: Port, **named_ports: Port):
        return current_graph_builder().add_call(self, ports, named_ports)
```

`PrimitiveModule` 只是被 GraphBuilder 读取的 authoring state：

- 不拥有 actor；
- 不负责调度；
- 不直接执行；
- 不保存 runtime state。

首版一个 PrimitiveModule instance 只能在 `forward()` 中形成一个 CompiledNode。
重复调用同一 instance 会产生 actor-pool/state-sharing 歧义，compiler 直接拒绝。若同一
UDF class 需要出现在两个 call sites，用户应声明两个 module instances：

```python
self.encode_left = mg.Map(Encoder, ...)
self.encode_right = mg.Map(Encoder, ...)
```

两者拥有独立 NodeRuntime/replica pool。未来若确有共享超大只读模型的需求，应单独
设计显式 shared residency，而不能复用原版 RayModule 的隐式 actor-handle sharing。

用户 API 保持简洁：

```python
self.ocr = mg.Map(
    OcrUdf,
    model_name=MODEL,
    replicas=4,
    batch_size=16,
    cpus_per_replica=1,
    gpus_per_replica=1,
    env="ocr",
    max_retries=1,
    timeout_s=600,
)
```

调度参数由 PrimitiveModule 消费并降低为 `ModuleConfig`，`model_name` 等其余参数
按调用顺序冻结进 OperatorFactory。`env` 继续复用原版 EnvRegistry 的命名环境。

output arity 规则保持显式且简单：

- Map/Expand/Reduce/Relate 默认 `num_outputs=1`；
- 多 output UDF 必须显式传 `num_outputs=N`；
- return annotation 只做一致性校验，不是唯一 authority；
- Filter output 数由被过滤的 target ports 数决定；
- `.sink(...)` 固定 `num_outputs=0`。

基础参数必须在 authoring 时校验：

```text
replicas >= 1
batch_size >= 1
cpus_per_replica >= 0
gpus_per_replica >= 0
max_retries >= 0
timeout_s is None or timeout_s > 0
```

### 4.2 Pipeline input 不是 Source primitive

```python
class ParseDocuments(mg.Pipeline):
    def forward(self, paths):
        pages = self.read_pages(paths)
        contents = self.ocr(pages)
        self.write(mg.group_by(paths, contents))
```

`Executor.run(..., inputs=...)` 将 path/key columns admission 为 CompiledInput
对应的 source PortData。真正读取 PDF/图片的操作发生在第一个 actor primitive 内。

driver 不会因为输入是文件路径就读取文件 payload。

因此 core 中没有：

```text
source=True
SourceOp
source actor
```

默认每个 `forward()` 参数是独立的 PositionKeys source domain。需要让多个 source
columns 表示同一批实体时，Pipeline 必须显式声明：

```python
class ScoreImages(mg.Pipeline):
    def __init__(self):
        super().__init__()
        self.input_group(
            "images",
            "metadata",
            key_mode=mg.ProvidedKeys(),
        )

    def forward(self, images, metadata):
        return self.score(images, metadata)
```

ProvidedKeys 输入使用显式 keyed column：

```python
inputs = {
    "images": mg.keyed(image_values, item_keys),
    "metadata": mg.keyed(metadata_values, item_keys),
}
```

`input_group()` 只引用 `forward()` 参数名；compiler 验证名称唯一归组，并为该组生成
一个 DomainId。authoring label 不进入 runtime identity。

### 4.3 zero-output terminal writer

写出节点仍是五类原语之一，只是没有 value output：

```python
self.write = mg.Reduce.sink(
    AssembleAndWrite,
    output_dir=output_dir,
)
```

`.sink(...)` 只是一层 authoring sugar，编译后为：

```text
ReduceOp + output_ports=()
```

core 中没有 `sink: bool` 和 `SinkOp`。

编译规则：

- zero-output node 必须是 graph leaf；
- UDF 成功返回 `None`；
- actor 只返回小型 acknowledgement；
- 不创建 dummy None column；
- 不创建没有消费者的 PortData；
- 即使 `forward()` 返回 `None`，该节点仍是 execution target；
- 产生 value 但未被消费、也未被 return 的 leaf 应报 authoring error。

首版不保证 external write exactly-once。terminal writer 必须对同一个 WorkUnit
可安全重试，例如：

- 稳定业务 key 覆盖写；
- 临时文件加 atomic rename；
- idempotent upsert。

### 4.4 Select 和普通扩展

`Select` 在 tracing 时完整降低：

```text
Select(inputs)
  -> Map 产生 mask + annotations
  -> Filter.by_mask(inputs, annotations, mask)
```

它不产生 SelectOp、kernel、WorkUnit type 或 provenance table。

普通扩展优先写成组合现有五类 primitive 的 Python authoring macro。真正新增语义时，
才使用第 10 节的内部 PrimitiveKernel 合同。

## 5. 唯一 CompiledGraph

临时 GraphBuilder 在编译后丢弃。

```python
@dataclass(frozen=True, slots=True)
class CompiledInput:
    port: CompiledPort


@dataclass(frozen=True, slots=True)
class CompiledInputGroup:
    domain: DomainId
    ports: tuple[PortId, ...]
    key_mode: PositionKeys | ProvidedKeys


@dataclass(frozen=True, slots=True)
class CompiledPort:
    id: PortId
    domain: DomainId
    grain: str
    relation: OutputRelation


@dataclass(frozen=True, slots=True)
class CompiledNode:
    id: NodeId
    operation: Operation
    factory: OperatorFactory
    input_ports: tuple[PortId, ...]
    output_ports: tuple[CompiledPort, ...]
    replicas: int
    batch_size: int
    resources: ReplicaResources
    runtime_env: str | None
    max_retries: int
    timeout_s: float | None


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    inputs: tuple[CompiledInput, ...]
    input_groups: Mapping[DomainId, CompiledInputGroup]
    nodes: tuple[CompiledNode, ...]  # topological order
    outputs: tuple[PortId, ...]
```

关键收敛：

- 删除 IdentityGroupId；
- source input group 直接以 DomainId 为键；
- 不保留 authored/verified/compiled 三套 graph；
- 不保存 durable graph schema/version。

dependencies、consumers、producer lookup、leaf lookup 和 port index 都是可重建索引，
不是重复语义事实。

GraphBuilder 必须支持 `forward() -> None`。执行完成条件依据 graph leaves，而不只依据
returned output ports。

## 6. Identity 与 source admission

```python
@dataclass(frozen=True, slots=True)
class RowRef:
    batch: BatchId
    port: PortId
    row: RowId


@dataclass(frozen=True, slots=True)
class IdentityKey:
    batch: BatchId
    domain: DomainId
    entity: EntityId
```

`RowRef` 表示一个具体 port representation；`IdentityKey` 表示逻辑实体。

### PositionKeys

- identity 来自调用者的 admission logical ordinal；
- 同 input group 内各 ports 必须等长且 ordinal 对齐；
- 调用者改变逻辑顺序会有意改变 identity；
- actor、shard、RPC、retry 和 completion order 不影响 identity。

### ProvidedKeys

- key 必须能 canonical encode，并在每个 source port 内唯一；
- 同一 group 的 ports 必须具有完全相同的 key set；
- admission 按 canonical key bytes 对齐；
- identity 来自 provided key，不来自物理输入位置。

不同 source domains 不会因为位置相同而自动对齐。

所有 ID 编码必须使用带类型 tag 的 canonical bytes 和稳定 digest，禁止使用 Python
`hash()`、对象地址、actor/shard/rank 或随机 completion 信息。

唯一构造入口：

```python
def source_entity(domain: DomainId, key: CanonicalValue) -> EntityId:
    ...


def child_entity(
    domain: DomainId,
    parent: IdentityKey,
    sibling_ordinal: int,
) -> EntityId:
    ...


def relation_entity(
    domain: DomainId,
    ordered_parents: tuple[IdentityKey, ...],
) -> EntityId:
    ...
```

Map/Filter 直接复用 primary EntityId，Reduce 直接复用 anchor EntityId，不再调用
新 ID constructor。Compiled DomainId 也必须由稳定 graph topology/slot 编码得到，以便
同一 Pipeline 的重复 compile 不因随机 UUID 改变 domain。

## 7. PortData、ValueShard 与 RunArena

一个 accepted actor batch 对每个 output port 直接产生一个完整 value shard：

```python
@dataclass(frozen=True, slots=True)
class ValueShard:
    ref: ray.ObjectRef
    row_count: int


@dataclass(frozen=True, slots=True)
class PortData:
    port: PortId
    entities: tuple[EntityId, ...]
    value_shards: tuple[ValueShard, ...]
    provenance: ProvenanceTable
```

逻辑 value column 为：

```text
concat(value_shards[0], value_shards[1], ...)
```

必须满足：

```python
sum(shard.row_count for shard in value_shards) \
    == len(entities) \
    == len(provenance)
```

首版不引入：

- ImmutableValueColumn protocol；
- ShardedValueColumn wrapper；
- ValueSegment；
- chunk lease；
- object manager。

完整 shard 模型依赖以下已确认规则：

- 一个 actor batch 只包含一个 invocation 的连续 WorkUnit range；
- binary split 保持连续子区间；
- 一个 actor RPC 不跨 microbatch co-batch。

每个 active microbatch 拥有一个 RunArena：

```python
class RunArena:
    batch: BatchId
    ports: dict[PortId, PortData]
    failures: dict[WorkUnitRef, TerminalFailure]
    # 未发布 invocation state 和 rebuildable indexes 保持 private。
```

RunArena 是 active-run 唯一 semantic store。索引只是 cache。

## 8. 类型化 direct provenance

每个 PortData 直接持有一个表：

```python
ProvenanceTable = (
    SourceRows
    | AliasRows
    | ChildRows
    | AggregateRows
    | RelatedRows
)
```

- SourceRows：admitted entities；
- AliasRows：Map/Filter 的 direct source RowRefs；
- ChildRows：一个 parent RowRef 加 sibling ordinal；
- AggregateRows：anchor 加 selector-member CSR；
- RelatedRows：每个声明 role 一个 direct parent RowRef。

统一结构投影：

```python
def parent_edges(row: RowRef) -> tuple[ParentEdge, ...]:
    ...
```

ancestor、dependency、role 和 aggregate-member 查询都是 direct edges 上的 sealed
policy。不会把 transitive ancestor dict 或完整 path 复制进每一行。

## 9. WorkUnit 与 payload layout

```python
@dataclass(frozen=True, slots=True)
class InvocationRef:
    batch: BatchId
    node: NodeId


@dataclass(frozen=True, slots=True)
class RowCoordinate:
    entity: EntityId


@dataclass(frozen=True, slots=True)
class FiberCoordinate:
    anchor: EntityId


@dataclass(frozen=True, slots=True)
class JoinKeyCoordinate:
    key: CanonicalValue


@dataclass(frozen=True, slots=True)
class WholeInvocationCoordinate:
    pass


WorkCoordinate = (
    RowCoordinate
    | FiberCoordinate
    | JoinKeyCoordinate
    | WholeInvocationCoordinate
)


@dataclass(frozen=True, slots=True)
class WorkUnitRef:
    invocation: InvocationRef
    coordinate: WorkCoordinate


@dataclass(frozen=True, slots=True)
class WorkRange:
    start: int
    stop: int
```

没有 opaque WorkUnitId 或 WorkUnit registry。InvocationRef 已经提供 batch/node，
coordinate 只保留最小语义 discriminator。WorkRange 是 ordered WorkUnit tuple 上的
半开物理区间，不进入 identity。

payload 将物理 value location 与语义 access shape 分开：

```python
@dataclass(frozen=True, slots=True)
class ShardTake:
    ref: ray.ObjectRef
    selector: slice | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PortSlice:
    takes: tuple[ShardTake, ...]


@dataclass(frozen=True, slots=True)
class ShardPayload:
    slices: tuple[PortSlice, ...]
    layout: WorkLayout


@dataclass(frozen=True, slots=True)
class RowAligned:
    work_unit_count: int


@dataclass(frozen=True, slots=True)
class FiberGrouped:
    # 每个 grouped input 一组 CSR offsets，长度均为 work_unit_count + 1。
    member_offsets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class RelationClosure:
    role_sizes: tuple[int, ...]
    # key Relate 为每个 WorkUnit 的 flat tuple CSR offsets；
    # custom WholeInvocation Relate 为 None。
    tuple_offsets_by_work_unit: tuple[int, ...] | None


WorkLayout = RowAligned | FiberGrouped | RelationClosure


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    invocation: InvocationRef
    work_units: tuple[WorkUnitRef, ...]
    payload: ShardPayload
    preterminal_failures: tuple[TerminalFailure, ...]
```

driver 将 semantic RowIds 转换为 ObjectRef selectors。下游 actor 拉取 unique refs，
在 actor 内执行 slice/take。大 payload 不进入 driver。

`ShardPayload.slices` 按 CompiledNode input_ports 顺序排列。WorkLayout 只描述本次调用
如何把 flat value columns 解释成 row/fiber/relation outer dimension，不重复保存
identity 或 provenance。

`InvocationPlan.work_units` 只包含需要执行的 ready WorkUnits；已由 upstream failure
确定为 Suppressed 的 units 放入 `preterminal_failures`。`ShardPayload` 提供一个
纯函数 `take_work_range(WorkRange)`，按 layout offsets 产生本次 actor RPC 的子 payload，
因此 NodeRuntime 不需要理解 primitive。

- Map/Filter/Expand 通常走 contiguous fast path；
- Reduce/Relate 的 fiber/role tuples 可能使用 sparse selectors。

Ray 通常以整个 referenced object 为传输单位，因此从一个大 ValueShard 中只选择少量
rows 仍可能搬运整个 shard。首版通过 actor `batch_size` 限制 shard 粒度，并在实验中
测量 transfer amplification；没有测量证据前不预先引入 shuffle/exchange service。

## 10. 唯一 PrimitiveKernel 扩展点

一个 semantic primitive 同时跨越 compiler、planner、worker ABI、materializer 和
failure semantics。若这些职责分散在多张 handler registry 中，会重新产生飞线。

V2.1 只保留一张内部静态 registry：

```python
@dataclass(frozen=True, slots=True)
class PrimitiveKernel:
    operation_type: type[Operation]
    validate: ValidateFn
    plan: PlanFn
    invoke: InvokeFn
    materialize: MaterializeFn


CORE_KERNELS = MappingProxyType({
    MapOp: MAP_KERNEL,
    FilterOp: FILTER_KERNEL,
    ExpandOp: EXPAND_KERNEL,
    ReduceOp: REDUCE_KERNEL,
    RelateOp: RELATE_KERNEL,
})
```

四个职责：

- `validate`：静态 relation/domain/cardinality contract；
- `plan`：ordered WorkUnits、WorkLayout 和已知 suppression；
- `invoke`：worker-side value-column UDF ABI，只产 raw values/evidence；
- `materialize`：driver 验证 cardinality，生成 EntityIds、typed provenance、
  failures 和 OutputBundle。

NodeRuntime 统一负责：

- batching；
- token；
- timeout；
- retry/split；
- actor capacity；
- atomic publication。

registry 是 private 且在一个安装版本内固定，不提供 public
`register_primitive()`。新增 core primitive 必须同时提供：

- Operation type；
- 完整 PrimitiveKernel；
- 必要的数据模型扩展；
- shared conformance tests；
- 对形式化边界的明确修改。

## 11. UDF ABI 与唯一 semantic authority

UDF 只接收 batched value columns，不接收：

```text
RowRef
IdentityKey
EntityId
WorkUnitRef
provenance
attempt token
```

### Map

```python
def run(self, col_a, col_b, *more_columns):
    return output_columns
```

每个 output 与 outer input dimension 一一对齐。

### Filter

```python
def run(self, target_a, target_b, *more_targets):
    return bool_mask
```

`Filter.by_mask()` 消费已有 mask。每个 target 产生一个 subset output，mask 是 control
input，不是 output。

### Expand

```python
def run(self, first_column, *more_columns):
    return nested_output_columns
```

outer dimension 是 parent WorkUnits；同一个 parent 下所有 outputs child count 相同。

### Reduce

```python
def run(self, anchor_values, grouped_a, grouped_b, *more_grouped):
    return output_columns
```

outer dimension 是 anchor/fiber WorkUnits；每个成功 anchor 产生一行。

### Key Relate

引擎形成 unique ordered parent tuples，role columns 和 output columns 与 tuples 对齐。

### Custom Relate

UDF 接收独立长度的 role columns，返回 role-local parent indexes 和对齐的 values。
duplicate ordered parent tuple 是 ContractViolation。

统一硬合同：

> 一个 WorkUnit 的业务输出不能依赖同一 actor RPC 中被放入了哪些其他 WorkUnits。

actor 只返回最小 raw evidence：

- Map/Filter/Expand/Reduce：无额外 evidence；Filter survivor 与 Expand child count
  已完整编码在 per-WorkUnit output offsets 中；
- key Relate：parent tuples 已在 plan 中；
- custom Relate：role-local parent indexes。

只有 driver materializer 生成 EntityId 和 typed provenance。

## 12. Ray P2P multi-return 协议

actor 不能：

```python
def bad_actor_run(self, values):
    ref = ray.put(values)
    return ref
```

这种 nested ObjectRef 的 owner 是 actor process；actor replacement 后可能出现
`OwnerDiedError`，也是 Ray 官方明确反对的模式。

CompiledNode 已知静态 output arity，因此 actor method 使用 Ray multiple returns：

```python
refs = actor.run.options(
    num_returns=1 + len(node.output_ports),
).remote(batch_attempt, shard_payload)

manifest_ref, *value_refs = refs
```

actor 直接返回：

```python
def run(self, batch_attempt, shard_payload):
    ...
    return raw_manifest, output_column_0, output_column_1
```

task-return ObjectRefs 由调用者/driver 持有。driver 只 `ray.get(manifest_ref)`，
value refs 留在 Ray object store，并直接路由给下游 actor。

```python
@dataclass(frozen=True, slots=True)
class BatchAttemptToken:
    value: UUID


@dataclass(frozen=True, slots=True)
class NoEvidence:
    pass


@dataclass(frozen=True, slots=True)
class CustomRelationEvidence:
    # 每个 role 一列 local parent indexes；各列与 output rows 等长。
    parent_indexes_by_role: tuple[tuple[int, ...], ...]


RawEvidence = NoEvidence | CustomRelationEvidence


@dataclass(frozen=True, slots=True)
class RawBatchManifest:
    token: BatchAttemptToken
    offsets_by_work_unit: tuple[int, ...]
    evidence: RawEvidence


@dataclass(frozen=True, slots=True)
class PendingBatch:
    actor_index: int
    invocation: InvocationRef
    work_range: WorkRange
    token: BatchAttemptToken
    manifest_ref: ray.ObjectRef
    value_refs: tuple[ray.ObjectRef, ...]
    deadline_ns: int | None


@dataclass(frozen=True, slots=True)
class AcceptedBatch:
    work_range: WorkRange
    manifest: RawBatchManifest
    value_refs: tuple[ray.ObjectRef, ...]
```

zero-output terminal writer 使用 `num_returns=1`，manifest 本身就是小型 ack。

worker exception 通过 `manifest_ref` 暴露。失败 actor call 没有可信 partial outputs。

`offsets_by_work_unit` 是所有 output ports 共享的 CSR offsets，长度为本 actor batch
WorkUnit 数加一。Map/Reduce 等一行输出是连续整数；Filter/Expand/Relate 使用实际
0/N cardinality。多 output ports 若不能共享同一 offsets，整个 batch 是
ContractViolation。zero-output terminal writer 使用空 offsets。

driver 仍会持有 EntityIds、typed provenance、offsets 和 custom relation parent
indexes。这些 metadata 至少是 `O(rows + direct_edges)`，并非零成本。V2.1 首版以
microbatch/max_inflight 限制其 active lifetime，不声称 metadata byte-bounded。
多节点实验必须单独测量 driver metadata CPU/RSS；若其成为瓶颈，再将同一列式语义
布局分区，而不是提前增加第二套 distributed provenance authority。

## 13. 五类 primitive 的 identity/provenance 规则

### Map

- 按 primary input EntityId 规划；
- 所有输入按同一 authoritative WorkUnit sequence gather；
- output identity 等于 primary identity；
- 所有 aligned branches 都保留为 direct representation/dependency parents。

### Filter

- 与 Map 一样规划；
- false 是合法 zero cardinality，不是 failure；
- survivor 保留 target identity 和已有 order evidence；
- 每个 target output 都是对应 target 的 SubsetOf。

### Expand

- 每 parent entity 一个 WorkUnit；
- child identity 来自 parent identity + sibling ordinal；
- output slot 不进入 identity；
- 多 output ports 共享 child domain 和 EntityId vector。

### Reduce

- 每 anchor fiber 一个 WorkUnit；
- complete empty group 合法；
- 因已记录 upstream failure 而不完整的 group 被 Suppressed；
- 成功 output 复用 anchor identity；
- provenance 记录 anchor 与完整 selector membership。

### Relate

- key Relate 每个 complete canonical join-key partition 一个 WorkUnit；
- custom Relate 默认 WholeInvocation；
- 每个 output row 对每个 role 恰有一个 parent；
- duplicate ordered parent tuples 非法；
- relation identity 来自 ordered role-parent identities。

### sparse failure propagation

fail-fast 是默认值；它在产生 sparse downstream result 前终止 run。

fail-closed 下：

- Map/Filter 按 EntityId 只 suppress 缺少 required aligned input 的 row；
- Expand 的 direct parent failure 记录 parent Row WorkUnit，不虚构未知 child rows；
- 合法 zero-child Expand 没有 TerminalFailure；
- Reduce 的 required member 因 upstream failure 缺失时 suppress 整个 anchor fiber；
- Relate 在 join key 尚无法确定时保守 suppress WholeInvocation；
- join-key partitions 已确定后，direct relation failure 可定位到对应 JoinKey WorkUnit；
- custom Relate 保守使用 WholeInvocation；
- suppression 只能引用已存在的 upstream terminal causes。

## 14. Diamond 对齐与按需顺序

diamond/fan-in 的正确性来自 IdentityKey join，而不是物理 zip：

```text
primary sequence: A2, A1, A3
other branch:     B1, B3, B2

gathered:         (A2,B2), (A1,B1), (A3,B3)
```

primary input 决定 WorkUnit sequence，其他 aligned inputs 通过 IdentityKey lookup。

默认 unordered relation 不做全局 canonical sort。发布时按照 planned WorkUnit ranges
拼接 accepted shards，因此 actor completion order 不会影响本次构造顺序。

只有 relation evidence 明确提供顺序时才保证 ordered equality：

- source admission ordinal；
- Alias/Subset 继承顺序；
- Expand sibling ordinal；
- Reduce ownership/anchor 顺序；
- 未来显式 order key。

Relate 和 ByRole member set 默认无序。没有 order evidence 时，跨执行比较按
IdentityKey set，而非强制序列相等。

## 15. NodeRuntime

每个 CompiledNode 对应一个普通 driver Python object：

```python
class NodeRuntime:
    node: CompiledNode
    actors: tuple[ActorHandle, ...]
    idle_actors: deque[int]
    invocations: dict[InvocationRef, InvocationState]
    round_robin: deque[InvocationRef]

    def enqueue(self, plan: InvocationPlan) -> None: ...
    def submit_one(self) -> PendingBatch | None: ...
    def on_manifest(
        self,
        pending: PendingBatch,
        raw: RawBatchManifest,
    ) -> None: ...
    def on_failure(
        self,
        pending: PendingBatch,
        error: BaseException,
    ) -> None: ...
```

NodeRuntime 没有：

- thread；
- asyncio loop；
- 自己的 `ray.wait`；
- manager actor；
- actor mailbox backlog。

Executor 拥有唯一 event loop。

每个 invocation 持有 ordered WorkUnit plan 和 contiguous range deque。replica 空闲时，
NodeRuntime 在 invocations 间 round-robin，并提交最多 `batch_size` 的一个 range。

多个 microbatches 可在同一 node 上交错执行，但一个 actor RPC 不会 co-batch 不同
invocations。这使 complete ValueShard、token 和 binary split 都保持简单。

## 16. Executor 主循环

```python
class Executor:
    def prepare(self, pipeline: Pipeline) -> None: ...

    def run(
        self,
        pipeline: Pipeline,
        inputs: Mapping[str, object],
        *,
        failure_mode: Literal["fail_fast", "fail_closed"] = "fail_fast",
    ) -> RunResult: ...

    def run_stream(
        self,
        pipeline: Pipeline,
        microbatches: Iterable[Mapping[str, object]],
        *,
        max_inflight: int,
        failure_mode: Literal["fail_fast", "fail_closed"] = "fail_fast",
    ) -> Iterator[RunResult]: ...
```

内部流程：

```text
admit microbatches，数量不超过 max_inflight
  -> 为 newly-ready node invocation 执行 kernel.plan
  -> 各 NodeRuntime 用 idle replicas 提交 batch
  -> ray.wait manifest refs，并检查 deadlines
  -> ray.get ready small manifests
  -> accept / retry / split / abort
  -> invocation settled 后 driver materialize
  -> atomic publish OutputBundle
  -> unlock consumers
  -> 释放所有消费者已结束的 intermediates
  -> yield lightweight RunResult
```

不引入 Python asyncio 或 async public API。Ray actor execution 和 object movement
本身仍然异步。

## 17. 资源启动与生命周期

graph preparation 是事务：

```text
compile + validate
  -> 计算所有 node replicas 的完整资源计划
  -> 创建 graph-level placement group
  -> 在 reserved bundles 创建全部 actors
  -> 每个 actor 构造常驻 UDF/model
  -> ping 全部 replicas
  -> 全部成功后才 admission 第一个 BatchId
```

任一步骤失败：

```text
kill 已创建 actors
remove placement group
clear NodeRuntimes
raise PreparationError
```

不会自动缩小 replicas，也不会留下“半个 graph 可运行”状态。

运行中 actor replacement 使用原 placement-group bundle。

`Executor.close()`：

- 拒绝新 run；
- cancel active runs；
- kill owned actors；
- remove owned placement groups；
- 释放内部 arena refs；
- 不调用全局 `ray.shutdown()`。

## 18. Attempt、retry、split 与 timeout

只有 coordinator 管理 attempts。primitive actor calls 关闭 Ray 自动 task retry。

### ContractViolation

driver 检测到 output/evidence/cardinality 违反合同后立即 abort，不 retry、不 split。

### InfrastructureFailure

actor crash 或 transport failure：

```text
mark token stale
replace actor
exact range retry，最多 max_retries
耗尽后 abort
```

没有数据局部性证据时不做 binary split。

### BadRecordError

`BadRecordError(index=i)` 中的 index 指 UDF outer batch dimension：

- Map/Filter：row WorkUnit；
- Expand：parent WorkUnit；
- Reduce：anchor/fiber WorkUnit；
- key Relate：flat tuple，再通过 offsets 定位 JoinKey WorkUnit；
- custom Relate：WholeInvocation。

index 越界是 ContractViolation。

actor exception 不携带可信 partial outputs。即使 index 已定位，失败 batch 中的健康
WorkUnits 在 fail-closed 下也必须按 bad unit 左右两侧的 contiguous ranges 重新
执行；bad unit 直接 terminal。fail-fast 则立即 abort，不继续执行该 batch peers。

没有 index 时，按 contiguous WorkUnit ranges 二分到 singleton。Fiber 和 JoinKey
closure 内部永远不拆。

只有显式 BadRecordError 在 fail-closed 下可成为 PermanentlyMissing。

### Generic UDF exception

exact range retry 最多 `max_retries`，仍失败则二分 contiguous WorkUnit ranges。
进入 isolation phase 后，derived child range 因 generic UDF exception 失败时直接
继续二分，不为每一层重新获得一份 `max_retries`，从而保持有限 amplification。
singleton 仍是 generic exception 时 abort，绝不静默转成 missing。

`max_retries` 的准确含义是“进入 isolation 前，同一原始 range 可额外 exact replay
多少次”。InfrastructureFailure 不提供 UDF/data outcome；它在任何 derived range 上
仍按同一有限 infrastructure budget exact retry，耗尽后直接 abort。

### Timeout

同步 actor UDF 无法依赖 cooperative `ray.cancel`。deadline 到达后：

```text
mark token stale
  -> ray.kill actor
  -> replacement + prepare
  -> retry/split range
  -> reject late manifest
```

public recovery API 只有：

- primitive `max_retries`；
- primitive `timeout_s`；
- run-level `failure_mode`。

split depth、retry priority、actor action 和 queue policy 都是固定 runtime 规则。

## 19. TerminalFailure 与原子发布

```python
@dataclass(frozen=True, slots=True)
class TerminalFailure:
    work_unit: WorkUnitRef
    outcome: PermanentlyMissing | Suppressed
    code: FailureCode
    message: str
    causes: tuple[WorkUnitRef, ...]


@dataclass(frozen=True, slots=True)
class OutputBundle:
    invocation: InvocationRef
    ports: tuple[PortData, ...]
    failures: tuple[TerminalFailure, ...]
```

规则：

- PermanentlyMissing 是 direct explicit BadRecord，无 causes；
- Suppressed 有一个或多个 upstream causes；
- 一个 WorkUnit 至多一个 TerminalFailure；
- 没有 failure explanation 的缺失数据是 ContractViolation；
- zero-output terminal node 发布 empty ports tuple；
- 一个 invocation 至多发布一次；
- downstream 只能观察完整 published bundle。

不引入：

- OutputBundleId；
- candidate state；
- semantic digest；
- mutable disposition；
- post-publication patch。

## 20. RunResult 与 observability

```python
@dataclass(frozen=True, slots=True)
class RunResult:
    batch: BatchId
    outputs: tuple[FinalPortView, ...]
    failures: tuple[TerminalFailure, ...]
    metrics: RunMetrics
```

`FinalPortView` 只是 detached final PortData 的薄只读视图，不是第二套 port data
model。具体 collect convenience 在实现期再定，不进入 semantic architecture。

这里的 detached 只表示不依赖 Executor、RunArena 或 replica actors。显式 returned
ports 仍持有 Ray ObjectRefs，因此必须在 Ray runtime 存活时读取；`Executor.close()`
不会使它们失效，但用户显式 `ray.shutdown()` 后不再保证可用。

常见 terminal-writer pipeline：

```text
outputs == ()
```

writer ack 后即可释放全部 payload refs，RunResult 只保留 metrics 和 terminal
semantic failures。

`RunResult.failures` 包含该 microbatch 执行图中产生的全部 TerminalFailure。已经恢复
的 retries、timeouts、stale results 和 actor replacements 只进入 metrics/events。

正常运行默认不写文件：

- metrics 在内存聚合；
- log 只含 compact run summary 和 sampled error；
- full provenance/events 仅由显式 experiment/debug 配置导出；
- RunResult 默认不持有完整 lineage。

## 21. 必须保持的系统不变量

1. Domain 只在 compile 阶段分配一次。
2. Input group 直接由 DomainId 标识，不存在第二个 identity-group ID。
3. PortData 的 entity/value/provenance cardinality 始终相等。
4. 每个 RowRef 都能在当前 RunArena 中解析。
5. shard、actor、RPC、attempt、completion order 不进入 EntityId/provenance。
6. actor 代码不生成 EntityId 或 typed provenance。
7. 一个 WorkUnit 的业务输出与 actor-batch peers 无关。
8. 一个 actor 同时至多一个 active RPC。
9. 一个 run 同时至多 `max_inflight` 个 active microbatches。
10. 一个 actor RPC 只含一个 invocation 的一个 contiguous range。
11. driver 不 `ray.get` 大型 intermediate value columns。
12. actor values 通过 direct task returns 返回，不返回 actor-owned `ray.put` refs。
13. accepted batch token 必须仍为 current；stale manifest 整体丢弃。
14. 每个 WorkUnit 在发布前必须成功 settle 或有一个 terminal outcome。
15. 每个 invocation 至多发布一个完整 OutputBundle。
16. downstream 只能读取 published bundles。
17. diamond inputs 按 IdentityKey gather，绝不直接 zip 物理 columns。
18. 只有存在 relation order evidence 时才主张 ordered equality。
19. generic UDF/backend/contract failure 不会静默变成 missing data。
20. zero-output terminal writer 的自动 retry 依赖公开的 idempotence contract。

## 22. 首版明确不做

- LocalExecutor 或 multi-backend abstraction；
- 同一 actor RPC 内 cross-microbatch co-batching；
- StageBatcher、byte/wait flush 或 hidden mailbox queue；
- byte-perfect admission control；
- custom object ownership/lease/manager actor；
- 首个 vertical slice 中的通用 distributed shuffle service；
- SourceOp/SinkOp 或 source/sink flags；
- public primitive/backend plugin registry；
- durable graph/trace serialization；
- 默认 full-lineage persistence；
- external-effect exactly-once；
- driver/node failure recovery；
- checkpoint/resume；
- post-publication selective downstream replay；
- 对 unordered relation 的全局 canonical sort。

## 23. PrimitiveKernel conformance

每个 core PrimitiveKernel 必须通过：

- empty input 和合法 zero-cardinality output；
- compile-time relation/domain validation；
- multi-output atomicity；
- batch-size/replica permutation；
- actor completion permutation；
- retry/split permutation；
- diamond identity alignment；
- fail-fast/fail-closed propagation；
- stale-result rejection；
- provenance RowRef validity；
- 物理扰动下 deterministic identity；
- single-node 和 multi-replica Ray execution。

该 suite 验证共享 engine contract，但不会自动证明任意新 primitive 的数学语义。
扩展 formal language 仍需单独定义并论证。
