# Multigrain V2.5 架构设计稿

状态：**V2.5 = Logical-Grain Lineage clean-slate design。**

V2.5 保留 V2.4 的目标、五原语用户面、persistent actor、value 不经过 driver 和精简纪律，
但更换语义核心：

> 不再把“已经产生的 item”当作 lineage 的唯一事实；一次可独立执行、重排、重试和发布的
> **Logical Grain** 才是语义事实，item 是 grain 的输入端点或有序 emission。

这是一次模型收敛，不是回到 V2.2。V2.5 没有重 IR、Kernel 层次、WorkUnit 层次、checkpoint
协议或 recovery graph。核心是一个 typed lineage DAG：正常 data-lineage skeleton 为
`item→grain→item` 二部图；failed-before-output 等负事实用同一 GrainId 增加稀疏
`grain→grain` cause edge。Ray 实现只需紧凑 manifest。

---

## 0. 一句话命题

推荐论文命题：

> **Multigrain decouples lineage-addressed logical grains from physical actor
> batches, enabling dynamic fan-out/fan-in ML pipelines to be freely rebatched
> and retried while preserving exact fiber closure and confining attributable
> record failures to their causal descendants under a value-only,
> grain-separable UDF interface.**

中文：

> **Multigrain 将带 lineage 的逻辑 grain 与物理 actor batch 解耦，使动态 fan-out/fan-in
> ML 管线可以跨 parent 重排、重新分批和有限重试，同时保持精确的 fiber 闭合，并把可归因的
> record 错误限制在其因果后继内；用户仍只编写 value-only、grain-separable UDF。**

该命题包含一个因果链，而不是三个并列 feature：

```text
logical grain
→ physical batch 可变
→ 跨 parent rebatching
→ lineage 不随 packing 改变
→ fan-in 可精确重建
→ failure 可归因到同一个 grain
→ 只截断 causal descendants
```

### 0.1 术语边界

- **elastic rebatching**：运行时改变 grain 的物理分组、顺序和 actor assignment；首版不声称
  actor autoscaling。
- **record-level isolation**：把可归因 UDF 错误定位到一个 logical grain，并保留无关 grains；
  不表示任意 Python 异常都天然可归因。
- **backward attribution**：沿 lineage 回看失败 grain 的 source/parents；错误不会“污染”
  上游已成功 item。
- **forward containment**：只 suppress 依赖失败 item/grain 的后续 grain。
- **recovery**：首版仅指 worker/transport failure 的有限 retry、opaque batch 的有限隔离，
  以及健康工作的继续执行；不指 driver crash recovery、checkpoint/resume 或外部 exactly-once。
- **M:N support**：数据模型可表达多父 relation；首版单 driver join 不构成 distributed M:N
  性能 claim。

---

## 1. 目标与明确不做

### 1.1 首版必须闭合

1. 一个输入 grain 可以产生 0、1 或多个有序输出 item。
2. 不同 parents 的 child grains 可以跨 parent 重排和重新组成 actor batches。
3. 在 structural retry-deterministic UDF 且 retry budget 未耗尽的成功 run 中，任意合法
   packing、completion order 和有限 retry 不改变 logical identity、parent bindings、output
   ordinal 或 terminal outcome。
4. `Expand→Map/Filter→Reduce` 能恢复完整、有序、允许为空的 fiber。
5. bad record 可被归因到 grain；无关 grains 不因共享物理 batch 而被永久丢弃。
6. failure 可向上追溯 direct causes，并向下 suppress 必需依赖。
7. UDF 只收发业务 values，不接触 `ItemRef`、`GrainId`、attempt 或 lineage。
8. 大业务 value 不经 driver 反序列化。
9. API 保持 `Pipeline + Map/Filter/Expand/Reduce/Relate` 和 RayModule 式配置。
10. 真实 document fan-out/fan-in workload 可复现已有性能现象，并由新核心重新测量。

### 1.2 首版明确不做

- driver crash recovery、checkpoint/resume；
- external exactly-once、事务 sink、side-effect rollback；
- durable 跨 run identity；
- distributed lineage authority；
- 通用 provenance 查询语言；
- 任意 Python UDF 的自动纯度或 separability 证明；
- actor autoscaling；
- arbitrary window/state/iteration；
- distributed/spill/hot-key join；
- 未知 key 的精准 Relate failure containment；
- publication 后 ObjectRef 丢失的局部修复；
- stage-global epoch、deferred drain、复杂 isolation budget；
- 多套 WorkUnit/RecoveryUnit/ClosureUnit 类型体系。
- 同一个 Expand node 的不同 output ports 产生不同 cardinality；异构 child families 使用多个
  Expand nodes 表达；
- nested Expand 或任意 cardinality-changing path 的通用 Reduce closure；首版只支持一个 origin
  Expand 后接 entity-preserving Map/Filter chain。

---

## 2. 为什么 item-only lineage 不信息完备

V2.4 的 `ItemMeta(id, parents, ordinal)` 能完整描述“已经产生的 item 来自哪里”，但不能描述
“某次逻辑计算为什么没有产生 item”。

以下五种状态在纯 item DAG 中完全相同：都没有 output item。

```text
尚未执行
Expand 成功返回 []
Filter 成功返回 false
UDF 在输出前失败
因 required input 失败而 suppressed
```

但 runtime 必须区分它们：

- 未执行必须继续等待或调度；
- zero-output Expand 必须关闭 empty fiber；
- Filter-false 必须把对应 child obligation 标为正常 terminal；
- UDF failure 必须进入 attribution/containment；
- suppressed 必须保留 direct cause，且不得调用 UDF。

`FiberLedger.expected`、phantom item、completion tombstone 或 `Port.failures` 都是在 item DAG
之外补回同一个事实：**曾经存在一次语义调用，并且它有 terminal outcome。**

因此 V2.5 采用 typed DAG；正常数据边构成二部 skeleton：

```text
Item --role-tagged input--> LogicalGrain --ordered emission--> Item
                                  |
                                  +-- Success(0..N outputs)
                                  +-- Failed
                                  +-- Suppressed
```

`Failed/Suppressed` 可用同一 `LineageCause` 指向 ItemRef 或 GrainId，因此 failed-before-output
会形成稀疏 grain→grain cause edge。这不是额外 recovery graph；它仍是同一 typed lineage DAG。

---

## 3. 两个平面：语义与物理严格分离

### 3.1 语义平面

语义平面回答：

- 哪一次 logical grain 被执行？
- 它绑定了哪些带 role 的 input items？
- 它成功、失败还是被 suppress？
- 成功时产生了哪些 port/ordinal emissions？
- 哪个 grain 生产了某个 item？

语义平面不包含：

- actor；
- physical batch；
- ObjectRef 位置；
- selector；
- attempt 历史；
- completion order；
- scheduling cost。

### 3.2 物理平面

物理平面回答：

- 当前把哪些 grains 放进一个 Ray RPC？
- 每个 input item 在哪个 shard/row？
- 当前 generation 是多少？
- 哪个 actor 空闲？
- value ObjectRefs 在哪里？
- RPC 是否 stale、timeout 或 transport failed？

物理平面可以随 retry/rebatch 改变，不得改变语义平面。

### 3.3 三种身份不得混用

```text
ItemRef    = 一个逻辑 port 上的数据项
GrainId    = 一次逻辑 primitive invocation
DispatchId = 一次物理 Ray RPC
```

- `ItemRef` 跨 rebatch/retry 稳定；
- `GrainId` 跨 rebatch/retry 稳定，可能产生 0..N 个 `ItemRef`；
- `DispatchId` 是短命 transport identity，可包含任意多个 grains。

`DispatchId`、actor、attempt 和 shard 位置绝不进入 `ItemRef` 或 `GrainId`。

---

## 4. 最小语义数据模型

以下 dataclass 是合同草图，不要求首版逐对象存储；规模化实现可用 arena/column/CSR。

```python
@dataclass(frozen=True, slots=True)
class PortId:
    node: int
    slot: int


@dataclass(frozen=True, slots=True)
class EntityId:
    raw: bytes


@dataclass(frozen=True, slots=True)
class ItemRef:
    port: PortId
    entity: EntityId


@dataclass(frozen=True, slots=True)
class RoleItems:
    role: str
    items: tuple[ItemRef, ...]


@dataclass(frozen=True, slots=True)
class GrainId:
    raw: bytes


LineageCause = ItemRef | GrainId


@dataclass(frozen=True, slots=True)
class AttemptToken:
    arena: int
    dispatch: int
    grain: GrainId
    generation: int


@dataclass(frozen=True, slots=True)
class Emission:
    item: ItemRef
    ordinal: int
```

Outcome 只有三种 terminal 结果：

```python
@dataclass(frozen=True, slots=True)
class Success:
    # 与 compiled output-port slot 平行；每个 port 合法为空。
    emissions_by_port: tuple[tuple[Emission, ...], ...]


@dataclass(frozen=True, slots=True)
class GrainFailure:
    kind: str
    message: str
    direct_causes: tuple[LineageCause, ...]


@dataclass(frozen=True, slots=True)
class Failed:
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class Suppressed:
    direct_causes: tuple[LineageCause, ...]


GrainOutcome = Success | Failed | Suppressed
```

Grain 本身只有一个三态执行生命周期：

```python
class GrainPhase(enum.Enum):
    READY = "ready"
    IN_FLIGHT = "in_flight"
    SEALED = "sealed"


@dataclass(slots=True)
class GrainRecord:
    id: GrainId
    node: int
    inputs: tuple[RoleItems, ...]
    output_slots: tuple[ItemRef, ...]
    phase: GrainPhase = GrainPhase.READY
    generation: int = 0
    active: AttemptToken | None = None
    outcome: GrainOutcome | None = None
    infra_failures: int = 0
```

`output_slots` 只保存执行前即可确定的 0/1 logical occurrences：

- Source/Map/Filter/Reduce/Relate：每个 output port 一个 expected slot；
- Expand：cardinality 未知，执行前为空；成功后的动态 children 只在 emissions 中出现。

slot 不表示 value 存在。它使 Source failure、Filter-false、Map failed-before-output 等负结果仍有
稳定的 port/entity coordinate，并使 ProducerIndex 可从事实源重建。

`GrainRecord` 为实现紧凑而共址存放两类字段：

- `id/node/inputs/output_slots/outcome` 是语义事实；
- `phase/generation/active/infra_failures` 是当前 executor 的短命状态。

这是有意选择“同一 arena row 共址、语义边界显式”，不拆成两套按 GrainId 关联的 records。
短命字段采用固定 allowlist：

```text
phase
generation
current active AttemptToken
infra failure count
```

它们不参与 GrainId、lineage、primitive semantics、跨 run identity 或持久化；`active` 只保存当前
attempt，不保存 history。除这四项外，不得再向 GrainRecord 增加 actor/resources/selectors/
physical-layout 等执行字段。

必须保持：

```text
READY     <=> active is None and outcome is None
IN_FLIGHT <=> active is not None and outcome is None
SEALED    <=> active is None and outcome is not None
```

若实现语言适合 tagged union，可以把这三个合法组合编码为一个 `GrainState`；不得允许第四种组合。

`ItemRef` 是一个逻辑 port/entity coordinate，不等于“value 已物化”。对于 Map/Filter/Reduce
这类 identity 可预知的 0/1 输出，planner 可以在 value 不存在时推导 expected `ItemRef`；
是否实际存在必须查看 producer grain outcome。Expand child refs 只有在成功得到 cardinality 后
才可枚举，失败时不得虚构 children。

- READY/IN_FLIGHT grain 的所有 required input ItemRefs 必须已物化并能解析到 ValueIndex；
- Failed grain 保存它执行时的确切 inputs；
- Suppressed grain 保存所有已知 logical bindings；若 origin Expand 在 cardinality 未知时失败，
  Reduce grain 可以只绑定 anchor/已知 members，并由 `Suppressed.direct_causes` 指向 origin
  GrainId。Suppressed 是终止收据，不会被交给 UDF，因此不要求虚构未知 child inputs。

### 4.1 为什么每个 output port 必须允许空 emissions

- Filter false：所有 ports 的 emissions 为空；
- Expand empty：所有 ports 的 emissions 为空；
- 多输出节点的某个端口为空：该端口没有 emission，但 grain 仍成功。

node kind 和 compiled output contract 解释“为什么为空”；不需要再增加
`FilteredOutcome/EmptyOutcome` 类层次。后文为简洁使用 `Success(0)` 表示“所有声明 output
ports 都成功产生 0 项”，实际 ABI 始终是 `emissions_by_port`。

表示层允许任一 port 为空；首版 primitive validator 仍执行 §8.6 的对称 cardinality 合同。
因此表示能力不等于允许 Map/Reduce/Relate 的 ports 具有不同 cardinality，也不允许 Expand
各 ports 使用不同 N。

### 4.2 Item 不再重复保存 parents

一个 item 的 direct parents 由以下路径唯一决定：

```text
ItemRef
→ producer GrainId
→ GrainRecord.inputs
```

Expand 的 N 个 children 不再各自重复保存同一个 parent ref。一个 Expand grain 保存一次
parent binding，并产生 N 个 ordered emissions。

### 4.3 运行时索引与 value authority

语义事实只有 `GrainRecord`、outcome 与 per-port emissions。以下语义索引可重建、可释放：

```text
ProducerIndex[ItemRef] -> GrainId
ConsumerIndex[ItemRef] -> [GrainId]
PortIndex[PortId] -> [ItemRef]
ExpandOriginIndex[EntityId] -> (anchor ItemRef, ordinal, origin GrainId)
```

`ProducerIndex` 还可缓存 Map/Filter/Reduce 的 expected 0/1 output slots，因此一个 ref 即使没有
进入 `PortIndex`，也能找到 producing grain 并区分 normal absence 与 failure。它仍可由
GrainRecord、compiled primitive contract 和 outcome 重建。

`ValueIndex[ItemRef] -> ValueLoc` 是 live run 的物理 value authority，不是可重建 cache；没有
checkpoint 时一旦它或底层 accepted ObjectRef 丢失，首版 abort 当前 microbatch arena；是否由
调用方重启整个 microbatch 不属于本次 run。

`ConsumerIndex` 只用于 forward trace/suppression 加速，不是事实源。

`ExpandOriginIndex` 在 accepted Expand commit 时建立。由于首版 Reduce path 中 Map/Filter 保持
child EntityId，后续 member、filtered 或 failed receipt 都能 O(1) 找回 `(anchor, ordinal)`；
它可由 Expand outcomes/emissions 重建，不是第二事实源。

---

## 5. Identity：provenance-addressed，不是 content-addressed

Map/Reduce 会改变 value 却复用 entity identity，因此不得称为 content-address。

V2.5 使用 run-scoped、provenance-derived opaque IDs。Phase 0 冻结以下最小、stdlib-only
identity protocol，不把它扩展成通用序列化框架：

```text
digest = BLAKE2b-128
personalization = ASCII "RayOrchMGV2.5"
payload = canonical_encode((domain, *parts))
```

`run_salt` 必须是恰好 16 bytes。所有 length/count 使用 unsigned 64-bit big-endian。
canonical grammar：

```text
None       = "n"
bool       = "b" + 00|01
int        = "i" + sign:u8 + magnitude_len:u64be + minimal_unsigned_be
bytes      = "y" + len:u64be + payload
str        = "s" + utf8_len:u64be + utf8
tuple      = "t" + count:u64be + encoded elements
PortId     = "p" + encode(node) + encode(slot)
EntityId   = "e" + raw16
ItemRef    = "r" + encode(port) + encode(entity)
RoleItems  = "o" + encode(role) + encode(items_tuple)
GrainId    = "g" + raw16
```

整数 sign `0` 表示零或正数，`1` 表示负数；zero 固定为 sign `0`、magnitude length `0`、空
magnitude。bool 在 int 之前判定，因此 `True` 与 `1` 编码不同。Identity 输入在调用 hash 前由
各 derivation function 构造成 tuple；encoder 明确拒绝 list/dict/set/float，不做隐式
normalization，不提供 decoder、type registry 或版本协商。不得使用 `pickle`、Python
`hash()`、`repr()`、dict iteration order 或旧 MGCV1 production codec。

role bindings 固定为 compiled role order 下的 `tuple[RoleItems, ...]`；不按 role 名排序。
`RoleItems.items` 也是 tuple，Reduce members 按 ordinal 排列。golden vectors 同时约束
canonical bytes、source/entity/grain IDs 和 oracle/production parity；在 golden 通过前不声称
两份实现必然生成相同 bytes。

概念公式：

```text
source entity = H("source-entity", run_salt, source_port, position)
map entity    = primary entity
filter entity = target entity
expand entity = H("expand-entity", run_salt, node, parent entity, ordinal)
reduce entity = anchor entity
relate entity = H("relate-entity", run_salt, node, ordered(role, parent ItemRef))
```

`ItemRef=(PortId, EntityId)` 才是唯一 occurrence：

- 同一 entity 可出现在不同 ports；
- multi-output Map 的不同 ports 可以共享 entity；
- diamond 分支可按 entity 对齐；
- 同 port 内不得出现重复 entity。

Grain identity 概念上：

```text
synthetic Source = H("source-grain", run_salt, source_port, position)
Map              = H("map-grain", run_salt, node, ordered role bindings)
Filter           = H("filter-grain", run_salt, node, ordered role bindings)
Expand           = H("expand-grain", run_salt, node, ordered role bindings)
Reduce           = H("reduce-grain", run_salt, node, anchor ItemRef)
Relate           = H("relate-grain", run_salt, node, canonical role-parent tuple)
```

约束：

1. role 顺序由 compiled graph 冻结，不依赖 Python dict iteration；
2. ordinal 来自单 grain 语义输出顺序，不来自 flattened batch position；
3. physical batch/actor/attempt/completion order 不参与 ID；
4. 首版不保证跨 run、跨版本或跨语言一致；
5. ID 只能比较、hash 和做 key；lineage 遍历必须走 Grain inputs，不能反解 ID；
6. 实现可用固定宽度 digest；论文正确性把 `H` 视为无碰撞命名函数。

---

## 6. Value-only 不等于 rebatch-safe

UDF 看不到 lineage 只是 API 隔离；要保证重排、重试和二分隔离正确，还需明确的
**grain separability contract**。

对于一批 grains：

```text
eval_batch([g1, ..., gn])
= concat(eval_grain(g1), ..., eval_grain(gn))
```

允许 vectorized kernel、continuous batching 和数值 jitter，但必须满足：

1. 一个 grain 的结构输出不依赖同 batch peers；
2. 不依赖 physical batch size、packing order 或 actor identity；
3. Map/Filter 的 output cardinality 符合原语合同；
4. Expand 的 cardinality 与 sibling ordinal 在 retry/split 后稳定；
5. Reduce 只依赖该 grain 的 anchor 与完整 fiber；
6. Relate 只依赖该 grain 的确切 role-parent tuple；
7. Filter mask、key projection 与 attributable-error classification 在 retry/split 后结构稳定；
8. 不使用跨 record mutable state；
9. 无不可安全重试的外部 side effect。

### 6.1 数值非确定性

若模型 kernel 存在数值 jitter：

- structural correctness 仍须精确：ItemRef、parents、ordinal、cardinality、status；
- value correctness 可使用 workload-specific tolerance；
- token-Jaccard 只能是 payload 质量指标，不能替代 structural oracle。

### 6.2 `error_policy="isolate"` 是用户承诺

开启 opaque-error 二分意味着用户确认该 UDF 对 batch peers 独立。Multigrain 首版不静态证明
任意 Python UDF 的 separability。用户同时承诺 singleton failure classification 在 retry 间
稳定；否则同一 grain 可能在不同 isolation path 中被不一致地判为成功/失败。

---

## 7. 用户 API 与 compiled graph

用户面保持有限：

```text
Pipeline
Map / Filter / Expand / Reduce / Relate
Port / keyed
Executor / RunResult
BadRecordError / ExecutionError / CompileError
```

配置风格参考 RayModule：

```python
class DocumentPipeline(mg.Pipeline):
    def __init__(self, model: str) -> None:
        self.read_pages = (
            mg.Expand(ReadPages)
            .pre_init(dpi=144)
            .ray_options(replicas=4, batch_size=8, num_cpus=2)
        )
        self.ocr = (
            mg.Map(Ocr)
            .pre_init(model=model)
            .ray_options(
                replicas=8,
                batch_size=16,
                num_gpus=1,
                error_policy="isolate",
                max_retries=1,
                cost_fn=estimate_page_cost,
            )
        )
        self.assemble = (
            mg.Reduce(Assemble)
            .ray_options(replicas=4, batch_size=8)
        )

    def forward(self, documents):
        pages = self.read_pages(documents)
        texts = self.ocr(pages)
        return self.assemble(anchor=documents, members=texts)
```

Reduce 的 `anchor` 必须显式存在，不能靠“直接上游是否为 Expand”猜测。

### 7.1 最小 compiled records

```python
class Primitive(enum.Enum):
    SOURCE = "source"  # compiled internal kind；不是用户原语
    MAP = "map"
    FILTER = "filter"
    EXPAND = "expand"
    REDUCE = "reduce"
    RELATE = "relate"


@dataclass(frozen=True, slots=True)
class InputBinding:
    role: str
    port: PortId


@dataclass(frozen=True, slots=True)
class NodeSpec:
    id: int
    kind: Primitive
    inputs: tuple[InputBinding, ...]
    output_ports: tuple[PortId, ...]
    udf_recipe: "UdfRecipe | None"
    execution: "ExecutionOptions | None"
    reduce_anchor: PortId | None = None
    reduce_members: PortId | None = None
    relate_keys: tuple["KeyProjection", ...] = ()
```

不建立 OperationSpec/Kernel class hierarchy。primitive 差异由五个 planner/validator 函数体现：

```text
plan_map
plan_filter
plan_expand
plan_reduce
plan_relate
```

### 7.2 compile-time 必查

- node/port/role 唯一；
- port slot 与 output arity 一致；
- graph 拓扑有序；
- internal Source node 无 inputs、恰好一个 output port，且无 UDF/execution recipe；
- 非 Source node 必须有 UDF/execution recipe；
- UDF constructor recipe 可序列化并已 snapshot；
- Map/Filter/Expand 的 primary/target/parent role 明确；
- Reduce 明确 anchor 和 members；
- members 到 anchor 满足首版 Reduce path grammar；
- Relate key role 与 input role 一致；
- diamond 与 Relate 不混用；
- resource/runtime_env 使用有限白名单；
- 不支持的 failure attribution 组合在 compile 或 dispatch 前明确拒绝。

---

## 8. 五原语的 Logical Grain 语义

### 8.0 Source admission

每个 admitted source row 由一个 synthetic Source grain 产生：

```text
inputs: ()
success: source port 上恰好一个 emission
entity: H("source-entity", run_salt, source_port, source_position)
grain:  H("source-grain", run_salt, source_port, source_position)
node:   source_port.node
```

因此“每个 emitted item 恰有一个 producer”没有 root-item 例外。Source grain identity 在读取
value 前即可由 input group 与 position 确定；source decode/admission failure 记录为该 grain 的
`Failed`，其 `output_slots` 仍保存预期 source `ItemRef`，但不产生 emission/value。输入枚举
完成且所有 source grains terminal 后，source port 才 sealed。

Source 是 runtime 内部 synthetic grain，不是第六个用户原语。compiled `Primitive` 包含
internal `SOURCE` kind，使任意 `source_port.node` 都能统一解析到一个 Source `NodeSpec`；
公开用户类仍只有 Map/Filter/Expand/Reduce/Relate。每个 Source node 恰好一个 output port，
Source `GrainRecord.node` 使用该 NodeSpec id，不增加全局 `-1` magic constant，也不把 node
再次混入已冻结的 source hash 公式。

`source_position` 是一次 Executor run 内、每个 source port 独立的单调 logical ordinal：

- 每次 run 创建新的 16-byte `run_salt`，各 source-port counter 从 0 开始；
- 同一 run 的后续 microbatch arena 继续 counter，不能按 arena 重置；
- ordinal 在 record 分配给 arena 前冻结；
- rebatching、retry、actor assignment 与 completion order 不改变它；
- 新 run 因 run_salt 不同，可以重新从 position 0 开始。

### 8.1 Map

一个 Map grain 对应一个按 role 对齐的 input tuple。

```text
inputs: primary item + optional aligned role items
success: 每个 output port 恰好一个 emission
entity: 复用 primary entity
```

多输入 Map 的普通 diamond 语义：

- primary port 驱动；
- secondary ports 按 entity 精确 1:1 对齐；
- 不按 physical row position zip；
- secondary 重复 entity 是结构错误；
- 正常不存在与 failed/suppressed 必须区分；
- M:N Cartesian product 只由 Relate 表达。

### 8.2 Filter

一个 Filter grain 对应一个 target item 和可选 control items。

```text
predicate true  -> Success(每个声明 port 恰一个同 entity emission)
predicate false -> Success(0)
predicate error -> Failed
```

false 是正常 terminal，不进入 failure。

Filter 可支持 score-then-filter 的单 actor 融合 facade：UDF 返回 mask 与可选 annotation columns，
runtime 只为 survivor 发布 emissions。它仍是一个 Filter node，不额外创建 Select primitive。

### 8.3 Expand

一个 Expand grain 对应一个 parent item。

UDF batch ABI 在语义上等价于：

```python
list[Parent] -> list[list[Child]]
```

多 output ports 的用户返回固定为 port-major tuple：

```python
tuple[list[list[PortValue]], ...]  # [P][G][N_g]
```

单 output 保留直接 `list[list[Child]]`，wrapper 规范化为长度 1 的 outer tuple。Phase 3
validator 对所有 shape/cardinality contract violation 统一 abort arena；它们是 UDF ABI
错误，不伪装成 record-level `Failed`。显式坏记录仍通过 `BadRecordError(index)` 表达。

对 grain `g` 返回 N 个 children 时：

```text
ordinal = 0..N-1
entity  = expand_entity(node, parent.entity, ordinal)
parent  = g.inputs["parent"]
```

每个 Expand output port 的 emissions 长度都是该 port 的 child-count。多输出 Expand 必须满足：

```text
所有 output ports 具有相同 N
每个 port 的 ordinal 集都精确等于 [0,N)
相同 ordinal 在各 ports 共享 entity
```

compiler 从 `members` path 追溯到 origin Expand 及其对应 output port。`expected=N` 锚定在
该 origin Expand 的 `Success.emissions_by_port[origin_slot]`，不能取最终 members port 的
survivor 数，也不能用所有 output ports 的 flat emission 总数。后续 Filter-false 只把对应
ordinal 记为 dropped，不改变 N。

- N>0：发布有序 children；
- N=0：grain 仍 SEALED(Success)，empty fiber 立即可闭合；
- retry/rebatch 不改变 N 与 ordinal。

多输出 Expand 的同一 ordinal 在不同 output ports 共享 entity，由 `PortId` 消歧。

### 8.4 Reduce

一个 Reduce grain 对应：

```text
一个 anchor item
+ 一个已闭合、按 ordinal 排序的 member fiber
```

UDF batch ABI：

```python
anchors: list[Anchor]
members: list[list[Member]]
```

一个 RPC 可包含多个 Reduce grains，但一个 fiber 不得拆到两个 grains 或两个 active attempts。

Reduce output 复用 anchor entity。

首版只支持以下可静态验证的 path grammar：

```text
anchor（必须是 origin Expand 的精确 input port）
→ exactly one Expand
→ zero or more entity-preserving Map/Filter nodes
→ members port
→ Reduce(anchor, members)
```

其中：

- 不接受仅 entity 相同的 aligned branch 作为 anchor；
- compiler 从 members 沿 primary/target path 唯一追溯到 Expand，并验证其 parent input
  `PortId == reduce_anchor`；
- 不允许 nested Expand、Relate 或第二个 cardinality-increasing node；
- Map 的 primary 必须沿该 child entity；
- Map 的 secondary/Filter control 若存在，必须按同一 child entity 1:1 对齐；
- primary path 上任一 Filter false 直接把对应 ordinal 标为 dropped，并把 normal absence 穿过
  后续 entity-preserving nodes 传播到 members port；不为缺 value 的后续 nodes 创建可执行 grain；
- secondary/control failure 使对应 obligation failed；
- required secondary/control port sealed 后仍正常缺少同 entity item，属于 1:1 path contract
  violation，立即 abort arena；不得解释成 pending，也不隐式当作 Filter drop；
- 无法证明唯一 Expand origin 或 ordinal continuity 时 compile 失败。

这条限制是首版 scope，不通过隐式 ancestor guessing 支持任意 DAG。

Reduce 只有 `fail_closed`，但 member failure 不等于立刻 semantic seal：

- 所有 child obligations terminal 且无 required failure：创建 Reduce grain；
- filtered child 是正常 dropped obligation，不进入 members；
- expected=0 或全部 filtered：创建合法 empty-fiber Reduce grain；
- origin Expand 在 cardinality 未知时 failed/suppressed：可立即创建
  SEALED(Suppressed) Reduce grain；
- 已知 expected=N 后任一 member failed/suppressed：fiber 已 doomed、绝不调用 Reduce UDF，
  但仍等待其余 ordinal terminal；全部 N 个 obligations settled 后，按 ordinal canonicalize
  inputs 与 causes，再创建 SEALED(Suppressed) grain；
- 尚有 child obligation 未 terminal：保持 open，不允许 partial Reduce。

已知 N 的 Suppressed Reduce inputs 按 ordinal 包含 present 和 failed/suppressed 的最终
members-port expected coordinates，dropped 不进入 members；direct causes 只取最终 members
receipt GrainId 并按 ordinal 排列。origin Expand 在 N 未知时失败则不虚构 children，Reduce
固定为 anchor-only binding，direct cause 指向 origin Expand GrainId。

### 8.5 Relate

Relate 首版是 value-key multi-parent relation。

```python
pairs = self.join(
    left=mg.keyed(left_rows, by=left_keys),
    right=mg.keyed(right_rows, by=right_keys),
)
```

规则：

- 等所有 required key ports sealed 后再确认 unmatched；
- 同 key 下每个 canonical role-parent tuple 对应一个 Relate grain；
- M:N 时枚举完整 Cartesian product；
- grain inputs 按 compiled role order 保存确切 parents；
- output identity 从完整 role-parent tuple 派生，不从 join 枚举位置派生；
- heterogeneous Python keys 不要求可排序；确定性展示顺序按 parent logical IDs；
- output cardinality guard、key bytes limit 和 chunked enumeration 必须存在。

key projection 被编译成普通的内部 Map node/port，使用相同 GrainOutcome、generation fencing 和
port sealing。driver 只读取该 node 明确声明的小 key values，不读取原始 payload。

Relate 使用 strict、type-sensitive canonical key；Python hash/equality 不是语义权威。支持类型、
规范编码、NaN/Unicode 等边界在 Phase 5 合同与 golden tests 中冻结，不提前扩展主架构。

若 key projection 在得到 key 前失败，runtime 不知道该 parent 会影响哪些 key tuples。首版：

- `error_policy="raise"`：abort 当前 microbatch arena；

不得虚构 key，也不得把该失败解释成正常 unmatched。

Phase 1 保留 `Relate/keyed/Primitive.RELATE/NodeSpec.relate_keys/plan_relate`、relation identity、
role-general DTO 与 `JoinIndex` 位置。Phase 3 integration prototype 只实现 bounded、sealed-port、
显式小型 int-key Port 的 driver-side Cartesian relation；完整 typed-key codec、key projection
failure boundary 与 chunked enumeration 留到 Phase 5。Phase 0 reference 覆盖 int-key
1:1、M:N、unmatched/sealing。

### 8.6 首版 output cardinality 合同

- Map：每个声明 output port 恰好 1 项；
- Filter true：每个声明 output port 恰好 1 项；
- Filter false：每个声明 output port 恰好 0 项；
- Expand：每个声明 output port 恰好 N 项，且所有 ports 共享 N/ordinal/entity layout；
- Reduce：每个声明 output port 恰好 1 项；
- Relate：每个 tuple grain 的每个声明 output port 恰好 1 项。

需要其他 0:N 行为时使用 Expand，不把 Map/Reduce/Relate 隐式变成第二种 fan-out 原语。

---

## 9. Fiber closure：derived barrier，不是第二份事实源

Expand grain 的 `Success.emissions_by_port` 和后续 grains 的 terminal outcomes 是权威事实。
`FiberBarrier` 只是 Reduce planner 的派生缓存。

```python
@dataclass(frozen=True, slots=True)
class FiberId:
    reduce_node: int
    anchor: ItemRef


@dataclass(slots=True)
class FiberBarrier:
    id: FiberId
    origin: GrainId
    expected: int | None
    present: dict[int, ItemRef]
    dropped: set[int]
    # ordinal -> (最终 members-port expected ItemRef, terminal receipt GrainId)
    failed: dict[int, tuple[ItemRef, GrainId]]
    blocked_by: set[GrainId]
```

状态：

```text
origin blocked_by 非空
→ suppressed

expected is None
→ open

len(present) + len(dropped) + len(failed) < expected
→ open

settled count > expected 或 ordinal 重复
→ cardinality violation

settled count == expected 且 failed 非空
→ suppressed

settled count == expected 且 failed 为空
→ ready
```

关键点：

- origin Expand grain Failed/Suppressed 时，把其 GrainId 放入 `blocked_by`；即使 cardinality 未知，
  该 anchor fiber 也会立即 suppressed，不会永久 open 或误判 empty；
- member-path grain Failed/Suppressed 时，通过其 input lineage 找到 `(anchor, ordinal)`，把失败
  的 expected coordinate 与 GrainId 记入 `failed[ordinal]`；
- `failed[ordinal]` 只记录 compiled members port 上的唯一 terminal receipt。若 failure 发生在
  更早节点，planner 沿剩余 path 确定性创建不执行 UDF 的 suppressed receipts，最终 member-port
  GrainId 通过 direct-cause chain 指回原始失败；同一 ordinal 第二个不同 terminal receipt 是
  invariant violation；
- 首个 member failure 只把 barrier 标成 operationally doomed，不提前生成 semantic Suppressed；
  其余 obligations 继续 settle，最终 inputs/causes 按 ordinal 规范化，因此不依赖 completion order；
- `expected=0` 立即 ready；
- 全部 filtered 也是 ready，Reduce 收到空 members；
- `present` 内按 ordinal 排序；
- barrier key 包含 Reduce node，避免同 anchor 的多个 fan-out/fan-in 分支碰撞；
- barrier 可在 Reduce grain sealed 后释放；
- barrier 不进入用户 API、identity 或 durable state。

members 经过 Map/Filter 等节点时，planner 使用 `ExpandOriginIndex[member.entity]` O(1) 取得
`(anchor, ordinal, origin GrainId)`。若索引缺失，再由 ProducerIndex/compiled path 重建并视为
cache repair；正常热路径不得逐 member 回溯 lineage。不得把完整 transitive ancestry 复制进
每个 item。

`LineageCause = ItemRef | GrainId` 是 typed DAG 节点的最小引用，不建立额外 cause 类型层次：

- 已物化 input/value 问题可指向 `ItemRef`；
- failed-before-output、origin Expand failure 和 suppressed invocation 必须指向 `GrainId`。

---

## 10. Failure、归因、containment 与恢复

### 10.1 三类错误

```text
Infrastructure failure
Explicit BadRecordError
Generic UDF error
```

#### Infrastructure failure

包括 actor crash、timeout、Ray transport failure：

- 当前 physical dispatch 的结果不提交；
- 只有当前 active token 对应的 grains 才回 READY，并清空 active；
- generation 不在 callback 中增加；下一次 dispatch reservation 时才增加；
- 替换 actor；
- 在有限预算内重新 packing/retry；
- 未耗尽前不产生 semantic `Failed`；
- 耗尽后 abort 当前 microbatch arena，不伪装成 filtered/missing。

#### Explicit BadRecordError

```python
raise mg.BadRecordError("corrupt page", index=bad_index)
```

- index 指向 dispatch 中的 grain；
- 失败 grain SEALED(Failed)；
- 同 RPC 的未确认结果不直接发布；
- 健康 peer grains 作为一个 batch 重新执行；
- direct causes 是该 grain 的 input ItemRefs；后续 suppression 可直接引用该 failed GrainId。

#### Generic UDF error

- `error_policy="raise"`：终止当前 microbatch；
- `error_policy="isolate"`：对 dispatch grains 做有界二分；
- singleton 仍失败时，该 grain SEALED(Failed)；
- 二分只改变 Dispatch，不创建新 GrainId；
- 子树共享原始 retry/isolation budget，不能无限重置。

### 10.2 backward attribution

从失败 grain：

```text
GrainRecord.inputs
→ each ItemRef
→ ProducerIndex
→ producer GrainRecord.inputs
→ source grains
```

`GrainFailure` 只存 direct causes；完整 source trace 按需惰性遍历，不在每次 failure 中复制。

### 10.3 forward containment

从失败 input/item：

```text
failed GrainId / expected ItemRef
→ CompiledGraph successor planners
→ ConsumerIndex for already-created grains
→ primitive planner
→ SEALED(Suppressed) or microbatch-arena abort
```

不能只遍历已经成功产生的 downstream item，因为 failure 发生时 downstream grain 可能尚未创建。
planner 必须读取 upstream terminal outcomes，并为已知语义调用创建 suppressed grain。

primitive 规则：

- Map：任一 required binding failed → grain suppressed；
- Filter：target/control failed → grain suppressed；
- Expand：parent failed → Expand grain suppressed，不虚构 children；
- Reduce：required child failed → anchor fiber suppressed；
- Relate：已知 parent tuple 含 failed input → tuple grain suppressed；key 未知时按 §8.5 abort。

### 10.4 “恢复”的准确范围

首版真正恢复：

- retry 同一 logical grain；
- actor replacement；
- stale result rejection；
- bad grain 隔离后健康 grains 重跑并继续；
- 无关 fibers 正常完成。

首版不恢复：

- deterministic poison record 本身；
- driver 丢失的 in-memory lineage；
- 已发布后丢失的 object；
- 外部 side effect；
- durable 跨 run workflow。

论文应使用 **lineage-directed fault containment and localized retry**，不使用 disaster recovery。

### 10.5 所有 completion 共享 generation fencing

fencing 不只验证成功 manifest。以下事件都必须携带或关联原始 immutable `DispatchPlan`：

```text
successful BatchManifest
explicit BadRecord report
generic UDF exception
actor/task failure
timeout callback
cancel/replacement callback
```

处理任何事件前，driver 对该 dispatch 的每个目标 grain 验证：

```text
(arena, dispatch, grain, generation) == GrainRecord.active
```

- 完全 stale：纯 no-op；
- dispatch fate 是原子的：timeout、retry、explicit/generic error 和 cancellation 都同时撤销该
  dispatch 仍 active 的全部 entries；正常实现不允许出现 mixed current/stale；
- 若 callback 检测到同一 dispatch 部分 token current、部分 stale，视为 executor invariant
  violation，abort arena，不能只提交 current 子集；
- explicit error 最好由 worker wrapper 作为带 token 的 error ack 返回；
- Ray task exception 则使用 driver 保存的 immutable DispatchPlan 作为 token evidence；
- timeout 只撤销仍指向该 dispatch 的 active tokens。

generation 只在 `READY→IN_FLIGHT` reservation 时递增并写入 active token。任何 callback 都
不得“顺手”递增 generation。

### 10.6 run-control failure 统一为 arena abort

首版没有 node-local partial abort。所有无法表达为 GrainOutcome 的 run-control failure 统一
abort 当前 microbatch arena：

```text
normal: RUNNING -> DELIVERED -> RECLAIMED
error:  RUNNING -> ABORTED   -> RECLAIMED
```

包括：

- infra retry budget 耗尽；
- `error_policy="raise"` generic UDF error；
- Relate/cost key projection failure；
- CommitDelta/apply invariant violation；
- mixed current/stale dispatch；
- accepted ObjectRef 丢失。

abort 原子地：

1. 标记 arena `ABORTED` 并使 arena token namespace 失效；
2. 清空 ready/in-flight ownership，best-effort cancel RPC；
3. 禁止任何 port 再 seal 或任何 output/failure partial delivery；
4. 所有晚到 callback 纯 no-op；
5. 释放未交付 ObjectRefs/indexes；
6. 向调用者抛出一个带已确认诊断快照的 `ExecutionError`。

`ABORTED` 是 microbatch run-control 状态，不扩充 Grain 的三态 lifecycle。

---

## 11. Scheduler 与物理 Dispatch

### 11.1 pending queue

node-local pending queue 只保存 `GrainId` 和可选小型 scheduling metadata，不复制 parents、
ordinal 或 value。

默认 packing：

1. 无 cost 时按 grain count；
2. 有 cost 时使用简单 LPT/greedy；
3. 尊重 `batch_size` 和 node resource limits；
4. 一个 grain 同一时刻最多一个 active generation；
5. parent/fiber 边界不限制 Map/Filter/Expand 的 batch；
6. 一个 Reduce grain 不拆分；
7. Relate tuple grains 可跨 key 共同 batch。

`cost_fn` 若依赖业务 value，必须通过明确的小型 cost projection worker 计算；driver 不读取大
payload。该 projection 也必须编译成普通内部 Map grain；projection failure 首版终止当前
microbatch arena，不建立单独执行协议。

#### 11.1.1 Batch trigger

每个 `(arena, node)` 只有一个 derived ready queue 和一个 tail wait timestamp，不创建 per-grain
timer。trigger 固定优先级：

```text
isolation forced group
→ ready_count >= batch_size
→ node admission closed
→ arena drain
→ max_batch_wait_ms expired
→ otherwise wait
```

`node admission closed` 表示所有 required input ports sealed，且所有 driving occurrences 已被
planner 分类。默认 `max_batch_wait_ms=2.0`；full batch、sealed tail 和 drain 不等待，只有
upstream 仍 open 的 underfilled tail 等待。full batch pop 后若留下不足 batch_size 的 tail，其
wait timestamp 从该次 pop 重新开始；新 grains 加入已有 tail 不延长 deadline。

只有 actor/pending capacity 已获得时才 reservation、增加 generation 并安装 AttemptToken；没有
capacity 时 grain 保持 READY，不预排 RPC。driver turn 固定先处理 completion/commit/planner/
sealing，再评估 triggers。timeout 通过 single event-loop 的 `ray.wait(timeout=next_deadline)`
驱动，不使用 node/actor timer thread。

### 11.2 transport DTO

`AttemptToken` 已在 §4 定义，并包含 `(arena, dispatch, grain, generation)`。其余 DTO：

```python
@dataclass(frozen=True, slots=True)
class RowTake:
    ref_slot: int
    row: int


@dataclass(frozen=True, slots=True)
class DispatchEntry:
    token: AttemptToken
    role_takes: tuple[tuple[RowTake, ...], ...]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    id: int
    node: int
    entries: tuple[DispatchEntry, ...]
```

driver 先分配 `DispatchPlan.id`，再 reservation 每个 grain、递增 generation，并构造
`token.dispatch == plan.id` 的 entries。reservation 与 active-token 安装在同一个不可中断段完成。

`RowTake` 的 `ref_slot` 指向 DispatchPlan 附带的 input ObjectRef 数组，避免多 role/multi-port
选择器歧义。

### 11.3 worker manifest

```python
@dataclass(frozen=True, slots=True)
class Span:
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class GrainAck:
    token: AttemptToken
    spans_by_port: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class BatchManifest:
    dispatch: int
    acks: tuple[GrainAck, ...]
    column_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DispatchErrorReport:
    dispatch: int
    kind: str                 # "bad_record" | "generic_udf"
    bad_token: AttemptToken | None
    message: str


WorkerManifest = BatchManifest | DispatchErrorReport
```

`start == stop` 显式表示该 grain 在该 port 成功零输出，因此不依赖 output row 才能确认完成。

- `BatchManifest` 只表示整次 vectorized call 正常返回，并要求每个 entry 恰好一个 ack；
- `BadRecordError(index)` 被 wrapper 转成带对应 `bad_token` 的 `DispatchErrorReport`；
- generic UDF exception 转成 `bad_token=None` 的 report；
- error report 不确认任何 peer output，所有 output columns 返回合法空值并整体不提交；
- actor crash/transport failure 没有 manifest，由 driver 保存的 DispatchPlan 处理。

### 11.4 RPC granularity：Grain 不是 RPC

Logical Grain 是 correctness/retry unit，**不等于一条 Ray RPC**。正常路径必须：

```text
many logical grains
→ one DispatchPlan
→ one actor method call
→ one manifest ObjectRef + one value ObjectRef per output port
```

最小合同：

1. scheduler 按 `batch_size` 或 cost budget 聚合 READY grains；
2. 未满 batch 可短暂等待 `max_batch_wait_ms`；port seal/arena drain 时允许 tail RPC；
3. Reduce 的一个 fiber 是一个 grain，但多个 ready fibers 仍可进入同一 RPC；
4. error isolation 只有二分叶子允许临时 singleton，健康 peers 重新合批；
5. 每个 `(Dispatch, output port)` 返回一个 column/block ObjectRef，GrainAck span 再切回 grains；
6. `max_pending_dispatches` 与 `max_pending_per_actor` 有界；
7. persistent actor 默认 method concurrency=1；vLLM 等后端使用其内部 continuous batching；
8. driver 用 bounded `ray.wait` 读取小 manifests，不一次 `ray.get` 全部业务 values。

禁止 per-grain `.remote()`、per-emission ObjectRef 和向单 actor 预排数千 calls。

首版只记录三个低成本指标：

```text
grains_per_rpc
pending_dispatches
tail_or_isolation_rpc_fraction
```

若 benchmark 发现 RPC 过碎，再增加 bytes/shards/flush-reason 等诊断，不提前建设复杂 credit 或
compatibility accounting。

---

## 12. Ray worker 与数据面

actor 是持久 UDF 实例：

```python
class Worker:
    def __init__(self, udf_cls, init_args, init_kwargs):
        self.udf = udf_cls(*init_args, **init_kwargs)

    def run(self, plan: DispatchPlan, *input_shards):
        # runtime wrapper:
        # 1. 根据 role_takes 构造 value columns
        # 2. UDF 只接收 values
        # 3. 校验 primitive cardinality contract
        # 4. 返回 manifest value + output column values
        ...
```

Ray 调用使用 direct multi-return：

```python
manifest_ref, *column_refs = worker.run.options(
    num_returns=1 + output_arity
).remote(plan, *input_refs)
```

driver 只：

```python
ready_manifest_refs, pending_manifest_refs = ray.wait(
    pending_manifest_refs,
    num_returns=min(completion_batch, len(pending_manifest_refs)),
)
manifests = ray.get(ready_manifest_refs)  # 只读取有界批次的小 manifest
```

业务 output columns 保持为 ObjectRefs，直接成为下游 actor 输入。禁止 actor 内 `ray.put()` 后把
nested ObjectRefs 包进 `WorkerResult` 返回。

### 12.1 lineage 如何“跨 actor”

- actor runtime wrapper 看见 `AttemptToken` 和 selectors；
- 用户 UDF 看不见任何 lineage；
- actor 返回 per-grain spans；
- driver/control plane 根据 accepted manifest 生成 semantic emissions；
- 下游 dispatch 再携带对应 grain tokens/selectors。

因此准确说法是：

> lineage is preserved across actor/operator boundaries

而不是每个 actor 保存完整 lineage graph。

### 12.2 Ray RSS、plasma mmap 与 allocator fragmentation 风险

记录一个已知工程风险：

- Ray issue [#53261](https://github.com/ray-project/ray/issues/53261) 在 Ray 2.44.1/PyTorch 2.6
  复现 actor 返回大量 tensors 后 RSS 不下降；截至该 issue 最新公开状态仍是 open、P1；
- Ray 社区关于
  [RAM not being released](https://discuss.ray.io/t/memory-ram-not-being-released-by-ray/7213)
  提醒 object-store mappings 可能在 cluster 生命周期中保留以供复用；
- 高频 serialization、高 actor concurrency 和 allocator arenas 也可能放大 worker RSS。

用户给出的最小复现每次返回 262144 个 float32，即约 1 MiB；3000 个 refs 在 clear 前本身就有约
2.93 GiB live data，并且一次 `ray.get(results)` 会把它们同时 materialize 到 caller。该峰值不是
“几十 MiB”。clear 后 RSS 不回落仍是需要在目标 Ray 版本上复测的风险。

首版只做四件事：

1. 多 grains 合成 RPC，每个 output port 返回一个 block ObjectRef；
2. pending Dispatch 有界，不一次 `ray.get` 全部 values；
3. actor method concurrency 默认 1；
4. arena 完成后及时释放中间 refs 与 metadata。

不在原型中预先实现 allocator profiler、credit ledger、自动 actor rotation 或复杂 RSS controller。
若长稳 benchmark 明确复现增长，再按证据选择 jemalloc、arena limit、graceful actor recycle 或
更细诊断。V2.5 不承诺 Ray actor RSS 精确回到 live payload 大小。

---

## 13. Generation fencing 与原子 publication

### 13.1 三态上限

```text
READY --dispatch(gen+1)--> IN_FLIGHT
  ^                            |
  |---- retryable infra -------|
                               |
                               +-- valid commit --> SEALED

stale result --> no-op
upstream failure --> SEALED(Suppressed)
```

不增加 `RETRYING/COMMITTING/CANCELLED/DRAINING`：

- retry 用 READY + generation；
- semantic failure 是 SEALED(Failed)；
- whole-arena cancellation 直接销毁 arena；
- commit 在单 driver event-loop 的不可中断段完成。

### 13.2 manifest 验证

提交前在同一个 single-thread event-loop turn 内纯计算并验证 `CommitDelta`，中间无 `await`、
callback 或 arena mutation。验证归并为四类：

1. Dispatch/AttemptToken 仍是当前 active generation，acks 完整；
2. spans、column lengths、output arity 与 primitive cardinality 合同一致；
3. 全部 ItemRefs/ordinals/parent bindings 可确定推导，且没有重复或错配；
4. ValueIndex、lineage indexes 与 fiber/join updates 已完整准备，可全量 apply。

详细断言属于 Phase 2 实现和测试清单，不扩展成独立 commit protocol。

### 13.3 atomic commit 顺序

只有完整 `CommitDelta` 验证成功后，才进入单 driver event-loop 中无 `await`、无用户代码、无
二次验证的 apply 段：

```text
apply shard refs + ValueIndex
→ apply ProducerIndex / PortIndex
→ update ConsumerIndex and derived barriers
→ write GrainOutcome
→ clear active and mark grain SEALED
→ enqueue newly-ready downstream grains
```

一个 accepted grain 的多 ports 全有或全无。验证失败不修改 arena；apply 段若出现内部异常，
abort 整个 arena，不暴露部分结果。

stale token：

- 不改变 GrainRecord；
- 不安装 locator；
- 不更新 fiber/join state；
- 对应 ObjectRef rows 变成无人引用垃圾，由 Ray 生命周期回收。

这是 **live-driver 内部 at-most-once visibility**，不是物理 UDF exactly-once。物理 UDF 可能因
retry 被执行多次。

Ray hidden task retry/restart 必须关闭或纳入同一个 token fencing；首版建议显式设置为 0，
由 Executor 统一管理。

---

## 14. Port sealing 与 absence 语义

一个 port 必须有 `open/sealed` 状态。

“当前已创建 grains 都 terminal”不足以 seal，因为动态上游可能尚未产生后续候选 grain。
首版按 primitive 使用以下可执行判据：

- Source：input enumeration 已结束，全部 source grains terminal；
- Map/Filter：primary 及所有 required alignment/control ports 已 sealed，每个 primary entity
  已被分类为 ready grain、normal absence 或 suppressed，且全部已创建 grains terminal；
- Expand：parent driving port 已 sealed，其 logical parent domain 中每个候选都已分类：
  materialized parent 对应 terminal Expand grain，failed/suppressed expected parent 对应 terminal
  suppressed receipt，normal absence 明确不创建 grain；
- Reduce：anchor port 已 sealed，每个 anchor 对应的 fiber 已 ready+terminal 或 suppressed；
- Relate：全部 key-projection ports 已 sealed，所有 key tuples 已完整枚举/分类，全部 tuple grains
  terminal；
- internal projection：其 driving port sealed 且所有 projection grains terminal。

只有 planner 能证明 producer domain 已闭合，port 才能 sealed。

Phase 1 planner 返回四种 occurrence action：

```text
wait | ensure executable | ensure suppressed | normal absence
```

contract/invariant violation 直接抛 planner error，Phase 2 转成 arena abort，不增加第五种 outcome。
`ensure` 是按 GrainId 的幂等 set-if-absent：相同 semantic template 在 READY/IN_FLIGHT/SEALED
任一 phase 都不重复插入；semantic fields 或 executable/suppressed 分类冲突则 abort。

multi-role 固定优先级：

1. driving occurrence normal absence 时直接传播 absence，不再等待其他 roles；
2. driving logical occurrence 存在时等待全部 required roles settled；
3. 任一非-driving required role sealed 后正常缺失是 contract violation；
4. 否则按 compiled role order 收集全部 Failed/Suppressed receipt GrainId 并 suppress；
5. 全部 present 才 ensure executable。

Filter false 是 normal terminal；后续缺 driving value 的 Map/Filter 不执行，absence 沿 compiled
entity-preserving path 传播到 members port。任何增量 cursor 都只能是可重建 derived cache。

absence 的解释依赖 seal：

```text
port open + item 不存在
→ pending/unknown

port sealed + Filter grain Success(0)
→ normal filtered

port sealed + Relate 没有 tuple grain
→ normal unmatched

producer grain Failed/Suppressed
→ failure-derived absence
```

没有 seal，join/reduce 不得把“当前没看到”解释为永久不存在。

Port 的语义是 keyed bag，不承诺按 actor completion order 稳定。结构正确性按 ItemRef 映射比较；
需要展示顺序时使用独立 canonical presentation sort。Reduce fiber 内部顺序只由 ordinal 决定，
不使用 PortIndex list order。

---

## 15. Executor 主循环

```text
compile graph
→ create per-microbatch arena
→ admit source grains and source emissions
→ repeat until graph quiescent:
     inspect upstream grain outcomes / port seals
     primitive planners create ready or suppressed grains
     scheduler packs READY grains into DispatchPlans
     dispatch to persistent actors
     accept valid manifests / reject stale manifests
     atomically publish emissions and outcomes
     update derived fiber/join indexes
     seal ports whose producer sets are terminal
→ deliver output refs + failures + metrics
→ snapshot compact diagnostics
→ detach RunResult from arena
→ reclaim arena control metadata
```

不同 bounded microbatches 可通过 `max_inflight` 重叠，但首版不跨 arena 合 batch。每个 arena
有独立 namespace、grain table、indexes 和 attempt tokens。

### 15.1 Arena reclamation 与持续运行内存上界

持续服务只保留 **live-set metadata**，不是 run history。live set 包括：

- READY/IN_FLIGHT grains；
- 已 terminal 但消费者或 port seal 尚未完成的 grains；
- 尚未闭合的 FiberBarrier/JoinIndex；
- 当前 retry/failure attribution 仍需的 direct lineage；
- 交付前的 final ItemRefs、failures 与 metrics。

正常 delivery 时：

1. 把 final output ObjectRefs、compact failure/source attribution 和 metrics 复制进独立
   `RunResult`；
2. `RunResult` 不得引用 GrainTable、indexes、arena callbacks 或 intermediate ObjectRefs；
3. arena 进入 DELIVERED 后立即 RECLAIMED，不等待用户释放 final ObjectRefs；
4. 用户长期持有 final ObjectRefs 只延长业务数据寿命，不得延长 lineage/control metadata；
5. intermediate ObjectRefs 在对应 ports 不再有后续 consumer/retry 后删除 executor ownership；
6. ABORTED arena 同样失效所有 callbacks 后进入 RECLAIMED。

首版不建设通用 ObjectRef lease/hold 系统：Executor 按 compiled consumers 与 active
DispatchPlans 管理其持有的 refs；arena token 失效后 late callbacks 只丢弃结果。若 benchmark
发现 ObjectRef ownership 难以正确回收，再引入专门 ledger，而不是现在预先增加一套生命周期 IR。

稳态 metadata 上界：

```text
O(max_inflight_arenas
  × max_live_metadata_per_arena
  + compiled graph
  + actor-pool control state)
```

不是 `O(service lifetime processed records)`。该上界只有在 admission/backpressure 与 reclamation
合同真实执行时成立，不能依赖 Python GC 最终“自己清理”。

---

## 16. 唯一事实源与可删缓存

### 16.1 唯一事实源

```text
CompiledGraph
ArenaState
GrainRecord.inputs
GrainRecord.output_slots
GrainRecord.outcome
Success.emissions_by_port
Port seal
```

live data plane 另有不可重建的物理 authority：

```text
Shard registry
ValueIndex
```

它们不是 semantic lineage，但没有 checkpoint 时不能归类为 cache。

### 16.2 可重建缓存

```text
ProducerIndex
ConsumerIndex
PortIndex
ExpandOriginIndex
FiberBarrier
JoinIndex
ready queues
cost estimates
display/source trace
```

缓存不得携带语义上独有的信息。删除缓存后必须能从事实源重建。

`GrainRecord` 状态必须满足 §4 的 tagged-state invariant；`phase=SEALED` 与
`outcome is not None` 是同一个状态的联合编码，不是两个可独立更新的事实。

---

## 17. Metadata scale

语义复杂度目标：

```text
O(live grains
  + role-tagged input edges
  + actual emissions
  + value locators
  + decoded join-key bytes
  + actual M:N relation cardinality)
```

不得出现：

- `O(attempt history)`；
- 每个 item 复制完整 lineage depth；
- 每次 failure 复制 source trace；
- Expand 每个 child 重复 parent tuple；
- 一次性物化整个 hot-key Cartesian product；
- nested tuple ItemId 随 DAG depth 增长。

原型可先用 dataclass/dict；scale milestone 前转为：

```text
GrainTable: node / phase / generation / input CSR / outcome offsets
EmissionTable: producer grain / port / entity / ordinal
ValueTable: ObjectRef slot / row
FailureTable: sparse failed grain rows
```

该规范化模型并非声称每个 1:1 Map 都比单个 `ItemMeta` 更省字节：1:1 stage 会多一个显式
grain/outcome row，换取 zero-output/failure/retry 的信息完备；高 fan-out 时则避免每个 child
重复 parent bindings。必须实测 bytes/item、bytes/grain，并报告相对 item-only positive-path
layout 的 break-even，而不是只凭 class 数量宣称更轻。

role、node、port、GrainId 可在 arena 内使用整数 handle；公开/跨进程 token 使用固定宽度 opaque
ID。

### 17.1 Hard admission 与 backpressure

首版只保留最小的有界性配置：

```text
max_inflight_arenas
max_grains_per_arena
max_pending_dispatches
max_fanout_per_grain
max_relation_cardinality
```

达到 arena/pending 上限后暂停 source admission 并 drain。hard structural limits 统一是
run-control arena abort，不产生 Grain `Failed`：

- `max_fanout_per_grain`：单个 Expand grain 的共享 N；
- `max_relation_cardinality`：单个 Relate node、单个 arena 的累计 tuple grains；
- `max_grains_per_arena`：arena GrainTable 的全部 rows。

任何会越界的 insertion/CommitDelta 必须在 mutation/publication 前原子拒绝。首版不引入
byte-credit、RSS controller、spill 或复杂 watermark 状态机。
若 benchmark 证明仅按 grain count 不足，再增加 byte-aware admission。

必须测量：

- metadata bytes/grain 和 bytes/emission；
- driver RSS/CPU；
- grains/RPC 与 pending dispatches；
- 长稳 benchmark 的 driver/worker RSS 趋势。

只有这些指标显示异常时，才启动 plasma/anonymous/PSS、allocator 和 ObjectRef ownership 的专项
诊断。

---

## 18. 必须保持的不变量

1. physical actor/batch/attempt/completion order 不进入 ItemRef 或 GrainId。
2. 一个 GrainId 同时最多一个 active generation。
3. stale generation 永远没有 publication 权。
4. 每个 admitted grain 最终至多一个 terminal outcome，且 GrainRecord 只能处于 §4 三种合法组合。
5. item 缺失不能代表 terminal 状态；zero-output 必须由 per-port empty `Success` 表示。
6. 每个 output slot/emitted ItemRef 恰有一个 producer grain；slot 存在不代表 value 已产生。
7. 一个 grain 的所有 output ports 原子发布。
8. Map/Filter/Reduce output entity 复用语义 anchor；PortId 负责 occurrence 消歧。
9. Expand output 与 `[0,n)` ordinal 一一对应。
10. Reduce 只消费 closed fiber；一个 fiber 对应一个 grain。
11. filtered obligations 计入 terminal closure，但不进入 Reduce members。
12. empty fiber 合法且必须触发一次 Reduce grain。
13. required failure 不得静默变成 normal missing。
14. diamond 是 entity 1:1 alignment；M:N 只由 Relate 表达。
15. Relate unmatched 只能在输入 scope sealed 后确认。
16. backward lineage 一律走 item→producer grain→inputs，不反解 ID。
17. failure 只存 direct causes；完整 trace 惰性生成。
18. 大 value 不经 driver 反序列化。
19. UDF 看不到 lineage/attempt/physical batch identity。
20. driver 存活期间，一个 logical output 至多可见一次；不声称物理执行一次。
21. failed-before-output 的 causal edge 指向 GrainId，不要求虚构 output ItemRef。
22. 所有 success/error/timeout/infra callbacks 都服从同一个 active token fencing。
23. output Port 是 keyed bag；只有 Reduce fiber ordinal 具有核心顺序语义。
24. Logical Grain 不等于 Ray RPC；正常路径按 batch_size/cost budget 聚合。
25. 每个 `(Dispatch, output port)` 至多一个 value ObjectRef，不逐 emission 创建对象。
26. inflight arenas、arena grains 与 pending Dispatches 必须有界。
27. actor method concurrency 默认 1；任何提高都必须有 UDF safety 与内存实验依据。
28. RunResult 不得引用 arena control state；delivery 后 arena metadata 可独立 reclaim。

---

## 19. 可形式化的正确性命题

### 19.1 Packing Confluence

在以下前提下：

- UDF grain-separable；
- structural cardinality、ordinal、key projection 和 attributable-error classification 在 retry
  间确定；
- source inputs 与 semantic failure set 固定；
- infra retry budget 未耗尽，run 正常达到 quiescence；
- 调度公平；

对同一 compiled graph，任意合法：

- batch partition；
- batch order；
- actor assignment；
- completion order；
- retry/repacking；
- stale-result arrival；

都产生相同的：

```text
ItemRef set
producer/input lineage
ordinal
terminal outcome
port seal state
```

都得到相同结构终态。infra budget 耗尽、driver abort 和 unknown-key projection failure 是
run-control failure，不属于该 confluence 命题。确定性 UDF 下 values 亦相同；数值非确定模型
按 workload tolerance 比较 payload。

### 19.2 Cardinality and Fiber Closure

- Expand grain 每个 output port 的 N 个 emissions 与 ordinal `[0,N)` 一一对应，各 ports 共享
  N/ordinal/entity layout；
- 每个 obligation 恰好进入 present/dropped/failed 之一；
- Reduce grain 当且仅当 fiber closed 且无 required failure 时被执行；
- empty fiber 被执行一次；
- 一个 child 不会进入错误 anchor 的 Reduce grain。

### 19.3 Relation Completeness

输入 ports sealed 后，每个 Relate key 下的 output grains 与 role multisets 的 Cartesian product
一一对应：

- 无遗漏；
- 无重复 parent tuple；
- 无错误 role；
- 不按物理位置 zip。

### 19.4 Fault-cut Exactness

在 attributable grain failure 假设下：

- 一个 planned grain 被 suppress，当且仅当它的 required ancestor closure 与 failure set 相交；
- 与 failure set 无 required dependency 的 grains 保持可执行；
- 健康 grain 的 structural outputs 与无故障 reference execution 一致。

首版 Relate unknown-key failure 会显式 abort microbatch arena，不纳入 exact cut claim。

---

## 20. 测试与实验

### 20.1 Reference interpreter

建立不依赖 Ray 和 production modules 的单线程解释器，以 Logical Grain 为语义单位。

精确比较：

- GrainId/canonical grain key；
- ItemRef；
- role bindings；
- emissions；
- ordinal；
- terminal outcomes；
- fiber membership；
- suppression causes；
- M:N parent tuples。

### 20.2 Property testing

随机生成小型五原语 DAG，随机化：

```text
fan-out cardinality
Filter pattern
batch partition
actor count
completion order
retry position
stale result
bad grain
empty fiber
M:N key multiplicity
```

所有结果与 reference interpreter 精确对比。

### 20.3 RQ1：rebatching efficiency

当 fan-out 与 child service time 呈 uniform/log-normal/Pareto/Zipf 分布时，比较：

```text
parent-bound batching
flat child batching without exact fan-in
Multigrain count-based rebatching
Multigrain cost-aware rebatching
```

指标：

- throughput；
- GPU/actor utilization；
- p50/p95/p99 parent completion；
- bubble ratio；
- pipeline overlap；
- scheduling overhead。

所有测量必须包括最终 Reduce output，不能只测 Expand 后 child throughput。

### 20.4 RQ2：fault containment

比较：

```text
whole-shard retry/drop
record isolation without lineage-aware fan-in
Multigrain logical-grain isolation
```

指标：

- weighted healthy recomputation；
- UDF calls；
- healthy parent completion；
- false suppression；
- missed suppression；
- duplicate/missing outputs；
- recovery latency；
- failure amplification。

### 20.5 RQ3：lineage cost

测量：

- no-lineage baseline overhead；
- bytes/grain 与 bytes/emission；
- driver RSS/CPU；
- worker RSS；
- grains/RPC 与 pending dispatches；
- max live grains；
- 至少一个长稳 soak test 的 RSS 趋势。

若 soak test 显示 worker RSS 持续增长，再单独复现
[Ray #53261](https://github.com/ray-project/ray/issues/53261)，区分 pinned refs、plasma mapping、
allocator fragmentation 和 metadata leak，并评估 jemalloc/actor recycle。该专项诊断不是首版
runtime 组件。

### 20.6 真实 workloads

首篇至少：

1. document→pages/blocks→model inference→document assembly；
2. video/media→frames/clips→inference→video aggregation。

Relate/M:N workload 只有在确实实现并测量 join closure 与 cardinality 边界时才进入核心实验；
否则作为 API generality case，不作为 headline 性能证据。

### 20.7 现有数字的地位

已有 368 PDF / 4×H20、1.72×、bubble 和 poison-record 数字是回归基线，不是 V2.5 证据。
必须在 V2.5 核心上重新运行后才能进入论文。

---

## 21. VLDB SDS 贡献表述

建议正文只保留三项：

1. **Logical-grain lineage model**  
   用 role-tagged inputs、ordered emissions 与显式 terminal outcome 统一动态 1:N、N:1 和
   multi-parent stage，特别表达 zero-output 与 failed-before-output。

2. **Packing-confluent execution**  
   把 actor batch 定义为 logical grains 的短命 partition，通过 generation fencing、原子
   publication 和 exact fiber closure 支持跨 parent rebatching 与有限 retry。

3. **Lineage-directed causal fault cut**  
   同一 grain lineage 用于 backward attribution 和 forward containment，使可归因坏记录只截断
   required descendants，同时保留健康 grains 的批量执行。

不把以下内容单独 claim 为 novelty：

- LPT；
- actor pool；
- lineage DAG 本身；
- retry 本身；
- flat_map/fan-out 本身；
- Ray integration 本身。

研究价值来自它们在 dynamic-cardinality fan-out/fan-in ML pipeline 中形成一个可形式化、
可测量的闭环。

---

## 22. Package 布局

起步保持约 6 个文件：

```text
rayorch/experimental/multigrain_v2_5/
├── __init__.py
├── api.py          # Pipeline、五原语、配置链
├── graph.py        # PortId、NodeSpec、CompiledGraph、五个 planner 合同
├── grain.py        # IDs、GrainRecord、Outcome、indexes、fiber barrier
├── worker.py       # persistent actor、DispatchPlan、BatchManifest
├── executor.py     # scheduler、generation fencing、atomic publication、recovery
└── metrics.py      # throughput/utilization/recompute/metadata
```

测试：

```text
test/experimental/multigrain_v2_5/
├── reference_semantics/
├── test_identity.py
├── test_grain_model.py
├── test_primitives.py
├── test_fiber_closure.py
├── test_failure_cut.py
├── test_dispatch_protocol.py
├── test_property_schedules.py
└── test_ray_integration.py
```

只有模块出现独立合同和测试压力时才拆分。首版不创建独立 `lineage.py/recovery.py/fanin.py`
类层次；它们先分别是 `grain.py/executor.py` 内的局部函数和派生索引。

---

## 23. 实施阶段

### Phase 0：reference semantics

- Logical Grain 独立解释器；
- 五原语 golden cases；
- zero-output/failed-before-output/empty fiber；
- compiled Source kind 与 run-global per-source-port position；
- canonical encoding/identity golden vectors；
- schedule/retry property generator；
- production/reference 双向 import guard。

### Phase 1：semantic core

- PortId/EntityId/ItemRef/GrainId；
- GrainRecord/Outcome/Emission；
- Producer/Consumer/Port/Value indexes；
- 四个 executable core planners（Map/Filter/Expand/Reduce）与 bounded `plan_relate` contract；
- ExpandOriginIndex 与 fiber barrier；
- compile validation。

不接 Ray，先通过 synthetic Source、unary `Expand→Map/Filter→Reduce` 的 reference parity
与 planner idempotence fixtures。
aligned secondary/control 语义保留在 §8.4，但在 unary exit gate 通过后实现；不为首个
document workload 阻塞 semantic core。packing/retry/stale-result confluence 属于 Phase 2，
Phase 1 不在没有 executor 时宣称已验证。

### Phase 2：single-process executor

- READY/IN_FLIGHT/SEALED；
- dispatch packing；
- manifest validation；
- atomic publication；
- stale generation；
- explicit failure 与 suppression。

### Phase 3：Ray actor execution

- persistent pools；
- multi-return；
- ObjectRef selectors；
- actor replacement；
- bounded retry；
- generic-error isolation。
- `Pipeline.forward` symbolic tracing 到同一 `CompiledGraph`；
- public `Executor(Pipeline).run(...)` event-loop driving；
- 每个 Dispatch/output-port 一个 coarse ObjectRef；
- bounded sealed int-key Relate 只作为 integration generality case。

### Phase 4：real workloads

- bridge 迁移；
- document pipeline；
- video fan-out/fan-in；
- baseline/ablation；
- metadata scale。

### Phase 5：Relate

- canonical typed-key encoding 与 golden tests；
- key projection；
- sealed-port join；
- M:N tuple grains；
- cardinality guards；
- failure boundary；
- 是否进入首篇论文由 workload 结果决定。

---

## 24. 防膨胀红线

### 24.1 Logical Grain 不能长成重 IR

GrainRecord 只允许两组共址字段：

```text
semantic facts:
identity
node
role-tagged logical inputs
output slots
terminal outcome

ephemeral lifecycle allowlist:
3-state phase
generation
current active AttemptToken
infra failure count
```

lifecycle 字段不参与 identity/lineage/primitive semantics，只为当前 arena 的 dispatch fencing 与
有限 retry 服务。采用共址 row 是为了避免两套 GrainId keyed records 和跨表状态不变量；这不是
允许 Grain 演化为 WorkUnit。

严禁加入：

- actor/resources；
- ObjectRef/selectors；
- physical batch contents/actor assignment（当前 active token 中的单个 DispatchId 除外）；
- layout/slice/rebase；
- checkpoint；
- attempt history；
- recursive recovery scope；
- Kernel/Matcher protocol；
- 每原语 Grain 子类。

### 24.2 outcome 只有三种

```text
Success
Failed
Suppressed
```

Filter-false 与 Expand-empty 的差异由 node kind + emissions 解释，不新增 outcome class hierarchy。

### 24.3 execution lifecycle 只有三态

```text
READY
IN_FLIGHT
SEALED
```

retry 用 generation；failure 存在 outcome；不增加平行 failure state machine。

### 24.4 primitive suppression 留在 planner

每个 primitive 有一个局部纯决策：

```text
input outcomes + closure/matching view
→ wait | create READY grain | create SUPPRESSED grain | normal absence
```

不创建独立 recovery-scope graph。

### 24.5 Reduce anchor 修复不能物化全祖先

只允许：

- compiled `anchor/members`；
- item→producer grain→inputs 遍历；
- node-local derived cache。

不允许给每个 item 增加 ancestors/op-path/full provenance。

### 24.6 UDF contract 不发展成类型证明系统

只做：

- 文档合同；
- opt-in `error_policy="isolate"`；
- runtime arity/cardinality checks；
- reference/property tests。

### 24.7 Relate 不倒逼重 matcher

unknown-key failure 诚实 abort；不为挽救过宽 claim 创建 matcher/evidence
协议、distributed provenance 或潜在 tuple 图。

### 24.8 不为“细粒度 lineage”创建细粒度 RPC

细粒度只存在于 semantic Grain/manifest spans；Ray transport 必须保持 coarse block：

- 不增加 per-item actor、per-grain remote method 或 per-emission ObjectRef；
- 不通过提高 actor `max_concurrency` 掩盖 batch trigger 错误；
- 不用频繁 actor kill 代替 bounded pending/reclamation；
- allocator 调优与 actor rotation 只在 benchmark 复现问题后考虑，不得替代 RPC coalescing；
- 新机制若使 steady-state RPC 数接近 grain 数，默认否决。

---

## 25. V2.5 成功标准

只有同时满足以下条件，核心闭环才算完成：

- document-like `Expand→Map/Filter→Reduce` 端到端运行；
- children 确实跨 parents 重排；
- empty、partial-filtered 和普通 fibers 都正确闭合；
- 任意合法 packing/retry 后 structural result 与 reference interpreter 精确一致；
- stale attempt 不产生任何可见 publication；
- 单 bad child 只 suppress 对应 required fiber；
- 其他 parent outputs 正常交付；
- infrastructure retry 不产生 duplicate/mismatched ItemRefs；
- UDF 不接触 lineage；
- 大 values 不经 driver；
- 正常路径按 batch_size/cost budget 合并 grains，tail/isolation 才允许 singleton；
- 每个 Dispatch/output-port 只返回一个 value block ObjectRef；
- inflight arenas、arena grains 与 pending dispatches 有界；
- delivery 后 arena metadata 与 RunResult 解耦并可立即 reclaim；
- 长稳 soak test 中 driver metadata 不随历史单调增长；worker RSS 若异常再进入专项诊断；
- no-failure overhead、metadata bytes 和 driver scale 可测；
- API 只暴露五原语与 RayModule 式配置；
- 所有论文性能数字来自 V2.5 实现自身。

在这些条件达成前，不增加 checkpoint、exactly-once、distributed metadata、通用 matcher 或
更多 primitive。

---

## 26. 最终架构判断

V2.5 的核心不是“给每个 item 挂更多 metadata”，而是承认一个不可消去的事实：

> 输出可以不存在，但语义调用仍然发生过。

Logical Grain 使这个负事实与普通成功 item 使用同一套 lineage：

```text
grain inputs
→ terminal outcome
→ zero or more ordered emissions
```

由此，同一模型同时服务：

- dynamic cardinality；
- reorder-safe rebatching；
- exact fan-in closure；
- multi-parent lineage；
- backward attribution；
- forward containment；
- retry identity；
- atomic publication。

它比 item-only + 多套补丁更信息完备，也比 V2.2 的重型 execution IR 更小。V2.5 的实现和论文
都应围绕这一条主线展开。
