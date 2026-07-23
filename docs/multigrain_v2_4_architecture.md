# Multigrain V2.4 架构设计稿（clean-slate，代码级）

状态：**V2.4 = 另起炉灶的优雅精简实现。** 不引用、不继承 V2/V2.1/V2.2/V2.3 的任何内部机制或
`experimental/multigrain/` 的重 IR。本文用具体 class / dataclass / 方法签名 / 模块依赖 /
数据流走查来落地设计——目的是让抽象经得起代码级检验，把"没想到的点"逼出来。

要保住的不是旧代码，而是**可复现性**：现有 `mg_bridge` 适配层调真实 MinerU/vLLM 函数，
E1/E3 数字（368 PDF / 4×H20 / 1.72× / record-vs-shard 0-vs-1.5×）产自 workload + 调度行为，
不产自旧 IR。V2.4 一旦暴露五原语 API，把 bridge 重指过来，**现有数字成为新核心的验收测试**
（见 §11）。

---

## 0. 一句话命题（论文头条 = 本文头条）

> **一根细粒度 lineage 同时给出 (a) reorder-safe 弹性重排 和 (b) record-level 故障隔离，
> 且通过 value-pure UDF API 对用户免费——这一组合在 cardinality-changing(1:N/M:N)的 AI
> pipeline 上，是 Ray Data / Spark / Trident 都不同时具备的。**

"高效 / 容灾 / 易用"是这句话的**结果**，不是三个并列卖点。本文每一个 class 都应能追溯到这句
thesis 或下面 6 个承重决策之一；不能追溯的，就是过度设计。

---

## 1. 六个承重决策（整个实现从这里推导）

| # | 决策 | 推出什么 |
|---|---|---|
| **D1** | 一个 item 只有 `id` / `value-ref` / `direct-parents(带 role)` 三样，**不物化传递祖先/op-path/display** | lean lineage；backward trace = 惰性图遍历 |
| **D2** | `id` 由 **driver 从 provenance 纯函数算出**，UDF 永不见 id | content-address ⇒ rebatch/retry 不变 + 幂等去重 + M1 value-purity 前提 |
| **D3** | fan-in 只产 multi-parent `ItemMeta`，用**两个函数**兑现（`reduce_fiber` + `join_by_key`），**不建抽象协议** | Reduce / diamond / Relate 复用同一数据模型与同一 join 函数 |
| **D4** | **value 永不过 driver**；driver 只持 metadata + ObjectRef | actor-to-actor 数据面（day-one 不变量）；直接修掉 driver-centric scale 软肋 |
| **D5** | **error 是数据**，与 value 同管道返回，无 collector actor | 多粒度恢复 = 同一路径的不同入口 |
| **D6** | **recovery unit ≡ 该节点的 lineage unit**；scope 由 value-purity + within-stage 冻结 | 不单独设计恢复；它是 lineage 的另一个视角 |

**反膨胀纪律（贯穿全文）：** 没有两个真实用例之前不做抽象；没有真实 workload 之前不加原语；
没有真实故障场景之前不加恢复档位。`join_by_key` 之所以成立，正因为它有 **diamond + Relate
两个真实用例**；`Matcher` 协议被否掉，正因为它是为想象中的更多 matcher 建的。

---

## 2. 模块依赖图（无环，自底向上）

```
identity.py ─┐
             ├─► item.py ─┬─► graph.py ─────────────────► api.py
             │            │                                 ▲
             │            ├─► fanin.py ────┐                 │
             │            ├─► lineage.py ──┤                 │
             │            ├─► recovery.py ─┼─► executor.py ──┘
             │            ├─► scheduler.py ┤        ▲
             │            └─► worker.py ───┘        │
             └────────────────────────────► metrics.py
```

| 模块 | 唯一职责 | 关键导出 |
|---|---|---|
| `identity.py` | provenance → `ItemId` 的纯函数 | `ItemId`, `source_id/map_id/expand_id/reduce_id/relate_id` |
| `item.py` | 被动数据模型 | `ItemRef`, `ParentRef`, `ItemMeta`, `Port`, `Shard`, `Failure` |
| `graph.py` | authoring → 不可变图 | `Primitive`, `NodeSpec`, `CompiledGraph`, `_Tracer` |
| `fanin.py` | fan-in 的两个具体函数 + fiber 结构 | `Group`, `FiberLedger`, `reduce_fiber()`, `join_by_key()` |
| `lineage.py` | parent↔dependent 索引 + trace + suppression | `LineageIndex` |
| `recovery.py` | 故障分类 + 最小策略 + quarantine | `RecoveryPolicy`, `BadRecordError`, `decide_record/shard` |
| `scheduler.py` | pending 队列 + cost-aware packing | `Scheduler`, `PendingItem`, `BatchPlan` |
| `worker.py` | Ray actor：selector→列→UDF→列+offset+error | `Worker`, `Invocation`, `WorkerResult`, `Offset` |
| `executor.py` | microbatch DAG 协调 | `Executor`, `RunResult` |
| `metrics.py` | 匿名 stage trace | `NodeMetric`, `RunMetrics` |
| `api.py` | 用户面（五原语） | `Pipeline`, `Map/Filter/Expand/Reduce/Relate`, `Port`, `keyed` |

**注意 `fanin.py` 里只有两个函数** —— 不是协议、不是类层次。`Matcher` 抽象已否决。

**Phase-0 起步塌缩**：先把 `fanin+lineage` 并入 item 邻域、`scheduler+recovery+worker` 并入
executor 邻域，起步 ~5 文件；某模块出现独立合同与测试需求时再拆。

---

## 3. `identity.py` —— content-address 的全部真相

**核心洞察：identity 是 DAG，不是 path。** Map 复用父 id、Expand 向下分叉、Reduce 向上收敛、
Relate 合并多父。`ItemId` 必须是可递归、可比较、**顺序无关**的规范值。

**`ItemId` 是 opaque 包装类，不是裸 tuple 别名。** 这道边界现在就立好（几乎免费），目的是让
将来的 interning 升级**永远关在 `identity.py` 一个文件里**。MVP 内部用可读嵌套 tuple（精确、
无碰撞假设、好调试）；升级 interning 时只改本模块，下游零改动（见"升级路径"）。

```python
@dataclass(frozen=True, slots=True)
class ItemId:
    _repr: tuple            # 内部表示；只有 identity.py 访问。__eq__/__hash__ 由 dataclass 生成
    # MVP：_repr = 可读嵌套 tuple；升级：_repr 换成 intern 表的 int handle，本类外无感知

def source_id(source: str, position: int) -> ItemId:
    return ItemId(("src", source, position))              # position 只在 admission 冻结一次

def map_id(parent: ItemId) -> ItemId:
    return parent                                         # Map/Filter 复用父 identity

def expand_id(node: str, parent: ItemId, ordinal: int) -> ItemId:
    return ItemId(("exp", node, parent._repr, ordinal))   # ordinal 取自 UDF 返回内层 list 位置

def reduce_id(anchor: ItemId) -> ItemId:
    return anchor                                         # 输出复用 parent identity

def relate_id(node: str, parents: dict[str, ItemId]) -> ItemId:
    items = tuple(sorted((r, p._repr) for r, p in parents.items()))
    return ItemId(("rel", node, items))                   # 排序 ⇒ 顺序无关

def explain(id: ItemId) -> str:                           # debug 显示走这里，不在别处拆 _repr
    ...
```

**升级路径（纯 driver 本地，~20 行，下游零改动）：** `ItemId` 全程**不跨进程**（worker 看不到
id，见 §9），故 interning 是纯内存的 driver 本地事，无分布式表难题。升级时在 `identity.py` 加一张
`_TABLE: dict[tuple,int]`，`derive_*` 把结构 `_intern()` 成 int、`_repr` 存 int handle；`==`/`hash`
自动变 O(1)。`LineageIndex`/`FiberLedger`/`join_by_key`/executor 只用 `==`/`hash`/dict-key，对
tuple 与 int 行为一致，**一行不改**。

**为什么下游天然不用拆 `ItemId`：** 因 D1 把直接父存在 `ItemMeta.parents` 里，所有 lineage
遍历（backward trace、diamond 找祖先 id）走 `parents`，**从不从 `ItemId` 反解结构**。opaque
边界是自然的，不是强加的。

**端口消歧（关键 corner case）：** Map/Filter/Reduce **复用**父 id，则 diamond 里
`page→OCR` 与 `page→figure` 两分支输出有**相同 ItemId**。靠 **`ItemRef = (port, ItemId)`**
区分——同 port 内唯一，跨 port 允许同 id，而这**正是 diamond 对齐的机制**。所以"Map 复用 id"
是 diamond 正确性的**前提**，不是妥协。

**diamond vs Relate 的输出 id（小设计规则）：**

- **diamond**（多父共享一个祖先 id，如都源自 page P）：输出**复用该共享 id**（enriched page 仍
  key 为 P），下游"按 document Reduce"天然继续可用。
- **Relate**（多父是不同实体，如 frame F + audio A，无共享 id）：输出用 `relate_id({F,A})`。

**reorder-invariance 义务（对应 M1）：** `derive_*` 的输入与调度序无关——`expand_id` 的
ordinal 取自 UDF 返回内层下标（rebatch 之前），`relate_id` 的父集排序。故 identity 是输入
multiset 的函数。

---

## 4. `item.py` —— 被动数据模型（D1 + D4）

```python
@dataclass(frozen=True, slots=True)
class ItemRef:
    port: str        # 生产该 item 的 node 输出端口
    id: ItemId

@dataclass(frozen=True, slots=True)
class ParentRef:
    role: str        # "primary"/"parent"/"left"/"right"/... 由原语与 UDF signature 决定
    ref: ItemRef

@dataclass(frozen=True, slots=True)
class ItemMeta:
    id: ItemId
    parents: tuple[ParentRef, ...]      # 直接父即执行所需 lineage 的全部；多父 = fan-in 输出
    ordinal: int | None = None          # 仅 Expand child 有意义

@dataclass(frozen=True, slots=True)
class Failure:
    ref: ItemRef
    node: str
    kind: str            # "infra" | "bad_record" | "udf" | "suppressed"
    retryable: bool
    message: str
    causes: tuple[ItemRef, ...]   # 归因到的上游 item（backward trace 结果快照）

@dataclass
class Shard:
    ref: "ray.ObjectRef"   # 指向 list[value]（或列式 block），driver 不反序列化
    n: int

@dataclass
class Port:
    """一个 node 的一个逻辑输出端口。metas/locations 平行数组；values 在 shards 里。"""
    node: str
    metas: list[ItemMeta]
    locations: list[tuple[int, int]]     # 平行于 metas：(shard_index, row_index)
    shards: list[Shard]
    failures: dict[ItemId, Failure] = field(default_factory=dict)
    keys: list | None = None             # 仅 Relate 前置投影时存在：平行于 metas 的小 join-key 列

    def alive_ids(self) -> set[ItemId]: return {m.id for m in self.metas}
```

**关键：多父 fan-in（diamond / Relate）不新增任何 dataclass**——它只是 `parents` 元组长度 > 1
的普通 `ItemMeta`。唯一为 Relate 增加的是 `Port.keys`：一个平行于 metas 的**小 hashable 列**
（不是复杂结构），且**仅在 value-key join 时存在**；diamond 用祖先 id 对齐，`keys` 恒为 None。

**D4 落地检查：** `Port` 无 `value` 字段；driver 只通过 `locations`→`shards[i].ref` 把 selector
交给 actor。`display_key` 之类 debug 信息不进核心，需要时挂旁路 dict，delivery 后释放。

---

## 5. `graph.py` —— authoring → 不可变图

```python
class Primitive(enum.Enum):
    MAP = "map"; FILTER = "filter"; EXPAND = "expand"; REDUCE = "reduce"; RELATE = "relate"

@dataclass(frozen=True)
class NodeSpec:
    id: str
    kind: Primitive
    udf_cls: type
    init_args: tuple; init_kwargs: dict          # pre_init 配置
    replicas: int; batch_size: int
    ray_resources: dict                          # num_cpus/num_gpus/... 白名单
    error_policy: str                            # "raise" | "isolate"
    recovery: "RecoveryPolicy"
    cost_fn: Callable | None
    inputs: tuple[str, ...]                      # 上游 node id（按 role 顺序）
    roles: tuple[str, ...]
    on: dict[str, str] | None = None             # 仅 Relate：role -> value 字段（value-key join）

@dataclass(frozen=True)
class CompiledGraph:
    nodes: dict[str, NodeSpec]
    order: tuple[str, ...]                       # 拓扑序
    sources: tuple[str, ...]
```

`api` 层用 `_Tracer` 在 `Pipeline.forward()` 里符号执行，把每个 `Map(...)(port)` 调用记成一条
`NodeSpec` 边，`compile()` 成不可变 `CompiledGraph`。authoring 阶段**不创建 actor**。

**多输入的两种 fan-in 由 `NodeSpec` 自动区分，无需新原语：**

- `Reduce` 单输入且上游是 `Expand` → `reduce_fiber`（fiber 重建）。
- `Reduce`/`Map` 多输入且 `on is None` → **diamond**，`join_by_key(key=祖先 id)`。
- `Relate`（`on` 非空）→ **value-key join**，`join_by_key(key=投影 key)`。

**diamond 没有独立用户面**：它就是"多输入节点按 id 自动对齐"（解决旧开放点 #3）。

---

## 6. `fanin.py` —— 两个函数兑现全部 fan-in（D3）

```python
@dataclass(frozen=True)
class Group:
    parents: tuple[ParentRef, ...]                 # 匹配到的父元组 → 输出 item 的 parents
    role_rows: dict[str, list[tuple[int, int]]]    # role -> [(shard,row)]，喂给 UDF
    complete: bool                                 # 必需 role 是否齐全
    missing_causes: tuple[ItemRef, ...]            # 缺失/失败的父（供 fail-closed 抑制）

@dataclass
class FiberLedger:                                 # Reduce 专用：每个 Expand parent 一个
    parent: ItemRef; expected: int
    succeeded: dict[int, ItemRef]; failed: dict[int, Failure]
    def state(self) -> str:                        # "ready" | "suppressed" | "pending"
        if self.failed: return "suppressed"        # 任一 required child 失败 → fail_closed
        if len(self.succeeded) == self.expected: return "ready"   # 空 fiber 也 ready
        return "pending"


def reduce_fiber(children: Port, ledger: dict[ItemRef, FiberLedger]) -> Iterator[Group]:
    """Reduce：按 child 的 parent id 聚组；用 ledger.expected 判 complete。
    完整 fiber 内按 child ordinal 恢复稳定序；一个 fiber 不跨 Group。"""
    ...


def join_by_key(inputs: list[Port], key_of, roles: tuple[str, ...]) -> Iterator[Group]:
    """一个函数服务两个真实用例：
      diamond:  key_of(port, i) = <该 item 的共享祖先 id>   ← 纯 metadata，零 value 访问
      relate:   key_of(port, i) = port.keys[i]              ← 前置投影出的 value key

    - inner join：某 role 缺该 key 且缺因是"根本不存在" → 正常无匹配，静默丢弃；
                  缺因是对方 port.failures 命中 → missing_causes 非空 → fail-closed。
    - M:N：同 key 多行时枚举完整笛卡尔积，按 (key, 各 role parent_id) 排序 ⇒ 输出集合与序确定
           （消除旧码 seen_keys 把 M:N 退化 1:N 的 bug）。
    """
    index: dict[Any, dict[str, list[int]]] = {}    # key -> role -> [row]
    fails: dict[Any, dict[str, ItemRef]] = {}
    for port, role in zip(inputs, roles):
        for i, meta in enumerate(port.metas):
            index.setdefault(key_of(port, i), {}).setdefault(role, []).append(i)
        for fid, f in port.failures.items():
            fails.setdefault(key_for_failed(port, fid), {})[role] = f.ref
    ...  # 对每个 key：齐 role 则出笛卡尔积 Group；缺 role 则按上面 inner/fail 分流
```

**为什么不是协议：** 实际只有 `reduce_fiber`（有 expected-count 特例）和 `join_by_key`（diamond
与 Relate 共用）两个用例。两个函数就是全部；`Matcher` Protocol + 3 类是为想象用例建的，否决。

### 6.1 value-key 的唯一额外成本：一个小投影列（不是复杂结构）

diamond 的 `key_of` 走 metadata（祖先 id），**零 value 访问**。Relate 的 key 是 value 函数，但
D4 规定 driver 无 value → 编译时给 Relate 前插一个**轻量 key-projection worker pass**：按
`NodeSpec.on[role]` 把字段投影成 `Port.keys`（小 hashable 列，value 仍在 object store）。
**这是唯一为 Relate 增加的东西**，一列而已。

### 6.2 明确划入 future（执行层，不动数据模型）

分布式/分区/spill join、hot-key 处理、大 join 基数——首版用**单节点/driver 内 key 索引**，
覆盖绝大多数治理任务（视频内对齐、对中等参考集匹配）。这些是**执行 scale**，将来接入时
`join_by_key` 的签名与 `ItemMeta` 数据模型都不变。

---

## 7. `lineage.py` —— 索引 + trace + suppression（D1/D6）

```python
class LineageIndex:
    """每 microbatch 构建，result delivery 后释放。"""
    def __init__(self): self._dependents: dict[ItemRef, list[ItemRef]] = {}

    def add(self, port: str, meta: ItemMeta) -> None:
        ref = ItemRef(port, meta.id)
        for p in meta.parents:
            self._dependents.setdefault(p.ref, []).append(ref)

    def dependents(self, ref: ItemRef) -> list[ItemRef]:      # forward
        return self._dependents.get(ref, [])

    @staticmethod
    def trace_sources(meta: ItemMeta, resolve) -> set[ItemRef]:  # backward = 惰性递归
        ...  # 沿 meta.parents 递归到无父 source；成本只在诊断时付
```

**D1 落地检查：** backward trace 不需存储结构（parents 就在 `ItemMeta` 里）；只有 forward 用的
`_dependents` 被物化，仅存活于当前 microbatch。砍掉了旧 `PortBatch` 的
`ancestors/ancestor_display/ordinals` 三套结构。**diamond 与 Relate 的多父在这里零特殊处理**——
一个多父 item 只是登记为多个 parent 的 dependent，"任一必需父失败 → 抑制"自动成立。

---

## 8. `recovery.py` —— 最小可用，多粒度同一条路（D5/D6）

**砍到最小**：只保留兑现"item 故障隔离"claim 所必需的档位。doc 15 的 deferred drain /
IsolationBudget work-factor / drain_scope / stage-global epoch **全部划入 future**（同一条
`quarantine→cascade` 路径的新入口，将来 workload 需要时再接）。

```python
class BadRecordError(Exception):
    def __init__(self, index: int, retryable: bool = False): ...  # 默认确定性 poison → 隔离

@dataclass(frozen=True)
class RecoveryPolicy:
    error_policy: str = "raise"        # "raise" | "isolate"
    max_shard_retries: int = 2         # infra 错误的有限预算
```

两字段。没有嵌套 budget、没有 timing、没有 drain scope。

**粒度阶梯（都汇入 `Port.failures` 一条数据路径）：**

```
infra 错误           → 丢弃 batch、换 actor、≤ max_shard_retries 重试（不生 ItemFailure）
显式 BadRecordError  → 隔离该行 → 健康行"作为一个 batch 重跑"（scar-tissue #2）→ 保留已确认
generic UDF 错误：
  error_policy=raise   → 终止当前 microbatch
  error_policy=isolate → 朴素有界二分定位坏行（见下 trim point）
missing consumer     → reduce_fiber/join_by_key 的 complete=False → fail-closed 抑制（正交）
```

**唯一可再裁的 trim point：** `error_policy="isolate"` 下对 generic 错误的**朴素有界二分**
（递归对半分 + 一个简单 call 上限，无 work-factor）。它保住"opaque 错误也能隔离"这个 claim，
约 40 行。若要更极致精简，可首版只认显式 `BadRecordError(index)`、generic 一律 raise——但那会
让"无法自报坏行的 UDF"退化成整 batch 失败。**默认保留朴素二分。**

**D5 落地检查：** sink 是被动的 `Port.failures` + `WorkerResult.errors`，随 batch 结果返回、
被 scheduler 合并——**无 collector actor**。`decide_record` 在 worker 包装层，`decide_shard` 在
scheduler，共享同一 passive policy。

---

## 9. `worker.py` / `scheduler.py` / `executor.py` —— 执行协议（D2/D4）

### 9.1 worker（Ray actor，持久 UDF 实例）

```python
@dataclass(frozen=True)
class Invocation:                       # 一次逻辑 UDF 调用的输入选择
    role_rows: dict[str, list[tuple[int, int]]]   # role -> [(shard,row)]

@dataclass(frozen=True)
class BatchPlan:
    node: str; attempt: int
    invocations: tuple[Invocation, ...]

@dataclass(frozen=True)
class Offset:                           # 输出行 → 回溯信息（driver 用来算 id）
    invocation: int                     # 属于第几个 invocation
    out_port: int                       # 多输出时的端口下标
    ordinal: int                        # Expand：内层 list 位置；其余=0

@dataclass
class WorkerResult:
    columns: list["ray.ObjectRef"]      # 每个输出端口一个 ObjectRef（value 不回 driver）
    offsets: list[Offset]
    errors: list[tuple[int, "BadRecordError"]]   # (invocation, error)

class Worker:
    def __init__(self, udf_cls, init_args, init_kwargs):
        self.udf = udf_cls(*init_args, **init_kwargs)      # pre_init 配置在此
    def run(self, plan: BatchPlan, *shard_refs) -> WorkerResult:
        # 1. 按 selector 从 resolved shard 构造 list 列
        # 2. 调 self.udf.run(...)（value-in / value-out，不见 id）
        # 3. 回 columns(ObjectRef) + offsets + invocation-local errors
        ...
```

**D2 落地检查：** worker 只回 invocation-local offsets 与 errors，**从不生成 ItemId**。driver 拿
`Offset` + 该 invocation 的父 `ItemMeta` → 调 `identity.derive_*` 算出输出 id。key-projection
pass 也是一个 `Worker`，UDF = 取 `on[role]` 字段。

### 9.2 scheduler（cost-aware packing，保持简单）

```python
@dataclass
class PendingItem:
    ref: ItemRef; parents: tuple[ParentRef, ...]; ordinal: int | None
    location: tuple[int, int]; cost: float

class Scheduler:
    def plan(self, node: NodeSpec, pending: list[PendingItem]) -> list[BatchPlan]:
        # cost_fn 缺省=每项权重 1；否则简单 LPT/greedy；尊重 batch_size；
        # 不变量：同一 item 一个 attempt 只进一个 active BatchPlan。
        ...
```

### 9.3 executor（microbatch DAG 协调，串起全部）

```python
class Executor:
    # driver 状态：
    #   ports:    dict[str, Port]                各 node 输出
    #   ledgers:  dict[ItemRef, FiberLedger]     Expand 子计数权威
    #   lineage:  LineageIndex                   当前 microbatch
    #   attempts: dict[str, int]                 stale-drop
    #   pools:    dict[str, list[Worker]]        持久 actor（scar-tissue #1）
    #   quarantine                               recovery（Failure 数据）
    def run(self, graph: CompiledGraph, sources) -> "RunResult": ...
```

microbatch 主循环：

```
admit source rows → 编 ItemMeta（source_id）
loop 直到所有 node quiesce:
  for node in ready(graph):
    if fan-in:                                   # 用 §6 两函数之一产 Group
      groups = reduce_fiber(...) / join_by_key(...)
      pending = [g for g in groups if g.complete]
      suppress(g for not complete)               # fail-closed → Port.failures
    else:
      pending = collect_inputs(node)             # Map/Expand/Filter：上游 items
    plans   = scheduler.plan(node, pending)      # LPT/rebatch
    results = dispatch(plans)                     # → 持久 pool 的 Worker.run，attempt token
    for r in results:
      metas, fails = ingest(r, node)             # driver 用 offsets+父算 id；errors→Failure
      recovery.apply(node, metas, fails)         # 入口，写回 Port/ledger/quarantine
      lineage.add(...); update_ledgers(...)      # Expand 发布 expected count
deliver RunResult（outputs + failures + metrics）
```

**stale-drop：** 每 `BatchPlan` 带 `attempt`；actor 死则 `attempts[node]++` 并重编 plan；旧
attempt 晚到比对 token 后整体丢弃。配合 content-address id ⇒ 幂等，不重复/不错配。

---

## 10. 必须保持的不变量（代码级）

1. `id` 不含物理 batch / actor / 完成序（`derive_*` 输入检查）。
2. Map/Filter/Reduce 输出复用父 `id`；diamond 复用共享祖先 id；跨 port 由 `ItemRef` 消歧。
3. Expand child `id` 由 `(node, parent.id, ordinal)` 唯一确定，ordinal 取自 UDF 返回内层位置。
4. `reduce_fiber` 只消费 `state()=="ready"` 的 fiber；一个 fiber 不跨 `BatchPlan`。
5. 多输入按 `ItemId`(diamond) 或投影 key(Relate) 对齐，**绝不按物理 position zip**。
6. 正常 Filter-false / Relate 无匹配（不在 `failures`）与 `Failure` 严格区分。
7. generic error 默认策略下不静默变 missing（`error_policy` 显式）。
8. failure 只沿 `LineageIndex.dependents` 传播，不静默丢整批。
9. 大 value 不经 driver 反序列化（`Port` 无 value 字段）。
10. 同一 item 同一时刻 ≤ 1 个 active `BatchPlan` attempt。
11. **`ItemId` 只可比较 / hash / 做 key；只有 `identity.py` 构造与解释它（`explain`）。lineage
    遍历一律走 `ItemMeta.parents`，绝不从 `ItemId` 反解结构。** 这条守住 interning 升级只关在
    `identity.py`。

---

## 11. Scope 边界（论文必须诚实标注）

- **五原语全实现**（含 Relate）；但 **Relate 首版 = 单节点/driver 内 key 索引 + 小投影列**，
  **分布式/分区/spill/hot-key join 划入 future**（执行层，不动 `ItemMeta` 与 `join_by_key` 签名）。
- **record 级恢复首版仅 Map。** Reduce/Relate/Expand/Filter 的 attributable unit 见 doc 15 §2，
  未落地前一律 fail-closed at grain；碰到未实现组合 `raise NotImplementedError`。
- **"多粒度"= stage 内故障归属阶梯**（record→二分→shard→document→job），**非跨 DAG lineage
  重放**；deferred drain / adaptive-split-as-tier 划入 future。
- **正确性 = keyed-equality modulo batching jitter**（vLLM 连续批处理对自己都非位级可复现）；
  correctness checker = token-Jaccard。
- **value-purity 是硬边界**：有状态/全局排序/跨记录去重/迭代 dataflow/不安全 side-effect 不在
  reorder & retry 保证内。
- **不做**：driver crash recovery、checkpoint/resume、exactly-once、durable 跨-run identity、
  distributed lineage authority、通用 provenance 查询语言、百万级 `ItemId` interning/列式压缩。

---

## 12. Scar-tissue（扔代码，不扔教训——day-one 就在核心里）

1. **持久 actor pool / factory 模式**：vLLM 只 load 一次——**1.57× 的真实来源，不是 LPT**。
   `Executor.pools` 从第一天就持久。
2. **record 隔离快路径**：隔离 bad row 后健康行**作为一个 batch 重跑**（非逐行序列化）；
   E3 里 `run_calls` 143→4 靠它。
3. **M:N 确定性枚举**（§6）。
4. **正确性判据 = token-Jaccard**（§11）。

---

## 13. Bridge-as-contract 复现计划（拆掉 clean-slate 的唯一风险）

1. 写 §3–§9 核心 + §12 scar-tissue → 用 dummy op 端到端跑通命题闭环
   （`Expand→Map→Reduce` + dynamic fan-out + rebatch + fiber 重建 + record 隔离）。
2. 把现有 `mg_bridge` op（调真实 `load_images_from_pdf`/`batch_two_step_extract`/`vlm_union_make`）
   重指到 V2.4 五原语——bridge 是薄适配，op 内部数学不变。
3. **现有数字变成验收测试**：重跑 368 PDF / 4×H20，期望 ≈1.72×、OCR bubble ≈0.065、output
   token-Jaccard median ≈0.9957；重跑 poison E3，期望 record-level 0 冗余、shard-level 1.5×。
4. 再推进多节点 scale（D4 的 actor-to-actor 兑现）+ Relate 的视频时间对齐 workload。

---

## 14. 包布局与命名

原型：`rayorch/experimental/multigrain_v2_4/`（版本号仅为与 v2.3 区分，研究期临时命名）。
**OSS 发布目标改名 `rayorch.multigrain`**，版本靠 release 管理。内部匿名证据 schema
（RunEnvelope/NodeMetric 见 doc 14）从 OSS 核心剥离，默认 example 跑公开语料。

---

## 15. 开放点（都不动核心结构）

已解决：
- **diamond 无独立用户面** —— 是多输入节点按 id 自动对齐（§5），非新原语。
- **Relate 保留** —— 数据结构零新增（多父 `ItemMeta`），仅 +1 小投影列；分布式 join 划 future。
- **fan-in 无协议** —— `reduce_fiber` + `join_by_key` 两函数。
- **`ItemId` 表示** —— 现在立 opaque 包装类边界（§3、不变量 #11），MVP 用可读 tuple；interning
  推迟，升级只关在 `identity.py`，下游零改动。

待你拍板（可后置）：
1. **opaque 错误二分**：默认保留朴素有界二分（§8 trim point）；若要更极致精简，可首版只认显式
   `BadRecordError`。
2. **首篇是否跑视频时间对齐 workload** 证 Relate 的 multi-parent 抑制（vs 仅 generality 声明）。
