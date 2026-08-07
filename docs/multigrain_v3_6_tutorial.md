# MultiGrain v3.6 从零理解与维护教程

这份教程面向第一次接触 MultiGrain、数据流编译器和细粒度 lineage 的读者。读完后，
你应该能够：

- 写出并运行一个 `RayModule + F.*` Pipeline；
- 解释 Port、Domain、Entity、Item、Grain 和 Expansion 分别是什么；
- 沿着 compiler、MicrobatchEngine、Executor、Worker 追踪一条数据；
- 判断 Filter、Expand、Reduce、Broadcast 和 Optional 如何改变状态；
- 定位常见编译、死锁、失败传播和 Worker ABI 问题；
- 在不引入飞线的前提下维护或增加一个 primitive。

本教程描述的是首次发布前已经完成 breaking change 的 v3.6。动态语义的精简合同见
[`multigrain_v3_6_design.md`](multigrain_v3_6_design.md)，compiler 设计边界见
[`multigrain_v3_6.md`](multigrain_v3_6.md)。

建议按目标选择阅读路径：

| 目标 | 阅读顺序 |
| --- | --- |
| 先会写 Pipeline | 1 → 2 → 3 → 4 → 5 |
| 理解一次执行为何得到这个结果 | 6 → 7 → 8 → 9 → 10 |
| 开始维护源码 | 11 → 12 → 13 → 14，再回查相关状态机 |

---

## 1. 先建立一个最小直觉

MultiGrain 做的事情可以先粗略理解为：

```text
用户写一张由 RayModule 和 F.* 组成的符号数据流图
    ↓
compiler 在启动 Ray 前验证并生成 RuntimePlan
    ↓
Executor 把输入切成多个 microbatch，每个 microbatch 拥有一个 MicrobatchEngine，
并让它们共享持久 Ray actors
    ↓
MicrobatchEngine 为每个逻辑 Entity 维护 Item、Expansion 等语义事实，
其 DispatchState 维护 Grain 生命周期
    ↓
Worker 只读取值、批量执行 UDF，并返回结构化报告
    ↓
MicrobatchEngine 提交报告、传播结构状态，最后按 lineage 顺序物化输出
```

最重要的边界是：

```text
RayModule 执行业务计算。
F.* 声明数据的结构关系。
MicrobatchEngine 管理身份与状态，但不解释业务 payload。
Executor 管理 Ray，但不重新解释 primitive 语义。
```

### 1.1 先看完整组件协作图

先不要深入类和字段。整个系统只有三段主流程：authoring/compile 建图，driver/runtime
推进状态，actor/worker 执行业务值计算。

```mermaid
flowchart LR
    subgraph Compile["Authoring 与编译期（Ray-free）"]
        User["Pipeline.forward<br/>RayModule + F.*"] --> Logical["LogicalProgram<br/>Port / Domain / Origin"]
        Logical --> Compiler["Compiler<br/>verify / analyze / canonicalize / lower"]
        Compiler --> Plan["RuntimePlan<br/>effects / indexes / layouts / pools"]
    end

    subgraph Driver["Driver 运行期"]
        Executor["Executor<br/>microbatch 并发 / actor capacity / RPC"]
        MicrobatchEngine["MicrobatchEngine<br/>Item / Expansion / Entity 语义传播"]
        Dispatch["DispatchState<br/>Grain phase / generation / queues"]
        Recovery["RecoveryPolicy<br/>纯决策"]
        Executor -->|"admit / reserve / commit"| MicrobatchEngine
        MicrobatchEngine -->|"唯一 Grain 状态入口"| Dispatch
        Executor -->|"failure facts"| Recovery
        Recovery -->|"RecoveryAction"| Executor
        MicrobatchEngine -->|"infra retry counts"| Recovery
        Recovery -->|"allow / deny"| MicrobatchEngine
    end

    subgraph Actor["Ray actor"]
        Worker["Worker<br/>恢复值 / batch UDF / 生成报告"]
        Store["Block store<br/>BlockRef / RowBinding"]
        Worker <-->|"get / put business values"| Store
    end

    Plan --> Executor
    Plan --> MicrobatchEngine
    Executor -->|"GrainPlan + CallOutputLayout"| Worker
    Worker -->|"WorkerResult"| Executor
```

读这张图时只记住五个 ownership：

| 组件 | 唯一拥有的东西 | 不做什么 |
| --- | --- | --- |
| Compiler | 从 logical facts 生成完整 RuntimePlan | 不启动 Ray、不执行 UDF |
| Executor | actors、ObjectRefs、microbatch 并发额度 | 不解释 Filter/Reduce 等语义 |
| MicrobatchEngine | Item/Expansion/Entity publication 与传播 | 不执行业务 UDF、不维护第二份 ready queue |
| DispatchState | Grain phase/generation 与 normal/immediate/tail queues | 不发布 Item/Expansion、不决定业务策略 |
| Worker | value-only batch UDF 和 Worker ABI | 不读取 Program/MicrobatchEngine、不创建逻辑身份 |

箭头表示主要协作，不要求为了形式上的单向依赖增加适配层。真正禁止的是跨层写状态，
或者同一个事实出现两个 authority。

后文 Mermaid 使用统一约定：实线箭头表示调用或 DTO 传递，虚线返回箭头表示返回值，
指向自身的箭头表示该组件修改自己拥有的状态；图中的“连接”不自动意味着对方可以读取
或修改组件内部表。

### 1.2 第一个可运行 Pipeline

```python
import rayorch.experimental.multigrain_v3_6 as mg


class Double:
    def run(self, values):
        # values 是一个 batch 列，不是单个值。
        return [value * 2 for value in values]


class DoublePipeline(mg.Pipeline):
    def __init__(self):
        self.double = mg.RayModule(Double).ray_options(
            batch_size=8,
            replicas=1,
            num_cpus=0,
        )

    def forward(self, values):
        return self.double(values)


with mg.Executor(DoublePipeline()) as executor:
    result = executor.run([1, 2, 3])

assert result.outputs == [2, 4, 6]
```

`Executor` 会在 Ray 尚未初始化时调用 `ray.init()`。退出 context manager 总会终止本
Executor 创建的 actors；如果 Ray 也是本 Executor 初始化的，它同时负责
`ray.shutdown()`。如果调用方已经初始化 Ray，Executor 视其为外部资源，关闭时不会
影响共享 runtime。

这里有三个容易误解的地方：

1. `Pipeline.forward()` 编译时执行一次，但参数是符号 `Port`，不是真实的 `[1, 2, 3]`。
2. `Double.run()` 运行时接收批量列，例如 `[1, 2, 3]`，返回值也必须按 batch 对齐。
3. `RunResult` 只持有输出和不可变指标快照，不持有活的 Engine 或 RuntimeState。

常用结果字段是：

```python
result.outputs
result.elapsed_s
result.calls          # tuple[CallMetrics, ...]
result.microbatches   # tuple[MicrobatchMetrics, ...]
result.rpc_count
result.actor_count
result.released_values
```

`CallMetrics` 用 `call_index + udf_name` 标识调用点。它的 `rpcs/grains/retries` 是本次
`run()` 的统计；`worker_snapshots[*].lifetime_calls` 则刻意表示持久 actor 的累计值。

可以在不执行 Ray 的情况下查看编译结果：

```python
compiled = DoublePipeline().compile()
print(compiled.explain_text())
```

`compile()` 只建立并验证静态数据流；`Executor` 才创建 actor 和处理真实值。

---

## 2. 七种必须分清的身份

理解整个框架的关键不是记住类名，而是分清静态位置、动态身份和执行行为。

| 概念 | 简化定义 | 静态/动态 |
| --- | --- | --- |
| Port | 数据流图上的一个逻辑位置 | 静态 |
| Domain | 一组同粒度 Entity 的身份空间 | 静态 |
| Entity | Domain 中一次逻辑 occurrence | 动态 |
| Item | 一个 Port 对一个 Entity 的结果 | 动态 |
| Call | 一个 RayModule 在图中的调用点 | 静态 |
| Grain | 一个 Call 对一个 Entity 的逻辑调用 | 动态 |
| Expansion | 一个 parent Entity 的 Expand 结果 | 动态 |

它们的身份公式是：

```text
EntityRef    = DomainRef × occurrence
ItemRef      = PortRef × EntityRef
GrainRef     = CallRef × EntityRef
ExpansionRef = child DomainRef × parent EntityRef
```

### 2.1 Port 不是数据容器

下面的 `doubled` 是一个符号 Port：

```python
def forward(self, values):
    doubled = self.double(values)
    return doubled
```

它表示“Double 这个 Call 的第 0 个逻辑输出位置”，不保存某一行真实值，也不保存
Ray `ObjectRef`。

同一个 Port 对不同 Entity 会形成不同 Item：

```text
ItemRef(doubled_port, entity_0)
ItemRef(doubled_port, entity_1)
ItemRef(doubled_port, entity_2)
```

### 2.2 Domain 表示粒度，不表示 Python 类型

假设 root 输入是两份文档：

```text
root Domain
├── document Entity 0
└── document Entity 1
```

每份文档展开出 pages 后，会产生新的 child Domain：

```text
document Domain
├── document 0
│   ├── page 0
│   └── page 1
└── document 1
    └── page 0
```

文档和页面即使内部整数编号相同，也不会意外对齐，因为 Domain 是 Entity 身份的一部分。

### 2.3 Entity 不会被 Filter 删除

Entity 表示“这个 occurrence 已经存在”。Filter 只改变某个目标 Port 是否包含它：

```text
page Entity 3：仍然存在

raw_page_port × page Entity 3      → PRESENT
selected_page_port × page Entity 3 → DROPPED
metadata_port × page Entity 3      → PRESENT
```

因此不同 branch 可以对同一个 Entity 有不同成员资格。`DROPPED` 属于 Item，不属于
Entity。

### 2.4 Grain 不是物理 batch

如果 `score` Call 在 page Domain 上执行，那么每个 page Entity 都对应一个 Grain：

```text
GrainRef(score_call, page_0)
GrainRef(score_call, page_1)
GrainRef(score_call, page_2)
```

Executor 可以把多个 Grain 打包进一个 actor RPC，但 batching 不改变 Grain 身份。
重试时 GrainRef 也不变，只提升 generation。

### 2.5 Expansion 为什么不能省略

没有 child Entity 可能表示四件不同的事：

```text
Expand 还没完成
Expand 成功，但结果为空
parent Item 被 DROPPED
上游失败，无法知道 cardinality
```

所以 Expansion 有独立终态：

```text
SUCCEEDED(children) | DROPPED | FAILED
```

`SUCCEEDED(())` 与 `DROPPED` 完全不同。前者是合法空 group，后者是该 Port 不包含
这个 parent occurrence。

---

## 3. RayModule 与 Worker 的批处理合同

`RayModule` 声明一个真正需要 Worker 的计算边界：

```python
self.module = (
    mg.RayModule(MyUDF, num_outputs=1)
    .pre_init(model_path)
    .ray_options(
        batch_size=16,
        replicas=2,
        recovery=mg.RecoveryPolicy.isolate_tail(infra_retries=1),
        num_gpus=1,
    )
)
```

### 3.1 UDF 类合同

有状态 UDF 使用类：

```python
class Model:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def run(self, images, thresholds):
        assert len(images) == len(thresholds)
        return [
            self.model(image, threshold)
            for image, threshold in zip(images, thresholds)
        ]
```

`pre_init()` 参数只在 actor 内构造 UDF 时使用。compiler 不实例化模型。

无状态函数可使用装饰器：

```python
@mg.function
def add(left, right):
    return [lhs + rhs for lhs, rhs in zip(left, right)]
```

无论类还是函数，输入和输出都是 batch columns。

### 3.2 多输出

```python
class Both:
    def run(self, values):
        return (
            [value * 2 for value in values],
            [value * 3 for value in values],
        )


self.both = mg.RayModule(Both, num_outputs=2)

def forward(self, values):
    doubled, tripled = self.both(values)
    return doubled, tripled
```

外层 tuple 按输出 Port 分列；每一列长度必须等于本次 batch 的 Grain 数。

### 3.3 positional 与 keyword 输入

```python
class Add:
    def run(self, values, *, increments):
        return [
            value + increment
            for value, increment in zip(values, increments)
        ]

def forward(self, values, increments):
    return self.add(values, increments=increments)
```

compiler 会生成 `CallInputLayout`。Worker 不反射 Pipeline，也不重新猜测 kwargs 顺序。

Logical IR 忠实保存 Python 调用形状，而不是提前伪装成 runtime slot：

```text
CallSpec
├── args: (CallInputSpec(values),)
└── kwargs: (("increments", CallInputSpec(increments)),)
```

`CallInputSpec` 只保存 `PortRef + InputMode`；关键字名字只存在 `CallSpec.kwargs` 的 key，
不在 value 中重复存一份。`CallSpec.ordered_inputs` 是按
`args + kwargs.values()` 计算出的只读 dense view，不是第二份状态。lowering 再生成：

```text
slots = (values, increments)
CallInputLayout(positional_count=1, keyword_names=("increments",))
```

MicrobatchEngine 用 dense slot 接收异步到达的 Item，Worker 根据唯一的 `CallInputLayout` 恢复
`run(values, increments=...)`。因此 logical 层易读，runtime 又不需要维护位置参数和
关键字参数两套队列。

### 3.4 Ray options 的两类含义

当前参考 Executor 自己消费：

- `batch_size`：每个 RPC 最多包含多少 Grain；
- `batch_scope`：`elastic` 或 `parent_bound`；
- `replicas`：该 Call 的持久 actor 数；
- `recovery`：该 Call 的强类型 `RecoveryPolicy`。

其他选项，例如 `num_cpus`、`num_gpus`、`max_restarts`，传给 Ray actor options。
compiler 会把两类选项归一化为一个按 `CallRef` 唯一索引的 `ActorPoolSpec`；runtime
不再反复解析 dict。`ActorPoolSpec` 没有独立身份：当前一个 Call 只对应一个同构 actor
pool，所以不存在冗余的 `PoolRef` 或 `CallRef -> PoolRef` 映射。
旧的 `max_retries` 同时混淆 UDF 与基础设施失败，已在编译期明确拒绝。

---

## 4. F.* 是结构关系，不是隐藏 Worker

当前公开结构 API 是：

```text
F.expand / F.expand_aligned
F.reduce / F.reduce_aligned
F.broadcast
F.filter
F.optional
```

除了 Expand 创建 child Entity，其他操作都不创建 Entity。所有 F.* 都不创建 actor、
RPC 或 Grain。

| 操作 | 新 Port | 新 Domain | 新 Entity | Worker |
| --- | --- | --- | --- | --- |
| RayModule Call | 是 | 否 | 否 | 是 |
| Filter | 是 | 否 | 否 | 否 |
| Broadcast | 是；同 Domain 时直接返回 source | 否 | 否 | 否 |
| Expand | 是 | 是 | 是 | 否；由上游 Call report 实现 |
| Reduce | 是 | 否；回到 parent | 否 | 否 |
| Optional | 否 | 否 | 否 | 否 |

### 4.1 Filter：改变 Port 成员资格

```python
selected = mg.F.filter(values, masks)
```

`values` 和 `masks` 必须属于同一 Domain。mask 为 `PRESENT(True)` 时复用 source
binding，为 `PRESENT(False)` 时目标 Item 变为 `DROPPED`。

```text
source 非 PRESENT → 原样传播 source outcome
source PRESENT     → 再由 mask 决定
```

若 Filter 输出继续作为下一个 Filter 的 mask，compiler 会把 control demand 反向传到
原始 bool producer。用户不需要手工声明 control manifest。

### 4.2 Optional：DROPPED 变成 MISSING

```python
optional_value = mg.F.optional(selected)
result = self.consume(optional_value)
```

Optional 只影响 Call 输入解释：

```text
PRESENT             → 正常值
OPTIONAL + DROPPED  → Worker 收到 mg.MISSING
FAILED/SUPPRESSED   → Call 不执行，输出 SUPPRESSED
```

UDF 应显式处理 `MISSING`：

```python
class FillDefault:
    def run(self, values):
        return [0 if value is mg.MISSING else value for value in values]
```

Call 可以全部由 optional 输入组成，因为 Entity 身份来自 Domain，而不是来自某个
“driving input”。

### 4.3 Expand：一对多创建 child Entity

```python
@mg.function
def split(values):
    return [[value, value + 1] for value in values]

def forward(self, values):
    children = mg.F.expand(split(values))
    return children
```

若 root 有两个 Entity，split 分别返回 2 个元素，那么 child Domain 有 4 个 Entity。

Expand 当前必须挂在 Call output 上，因为 Worker report 提供真实 rows 和 cardinality。
它不是对任意 Python list 做 driver 侧展开。

### 4.4 Reduce：沿 lineage 回收一级

```python
def forward(self, values):
    children = mg.F.expand(split(values))
    transformed = self.transform(children)
    return mg.F.reduce(transformed)
```

Reduce 输出位于 child Domain 的 parent Domain。默认 `members=transformed`：

```text
members PRESENT   → child 入组
members DROPPED   → child 排除
members 失败      → parent group SUPPRESSED
```

如果成员资格和实际值来自不同 Port，可以显式指定：

```python
grouped_scores = mg.F.reduce(scores, members=selected_pages)
```

只有 `selected_pages` 为 PRESENT 的 child 才会从 `scores` 取值。

### 4.5 Broadcast：把祖先值投影给后代 Entity

```python
leaf_thresholds = mg.F.broadcast(root_thresholds, like=leaves)
```

`root_thresholds` 必须来自 `leaves` Domain 的某个 ancestor Domain。`like` 只选择目标
Domain，不复制它的成员资格。

因此下面两件事不同：

```python
# 每个 leaf Entity 都获得 ancestor threshold。
thresholds = mg.F.broadcast(root_thresholds, like=selected_leaves)

# 若还要服从 selected_leaves 的成员资格，需要显式 Filter。
selected_thresholds = mg.F.filter(thresholds, leaf_mask)
```

### 4.6 aligned API

`expand_aligned(*ports)` 让同一个 Call 的多个 group outputs 共享 Expansion：

```python
left_group, right_group = self.produce(values)
left, right = mg.F.expand_aligned(left_group, right_group)
```

每个 parent 的左右 cardinality 必须完全相同，否则整个成功 report 被拒绝，不产生局部
children。

`reduce_aligned(*ports, members=...)` 则让多个 value Ports 使用同一成员集合回收。

---

## 5. 一个包含完整结构关系的例子

下面的 Pipeline 把 document 展成片段，把 root threshold 广播到片段，筛选片段，再按
document 聚回去。

```python
import rayorch.experimental.multigrain_v3_6 as mg


class Split:
    def run(self, documents):
        return [
            [document[index:index + 2] for index in range(0, len(document), 2)]
            for document in documents
        ]


class LongEnough:
    def run(self, chunks, thresholds):
        return [
            len(chunk) >= threshold
            for chunk, threshold in zip(chunks, thresholds)
        ]


class Summarize:
    def run(self, documents, chunk_groups):
        return [
            (document, tuple(chunks))
            for document, chunks in zip(documents, chunk_groups)
        ]


class DocumentPipeline(mg.Pipeline):
    def __init__(self):
        self.split = mg.RayModule(Split)
        self.long_enough = mg.RayModule(LongEnough)
        self.summarize = mg.RayModule(Summarize)

    def forward(self, documents, thresholds):
        chunks = mg.F.expand(self.split(documents))
        chunk_thresholds = mg.F.broadcast(thresholds, like=chunks)
        masks = self.long_enough(chunks, chunk_thresholds)
        selected = mg.F.filter(chunks, masks)
        chunk_groups = mg.F.reduce(selected)
        return self.summarize(documents, chunk_groups)


with mg.Executor(DocumentPipeline()) as executor:
    result = executor.run(
        ["abcde", "xy"],
        [2, 2],
    )

assert result.outputs == [
    ("abcde", ("ab", "cd")),
    ("xy", ("xy",)),
]
```

第一份文档的 `"e"` 对应 child Entity 仍存在，只是 `selected` Port 上的 Item 为
`DROPPED`。Reduce 的 members 默认是 `selected`，所以它不会进入最终 group。

这张图可以读成：

```mermaid
flowchart LR
    D["documents<br/>root Domain"] --> S["Split Call"]
    S --> E["Expand<br/>chunk Domain"]
    T["thresholds<br/>root Domain"] --> B["Broadcast"]
    E --> L["LongEnough Call"]
    B --> L
    E --> F["Filter source"]
    L --> F
    F --> R["Reduce to root"]
    D --> M["Summarize Call"]
    R --> M
```

---

## 6. 编译期：从 Python authoring 到 RuntimePlan

### 6.1 forward 是符号追踪

`Pipeline.compile()` 会检查 `forward` 参数，并为每个参数创建 root source Port。随后在
一个 trace context 中调用 `forward(*symbolic_ports)`。

例如：

```python
def forward(self, documents, thresholds):
    chunks = mg.F.expand(self.split(documents))
    ...
```

执行的不是 Split UDF，而是：

```text
self.split(documents Port)
→ 登记 CallSpec
→ 创建 CallOutputOrigin Port

F.expand(group Port)
→ 创建 child DomainSpec
→ 创建 ExpandOrigin Port
```

最终得到不可变 `LogicalProgram`：

```text
calls
ports
domains
source_ports
output_tree
```

### 6.2 PortOrigin 是 typed AST

每个 Port 有且只有一个 origin：

```text
SourceOrigin
CallOutputOrigin
ExpandOrigin
ReduceOrigin
BroadcastOrigin
FilterOrigin
```

可以把它理解成编译器 AST node。具体 Origin 只允许由
[`semantics.describe_origin()`](../rayorch/experimental/multigrain_v3_6/semantics.py)
解析。

`describe_origin()` 把不同 AST node 统一成 `PrimitiveSemantics`：

```text
kind
inputs + InputRole
control_demands
control_predecessors
rejects_control
producing_call/output_index/source_index
```

compiler 其他阶段不再 `isinstance(XxxOrigin)`；runtime 完全不认识 Origin。

这对组件的通信只有一个窄口：

```mermaid
flowchart LR
    Origin["PortOrigin<br/>typed AST node"] -->|"describe_origin(origin)"| Parser["semantics.py<br/>唯一 parser"]
    Parser --> Semantic["PrimitiveSemantics<br/>kind / inputs / roles / control"]
    Semantic --> Verify["Verifier"]
    Semantic --> Analyze["Analysis"]
    Semantic --> Lower["Lowering"]
```

因此新增 primitive 时，应扩展 parser 返回的统一描述；不应让 Verify、Analyze、Lower
分别反射具体 Origin 并各写一套 `isinstance`。

### 6.3 固定 compiler pipeline

[`compile_logical()`](../rayorch/experimental/multigrain_v3_6/compiler.py) 的顺序固定：

```text
verify LogicalProgram
→ analyze
→ optional canonicalize
→ lower RuntimePlan
→ verify RuntimePlan
```

这不是可插拔 PassManager。v3.6 当前不需要插件注册表、cost model 或通用 SSA。

```mermaid
sequenceDiagram
    participant P as Pipeline.forward
    participant T as Trace context
    participant S as semantics.py
    participant C as Compiler
    participant R as RuntimePlan

    P->>T: 用 symbolic Ports 调用 RayModule / F.*
    T-->>C: freeze LogicalProgram
    C->>S: describe_origin（逐 Port）
    S-->>C: PrimitiveSemantics
    C->>C: verify logical graph
    C->>C: analyze derived facts
    opt optimize=True
        C->>C: canonicalize transparent rewrites
    end
    C->>C: lower Effects / trigger indexes / layouts / pools-by-Call
    C->>C: verify RuntimePlan completeness
    C-->>R: immutable RuntimePlan + ProgramExplanation
```

#### Verify

验证内容包括：

- 引用是否存在；
- Domain parent 是否无环；
- 所有 Call inputs 是否属于 execution Domain；
- Port dependency 是否无环；
- Expand 是否只下降一级并来自 Call output；
- Group value/members 是否同 child Domain；
- Broadcast source 是否为目标 Domain 的祖先；
- Filter source/mask 是否同 Domain。

这些错误在 Ray actor 启动前暴露。

#### Analyze

[`analysis.py`](../rayorch/experimental/multigrain_v3_6/analysis.py) 计算可丢弃、可重算的
事实：

```text
consumers_by_port
outputs_by_call
expansion_sources_by_domain
control_ports fixed point
group_depth_by_port
```

例如 chained Filter 的 control demand 会一直反传到真正产生 bool 的 Port。

#### Canonicalize

当前唯一优化是折叠透明 Broadcast 链：

```text
broadcast(broadcast(root, child), grandchild)
→ broadcast(root, grandchild)
```

`optimize=False` 跳过 canonicalization，但仍走相同 verify/analyze/lower。这给优化提供
语义 correctness baseline。

#### Lower

lowering 把逻辑依赖变成 RuntimePlan：

```text
CallInputEffect
FilterEffect
ReduceEffect
BroadcastEffect
ExpandEffect
CallInputLayout / CallOutputLayout
ActorPoolSpec
```

每个 structural Port 只产生一个冻结的完整 Effect。`structural_effects_by_target[target_port]`
负责按目标定位；`item_effects_by_source[source_port]`、
`reduce_effects_by_child_domain[domain]` 和 `broadcast_effects_by_target_domain[domain]`
只是触发索引，并且都引用同一个 Effect 对象。MicrobatchEngine 收到 Effect 后已拥有执行所需的
全部 Port/Domain/control 信息，不需要先解析 target-only Route，再回查另一份 Rule。

`ActorPoolSpec` 由 `RuntimePlan.actor_pools_by_call[CallRef]` 唯一定位，不重复保存 Call 或 Pool
身份。只有 Call 创建 actor pool；任何结构 Port 都不会进入这张表。

RuntimePlan 已经包含 MicrobatchEngine 需要的完整物理事实，所以 MicrobatchEngine 不回读 LogicalProgram。

### 6.4 用 explain 调试编译结果

```python
compiled = DocumentPipeline().compile()
print(compiled.explain_text())
```

逐 Port 输出会说明：

```text
逻辑 kind
所属 Domain
逻辑输入 Ports
物理 lowering rule
是否 demand control
是否发生 canonical rewrite
```

遇到“为什么这个 mask 没有 control”或“为什么 Broadcast 指向另一个 source”时，先看
explain，不要直接进 MicrobatchEngine 猜。

---

## 7. 运行期：一个事实 publication 如何推动全图

Executor 的一次运行大致经历：

```mermaid
sequenceDiagram
    participant User
    participant E as Executor
    participant A as MicrobatchEngine
    participant D as DispatchState
    participant P as RecoveryPolicy
    participant W as Worker actor
    participant S as Block store

    User->>E: run(source columns)
    E->>S: put(source column blocks)
    S-->>E: BlockRefs
    E->>A: admit_sources(RowBindings)
    A->>A: publish source Items / advance
    A->>D: inputs_ready(Grain, batch_key)

    loop while MicrobatchEngine is incomplete
        E->>A: dispatch_priority / reserve_dispatch
        A->>D: reserve(Call, batch contract)
        D-->>A: DispatchBatch
        A-->>E: DispatchBatch
        E->>A: grain_plan(each Grain)
        A-->>E: InvocationPlans
        E->>W: execute(grain_plans, layouts)
        W->>S: get input rows / put output blocks
        W-->>E: WorkerResult
        alt GrainReport / GrainFailureReport
            E->>A: commit_report
            A->>D: validate generation / seal Grain
            A->>A: publish outputs / advance
        else DispatchFailure(UDF_ERROR)
            E->>P: decide_udf(completed retries, group size)
            P-->>E: RecoveryAction
            E->>A: apply_udf_recovery
            alt retry or split
                A->>D: requeue exact group(s)
            else failed singleton
                A->>A: publish FAILED semantic facts
            end
        else Ray actor/RPC exception
            E->>A: retry_infrastructure_dispatch
            A->>D: read infra failure counts
            D-->>A: per-Grain counts
            A->>P: allows_infrastructure_retry(counts)
            P-->>A: allow / deny
            opt allowed
                A->>D: generation++ / immediate requeue
                E->>E: replace untrusted actor
            end
        end
    end

    E->>A: read ordered terminal facts
    E->>S: get final payloads
    E->>A: release_values
    E-->>User: RunResult
```

组件两两通信可以压缩成下面五份合同：

| 组件对 | 请求 | 返回 | 状态由谁修改 |
| --- | --- | --- | --- |
| Executor ↔ MicrobatchEngine | admission、reserve、commit、materialize | DispatchBatch、GrainPlan、完成状态 | MicrobatchEngine 修改语义表 |
| MicrobatchEngine ↔ DispatchState | inputs ready/terminal、reserve、seal、recover | DispatchBatch、priority、校验结果 | DispatchState 独占 Grain/queue |
| Executor ↔ Worker | `GrainPlan + CallOutputLayout` | `WorkerResult` | Worker 只修改 actor 内 UDF 状态 |
| MicrobatchEngine ↔ transitions | 当前事实 tuple | 纯 action/outcome | transition 不修改任何状态 |
| Worker ↔ Block store | `RowBinding.get`、column `put` | 业务值、`BlockRef` | store 拥有 payload；MicrobatchEngine 只存引用 |

这张表是定位代码的最快入口：先判断当前问题发生在哪一对组件之间，再检查该边界上的
DTO 和 authority，而不是从 `Executor.run()` 一路单步进入所有模块。

### 7.1 MicrobatchEngine admission

`Executor.run()` 要求 source columns 数量等于 `forward` 参数数，而且所有列等长。

`microbatch_size` 把 source rows 切成多个独立 MicrobatchEngine：

```python
result = executor.run(
    values,
    microbatch_size=32,
    max_active_microbatches=4,
)
```

最多 4 个 MicrobatchEngine 同时活跃，但它们共享同一批持久 actors。一个 RPC 不混合多个 MicrobatchEngine
的 Grain。

```mermaid
sequenceDiagram
    participant E as Executor
    participant S as Block store
    participant A as MicrobatchEngine

    E->>E: slice row-aligned source columns
    E->>S: put(tuple(values))
    S-->>E: BlockRef
    E->>E: build RowBinding(block, row)
    E->>A: admit_sources(bindings, controls)
    A->>A: create root Entities
    A->>A: publish source Items
    E->>A: close_admission()
```

这里 Executor 决定 MicrobatchEngine 切片和物理 block；MicrobatchEngine 决定 source Entity/Item 身份。
双方都不读取 source 的业务含义。

### 7.2 FactEvent queue 与 advance

MicrobatchEngine 的 Item、Expansion、Entity 分别只通过 `_publish_item()`、`_publish_expansion()`、
`_publish_entity()` 写入 canonical tables。首次 publication 把现有不可变身份放入
同一个私有联合队列：

```python
_FactEvent = ItemRef | ExpansionRef | EntityRef
```

Event 只表示“这个事实刚刚出现”，不复制 outcome、binding、children 或 lineage。
完全相同的幂等 publication 不会再次入队。

`advance()` 穷尽匹配三种事实，并使用 RuntimePlan 对应的 Effect 索引：

```text
ItemRef  → item_effects_by_source
ExpansionRef → reduce_effects_by_child_domain
EntityRef → broadcast_effects_by_target_domain
```

Item Effect 再封闭匹配为 `CallInputEffect | FilterEffect | ReduceEffect |
BroadcastEffect`。它持续运行到 FactEvent queue 为空，即达到当前局部不动点。

```mermaid
flowchart LR
    Item["_publish_item"] --> Facts["FactEvent FIFO"]
    Expansion["_publish_expansion"] --> Facts
    Entity["_publish_entity"] --> Facts
    Facts --> Advance{"advance(): match fact"}
    Advance -->|"ItemRef"| ItemEffects["item_effects_by_source"]
    Advance -->|"ExpansionRef"| ShapeEffects["reduce_effects_by_child_domain"]
    Advance -->|"EntityRef"| EntityEffects["broadcast_effects_by_target_domain"]
    ItemEffects --> Transitions["Call / Filter / Group / Broadcast"]
    ShapeEffects --> Transitions
    EntityEffects --> Transitions
    Transitions --> NewFacts["new canonical facts"]
    NewFacts -->|"first publication only"| Facts
```

RuntimePlan 只告诉 MicrobatchEngine “这次 publication 应用哪些完整 Effects”；纯 transition 决定结果，
三个 typed publication 入口才拥有写事实的权限。有限 DAG、单调终态和每个事实只入队一次，
共同保证局部固定点终止。

### 7.3 Call inputs 如何成为 Grain

同一个 Call、同一个 Entity 的输入槽由 `PendingGrain` 收集。动态规则由
[`transitions.py`](../rayorch/experimental/multigrain_v3_6/transitions.py) 唯一定义：

```text
任意 FAILED/SUPPRESSED → outputs SUPPRESSED
否则仍有未决输入      → WAIT
否则 REQUIRED DROPPED  → outputs DROPPED
否则                   → Grain READY
```

所有输入对 Entity 身份是对称的，不存在 `driven_by`。

```mermaid
sequenceDiagram
    participant A as MicrobatchEngine
    participant T as call_transition
    participant D as DispatchState

    A->>A: collect PendingGrain slots
    A->>T: modes + current Item outcomes
    T-->>A: WAIT / READY / DROP_OUTPUTS / SUPPRESS_OUTPUTS
    alt READY
        A->>D: inputs_ready(GrainRef, batch_key)
        D->>D: create READY Grain + normal queue entry
    else DROP or SUPPRESS
        A->>D: inputs_terminal(GrainRef)
        D->>D: create SEALED Grain
        A->>A: publish terminal output Items
    else WAIT
        A->>A: keep PendingGrain only
    end
```

`call_transition` 不知道 queue，`DispatchState` 不知道 REQUIRED/OPTIONAL，MicrobatchEngine 负责把
纯语义动作接到唯一物理状态机上。

### 7.4 reserve、batch 与 GrainPlan

Executor 从 MicrobatchEngine 预留 READY Grains：

```text
READY + RESERVE → IN_FLIGHT
```

MicrobatchEngine 再把每个 Grain 投影为 Worker DTO：

```text
RowBinding
GroupInput
MissingInput
```

Worker 只看到这些 value takes、generation 和 output layouts，看不到 Program、PortOrigin
或 MicrobatchEngine tables。

```mermaid
sequenceDiagram
    participant E as Executor
    participant A as MicrobatchEngine
    participant D as DispatchState
    participant W as Worker

    E->>A: dispatch_priority(CallRef)
    A->>D: priority(CallRef)
    D-->>A: immediate / normal / tail / none
    E->>A: reserve_dispatch(CallRef, ActorPoolSpec batching)
    A->>D: reserve(...)
    D-->>A: DispatchBatch(exact Grains, udf_retries)
    loop each Grain
        E->>A: grain_plan(GrainRef)
        A-->>E: GrainPlan(generation, inputs)
    end
    E->>W: execute(grain_plans, output layouts)
```

`DispatchBatch` 是一次 RPC 的精确 lease。Executor 保存它以关联 ObjectRef 和失败，
但不能自行把 Grain 放回 ready queue。

### 7.5 Worker report 与原子提交

Worker 返回：

```text
GrainReport        成功，含全部 outputs
GrainFailureReport 某个 Grain 的业务失败
```

MicrobatchEngine 在写任何结果前先检查：

- Grain 是否 `IN_FLIGHT`；
- generation 是否仍然有效；
- output Ports 是否完整且无重复；
- scalar/expanded layout 是否匹配；
- control manifest 是否准确；
- aligned Expand cardinality 是否一致。

验证完成后才一次性封闭 Grain 并发布结果。multi-output 中任一列把某个位置返回为
`RecordFailure` 时，该 Grain 的所有输出都 `FAILED`，不会暴露半成功状态。

```mermaid
sequenceDiagram
    participant W as Worker
    participant E as Executor
    participant A as MicrobatchEngine
    participant D as DispatchState

    W-->>E: tuple[GrainReport | GrainFailureReport]
    E->>A: commit_report(report)
    A->>D: validate_in_flight(grain, generation)
    D-->>A: valid current attempt
    A->>A: preflight all outputs / expansions / controls
    A->>D: seal(grain, generation)
    A->>A: publish all terminal facts
    A->>A: advance to local fixed point
```

preflight 失败时不会先发布半个 multi-output report；generation 校验则阻止旧 attempt 的
迟到报告覆盖新状态。

### 7.6 完成、物化与释放

一个 MicrobatchEngine 完成需要：

```text
admission 已关闭
FactEvent queue/pending/ready 均为空
所有 Grain SEALED
所有输出 Port × 已存在 Entity 都有 Item 终态
```

`materialize_tree()`：

- PRESENT scalar：读取 `RowBinding`；
- PRESENT group：按 `GroupLayout` 恢复嵌套 list；
- 非 PRESENT：在结果中保留 `ItemOutcome`。

最终业务值复制到用户结果后，MicrobatchEngine 清空 ValueTable。Executor 随即冻结
`MicrobatchMetrics` 并释放 Engine；`RunResult` 不保留可变内部状态。需要逐 Grain 检查时，
应在 `DispatchState` 或 `MicrobatchEngine` 的所属单元测试中使用只读 snapshot，而不是
让公开结果承担调试后门。

---

## 8. 四套状态机

### 8.1 Entity

```text
不存在
  ├─ source admission → root Entity
  └─ Expand success   → child Entity
```

Entity 创建后不删除、不失败、不 dropped。只有 Expand 创建 child Entity。

### 8.2 Item

```text
UNRESOLVED
  ├─ PRESENT
  ├─ DROPPED
  ├─ FAILED
  └─ SUPPRESSED
```

| Outcome | 意义 |
| --- | --- |
| PRESENT | 此 Port 对此 Entity 有值 |
| DROPPED | 已确认此 Port 不包含该 Entity |
| FAILED | 直接生产者失败，或透明 view 保留该失败 |
| SUPPRESSED | 生产者因依赖失败而未执行 |

Item/Expansion/Entity 事实不可改写；完全相同的 publication 可以幂等重放，冲突 publication
直接失败。

### 8.3 Grain

```text
WAITING → READY → IN_FLIGHT → SEALED
    └───────────────────────→ SEALED
               IN_FLIGHT → READY  (retry)
```

`WAITING` 由 pending slots 表示，不是 GrainRecord phase。GrainRecord 只保存：

```text
phase
generation
infra_failures
```

Grain 的业务结果由输出 Items 推导，不再维护重复 GrainOutcome。
可变 GrainRecord 完全由 `DispatchState` 私有持有；MicrobatchEngine 与 Executor 只能读取冻结的
`GrainSnapshot`，不能通过 `RuntimeState` 获得第二条修改路径。

### 8.4 Expansion

```text
UNRESOLVED
  ├─ SUCCEEDED(children)
  ├─ DROPPED
  └─ FAILED
```

成功 Expansion 直接拥有有序 child Entity tuple，cardinality 由 `len(children)` 派生。
失败 Expansion 不虚构 children。

---

## 9. payload、control 与 lineage 为什么分开

### 9.1 MicrobatchEngine 不读取业务 payload

业务值存成粗粒度 block：

```text
BlockRef → 一个 Ray ObjectRef 或测试内存块
RowBinding(block, row) → 块中的一行
```

MicrobatchEngine 的 ValueTable 只保存 ItemRef 到 RowBinding/GroupBinding 的映射。只有 Worker 和最终
materializer 调用 store.get()。

这避免 driver 为调度而反序列化图片、页面或模型输出。

```mermaid
sequenceDiagram
    participant A as MicrobatchEngine
    participant E as Executor transport
    participant W as Worker
    participant S as Block store

    A-->>E: RowBinding / GroupInput（只含 RowBindings）
    E->>W: GrainPlan
    W->>S: get(RowBinding)
    S-->>W: business value
    W->>W: run batch UDF
    W->>S: put(output column)
    S-->>W: BlockRef
    W-->>E: OutputReport(RowBinding, optional control)
    E->>A: commit_report
    A->>A: store binding/control，不 get payload
```

Executor 在这对组件之间仅做 DTO transport；它不检查业务值，也不生成 Item 身份。

### 9.2 control manifest 是受控投影

Filter 必须知道 mask 是 True 还是 False，但 MicrobatchEngine 又不能读取业务 payload。因此 compiler
计算 `control_ports`，Worker 只为被 demand 的 bool Port 返回小型 control manifest。

```text
ValueTable        保存真实业务值的位置
ItemRecord.control 保存调度需要的 bool 副本
```

这不是重复业务值，而是明确的 data plane/control plane 分离。

```mermaid
flowchart LR
    Worker["Worker output"] --> Binding["RowBinding<br/>business data location"]
    Worker --> Control["bool control manifest<br/>only when demanded"]
    Binding --> Values["RuntimeState.values"]
    Control --> Item["ItemRecord.control"]
    Lineage["Entity / Expansion / lineage"] --> MicrobatchEngine["MicrobatchEngine transitions"]
    Item --> MicrobatchEngine
    Values -. "Worker/materializer only" .-> Store["Block store payload"]
```

三者的生命周期也不同：payload 可在 materialize 后释放，control 是小型语义投影，
lineage/terminal facts 则保留用于审计。

### 9.3 GroupBinding 与 CSR Expansion

Reduce 不把大 group payload 复制到 MicrobatchEngine，而是保存：

```text
GroupBinding(
    GroupLayout(offsets_by_level),
    flat_items,
)
```

`flat_items` 指向有序叶子 Item；`offsets_by_level` 表示嵌套边界。Worker 在真正调用 UDF
前才按 offsets 恢复 Python nested lists。

---

## 10. 失败分类与恢复策略

### 10.1 业务记录失败

UDF 可以只让 batch 中某些 Grain 失败：

```python
from rayorch.experimental.multigrain_v3_6.protocol import RecordFailure


class Parse:
    def run(self, values):
        return [
            RecordFailure("invalid input") if value is None else parse(value)
            for value in values
        ]
```

Worker 把对应 Grain 转成 `GrainFailureReport`。其他 batch positions 仍然成功。

### 10.2 四类 failure 不共用一条恢复路径

| failure | 表示什么 | 恢复合同 |
| --- | --- | --- |
| `RecordFailure` | UDF 已定位到一个 Grain 的业务失败 | 直接提交该 Grain 为 FAILED，不重试 |
| `CONTRACT_ERROR` | UDF 返回值违反编译后的 Worker ABI | 立即抛 `ExecutionError`，永不重试 |
| `UDF_ERROR` | 一次不透明 UDF dispatch 抛异常 | 按该 Call 的 `RecoveryPolicy` 决策 |
| `INFRA_FAILURE` | actor/RPC/transport 失效，结果不可信 | 换新 actor，按独立 infra budget 立即重放 |

```mermaid
flowchart TD
    Failure{"失败在哪个边界被识别？"}
    Failure -->|"UDF 返回 RecordFailure(position)"| Record["GrainFailureReport"]
    Failure -->|"Worker normalization/ABI"| Contract["CONTRACT_ERROR"]
    Failure -->|"UDF 整次抛异常"| Udf["UDF_ERROR"]
    Failure -->|"ray.get / actor / transport"| Infra["INFRA_FAILURE"]

    Record --> Commit["MicrobatchEngine.commit_failure<br/>只影响对应 Grain"]
    Contract --> Fast["ExecutionError<br/>fail-fast"]
    Udf --> Policy["RecoveryPolicy.decide_udf"]
    Policy --> Abort["ABORT"]
    Policy --> Retry["RETRY_IMMEDIATE / RETRY_TAIL"]
    Policy --> Split["SPLIT_TAIL"]
    Policy --> Single["FAIL_SINGLETON"]
    Retry --> Dispatch["DispatchState requeue exact group"]
    Split --> Dispatch
    Single --> Commit
    Infra --> Budget{"infra budget remains?"}
    Budget -->|"yes"| Replace["generation++ / fresh actor / immediate queue"]
    Budget -->|"no"| Fast
```

Worker 只负责把异常快照为可序列化的 `DispatchFailure`。它不知道重试策略；
Executor 只把 failure 分类交给纯 `RecoveryPolicy`；MicrobatchEngine 内唯一的 `DispatchState` 拥有
READY、generation 和 recovery queues。这样不会在 Executor 或 Engine 中再藏一份
可运行 Grain。

合同错误的消息包含 Call/UDF、精确 Grain group、generation、ABI 说明和 Worker
traceback。UDF 自己恰好抛出名为 `WorkerContractError` 的异常仍属于
`UDF_ERROR`；分类依据异常发生的边界，而不是异常类名。

### 10.3 UDF recovery policy

```python
mg.RecoveryPolicy.abort(infra_retries=1)          # 默认
mg.RecoveryPolicy.retry_batch(attempts=2)
mg.RecoveryPolicy.retry_tail(attempts=2)
mg.RecoveryPolicy.isolate_tail(infra_retries=1)
```

| policy | UDF 失败后的动作 |
| --- | --- |
| `abort` | 立即终止本次 run |
| `retry_batch` | 原精确 Grain group 进入 immediate queue |
| `retry_tail` | 原精确 Grain group 进入 tail queue，让正常工作先行 |
| `isolate_tail` | 整组 tail retry 一次；仍失败则有界二分，singleton 才提交 FAILED |

`attempts` 是额外 UDF replay 次数，不是总执行次数。`isolate_tail` 没有
`max_depth`、`min_batch` 等第二组旋钮：每次非 singleton 只二分一次，所以有限性由
group cardinality 直接证明。immediate、normal、tail 的优先级也是 MicrobatchEngine 的显式
调度状态，不是 Executor 临时拼接的列表。

`isolate_tail` 的组件协作如下：

```mermaid
sequenceDiagram
    participant E as Executor
    participant P as RecoveryPolicy
    participant A as MicrobatchEngine
    participant D as DispatchState

    E->>P: decide_udf(completed=0, grains=N)
    P-->>E: RETRY_TAIL
    E->>A: apply_udf_recovery(batch, RETRY_TAIL)
    A->>D: recover_udf
    D->>D: exact group → tail queue, udf_retries=1
    Note over E,D: 若整组再次失败
    E->>P: decide_udf(completed=1, grains=N)
    alt N > 1
        P-->>E: SPLIT_TAIL
        E->>A: apply_udf_recovery
        A->>D: ordered halves → tail queue
    else N = 1
        P-->>E: FAIL_SINGLETON
        E->>A: apply_udf_recovery
        A->>A: publish FAILED Item / Expansion facts
    end
```

### 10.4 RPC/actor 异常

如果 Ray RPC 抛异常，Executor 根据 `RecoveryPolicy.infra_retries`：

```text
IN_FLIGHT → READY
generation += 1
infra_failures += 1
```

旧 generation 的迟到报告不能覆盖新 attempt。actor crash 时 Executor 丢弃不可信 handle，
创建 fresh actor，并把同一精确 group 放进 immediate queue；若失败发生在 UDF recovery
中，`DispatchBatch.udf_retries` 会随精确 group 保留。超过 infra budget 后抛
`ExecutionError`，不会伪造一个业务 FAILED Item。

### 10.5 failure propagation

普通 Call 的输入优先级是：

```text
FAILED/SUPPRESSED > unresolved > REQUIRED DROPPED > READY
```

所以 failure 与 required drop 同时出现时，输出是 `SUPPRESSED`，而不是依赖输入顺序。

Filter 和 Reduce 是带角色的 gate：

- Filter 先看 source，再看 mask；
- Reduce 先看 Expansion，再看 members，只读取 survivors 的 values。

这些不对称写在纯 transition 中，不允许由某个默认输入槽暗中决定。

---

## 11. 源码地图与推荐阅读顺序

建议按以下顺序阅读，而不是直接从 MicrobatchEngine 开始：

1. [`model.py`](../rayorch/experimental/multigrain_v3_6/model.py)：核心 Ref 和终态枚举。
2. [`logical.py`](../rayorch/experimental/multigrain_v3_6/logical.py)：typed logical IR。
3. [`functional.py`](../rayorch/experimental/multigrain_v3_6/functional.py)：F.* 公开表面。
4. [`semantics.py`](../rayorch/experimental/multigrain_v3_6/semantics.py)：唯一 Origin parser。
5. [`transitions.py`](../rayorch/experimental/multigrain_v3_6/transitions.py)：动态状态代数。
6. [`recovery.py`](../rayorch/experimental/multigrain_v3_6/recovery.py)：Ray-free 纯恢复策略。
7. [`plan.py`](../rayorch/experimental/multigrain_v3_6/plan.py)：完整不可变 Runtime Effects 与触发索引。
8. [`compiler.py`](../rayorch/experimental/multigrain_v3_6/compiler.py)：固定 passes。
9. [`runtime/state.py`](../rayorch/experimental/multigrain_v3_6/runtime/state.py)：被动 tables。
10. [`runtime/dispatch.py`](../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py)：唯一 Grain/queue 状态机。
11. [`runtime/engine.py`](../rayorch/experimental/multigrain_v3_6/runtime/engine.py)：Item/Expansion/Entity 传播层。
12. [`protocol.py`](../rayorch/experimental/multigrain_v3_6/protocol.py)：稳定 DTO/Worker ABI。
13. [`worker.py`](../rayorch/experimental/multigrain_v3_6/worker.py)：值恢复和报告生成。
14. [`executor.py`](../rayorch/experimental/multigrain_v3_6/executor.py)：Ray actor 与多 microbatch 调度。
15. [`materialize.py`](../rayorch/experimental/multigrain_v3_6/materialize.py)：最终输出恢复。

源码层的主要 knowledge/call 关系是：

```mermaid
flowchart LR
    subgraph Frontend["Authoring / compiler"]
        API["api / functional"] --> Logical["logical.py"]
        Logical --> Semantics["semantics.py"]
        Semantics --> Analysis["analysis.py"]
        Analysis --> Compiler["compiler.py"]
        Compiler --> Plan["plan.py<br/>RuntimePlan"]
    end

    subgraph Pure["Pure contracts"]
        Transitions["transitions.py"]
        Recovery["recovery.py"]
        Protocol["protocol.py<br/>DTOs"]
    end

    subgraph Runtime["Driver runtime"]
        Executor["executor.py"] -->|"MicrobatchEngine façade"| MicrobatchEngine["runtime/engine.py"]
        MicrobatchEngine -->|"Grain authority"| Dispatch["runtime/dispatch.py"]
        Executor --> Materialize["materialize.py"]
        Materialize -->|"read terminal facts"| MicrobatchEngine
    end

    subgraph ActorSide["Actor / values"]
        Worker["worker.py"]
        Store["block store"]
        Worker <-->|"get / put"| Store
    end

    Plan --> Executor
    Plan --> MicrobatchEngine
    MicrobatchEngine -->|"facts → decision"| Transitions
    Dispatch -->|"phase transition"| Transitions
    Executor -->|"UDF recovery"| Recovery
    MicrobatchEngine -->|"infra budget"| Recovery
    Executor --> Protocol
    MicrobatchEngine --> Protocol
    Worker --> Protocol
    Executor <-->|"GrainPlan / WorkerResult"| Worker
    Materialize -->|"get payload"| Store
```

图表示主要调用与权责关系，而不是禁止所有反向依赖的形式规则。真正需要阻止的是跨层
写状态、同一事实出现两个 authority，或为了依赖图好看而增加无语义适配层。

---

## 12. 如何维护而不引入飞线

### 12.1 增加一个新 primitive

新增 primitive 前先回答：

```text
它创建哪种身份？
它消费哪些 Port，角色分别是什么？
它是否改变 Domain？
它产生哪些 Item/Expansion 终态？
control demand 如何传播？
它是否真的需要 Worker？
为什么现有 primitive 无法表达？
```

若确实需要，按固定顺序修改：

1. 在 `logical.py` 增加 `XxxOrigin` 并加入 `PortOrigin` 联合。
2. 在 `api.py/functional.py` 构造新 Port/Domain 关系。
3. 在 `semantics.describe_origin()` 增加穷尽 case，声明 InputRole/control。
4. 在 compiler verifier 增加局部合法性 case。
5. 在 `plan.py` 定义一个包含执行所需全部静态信息的最小 `XxxEffect`。
6. 在 lowering 中只构造一次 Effect，并让所有触发索引引用它；即使是 pass 也写出 case。
7. 若有新的动态组合，在 `transitions.py` 增加纯函数。
8. MicrobatchEngine 只应用 transition 和发布事实，不重复写状态优先级。
9. 增加笛卡尔积测试、compiler boundary 测试和 explain 断言。
10. optimized/unoptimized 必须结果等价。

```mermaid
flowchart LR
    Author["api / functional<br/>构造 XxxOrigin"] --> Parser["semantics.describe_origin<br/>统一角色描述"]
    Parser --> Verify["verify local legality"]
    Parser --> Facts["analyze derived facts"]
    Facts --> Lower["lower one immutable XxxEffect"]
    Lower --> Plan["verify RuntimePlan"]
    Plan --> Transition["pure transition<br/>若有新动态组合"]
    Transition --> Engine["MicrobatchEngine applies decision<br/>and publishes facts"]
    Engine --> Tests["Cartesian + compiler + integration"]
```

如果一项修改需要绕过这条链直接让 Executor/Worker 读取 Origin，通常说明缺少静态 Effect、
DTO 或纯 transition，而不是需要一条快捷飞线。

### 12.2 修改状态语义

不要先改 MicrobatchEngine 的某个 `if`。正确顺序是：

```text
更新 v3.6 语义合同
→ 修改 transitions.py
→ 修改笛卡尔积期望
→ 让 MicrobatchEngine 适配 transition result
→ 跑完整回归
```

如果一个 outcome 规则无法在纯 transition 中表达，通常说明输入事实或 primitive role
还没有建模清楚。

### 12.3 修改 Worker ABI

必须同步检查：

```text
protocol DTO
compiler CallInputLayout/CallOutputLayout
MicrobatchEngine grain_plan/commit validation
Worker restore/normalize/report
Ray actor boundary integration test
```

不要把 LogicalProgram 或 MicrobatchEngine 对象直接传进 Worker。

### 12.4 增加编译优化

优化必须满足：

- `optimize=False` 仍是完整可运行 baseline；
- 不改变 Call 是否执行，除非已经定义副作用合同；
- 不改变 ItemOutcome、control、membership 或 failure provenance；
- `explain` 记录 rewrite；
- optimized/unoptimized 有等价测试；
- 有 profile 证明值得增加复杂度。

当前 Filter 本身没有 actor/RPC，单纯融合相邻 Filter 通常没有明显性能收益。

### 12.5 不允许出现的飞线

维护时可以用这份硬性检查表：

- compiler 除 `describe_origin()` 外不识别具体 `XxxOrigin`；
- runtime 不导入 LogicalProgram/PortOrigin；
- 任一输入 Port 不得成为 Grain/Entity 隐式 driver；
- 只有 Expand 创建 child Entity；
- F.* 结构 primitive 不得偷偷创建 actor/RPC/Grain；
- Item、Expansion、Entity 只能通过各自唯一 publication 入口进入 canonical tables；
- MicrobatchEngine 不读取业务 payload；
- Executor 不重新实现 Filter/Reduce 等 primitive 语义；
- Worker 不读取 Program/MicrobatchEngine，不生成逻辑身份；
- derived index/queue 不得成为第二套语义权威表。

---

## 13. 测试与回归入口

### 13.1 Ray-free 单元与 MinerU 静态门禁

```bash
pytest -q \
  test/experimental/multigrain_v3_6/unit \
  test/experimental/multigrain_v3_6/benchmark/test_mineru_pipeline.py
```

这里覆盖 compiler 边界、control fixed point、状态笛卡尔积、nested group、recovery
policy reducer、MicrobatchEngine recovery queues、retry fencing、aligned atomicity 和 MinerU
Pipeline 静态结构。

### 13.2 真实 Ray integration

```bash
pytest -q test/experimental/multigrain_v3_6/integration/test_executor.py
```

覆盖空输入、重复 run 指标、持久 actor、多 microbatch、同步 actor 构造清理、keyword
ABI、actor crash replacement、infra exhaustion、RecordFailure、UDF immediate/tail retry、
poison isolation、合同 fail-fast 和 multi-output 原子性。某些 Ray 版本若误判 uv runtime
environment，可临时设置：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
pytest -q test/experimental/multigrain_v3_6/integration/test_executor.py
```

测试后应确认 fixture 已 `ray.shutdown()`，并用 `ray status` 验证没有残留实例。

### 13.3 静态类型

```bash
pyright \
  rayorch/experimental/multigrain_v3_6/semantics.py \
  rayorch/experimental/multigrain_v3_6/compiler.py \
  rayorch/experimental/multigrain_v3_6/transitions.py \
  rayorch/experimental/multigrain_v3_6/runtime
```

新增 `PortOrigin`、`InputRole` 或 Effect 联合成员后，`assert_never` 应帮助暴露未处理 case。

### 13.4 真实性能证据

当前实现已经完成 MinerU 368-PDF、Docling 48-PDF、Caption 256-video 和双 Domain
Multimodal 256-video 的两轮交替顺序 paired trial。相对基线的平均变化依次为 -0.551%、
+1.260%、+1.990% 和 +0.492%，身份与结构合同全部通过。完整均值、样本方差、RPC/batch
诊断和业务非确定性分析见
[`2026-08-08_release_regression.md`](experiments/multigrain_v3_6/2026-08-08_release_regression.md)。

Caption 还验证了一个重要反例：microbatch 4、active 2 会让 V3.5/V3.6 都因 admission
window 太小而增加约 50% RPC。冻结性能窗口是 32×4，benchmark 默认值和 artifact 已显式
固定这两个字段。维护者应把它们理解为容量/吞吐配置，而不是与性能无关的安全限制。

后续性能修改仍不能只看 unit test；至少应比较 correctness、wall time、RPC 数、平均
batch、driver RSS 和 actor 数。

---

## 14. 常见问题的排查顺序

### 14.1 编译期报 Domain mismatch

先问两个 Port 是否真的同粒度：

- ancestor 值给 descendant：使用 Broadcast；
- child values 回 parent：使用 Reduce；
- 两个无共同 lineage 的 Domain：当前不支持隐式 join。

不要通过选择某个 driving input 强行对齐。

### 14.2 Filter 报 control manifest 问题

按顺序检查：

1. `compiled.analysis.control_ports` 是否包含 mask producer；
2. `compiled.explain_text()` 是否显示 `control`；
3. chained Filter 是否通过 `control_predecessors` 回传；
4. Worker output 是否真的是逐 Grain bool；
5. expanded bool rows 是否与 rows 等长。

### 14.3 MicrobatchEngine deadlocked

查看：

```python
engine.progress_summary()
engine.grain_snapshots()
engine.expansion_count
```

常见原因是：

- 某个依赖 Port 没有进入 Effect 触发索引；
- Expansion 已终态但 Group 的 member/value Item 没发布；
- Worker report layout 不完整；
- 新 primitive 在 lowering 中被遗漏而不是显式 pass；
- output Port 对某些已存在 Entity 没有终态。

### 14.4 CommitError

CommitError 通常不是普通业务失败，而是合同冲突：

- stale generation；
- output ports 不完整或重复；
- scalar/expanded report 与 plan 不一致；
- aligned cardinality 不一致；
- control demand 不一致；
- Item/Expansion/Entity 被冲突发布。

先检查 WorkerReport 和 RuntimePlan，不要尝试覆盖已有状态。

### 14.5 输出中出现 ItemOutcome

这是预期的显式失败/成员语义：

```python
ItemOutcome.DROPPED
ItemOutcome.FAILED
ItemOutcome.SUPPRESSED
```

materializer 不会把它们静默转换为 `None`，因为三者含义不同。

---

## 15. 最后用五句话复述整个框架

1. `Pipeline.forward()` 用符号 Port 写图，RayModule 表示计算，F.* 表示结构关系。
2. compiler 把 typed PortOrigin AST 验证、分析并 lower 成不含 Origin 的 RuntimePlan。
3. Entity 定义 occurrence，Item 定义 Port 上的终态，Grain 定义 Call 的执行，Expansion 定义
   Expand children。
4. MicrobatchEngine 独占语义表，DispatchState 独占 Grain/queues；Executor 管 Ray，Worker 只管值。
5. 身份、业务值、control、物理 batching 和 failure propagation 相互分离，因此可以独立
   优化而不改变逻辑语义。

当你能根据一个新需求明确指出它属于这五层中的哪一层，并说明它不应该进入哪些层，
就已经具备维护 v3.6 的核心能力。
