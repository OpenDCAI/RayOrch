# Multigrain V2.2 权威架构合同

状态：`rayorch.experimental.multigrain_v2` 的唯一规范性架构文档。

本文取代 V2、V2.1 的架构草案。实现、测试和论文不得从旧文档恢复已删除的
API 或机制。配套文档：

- `multigrain_v2_2_implementation_plan.md`
- `multigrain_v2_2_paper_and_experiments.md`

本文使用 MUST、MUST NOT、SHOULD 表示强制合同、禁止行为和推荐行为。

## 1. 系统命题

Multigrain 是面向 applied-ML dataflow 的 Ray 原生执行引擎。它将五类动态基数原语：

```text
Map / Filter / Expand / Reduce / Relate
```

编译为带稳定 identity、类型化 direct provenance 和语义恢复闭包的 WorkUnits。

系统只维护一条语义链：

```text
Pipeline authoring
-> immutable CompiledGraph
-> per-microbatch RunArena
-> PrimitiveKernel planning/invoke/bundle materialization
-> Ray NodeRuntime transport
-> atomic OutputBundle publication
```

核心边界：

- driver 是 identity、typed provenance、WorkUnit 和 publication 的唯一 authority；
- actors 只计算 value columns 与最小 raw evidence；
- 大型业务 payload 通过 Ray ObjectRefs P2P 路由，不经过 driver；
- RPC batch、actor、replica、attempt 和完成顺序不进入逻辑 identity；
- publication 前允许 WorkUnit 范围恢复；
- publication 后 intermediate ObjectRef 丢失只允许整 microbatch 重启；
- 已交付结果不依赖存活的 Executor，但仍依赖调用者持有的 Ray runtime。

## 2. 用户只需要理解的概念

顶层 public API 只包含：

```text
Pipeline
Map / Filter / Expand / Reduce / Relate
Select
Port / keyed / provided_keys / by_role
RelationResult / UdfContext
Executor / RunStream
RunResult / ResultPort
FailureRecord / RunMetrics / ExecutionErrorSummary
BadRecordError
CompileError / ExecutionError / MultigrainError
```

以下类型全部是 internal：

```text
CompiledGraph / RunArena / PortData / RowRef / IdentityKey
WorkUnitRef / InvocationPlan / PrimitiveKernel
ValueShard / RpcShardTake / RpcManifest / OutputBundle
TerminalFailure / RpcAttemptToken / MicrobatchAttempt
```

## 3. 完整用户示例

```python
import ray
import rayorch.experimental.multigrain_v2 as mg


class ReadPages:
    def __init__(self, dpi: int) -> None:
        self.dpi = dpi

    def run(self, paths: list[str]) -> list[list[Page]]:
        return [decode_pages(path, dpi=self.dpi) for path in paths]


class Ocr:
    def __init__(self, model: str) -> None:
        self.model = load_model(model)

    def run(
        self,
        pages: list[Page],
    ) -> tuple[list[str], list[float]]:
        return run_ocr(self.model, pages)


class KeepHighScore:
    def run(self, scores: list[float]) -> list[bool]:
        return [score >= 0.8 for score in scores]


class Assemble:
    def run(
        self,
        paths: list[str],
        grouped_texts: list[list[str]],
    ) -> list[Document]:
        return assemble_documents(paths, grouped_texts)


class WriteDocuments:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def run(
        self,
        documents: list[Document],
        *,
        context: mg.UdfContext,
    ) -> None:
        write_idempotently(
            self.output_dir,
            context.item_keys,
            documents,
        )


class DocumentPipeline(mg.Pipeline):
    def __init__(self, model: str, output_dir: str) -> None:
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
                num_cpus=1,
                num_gpus=1,
                max_retries=1,
                timeout_s=600,
            )
        )
        self.make_mask = mg.Map(KeepHighScore)
        self.keep = mg.Filter.by_mask()
        self.assemble = mg.Reduce(Assemble)
        self.write = (
            mg.Map(WriteDocuments)
            .pre_init(output_dir=output_dir)
            .ray_options(replicas=2, batch_size=8, max_retries=1)
        )

    def forward(self, paths: mg.Port[str]) -> None:
        pages = self.read_pages(paths)
        texts, scores = self.ocr(pages)
        keep_mask = self.make_mask(scores)
        kept_texts = self.keep(texts, mask=keep_mask)
        documents = self.assemble(paths, kept_texts)
        self.write(documents)


ray.init(address="auto")

pipeline = DocumentPipeline(model="/models/ocr", output_dir="/output")
with mg.Executor(pipeline, max_inflight=4) as executor:
    result = executor.run(["a.pdf", "b.pdf"])
    result.raise_for_status()

ray.shutdown()
```

示例中的 `Filter` 仅用于说明接线形态；当 mask 由另一个 UDF 产生时，规范写法是
`Filter.by_mask()`。`Select` 是打分/标注后立即过滤的单node fused facade，编译为
`FilterSpec(mode="select")`；UDF、mask生成与subset在同一actor RPC内完成。

## 4. Module authoring 合同

### 4.1 Module 是 node recipe

`PrimitiveModule` 是被 compiler 读取的可配置 recipe：

- 不创建 actor；
- 不持有运行时状态；
- 不直接执行 UDF；
- 被调用的instance必须通过Pipeline直接属性或递归tuple slot获得唯一attribute path；
- 一个 instance 在一张图中 MUST 形成且只形成一个 callsite；
- 两个 nodes 即使使用相同 UDF class，也 MUST 声明两个 module instances。

```python
self.left = mg.Map(Encoder).pre_init(model=MODEL)
self.right = mg.Map(Encoder).pre_init(model=MODEL)
```

共享 module instance 不是共享模型驻留机制。未来若需要共享只读模型，必须增加显式
residency abstraction，不能复用 module object。

### 4.2 分段配置

唯一配置链为：

```text
bind UDF class
-> pre_init UDF constructor
-> ray_options node execution
-> call ports
```

```python
self.ocr = (
    mg.Map(Ocr)
    .pre_init(model=MODEL, decoder="beam")
    .ray_options(
        replicas=4,
        batch_size=16,
        num_cpus=1,
        num_gpus=1,
        resources={"accelerator_type:A100": 0.001},
        runtime_env={"env_vars": {"MODEL_PROFILE": "ocr"}},
        max_retries=1,
        timeout_s=600,
    )
)
```

`pre_init()` 的参数全部传给 UDF constructor。`ray_options()` 只允许固定 whitelist：

```python
@dataclass(frozen=True, slots=True)
class ModuleConfig:
    replicas: int = 1
    batch_size: int = 64
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: tuple[tuple[str, float], ...] = ()
    runtime_env: Mapping[str, object] | None = None
    max_retries: int = 1
    timeout_s: float | None = None
```

`runtime_env`按用户填写内容原样保留，不做merge、排序或自定义结构转换。compiler只通过
cloudpickle round-trip生成与用户对象断开引用的snapshot并验证可传输性；后续用户修改
原dict/list不影响CompiledGraph，internal runtime也MUST NOT修改该snapshot。

`max_restarts`、`max_task_retries`、`scheduling_strategy`、actor name 和 placement group
不允许用户透传。它们由 Executor 保留。

首版每个配置方法最多调用一次；第二次调用立即 `ValueError`。链式方法原地返回同一
recipe，首次成功 compile 后冻结。compile 失败不冻结；重复 compile 返回同一语义快照。
配置对象不支持并发修改。

### 4.3 typing 边界

实现 SHOULD 使用 ParamSpec 为 `.pre_init()` 和 UDF `run()` 参数名提供 best-effort
IDE 补全：

```python
IP = ParamSpec("IP")
RP = ParamSpec("RP")


class UdfProtocol(Protocol[IP, RP]):
    def __init__(self, *args: IP.args, **kwargs: IP.kwargs) -> None: ...
    def run(self, *args: RP.args, **kwargs: RP.kwargs) -> object: ...
```

Python typing 无法稳定地把任意 `run(T) -> tuple[A, B]` 映射为
`Port[T] -> tuple[Port[A], Port[B]]`。因此：

- IDE typing 是 best effort；
- compiler 的 `inspect.Signature.bind()` 是 input binding authority；
- compiler 的 normalized return annotation 是 output arity authority；
- runtime shape validation 是最终 safety gate。

不引入 mypy/Pyright plugin。

### 4.4 output arity

compiler 对 `inspect.unwrap(udf_cls.run)` 调用
`typing.get_type_hints(..., include_extras=True)`，再应用唯一规则：

```text
annotation absent or Any       -> 1
None or NoneType               -> 0
tuple[T1, ..., TN] fixed tuple -> N
RelationResult[Outputs]        -> recursively infer Outputs
all other annotations          -> 1
```

显式 annotation 无法解析、bare tuple、`tuple[T, ...]` 或空 tuple MUST
`CompileError`。NamedTuple、dataclass 和普通 tuple-valued row 类型均视为一个 output；
多 port 返回必须使用顶层固定 tuple annotation。

runtime MUST 验证：

```text
arity 0 -> return is None
arity 1 -> return is one list column
arity N -> return is tuple of N list columns
```

Filter predicate/by-mask的output arity由target ports数量决定。Select先按上述规则推导
UDF returns：`list[bool]`表示零annotations；固定tuple第一项为mask，其余项数为
annotation arity；最终output arity为input targets数加annotation arity。

一个primitive invocation的所有非零outputs MUST共享同一semantic cardinality/identity
layout；首版禁止单node heterogeneous output cardinality。跨粒度结果用显式分支：

```python
metadata, page_groups = self.parse(documents)  # Map: both one row/document
pages = self.expand_pages(page_groups)         # Expand: many rows/document
```

其中`page_groups`的每个`list[Page]`只是一个Map row value。Map和Expand可以由未来physical
optimizer融合，但logical CompiledGraph、identity和PortRelation不改变。

zero-output支持矩阵：

```text
Map       allowed
Reduce    allowed
KeyRelate allowed
CustomRelate allowed
Filter    forbidden
Expand    forbidden
```

Filter/Expand UDF annotation为None时 compiler MUST 拒绝。

### 4.5 可选 UDF execution context

zero-output writer需要稳定幂等键，但 internal WorkUnitRef 不进入普通public API。UDF 可以
声明一个保留的 keyword-only 参数：

```python
@dataclass(frozen=True, slots=True)
class UdfContext:
    invocation_key: str
    item_keys: tuple[str, ...]
```

```python
class WriteDocuments:
    def run(
        self,
        documents: list[Document],
        *,
        context: mg.UdfContext,
    ) -> None:
        write_once(context.item_keys, documents)
```

compiler 只在参数名为`context`、keyword-only且annotation恰为`mg.UdfContext`时将其识别为
reserved context；它不形成input port或Relate role。actor为每个RPC注入context。

两个字段都是opaque strings，用户只能比较和持久化，MUST NOT解析其格式。public ABI只
冻结semantic stability，不冻结hash、编码或字符串格式：

```text
invocation_key scope -> (BatchId, NodeId)

item semantic coordinate:
Map / Filter / Expand -> primary/parent IdentityKey
Reduce                -> anchor IdentityKey
Key Relate            -> ordered role-parent IdentityKey tuple
Custom Relate         -> 不可事前确定，因此item_keys=()
```

同一active microbatch内，相同scope/coordinate跨RPC、retry、split和whole restart必须
产生相同key；同一invocation内不同coordinates必须产生不同item keys。
MicrobatchAttempt、RPC、actor和WorkRange不进入scope。实现可以使用内部`H()`，但其bytes
不是public durable ABI。

alignment规则：

```text
Map / Filter / Expand -> UDF outer row/parent dimension
Reduce                -> anchors
Key Relate            -> plan-known flat parent tuples
Custom Relate         -> empty，因为parent pairs只能由UDF返回后确定
```

普通Map/Reduce/Key Relate writer必须按item_keys幂等写入；一个invocation可能拆成多个RPC，
不得只使用invocation_key作为每个RPC的写入key。Custom zero-output Relate使用
invocation_key幂等提交整个WholeInvocation。外部重新submit产生新BatchId，keys不保证与
前一次run相同；跨engine version也不保证字符串相同。系统仍不承诺external exactly-once。

首版terminal writer采用明确的at-least-once invocation前提：timeout、retry或whole
restart可能重复调用相同semantic write，writer UDF及其外部sink MUST自行保证幂等。
Multigrain不提供事务、commit/abort或副作用回滚协议。

### 4.6 Pipeline outputs

`Pipeline.forward()` 只允许返回：

```text
Port
tuple[Port, ...]
None
```

分别对应一个、多个和零个 `ResultPort`。首版不保存 dict、NamedTuple 或通用 PyTree
output structure。

### 4.7 static authoring subset

compiler以symbolic Port执行一次`Pipeline.forward()`。普通Python control flow只要不读取
Port业务值即可使用，包括基于静态config的condition、遍历静态tuple和调用helper
function。首版module discovery只递归遍历Pipeline直接属性与tuple slots：

```python
self.layers = (mg.Map(A), mg.Map(B))

def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

对应稳定paths为`layers[0]`和`layers[1]`。list、dict、custom container和nested Pipeline
首版不支持。未被当前静态分支调用的module可存在但不进入CompiledGraph；同一个module
object出现在两个paths、调用未绑定module或同一instance调用两次均为CompileError。
同一Port被多次消费属于合法fan-out。

Port不提供运行时值。`bool(port)`、迭代、`len(port)`和`port[index]`必须立即抛出
`CompileError(PORT_VALUE_NOT_AVAILABLE)`；data-dependent control flow必须由未来具有
明确dataflow semantics的primitive表达。其他forward异常统一包装为
`CompileError(AUTHORING_TRACE_FAILED)`并保留cause。

## 5. 唯一 CompiledGraph

不保留 AuthoringGraph、VerifiedGraph 或 backend-specific graph。唯一 graph IR：

```python
@dataclass(frozen=True, slots=True)
class CompiledPort:
    id: PortId
    node: NodeId | None
    slot: int
    domain: DomainId


@dataclass(frozen=True, slots=True)
class SourceRelation:
    pass


@dataclass(frozen=True, slots=True)
class SameAs:
    role_names: tuple[str, ...]
    parents: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class SubsetOf:
    target: PortId
    controls: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class ChildrenOf:
    parent: PortId
    context_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class AggregateOf:
    anchor: PortId
    role_names: tuple[str, ...]
    member_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class RelatedFrom:
    role_names: tuple[str, ...]
    role_ports: tuple[PortId, ...]


PortRelation = (
    SourceRelation
    | SameAs
    | SubsetOf
    | ChildrenOf
    | AggregateOf
    | RelatedFrom
)


@dataclass(frozen=True, slots=True)
class MapSpec:
    primary: PortId
    role_names: tuple[str, ...]
    inputs: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class FilterSpec:
    mode: Literal["predicate", "mask", "select"]
    role_names: tuple[str, ...]
    control_inputs: tuple[PortId, ...]
    targets: tuple[PortId, ...]
    annotation_arity: int = 0


@dataclass(frozen=True, slots=True)
class ExpandSpec:
    parent: PortId
    role_names: tuple[str, ...]
    inputs: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class ReduceMemberSpec:
    role: str
    port: PortId


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    anchor: PortId
    members: tuple[ReduceMemberSpec, ...]


@dataclass(frozen=True, slots=True)
class RelateRoleSpec:
    name: str
    value_port: PortId
    key_port: PortId | None


@dataclass(frozen=True, slots=True)
class RelateSpec:
    mode: Literal["key", "custom"]
    roles: tuple[RelateRoleSpec, ...]


OperationSpec = MapSpec | FilterSpec | ExpandSpec | ReduceSpec | RelateSpec


@dataclass(frozen=True, slots=True)
class CompiledNode:
    id: NodeId
    spec: OperationSpec
    factory: OperatorFactory | None
    config: ModuleConfig
    accepts_context: bool
    output_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class CompiledInputGroup:
    name: str
    position: int
    domain: DomainId
    port: PortId


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    nodes: tuple[CompiledNode, ...]
    ports: tuple[CompiledPort, ...]
    inputs: tuple[CompiledInputGroup, ...]
    outputs: tuple[PortId, ...]
    topological_nodes: tuple[NodeId, ...]
```

FilterSpec validation：

```text
predicate -> one control_input, one or more targets, annotation_arity=0
mask      -> one mask control_input, one or more targets, annotation_arity=0
select    -> non-empty control_inputs == targets in UDF argument order,
             annotation_arity >= 0
```

`CompiledGraph` MUST 自包含所有 runtime 所需语义；NodeRuntime 不允许回读 Pipeline、
module attributes 或 Python AST。
immutable指detached semantic snapshot：graph不得保存live module/UDF instance或指向用户
原始可变配置的引用。OperatorFactory保存bytes，runtime_env保存序列化round-trip后的独立
mapping；internal runtime不得修改CompiledGraph中的对象。

`OperationSpec` 是 primitive binding 和output relation的唯一 authority。flat RPC input 顺序由
`ordered_input_ports(spec)` 纯函数生成，不在 `CompiledNode` 中重复保存。Key Relate 的
planning-key demand 直接来自 `RelateRoleSpec.key_port`，不增加第二份 demand list。

`Filter.by_mask()` 的 `CompiledNode.factory` 为 None；其余需要 UDF 的 nodes 必须具有
factory。compiler 必须在生成 graph 前验证该矩阵。

`PortRelation` 是按需生成的port-centric semantic view，不存入CompiledPort或其他
serialized graph state。唯一总函数：

```python
def relation_of(
    graph: CompiledGraph,
    port: PortId,
) -> PortRelation:
    ...
```

source port直接返回SourceRelation；node output通过producing CompiledNode.spec和
output slot推导：

```text
Map output       -> SameAs(all value input roles)
Filter output    -> SubsetOf(identity target, additional control/value parents)
Expand output    -> ChildrenOf(primary parent, aligned context ports)
Reduce output    -> AggregateOf(anchor, member roles)
Relate output    -> RelatedFrom(ordered value roles)
source port      -> SourceRelation
```

`relation_of()`是唯一投影实现。compiler用它验证DomainId propagation，
`materialize_bundle()`用它选择provenance constructor，visualizer/debugger也只调用该
函数。允许实现进程内private cache，但cache不得序列化、参与graph equality或成为
semantic authority。

`SubsetOf.target`提供output identity/domain anchor；`controls`保存除target外影响
membership或derived value的direct parents，并按OperationSpec顺序去重。predicate默认
target自身时controls为空；by-mask为`(mask,)`。Select前`len(targets)`个outputs分别以
对应target为anchor、其余UDF inputs为controls；全部annotation outputs固定以
`targets[0]`为anchor、其余UDF inputs为controls。因此single-node Select仍保留全部
direct parents而不虚构内部mask Port，也不增加重复的annotation-anchor字段。

DomainId规则：

```text
source input group -> fresh domain
SameAs             -> primary domain，所有aligned roles必须同domain
SubsetOf           -> target domain
ChildrenOf         -> fresh callsite domain，所有outputs共享
AggregateOf        -> anchor domain
RelatedFrom        -> fresh callsite domain，所有outputs共享
```

`OperatorFactory` 只保存可序列化 constructor recipe：

```python
@dataclass(frozen=True, slots=True)
class OperatorFactory:
    payload: bytes

    def build(self) -> object:
        udf_cls, args, kwargs = cloudpickle.loads(self.payload)
        return udf_cls(*args, **kwargs)
```

compiler MUST 对`(udf_cls, copied_args, copied_kwargs)`执行一次cloudpickle round-trip，
并只把生成的immutable bytes保存进CompiledGraph；失败时抛
`CompileError(code="UNSERIALIZABLE_UDF_FACTORY")`，不得延迟到actor startup。

## 6. Identity 与 admission

### 6.1 基础标识

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

`BatchId` 隔离 microbatches。`DomainId` 表示可对齐的 identity 空间。`EntityId`
表示该 domain 中的语义实体；它 MUST NOT 包含 BatchId、PortId、actor、shard、
attempt 或完成顺序。

### 6.2 source keys

V2.2只支持external-feed-driven admission，没有zero-input Source primitive。feed records
可以是业务rows，也可以是轻量file paths、manifest entries、row-group/index shards或
offset ranges；reader UDF在actors中读取实际payload。

一个完整root path可以作为一条输入驱动pipeline，但只形成一个source WorkUnit。大规模
distributed read应由外部lazy iterator/index按可并行descriptor粒度组成microbatches，
再交给`run_stream()`；driver只枚举和传递descriptors，不读取业务payload。

每个显式 `Pipeline.forward()` 参数恰好形成一个独立 `CompiledInputGroup`：

```python
class UserEvents(mg.Pipeline):
    def forward(
        self,
        users: mg.Port[User],
        events: mg.Port[Event],
    ) -> mg.Port[Pair]:
        ...
```

首版不允许一个 input group 包含多个 source ports。需要多列时，用户把一行表示为
dataclass/dict/tuple row value，或在第一个 Map 中拆成多个 ports。

单参数 pipeline 可使用 list shorthand：

```python
executor.run(users)
```

多参数 pipeline 必须使用按 forward 参数名绑定的 mapping：

```python
executor.run(
    {
        "users": users,
        "events": mg.provided_keys(events, event_ids),
    },
    failure_mode="fail_closed",
)
```

`run_stream()` 的每个元素使用相同 binding 形式。缺参数、多参数、未知参数或 positional
歧义均为 `INVALID_INPUT` aborted，不调用 Pipeline.forward()。

plain `list[T]` admission 使用 PositionKeys，EntityId 由 input ordinal 生成。用户提供
稳定业务 key 时使用：

```python
mg.provided_keys(values, keys)
```

```python
@dataclass(frozen=True, slots=True)
class ProvidedInput(Generic[T]):
    values: list[T]
    keys: list[object]


@dataclass(slots=True)
class AdmittedInputGroup:
    spec: CompiledInputGroup
    retained_values: list[object]
    entities: tuple[EntityId, ...]


@dataclass(slots=True)
class AdmittedInputs:
    groups: tuple[AdmittedInputGroup, ...]
```

`ProvidedInput` 合同：

- keys 必须 canonical-encodable；
- keys 与 values 等长且唯一；
- identity 来自 key，不来自输入位置。

`mg.provided_keys()` 只用于 concrete admission；它与 graph authoring 中的
`mg.keyed(values, by=key_port)` 是不同类型。

`AdmittedInputs`是microbatch-scoped source closure：Executor在result delivery前强引用
原Python value lists和admission时已canonicalized的entities，不deep-copy业务values。
Provided keys完成canonicalization后无需继续保留原key对象。

每个MicrobatchAttempt从AdmittedInputs重新`ray.put()`并在该attempt的RunArena中构造
source PortData；source ObjectRefs和locators不存入AdmittedInputs。source rows MUST NOT
是Ray ObjectRef，因此driver存活时whole restart可重新put source values。

caller在对应RunResult delivery前MUST NOT修改已admit的value list或其中row objects；
否则行为未定义。serialization、Ray session或driver级故障属于global ExecutionError，
不降级为batch-local source loss。

### 6.3 top-level empty

每个 `CompiledInputGroup` 可以独立为空。只有所有 input groups 均为零行时，
该 microbatch 才以 `EMPTY_INPUT` 返回 aborted。部分 group 为空由图语义正常传播。

`run_stream([])` 表示没有 microbatch，正常结束且不产生 aborted result。

### 6.4 derived identity

```text
Map              -> primary input EntityId
Filter survivor  -> filtered target EntityId
Expand child     -> H(callsite salt, parent DomainId, parent EntityId, sibling ordinal)
Reduce output    -> anchor EntityId
Relate output    -> H(callsite salt, ordered role (DomainId, EntityId) tuple)
```

output slot 不进入 EntityId；同一 Expand/Relate invocation 的多个 output ports
共享 DomainId 与 EntityId vector。BatchId、attempt 和 physical information MUST NOT
进入上述 hash input。

### 6.5 canonical bytes 与 stable hash

V2.2 的 `CanonicalValue` 只支持：

```text
None
bool
arbitrary-precision int
finite binary64 float
UTF-8 string
bytes
recursive tuple[CanonicalValue, ...]
```

list、dict、set、自定义对象、NaN、正负Infinity全部拒绝。这样避免容器可变性、cycle和
跨语言mapping-order歧义。

canonical byte grammar版本固定为`MGCV1`。`uleb128(n)` 是标准unsigned LEB128：
每byte低7位承载数据、最高位表示后续byte，least-significant group first；必须使用最短
编码。`frame(payload) = uleb128(len(payload)) || payload`。

```text
value := one-byte type tag || unsigned-varint payload-length || payload
None  := tag N, empty payload
bool  := tag B, payload 00 or 01
int   := tag I, minimal ASCII decimal, 0 has no sign
float := tag F, IEEE-754 binary64 big-endian; -0 normalized to +0
str   := tag S, Unicode NFC then UTF-8
bytes := tag Y, raw bytes
tuple := tag T, concatenated framed child values
```

编码type-sensitive，因此`1`与`1.0`不同。decoder MUST 拒绝non-minimal representation。

所有stable IDs使用完整32-byte SHA-256：

```text
H(domain_tag, fields...)
= SHA256(
    b"MGID/v1"
    || frame(canonical_encode(domain_tag))
    || frame(canonical_encode(field_0))
    || ...
)
```

H的每个参数必须是CanonicalValue：字符串按S编码，raw NodeId/DomainId/EntityId按Y编码，
ordinal按I编码。任何调用点不得自行拼接bytes。

compiler要求每个被调用的PrimitiveModule instance恰好绑定一个Pipeline直接属性或递归
tuple slot。tuple索引使用十进制`[n]`进入attribute path。NodeId/callsite salt派生自：

```text
H("node", pipeline class module, qualname, attribute path, primitive kind)
```

Select直接生成一个`FilterSpec(mode="select")` node，不追加内部callsite suffix。别名
指向同一module或无法找到唯一attribute path时CompileError。

```text
DomainId(source)   = H("source-domain", NodeId-like input parameter salt)
DomainId(expand)   = H("expand-domain", NodeId)
DomainId(relate)   = H("relate-domain", NodeId)
EntityId(position) = H("source-position", DomainId, ordinal)
EntityId(provided) = H("source-key", DomainId, canonical key bytes)
```

其中input parameter salt精确定义为：

```text
H("input", pipeline class module, qualname, parameter name, parameter position)
```

MGCV1只服务stable IDs和`provided_keys()` source identity，不参与Key Relate运行时判等。
Key Relate采用第11.5节定义的Python value equality。

reference oracle必须独立实现上述公开byte grammar与hash vectors。

## 7. list column、ValueShard 与 RowId locator

首版 UDF value column 唯一 ABI 是 `list[T]`：

```text
len(column)
column[start:stop]
[column[i] for i in indexes]
left + right
```

NumPy array、Tensor、Arrow object 可以作为单个 row value，但不是首版 column container。

```python
@dataclass(frozen=True, slots=True)
class ValueShard:
    ref: ray.ObjectRef
    row_count: int


@dataclass(frozen=True, slots=True)
class ShardPosition:
    shard: int
    offset: int


@dataclass(frozen=True, slots=True)
class KeyIndex:
    key_by_row: tuple[Hashable, ...]
    rows_by_key: tuple[tuple[Hashable, tuple[RowId, ...]], ...]


@dataclass(frozen=True, slots=True)
class PortData:
    port: PortId
    domain: DomainId
    entities: tuple[EntityId, ...]
    shards: tuple[ValueShard, ...]
    locator: tuple[ShardPosition, ...]
    provenance: ProvenanceTable
    key_index: KeyIndex | None
```

RowId 是 published port 中的 canonical ordinal。bundle materializer 按：

```text
canonical WorkUnit order
-> unit-local output ordinal
```

分配连续 RowIds。`locator[row_id]` 是 RowId 到 ValueShard/offset 的唯一解析路径。
actor completion order 不影响 RowId。

whole-microbatch restart 丢弃旧 RunArena、PortData 和 locator。不同 attempts 的 RowRef
MUST NOT 同时可解析；attempt ID 只进入 trace，不进入 RowRef。

```python
@dataclass
class RunArena:
    batch: BatchId
    attempt: MicrobatchAttempt
    ports: dict[PortId, PortData]
    bundles: dict[NodeId, OutputBundle]
```

## 8. 类型化 direct provenance

只保存 direct parents，不保存传递闭包：

```python
@dataclass(frozen=True, slots=True)
class SourceRows:
    pass


@dataclass(frozen=True, slots=True)
class AliasRows:
    role_names: tuple[str, ...]
    parents_by_role: tuple[tuple[RowRef, ...], ...]


@dataclass(frozen=True, slots=True)
class ChildRows:
    role_names: tuple[str, ...]
    parents_by_role: tuple[tuple[RowRef, ...], ...]
    sibling_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AggregateRows:
    anchors: tuple[RowRef, ...]
    role_names: tuple[str, ...]
    offsets_by_role: tuple[tuple[int, ...], ...]
    members_by_role: tuple[tuple[RowRef, ...], ...]


@dataclass(frozen=True, slots=True)
class RelatedRows:
    role_names: tuple[str, ...]
    parent_rows_by_role: tuple[tuple[RowRef, ...], ...]


ProvenanceTable = (
    SourceRows
    | AliasRows
    | ChildRows
    | AggregateRows
    | RelatedRows
)
```

column contract：

- `AliasRows.parents_by_role[r]` 长度等于 output row count；
- `ChildRows.parents_by_role[r]` 长度等于 child row count；
- `AggregateRows.anchors` 长度等于 aggregate output count；
- 每个 aggregate role offsets 长度为 output count + 1，首项0、末项等于该 role
  members长度；
- `RelatedRows.parent_rows_by_role[r]` 长度等于 relation output count；
- role_names 与对应 role-major columns 一一对应且唯一。

Map 的所有 value inputs 都是 direct parents。Filter output 的 identity target row 与
决定其survival/value的control input rows都是direct parents。Expand 的 primary parent和所有
aligned context inputs都是 direct parents。control parent进入dependency/provenance，
但不进入output identity。

`parent_edges()` 是唯一通用 ancestry projection。primitive-specific closure 仍由
PrimitiveKernel 解释，不能退化为无类型 edge bag。

## 9. WorkUnit 与 payload

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
    key: Hashable


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


@dataclass(frozen=True, slots=True)
class RowAlignedLayout:
    work_unit_count: int


@dataclass(frozen=True, slots=True)
class FiberGroupedLayout:
    role_names: tuple[str, ...]
    offsets_by_role: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class KeyRelationLayout:
    role_names: tuple[str, ...]
    tuple_offsets_by_work_unit: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CustomRelationLayout:
    role_names: tuple[str, ...]
    role_lengths: tuple[int, ...]


WorkLayout = (
    RowAlignedLayout
    | FiberGroupedLayout
    | KeyRelationLayout
    | CustomRelationLayout
)
```

WorkUnit closure：

```text
Map / Filter / Expand -> one primary row or parent
Reduce                -> one anchor and its complete member fiber
Key Relate            -> one complete Python-equality join-key partition
Custom Relate         -> whole invocation
```

一个 WorkUnit 的业务输出 MUST 与同一 actor RPC 中有哪些 peer WorkUnits 无关。

driver payload 使用 ref slots，确保 ObjectRefs 作为 actor RPC 顶层参数传递：

```python
@dataclass(frozen=True, slots=True)
class RpcShardTake:
    ref_slot: int
    selector: slice | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RpcPortSlice:
    takes: tuple[RpcShardTake, ...]


@dataclass(frozen=True, slots=True)
class RpcPayload:
    slices: tuple[RpcPortSlice, ...]
    layout: WorkLayout
    context: UdfContext | None


@dataclass(frozen=True, slots=True)
class PlannedRpc:
    payload: RpcPayload
    value_refs: tuple[ray.ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    invocation: InvocationRef
    work_units: tuple[WorkUnitRef, ...]
    rpc: PlannedRpc
    preterminal_failures: tuple["TerminalFailure", ...]
```

`InvocationPlan.work_units` 只包含ready、需要执行的canonical sequence；
preterminal failures不占WorkRange index。layout cardinality规则：

```text
RowAligned.work_unit_count = len(work_units)
Fiber role offsets长度 = len(work_units)+1
Key tuple offsets长度 = len(work_units)+1
Custom work_units恰好一个WholeInvocation，role_lengths与roles一一对应
```

`take_work_range(plan, range) -> PlannedRpc` 是唯一切片算法：

```text
RowAligned:
  每个input column按[start:stop]切
FiberGrouped:
  anchor按WorkRange切；每个member role按其offset区间切并rebase offsets
KeyRelation:
  每个aligned role column按tuple_offsets[start:stop]切并rebase offsets
CustomRelation:
  只允许range=(0,1)，传递每个独立role完整column
```

返回的RpcPayload使用local offsets。UdfContext.item_keys在Row/Reduce按range内
WorkUnits对齐，在KeyRelation按plan-known flat parent tuples对齐；Custom Relate中始终
为空，因为parent pairs只能由UDF返回后确定。
`CompiledNode.accepts_context=False`时context必须为None；为True时plan创建full context，
`take_work_range()`同步切分item_keys；invocation_key在所有ranges中相同。

切片后只保留仍被takes引用的ObjectRefs，并按`RpcPayload.slices`、`takes`扫描时首次出现
顺序重新构造`PlannedRpc.value_refs`和从0连续编号的ref_slots。不存在slot悬空或引用
value_refs范围外位置。

调用形式：

```python
refs = actor.run.options(num_returns=1 + output_arity).remote(
    token,
    planned_rpc.payload,
    *planned_rpc.value_refs,
)
```

Ray 在 actor method 执行前解析顶层 refs。actor MUST NOT 对 value refs 调用 `ray.get()`。
`PlannedRpc.value_refs` 按 `ordered_input_ports(spec)` 顺序扫描各 PortSlice，并按
shard首次出现顺序去重；`ref_slot` 是该 tuple 中的稳定位置。

## 10. PrimitiveKernel 唯一扩展点

```python
@dataclass(frozen=True, slots=True)
class KernelInvokeResult:
    columns: tuple[list[object], ...]
    offsets: tuple[int, ...]
    evidence: CustomRelationEvidence | None = None


class PrimitiveKernel(Protocol):
    def validate(self, node: CompiledNode, graph: CompiledGraph) -> None: ...

    def plan(
        self,
        node: CompiledNode,
        arena: RunArena,
    ) -> InvocationPlan: ...

    def invoke(
        self,
        node: CompiledNode,
        udf: object | None,
        values: tuple[list[object], ...],
        layout: WorkLayout,
        context: UdfContext | None,
    ) -> KernelInvokeResult: ...

    def materialize_bundle(
        self,
        node: CompiledNode,
        graph: CompiledGraph,
        plan: InvocationPlan,
        accepted: tuple[AcceptedRpc, ...],
        arena: RunArena,
    ) -> OutputBundle: ...
```

静态 private registry：

```python
KERNELS = {
    MapSpec: MAP_KERNEL,
    FilterSpec: FILTER_KERNEL,
    ExpandSpec: EXPAND_KERNEL,
    ReduceSpec: REDUCE_KERNEL,
    RelateSpec: RELATE_KERNEL,
}
```

新增 core primitive 必须提供完整 kernel bundle 和 conformance tests。普通用户通过 UDF
或 authoring macro 扩展，不注册半套 handler。

Ray actor只import`kernels.dispatch.invoke_rpc(...)` facade。它是唯一transport adapter：
按每个RpcPortSlice的takes顺序，从Ray已解析的columns中执行selector并拼成一个逻辑input
column，检查ref_slot/selector/layout边界，然后在`kernels/`内查询上述registry：

```python
def invoke_rpc(
    node: CompiledNode,
    udf: object | None,
    payload: RpcPayload,
    resolved_columns: tuple[list[object], ...],
) -> KernelInvokeResult:
    values = tuple(
        gather_port_slice(port_slice, resolved_columns)
        for port_slice in payload.slices
    )
    return KERNELS[type(node.spec)].invoke(
        node=node,
        udf=udf,
        values=values,
        layout=payload.layout,
        context=payload.context,
    )
```

`ray/actor.py`不import concrete kernel modules。dispatcher不理解primitive semantics，
不生成identity/provenance，也不做recovery；actor-side primitive normalization仍由
`PrimitiveKernel.invoke`唯一负责。禁止actor或kernel分别重复解析ref slots。

唯一 normalization：

```text
Map:
  UDF columns长度 = WorkUnit count；offsets = 0..N
Filter predicate:
  UDF返回bool mask；kernel对targets做subset；offset增量为0或1
Filter.by_mask:
  无factory；kernel直接验证并应用mask
Filter select:
  UDF第一项为bool mask、其余为annotations；kernel在同一RPC过滤inputs和annotations
Expand:
  每个output是outer长度N的list[list[T]]；逐parent child counts必须跨outputs相同；
  kernel flatten inner lists并生成prefix offsets
Reduce:
  UDF columns长度 = anchor WorkUnit count；offsets = 0..N
Key Relate:
  UDF columns长度 = flat parent tuple count；offsets按JoinKey partition tuple counts生成
Custom Relate:
  kernel拆RelationResult.values与parents；offsets = (0, relation_count)
zero-output:
  columns = ()；offsets为长度N+1的全0 prefix；manifest只作每个WorkUnit成功ack
```

## 11. 五类 primitive

### 11.1 Map

```python
def run(
    self,
    first: list[A],
    second: list[B],
) -> list[C]:
    ...
```

- primary identity 规划 Row WorkUnits；
- aligned inputs 按 IdentityKey gather；
- 每个成功 WorkUnit 恰好产生一行；
- outputs 复用 primary identity；
- 所有输入 representation parents 进入 AliasRows。

### 11.2 Filter

predicate 形式：

```python
def run(self, values: list[A]) -> list[bool]:
    ...
```

已有 mask：

```python
self.keep = mg.Filter.by_mask()
survivors = self.keep(values, mask=mask)
```

- predicate mode 默认把唯一predicate input同时作为唯一target；
- predicate UDF首版必须恰好接收一个list column，并返回一个`list[bool]`；
- 额外/不同targets使用`module(predicate_input, targets=(a, b))`；
- by-mask mode使用`module(a, b, mask=mask)`，位置参数全部是targets；
- 一个target返回一个Port，多个targets按传入顺序返回tuple[Port, ...]；
- mask 是 control input，不是 output；
- false 是合法零基数，不是 failure；
- survivor 保留 target identity；
- 多 targets 各自产生 subset output；
- mask 长度和 targets 长度不一致是 contract violation。
- runtime要求mask每个元素的type恰为bool；整数0/1不隐式接受。

`Select` 是同一FilterKernel的fused select mode，不产生中间Map node、mask Port或第二个
actor pool：

```python
class Score:
    def run(
        self,
        texts: list[str],
    ) -> tuple[list[bool], list[float]]:
        scores = score(texts)
        return [value >= 0.8 for value in scores], scores


self.select = mg.Select(Score).ray_options(replicas=8, num_gpus=1)
kept_texts, kept_scores = self.select(texts)
```

固定合同：

- UDF接收一个或多个按IdentityKey完全对齐的`list` columns；
- 返回`list[bool]`表示只有mask，或固定tuple，其第一项是`list[bool]`、其余项是annotation
  columns；
- runtime要求mask元素type恰为bool，所有annotation长度与input outer dimension相同；
- output依次为filtered input ports、filtered annotation ports，mask不作为output；
- 一个总output返回Port，多个返回tuple；
- 所有outputs共享同一0/1-per-input-row layout；
- `ray_options()`只配置这个fused node的actor pool；
- FilterKernel在同一RPC内调用UDF并完成subset，未过滤mask/annotations不进入Object Store。

### 11.3 Expand

```python
def run(self, parents: list[A]) -> list[list[B]]:
    ...
```

outer dimension 是 parent WorkUnits。actor manifest 的 per-WorkUnit offsets 描述每个
parent 的 child count。多 outputs 必须共享 offsets。child identity 来自 parent identity
与 sibling ordinal；合法 zero-child 不产生 failure。

### 11.4 Reduce

```python
def run(
    self,
    anchors: list[A],
    grouped_members: list[list[B]],
) -> list[C]:
    ...
```

authoring：

```python
reduced = self.reduce(anchor_port, member_port)
```

role 自连接有歧义时才使用：

```python
reduced = self.reduce(
    mg.by_role(anchor=anchors, positive=pos, negative=neg)
)
```

`anchor` 是 `by_role()` 的保留keyword，并绑定UDF signature中第一个非context参数；
其余keywords按signature member role顺序绑定。首版所有member roles都是required，不提供
optional member API。

- 每个 anchor fiber 一个 WorkUnit；
- complete empty member group 合法，仍调用 UDF；
- required member 因 upstream failure 缺失时 suppress 整个 fiber；
- output 复用 anchor identity；
- AggregateRows 保存 anchor 和完整 direct members。

fiber membership 只有一个算法。对 member row `m`，沿 typed `parent_edges()` 递归向上，
收集位于 anchor port DomainId 且存在于当前 anchor PortData 的 ancestors：

```text
exactly one anchor -> m属于该anchor fiber
zero anchors       -> ContractViolation(UNBOUND_REDUCE_MEMBER)
multiple anchors   -> ContractViolation(AMBIGUOUS_REDUCE_MEMBER)
```

每个 `ReduceMemberSpec` 是一个独立 role。role 中没有成员是 complete empty group，合法；
若该 role 的 upstream TerminalFailure 可归因到某个 anchor，则 suppress该anchor
fiber。failure可归因到多个/零个anchors时，为避免不完整聚合，suppress whole Reduce
invocation。

### 11.5 Key Relate

```python
relations = self.join(
    user=mg.keyed(users, by=user_ids),
    event=mg.keyed(events, by=event_user_ids),
)
```

`mg.keyed()` 只构造 symbolic `KeyedRole`。compiler 生成：

```python
RelateRoleSpec(name="user", value_port=users.id, key_port=user_ids.id)
RelateRoleSpec(name="event", value_port=events.id, key_port=event_user_ids.id)
```

规则：

- canonical role order 来自 UDF `run()` signature，不来自 keyword 调用顺序；
- role names 必须与 signature 精确绑定；
- primary role 是 signature 中第一个非context role；
- value/key ports 必须具有相同domain；unexplained IdentityKey set mismatch是
  ContractViolation，explained missing先按11.8 closure matrix处理，再验证剩余sets；
- key values必须是Hashable，分组直接采用Python `__hash__`/`__eq__`语义；
- driver 只 materialize planning key ports；
- repeated keys 在 role order 下形成完整笛卡尔 parent tuples；
- unmatched key 不产生 WorkUnit；
- partition 顺序为 primary role key 首次出现顺序，不做全局 sort；
- relation identity 来自 ordered role-parent `(DomainId, EntityId)` tuple，不来自BatchId或
  join key。

同一key partition内，各role rows按其PortData RowId顺序，笛卡尔积按signature role顺序
做lexicographic enumeration。该顺序仅是稳定presentation/WorkRange顺序；unordered
relation的语义比较仍按relation IdentityKey set，不宣称业务排序。

UDF 接收与 parent tuples 对齐的 role columns，返回普通 output columns。

driver按key port RowId/locator顺序`ray.get`其ValueShards，重建一个list column并直接
建立Python-equality KeyIndex。相同key port被多个Relate消费时复用PortData上的KeyIndex。
unhashable key或`__hash__`/`__eq__`异常产生batch-local `INVALID_JOIN_KEY` aborted，不
调用Relate UDF。NaN、`True == 1`等行为遵循Python本身，不额外规范化。

### 11.6 Custom Relate

plain named role ports 表示 Custom Relate：

```python
matches = self.match(image=images, detection=detections)
```

```python
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RelationResult(Generic[T]):
    parents: Mapping[str, list[int]]
    values: T
```

```python
class SpatialMatch:
    def run(
        self,
        image: list[Image],
        detection: list[Detection],
    ) -> mg.RelationResult[list[Match]]:
        image_indexes, detection_indexes, matches = match_all(
            image,
            detection,
        )
        return mg.RelationResult(
            parents={
                "image": image_indexes,
                "detection": detection_indexes,
            },
            values=matches,
        )
```

contract：

- custom mode 所有 roles 必须是 plain ports；与 KeyedRole 混用非法；
- roles 接收独立长度 columns；
- `parents` keys 与 signature role names 完全相同；
- parent index columns 等长且不得越界；
- duplicate ordered parent tuple 非法；
- values cardinality 与 parent tuple count 相同；
- multi-output 使用 `RelationResult[tuple[list[A], list[B]]]`；
- zero-output custom writer annotation 为 `None`，只记录 WholeInvocation closure，不声称
  知道 UDF 内部具体 pairs。

### 11.7 aligned fan-in 与 diamond

Map/Expand 的第一个 UDF role 是 primary，定义 authoritative WorkUnit sequence。对每个
primary IdentityKey，所有 required aligned roles 必须能唯一lookup：

```text
primary key存在于secondary
-> gather对应row

secondary含额外keys
-> 合法，当前invocation忽略

primary key因secondary upstream TerminalFailure缺失
-> fail_closed suppress该primary WorkUnit

primary key仅因secondary合法Filter false而缺失
-> ContractViolation(STRUCTURAL_ALIGNMENT_MISMATCH)
```

因此 filtered/unfiltered diamond 的规范接线是把 filtered subset 作为 primary，再从
unfiltered superset按IdentityKey gather；反向接线非法。若业务需要两个不同subsets的
intersection，必须先显式构造共同mask或relation，Map不会隐式取intersection。

Filter predicate/mask/select的所有control inputs与targets要求exact IdentityKey set。
任何fan-in都禁止按物理位置zip。该规则同时定义了M3 diamond theorem的合法set premise。

### 11.8 fail-closed upstream closure matrix

fail_fast在第一个TerminalFailure产生时终止batch。只有fail_closed进入以下传播：

```text
Map:
  required aligned row缺失且可按IdentityKey归因 -> suppress该Row WorkUnit
  无法归因 -> suppress whole invocation

Filter:
  target或任一control input row的explained failure -> suppress对应IdentityKey WorkUnit
  合法false -> 正常zero cardinality，不是failure
  无解释的exact-set mismatch -> ContractViolation

Expand:
  primary/aligned parent failure -> suppress该Parent WorkUnit，不虚构未知children

Reduce:
  member failure唯一归因anchor -> suppress该Fiber WorkUnit
  零/多anchor归因 -> suppress whole invocation

Key Relate:
  planning key port在KeyIndex完成前有任何explained missing
    -> 生成InvocationRef-scoped failure并suppress whole invocation，不把未知key猜成unmatched
  KeyIndex完成后value row failure
    -> suppress所有包含该row的JoinKey WorkUnits

Custom Relate:
  任一role有explained missing -> suppress WholeInvocation
```

source ValueShard loss不是TerminalFailure；它使用17.2的whole-attempt restart/re-put。

## 12. Ray multi-return 与 RpcManifest

所有 NodeRuntime actors 显式设置：

```python
ActorClass.options(
    max_restarts=0,
    max_task_retries=0,
    scheduling_strategy=placement_group_strategy,
).remote(...)
```

Ray hidden retry/reconstruction MUST 关闭。`ModuleConfig.max_retries` 是 Multigrain
显式 retry budget，不是 Ray option。

actor method 直接返回：

```python
def __init__(self, compiled_node: CompiledNode) -> None:
    self.node = compiled_node
    self.udf = (
        compiled_node.factory.build()
        if compiled_node.factory is not None
        else None
    )


def run(self, token, payload, *resolved_columns):
    result = kernels.dispatch.invoke_rpc(
        self.node,
        self.udf,
        payload,
        resolved_columns,
    )
    rpc_manifest = RpcManifest(
        token=token,
        offsets_by_work_unit=result.offsets,
        evidence=result.evidence,
    )
    if not result.columns:
        return rpc_manifest
    return rpc_manifest, *result.columns
```

禁止 actor 内：

```python
return ray.put(output_column)
```

因为 nested ObjectRef 的 owner 会变成 actor。

```python
@dataclass(frozen=True, slots=True)
class RpcAttemptToken:
    value: UUID


@dataclass(frozen=True, slots=True)
class CustomRelationEvidence:
    parent_indexes_by_role: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class RpcManifest:
    token: RpcAttemptToken
    offsets_by_work_unit: tuple[int, ...]
    evidence: CustomRelationEvidence | None


@dataclass(frozen=True, slots=True)
class PendingRpc:
    actor_index: int
    invocation: InvocationRef
    work_range: WorkRange
    token: RpcAttemptToken
    manifest_ref: ray.ObjectRef
    value_refs: tuple[ray.ObjectRef, ...]
    deadline_ns: int | None


@dataclass(frozen=True, slots=True)
class AcceptedRpc:
    work_range: WorkRange
    manifest: RpcManifest
    value_refs: tuple[ray.ObjectRef, ...]
```

driver 只 `ray.get(manifest_ref)`。planning key ports 是唯一 intermediate value exception；
它们被 compiler 显式标记并由 driver materialize。final ResultPort 由用户显式 collect。

`offsets_by_work_unit` 是 prefix offsets：

- 长度必须为 `WorkRange` WorkUnit count + 1；
- 第一项必须为0；
- 单调不下降；
- 每个 output column 长度必须等于末项；
- 一个 actor batch 的所有 output columns共享同一offsets；
- zero-output writer offsets全为0，`num_returns=1`，actor直接返回RpcManifest而不是一元
  tuple；manifest是每个WorkUnit的ack。调用端将单个ObjectRef归一化为一元素refs tuple。

accepted WorkRanges 必须无重叠，并完整覆盖全部settled-success WorkUnits；terminal
WorkUnits不出现在AcceptedRpc。bundle materializer为每个output slot分别调用：

```python
def build_locator(
    output_slot: int,
    accepted: tuple[AcceptedRpc, ...],
) -> tuple[ShardPosition, ...]:
    ...
```

`value_refs[output_slot]` 与该 slot 的 ValueShard 一一对应。

## 13. 原子 publication

node-invocation barrier：

```text
plan all ready WorkUnits
-> execute/recover every WorkRange
-> every WorkUnit settles as success or WorkUnit-scoped TerminalFailure,
   or one InvocationRef-scoped failure short-circuits the whole invocation
-> sort accepted batches by WorkRange
-> PrimitiveKernel.materialize_bundle once
-> atomically publish one OutputBundle
```

```python
@dataclass(frozen=True, slots=True)
class OutputBundle:
    invocation: InvocationRef
    ports: tuple[PortData, ...]
    failures: tuple[TerminalFailure, ...]
```

每个 invocation 每个 microbatch attempt 最多发布一次。downstream 不得读取 partial
actor results。stale token 的 manifest 和 value refs 整体丢弃。
本文的`node publication`专指OutputBundle进入RunArena；`result delivery`专指RunResult
由run返回或run_stream yield。两者不得混称。

同一WorkRange同时最多一个active RPC。普通failure必须先返回terminal outcome再retry；
timeout时先从wait set移除旧refs、kill旧actor并完成replacement，再提交新RPC。系统不比较
新旧payload或cardinality，retry结果就是该WorkRange结果；manifest token只校验返回值
属于当前PendingRpc，不维护额外range epoch/registry。

## 14. NodeRuntime 与调度

```python
@dataclass
class NodeRuntime:
    node: CompiledNode
    actors: list[ActorHandle]
    idle_actors: deque[int]
    ready_by_invocation: dict[InvocationRef, deque[WorkRange]]
    pending: dict[ray.ObjectRef, PendingRpc]
```

调度规则：

- replicas 是该 node 的持久 actor capacity；
- `max_inflight` 是 Executor 的 active microbatch admission count；
- 一个 actor 同时最多一个 RPC；
- 一个 RPC 只含一个 invocation 的 contiguous WorkRange；
- 不在同一 RPC 中 co-batch 多个 microbatches；
- 多个 ready invocations round-robin；
- actor mailbox 不作为隐式 queue。

## 15. Executor 生命周期

caller 必须先初始化 Ray：

```python
ray.init(...)
with mg.Executor(pipeline, max_inflight=4) as executor:
    ...
ray.shutdown()
```

Executor 不调用 `ray.init()` 或 `ray.shutdown()`。

```python
class ExecutorState(Enum):
    NEW = auto()
    PREPARED = auto()
    RUNNING = auto()
    POISONED = auto()
    CLOSED = auto()
```

状态合同：

```text
NEW -> PREPARED       context enter / prepare succeeds
PREPARED -> RUNNING   run or run_stream starts
RUNNING -> PREPARED   normal completion
RUNNING -> POISONED   global invariant or fatal Ray lifecycle failure
RUNNING -> POISONED   input iterator exception before explicit cleanup
RUNNING -> CLOSED     explicit early stream cancellation cleanup
any non-CLOSED -> CLOSED
```

prepare：

```text
validate ray initialized
-> compute graph-wide resources
-> create one placement group
-> create all replicas in assigned bundles
-> ping all actors
-> commit PREPARED
```

placement group 为每个 `(NodeId, replica_index)` 分配一个固定bundle，bundle resources
精确来自ModuleConfig。mapping在prepare期间冻结：

```python
ReplicaPlacement(
    node=node_id,
    replica=replica_index,
    bundle_index=bundle_index,
)
```

actors 使用 `PlacementGroupSchedulingStrategy` 指向各自bundle。replacement MUST 复用
原bundle index；无法创建replacement是global `ExecutionError(code="REPLACEMENT_FAILED")`。

任一步失败必须回滚全部 owned resources。`close()` 幂等，kill actors、remove placement
group、flush trace；不关闭 Ray。

## 16. run、run_stream 与 backpressure

`run()` 执行一个 microbatch：

```python
result = executor.run(inputs, failure_mode="fail_fast")
```

`failure_mode` 只允许 `"fail_fast"` 或 `"fail_closed"`，默认 fail_fast。它是整个
microbatch policy，不允许每个node各自配置。

`run_stream()` overlap 最多 `max_inflight` 个 microbatches，按完成顺序 yield：

```python
for result in executor.run_stream(
    batches,
    failure_mode="fail_fast",
):
    consume(result)
```

只在有 admission slot 时读取 input iterator。generator 消费暂停时不 admit 新 batch。
一个RunStream在创建时冻结单一failure_mode，不能逐batch改变。
`run()` 与 stream scheduler 共用实现，语义上等价于单元素、`max_inflight=1` 的 stream；
它在返回唯一 RunResult 前必须消费内部 StopIteration 并完成stream finalization，使
Executor回到PREPARED。

`run_stream()` 返回轻量 `RunStream`：

```python
class RunStream(Iterator[RunResult]):
    def close(self) -> None: ...
    def __enter__(self) -> "RunStream": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

完整消费可直接 `for`，自然耗尽后 Executor 回到 PREPARED。可能 early break 时 MUST：

```python
with executor.run_stream(batches) as stream:
    for result in stream:
        if should_stop(result):
            break
```

提前 `close()` 或 context exit：

```text
stop admission
-> mark active tokens stale
-> best-effort cancel pending RPCs
-> Executor.close()
-> transition CLOSED
```

`RunStream.__exit__()` 先检查是否已自然耗尽：已耗尽只确保stream closed并保留Executor
PREPARED；未耗尽或block内抛异常才执行上述Executor cleanup。`close()`幂等。

首版不修复 actor pool 供取消后复用。直接 `for` 后未 close 即 break 时状态保持
RUNNING且active stream仍登记在Executor中；任何新run操作得到
`ExecutionError(code="ACTIVE_STREAM")`。只有显式`RunStream.close()`或
`Executor.close()`负责cleanup。`__del__`只能发ResourceWarning，不能作为correctness
mechanism。

## 17. failure 与两层恢复

### 17.1 publication 前

```text
ContractViolation
-> raise ExecutionError(CONTRACT_VIOLATION)
-> poison Executor
```

`ModuleConfig.max_retries` 只表示原始 WorkRange 在generic UDF exception后可获得的额外
exact retries。每次logical range execution另有固定internal
`infrastructure_retries=1`，不进入public API。

```python
def recover(work_range, udf_retries_left):
    outcome = execute_with_one_infrastructure_retry(work_range)
    if outcome.is_success:
        return accept(outcome)
    if outcome.is_contract_violation:
        raise_global(outcome)
    if outcome.is_bad_record:
        return handle_bad_record(work_range, outcome)
    if outcome.is_generic and udf_retries_left > 0:
        return recover(work_range, udf_retries_left - 1)
    if len(work_range) == 1:
        return abort_batch(outcome)
    left, right = split_contiguous(work_range)
    return recover(left, 0), recover(right, 0)
```

`execute_with_one_infrastructure_retry()` 在第一次InfrastructureFailure时确认旧RPC已经
terminal再replace/retry；Timeout时先移除旧refs并kill旧actor，再replace/retry同一range。
第二次直接abort batch。replacement无法在原placement-group bundle创建属于global
ExecutionError并poison Executor。

对于含 `n` 个WorkUnits的原始range，在没有infrastructure failure时，generic isolation
logical RPC调用数上界为：

```text
max_retries + 2n - 1
```

考虑每个logical call最多一次infrastructure retry，physical RPC上界为该值的两倍。
split children至少执行一次且不重新获得UDF retry budget。

`BadRecordError(index: int | None)` 的index是当前RPC UDF outer dimension：

```text
Map/Filter -> row
Expand     -> parent
Reduce     -> anchor
KeyRelate  -> flat parent tuple，再由offsets映射到不可拆的JoinKey WorkUnit
CustomRelate -> 只允许None，定位WholeInvocation
```

负index或超出当前outer dimension的index是ContractViolation并poison Executor。
fail_fast立即abort。fail_closed下，显式index直接定位对应WorkUnit；None则对WorkRange
二分到singleton。Row/Fiber/JoinKey/WholeInvocation closure内部从不拆。定位后的
WorkUnit记为PermanentlyMissing，左右健康ranges重新执行。

no-index BadRecord isolation与显式index后的健康range replay，其logical RPC总数同样不
超过`2n - 1`，含每次最多一次infrastructure retry时physical RPC不超过`2(2n - 1)`。

只有显式 `BadRecordError` 可以在 fail-closed 下变成 PermanentlyMissing。generic/system
错误不得静默丢行。

### 17.2 result delivery前的accepted source/intermediate ObjectRef loss

只在 microbatch active 且尚未 yield 时自动处理：

```text
detect ObjectLostError for source/accepted intermediate/planning key ref
-> increment fixed MicrobatchAttempt
-> stop admitting work for this microbatch
-> kill/replace仍在执行旧attempt RPC的actors并drop全部旧refs
-> discard old RunArena, bundles, provenance and locators
-> retain same BatchId
-> rebuild from still-available AdmittedInputs
-> rerun whole graph once
```

Object loss可能在driver `ray.get(planning_key_ref)` 直接出现，也可能作为下游actor顶层
dependency resolution失败的Ray exception cause出现。transport classifier必须沿Ray
exception cause chain确认丢失ref属于当前RunArena的accepted ValueShard；未知
RayTaskError不得误分类为published-value loss。

整批 restart budget 首版固定为 1，不暴露 public knob，不消耗 node `max_retries`。
第二次published-value loss或batch-local UDF/data重跑失败返回aborted。

在V2.2 public input ABI中AdmittedInputs强引用Python lists，因此driver存活时source
closure始终可重建。serialization failure、Ray session loss、ContractViolation和
replacement资源失败保持global ExecutionError并poison；只有batch-local UDF/data失败或
第二次published-value loss返回aborted。

旧 attempt 中已成功的 terminal writer 可能再次执行，因此所有 zero-output writers MUST
使用`UdfContext`稳定keys或业务keys实现幂等。`succeeded`只表示latest valid writer
attempt返回ack，不表示writer只调用一次；`aborted`也不表示外部一定没有副作用。系统不
承诺external exactly-once。

final actor RPC正常完成、manifest校验通过并发布OutputBundle后，Executor直接返回
RunResult，不主动probe或读取final payload。此后ResultPort丢失不触发自动restart；
`collect()`或`iter_shards()`得到`ExecutionError(code="RESULT_UNAVAILABLE")`，上层可按需
重新提交原microbatch。

## 18. TerminalFailure 与 RunResult

```python
@dataclass(frozen=True, slots=True)
class TerminalFailure:
    scope: WorkUnitRef | InvocationRef
    outcome: PermanentlyMissing | Suppressed
    code: FailureCode
    message: str
    causes: tuple[WorkUnitRef | InvocationRef, ...]
```

WorkUnit scope用于可定位row/fiber/join-key failure；InvocationRef scope用于在WorkUnits
尚不可构造或必须保守放弃整个invocation时表达failure。Invocation-scoped failure不进入
可调度`InvocationPlan.work_units`，只进入preterminal_failures和OutputBundle。

batch-local 执行失败返回结果，不 raise：

```python
@dataclass(frozen=True, slots=True)
class FailureRecord:
    node: str
    coordinate: str
    outcome: Literal["permanently_missing", "suppressed"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionErrorSummary:
    code: str
    message: str
    causes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunMetrics:
    counters: Mapping[str, int | float]
    by_node: Mapping[str, Mapping[str, int | float]]


@dataclass(frozen=True, slots=True)
class RunResult:
    batch_id: str
    status: Literal["succeeded", "aborted"]
    outputs: tuple[ResultPort, ...]
    failures: tuple[FailureRecord, ...]
    metrics: RunMetrics
    error_summary: ExecutionErrorSummary | None

    def raise_for_status(self) -> None:
        if self.status == "aborted":
            raise ExecutionError.from_summary(self.error_summary)
```

规则：

- fail-closed 且有 explained TerminalFailures 仍可 succeeded；
- aborted 的 outputs 必须为空，不暴露 partial terminal branches；
- terminal-writer success 的 outputs 同样为空，但 status 是 succeeded；
- error_summary 为 None 当且仅当 succeeded；
- batch aborted 后 stream 可以继续；
- global invariant、Ray session failure和input iterator exception直接raise并poison
  Executor；ACTIVE_STREAM只做non-poisoning rejection。

```python
class ResultPort:
    @property
    def row_count(self) -> int: ...

    def collect(self) -> list[object]: ...

    def iter_shards(self) -> Iterator[list[object]]: ...
```

`iter_shards()` 按 logical RowId 顺序返回 list columns；`collect()` 等价于按该顺序连接
所有 shards。两者不暴露 ObjectRef。

Executor.close 后只要 Ray runtime 仍存活，ResultPort 可读；caller `ray.shutdown()`
后调用collect/iter_shards必须抛`ExecutionError(code="RESULT_UNAVAILABLE")`。

## 19. public exceptions

```python
class MultigrainError(Exception): ...


class CompileError(MultigrainError):
    code: str
    path: str | None


class ExecutionError(MultigrainError):
    code: str
    batch_id: str | None
    node_id: str | None
    causes: tuple[str, ...]

    @classmethod
    def from_summary(
        cls,
        summary: ExecutionErrorSummary | None,
    ) -> "ExecutionError":
        ...


class BadRecordError(MultigrainError):
    def __init__(
        self,
        message: str,
        *,
        index: int | None = None,
    ) -> None:
        ...
```

internal `InfrastructureFailure`、`PublishedValueLost`、`ContractViolation` 等通过
FailureCode、TerminalFailure 或 ExecutionErrorSummary 表达，不从顶层导出。

## 20. observability

默认使用内存 metrics 和 compact logging。显式 experiment/debug 模式使用
`JsonlTraceWriter`：

```python
class TraceWriter(Protocol):
    def write(self, event: TraceEvent) -> None: ...
    def close(self) -> None: ...
```

trace 必须包含：

```text
BatchId / MicrobatchAttempt / NodeId / WorkRange
RPC token / actor replacement / retry / split / stale rejection
publication / batch status / error code
active entities / direct edges / planning-key count and estimated bytes
selected rows / unique source shard rows / selector density
```

trace 不持久化业务 payload。Trace schema 必须版本化。

## 21. 不变量

1. CompiledGraph 是 runtime 唯一静态语义输入。
2. DomainId 只在 compile 阶段分配。
3. PortData entities、locator 与 provenance cardinality 始终一致。
4. 每个可见 RowRef 在当前 RunArena 中有唯一 locator。
5. 不同 microbatch attempts 的 RowRefs 不共存。
6. actor 不生成 EntityId 或 typed provenance。
7. WorkUnit 结果与 RPC batch peers 无关。
8. actor、shard、RPC、attempt 和 completion order 不进入 identity。
9. 一个 actor 同时至多一个 RPC。
10. 一个 RPC 只含一个 invocation 的 contiguous WorkRange。
11. 一个 run 同时至多 max_inflight 个 active microbatches。
12. 大型业务 intermediate values 不经过 driver。
13. planning key 是唯一显式 intermediate value exception。
14. value ObjectRefs 必须作为 actor RPC 顶层参数传递。
15. actor 不返回 actor-owned `ray.put` refs。
16. stale token 的整个 actor result 被拒绝。
17. 每个 WorkUnit publication 前成功settle或有terminal outcome；Invocation-scoped
    failure可在WorkUnits不可安全构造时短路整个invocation。
18. 每个 invocation 每 attempt 最多发布一个完整 OutputBundle。
19. downstream 只读取完整 published bundles。
20. diamond inputs 按 IdentityKey gather，绝不 zip 物理 columns。
21. generic/backend/contract failure 不会静默变成 missing。
22. whole replay 丢弃整个旧 RunArena，不混用 generations。
23. Ray hidden task retry 和 actor restart 始终关闭。
24. terminal writer retry 和 whole replay 都依赖幂等合同。

## 22. 明确 non-goals

- LocalExecutor 或多 backend abstraction；
- cross-microbatch co-batching；
- byte-perfect admission control；
- byte-sparse ObjectRef transfer；
- distributed provenance authority；
- manager actor、custom lease 或 ownership protocol；
- SourceOp、SinkOp 或 source/sink flags；
- per-batch public cancellation；
- cancel 后修复并复用 Executor；
- graph durable save/load；
- default full-lineage persistence；
- checkpoint/driver restart；
- external exactly-once；
- arbitrary custom primitive plugin；
- window、iteration、union 或 stateful streaming。

## 23. Conformance 要求

任何实现只有通过以下 gates 才能声称符合 V2.2：

- 五类 kernels 的 identity/provenance golden tests；
- deterministic、WorkUnit-separable UDF的randomized physical reorder/shard/retry
  invariance；
- 同一WorkRange最多一个active RPC；timeout retry前旧actor已kill且旧refs已drop；
- diamond IdentityKey gather 反例；
- annotation arity 与 runtime shape tests；
- `RelationResult[Outputs]`、named roles 和 M:N key tests；
- source `provided_keys` 与 relation `keyed` 类型隔离；
- list column slice/take/concat/collect ordering；
- stale token、timeout、actor replacement 和 finite split bound；
- published-ref whole restart、source re-put、global source failure和delivered-result loss；
- terminal writer replay idempotence fixture；
- top-level empty 与 graph-internal empty 区分；
- RunStream exhaustion、early close、poison 和 cleanup；
- Executor prepare rollback、close idempotence 和 caller-owned Ray；
- metadata/key-byte 与 sparse transfer amplification metrics。

