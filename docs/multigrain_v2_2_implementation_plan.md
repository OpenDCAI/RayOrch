# Multigrain V2.2 实现计划

状态：V2.2 唯一实施顺序。语义合同以
`multigrain_v2_2_architecture.md` 为准。

目标路径：

```text
rayorch/experimental/multigrain_v2/
```

旧 `rayorch/experimental/multigrain/` 保留为历史 prototype，不做兼容迁移，不从旧
runtime object、pickle、trace 或 private extension point 继承合同。

## 1. 实现原则

1. 先完成纯语义，再接 Ray。
2. 每个 phase 有可执行 exit gate；未通过不得进入下一 phase。
3. CompiledGraph 是 runtime 唯一静态输入。
4. PrimitiveKernel 是五类 primitive 唯一语义入口。
5. actors 不生成 identity/provenance。
6. list column 是首版唯一 value-column ABI。
7. Ray hidden retries 始终关闭。
8. 不为未来 distributed metadata、byte blocks 或 cross-batch batching提前建 abstraction。
9. public API 只在 top-level `__init__.py` 导出权威合同中的类型。
10. 每个源码模块只能依赖 package tree 中更低层，不允许环形 import 飞线。

## 2. 目标 package layout

```text
multigrain_v2/
├── __init__.py
├── errors.py
├── authoring.py
├── compiler.py
├── graph.py
├── identity.py
├── column.py
├── data.py
├── provenance.py
├── work.py
├── relation.py
├── result.py
├── trace.py
├── kernels/
│   ├── base.py
│   ├── dispatch.py
│   ├── map.py
│   ├── filter.py
│   ├── expand.py
│   ├── reduce.py
│   └── relate.py
├── runtime/
│   ├── manifest.py
│   ├── node.py
│   ├── recovery.py
│   ├── stream.py
│   └── executor.py
└── ray/
    ├── actor.py
    ├── placement.py
    └── transport.py
```

依赖方向：

```text
errors
-> identity / column
-> authoring / graph / provenance / relation
-> work
-> data
-> runtime manifest
-> kernels
-> compiler
-> runtime recovery
-> result/trace
-> ray transport/actor/placement
-> runtime node/executor/stream
-> public __init__
```

`runtime/manifest.py`只定义RpcAttemptToken、CustomRelationEvidence以及三个生命周期
records：PendingRpc/RpcManifest/AcceptedRpc；它只能依赖identity、work与Ray ObjectRef
类型。`ray/`不允许import concrete kernels；它只执行transport ABI。`kernels/`不允许
创建actor或调用`ray.get()`。

## 3. Phase 0：合同 fixtures 与 break tests

先建立不会依赖 Ray 的 fixtures：

```text
linear Map
Map multi-output
Filter all-false
Select fused mask + annotations
Expand zero/one/many children
Reduce complete-empty fiber
Reduce missing required member
Key Relate repeated keys
Custom Relate named parent indexes
diamond reorder/filter/suppression
zero-output terminal writer
multiple independent input groups
external path/index descriptor feed
```

同时建立`tests/experimental/multigrain_v2/reference_semantics/`。reference interpreter
MUST NOT import production canonical encoding、EntityId derivation、provenance constructors、
kernels或failure-policy helpers；import-guard test强制该边界。Ray attempt/stale/restart由
独立trace state-machine verifier验证，不塞进单线程semantic oracle。

先写必须失败的 tests：

- module instance 两个 callsites；
- module alias、unbound module、list/dict/custom container与nested Pipeline；
- Port参与bool/iteration/len/index；
- 第二次 `.pre_init()` 或 `.ray_options()`；
- 显式 annotation 无法解析；
- variable-length tuple return；
- runtime return arity/column type不符；
- source `provided_keys`长度不等、重复或unsupported key；
- relation value/key identity set 不同；
- mixed KeyedRole/plain roles；
- Custom Relate parents 缺 role、多 role、越界、长度不等；
- duplicate ordered parent tuple；
- physical zip 造成 diamond misalignment；
- stale token 被接受；
- generic exception 被静默变为 missing。

Exit gate：fixtures 的 expected identities、direct provenance、WorkUnits 和 failures 以手工
golden 固定，旧实现无法全部通过。

## 4. Phase 1：identity、column 与 graph

### 4.1 `errors.py`

只导出：

```python
class MultigrainError(Exception): ...
class CompileError(MultigrainError): ...
class ExecutionError(MultigrainError): ...
class BadRecordError(MultigrainError): ...
```

WorkUnit/Invocation-scoped internal failure kinds放在`work.py`，不加入public hierarchy。
CompileError固定`code/path`；ExecutionError固定`code/batch_id/node_id/tuple[str] causes`
和`from_summary()`；BadRecordError固定keyword-only `index: int | None`。错误对象shape用
public tests锁定。

### 4.2 `identity.py`

实现：

```text
BatchId / NodeId / PortId / DomainId / RowId / EntityId
MicrobatchAttempt
RowRef / IdentityKey
CanonicalValue
canonical_encode()
source/expand/relation entity derivation
```

canonical encoding 必须：

- type-sensitive；
- process-independent；
- 不使用 Python randomized `hash()`；
- 严格实现MGCV1 framed byte grammar；
- ULEB128最短编码与`frame=uleb128(length)+payload`；
- 只支持None/bool/int/finite-float/str/bytes/recursive tuple；
- Unicode NFC、float -0 normalization、non-finite rejection；
- 使用带domain tag的完整SHA-256；
- NodeId来自pipeline module/qualname/unique attribute path/primitive kind；
- Position/Provided/Expand/Relate IDs有独立golden vectors；
- provided key唯一性只比较MGCV1 bytes；Key Relate分组使用Python hash/equality；
- 拒绝list/dict/set/custom objects。

### 4.3 `column.py`

只支持 `list[T]`：

```python
def validate_column(value: object) -> list[object]: ...
def take(column: list[T], selector: slice | tuple[int, ...]) -> list[T]: ...
def concat(columns: Sequence[list[T]]) -> list[T]: ...
```

所有函数必须保持逻辑顺序，不接受 tuple/NumPy/Arrow 隐式转换。

### 4.4 `graph.py`

实现架构文档中的 frozen dataclasses：

```text
Source/SameAs/SubsetOf/ChildrenOf/AggregateOf/RelatedFrom derived PortRelations
Map/Filter/Expand/Reduce/Relate tagged OperationSpecs
CompiledPort / RelateRoleSpec / CompiledNode
CompiledInputGroup / CompiledGraph
relation_of(graph, port)
```

Exit gate：

- canonical encoding golden 跨进程一致；
- derived EntityId 不含 BatchId/physical data；
- graph dataclasses 可被 Ray cloudpickle；
- graph validation 不读取 Pipeline 或 AST。
- CompiledPort和serialized graph不保存PortRelation；
- relation_of对每类OperationSpec/output slot都有golden，并可选cache不参与graph equality；

## 5. Phase 2：authoring 与 compiler

### 5.1 `authoring.py`

实现：

```text
Port[T]
Pipeline
PrimitiveModule
Map / Filter / Expand / Reduce / Relate
Select fused facade -> FilterSpec(mode="select")
KeyedRole / keyed()
ProvidedInput / provided_keys()
ByRole / by_role()
RelationResult[Outputs]
UdfContext
OperatorFactory / ModuleConfig
```

public `UdfContext`只含opaque `invocation_key`和`item_keys`，不冻结hash或字符串格式。
tests必须验证同一semantic coordinate跨RPC/retry/split/whole restart稳定、不同coordinates
唯一、新BatchId不要求相同、Custom Relate为空；batch/node/work细节只进入internal trace。

配置状态：

```text
new recipe
-> optional one pre_init
-> optional one ray_options
-> trace one callsite
-> successful compile freezes
```

compile 失败必须回滚 temporary trace state，不冻结 recipe。

input authoring/admission固定为：

- 只有external feed可启动execution，不实现zero-input Source primitive；
- file path、manifest entry、row-group/index shard都只是普通source row values；
- 每个forward参数一个single-port CompiledInputGroup；
- single-input run允许list shorthand；
- multi-input run使用参数名到list/ProvidedInput的mapping；
- run_stream每个元素使用同一binding形态；
- AdmittedInputs是microbatch-scoped，强引用Python value lists和canonical entities直到
  result delivery，不保存ObjectRef；
- caller在result delivery前不得修改已admit list或row objects，Executor不deep-copy；
- source row为Ray ObjectRef时拒绝；
- PositionKeys/ProvidedInput属于microbatch admission，不进入CompiledGraph；
- 每个attempt重新ray.put并把source ValueShard/PortData放入该attempt RunArena。

`ModuleConfig.runtime_env`保存用户填写mapping的cloudpickle round-trip snapshot，不做
merge、排序或自定义FrozenJson转换。compiler同时probe UDF factory并将其round-trip
payload保存为bytes；CompiledGraph不得保留指向用户原始可变配置的引用。

symbolic trace合同：

- module discovery只递归Pipeline直接属性和tuple slots，path使用`attr[n]`；
- static config condition、tuple loop和helper function合法；
- 未调用module不进入graph，被调用module必须有唯一path且只调用一次；
- list/dict/custom container、nested Pipeline和forward内临时module拒绝；
- Port的bool/iteration/len/index抛`PORT_VALUE_NOT_AVAILABLE`；
- forward其他异常包装为`AUTHORING_TRACE_FAILED`并保留cause；
- 同一Port fan-out合法。

### 5.2 arity inference

建立一个独立纯函数：

```python
def infer_output_arity(
    run_fn: Callable[..., object],
    operation: OperationSpec,
) -> int:
    ...
```

测试必须覆盖：

```text
missing / Any / None
fixed tuple
NamedTuple as one value
bare tuple / tuple[T, ...] rejection
postponed string annotations
Annotated
RelationResult[list[T]]
RelationResult[tuple[list[A], list[B]]]
Select list[bool] / tuple[list[bool], annotation columns...]
unresolvable explicit forward reference
```

### 5.3 compiler

compiler 执行：

1. trace `Pipeline.forward()`；
2. 使用 `Signature.bind()` 校验 call args；
3. 检查 single callsite；
4. 将Select直接lower为单node `FilterSpec(mode="select")`；
5. 分配 NodeId、PortId、DomainId；
6. 构造唯一tagged OperationSpec，并用relation_of验证每个output；
7. 从RelateRoleSpec直接获得planning-key demand；
8. 推导 output arity；
9. 将reserved UdfContext参数降低为CompiledNode.accepts_context；
10. 调用每个 PrimitiveKernel.validate；
11. 生成detached、logically immutable CompiledGraph snapshot；
12. 仅在全部成功后冻结 recipes。

不从 assignment variable names 推导 semantic role；role name 只来自 UDF signature。
本Phase同时建立`kernels/base.py` protocol、registry，以及五个内建kernel的静态
`validate()`；plan/invoke/materialize_bundle留到Phase 4。该分阶段仅是开发顺序，不开放
半套kernel注册接口。

Exit gate：

- 固定pyright版本的`reveal_type` golden验证pre_init/call参数名best-effort补全；
- compiler 对全部非法 graph 给出稳定 CompileError code/message；
- 重复 compile 语义相同；
- compile后修改原args/kwargs/runtime_env对象不改变CompiledGraph snapshot；
- module attribute alias与非唯一callsite salt被拒绝；
- 直接attribute与递归tuple生成稳定callsite paths；static condition/loop golden稳定；
- `provided_keys` 与 `keyed` 运行时类型不同；
- CompiledGraph 包含 runtime 所需全部信息。

## 6. Phase 3：基础 Work types、PortData 与 publication model

先在`work.py`实现不依赖data/kernel的基础类型：

```text
InvocationRef
Row/Fiber/JoinKey/WholeInvocation coordinates
WorkUnitRef / WorkRange
FailureCode / PermanentlyMissing / Suppressed / TerminalFailure
```

再实现：

```text
ValueShard / ShardPosition / KeyIndex / PortData / OutputBundle / RunArena
SourceRows / AliasRows / ChildRows / AggregateRows / RelatedRows
parent_edges()
```

最后在`runtime/manifest.py`实现：

```text
RpcAttemptToken / CustomRelationEvidence
PendingRpc / RpcManifest / AcceptedRpc
```

规则：

- locator cardinality 等于 entities/provenance cardinality；
- key index 只在 compiled demand 存在时构建；
- whole attempt drop 后旧 arena 不可解析。

Exit gate：

- direct provenance golden 通过；
- parent_edges 不丢 direct parents；
- OutputBundle/RunArena可在不import kernels/runtime managers的前提下构造；
- manifest records只依赖identity/work/Ray ObjectRef且可cloudpickle；
- graph-internal empty PortData 合法。

## 7. Phase 4：Work layouts、invocation planning 与五类纯 PrimitiveKernel

### 7.1 `work.py` layouts 与 invocation planning

实现：

```text
RowAlignedLayout / FiberGroupedLayout
KeyRelationLayout / CustomRelationLayout
RpcShardTake / RpcPortSlice / RpcPayload
PlannedRpc(value_refs + payload)
InvocationPlan
```

同一Phase实现`assign_row_ids()`和`build_locator()`。accepted RPCs先按WorkRange排序；
manifest offsets必须为长度work_count+1、首项0的单调prefix sums；每个output column长度
等于末项；accepted ranges无重叠并覆盖全部settled-success WorkUnits；unit-local output
ordinal决定同一WorkUnit内顺序。随机completion order必须得到相同RowIds/locator。

`take_work_range()` 必须是唯一纯函数，并分别实现row slice、fiber CSR slice/rebase、
key tuple CSR slice/rebase和custom whole-invocation restriction。layout cardinality和
item context keys与architecture合同一致。切片后按surviving takes首次出现顺序compact
ObjectRefs并重写连续ref_slots；RpcPayload.context只在CompiledNode.accepts_context时存在，
Custom Relate item_keys始终为空。

### 7.2 kernels

每个 kernel 实现完整四方法：

```text
validate / plan / invoke / materialize_bundle
```

不能把 primitive-specific switch 放进 Executor、NodeRuntime 或 Ray actor。
`kernels/dispatch.py::invoke_rpc()`是唯一transport adapter：它按RpcPortSlice gather
Ray已解析的Python list shards，验证ref_slot/selector/layout边界，再把values、layout和
context传给对应`PrimitiveKernel.invoke()`。dispatcher不得实现primitive semantics、
identity、provenance或recovery；actor与concrete kernels不得重复解析RpcPayload。

纯测试矩阵：

- Map aligned gather；
- Map nested collection row + explicit downstream Expand；
- 所有multi-output primitive共享一个semantic layout，heterogeneous cardinality拒绝；
- Filter survivor/subset；
- Select单node UDF+subset、filtered annotations与完整direct parents；
- Expand offsets与child identity；
- Reduce typed-ancestor fiber closure、role-major provenance、complete-empty与requiredness；
- Key Relate Python-equality KeyIndex、M:N tuples、unmatched keys与unhashable rejection；
- Custom Relate named parents、RelationResult、多输出；
- zero-output仅允许Map/Reduce/Relate，Filter/Expand拒绝；
- sparse fail-closed suppression。

Exit gate：

- reference interpreter 与 kernels 在所有 fixtures 上 keyed-equal；
- deterministic fixtures中randomized shard/order/batch changes不改变value/identity/provenance；
- nondeterministic fixtures只验证retry结果生效、旧timeout refs被drop与attempt isolation；
- duplicate relation tuple、offset mismatch、missing failure explanation 全部拒绝。
- actor-side `KernelInvokeResult(columns, offsets, evidence)` normalization逐primitive通过。

## 8. Phase 5：Ray transport spike

只验证 transport，不接完整 scheduler。

最小 actor：

```python
@ray.remote(max_restarts=0, max_task_retries=0)
class SpikeActor:
    def run(self, token, payload, *columns):
        return manifest, output_0, output_1
```

验证：

- `num_returns=1+N`；
- `N=0`时actor直接返回RpcManifest，调用端将单个ObjectRef归一化为refs tuple；
- task-return refs 由 driver 持有；
- unique refs 作为顶层参数会在 actor 前解析；
- ref slots按operation input顺序、shard首次出现顺序稳定去重；
- actor 接收 Python list columns；
- actor只通过`kernels.dispatch.invoke_rpc`得到`KernelInvokeResult`；
- multi-shard/duplicate-ref/strided-selector fixtures证明唯一adapter正确gather且不调用
  `ray.get()`；
- manifest offsets是prefix sums且每个output长度等于末项；
- actor crash、timeout、ObjectLostError 的实际 Ray exception；
- 下游top-level dependency loss的exception cause chain与lost ref定位；
- `ray.put()` nested owner 反例；
- `max_task_retries=0` 不透明重建 actor task output；
- `max_restarts=0` 不自动重启 actor；
- placement group bundle 内显式 replacement 可创建。

Exit gate：形成版本锁定的 Ray behavior tests。若 Ray 行为与合同不符，停止实现并修改
transport，不在 semantic layer 增加补丁。

## 9. Phase 6：NodeRuntime 与最小 vertical slice

首条真实链路：

```text
PositionKeys input
-> Map
-> Map
-> ResultPort.collect()
```

实现：

```text
NodeRuntime actor pool
round-robin invocation queues
contiguous WorkRange batching
ray.wait driver loop
node-invocation barrier
atomic OutputBundle publication
```

必须证明：

- active RPC 不超过 replicas；
- actor mailbox 不持有 hidden work；
- driver 不 gather业务 intermediate columns；
- actor 不生成 EntityId/provenance；
- stale result 整体拒绝；
- ResultPort logical shard order稳定。

Exit gate：deterministic UDF下改变replicas、batch_size、actor completion order后，
输出/identity/provenance保持keyed-equal；nondeterministic UDF不比较payload，只验证
同一WorkRange最多一个active RPC、retry结果生效且单个bundle不混代。

## 10. Phase 7：完整 data path

按顺序接入：

1. Filter predicate、`Filter.by_mask()`与Select fused mode；
2. Expand；
3. Reduce；
4. Key Relate planning-key materialization；
5. Custom Relate；
6. zero-output Map/Reduce/Relate。

Filter gate：

- predicate UDF只允许一个input并返回一个list[bool]；
- bool元素使用exact type check；
- by-mask node无OperatorFactory；
- Select只有一个actor pool；UDF与subset在同一RPC完成，不发布内部mask/annotation shards；
- Select返回filtered inputs后接filtered annotations，所有outputs共享0/1 layout；
- 多输入Select的target outputs各自以对应input为identity anchor；所有annotation outputs
  固定以第一个input为anchor，其余inputs为controls，不保存额外anchor字段；
- multi-target返回顺序与targets一致；
- predicate/mask/select controls和targets exact IdentityKey sets；
- relation_of派生的SubsetOf含identity target与去重controls，materialized provenance包含
  全部direct parents。

Key Relate gate：

- 只 `ray.get` compiled planning key refs；
- 不复制 KeyEvidence 到 manifest；
- planning-key count与estimated serialized bytes纳入metrics；
- unexplained value/key identity set mismatch失败；explained missing先走closure matrix；
- repeated keys 形成正确笛卡尔积；
- role order来自 signature；
- relation values 保持 P2P。

Custom Relate gate：

- `RelationResult[Outputs]` arity；
- named parents 降低为 canonical role tuple；
- parent index/evidence validation；
- zero-output返回 None。

fail-closed matrix gate逐项覆盖Map、Filter、Expand、Reduce、Key Relate和Custom Relate；
planning-key未完成时的explained missing必须suppress whole Key Relate invocation，不能被
当作unmatched key。该路径使用InvocationRef-scoped TerminalFailure，不虚构JoinKey
WorkUnit。source ref loss必须进入whole-attempt restart/re-put。

## 11. Phase 8：publication 前恢复

实现顺序：

1. PendingRpc/RpcAttemptToken返回归属验证；
2. exact-range retry；
3. explicit actor replacement；
4. BadRecordError(index)；
5. no-index binary isolation；
6. generic exception finite split；
7. timeout kill/replace；
8. fail_fast/fail_closed；
9. TerminalFailure propagation。

必须固定 amplification bound。设原始 range 有 `n` 个 WorkUnits：

- generic UDF exception只给原始range额外`max_retries`次exact retry；
- isolation tree 每个 derived range 不重新获得 retry budget；
- singleton generic failure abort；
- 每个logical range execution固定最多一次internal infrastructure retry；
- logical RPC上界为`max_retries + 2n - 1`，physical RPC上界为其两倍；
- Row/Fiber/JoinKey/WholeInvocation closure内部不拆；
- BadRecord index按Map/Filter row、Expand parent、Reduce anchor、KeyRelate flat tuple定义；
- CustomRelate只允许index=None并定位WholeInvocation。

Exit gate：

- adversarial tests 不能无限 retry/split；
- healthy WorkUnits 只按算法要求重跑；
- generic exception 永不变为 missing；
- failures 恰好解释所有缺失 output。

## 12. Phase 9：whole-microbatch restart

实现内部：

```text
MicrobatchAttempt
AdmittedInputs
fixed batch_restart_budget = 1
attempt-scoped RunArena
```

触发器只包括：source、已接受intermediate或planning-key ref的确定ObjectLostError。
普通UDF、manifest contract或source iterator error不进入该路径。

流程：

```text
stale attempt RPC tokens
-> stop该batch新提交
-> drop old arena/bundles/locators
-> preserve BatchId
-> re-ray.put retained Python list inputs
-> create fresh attempt
-> rerun whole graph once
```

tests：

- nondeterministic Map shape/value变化不与旧 attempt 混用；
- diamond 两分支不跨 attempt；
- old attempt refs从wait set移除且永不进入acceptance；
- source serialization/Ray session failure保持global ExecutionError；
- second published loss aborted；
- terminal writer fixture通过UdfContext证明幂等 key跨attempt稳定；
- terminal writer ack-loss会重复调用，但fake idempotent sink不产生重复记录；
- writer已经产生副作用后batch aborted的fixture不声称自动回滚；
- accepted final ResultPort不主动probe；后续loss只在读取时报RESULT_UNAVAILABLE。

Exit gate：trace 可以唯一重建每个 attempt 的开始、废弃和最终 outcome。

## 13. Phase 10：Executor、RunStream 与 lifecycle

实现：

```text
ExecutorState
transactional placement-group prepare
run()
RunStream iterator/context manager/close
completion-order yield
count-bounded admission
session poison/close
caller-owned Ray
```

RunStream tests：

- full direct-for exhaustion 回到 PREPARED；
- run_stream创建时冻结一个failure_mode；
- `with RunStream` early break 确定性 cleanup；
- explicit `close()` 等价；
- direct-for early break保持RUNNING/active-stream登记，新run得到ACTIVE_STREAM；
- ACTIVE_STREAM rejection不poison，显式close后进入CLOSED；
- input iterator exception poison；
- KeyboardInterrupt cleanup 后原样传播；
- batch-local aborted yield 后 stream继续；
- global failure raise并poison；
- no input batches 正常结束；
- single root path作为一个source row正常驱动reader UDF；
- external descriptor iterator按max_inflight lazy admission；
- max_inflight 从不超限。

RunResult/empty tests：

- `run_stream([])`不产生RunResult并正常回到PREPARED；
- 所有input groups为空的一个microbatch返回`EMPTY_INPUT` aborted；
- 部分独立input groups为空仍进入graph执行；
- aborted outputs必须为空且error_summary非None；
- fail-closed succeeded可以有failures但error_summary为None；
- zero-output terminal writer succeeded时outputs为空且error_summary为None。

Lifetime tests：

- prepare失败完整回滚；
- active microbatch强引用AdmittedInputs，result delivery后释放；
- whole restart从同一AdmittedInputs重建source refs，不复用旧RunArena refs；
- 每个node replica固定placement-group bundle index；
- replacement复用原bundle，创建失败raise REPLACEMENT_FAILED并poison；
- close幂等；
- close不调用 ray.shutdown；
- 多 Executors 共享 Ray session；
- Executor close后 ResultPort可读；
- Ray shutdown后 collect抛 RESULT_UNAVAILABLE。

## 14. Phase 11：trace、metrics 与 experiment harness

实现版本化 JSONL events：

```text
admission / attempt_start / attempt_discard
rpc_submit / rpc_accept / stale_reject
retry / split / actor_replace / timeout
bundle_publish / batch_complete / batch_abort
stream_close / executor_poison / executor_close
fault_injected / object_lost / result_delivered
```

fault events必须含stable injection id、NodeId/WorkCoordinate logical occurrence、fault
boundary和lost-ref producing node/port/attempt，供state-machine verifier关联。

必备 metrics：

```text
active rows / direct edges / planning-key count and estimated serialized bytes
retained source payload bytes / unreleased ResultPort metadata
driver materialize CPU / join plan CPU / peak RSS
selected rows / unique source shard rows
selector density / row transfer amplification
requested and physically transferred bytes / network counters
retry work / recovered work / good-work preservation
```

默认不开 full lineage export。experiment/debug 配置只输出 schema-versioned metadata，
不写业务 payload。

artifact exit gate：

- public/synthetic workload不依赖私人CEPH；
- fault injection scripts覆盖fault matrix；
- configs、seeds、checksums和baseline tuning records版本化；
- 干净container中one-command smoke成功；
- raw metrics可重新生成论文图；
- controlled payload校准验证physical-byte/network counter误差；
- oracle import guard通过。

## 15. 全局 CI gates

实现完成必须同时通过：

1. 全部 public snippets 可执行或可静态解析；
2. 所有 Python prototype 和源码通过 formatter/linter/type check；
3. semantic reference oracle/property tests；
4. Ray behavior tests；
5. five-kernel conformance；
6. deterministic、WorkUnit-separable UDF的physical execution invariance；
7. finite recovery amplification；
8. whole-attempt isolation；
9. lifecycle/resource leak tests；
10. metadata/transfer metric completeness；
11. 禁止 import 旧 multigrain runtime types；
12. public `__all__` 不泄漏 internal types。

采用三级验证：

```text
每个PR:
  Python 3.10/3.11/3.12的pure/compiler tests
  minimum-supported与latest-validated Ray的单机transport/E2E tests
  formatter/linter/type check/public snippets/import guards

nightly:
  至少双Ray nodes
  P2P transfer、max_inflight isolation、actor crash/timeout/replacement
  whole-microbatch restart、placement-group cleanup、writer idempotency fixture

release:
  clean container one-command install/smoke
  单机与双节点通过
  public/synthetic workloads与artifact reproduction
  dependency lock、license、credential/private-path scan
```

Phase 5 transport spike通过后必须冻结`minimum supported Ray`和`latest validated Ray`，并将
项目依赖从裸`ray`改为已验证的版本范围；具体版本不得在行为测试前猜测。

resource leak硬门槛只检查Executor拥有的actors、placement groups、active streams和
pending RPCs在固定grace period后归零，并验证重复run/close不会使这些数量单调增长。
RSS只记录trend，不设环境敏感的首版绝对阈值。

PR性能测试只阻止数量级退化；固定workload的median/p95回归在nightly报告，持续超过20%
进入人工确认。测试逻辑自身不自动retry；CI基础设施失败最多重跑一次。同一Ray integration
test在近期nightly出现两次非基础设施失败即为release blocker。

## 16. 推荐提交切分

```text
1. add v2 identity, errors, column and graph contracts
2. add authoring compiler and arity inference
3. add PortData provenance and RowId locator
4. add WorkUnit layouts and pure kernels
5. validate Ray multi-return transport
6. add NodeRuntime and Map vertical slice
7. add Filter and Expand
8. add Reduce and Key Relate
9. add Custom Relate and zero-output writers
10. add pre-publication recovery
11. add whole-microbatch restart
12. add Executor RunStream lifecycle
13. add trace metrics and experiment harness
```

每个提交必须独立通过已有 gates；不提交不可运行的“大爆炸重构”。

## 17. Definition of Done

只有满足以下条件才称为 V2.2 implemented：

- 用户只看 architecture public API 即可写 pipeline；
- contributor 只看 architecture + implementation plan 即可定位每个类型和职责；
- 不需要回读 V2/V2.1 或聊天记录解释语义；
- deterministic UDF的randomized physical execution不改变keyed semantic result；
- failure injection 不产生 silent drop、mixed attempts 或无限恢复；
- Ray resources 在 success/aborted/exception/early-close 下均可回收；
- paper experiments 可以从 versioned trace 和 harness 完整复现。

