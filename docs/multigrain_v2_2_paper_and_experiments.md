# Multigrain V2.2 论文定位与实验合同

状态：V2.2 唯一研究叙事与评测合同。架构事实以
`multigrain_v2_2_architecture.md` 为准。

本文不保证论文录用。它定义达到 VLDB Scalable Data Science 基本配位所需的问题、
claims、形式化边界、实验和 artifact。

## 1. 一句话 thesis

> Multigrain 将 applied-ML dataflow 中不同基数变化原语编译为带稳定 identity 和
> typed provenance 的 semantic WorkUnits；对deterministic、cardinality/order-stable、
> WorkUnit-separable UDF，它保持物理分片、乱序与有限重试下的逻辑结果，对
> nondeterministic UDF则保证attempt隔离和无混代，并以可测量的集中metadata与
> shard-transfer成本保留健康工作。

论文主轴是 semantic recovery closure，不是“又一个 Ray actor pipeline”，也不是
cross-microbatch batching scheduler。

## 2. 研究问题

Applied-ML pipelines 同时包含：

```text
row-preserving transforms
selective filtering
one-to-many expansion
group/fiber aggregation
M:N relation construction
```

普通 row-only DAG runtime 难以统一回答：

1. 动态基数变化后，一条输出究竟是哪一个语义实体？
2. diamond branches 物理乱序后如何重新一一对齐？
3. 一个失败应重跑一行、一个parent closure、一个fiber、一个join partition，还是整批？
4. 如何隔离坏数据而不把系统错误静默当作missing？
5. 如何让 payload 保持P2P，同时由一个 authority验证identity/provenance？

核心研究问题：

> Primitive-derived WorkUnits 能否在动态基数 DAG 中同时保证 identity/provenance
> physical invariance 与安全故障闭包，并以可接受的 driver metadata 和 shard transfer
> 成本减少重算放大？

## 3. 与普通 Ray DAG/retry 的区别

Ray 提供：

- task/actor execution；
- ObjectRef dependency tracking；
- scheduling、object store 与故障信号；
- task-level lineage reconstruction。

Ray 不知道：

- Expand child 的父实体与 sibling ordinal；
- Reduce fiber 的完整成员闭包；
- Relate output 的 ordered role parents；
- Filter false 与 missing failure 的区别；
- diamond branches 是否仍属于同一 IdentityKey；
- 哪些 UDF exceptions 可以成为 PermanentlyMissing；
- 已发布 typed provenance 是否与透明重算值仍一致。

Multigrain 使用 Ray 作为 transport/resource substrate，但关闭 Ray hidden actor-task
reexecution，由 semantic coordinator 管理恢复。

related-work章节必须分别连接三条数据库研究脉络：

- scientific workflow与data provenance：比较direct typed provenance和通用lineage DAG；
- distributed dataflow recovery：比较task/stage lineage、checkpoint与semantic closure；
- ML data systems/applied-ML pipelines：比较动态基数、GPU actor residency和relation-aware
  authoring。

Ray/Ray Data只是实现substrate和工程baseline，不能替代上述数据库领域定位。

## 4. 可辩护贡献

### C1. Primitive-to-WorkUnit compilation

统一编译五类 primitive：

```text
Map/Filter/Expand -> Row/Parent WorkUnit
Reduce            -> Fiber WorkUnit
Key Relate        -> JoinKey WorkUnit
Custom Relate     -> WholeInvocation WorkUnit
```

同一 WorkUnit 定义同时驱动：

- payload gather；
- RPC batching；
- failure localization；
- retry/split边界；
- output identity；
- direct provenance；
- sparse downstream suppression。

### C2. Identity-safe typed provenance execution

系统用 `IdentityKey(BatchId, DomainId, EntityId)` 对齐逻辑实体，用 typed direct
provenance 表达：

```text
SourceRows / AliasRows / ChildRows / AggregateRows / RelatedRows
```

diamond merge 按 IdentityKey gather，不按物理位置 zip。actor、shard、completion order
和 attempt 不进入 identity。

### C3. Single-authority semantics over P2P payload

业务 values 通过 Ray task-return ObjectRefs P2P 传输；actor 只返回 raw offsets/evidence；
driver bundle materializer 是 EntityId、RowId、typed provenance 和 atomic OutputBundle 的唯一
authority。

显式 planning key columns 是唯一 intermediate materialization exception。

### C4. Bounded semantic recovery

publication 前恢复包括：

- exact WorkRange retry；
- actor replacement；
- BadRecordError localization；
- finite binary isolation；
- sparse TerminalFailure propagation。

published intermediate loss 不做 selective patch，而是 active microbatch whole restart，
避免 mixed generations。

## 5. 贡献收敛

论文正文 SHOULD 将贡献合并为三项：

1. semantic WorkUnit model 与 compilation；
2. identity/provenance invariance 与 relation-aware recovery；
3. Ray implementation、P2P/central metadata trade-off 和系统评测。

whole-microbatch restart 是 fault-boundary mechanism，不单独包装为主要创新。

## 6. 明确不主张

不声称：

- arbitrary Python program semantics；
- 所有 UDF deterministic；
- external exactly-once；
- post-yield result reconstruction；
- driver/checkpoint recovery；
- byte-bounded memory；
- byte-sparse ObjectRef transfer；
- distributed metadata；
- cross-microbatch co-batching；
- universal optimal scheduler；
- complete relational algebra；
- general streaming/window/state runtime；
- 比所有 Ray Data workload 更快。

terminal writer采用at-least-once invocation与用户幂等sink前提；实验只验证稳定key下
重复attempt不产生重复记录，不把外部事务、回滚或exactly-once包装为系统贡献。

## 7. 形式化对象

### 7.1 Port state

对 batch `b` 和 port `p`：

```text
P[b,p] = (D, E, V, Π)
```

其中：

- `D` 是 DomainId；
- `E = <e0, ..., en-1>` 是 EntityId sequence；
- `V` 是与 E 等长的 value column；
- `Π` 是与 E 对齐的 typed direct provenance。

逻辑观察使用 keyed map：

```text
K(P) = { (b, D, ei) -> (Vi, Πi) }
```

物理 shard layout、RowId locator 和完成顺序不进入 `K(P)`。

### 7.2 primitive denotation

Map：

```text
每个primary entity最多一个成功output；
output identity = primary identity。
```

Filter：

```text
输出是target identity set的合法subset；
false不产生failure。
```

Expand：

```text
parent e的第j个child identity =
H(callsite salt, parent DomainId, parent EntityId, j)；
合法child count可以为0。
```

Reduce：

```text
每个anchor a的closure包含a及其complete direct member fiber；
成功output identity = anchor identity。
```

Relate：

```text
每个relation row由ordered role-parent tuple r定义；
identity = H(callsite salt, ordered role (DomainId, EntityId) tuple)。
```

BatchId、attempt和physical information不进入derived EntityId。

### 7.3 WorkUnit partition

对 node invocation `I`，compiler/kernel 生成 ordered WorkUnits：

```text
W(I) = <w0, ..., wm-1>
```

每个 WorkUnit closure `C(w)` 包含计算该 WorkUnit 输出所需的全部 direct inputs。

WorkUnit-separable 前提：

```text
eval(w, C(w), peers) = eval(w, C(w))
```

即一个 WorkUnit 的业务结果不依赖同一 RPC 中有哪些 peer WorkUnits。

### 7.4 execution equivalence

两个 execution plans `x` 和 `y` 可以使用不同：

```text
replicas / WorkRange partition / shard layout
RPC batch size / completion order / finite retries
```

若最终成功/terminal outcomes 相同，则 keyed equality：

```text
K(Px) = K(Py)
```

ordered equality 只在存在显式 order evidence 时主张。

## 8. 形式化 claims

### M1. Compilation soundness

对通过 compiler/kernel validation 的 Pipeline，CompiledGraph 中每个 node 的 WorkUnit
partition、domain propagation、identity derivation 和 provenance constructor 与该
primitive denotation一致。

证明方式：对 topological node sequence 做归纳，五类 kernels 分情况证明。

### M2. Physical execution invariance

在以下前提下：

- canonical input identity相同；
- UDF对每个closure deterministic（包括value、cardinality和unit-local order）；
- UDF满足WorkUnit-separable；
- 每个 accepted WorkUnit使用一个完整closure；
- stale results被拒绝；
- generic/system errors不伪装为missing；

改变合法 physical partition/reorder/retry 不改变最终 keyed values、typed provenance 和
terminal failure explanation。

比较故障执行时必须使用相同semantic fault schedule：

```text
(NodeId, WorkCoordinate, logical occurrence, fault code)
```

fault injection不得依赖actor、RPC index或wall-clock timing。failure equality比较
normalized code、coordinate和cause DAG，不比较message文本；cause DAG来自versioned
trace，public FailureRecord只用于简化用户诊断。

对于 nondeterministic UDF，不主张不同physical plans或retries产生value/identity
equality。系统只保证：

- 同一WorkRange最多一个active RPC；timeout retry前旧actor已kill且旧refs已drop；
- 一个published bundle来自一个attempt内一致的accepted results；
- whole restart废弃旧attempt；
- downstream不会观察mixed generations。

### M3. Diamond alignment safety

对Map/Expand fan-in，primary IdentityKey set必须是每个required secondary set的subset；
Filter predicate/mask/select的controls与targets要求exact set。Select target outputs各自以
对应input为identity anchor，annotation outputs以第一input为anchor。满足该premise时
IdentityKey gather产生唯一完整对齐；物理zip不属于合法实现。

若 required secondary 的primary key因explained upstream failure缺失，kernel按primitive
closure生成Suppressed；仅因secondary合法Filter false缺失属于structural mismatch，
除非filtered branch本身被选为primary。无failure explanation的required缺失是
ContractViolation。

Key Relate在planning key index完成前无法安全构造JoinKey WorkUnits；此时explained
missing使用InvocationRef-scoped TerminalFailure保守suppress whole invocation，而不是
把未知key当作unmatched。index完成后才允许定位到JoinKey scope。

### M4. Recovery closure safety

publication 前：

- retry/split只重新执行所选WorkUnit closures；
- Fiber/JoinKey closure内部不拆；
- fail-closed只把显式BadRecordError变成PermanentlyMissing；
- recovery algorithm有限终止。

对含n个WorkUnits的原始range，generic isolation logical RPC调用数至多
`max_retries + 2n - 1`；每个logical call至多一次internal infrastructure retry。
BadRecordError(None)与显式index后的健康range replay logical调用数至多`2n - 1`。

publication 后：

- selective downstream patch不在claim内；
- active execution实际观察到的source/intermediate/planning-key loss触发整个microbatch
  attempt废弃与一次whole restart；
- final ResultPort不主动probe，正常delivery后的loss只在读取时报RESULT_UNAVAILABLE；
- 不混用旧/新attempt values、RowRefs或provenance。

### M5. Direct provenance adequacy

对受支持五类 primitives，Source/Alias/Child/Aggregate/Related direct tables足以恢复每个
output的direct semantic parents。Alias/Child按input role保存per-output parents，
Aggregate按member role保存CSR fibers，Related按relation role保存parents；Filter controls
也是direct dependencies，fused Select annotation保存全部UDF inputs且不虚构内部mask
Port。通过递归`parent_edges()`获取ancestor traversal。

不主张对任意用户定义relation language完备。

## 9. fault model

支持并注入：

- UDF BadRecordError；
- generic UDF exception；
- actor process crash；
- actor timeout/hang；
- Ray transport failure；
- stale late result；
- accepted intermediate ObjectRef loss；
- planning key ObjectRef loss；
- terminal writer retry；
- source ValueShard loss followed by retained-list re-put；
- stream early cancellation。

不支持：

- driver death；
- Ray cluster整体不可恢复；
- caller `ray.shutdown()` 后恢复ResultPort；
- checkpoint restart；
- arbitrary external transaction rollback。
- source reserialization failure作为global error而非batch recovery。

必须维护机器可读 fault matrix。每一项固定：

```text
fault code
injection point
pre-publication / node-published-active / result-delivered boundary
expected retry layer
expected batch status
expected Executor state
required trace assertions
```

matrix必须逐项覆盖本节支持列表，尤其包括transport failure、stale late result、
planning-key loss、writer执行后ack loss和second published-value loss。

## 10. reference oracle

必须实现独立、单线程、无Ray的 reference interpreter：

```text
canonical input admission
-> primitive denotation
-> identity derivation
-> direct provenance
-> failure policy
```

它位于独立test package，不能import production canonical encoding、EntityId derivation、
provenance constructors、PrimitiveKernel或failure-policy helpers；import guard和独立
golden vectors强制该边界。Ray retry/stale/restart另由trace state-machine verifier验证，
不让semantic interpreter模拟transport faults。

property test 生成：

- 合法随机 DAG；
- input domains/keys；
- zero/one/many cardinalities；
- repeated relation keys；
- branch reorder；
- injected failures；
- random physical shard/retry plans。

比较：

```text
keyed values
EntityIds
normalized direct provenance
TerminalFailure causes
batch status
```

fault runs同时由trace verifier比较normalized failure cause DAG和attempt boundaries。

## 11. workloads

### W1. Document parsing

```text
document paths
-> Expand pages
-> Map layout/OCR
-> Filter invalid blocks
-> Reduce document assembly
-> terminal writer
```

覆盖 nested cardinality、GPU replicas、fiber recovery 和真实外部写出。

### W2. Multimodal/video curation

```text
videos
-> Expand clips/frames
-> Map embeddings/scores
-> Filter candidates
-> Relate cross-modal entities
-> Reduce selected assets
```

覆盖大型payload、sparse selectors、diamond和relation fanout。

### W3. Synthetic relation stress

可控参数：

```text
rows / domains / expand fanout / relation roles
key cardinality / skew / duplicate rate
diamond depth / failure rate / selector density
```

用于极限曲线和形式化反例，不替代真实 workloads。

## 12. baselines

### B1. Original RayOrch

代表actor DAG、node barrier和whole-node处理，但没有V2.2 semantic WorkUnits/typed
provenance。

### B2. Row-only runtime

所有操作退化为row/shard粒度，Reduce/Relate失败时使用whole-node fallback。

### B3. Whole-microbatch recovery

任一failure重跑整batch。用于量化semantic recovery closure价值。

### B4. Ray Data

使用合理的 `map_batches`、actor pools、batch size和资源配置。只比较双方都能表达的
throughput/latency，不声称语义等价。

### B5. Ablated Multigrain

分别关闭：

- IdentityKey gather，使用physical zip；
- typed provenance，只存untyped edges；
- WorkUnit recovery，使用whole-node retry；
- P2P values，经driver gather；
- planning-key demand，actor-sideWholeInvocation join。

所有 baselines 使用相同模型、数据和hardware。实验分成两组：

- throughput组比较共同可表达的无故障data path，包含Ray Data；
- semantic-recovery组只比较能实现相同completeness/failure policy的systems，Ray Data
  不被强行包装成语义等价baseline。

## 13. 核心 experiments

### E1. Semantic correctness

问题：物理执行变化是否保持逻辑结果？

变化：

```text
replicas / batch_size / max_inflight
completion order / shard partition
retry/split pattern / actor replacement
```

报告：

- keyed value equality；
- EntityId equality；
- normalized provenance equality；
- failure explanation equality；
- oracle mismatches；
- silent-drop count。

必须包含 physical zip 会失败而 IdentityKey gather 成功的 diamond反例。
keyed value/identity equality只在deterministic、cardinality/order-stable UDF组报告；
nondeterministic组只验证retry结果生效、旧timeout refs不被接受、attempt isolation和无
内部mixed generations。

### E2. Recovery precision

对五类 primitive 注入相同 failure：

```text
Map row
Expand parent
Reduce fiber
Key Relate partition
Custom Relate invocation
```

比较 whole-node、whole-microbatch、exact-range、binary row isolation 和 semantic
WorkUnit isolation。

结果分三层报告，不能混为同一“恢复成功率”：

```text
explicit BadRecord + fail_closed -> 可成功完成并量化good-work preservation
transient infrastructure fault   -> 可恢复后比较额外成本
persistent generic singleton     -> 按合同aborted，只比较定位/重算放大
```

指标：

- recomputation amplification；
- good-work preservation；
- recovered/failed WorkUnits；
- latency to isolate；
- final completeness；
- false missing；
- total UDF invocations。

### E3. Whole-attempt restart

在不同node-bundle publication points注入active execution可观察的source、intermediate
或planning-key ObjectRef loss：

- 验证旧 attempt 永不与新 attempt混合；
- 验证 nondeterministic shape/value不会污染diamond；
- 测量整批重跑额外成本；
- 验证source ref可由retained list重建、Ray global failure仍raise、second loss正确aborted；
- 验证terminal writer idempotent fixture。
- 单独验证final ResultPort正常delivery后不主动probe，loss在读取时报告
  RESULT_UNAVAILABLE。

该实验是安全边界验证，不包装为常见failure性能优势。

### E4. Driver semantic metadata

sweep：

```text
rows / direct edges / key bytes
relation skew / max_inflight / diamond depth
```

报告：

- peak driver RSS；
- metadata bytes estimate；
- planning-key count与estimated serialized bytes；
- materialize CPU；
- join planning CPU；
- parent-edge traversal CPU；
- payload bytes绕过driver比例。

live metadata复杂度声明：

```text
O(
  active rows
  + direct edges
  + decoded planning-key objects and estimated serialized bytes
  + pending WorkRanges/ref state
  + retained active source payload bytes
  + unreleased ResultPort metadata
)
```

不声称byte-bounded。
P2P claim只针对actor-produced intermediate business values；source admission、
planning keys和final collect明确经过driver。

### E5. P2P transfer amplification

首版 selector 是逻辑 sparse，不保证byte-sparse。

sweep：

```text
producer shard rows
selector density
fanout consumers
value size
object locality
```

报告：

- selected rows；
- unique source shard rows；
- unique refs per RPC；
- row transfer amplification；
- requested/transferred bytes；
- network throughput；
- actor deserialize/gather CPU。

必须明确一行selector仍可能搬运整个ValueShard。
若无法通过Ray metrics、network counters或受控payload instrumentation可靠得到物理bytes，
论文必须删除byte-transfer效率claim；该测量是go/no-go gate，不是可选指标。

### E6. Concurrency model

分别改变：

- per-node replicas；
- node batch_size；
- Executor max_inflight。

证明三者职责正交：

```text
replicas增加node capacity
batch_size改变RPC packing
max_inflight改变pipeline overlap
```

报告 utilization、queue wait、stage throughput、end-to-end throughput、tail latency和
driver event-loop CPU。

### E7. Lifecycle

长时间stream覆盖：

- success/aborted混合；
- actor replacements；
- early RunStream close；
- input iterator exception；
- repeated Executor create/close；
- multiple Executors共享Ray；
- ResultPort在Executor close后读取。

报告资源泄漏、object-store occupancy、actor/placement-group count和cleanup latency。

### E8. End-to-end value

在W1/W2上回答：

- 相比Original RayOrch，是否降低failure重算；
- 相比whole-microbatch baseline，是否保留更多good work；
- 语义metadata开销是否可接受；
- 相比Ray Data，共同可表达部分的throughput差距；
- 哪些relation-aware能力是baselines无法原生提供的。

受SDS正文篇幅限制，正文将E1/E3归入semantic correctness，E2归入recovery precision，
E4/E5/E6归入cost envelope；lifecycle、完整fault matrix和额外scale曲线进入补充材料。

## 14. ablations

至少包含：

```text
- WorkUnit-specific closure
- binary isolation
- fail-closed
- typed provenance
- IdentityKey gather
- P2P values
- planning-key driver materialization
- node-invocation barrier
```

每项 ablation 必须说明改变的是 correctness、recovery precision还是performance；不能把
非法实现作为性能优化后仍声称等价。

## 15. metrics schema

Performance：

```text
throughput / p50,p95,p99 latency
actor utilization / queue wait / driver CPU
object-store and network bytes
```

Recovery：

```text
attempts / retries / splits / replacements
recomputed WorkUnits
good-work preservation
recomputation amplification
stale rejections / aborted batches
whole microbatch restarts
```

Semantics：

```text
entities / direct edges / key bytes
oracle mismatches
diamond alignment failures
PermanentlyMissing / Suppressed
unexplained missing count
```

Transfer：

```text
selected rows / source shard rows
selector density / unique refs
row and byte amplification
```

统计协议必须预先固定：

- seeds；
- warmup；
- measured repetitions；
- confidence intervals；
- outlier policy；
- baseline tuning budget；
- cluster/model/data versions。

## 16. artifact

开源 artifact 必须包含：

- reference interpreter；
- randomized DAG/property generators；
- fault injector；
- baseline adapters；
- versioned trace/event schema；
- correctness comparator；
- workload configs；
- fixed public/synthetic datasets；
- model/version download说明；
- container/environment lock与经过transport behavior tests验证的Ray精确版本；
- 单机smoke和至少双Ray node的P2P/fault/lifecycle reproduction；
- raw metrics与绘图脚本；
- baseline tuning records；
- one-command smoke与reproduction入口。

artifact 不依赖私人 CEPH 路径，不保存业务 payload，不要求访问内部聊天或设计计划。

## 17. VLDB SDS 基本配位判断

V2.2 与 Scalable Data Science 的基本配位来自：

- 面向真实 applied-ML data pipelines；
- 解决动态基数、关系型数据和故障恢复的系统问题；
- 有明确新机制，而非仅API封装；
- 有形式化correctness边界；
- 有Ray上的分布式实现；
- 有真实workloads、baselines、ablation和scale/failure实验；
- 有可复现artifact。

基本配位不等于竞争力充分。最主要 rejection risks：

1. WorkUnit被评为常规task partition，未证明semantic closure价值；
2. provenance被评为metadata bookkeeping，未展示diamond/correctness反例；
3. driver metadata先成为瓶颈；
4. sparse selector产生严重transfer amplification；
5.只完成synthetic workload；
6. baselines调优不公平；
7.形式化claim与实现fault model不一致。

## 18. go/no-go

进入投稿写作前必须满足：

- V2.2 architecture invariants 全部有tests；
- reference oracle与deterministic、WorkUnit-separable UDF的randomized physical
  invariance通过；
- W1/W2至少两个真实end-to-end workloads；
- recovery precision在至少Reduce和Relate上显著优于whole-batch baseline；
- driver metadata和transfer amplification有完整曲线；
- Ray Data/Original RayOrch baselines公平调优；
- artifact在干净环境可运行；
- 所有claims能由定理、实验或明确non-goal支撑。

若只能展示“persistent actors + retry”，则 no-go；若 semantic WorkUnit closure、
identity-safe diamond和typed provenance在真实pipeline中同时展示correctness与恢复收益，
则达到VLDB SDS基本投稿条件。

