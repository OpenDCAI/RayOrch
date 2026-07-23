# Multigrain V2.1 论文定位与实验设计

状态：已被 `multigrain_v2_2_paper_and_experiments.md` 取代的历史研究计划。

本文档仅保留 V2.1 研究演进记录。论文 claims、fault model、实验与 artifact 必须以
`multigrain_v2_2_paper_and_experiments.md` 为准。

## 1. 论文定位

推荐定位：

```text
Applied-ML Enabling Infrastructure
```

系统针对具有动态基数和多粒度关系的 ML/data-science pipeline：

```text
document -> pages -> blocks
asset -> frames/clips
items <-> relation tuples
children -> aggregate owner
```

这些 pipeline 不能被简单等长 row batch 完整描述。错误恢复也不能总是安全地以物理
shard 或整个 microbatch 为边界。

## 2. 核心 thesis

建议主张：

> 类型化关系合同将动态基数 ML dataflow 编译为语义闭合的 WorkUnits；这些
> WorkUnits 在 persistent actor pipeline 中统一驱动 identity-safe 对齐、批处理、
> provenance 物化、错误隔离、replay closure 和 node-level 原子发布。

论文范围是：

```text
有限 DAG
单 microbatch 内闭合
Map/Filter/Expand/Reduce/Relate
发布前恢复
```

不再把 cross-microbatch actor co-batching 写进首版核心 thesis。

## 3. 为什么不是“又一个 Ray actor pipeline”

原版 RayOrch 已经具有：

- persistent actors；
- pipeline tracing；
- replicas；
- microbatch inflight；
- synchronous `ray.wait`；
- completion-driven DAG overlap。

因此这些不能作为新贡献。

原版的局限是：

```text
一次 logical call fanout 全部 replicas
按等长 list row shard
driver 收集 stage payload
没有 relation/domain identity
没有 fiber/join-key replay closure
没有 causal sparse failure
```

V2.1 复用同一控制骨架，改变的是 semantic unit 和 data/recovery plane。这使原版
RayOrch 成为一个特别有价值的内部 baseline：两者控制结构接近，差异更容易归因于
Multigrain 机制，而不是“是否用了 persistent actors”。

## 4. 可辩护贡献

### C1. 类型化 relation-to-WorkUnit compilation

静态 primitive/relation 合同生成：

- Row WorkUnit；
- parent WorkUnit；
- anchor Fiber WorkUnit；
- JoinKey partition WorkUnit；
- WholeInvocation WorkUnit。

同一 WorkUnitRef 被 planning、batching、materialization、failure isolation 和 replay
共同使用。

需要证据：

- 五类 primitive 的形式化 denotation；
- 每类 WorkUnit closure 定义；
- compiler/validator rules；
- conditional compilation soundness；
- 随机合法 DAG property tests。

### C2. identity/provenance-aware execution

V2.1 将逻辑 identity 与物理 row/shard/actor 解耦：

- diamond/fan-in 按 IdentityKey gather；
- Expand/Reduce/Relate 具有明确 domain derivation；
- typed direct provenance 支持 fiber 和 relation closure；
- actor batch peers 不进入语义；
- actor completion/retry 不改变 identity。

需要证据：

- physical partition/reordering invariance；
- diamond alignment tests；
- provenance query 和 storage overhead；
- 相对于 row-id/path-string lineage 的表达能力案例。

### C3. P2P value shards 与 driver semantic materialization

actor 通过 Ray static multiple returns 直接产生 caller-owned value refs；driver 只读取
compact manifest，并根据 identity/provenance 规划下游 PortSlice。

机制价值：

- 不把大 payload 拉回 driver；
- actor 不复制 semantic metadata authority；
- Reduce/Relate 可以按 fiber/role selectors gather；
- accepted values 不依赖 actor-owned `ray.put` refs。

需要证据：

- driver/network/object-store traffic；
- driver CPU/RSS；
- payload size 和 fanout sweep；
- 与原版 driver collect 的对照。

### C4. bounded pre-publication isolation

错误发生时：

1. stale attempt 不可发布；
2. actor/infrastructure failure exact retry；
3. UDF failure可按完整 WorkUnit ranges 隔离；
4. explicit BadRecord 才能成为 fail-closed missing；
5. downstream absence 具有 causal Suppressed 解释；
6. node 只在全部 WorkUnits settle 后原子发布。

比较目标必须在相同 output-completeness policy 下：

- whole actor batch retry；
- whole microbatch retry；
- row-only binary isolation；
- relation-aware WorkUnit isolation。

### C5. physical-execution invariance

对 deterministic、WorkUnit-separable UDF：

- batch_size；
- replica assignment；
- physical shard boundaries；
- actor completion order；
- legal retry/split path

不改变：

- IdentityKeys；
- typed provenance；
- TerminalFailures；
- values 的 keyed equivalence；
- 有 order evidence 时的顺序。

对无 order evidence 的 relation 只主张 set equivalence。

对 nondeterministic UDF 只主张：

- active run 内原始 closure 可重跑；
- first accepted result；
- identity/provenance/cardinality 结构合法；
- atomic publication。

不主张 payload equality。

## 5. 明确排除的 claims

首版不得宣称：

- fragment-level inter-stage streaming；
- 消除 node-invocation barrier；
- cross-microbatch actor co-batching；
- post-publication selective replay；
- checkpoint/resume；
- driver failure recovery；
- byte-bounded admission；
- external writer exactly-once；
- nondeterministic VLM payload equality；
- unordered Relate 的全局 canonical order；
- 任意 custom Relate/不可拆 hot fiber 的可扩展性；
- Union/outer/window/state/iteration 的语言完备性；
- normal RunResult 中 durable full provenance。

## 6. fault model

支持：

- explicit BadRecordError；
- generic UDF exception；
- actor process death；
- RPC/transport failure；
- bounded straggler timeout；
- actor kill/replacement；
- stale/late manifest。

不支持：

- driver death；
- published value 所在 cluster node 丢失后的重建；
- source loss；
- checkpoint restart；
- arbitrary external-service transaction；
- node 发布后的 downstream patch。

value ownership 必须明确：

- actor 不返回 `ray.put` refs；
- outputs 是 caller-owned actor-task direct returns；
- driver 保持 accepted refs 到全部 consumers 完成；
- terminal writer ack 后释放 payload；
- returned final refs 由 RunResult/caller 保持。

## 7. 形式化边界

### 7.1 语言覆盖命题

目标不是证明五类 primitive 对所有 dataflow 完备，而是证明：

> 对由五类 direct-parent contracts 构成的有限合法 DAG，compiler 能构造一个
> CompiledGraph，使每个 output 的 identity、direct provenance、cardinality 和
> failure closure 与 primitive denotation 一致。

明确反例：

- optional parent Union/outer join；
- cross-batch window；
- recursive graph；
- semantic sort；
- stateful external effects。

### 7.2 batch separability 前提

对每个 WorkUnit `w`：

```text
output(w)
```

不能依赖同一 actor RPC 中集合：

```text
batch_peers(w)
```

该前提保证 actor batch 是物理容器，不是语义 scope。

### 7.3 reordering invariance

证明和测试必须区分：

- identity-keyed equivalence；
- relation set equivalence；
- explicit order evidence 下的 ordered equality。

不得重新引入“所有输出都 canonical sort 后序列相等”的过强主张。

## 8. workloads

至少包含两个结构不同的公开 workload，加一个 synthetic sweep。

### W1. Document parsing

```text
PDF paths
  -> Expand pages
  -> GPU layout/OCR Map
  -> optional block/evidence Relate
  -> Reduce assemble
  -> zero-output terminal writer
```

覆盖：

- nested 1:N/N:1；
- GPU batching；
- page/document skew；
- bad page；
- incomplete document suppression；
- idempotent file writer；
- outputless long-running pipeline。

### W2. Multimodal/video curation

```text
asset
  -> Expand frames/clips
  -> VLM embedding/caption
  -> key/content Relate
  -> Reduce asset summary
  -> terminal writer
```

覆盖：

- 不同 fanout；
- M:N density；
- model latency variation；
- nondeterministic values；
- role-based grouping。

### W3. Synthetic relation workload

sweep：

- source microbatch size；
- `max_inflight`；
- replicas；
- actor `batch_size`；
- Expand fanout/depth；
- fiber size/skew；
- join-key skew；
- relation density；
- payload bytes；
- service-time tail；
- failure type/rate/locality；
- terminal writer 与 returned-port 两种结果形态。

## 9. baselines

### B1. 原版 RayOrch

使用：

- 相同 UDF/model；
- 相同 replicas；
- 相同资源；
- 原版 contiguous replica sharding；
- stage payload driver collect；
- 无 Multigrain semantic runtime。

该 baseline 用于回答：

> 在控制骨架相近时，relation-aware WorkUnit 和 P2P data plane 带来什么？

### B2. Row-only runtime

保留 row identity 和 binary isolation，但没有 Expand/Reduce/Relate closure。

### B3. Whole-microbatch recovery

相同 persistent actors/P2P transport，错误后重跑整个 microbatch。

### B4. Ray Data

使用相同模型、资源、warmup 和 tuning budget。

### B5. 相关 cardinality-changing ML dataflow system

优先选择可真实运行的最强相关系统，例如 Trident 或投稿时更合适的公开实现。

Spark 仅在 accelerator UDF 和 failure semantics 可公平配置时纳入。

禁止用自己模拟的“竞争系统”代替真实 baseline。

## 10. 核心 experiments

### E1. Semantic correctness

随机改变：

- batch_size；
- replicas；
- physical range partition；
- actor completion order；
- Python hash seed；
- retry/split tree；
- actor failure point。

deterministic UDF 比较：

- values keyed equivalence；
- EntityIds/IdentityKeys；
- normalized typed provenance；
- TerminalFailures；
- order-evidence-aware ordering。

nondeterministic UDF 只检查结构和 completeness。

### E2. Semantic metadata overhead

测量：

- identity materialization；
- provenance bytes/row 和 bytes/edge；
- parent/fiber/role query time；
- graph compile time；
- driver metadata CPU/RSS；
- 与 no-provenance 和 row-path lineage 对比。

### E3. P2P data plane

比较：

```text
原版 driver collect
vs
V2.1 manifest + ObjectRef routing
```

sweep payload size、fanout、nodes、consumer count。

报告：

- driver bytes；
- network bytes；
- object-store peak；
- serialization time；
- driver CPU/RSS；
- throughput/latency。

### E4. replicas、batch_size 与 max_inflight

分别 sweep，避免把三个概念混在一起：

- replicas：node actor capacity；
- batch_size：one RPC WorkUnit count；
- max_inflight：pipeline microbatch overlap。

报告 GPU utilization、batch fill、queue wait、stage overlap 和 end-to-end goodput。

不把 persistent actor 或 inflight 本身包装成新贡献。

### E5. Relation-aware recovery

注入：

- direct indexed BadRecord；
- unindexed BadRecord；
- generic exception；
- actor crash；
- timeout；
- Reduce fiber failure；
- Relate JoinKey failure。

比较：

- abort-only；
- whole batch retry；
- whole microbatch retry；
- row-only binary split；
- semantic WorkUnit split。

相同 failure/output policy 下报告：

- useful accepted work；
- redundant work；
- attempted WorkUnits；
- RPC count；
- isolation depth；
- replacement/model reload cost；
- recovery latency；
- final completeness；
- suppressed outputs。

### E6. Long-running lifecycle

重点使用 zero-output terminal writer，避免 consumer 持有结果影响结论。

报告：

- active RunArena count；
- driver/actor RSS；
- object-store usage；
- live refs；
- model actor stability；
- throughput over time；
- timeout/replacement 后是否恢复稳定态。

### E7. Scale

目标：

- 4–8 nodes；
- 尽可能达到 32+ accelerators；
- strong scaling；
- weak scaling；
- 多次 steady-state runs；
- confidence intervals。

若资源达不到目标，论文必须缩小 claim，而不是用单节点结果推断多节点。

## 11. ablations

至少包含：

- no typed provenance；
- row-only WorkUnits；
- WholeInvocation-only WorkUnits；
- driver payload gather；
- no identity-key diamond gather；
- no isolation；
- whole-batch retry；
- relation-unaware binary split；
- one vs multiple replicas；
- one vs multiple max_inflight；
- batch_size sweep；
- node barrier wait contribution；
- Reduce fiber skew；
- Relate join-key skew；
- timeout disabled vs enabled。

不再包含：

- StageBatcher；
- cross-microbatch co-batching；
- Local/Ray backend comparison；
- post-publication replay。

这些不是首版实现或主张。

## 12. metrics

### Performance

- end-to-end wall time；
- goodput；
- rows/parents/fibers/relation tuples per second；
- p50/p95/p99 latency；
- real GPU utilization；
- actor batch fill/distribution；
- RPC count；
- queue wait/service/materialization/publication wait；
- network/object-store bytes；
- driver/actor/object-store peak memory。

### Recovery

- attempts by type；
- accepted/retried/redundant WorkUnits；
- isolation depth/RPCs；
- stale result count；
- timeout/kill/replacement latency；
- PermanentlyMissing/Suppressed count；
- output completeness。

### Semantic metadata

- EntityId/provenance materialization time；
- bytes per row/edge；
- identity index build/query time；
- fiber/role lookup time；
- port cardinality；
- relation density。

不得把 proxy metric 错标为 GPU utilization、wall time、RSS 或 network bytes。

## 13. invariant-based CI gates

CI 优先使用不受机器速度影响的确定性 gate。

### Fairness

持续 READY 的 `K` 个 invocation queues、`R` 个 replicas：

```text
每个 invocation 在 ceil(K / R) 个 refill rounds 内获得 slot。
```

### Timeout state machine

fake clock 精确检查：

```text
deadline
-> token stale
-> actor invalid
-> replacement exactly once
-> late manifest rejected
```

真实 Ray timeout 只做 smoke/integration，不以严格毫秒阈值作为 correctness gate。

### Isolation bound

单个持续失败 WorkUnit：

- unindexed BadRecord：attempted WorkUnits `< 3n`；
- 一次 exact retry 后 generic isolation：`< 4n`；
- RPC 数不超过 `initial + retries + 2*ceil(log2(n))`。

### Lifetime

- active arenas `<= max_inflight`；
- terminal writer completion 后无 payload refs；
- stale refs 不发布；
- Executor close 后内部 runtime state 为零。

wall-clock、bytes 和 GPU 百分比先作为实验测量项，不预先编造 acceptance 数字。

## 14. artifact

公开 artifact 至少包含：

- public data manifests/download scripts；
- pinned container/dependencies/models；
- cluster launch/config；
- baseline tuning records；
- anonymous raw experiment events；
- plotting scripts；
- correctness comparator；
- fault definitions；
- smoke/full-scale reproduction guide。

核心结果不能依赖私人 CEPH 路径。

## 15. 主要 rejection risks

以下情况出现时，论文仍不具备 SDS 竞争力：

- 提升主要来自 persistent actor 或普通 inflight；
- WorkUnitRef 只是 datatype，没有驱动 batching/recovery；
- Reduce/Relate 在 claimed scale 仍由 driver gather payload；
- isolation baseline 使用不同 completeness policy；
- 只有一个 workload 或一个 node；
- node barrier/hot-key/selector transfer cost 被隐藏；
- nondeterministic VLM 被描述成 deterministic equivalence；
- unordered relation 被错误宣称全局有序；
- external writer 被错误宣称 exactly-once；
- actor-owned `ray.put` refs 破坏 fault model；
- competitor 只有模拟没有真实实现；
- artifact 无法复现主结果。

## 16. go/no-go

架构足以进入实现，不等于论文已经成立。

进入完整论文写作前必须证明：

1. relation contracts 确实产生有价值的语义执行边界；
2. WorkUnits 同时改善 identity alignment 和 failure isolation；
3. P2P data path 在真实 payload 下优于 driver gather；
4. semantic metadata 成本可接受；
5. deterministic 语义在合法物理扰动下保持；
6. 机制在多节点真实 workload 中仍成立；
7. 相对于 tuned baselines 的收益不是 persistent actor/inflight 假象。
