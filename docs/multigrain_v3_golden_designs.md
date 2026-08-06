# MultiGrain V3 Golden Designs

> 这是一份设计记忆清单，不替代各版本架构文档。它只回答：每个版本留下了什么，以及
> V4 最终不能漏掉什么。实现变化时更新链接和结论，不在这里复制实现细节。

## 1. 不应丢失的设计初心

- **语义与物理解耦**：Port、Domain、Entity、Item、Grain 的逻辑含义不由 actor、batch、
  ObjectRef、dispatch 或完成顺序定义。
- **计算与结构解耦**：只有用户 Call 执行计算；expand、reduce、broadcast、filter 是结构
  关系，不能因为实现方便就永久变成 adapter actor/RPC。
- **单写者与原子提交**：worker 只返回事实，runtime 是语义状态唯一写入口；一个 attempt
  要么完整 commit，要么不可见。
- **动态结构必须有稳定身份**：fan-out、空 group、nested group、drop、failure 都能沿
  lineage 确定性恢复，不能用“长度恰好相等”推断对齐。
- **控制面有界且分层**：Arena admission、Stage batch、actor concurrency、outstanding RPC
  window 和资源副本各自有明确 owner。
- **性能优化不得改写语义**：fusion、elastic rebatching、prefetch 和 placement 只能改变
  physical plan，不能改变 logical result、failure attribution 或 lineage。

## 2. 版本演进索引

| 版本 | 必须记住的贡献 | 文档 | 实现 |
| --- | --- | --- | --- |
| V3 | General DAG；Item/Grain/lineage；Arena 单写者；ordered/nested Reduce；persistent Stage actor pool；elastic batching；局部 recovery | [架构导读](./multigrain_v3_architecture.md) | [multigrain_v3](../rayorch/experimental/multigrain_v3/) |
| V3.1 | 验证 RayModule + functional authoring；trace-only metadata 不得泄漏到 runtime；authoring sugar 必须无损 lowering | [V3.1](./multigrain_v3_1.md) | [multigrain_v3_1](../rayorch/experimental/multigrain_v3_1/) |
| V3.2 | 建立 Node/Port/Domain 正交模型；GroupValue 成为 Port layout；独立 expand 不隐式 zip；logical IR 与 physical lowering 分层 | [V3.2](./multigrain_v3_2.md) | [multigrain_v3_2](../rayorch/experimental/multigrain_v3_2/) |
| V3.3 | 独立 Port/Domain 内核；Call 是唯一计算、actor、RPC、Grain 边界；结构操作零 actor；CommitDelta 原子提交；四层模型定型 | [V3.3](./multigrain_v3_3.md) | [multigrain_v3_3](../rayorch/experimental/multigrain_v3_3/) |
| V3.4 | 保留 V3.3 语义，删除 Local/Ray 双执行路径；完整 Pipeline 统一使用一个 Ray Executor；组件仍可 Ray-free 单测 | [V3.4](./multigrain_v3_4.md) | [multigrain_v3_4](../rayorch/experimental/multigrain_v3_4/) |

历史原型可以退出，但表中对应的设计结论不能因目录退出而消失。

## 3. V3 实验留下的控制面结论

- 多 Arena 通过共享、独立的 Stage actor pools 形成 pipeline overlap；`max_inflight_arenas`
  只负责 admission，不能代替 Stage/actor backpressure。
- Stage `batch_size` 决定一个物理 RPC 的 grain 数；actor outstanding window 不改变 batch。
- `max_outstanding_per_actor` 统计执行中与 mailbox 中的全部 RPC，并满足
  `max_outstanding_per_actor >= actor_max_concurrency`。对 Docling 的长且不均匀任务，
  默认 `1/1` 的浅窗口能减少过早绑定、HOL blocking 和 actor failure blast radius。
- 跨 Arena round-robin 不属于必要语义。full368 消融未证明其吞吐收益，因此回到
  Arena-major；actor replicas 内部的 slot 轮转仍是 ExecutionPool 的物理策略。
- actor failure 必须按 worker generation 收口该代全部 RPC，并精确释放 credit；Arena
  只接收 failure event，不读取 transport pending 状态。

对应证据与实现：

- [Docling 调度优化记录](./experiments/multigrain_v3/2026-08-04_docling_optimization_debug.md)
- [Docling 直接模型 UDF 与 V2 batch 记录](./experiments/multigrain_v3/2026-08-05_docling_direct_v2.md)
- [V3 control plane](../README.md#multigrain-v3-control-plane)
- [outstanding credit 实现](../rayorch/experimental/multigrain_v3/execution.py)

## 4. 当前设计前沿：细粒度 Lineage

- 细粒度 Entity/Item/Grain lineage 是语义底线；退化成 block/parent 级粗血缘会损失
  record failure attribution、最小 replay、Filter membership、nested Reduce 顺序和分支隔离。
- 当前 parent + ordinal 只为每个 child 保存一条父链接，不复制完整 ancestry。固定图深度下，
  metadata 随 Entity、Item、Grain 和显式 relation edge 数量线性增长；多轮 Expand 的乘积首先
  是真实业务基数，不是 lineage 路径重复。
- V3.4 保留现有表示，不预先引入 dense handle、range、column、bitmap、hash-cons 或仅为
  metadata 保险服务的 structural hard-limit 事务；它面向由 admission、backpressure 和 Arena
  生命周期约束的有限、可物化工作集。
- 后续版本先测量 bytes/entity、bytes/item、Python object count 与 metadata CPU；只有证据表明
  表示成为瓶颈时，才在不降低逻辑粒度的前提下使用 range/column/bitmap、稀疏例外和可选的
  lineage fingerprint 压缩物理状态。

## 5. V4 必须保留的核心优势

V4 设计和验收时逐项核对：

- [ ] RayModule + symbolic Port 的简洁 authoring，不把运行时细节暴露给用户。
- [ ] Call/Port/Domain/Entity/Item/Grain 稳定身份，以及逻辑引用与物理引用严格分离。
- [ ] Call 是唯一计算边界；结构操作默认零 actor、零 RPC、零 Grain。
- [ ] 显式 expand/reduce/broadcast/filter/optional 语义；禁止按长度隐式对齐独立 Domain。
- [ ] ordered、empty、nested、aligned group 与多输入/多输出、分支、汇合 General DAG。
- [ ] 单写者增量 runtime、CommitDelta/等价原子提交，以及 generation fencing。
- [ ] PRESENT、DROPPED、FAILED、SUPPRESSED、MISSING 等终态传播不混淆。
- [ ] grain/record 级 failure attribution、bounded retry、局部 replay，失败不污染健康 sibling。
- [ ] coarse ObjectRef blocks、persistent actor pools，以及 Worker ABI 不持有完整 DAG/runtime。
- [ ] elastic 与 parent-bound batching 都是可解释的 physical policy，不进入逻辑语义。
- [ ] 有界 multi-Arena overlap、Arena-major 调度和独立的 admission/batch/replica 控制面。
      replica 必须按 stage 独立调优；Docling V1 的正例是保持 Arena、window 与模型 batch
      不变，仅将 CPU 汇合 pools 从 3 扩到 4，而盲目扩大 inflight/Table batch/GPU replicas
      均被真实性能 gate 否决。
- [ ] per-Stage shallow outstanding window、actor concurrency 约束和 generation 级 credit 回收。
- [ ] 完整 Pipeline 只有一个生产执行路径；允许组件级 reference/unit test，但不复制第二套 runtime。
- [ ] workload UDF/DTO/kernel 不绑定 authoring API；从 V3 Pipeline 迁移到 RayModule + F.* 只替换图表达与 lowering。
- [ ] 模型 adapter 优先调用上游稳定 facade；独立 batch kernel 只替换 UDF，不以 private-field monkeypatch 改写数据流。
- [ ] 区分“batch 抽象正确”和“具体模型可晋升”：同模型 serial/batch exact 只证明 kernel；模型
      相对 golden 的质量与吞吐仍须独立 gate。Docling V1 clean batch 是双 gate 通过的正例，
      V2 是抽象通过、模型晋升否决的反例；证据见
      `experiments/multigrain_v3/2026-08-06_docling_v1_batch_kernel.md`。
- [ ] lineage/explain、结构与失败审计、batch fill/queue wait 等可观测性足以解释 correctness 和性能。
- [ ] V3/V3.3/V3.4、Ray Data、Native 的真实 workload 回归可比较，历史数据不冒充新版本结果。

V4 可以替换数据结构、调度算法和文件组织，但若删除上述任一项，必须在设计文档中明确
说明替代机制与证据，不能静默丢失。
