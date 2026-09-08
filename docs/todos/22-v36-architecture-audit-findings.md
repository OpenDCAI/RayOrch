# V3.6 架构审计待确认项

状态：**等待逐项 QA；本文件不授权修改源码**
记录日期：2026-08-09

本文只保存本轮源码与文档审计中有直接证据、值得用户决策的事项。教学文档继续描述已经成立的
合同，不把候选修复写成既定语义；每项确认后再单独实现、测试和提交。

---

## 1. 总览

| ID | 级别 | 结论 | 是否值得处理 |
| --- | --- | --- | --- |
| V36-A01 | P1 发布门禁 | 当前目录重组后的精确工作树尚未重新完成真实 Ray/性能证据闭环 | 发布或提交前必须明确门禁 |
| V36-A02 | P1 生命周期 | 已采用 fail-stop：执行开始后的终止失败会关闭 Executor 及其 actors | 已解决 |
| V36-A03 | P2 解耦 | 通用 `Worker.observe()` 仍探测 OCR/Table workload 字段名 | 值得做小型收口 |
| V36-A04 | P2 易用性 | RayModule symbolic call 尚不能诚实穿透 UDF 的 Python 泛型签名 | 值得继续设计，不宜立即实现 |

当前没有发现需要推翻 `LogicalProgram → ProgramAnalysis → RuntimePlan`、
`MicrobatchEngine + DispatchState + Executor` 或七种身份体系的证据。

---

## 2. V36-A01：当前工作树与已发布回归的证据范围不完全相同

### 源码与文档事实

[`2026-08-08 发布回归`](../experiments/multigrain_v3_6/2026-08-08_release_regression.md)明确记录：

- 核心实现对应 `cdd01d0`；
- benchmark adapter 修正随后收口在 `c596cef`；
- 该版本完成 MinerU 368 PDF、Docling 48 PDF 和两类视频 paired gate。

当前未提交工作树又把平铺的 compiler/executor/worker 等模块迁入：

```text
program/    static declaration, analysis, verification and lowering
runtime/    microbatch facts, transitions and dispatch
execution/ Ray adapter, Worker and Executor
```

这轮迁移大部分是 import/path 和注释重组，Ray-free unit tests 已通过；但现有实验报告并没有测试
“当前精确工作树”。尤其 Worker actor 的远程 import、序列化和 package boundary 只有真实 Ray
才能覆盖。

### Bad case

本地 unit test 可以成功 import driver 侧对象，但 Ray worker 进程使用旧模块路径、遗漏 package
export，或 benchmark adapter 在远端解析到不同类。此时算法完全没变，真实执行仍会在 actor
初始化或 RPC 反序列化阶段失败。

### 判断

这不是已经观察到的性能退化，而是**证据版本漂移**。历史 368 PDF 结果仍能证明 V3.6 主架构
没有系统性额外开销，却不能无条件证明之后每个未提交结构修改都通过了同一门禁。

### 建议

提交或宣称当前版本冻结前至少完成：

1. 当前工作树的全部 Ray-free tests；
2. 当前工作树的完整真实 Ray integration；
3. 一个短 paired MinerU smoke，确认远端 import、batch/RPC 统计与结果身份不变；
4. 只有 smoke 出现行为/时序差异，或 release policy 要求“精确提交全量认证”时，再重跑 full-368。

待 QA：目录重组是否要求 full-368 重跑，还是“完整真实 Ray integration + paired smoke”足以更新
证据边界？

---

## 3. V36-A02：终止失败后的 Executor fail-stop 合同（已解决）

### 源码事实

[`Executor.run()`](../../rayorch/experimental/multigrain_v3_6/execution/executor.py#L125-L257)
把 `active` 和 `pending_rpcs` 保存在单次调用的局部变量中。dispatch 时会先把 actor 标为
`busy=True`，并登记 `ObjectRef → _PendingRpc`；处理一个完成 ref 时，在对应 pending RPC 的
`finally` 中释放对应 actor。

如果以下任一路径抛出终止异常：

- Worker contract error；
- UDF recovery 决定 `ABORT`；
- infrastructure retry 耗尽；
- `commit_reports()` 拒绝完整 WorkerDispatchResult；
- `KeyboardInterrupt` 或其他异常退出；

当前实现不会把这些在途请求视为可复用的干净状态。异常逃离 active execution 时，`run()` 的
fail-stop 边界调用 `close()`，释放本 Executor 的 actors，并保留原始异常。

### Bad case

一张图有两个可并行 Call，两个 Call 各只有一个 actor：

1. Call A 和 Call B 都已有 pending RPC；
2. A 首先返回确定性合同错误，`run()` 抛出 `ExecutionError`；
3. B 的 ref 不再被 wait/get；
4. fail-stop 边界关闭 Executor 与其 actor slots；
5. 后续 `run()` 由 `_closed` 检查直接拒绝，不会以永久 busy actor 的形式 deadlock。

现有测试覆盖“成功后重复 run”，失败测试则都放在 `with Executor(...)` 内；context manager 退出会
`close()`，因此没有覆盖“捕获失败后复用仍打开的 Executor”。

### 为什么值得修

透明复用需要取消或 drain 所有 RPC、恢复所有 actor capacity、处理已经完成但未消费的结果，
并判断每个 microbatch Engine 是否还能安全重用，抽象成本很高。因此实现选择了较小且明确的
fail-stop 合同。

### 已采用方案

采用明确、轻量的 **Executor fail-stop**：

- source 数量、列长和参数范围等纯输入预检在进入执行态前完成；这类 preflight `ValueError`
  不污染 Executor；
- 一旦 admission/dispatch 已开始，`run()` 再以终止异常离开，该 Executor 自动关闭自己拥有的
  actors，并进入不可再次运行的 terminal 状态；
- 若 Ray runtime 是外部传入，只回收本 Executor 的 actors，不关闭外部 cluster；
- 原异常原样继续抛出，cleanup 失败不能覆盖首要异常；
- 下一次 `run()` 立即给出“previous run failed; create a new Executor”一类可读错误。

这与 microbatch Engine 的语义恢复没有混合：recovery policy 能处理的失败仍在同一次 `run()`
内恢复；只有已经决定向用户抛出的终止失败才关闭物理执行器。

对应回归验证终止失败后 Executor 已关闭，且不会遗留可被后续 run 误判为可用的 busy actor。

---

## 4. V36-A03：Worker observation 仍有 workload 名称飞线

### 源码事实

[`Worker.observe()`](../../rayorch/experimental/multigrain_v3_6/execution/worker.py#L231-L256)
先调用通用 `batch_audit()`，若不存在则继续探测：

```text
last_batch_audit
last_table_batch_audit
last_ocr_batch_audit
```

后两个名称来自 Docling/OCR workload。通用 execution 层知道它们，意味着新 workload 若使用
`last_layout_batch_audit`，开发者会再次修改 Worker，形成不断增长的名称分支。

### 影响

它只影响 best-effort diagnostics，不参与 Item/Grain 语义、调度或恢复，因此不是 P1。现有相关
workload 类已经提供统一 `batch_audit()` 方法，收口的兼容风险也较低。

### 建议方案

只保留一个显式 observation protocol：UDF 可选实现 `batch_audit() -> Mapping[str, scalar]`；
未实现则返回空 audit。Worker 只负责过滤可序列化的小型 scalar，不再猜测属性名。

需要新增测试：

- 有 `batch_audit()` 时正常冻结 scalar snapshot；
- 没有时返回空 audit；
- workload 风格属性不再被框架隐式识别；
- audit 抛错仍由 Executor 的 best-effort observation 边界隔离。

待 QA：确认删除三种属性 fallback，只保留 `batch_audit()`。

---

## 5. V36-A04：RayModule symbolic typing 是开放设计，不是状态机缺陷

完整分析见
[`21-v36-raymodule-symbolic-typing.md`](21-v36-raymodule-symbolic-typing.md)。当前矛盾是：

```text
UDF runtime: list[Image] -> list[Text]
authoring:   Port        -> Port
```

直接复制原版 `RayModule[InitP, RunP, R]` 会向类型检查器谎称 `Port` 是任意业务类型；Python
`ParamSpec` 也不能普遍表达“保留参数名，同时把每个参数类型映射为 Port”。

它影响 IDE 补全和开源易用性，不影响当前身份、状态 owner 或执行正确性。建议先维持诚实的
symbolic API 与运行时 signature binding，等候选方案能通过 TODO 中的验收矩阵后再实现。

待 QA：本轮是否继续保持 deferred，不为表面自动补全引入复杂 proxy/type plugin？

---

## 6. 已审视但当前不建议修改

### `MicrobatchEngine.commit_reports()` 较长

它目前按“完整 batch identity 预检 → barrier discovery → 全部 live output 预检 → mutation
frontier → 依赖序发布”分成清楚的阶段，而且只有一种 Worker report schema。此时抽出
`ReportValidator` 会增加 DTO 和跨文件跳转，却没有消除第二份实现。

只有将来出现第二种 report/backend schema，需要共享同一个纯 validation result 时才值得抽取。

### 989 行 Engine 本身

Engine 较长，但 canonical RuntimeState 的唯一写入、Fact FIFO、Effect interpreter 和 lineage
导航是同一个 microbatch 语义 owner。继续按函数拆文件但共享 `_state/_facts`，只会把一个状态机
拆成多个能互相摸内部表的对象，反而增加飞线。当前更合适的手段是分块注释与独立源码导读。

### 核心静态/动态分层

本轮没有发现 runtime 回读 Logical Origin、Worker 持有 RuntimePlan、Executor 直接修改 Item/
Entity tables，或 F.* 隐式创建 actor 的路径。`program/runtime/execution` 的依赖门禁和当前
Ray-free tests 也支持这一结论；因此不建议再增加新的 IR 或 adapter 层。
