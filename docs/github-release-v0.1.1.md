# RayOrch v0.1.1 Release Note

## 🇨🇳 中文

### 🚀 RayOrch v0.1.1：跨 DAG Stage 复用同一个 Actor Pool

RayOrch v0.1.1 是一个保持 0.1 编程接口兼容的小版本更新，新增了**跨 DAG Stage 共享 Actor Pool**的能力。对于需要连续调用同一套常驻模型的流水线，用户可以继续把不同阶段表达为独立的逻辑 Call，同时避免重复创建 Actor 和重复加载模型。

📄 **论文：** [RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation](https://arxiv.org/abs/2609.18703)

### ✨ 主要更新

#### 1. 同一个 `RayModule` 可以服务多个逻辑 Call

在 `Pipeline.forward()` 中重复调用同一个 `RayModule` 对象时，这些调用会保留各自独立的 DAG 依赖、batching 和恢复状态，但共享同一个物理 Actor Pool：

```python
import rayorch as ro


class ModelStack:
    def run(self, items, *, stage):
        if stage == "infer":
            return infer(items)
        return finish(items)


class SharedModelPipeline(ro.Pipeline):
    def __init__(self):
        self.model = ro.RayModule(ModelStack).ray_options(
            replicas=4,
            batch_size=16,
            num_gpus=1,
        )

    def forward(self, inputs):
        inferred = self.model(inputs, stage="infer")
        return self.model(inferred, stage="finish")
```

这里的 `Infer` 与 `Finish` 是两个独立的逻辑 Call，但四个 Actor 只初始化四套模型，而不是八套。请求状态仍通过 Port 显式传递，不依赖任务再次调度到同一个副本。

#### 2. 支持静态关键字参数

`RayModule` 调用现在可以同时接收动态 Port 和普通静态关键字参数，例如 `stage="infer"`。静态参数会进入编译后的 Call contract，并在每次执行时传给 UDF，不需要把控制信息包装进业务数据。

#### 3. 调度与指标按逻辑 Call、物理 Pool 分层

每个逻辑 Call 保留独立的 batch、恢复策略和调用指标；副本数、Ray 资源和 UDF 实例归属于共享 Pool。调度器会在共享同一 Pool 的 Call 之间轮转选择就绪工作，同时继续遵守输入 batch 隔离、失败恢复与血缘顺序语义。

### ⚡ Flash-MinerU 验证

Flash-MinerU 的 `v4-advanced-shared` 管线使用该能力，将 MinerU 4 Advanced 的 `Infer` 与 `Finish` 拆成两个可独立 rebatch 的阶段，同时复用每张 GPU 上常驻的 Layout、Two-Step VLM、MFR 和 OCR 模型栈。

在同一组 **368 个 PDF、7,072 页、4× NVIDIA H20** 上：

- Native MinerU：1,942.44 秒；
- Flash-MinerU `v4-advanced-local`：1,350.25 秒；
- Flash-MinerU `v4-advanced-shared`：1,234.60 秒；
- 相对 Native 执行阶段加速：**1.57×**；
- 相对未拆分的 local pipeline 进一步提升：**9.4%**；
- 368/368 个文档和 7,072/7,072 页完成，无页数或血缘错位；
- 平均 token Jaccard：**0.9945**，平均 token multiset F1：**0.9906**。

这些结果来自指定负载和硬件，不代表所有流水线都能获得相同收益。共享 Actor Pool 主要适合多个 DAG Stage 需要复用同一套昂贵常驻状态、且阶段之间仍存在独立 rebatching 机会的场景。

### ✅ 兼容性与验证

- `Pipeline`、`RayModule`、`F.expand`、`F.reduce`、`Executor` 和 `pipeline.run()` 的公开使用方式保持不变。
- 原有“一条逻辑 Call 对应一个 `RayModule`”的 Pipeline 无需修改。
- 同一个 `RayModule` 只调用一次时，执行行为与 v0.1.0 一致。
- `RunResult.actor_count` 继续提供物理 Actor 总数；共享 Pool 不会被多个逻辑 Call 重复计数。
- 完整测试集：**267 passed, 2 skipped**。
- shared actor 核心单元与 Executor 集成测试：**212 passed**。
- Flash-MinerU 完整 368 PDF 实验及端到端质量校验通过。

### 📦 安装

```bash
pip install rayorch==0.1.1
```

---

## 🇬🇧 English

### 🚀 RayOrch v0.1.1: reuse one Actor Pool across DAG stages

RayOrch v0.1.1 is a backward-compatible update to the 0.1 programming model that adds **shared actor pools across DAG stages**. Pipelines that invoke the same resident model stack in consecutive stages can keep those stages as separate logical Calls while avoiding duplicate actors and duplicate model initialization.

📄 **Paper:** [RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation](https://arxiv.org/abs/2609.18703)

### ✨ Highlights

#### 1. One `RayModule` can serve multiple logical Calls

Calling the same `RayModule` object more than once in `Pipeline.forward()` creates independent logical Calls with their own DAG dependencies, batching, and recovery state, backed by one physical Actor Pool:

```python
import rayorch as ro


class ModelStack:
    def run(self, items, *, stage):
        if stage == "infer":
            return infer(items)
        return finish(items)


class SharedModelPipeline(ro.Pipeline):
    def __init__(self):
        self.model = ro.RayModule(ModelStack).ray_options(
            replicas=4,
            batch_size=16,
            num_gpus=1,
        )

    def forward(self, inputs):
        inferred = self.model(inputs, stage="infer")
        return self.model(inferred, stage="finish")
```

`Infer` and `Finish` are separate logical Calls, but four actors initialize four model stacks rather than eight. Request state remains explicit in Ports and does not rely on the next task reaching the same replica.

#### 2. Static keyword arguments on Calls

A `RayModule` call may now combine dynamic Ports with ordinary static keyword arguments such as `stage="infer"`. Static values become part of the compiled Call contract and are passed to the UDF on every execution, without embedding control information in business records.

#### 3. Logical Call and physical Pool responsibilities are separated

Batching, recovery policy, and call metrics remain per logical Call. Replicas, Ray resources, and initialized UDF instances belong to the physical Pool. The scheduler rotates among ready Calls sharing a Pool while preserving input-batch isolation, recovery behavior, lineage, and ordering.

### ⚡ Validation with Flash-MinerU

Flash-MinerU's `v4-advanced-shared` pipeline uses this feature to separate MinerU 4 Advanced `Infer` and `Finish` into independently rebatchable stages while reusing the resident Layout, Two-Step VLM, MFR, and OCR stack on each GPU.

On the same **368 PDFs, 7,072 pages, and 4× NVIDIA H20**:

- Native MinerU: 1,942.44 seconds;
- Flash-MinerU `v4-advanced-local`: 1,350.25 seconds;
- Flash-MinerU `v4-advanced-shared`: 1,234.60 seconds;
- execution speedup over Native: **1.57×**;
- additional improvement over the unsplit local pipeline: **9.4%**;
- 368/368 documents and 7,072/7,072 pages completed with no page-count or lineage mismatch;
- mean token Jaccard: **0.9945**, mean token multiset F1: **0.9906**.

These measurements belong to the stated workload and hardware and are not universal performance guarantees. Shared actor pools are most useful when several DAG stages need the same expensive resident state but still expose independent rebatching opportunities.

### ✅ Compatibility and validation

- Public usage of `Pipeline`, `RayModule`, `F.expand`, `F.reduce`, `Executor`, and `pipeline.run()` is unchanged.
- Existing pipelines that use one `RayModule` per logical Call require no changes.
- A `RayModule` invoked once retains the v0.1.0 execution behavior.
- `RunResult.actor_count` continues to report the physical actor total without double-counting a Pool shared by several Calls.
- Full test suite: **267 passed, 2 skipped**.
- Shared-actor core unit and Executor integration suite: **212 passed**.
- The complete Flash-MinerU 368-PDF run and end-to-end quality validation passed.

### 📦 Installation

```bash
pip install rayorch==0.1.1
```

---

## 🔗 Links

- [Paper](https://arxiv.org/abs/2609.18703)
- [English documentation](https://opendcai.github.io/RayOrch-doc/en/)
- [中文文档](https://opendcai.github.io/RayOrch-doc/zh/)
- [Flash-MinerU](https://github.com/OpenDCAI/Flash-MinerU)
- [Full Changelog](https://github.com/OpenDCAI/RayOrch/compare/v0.1.0...v0.1.1)
