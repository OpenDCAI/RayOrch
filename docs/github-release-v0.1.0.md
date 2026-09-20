# RayOrch v0.1.0 Release Note

## 🇨🇳 中文

### 🚀 RayOrch v0.1.0：全新多粒度运行时的首个公开版本

RayOrch v0.1.0 是全新多粒度运行时的首个公开版本，面向 PDF 理解、视频处理、多模型视觉和多阶段 LLM 推理等大规模模型托管数据流水线。它解决的核心问题是：一个输入可能动态展开为数量不定的页面、帧或区域，GPU 需要跨输入共享批次以提高利用率，而系统仍必须保存每个子项的归属和顺序，在父项所需结果全部结束后立即组装，并将失败限制在正确的作用域内。

📄 **论文：** [RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation](https://arxiv.org/abs/2609.18703)（[PDF](https://arxiv.org/pdf/2609.18703)）

---

## 🔑 主要能力

### 🧩 精简、显式的编程模型

用户只需要编写普通批处理 UDF，通过 `RayModule` 配置副本、批大小、CPU/GPU 和运行环境，再在 `Pipeline.forward()` 中声明数据依赖：

```python
import rayorch as ro


class AddOne:
    def run(self, values):
        return [value + 1 for value in values]


class MyPipeline(ro.Pipeline):
    def __init__(self):
        self.add_one = ro.RayModule(AddOne).ray_options(
            replicas=2,
            batch_size=8,
        )

    def forward(self, values):
        return self.add_one(values)


result = MyPipeline().run(
    [1, 2, 3],
    input_batch_size=2,
    max_active_input_batches=2,
)
print(result.outputs)  # [2, 3, 4]
```

`pipeline.run(...)` 适合一次有限输入；需要多次调用并复用已启动 Actor 和已加载模型时，可以保留一个 `Executor`。兼容的函数式入口 `rayorch.run(pipeline, ...)` 仍然可用。

### 🌳 原生表达 `1 → M → 1` 多粒度数据流

`F.expand`、`F.filter`、`F.broadcast` 和 `F.reduce` 将展开、筛选、跨层依赖与有序归并直接表达在 Pipeline 中。运行时持续保存父子关系、成员集合、不可变序号和终态，因此执行批次可以混合来自不同 PDF 或视频的就绪子项，而结果仍会返回正确父项并按原顺序重建。

### ⚡ Completion-driven CPU/GPU 流水线执行

RayOrch 不要求整个输入批次在每一阶段同步前进；任何阶段只要依赖满足即可进入调度。持久 Actor 池让 CPU 预处理、GPU 推理和结果组装并行重叠，`input_batch_size` 控制单个输入生命周期的规模，`max_active_input_batches` 控制可重叠的输入批次数量。

### 🛡️ 明确的失败与恢复边界

v0.1.0 区分业务结果、逻辑 Grain 和物理执行尝试，支持基础设施故障重试、UDF 失败隔离、父级作用域抑制和显式 `OutputIssue`。失败重试不会改变子项身份或顺序，无关父项可以继续执行。

### 🌐 多机、多卡与跨环境 Stage

同一个 Pipeline 可以在本地 Ray 或现有 Ray 集群上运行。每个 `RayModule` 可以独立声明 CPU/GPU、资源标签和 Ray `runtime_env`，因此 SGLang、vLLM 或其他模型 Stage 可以运行在不同 Conda 环境中，而 Driver 不需要安装所有模型依赖。

### 📊 Lazy Benchmark 与 Ray Job 提交

Benchmark 基础设施与具体负载已经分离。内置案例采用清晰的 `udfs.py + pipeline.py + env.json + benchmark.py` 结构，覆盖 MinerU、文档拓扑、视频、多模态、YOLO → SAM、双 vLLM 和 SGLang → vLLM。Benchmark 支持配置输入数量、GPU 数量、批大小、模型和输入输出路径，可直接运行或通过 Ray Jobs 提交，并生成统一的吞吐、调度、资源与产物报告。

---

## 📈 论文结果

论文在 NVIDIA H20 集群上报告：MinerU 从 4 卡扩展到 64 卡获得 **15.14×** 加速，视频流水线从 8 卡扩展到 64 卡获得 **7.82×** 加速；在 MinerU 上，相比 Ray Data 和 Daft，端到端时间分别降低 **13.1%** 和 **29.0%**，在 Docling 上相比 Ray Data 降低 **16.0%**。这些数字来自论文规定的实验配置，不应被理解为任意模型、数据集或集群上的通用性能承诺；完整方法、实验设置和消融结果请参阅[论文](https://arxiv.org/abs/2609.18703)。

---

## ✅ 发布验证

- Python 3.11 / 3.12 GitHub CI 全部通过。
- 默认测试集：`234 passed, 2 skipped`；两个可选跨 Conda 测试单独启用后 `2 passed`。
- 两节点 Ray Cluster 集成测试通过；该测试在单台机器上创建两个 Ray 节点，不等同于物理多机验收。
- wheel 与 sdist 构建及 `twine check` 通过，并在源码目录之外安装 wheel 后完成真实本地 Ray `Pipeline.run()`。
- Flash-MinerU 迁移路径使用安装后的 RayOrch wheel 完成 1 卡和 2 卡 MinerU2.5 推理，双副本输出保持输入顺序。
- DataFlow 适配路径完成 9 项 CPU/Ray 集成测试，并验证多次调用复用同一个 warm `Executor`。

---

## 📦 安装

RayOrch v0.1.0 需要 Python 3.11 或更高版本：

```bash
pip install rayorch==0.1.0
```

从 `0.0.1` 升级的用户请阅读 [API Migration Guide](https://github.com/OpenDCAI/RayOrch/blob/main/docs/api_migration.md)。RayOrch 目前仍处于 Alpha 阶段，`0.1` 版本线优先保证编程模型简洁、运行时语义明确以及实际多模型负载可用；在 `1.0` 前，API 仍可能继续演进。

---

## 🔗 相关链接

- [论文 / Paper](https://arxiv.org/abs/2609.18703)
- [英文文档](https://opendcai.github.io/RayOrch-doc/en/)
- [中文文档](https://opendcai.github.io/RayOrch-doc/zh/)
- [Quickstart](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html)
- [Benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/)
- [完整变更 / Full Changelog](https://github.com/OpenDCAI/RayOrch/compare/v0.0.1...v0.1.0)

---

## 🇬🇧 English

### 🚀 RayOrch v0.1.0: the first public release of the new multi-grain runtime

RayOrch v0.1.0 is the first public release of the new multi-grain runtime for large-scale, model-hosted pipelines such as PDF understanding, video processing, multi-model vision, and multi-stage LLM inference. It addresses a recurring systems problem: one input may expand into an input-dependent number of pages, frames, or regions; GPUs should batch ready children across inputs for utilization; and the system must still preserve ownership and order, finalize each parent as soon as its required children become terminal, and contain failures within the correct scope.

📄 **Paper:** [RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation](https://arxiv.org/abs/2609.18703) ([PDF](https://arxiv.org/pdf/2609.18703))

---

## 🔑 Highlights

### 🧩 A small and explicit programming model

Users write ordinary batched UDFs, configure replicas, batching, CPU/GPU resources, and environments with `RayModule`, and declare dependencies in `Pipeline.forward()`:

```python
import rayorch as ro


class AddOne:
    def run(self, values):
        return [value + 1 for value in values]


class MyPipeline(ro.Pipeline):
    def __init__(self):
        self.add_one = ro.RayModule(AddOne).ray_options(
            replicas=2,
            batch_size=8,
        )

    def forward(self, values):
        return self.add_one(values)


result = MyPipeline().run(
    [1, 2, 3],
    input_batch_size=2,
    max_active_input_batches=2,
)
print(result.outputs)  # [2, 3, 4]
```

Use `pipeline.run(...)` for one finite execution. Keep an `Executor` when repeated calls should reuse already-started actors and loaded models. The compatible functional form `rayorch.run(pipeline, ...)` remains available.

### 🌳 Native `1 → M → 1` multi-grain dataflows

`F.expand`, `F.filter`, `F.broadcast`, and `F.reduce` express expansion, membership changes, cross-level dependencies, and ordered reconstruction directly in the Pipeline. The runtime preserves parent-child lineage, declared membership, immutable ordinals, and terminal states, allowing physical batches to mix ready children from different PDFs or videos without losing ownership or order.

### ⚡ Completion-driven CPU/GPU pipeline execution

RayOrch does not require an entire input batch to advance through every stage in lockstep. A stage becomes schedulable as soon as its dependencies are ready. Persistent actor pools overlap CPU preprocessing, GPU inference, and output assembly; `input_batch_size` controls one input lifecycle, while `max_active_input_batches` controls how many lifecycles may overlap.

### 🛡️ Explicit failure and recovery boundaries

v0.1.0 separates business results, logical Grains, and physical execution attempts. It supports infrastructure retries, UDF failure isolation, parent-scoped suppression, and explicit `OutputIssue` results. Retries do not change child identity or order, and unrelated parents can continue.

### 🌐 Multi-node, multi-GPU, and cross-environment stages

The same Pipeline can run on local Ray or connect to an existing Ray cluster. Every `RayModule` may declare its own CPU/GPU requirements, custom resources, and Ray `runtime_env`, allowing SGLang, vLLM, or other model stages to run in separate Conda environments without installing every model dependency in the driver.

### 📊 Lazy Benchmarks and Ray Job submission

Reusable Benchmark infrastructure is separated from workload implementations. Built-in cases follow an inspectable `udfs.py + pipeline.py + env.json + benchmark.py` layout and cover MinerU, document topology, video, multimodal processing, YOLO → SAM, dual-vLLM, and SGLang → vLLM. Benchmarks expose input count, GPU count, batching, model, and path configuration, run locally or through Ray Jobs, and emit standardized throughput, scheduling, resource, and artifact reports.

---

## 📈 Results reported in the paper

On NVIDIA H20 GPUs, the paper reports **15.14×** processing-time speedup when scaling MinerU from 4 to 64 GPUs and **7.82×** when scaling the video pipeline from 8 to 64 GPUs. It reports end-to-end time reductions of **13.1%** versus Ray Data and **29.0%** versus Daft on MinerU, and **16.0%** versus Ray Data on Docling. These results belong to the paper's specified workloads and experimental setup and are not universal performance guarantees for arbitrary models, datasets, or clusters. See the [paper](https://arxiv.org/abs/2609.18703) for the methodology, configurations, and ablations.

---

## ✅ Release validation

- The Python 3.11 / 3.12 GitHub CI matrix passed.
- Default suite: `234 passed, 2 skipped`; the two opt-in cross-Conda tests passed when enabled separately.
- A two-node Ray Cluster integration test passed. It creates two Ray nodes on one host and is not a physical multi-host acceptance run.
- Wheel and sdist builds passed `twine check`; the wheel was installed outside the source checkout and completed a real local-Ray `Pipeline.run()`.
- The Flash-MinerU migration path completed real MinerU2.5 inference with the installed RayOrch wheel on one and two GPUs, with ordered outputs across two replicas.
- The DataFlow integration path passed nine CPU/Ray tests and verified that repeated calls reuse one warm `Executor`.

---

## 📦 Installation

RayOrch v0.1.0 requires Python 3.11 or newer:

```bash
pip install rayorch==0.1.0
```

Users upgrading from `0.0.1` should read the [API Migration Guide](https://github.com/OpenDCAI/RayOrch/blob/main/docs/api_migration.md). RayOrch remains an Alpha project: the `0.1` line prioritizes a small authoring model, explicit runtime semantics, and practical multi-model workloads, while APIs may continue to evolve before `1.0`.

---

## 📝 引用 / Citation

If RayOrch is useful in your work, please cite:

```bibtex
@misc{ma2026rayorchprogrammingexecutinglineagecontrolled,
      title={RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation},
      author={Xiaochen Ma and Zimo Meng and Junzhu Liang and Youhe Jiang and Yue Cheng and Hao Liang and Bohan Zeng and Dengchun Li and Lu Ma and Zhengyang Zhao and Zhen Hao Wong and Runming He and Meiyi Qiang and Jiangtao Guan and Binhang Yuan and Wentao Zhang},
      year={2026},
      eprint={2609.18703},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2609.18703},
}
```

---

## 🔗 Links

- [Paper](https://arxiv.org/abs/2609.18703)
- [English documentation](https://opendcai.github.io/RayOrch-doc/en/)
- [中文文档](https://opendcai.github.io/RayOrch-doc/zh/)
- [Quickstart](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html)
- [Benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/)
- [Full Changelog](https://github.com/OpenDCAI/RayOrch/compare/v0.0.1...v0.1.0)
