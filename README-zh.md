<div align="center">
  <img src="https://raw.githubusercontent.com/OpenDCAI/RayOrch-doc/main/docs/.vuepress/public/rayorch-mark.svg" width="104" alt="RayOrch logo" />

# RayOrch

**数据一旦就绪，立即运行下一阶段。**

基于 Ray、面向多阶段多模型 AI 负载的完成驱动数据流编排框架。

[![GitHub Stars](https://img.shields.io/github/stars/OpenDCAI/RayOrch?style=social)](https://github.com/OpenDCAI/RayOrch)
[![CI](https://img.shields.io/github/actions/workflow/status/OpenDCAI/RayOrch/ci.yml?label=CI)](https://github.com/OpenDCAI/RayOrch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rayorch)](https://pypi.org/project/rayorch/)
[![Python](https://img.shields.io/pypi/pyversions/rayorch)](https://pypi.org/project/rayorch/)
[![License](https://img.shields.io/github/license/OpenDCAI/RayOrch)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-English-5b5bd6)](https://opendcai.github.io/RayOrch-doc/en/)
[![中文文档](https://img.shields.io/badge/docs-中文-28a745)](https://opendcai.github.io/RayOrch-doc/zh/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18703-b31b1b.svg)](https://arxiv.org/abs/2609.18703)

[完整文档](https://opendcai.github.io/RayOrch-doc/zh/) · [快速上手](https://opendcai.github.io/RayOrch-doc/zh/guide/first-pipeline.html) · [Benchmarks](https://opendcai.github.io/RayOrch-doc/zh/benchmarks/) · [API 参考](https://opendcai.github.io/RayOrch-doc/zh/api/)

[English](README.md) | [简体中文](README-zh.md)

</div>

---

## RayOrch 是什么？

RayOrch 是一个基于 Ray、面向大规模模型托管多模态数据流的轻量编程与执行层。用户可以用普通 Python UDF 描述 CPU/GPU 流水线，RayOrch 负责持久化模型 Actor、阶段级资源配置、跨输入组批、多机执行，以及 `1 → M → 1` 数据流的保序重建。

典型负载包括 PDF 理解、视频处理、多模型视觉和多阶段 LLM 推理。RayOrch 保持编程模型显式且精简：**UDF 编写业务逻辑，`Pipeline.forward()` 描述依赖关系，RayOrch 在不丢失数据血缘的前提下调度每一个就绪任务。**

## 工作机理

```mermaid
flowchart LR
    Pipeline["Pipeline.forward()"] --> Graph["静态数据流<br/>依赖 + 基数关系"]
    Graph --> Ready["完成驱动的<br/>READY 调度"]
    Ready --> Actors["持久化 Ray Actor<br/>资源 + 组批"]
    Actors --> Lineage["基于血缘的<br/>保序重建"]
    Lineage --> Result["RunResult"]
```

- **显式数据流：** `RayModule` 声明可执行阶段，`F.expand()` 和 `F.reduce()` 在 Pipeline 中明确表达 `1 → M → 1` 关系。
- **完成驱动执行：** 一个下游数据项只要自己的依赖就绪即可运行，不需要等待整个全局阶段结束。
- **高效模型托管：** 不同输入中的就绪项可以进入同一个持久化 CPU/GPU Actor 池组批，昂贵模型只需加载一次。
- **稳定数据血缘：** 物理执行可以乱序完成，但每个子项始终保留父级和位置，因此能够确定性地保序还原。

RayOrch 负责这些逻辑数据流语义，Ray 继续负责节点、资源、Actor、RPC 和对象传输。编译器、运行时和源码流程详见[框架设计](https://opendcai.github.io/RayOrch-doc/zh/architecture/)。

## 一条 Pipeline

下面是一类典型多模态负载：每个文档展开为数量不同的页面，不同文档的就绪页面可以共享模型批次，处理完成后再按照来源和原始顺序还原为文档结果。

```mermaid
flowchart LR
    Documents["文档"] --> Split["SplitPages<br/>CPU Actor"]
    Split --> Expand["F.expand"]
    Expand --> Pages["来自不同文档的<br/>独立页面"]
    Pages --> Model["ProcessPage<br/>持久化 Actor 池"]
    Model --> Reduce["F.reduce<br/>按父级和顺序归并"]
    Reduce --> Assemble["AssembleDocument"]
    Assemble --> Results["文档结果"]
```

```python
import rayorch as ro


class SplitPages:
    def run(self, documents):
        # 为每个输入文档返回一个有序页面列表。
        return [
            [f"{document['name']}:page-{page}" for page in range(document["pages"])]
            for document in documents
        ]


class ProcessPage:
    def run(self, pages):
        # 同一个批次可以包含来自不同文档的页面。
        return [page.upper() for page in pages]


class AssembleDocument:
    def run(self, page_groups):
        # F.reduce 已按原始文档和页面顺序完成重建。
        return [" | ".join(pages) for pages in page_groups]


class DocumentPipeline(ro.Pipeline):
    def __init__(self):
        self.split = ro.RayModule(SplitPages).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=1,
        )
        self.process = ro.RayModule(ProcessPage).ray_options(
            replicas=2,
            batch_size=3,
            num_cpus=1,  # GPU 模型阶段可改为 num_gpus=1。
        )
        self.assemble = ro.RayModule(AssembleDocument).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=1,
        )

    def forward(self, documents):
        page_groups = self.split(documents)
        pages = ro.F.expand(page_groups)           # 文档 → 页面
        processed_pages = self.process(pages)      # 页面独立进入调度
        ordered_groups = ro.F.reduce(processed_pages)  # 页面 → 文档
        return self.assemble(ordered_groups)


inputs = [
    {"name": "guide", "pages": 2},
    {"name": "paper", "pages": 3},
]
result = DocumentPipeline().run(inputs)
print(result.outputs)
```

安装并运行：

```bash
pip install rayorch
python your_pipeline.py
```

同一条 Pipeline 通过 `pipeline.run(inputs, address="auto")` 即可连接已有 Ray 集群。将上面的三个 UDF 替换为 PDF 渲染与 OCR、视频解码与 VLM 推理，或其他批处理 Python/模型逻辑，整体数据流结构不需要改变。

## 文档

README 只展示核心编程模型和工作机理；完整教程、架构与源码流程、分布式执行契约、Benchmark 系统、真实模型案例和 API 参考统一放在文档站：

- [安装](https://opendcai.github.io/RayOrch-doc/zh/guide/installation.html)
- [第一条 Pipeline](https://opendcai.github.io/RayOrch-doc/zh/guide/first-pipeline.html)
- [一对多与有序归并](https://opendcai.github.io/RayOrch-doc/zh/guide/fan-out-and-reduce.html)
- [框架设计](https://opendcai.github.io/RayOrch-doc/zh/architecture/)
- [多机与跨环境执行](https://opendcai.github.io/RayOrch-doc/zh/distributed/)
- [内置 Benchmarks](https://opendcai.github.io/RayOrch-doc/zh/benchmarks/)
- [API 参考](https://opendcai.github.io/RayOrch-doc/zh/api/)
- [论文与实验复现](https://opendcai.github.io/RayOrch-doc/zh/paper/)

## 引用 RayOrch

如果 RayOrch 对你的研究有所帮助，请引用 [RayOrch 论文](https://arxiv.org/abs/2609.18703)：

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

## 开源协议

RayOrch 基于 [Apache License 2.0](LICENSE) 开源。
