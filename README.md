<div align="center">
  <img src="https://raw.githubusercontent.com/OpenDCAI/RayOrch-doc/main/docs/.vuepress/public/rayorch-mark.svg" width="104" alt="RayOrch logo" />

# RayOrch

**Run every stage as soon as its data is ready.**

Completion-driven dataflow orchestration for multi-stage, multi-model AI workloads on Ray.

[![GitHub Stars](https://img.shields.io/github/stars/OpenDCAI/RayOrch?style=social)](https://github.com/OpenDCAI/RayOrch)
[![CI](https://img.shields.io/github/actions/workflow/status/OpenDCAI/RayOrch/ci.yml?label=CI)](https://github.com/OpenDCAI/RayOrch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rayorch)](https://pypi.org/project/rayorch/)
[![Python](https://img.shields.io/pypi/pyversions/rayorch)](https://pypi.org/project/rayorch/)
[![License](https://img.shields.io/github/license/OpenDCAI/RayOrch)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-English-5b5bd6)](https://opendcai.github.io/RayOrch-doc/en/)
[![中文文档](https://img.shields.io/badge/docs-中文-28a745)](https://opendcai.github.io/RayOrch-doc/zh/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18703-b31b1b.svg)](https://arxiv.org/abs/2609.18703)

[Documentation](https://opendcai.github.io/RayOrch-doc/en/) · [Quickstart](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html) · [Benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/) · [API Reference](https://opendcai.github.io/RayOrch-doc/en/api/)

[English](README.md) | [简体中文](README-zh.md)

</div>

---

## What is RayOrch?

RayOrch is a small programming and execution layer for large-scale, model-hosted multimodal dataflows on Ray. It lets ordinary Python UDFs describe pipeline-parallel CPU/GPU workloads while handling persistent model actors, per-stage resources, cross-input batching, multi-node execution, and ordered reconstruction of `1 → M → 1` dataflows.

Typical workloads include PDF understanding, video processing, multi-model vision, and multi-stage LLM inference. RayOrch keeps the programming model explicit—**UDFs contain business logic, `Pipeline.forward()` describes dependencies, and RayOrch schedules each ready item without losing its lineage.**

## How it works

```mermaid
flowchart LR
    Pipeline["Pipeline.forward()"] --> Graph["Static dataflow<br/>dependencies + cardinality"]
    Graph --> Ready["Completion-driven<br/>READY scheduling"]
    Ready --> Actors["Persistent Ray actors<br/>resources + batching"]
    Actors --> Lineage["Lineage-aware<br/>ordered reconstruction"]
    Lineage --> Result["RunResult"]
```

- **Explicit dataflow:** `RayModule` declares executable stages, while `F.expand()` and `F.reduce()` make `1 → M → 1` relationships visible in the Pipeline.
- **Completion-driven execution:** a downstream item runs as soon as its own dependencies are ready instead of waiting for an entire global stage.
- **Efficient model hosting:** ready items from different inputs can share batches on persistent CPU/GPU actor pools, so expensive models stay loaded.
- **Stable lineage:** physical execution may finish out of order, but every child retains its parent and position, allowing deterministic reconstruction.

RayOrch owns these logical dataflow semantics; Ray continues to provide nodes, resources, actors, RPC, and object transport. See [Framework Design](https://opendcai.github.io/RayOrch-doc/en/architecture/) for the compiler, runtime, and source-code walkthrough.

## One Pipeline

The example below represents a common multimodal workload: each document expands into a variable number of pages, ready pages from different documents can share model batches, and processed pages are restored to their original document and order.

```mermaid
flowchart LR
    Documents["Documents"] --> Split["SplitPages<br/>CPU actors"]
    Split --> Expand["F.expand"]
    Expand --> Pages["Independent pages<br/>from all documents"]
    Pages --> Model["ProcessPage<br/>persistent actor pool"]
    Model --> Reduce["F.reduce<br/>group by parent and order"]
    Reduce --> Assemble["AssembleDocument"]
    Assemble --> Results["Document results"]
```

```python
import rayorch as ro


class SplitPages:
    def run(self, documents):
        # Return one ordered page list for each input document.
        return [
            [f"{document['name']}:page-{page}" for page in range(document["pages"])]
            for document in documents
        ]


class ProcessPage:
    def run(self, pages):
        # One batch may contain pages from different documents.
        return [page.upper() for page in pages]


class AssembleDocument:
    def run(self, page_groups):
        # F.reduce restores each document's pages in source order.
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
            num_cpus=1,  # Use num_gpus=1 for a GPU model stage.
        )
        self.assemble = ro.RayModule(AssembleDocument).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=1,
        )

    def forward(self, documents):
        page_groups = self.split(documents)
        pages = ro.F.expand(page_groups)           # document → pages
        processed_pages = self.process(pages)      # schedule pages independently
        ordered_groups = ro.F.reduce(processed_pages)  # pages → document
        return self.assemble(ordered_groups)


inputs = [
    {"name": "guide", "pages": 2},
    {"name": "paper", "pages": 3},
]
result = DocumentPipeline().run(inputs)
print(result.outputs)
```

Install and run with:

```bash
pip install rayorch
python your_pipeline.py
```

The same Pipeline can connect to an existing Ray cluster with `pipeline.run(inputs, address="auto")`. Replace the three UDFs with PDF rendering and OCR, video decoding and VLM inference, or any other batched Python/model logic; the dataflow structure remains the same.

## Documentation

The README intentionally shows only the core model and mechanism. The complete guides, architecture and source-code walkthrough, distributed execution contract, Benchmark system, real-model cases, and API reference live in the documentation:

- [Installation](https://opendcai.github.io/RayOrch-doc/en/guide/installation.html)
- [Your First Pipeline](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html)
- [Fan-out and Ordered Reduction](https://opendcai.github.io/RayOrch-doc/en/guide/fan-out-and-reduce.html)
- [Framework Design](https://opendcai.github.io/RayOrch-doc/en/architecture/)
- [Multi-node and Cross-environment Execution](https://opendcai.github.io/RayOrch-doc/en/distributed/)
- [Built-in Benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/)
- [API Reference](https://opendcai.github.io/RayOrch-doc/en/api/)
- [Paper and Reproduction](https://opendcai.github.io/RayOrch-doc/en/paper/)

## Cite RayOrch

If RayOrch helps your research, please cite the [RayOrch paper](https://arxiv.org/abs/2609.18703):

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

## License

RayOrch is released under the [Apache License 2.0](LICENSE).
