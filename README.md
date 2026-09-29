<div align="center">
  <img src="https://raw.githubusercontent.com/OpenDCAI/RayOrch-doc/main/docs/.vuepress/public/rayorch-mark.svg" width="104" alt="RayOrch logo" />

# RayOrch

**Write multimodal AI data pipelines like programs. Scale them across Ray.**

Run each stage as soon as its data is ready, while preserving lineage, order, and local failure boundaries.

[![GitHub Stars](https://img.shields.io/github/stars/OpenDCAI/RayOrch?style=social)](https://github.com/OpenDCAI/RayOrch)
[![CI](https://img.shields.io/github/actions/workflow/status/OpenDCAI/RayOrch/ci.yml?label=CI)](https://github.com/OpenDCAI/RayOrch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rayorch)](https://pypi.org/project/rayorch/)
[![Python](https://img.shields.io/pypi/pyversions/rayorch)](https://pypi.org/project/rayorch/)
[![License](https://img.shields.io/github/license/OpenDCAI/RayOrch)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18703-b31b1b.svg)](https://arxiv.org/abs/2609.18703)

[Documentation](https://opendcai.github.io/RayOrch-doc/en/) · [Quickstart](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html) · [Benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/) · [API Reference](https://opendcai.github.io/RayOrch-doc/en/api/)

[English](README.md) | [简体中文](README-zh.md)

</div>

## Why this exists

Many multimodal workloads have the same shape: **one input becomes a variable number of items, a model processes those items, and the items become one result again**.

```text
PDF ──► pages ──► OCR / VLM ──► document
video ──► frames ──► vision model ──► summary
```

This `1 → M → 1` pattern is where a flat batch API starts to lose information. A PDF with 2 pages and a PDF with 48 pages should share ready GPU work, but their pages must never be confused; a short document should finish without waiting for a long one; and a failed page should not silently invalidate unrelated inputs.

RayOrch makes that relationship part of the program. You write ordinary batched UDFs, declare `F.expand` and `F.reduce`, and let RayOrch retain ownership, order, readiness, and failure scope while Ray supplies actors, resources, and placement.

<p align="center">
  <img src="docs/assets/readme/paper-motivation.svg" width="100%" alt="Input-dependent fan-out, cross-input batching, and lineage-controlled continuation" />
</p>

The physical execution microbatch is temporary. It may mix children from different parents, but it never becomes the source of truth for membership, ordering, or completion.

## What you gain

- **More useful accelerator time:** ready children from different inputs can share a model batch, while independent parents continue as soon as their own work is complete.
- **Correct reconstruction by construction:** parent identity and child order stay in runtime-owned lineage instead of being rediscovered from a flat result table.
- **Local recovery:** retries and typed failures stay scoped to the logical call and its parent group.
- **A small authoring surface:** Python UDFs plus a few explicit structural operations; RayOrch does not replace Ray's actor, resource, or environment model.

The benchmark timeline below is from the paper's MinerU comparison. It shows why the distinction matters: RayOrch overlaps the stages while preserving the same document-level result contract.

<p align="center">
  <img src="docs/assets/readme/mineru-comparison.svg" width="100%" alt="MinerU cold end-to-end timeline comparing RayOrch, Ray Data, and Daft" />
</p>

The exact numbers depend on model, data, storage, and hardware. Use the [benchmark guide](https://opendcai.github.io/RayOrch-doc/en/benchmarks/) for reproducible runs and interpretation.

## Install and run

```bash
pip install rayorch
```

The smallest pipeline is ordinary Python:

```python
import rayorch as ro


class AddOne:
    def run(self, values):
        return [value + 1 for value in values]


class MyPipeline(ro.Pipeline):
    def __init__(self):
        self.first = ro.RayModule(AddOne).ray_options(replicas=2, batch_size=8)
        self.second = ro.RayModule(AddOne).ray_options(replicas=2, batch_size=8)

    def forward(self, values):
        return self.second(self.first(values))


result = MyPipeline().run(
    [1, 2, 3],
    input_batch_size=2,
    max_active_input_batches=2,
)
print(result.outputs)  # [3, 4, 5]
```

`forward()` is traced once to build a static graph; it does not execute the UDF. `pipeline.run(...)` manages a temporary `Executor`. Keep an `Executor` open when repeated runs should reuse already-started actors and loaded models.

## Programming model

<p align="center">
  <img src="docs/assets/readme/paper-program-execution.svg" width="100%" alt="A RayOrch program lowered into structural lineage and per-Call actor execution" />
</p>

Read the diagram from left to right through this concrete PDF parsing pipeline:

```python
import rayorch as ro
from rayorch import F
from rayorch.benchmarks.mineru.udfs import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)


class PdfPipeline(ro.Pipeline):
    def __init__(self, model_path, output_dir):
        # Python program: declare the stages and their physical policies.
        self.render = (
            ro.RayModule(MinerUPdfToPages)
            .pre_init(dpi=200)
            .ray_options(replicas=2, batch_size=1, num_cpus=1)
        )
        self.ocr = (
            ro.RayModule(MinerUVlmOcrPage)
            .pre_init(model=model_path, gpu_memory_utilization=0.8)
            .ray_options(replicas=2, batch_size=32, num_gpus=1, num_cpus=1)
        )
        self.metadata = ro.RayModule(PdfMetadata).ray_options(
            replicas=1, batch_size=32, num_cpus=1
        )
        self.assemble = (
            ro.RayModule(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(replicas=2, batch_size=4, num_cpus=1)
        )

    def forward(self, pdfs):
        # Structural control plane: PDF -> ordered page Entities (1:M).
        pages = F.expand(self.render(pdfs))
        contents = self.ocr(pages)       # Physical work is page-level (M).
        stems = self.metadata(pdfs)      # Metadata stays at document scope.

        # Structural control plane: pages -> ordered documents (M:1).
        grouped_contents, grouped_pages = F.reduce_aligned(
            contents, pages, members=contents
        )
        return self.assemble(grouped_contents, grouped_pages, stems)
```

The three layers in the figure correspond to this code:

1. **Python program.** `forward()` composes ordinary `RayModule` calls and `F` operations. It is traced once; the UDFs do not run while the graph is being built.
2. **Structural control plane.** `F.expand` records each PDF's ordered page membership. `F.reduce_aligned` records how page outputs and page records close back into the same document. These symbolic Ports carry lineage, order, and readiness; they do not create actors.
3. **Physical execution plane.** The `ray_options(...)` values create the actor pools. At runtime, ready page `Grain`s from different PDFs can share one execution microbatch on the OCR actors. The physical batch may change, while the control-plane lineage keeps each document's pages separate and ordered.

The complete UDF implementations live in [`rayorch/benchmarks/mineru/udfs.py`](rayorch/benchmarks/mineru/udfs.py); the full configurable pipeline is [`rayorch/benchmarks/mineru/pipeline.py`](rayorch/benchmarks/mineru/pipeline.py).

The public model has six pieces:

| Object | Meaning |
| --- | --- |
| UDF | A Python class or function that processes one runtime batch. |
| `RayModule` | UDF construction, actor resources, replicas, batch size, and recovery policy. Reusing the same object reuses its actor pool. |
| `Pipeline` | Static topology declared in `forward()`. |
| `rayorch.F` | Explicit expansion, filtering, broadcast, and ordered reduction. |
| `Executor` | Persistent actor ownership and one run's execution lifecycle. |
| `RunResult` | Ordered outputs and immutable run metrics. |

### UDF contract

- Each input Port is passed as one column. All input columns in a runtime batch have the same row count.
- Each output Port returns one value per input row. Ordinary Python values, including `None` and empty containers, are valid business values.
- An output consumed by `F.expand` returns one ordered child sequence per input row. `F.expand` turns those children into independently schedulable entities.
- `F.reduce` collects surviving child values back into their declared parent order. An empty successful group becomes an empty list.
- Return `ro.RecordFailure(cause)` for one logical record or `ro.GroupFailure(cause)` to suppress its siblings under the same direct parent.
- Final non-success results are exposed as `ro.OutputIssue` with `ItemOutcome.DROPPED`, `FAILED`, or `SUPPRESSED`. There is no hidden missing-value sentinel; `None` remains an ordinary value.

### Terms

| Term | Meaning |
| --- | --- |
| Domain | One logical hierarchy level, such as documents or pages. |
| Entity | One data unit inside a Domain, such as document A or page A/0. |
| Call | One configured use of a `RayModule` in the static graph. |
| Grain | One `Call × Entity` invocation, the unit of scheduling and retry. It is not a Ray task. |
| Port | One logical input or output edge in the graph. |
| Item | The value or terminal outcome at one `Port × Entity`. |
| Execution microbatch | A temporary physical batch of ready Grains sent to one actor. It may mix parents and does not define lineage or output order. |

### Structural operations

| API | Shape change | Use |
| --- | --- | --- |
| `F.expand(groups)` | One ordered group → many child entities | Pages, frames, regions, or other variable-cardinality work. |
| `F.filter(values, mask)` | Keep or drop entities in the same Domain | Selection without changing parentage or order. |
| `F.broadcast(value, like=descendants)` | One ancestor value → descendant views | Make document metadata available to every page. |
| `F.reduce(values, members=...)` | Many children → one ordered parent group | Reconstruct a document, video, or nested result. |

`F.expand_aligned` and `F.reduce_aligned` apply the same membership and order to several Ports at once. These operations declare structure during compilation; they do not create actors or execute user code.

RayOrch owns logical dependencies, cardinality, lineage, readiness, recovery, and reconstruction. Ray owns placement, actors, RPCs, resources, and object storage. Batching, placement, and retry timing may change while the logical result remains schedule-invariant.

## Reuse one model across stages

Keep one `RayModule` when several Calls should share an initialized model stack:

```python
import rayorch as ro
from my_model_library import load_model


class ModelStack:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def run(self, values, *, stage):
        if stage == "layout":
            return self.model.layout(values)
        if stage == "recognize":
            return self.model.recognize(values)
        raise ValueError(stage)


class DocumentPipeline(ro.Pipeline):
    def __init__(self, model_path):
        self.model = (
            ro.RayModule(ModelStack)
            .pre_init(model_path)
            .ray_options(replicas=4, num_gpus=1, batch_size=16)
        )

    def forward(self, pages):
        layouts = self.model(pages, stage="layout")
        return self.model(layouts, stage="recognize")
```

The two Calls keep independent dependencies, readiness, recovery, and metrics while sharing the actors owned by `self.model`.

## Workloads and integrations

- **[Flash-MinerU](https://github.com/OpenDCAI/Flash-MinerU)** is the clearest maintained application: PDF → pages → GPU VLM/OCR → ordered document assembly. Its current pipeline-parallel path reports about **8.5 minutes for 368 PDFs on one 8×A100 host**, versus about **14 minutes** for its eight-process MinerU baseline (roughly **1.7× faster** in that setup). Start with the [RayOrch application guide](https://opendcai.github.io/RayOrch-doc/en/benchmarks/flash-mineru.html), then see the [Flash-MinerU benchmark notes](https://github.com/OpenDCAI/Flash-MinerU#-benchmark) for the dataset, versions, and commands; these are workload results, not universal guarantees.
- [Built-in benchmarks](https://opendcai.github.io/RayOrch-doc/en/benchmarks/) cover MinerU, YOLO → SAM, dual vLLM, SGLang → vLLM, nested documents, video captioning, and multimodal video.
- [DataFlow integration](https://github.com/OpenDCAI/DataFlow) demonstrates wrapping row-independent operators without changing their storage-facing API.

Operators that depend on global cross-row state should keep their original execution path rather than being forced into data parallelism.

## Compatibility

RayOrch currently targets Python 3.11 and 3.12 with Ray 2.x. It is published as an Alpha project. The examples and benchmarks cover finite, row-aligned input batches executed through Ray CPU/GPU actor pools; framework-specific model dependencies remain the responsibility of each workload environment.

Start with the [installation guide](https://opendcai.github.io/RayOrch-doc/en/guide/installation.html) and [first pipeline](https://opendcai.github.io/RayOrch-doc/en/guide/first-pipeline.html). Then follow the [1 → M → 1 guide](https://opendcai.github.io/RayOrch-doc/en/guide/fan-out-and-reduce.html), the [cardinality and lineage concepts](https://opendcai.github.io/RayOrch-doc/en/concepts/cardinality.html), and [distributed execution](https://opendcai.github.io/RayOrch-doc/en/distributed/). For a complete application, use the [MinerU benchmark](https://opendcai.github.io/RayOrch-doc/en/benchmarks/mineru.html); for every public symbol, use the [API reference](https://opendcai.github.io/RayOrch-doc/en/api/).

## Cite and license

If RayOrch helps your research, please cite the [RayOrch paper](https://arxiv.org/abs/2609.18703):

```bibtex
@article{ma2026rayorch,
  title={RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation},
  author={Ma, Xiaochen and Meng, Zimo and Liang, Junzhu and Jiang, Youhe and Cheng, Yue and Liang, Hao and Zeng, Bohan and Li, Dengchun and Ma, Lu and Zhao, Zhengyang and others},
  journal={arXiv preprint arXiv:2609.18703},
  year={2026}
}
```

RayOrch is released under the [Apache License 2.0](LICENSE).
