# MinerU scale Benchmark

This Benchmark packages the business UDFs used by RayOrch's completed
8-node/64-H20 MinerU experiment behind the current public RayOrch API. It runs
without patches to RayOrch, Flash-MinerU, or the execution engine.

The historical reference processed 3,690 PDFs (174,744 pages) with one failed
document at 48.1738 pages/s. That number is provenance, not a performance
guarantee. The exact historical contract is recorded in
`success_contract.json`.

## Topology

```text
PDF URI ─┬─> render ─> Page ─> MinerU/vLLM OCR ─> ordered reduce ─┐
         └─> metadata/stem ───────────────────────────────────────┤
                                                                  v
                                               Markdown + JSON + images
```

The four UDFs in `udfs.py` preserve the scale-specific behavior that mattered
in the successful run:

- local or HDFS PDF reads with retry and filesystem-client refresh;
- one persistent MinerU/vLLM client per GPU actor;
- node-local vLLM initialization locking and non-overlapping port blocks;
- lightweight document metadata rather than forwarding PDF bytes;
- ordered document assembly with atomic `_SUCCESS` publication and resume.

## Requirements

- Python 3.11 or 3.12 and a Ray cluster compatible with the installed RayOrch;
- the MinerU model at the same local/shared path on every GPU node;
- Flash-MinerU importable as `flash_mineru` on every actor;
- PyArrow plus a working HDFS client configuration when using `hdfs://`;
- a shared POSIX/Ceph output path, or an HDFS output URI.

The reference Flash-MinerU revision is:

```text
7246a353554cee35674d7ef82f2c061db2f43003
```

`env.json` records the successful dependency line. On a prepared GPU image,
prefer reusing the installed environment instead of reinstalling vLLM and
Torch during every Ray Job.

## Direct run

From the RayOrch checkout, make Flash-MinerU importable and run the module:

```bash
export PYTHONPATH=/path/to/Flash-mineru:${PYTHONPATH}

python -m rayorch.benchmarks.mineru_scale \
  --input hdfs://namenode/path/to/pdfs-a \
  --input hdfs://namenode/path/to/pdfs-b \
  --model /shared/models/MinerU2.5-2509-1.2B \
  --output /shared/results/mineru-scale-run \
  --artifact-dir /shared/results/mineru-scale-reports \
  --ray-address auto
```

No source constants need editing. Every location and resource count is a CLI
argument. Repeat `--input` to combine roots; duplicate PDF stems are rejected
before actors start.

The defaults reproduce the successful 64-GPU resource shape:

| Setting | Default |
| --- | ---: |
| render actors | 256 CPU actors |
| OCR actors | 128 actors |
| GPU per OCR actor | 0.5 |
| total OCR reservation | 64 GPUs |
| assemble actors | 64 CPU actors |
| OCR batch size | 64 pages |
| input batch size | 24 PDFs |
| active input batches | 24 |
| vLLM GPU memory utilization | 0.32 |

For a small smoke run, override the scale explicitly:

```bash
python -m rayorch.benchmarks.mineru_scale \
  --input /shared/pdfs \
  --input-limit 2 \
  --model /shared/models/MinerU2.5-2509-1.2B \
  --output /shared/results/mineru-smoke \
  --render-replicas 2 \
  --ocr-replicas 1 \
  --assemble-replicas 1 \
  --gpus-per-ocr-actor 1 \
  --ray-address auto
```

## Python API

```python
from rayorch.benchmark import MinerUScaleBench

bench = MinerUScaleBench(
    input_paths=(
        "hdfs://namenode/path/to/pdfs-a",
        "hdfs://namenode/path/to/pdfs-b",
    ),
    model="/shared/models/MinerU2.5-2509-1.2B",
    output_dir="/shared/results/mineru-scale-run",
    artifact_dir="/shared/results/mineru-scale-reports",
)
report = bench.run(ray_address="auto")
report.print_summary()
```

For advanced placement or runtime environments, use `stage_options` with the
stage names `render`, `ocr`, `metadata`, and `assemble`.

## Ray Jobs

The Benchmark is registered as `mineru_scale`, so normal RayOrch submission
works without registry edits:

```python
from rayorch.benchmark import LocalSource, MinerUScaleBench

bench = MinerUScaleBench(
    input_paths=("hdfs://namenode/path/to/pdfs",),
    model="/shared/models/MinerU2.5-2509-1.2B",
    output_dir="/shared/results/mineru-scale-run",
    artifact_dir="/shared/results/mineru-scale-reports",
)
run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(
        project_root="/path/to/RayOrch",
        modules=("/path/to/Flash-mineru/flash_mineru",),
        install_dependencies=False,
    ),
)
report = run.wait(timeout_s=21600)
```

`artifact_dir` must be visible to both the Ray Job driver and the submitting
client. Input, model, and output data are not uploaded by `LocalSource`.

## Output and resume semantics

Each document is written below `<output>/<pdf-stem>/` and becomes complete only
when its `_SUCCESS` JSON exists. A rerun validates and reuses committed
documents. Partial local documents keep `.rayorch-incomplete`; HDFS writes use
a staging directory followed by a move into the final path.

The standard Benchmark report is written below
`<artifact-dir>/<run-id>/summary.json` with document count, failed-document
count, page count, throughput, actor/RPC metrics, and the complete output list.

## Codex parameter-tuning Skill

The repository includes a reusable tuning Skill with the historical evidence,
decision workflow, and a deterministic starting-configuration generator:

```text
skills/mineru-scale-tuning/
```

Install it into Codex from the RayOrch checkout:

```bash
cp -R rayorch/benchmarks/mineru_scale/skills/mineru-scale-tuning \
  "${CODEX_HOME:-$HOME/.codex}/skills/"
```

It can then be invoked as `$mineru-scale-tuning`. The calculator is also usable
without Codex:

```bash
python rayorch/benchmarks/mineru_scale/skills/mineru-scale-tuning/scripts/recommend.py \
  --gpus 64 \
  --gpu-memory-gb 96 \
  --cluster-cpus 512 \
  --profile shared-h20 \
  --format cli
```
