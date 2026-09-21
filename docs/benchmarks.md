# Writing a Benchmark

A RayOrch workload has three essential parts:

```text
UDFs + Pipeline + environment
```

A reusable Benchmark adds one small Python entrypoint that connects user
configuration and input data to those three parts.

The built-in MinerU example therefore contains only:

```text
rayorch/benchmarks/mineru/
  udfs.py             # business computation
  pipeline.py         # dataflow and actor resources
  env.json            # dependencies for Ray Jobs
  benchmark.py        # user configuration -> inputs -> Pipeline
  README.md           # meaning, topology, run, and results
  __init__.py         # package marker
```

There is no workload-specific base class, plugin file, CLI, runner, or
submission implementation.

## Runnable reference workloads

Several historical workload-shape tests are also available as small,
dependency-free Benchmarks:

```python
from rayorch.benchmark import (
    DocumentTopologyBench,
    VideoCaptionTopologyBench,
    VideoMultimodalTopologyBench,
)

report = DocumentTopologyBench(
    output_dir="./results",
    document_count=8,
    pages_per_document=6,
    tables_per_page=2,
    workers=4,
).run()
```

They live in separate directories under `rayorch/benchmarks/`, each with the
same `udfs.py + pipeline.py + env.json + benchmark.py` shape as MinerU:

| Benchmark | Topology demonstrated |
| --- | --- |
| `DocumentTopologyBench` | nested Document -> Page -> TableJob -> Page -> Document |
| `VideoCaptionTopologyBench` | Video -> Frame -> Caption -> Video |
| `VideoMultimodalTopologyBench` | sibling Audio and Frame domains joined at Video |

These use synthetic deterministic UDFs so users can run and inspect the
topologies without downloading models or datasets. They preserve the graph
shapes of the old test cases; they are not production Docling, VLM, ASR, or ViT
implementations.

The earliest RayOrch examples are also preserved as real workload adapters:

| Benchmark | Topology | Optional dependencies |
| --- | --- | --- |
| `YoloSamBench` | Image -> YOLO -> SAM -> Overlay | Ultralytics, Segment Anything, Torch |
| `Panda70MBench` | Panda source -> clips -> fused Qwen caption teachers -> source | OpenCV or PyAV, PyArrow, vLLM |
| `DualVllmBench` | Prompt -> Model A -> refinement -> Model B | vLLM, CUDA |
| `SglangVllmBench` | Prompt -> SGLang -> handoff -> vLLM | SGLang and vLLM in separate Conda environments |

Each built-in directory has its own `README.md` with the workload's intent,
topology diagram, exact run/submit examples, resource requirements, and result
layout.

## Read the files in this order

### 1. `udfs.py`

Write ordinary batched Python classes. Heavy dependencies belong inside
constructors or `run()` methods so importing the Benchmark remains cheap.

```python
class Decode:
    def run(self, paths):
        return [decode(path) for path in paths]
```

### 2. `pipeline.py`

Connect the UDFs and state their Ray resources.

```python
class ImagePipeline(Pipeline):
    def __init__(self, workers=4):
        self.decode = RayModule(Decode).ray_options(
            replicas=workers,
            batch_size=16,
            num_cpus=1,
        )

    def forward(self, paths):
        return self.decode(paths)
```

### 3. `env.json`

List only the dependencies required by the workload:

```json
{
  "schema_version": 1,
  "pip": {
    "packages": ["pillow"]
  }
}
```

### 4. `benchmark.py`

Define a small dataclass with user-facing parameters. Its `run()` method loads
inputs, builds the Pipeline, and delegates the lifecycle to `run_benchmark()`.
Its `submit()` method delegates to `submit_benchmark()`.

The framework owns:

- Ray and Executor lifecycle;
- run IDs;
- timing and common execution metrics;
- profile sampling;
- report and artifact writing;
- Ray Jobs submission and result waiting.

The workload owns:

- its user parameters;
- input discovery and validation;
- Pipeline construction;
- optional workload-specific metrics.

This is composition through ordinary functions, not a `BaseBenchmark`
inheritance hierarchy.

## Register the public class

Built-ins have one import-free record in `rayorch/benchmark/registry.py`:

```python
"images": (
    "ImageBench",
    "my_package.images.benchmark:ImageBench",
    "my_package.images",
    "env.json",
)
```

External packages can register the same four values:

```python
from rayorch.benchmark import register

register(
    "images",
    public_name="ImageBench",
    benchmark_class="my_package.images.benchmark:ImageBench",
    runtime_package="my_package.images",
)
```

Registration stores strings only. It does not import the workload.

## Run locally or on an existing cluster

```python
from rayorch.benchmark import MinerUBench

bench = MinerUBench(
    input_path="/shared/pdfs",
    model="/shared/models/mineru",
    output_dir="/shared/output",
    input_limit=100,
    num_gpus=8,
)

report = bench.run(ray_address="auto")
```

`input_path` may be one PDF or a directory of PDFs.

Common controls remain explicit constructor arguments. For experiments that
need to tune one stage more deeply, `stage_options` is a small escape hatch:

```python
bench = MinerUBench(
    input_path="/shared/pdfs",
    model="/shared/models/mineru",
    output_dir="/shared/output",
    num_gpus=8,
    batch_size=64,
    stage_options={
        "render": {"replicas": 4, "num_cpus": 2},
        "ocr": {"batch_size": 32},
        "assemble": {"replicas": 2, "batch_size": 8},
    },
)
```

The valid stage names are `render`, `ocr`, `metadata`, and `assemble`.
Each mapping is passed to that stage's `RayModule.ray_options()` after its
defaults, so specified values override defaults while unspecified values stay
unchanged. Business configuration such as model and output paths remains at
the top level.

## Per-stage runtime environments

`RayModule.ray_options()` accepts normal Ray actor options. RayOrch consumes
only `replicas`, `batch_size`, and `recovery`; options such as `num_cpus`,
`num_gpus`, `resources`, and `runtime_env` are passed unchanged to Ray.

This makes a stage-specific Conda environment explicit:

```python
self.infer = RayModule(Infer).ray_options(
    replicas=2,
    num_gpus=1,
    runtime_env={"conda": "model-environment"},
)
```

Benchmark users configure the same option through `stage_options`:

```python
bench = YoloSamBench(
    ...,
    stage_options={
        "yolo": {"runtime_env": {"conda": "yolo-environment"}},
        "sam": {"runtime_env": {"conda": "sam-environment"}},
    },
)
```

Ray creates that stage's persistent actors in the requested environment. The
driver does not need the stage's heavy dependency merely to construct or
validate the Benchmark; imports happen when the UDF is initialized inside the
actor. Every node that may host the actor must have the named environment, with
compatible Python and Ray versions and access to RayOrch.

This intentionally uses Ray's native contract directly. RayOrch does not scan
Conda environments, merge dependency sets, infer environment containment, or
maintain a second environment registry.

## Submit through Ray Jobs

If RayOrch and workload dependencies are already installed on the cluster:

```python
run = bench.submit("http://ray-head:8265")
report = run.wait(timeout_s=3600)
```

During development, upload local source and let Ray install
`env.json`:

```python
from rayorch.benchmark import LocalSource

run = bench.submit(
    "http://ray-head:8265",
    source=LocalSource(
        project_root="/path/to/RayOrch",
        modules=("/path/to/Flash-mineru/flash_mineru",),
    ),
)
```

Set `install_dependencies=False` only when the cluster environment is already
prepared.

Inputs, models, outputs, and report paths are not uploaded. They must be
available at the same paths on all eligible cluster nodes.

## Multi-node behavior

RayOrch uses normal Ray actors rather than introducing another cluster
scheduler. `replicas=N, num_gpus=1` requests N one-GPU actors and Ray places
them on nodes with available resources. If the resources are unavailable, the
actors remain pending rather than silently using fewer GPUs.

The repository verifies:

- two GPU actors can be placed on two independent local raylets;
- a Ray-only environment can receive RayOrch and workload source with
  `LocalSource`, install dependencies, load the real MinerU model, and complete
  a real PDF;
- the same workload topology has run on four H20 GPUs.

Physical multi-host deployment still requires consistent networking, drivers,
and shared paths.

## Profiling boundary

Every report contains common scheduler and batching metrics. Optional profiling
records driver RSS and GPUs visible to the driver process. It is intentionally
best effort and is not presented as cluster-wide monitoring; use Ray Dashboard
or the deployment's metrics system for that.
