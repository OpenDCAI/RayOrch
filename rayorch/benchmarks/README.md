# Built-in Benchmarks

This directory contains workload implementations, not Benchmark framework code.

Every workload has four implementation files and one local guide:

- `udfs.py`: batched business functions;
- `pipeline.py`: the RayOrch graph and actor resources;
- `env.json`: dependencies used by Ray Jobs;
- `benchmark.py`: a thin user configuration and input adapter.
- `README.md`: meaning, topology, execution, and result format.

Users import built-ins from `rayorch.benchmark`, for example:

```python
from rayorch.benchmark import MinerUBench
```

The built-ins currently include:

- `MinerUBench`: the real MinerU PDF workload;
- `MinerUScaleBench`: the production-hardened local/HDFS MinerU workload with
  the completed 64-GPU experiment's resource defaults;
- `YoloSamBench`: the original real YOLO -> SAM image workload;
- `DualVllmBench`: the original real two-model vLLM workflow;
- `SglangVllmBench`: SGLang -> vLLM with one Conda environment per model stage;
- `DocumentTopologyBench`: nested Document -> Page -> TableJob reductions;
- `VideoCaptionTopologyBench`: one Video -> Frame -> Video relation;
- `VideoMultimodalTopologyBench`: sibling audio/frame relations joined at Video.

The three topology Benchmarks are deterministic and dependency-free. They are
runnable examples promoted from the historical workload-shape regression tests,
not claims of production Docling, caption-model, or multimodal-model adapters.

User-facing controls belong on the Benchmark dataclass. A workload may expose
one small `stage_options` mapping for advanced per-stage `RayModule.ray_options`
overrides instead of flattening every scheduler option into top-level fields.
This includes Ray-native `runtime_env` overrides for cross-environment stages.

To add a workload, copy the shape of one directory here or create the same
layout in an external package. Do not add a workload-specific runner, CLI,
plugin class, or execution lifecycle. Keep heavy imports inside UDF methods so
registry and configuration imports stay lightweight.
