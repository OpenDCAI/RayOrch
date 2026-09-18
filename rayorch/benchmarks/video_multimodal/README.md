# Video Multimodal Topology Benchmark

## Why this case exists

This dependency-free case preserves the historical dual-modality video graph.
It demonstrates two sibling child domains that execute independently and then
join at their shared parent.

## Topology

```text
                  ┌► AudioChunk ─► Transcribe ─► reduce ─┐
Video ────────────┤                                      ├► Merge ─► Video result
                  └► Frame ──────► Vision ─────► reduce ─┘
```

The audio and frame branches have different cardinalities but preserve
alignment when they return to the Video domain.

## Run

```python
from rayorch.benchmark import VideoMultimodalTopologyBench

report = VideoMultimodalTopologyBench(
    output_dir="./results",
    video_count=8,
    frames_per_video=16,
    audio_chunks_per_video=6,
    workers=4,
    batch_size=8,
    input_batch_size=1,
    max_active_input_batches=4,
).run()
```

Inputs and outputs are deterministic, so the example needs no files, models, or
optional dependencies. It supports the normal `submit()` and `LocalSource`
flow as well.

## Results

Each output contains one video name plus its ordered audio-chunk and frame IDs.
The report adds total `audio_chunks` and `frames` to the common RayOrch metrics.

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

This is a topology example, not a production ASR or vision-model adapter.
