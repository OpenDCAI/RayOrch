# Video Caption Topology Benchmark

## Why this case exists

This dependency-free case preserves the historical video-caption workload
shape. It demonstrates dynamic frame fan-out, cross-video actor batching, and
ordered reduction back to one result per video.

## Topology

```text
Video
  │
  ▼
DecodeFrames ─► Frame ─► CaptionFrames
                              │
                              ▼
                       ordered reduce
                              │
                              ▼
                       Video summary
```

## Run

```python
from rayorch.benchmark import VideoCaptionTopologyBench

report = VideoCaptionTopologyBench(
    output_dir="./results",
    video_count=8,
    frames_per_video=16,
    workers=4,
    batch_size=8,
    input_batch_size=1,
    max_active_input_batches=4,
).run()
```

The Benchmark generates deterministic logical videos and frames, so it needs no
video files, model weights, or optional dependencies. It can also use the normal
`submit()` and `LocalSource` Ray Job flow.

## Results

Each output contains a video name, ordered frame IDs, and deterministic caption
strings. `report.metrics["frames"]` reports total frame work in addition to the
common execution metrics.

```text
output_dir/.rayorch-benchmark/<run-id>/
  config.json
  summary.json
  gpu_samples.jsonl
```

This is a runnable topology example, not a production VLM adapter.
