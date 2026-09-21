# Panda-70M video caption Benchmark

This workload keeps Panda source ownership through clip fan-out, decodes four
temporal frames per clip, sends the frames to a fused Qwen2.5-VL teacher, and
reduces the selected clip captions back to the source video.

The package follows the RayOrch Benchmark layout:

```text
udfs.py + pipeline.py + env.json + benchmark.py
```

`VideoCaptionUDF` loads vLLM once per persistent actor. `PandaFusedTeacher`
packs the four temporal prompts into one request batch for every clip batch.
The selector uses a reference-free length and diversity score during the run;
reference F1 values are recorded as diagnostics when the manifest includes
reference captions.

## Manifest

The manifest is a JSON object containing `sources`. Each source has a stable
`source_id` and ordered `clips`:

```json
{
  "sources": [
    {
      "source_id": "video-0001",
      "clips": [
        {
          "clip_index": 0,
          "path": "/shared/videos/video-0001.mp4",
          "clip_start_fraction": 0.0,
          "clip_end_fraction": 1.0,
          "reference_caption": "a person opens a door"
        }
      ]
    }
  ]
}
```

`path` may use `hdfs://`. Set `RAYORCH_DEFER_LOCAL_PATH_CHECK=1` when the
manifest is visible on the driver and video files are mounted only on workers.

## Run

```python
from rayorch.benchmark import Panda70MBench

report = Panda70MBench(
    manifest="/shared/panda70m/val.audit.json",
    output_dir="/shared/results/panda70m",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    teacher_replicas=2,
    decode_replicas=4,
).run(ray_address="auto")
```

The benchmark writes one JSON file per source and the standard RayOrch report
under `output_dir/.rayorch-benchmark/<run-id>/`.

## Submit

```python
run = Panda70MBench(
    manifest="/shared/panda70m/val.audit.json",
    output_dir="/shared/results/panda70m",
).submit("http://ray-head:8265")
report = run.wait(timeout_s=3600)
```

The cluster must provide the video paths, model weights, and the dependencies
listed in `env.json` on every eligible worker.
