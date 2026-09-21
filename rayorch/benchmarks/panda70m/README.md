# Panda-70M Video Benchmark

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

## Download the data

Panda-70M is downloaded from the upstream [Panda-70M repository](https://github.com/snap-research/Panda-70M).
The upstream project publishes metadata CSVs (including a small validation
split) and a repository-specific `video2dataset` fork. The video files are not
packaged with RayOrch, and the full training split needs roughly 36 TB, so use
the validation or 2M metadata split for a first run.

```bash
git clone https://github.com/snap-research/Panda-70M.git
cd Panda-70M/dataset_dataloading/video2dataset
pip install -e .
cd ../..
```

Download a metadata CSV from the [upstream download table](https://github.com/snap-research/Panda-70M/blob/main/dataset_dataloading/README.md)
and run the downloader supplied by that repository (its CSV uses `url`,
`caption`, and `timestamp` columns):

```bash
video2dataset \
  --url_list=/data/panda70m/validation.csv \
  --url_col=url \
  --caption_col=caption \
  --clip_col=timestamp \
  --output_folder=/data/panda70m/validation \
  --save_additional_columns='[matching_score]' \
  --config=Panda-70M/dataset_dataloading/video2dataset/video2dataset/configs/panda70m.yaml
```

The downloader writes `.mp4`, `.txt`, and `.json` files in shards. Build the
RayOrch manifest by grouping those clips under one `source_id`, preserving a
zero-based `clip_index`, and setting each `path` to the worker-visible `.mp4`
path. A minimal manifest is shown below; `reference_caption` is optional and is
only used for diagnostics.

```json
{
  "sources": [
    {
      "source_id": "video-0001",
      "clips": [
        {"clip_index": 0, "path": "/data/panda70m/validation/00000/0000000_00000.mp4"}
      ]
    }
  ]
}
```

Keep the manifest and videos on a shared filesystem (or use `hdfs://` paths).
When the driver cannot see worker-local files, set
`RAYORCH_DEFER_LOCAL_PATH_CHECK=1` before constructing the benchmark.

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
from rayorch.benchmark import VideoPanda70MBench

report = VideoPanda70MBench(
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
run = VideoPanda70MBench(
    manifest="/shared/panda70m/val.audit.json",
    output_dir="/shared/results/panda70m",
).submit("http://ray-head:8265")
report = run.wait(timeout_s=3600)
```

The cluster must provide the video paths, model weights, and the dependencies
listed in `env.json` on every eligible worker.
