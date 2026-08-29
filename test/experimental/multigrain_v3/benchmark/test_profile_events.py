from __future__ import annotations

import json
from pathlib import Path

from rayorch.experimental.multigrain_v3.benchmark.profile_events import (
    ProfileEventWriter,
    aggregate_gpu_rows,
    merge_profile_events,
    write_gpu_samples,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_actor_local_events_merge_with_paired_clocks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rayorch.experimental.multigrain_v3.benchmark.profile_events._ray_identity",
        lambda: ("node-a", "actor-a"),
    )
    writer = ProfileEventWriter(
        str(tmp_path), system="rayorch", stage="ocr", role="model"
    )
    writer.actor_ready(started_epoch_s=100.0, started_monotonic_s=10.0)
    writer.stage_batch(
        started_epoch_s=101.0,
        started_monotonic_s=11.0,
        items=24,
        batch_size=64,
    )

    result = merge_profile_events(tmp_path, tmp_path / "profile_events.jsonl")
    rows = _jsonl(tmp_path / "profile_events.jsonl")
    assert result["files"] == 1
    assert [row["type"] for row in rows] == ["actor_init", "stage_batch"]
    assert rows[0]["started_epoch_s"] == 100.0
    assert rows[0]["started_monotonic_s"] == 10.0
    assert rows[0]["ready_epoch_s"] >= rows[0]["started_epoch_s"]
    assert rows[0]["ready_monotonic_s"] >= rows[0]["started_monotonic_s"]
    assert rows[1]["batch_size"] == 64
    assert rows[1]["items"] == 24
    assert rows[1]["finished_epoch_s"] >= rows[1]["started_epoch_s"]
    assert rows[1]["finished_monotonic_s"] >= rows[1]["started_monotonic_s"]


def test_distributed_gpu_samples_become_cluster_wide_rows(tmp_path):
    records = []
    for node_index in range(8):
        records.append(
            {
                "node_id": f"node-{node_index}",
                "samples": [
                    {
                        "epoch_s": 100.1 + node_index / 100.0,
                        "monotonic_s": 10.1 + node_index / 100.0,
                        "utilization": [90.0 + gpu for gpu in range(8)],
                        "memory_used": [1_000 + gpu for gpu in range(8)],
                        "power_w": [250.0 + gpu for gpu in range(8)],
                    }
                ],
            }
        )

    result = write_gpu_samples(records, tmp_path / "gpu_samples.jsonl")
    rows = _jsonl(tmp_path / "gpu_samples.jsonl")
    assert result["samples"] == 1
    assert len(rows[0]["node_ids"]) == 8
    assert len(rows[0]["gpu_ids"]) == 64
    assert len(rows[0]["utilization"]) == 64
    assert len(rows[0]["memory_used"]) == 64
    assert len(rows[0]["power_w"]) == 64
    assert isinstance(rows[0]["epoch_s"], float)
    assert isinstance(rows[0]["monotonic_s"], float)


def test_gpu_aggregation_is_deterministic_by_node_id():
    rows = aggregate_gpu_rows(
        [
            {
                "node_id": "node-b",
                "epoch_s": 100.2,
                "monotonic_s": 10.2,
                "gpu_ids": ["node-b:0"],
                "utilization": [20],
            },
            {
                "node_id": "node-a",
                "epoch_s": 100.1,
                "monotonic_s": 10.1,
                "gpu_ids": ["node-a:0"],
                "utilization": [10],
            },
        ]
    )
    assert rows[0]["node_ids"] == ["node-a", "node-b"]
    assert rows[0]["gpu_ids"] == ["node-a:0", "node-b:0"]
    assert rows[0]["utilization"] == [10, 20]
