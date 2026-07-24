"""CPU-only tests for reproducible MinerU paired-ablation orchestration."""

from __future__ import annotations

import json

import pytest

from rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation import (
    build_run_command,
    load_config,
)


def test_ablation_config_requires_unique_paired_runs(tmp_path):
    """The checked-in schema rejects missing fields and duplicate run tags."""

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "common": {},
                "runs": [
                    {
                        "tag": "same",
                        "pair": "p",
                        "mode": "elastic",
                        "batch_size": 64,
                    },
                    {
                        "tag": "same",
                        "pair": "p",
                        "mode": "parent_bound",
                        "batch_size": 64,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate run tag"):
        load_config(path)


def test_ablation_config_requires_both_modes_in_every_pair(tmp_path):
    """A pair cannot silently compare two runs of the same packing mode."""

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "common": {},
                "runs": [
                    {
                        "tag": "a",
                        "pair": "p",
                        "mode": "elastic",
                        "batch_size": 64,
                    },
                    {
                        "tag": "b",
                        "pair": "p",
                        "mode": "elastic",
                        "batch_size": 64,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="one elastic and one parent_bound"):
        load_config(path)


def test_ablation_builds_isolated_mineru_cli_command(tmp_path):
    """One schedule entry maps to one resumable child-process invocation."""

    common = {
        "limit": 48,
        "replicas": 4,
        "microbatch_size": 24,
        "max_inflight_arenas": 3,
        "max_batch_wait_ms": 10,
        "gpu_memory_utilization": 0.9,
        "render_replicas": 4,
        "reduce_replicas": 1,
        "num_cpus": 32,
        "object_store_gb": 60,
    }
    run = {
        "tag": "r1_elastic_bs64",
        "pair": "r1_bs64",
        "mode": "elastic",
        "batch_size": 64,
    }
    command = build_run_command(
        common,
        run,
        tmp_path,
        python="/usr/bin/python3",
        flash_repo="/data/flash",
        model="/data/model",
    )
    assert command[:4] == [
        "/usr/bin/python3",
        "-u",
        "-m",
        "rayorch.experimental.multigrain_v2_5.benchmark.mineru_cli",
    ]
    assert command[command.index("--mode") + 1] == "elastic"
    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs/r1_elastic_bs64"
    )
    assert command[command.index("--timeline-dir") + 1].endswith(
        "timelines/r1_elastic_bs64"
    )
    assert command[command.index("--flash-repo") + 1] == "/data/flash"
    assert command[command.index("--model") + 1] == "/data/model"
