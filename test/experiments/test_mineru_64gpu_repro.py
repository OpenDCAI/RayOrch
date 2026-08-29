from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from experiments.mineru_64gpu_repro import reproduce


@pytest.mark.parametrize("engine", ["rayorch", "raydata"])
def test_frozen_config_is_valid(engine: str) -> None:
    config = reproduce.load_config(engine)
    compute = config["compute"]
    hyperparameters = config["launcher_hyperparameters"]

    assert compute["gpu_total"] == 64
    assert hyperparameters["ocr_replicas"] == 128
    assert hyperparameters["gpus_per_ocr_actor"] == 0.5
    assert (
        hyperparameters["ocr_replicas"]
        * hyperparameters["gpus_per_ocr_actor"]
        == compute["gpu_total"]
    )


def test_configs_keep_comparison_inputs_and_compute_identical() -> None:
    rayorch = reproduce.load_config("rayorch")
    raydata = reproduce.load_config("raydata")

    assert rayorch["compute"] == raydata["compute"]
    assert rayorch["workload"] == raydata["workload"]
    assert (
        rayorch["launcher_hyperparameters"]
        == raydata["launcher_hyperparameters"]
    )


def test_configure_writes_ignored_credential_reference(
    tmp_path: Path,
) -> None:
    cmk_file = tmp_path / "cmk"
    cmk_file.write_text("secret-not-copied", encoding="utf-8")
    profile_file = tmp_path / "profile.local.toml"
    args = argparse.Namespace(
        cmk_file=cmk_file,
        profile_file=profile_file,
        remote_workdir="hydp_runs/test/mineru_64gpu",
        force=False,
    )

    assert reproduce.configure_profile(args) == 0
    profile = profile_file.read_text(encoding="utf-8")
    assert str(cmk_file) in profile
    assert "secret-not-copied" not in profile
    assert "compute_h20_8node_8gpu_zw_pretrain.json" in profile


def test_configure_refuses_to_overwrite_profile(tmp_path: Path) -> None:
    cmk_file = tmp_path / "cmk"
    cmk_file.write_text("secret", encoding="utf-8")
    profile_file = tmp_path / "profile.local.toml"
    profile_file.write_text("existing", encoding="utf-8")
    args = argparse.Namespace(
        cmk_file=cmk_file,
        profile_file=profile_file,
        remote_workdir="hydp_runs/test/mineru_64gpu",
        force=False,
    )

    with pytest.raises(reproduce.ReproductionError, match="already exists"):
        reproduce.configure_profile(args)
