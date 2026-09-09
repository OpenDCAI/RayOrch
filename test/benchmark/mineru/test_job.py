from __future__ import annotations

import email
import zipfile
from pathlib import Path

from rayorch.benchmark.mineru import job


def test_transport_wheel_has_no_runtime_dependencies_or_top_level_data(
    tmp_path: Path,
):
    project = tmp_path / "workload"
    package = project / "workload"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.0.0"\n')
    (project / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "transport-wheel-test"
version = "1.0.0"
dynamic = ["dependencies"]
[tool.setuptools]
packages = ["workload"]
[tool.setuptools.dynamic]
dependencies = {file = "requirements.txt"}
""".strip()
        + "\n"
    )
    (project / "requirements.txt").write_text("requests\n")
    (project / "private-input.pdf").write_bytes(b"not package data")

    wheel = job.build_wheel(project, tmp_path / "wheels")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
    assert metadata.get_all("Requires-Dist", []) == []
    assert not any(name.endswith("private-input.pdf") for name in names)


def test_create_bundle_builds_two_wheels_and_forwards_auto_address(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "rayorch"
    flash = tmp_path / "flash"
    project.mkdir()
    flash.mkdir()
    (project / "pyproject.toml").write_text("[build-system]\n")
    (flash / "pyproject.toml").write_text("[build-system]\n")

    def fake_build(source: Path, wheel_dir: Path) -> Path:
        return wheel_dir / f"{source.name}.whl"

    monkeypatch.setattr(job, "build_wheel", fake_build)
    bundle = job.create_bundle(
        project_root=project,
        flash_repo=flash,
        benchmark_args=["--limit", "4"],
        wheel_dir=tmp_path / "wheels",
    )

    assert bundle.entrypoint.endswith(
        f"--limit 4 --flash-repo {flash} --ray-address auto"
    )
    assert [path.name for path in bundle.wheels] == ["rayorch.whl", "flash.whl"]
    assert bundle.runtime_env.value["py_modules"] == [
        str((tmp_path / "wheels/rayorch.whl").resolve()),
        str((tmp_path / "wheels/flash.whl").resolve()),
    ]
