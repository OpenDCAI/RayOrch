"""Public Benchmark configuration, reports, and Ray Jobs submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from rayorch.benchmark import (
    BenchmarkReport,
    BenchmarkRun,
    LocalSource,
    MinerUBench,
    benchmark_config,
)


def test_mineru_configuration_is_json_safe_and_side_effect_free(tmp_path: Path):
    before = set(sys.modules)
    benchmark = MinerUBench(
        input_path=tmp_path / "pdfs",
        model=tmp_path / "model",
        output_dir=tmp_path / "out",
        num_gpus=8,
        batch_size=32,
        stage_options={
            "render": {"replicas": 4},
            "ocr": {"batch_size": 16, "resources": {"accelerator": 1}},
        },
    )

    config = benchmark_config(benchmark)

    assert json.loads(json.dumps(config)) == config
    assert MinerUBench(**config) == benchmark
    assert benchmark.num_gpus == 8
    assert benchmark.batch_size == 32
    assert benchmark.stage_options == {
        "render": {"replicas": 4},
        "ocr": {"batch_size": 16, "resources": {"accelerator": 1}},
    }
    assert "ray" not in set(sys.modules) - before
    assert "vllm" not in set(sys.modules) - before


def test_report_round_trip_and_default_artifact_layout(tmp_path: Path):
    summary = tmp_path / "out" / ".rayorch-benchmark" / "run-1" / "summary.json"
    report = BenchmarkReport(
        benchmark="mineru",
        run_id="run-1",
        config={"input_path": "/data"},
        metrics={"measured_wall_s": 1.5},
        profile={"status": "disabled"},
        artifacts={"summary": str(summary)},
    )

    report.write_json(summary)
    restored = BenchmarkReport.from_dict(json.loads(summary.read_text()))

    assert restored == report
    assert restored.elapsed_s == 1.5


def test_code_sources_have_two_explicit_modes(tmp_path: Path):
    workload_env = {
        "pip": {"packages": ["heavy==1"], "pip_check": True},
        "env_vars": {"TOKENIZERS_PARALLELISM": "false"},
    }
    source = tmp_path / "rayorch"
    module = tmp_path / "flash_mineru"
    source.mkdir()
    module.mkdir()

    prepared = LocalSource(source, (module,)).runtime_env(workload_env)
    assert prepared["working_dir"] == str(source.resolve())
    assert prepared["py_modules"] == [str(module.resolve())]
    assert prepared["pip"] == workload_env["pip"]

    prepared = LocalSource(
        source,
        (module,),
        install_dependencies=False,
    ).runtime_env(workload_env)
    assert "pip" not in prepared


def test_submit_uses_ray_jobs_without_building_wheels(tmp_path: Path, monkeypatch):
    benchmark = MinerUBench(
        input_path=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    captured = {}

    class Client:
        def __init__(self, address):
            captured["address"] = address

        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "job-1"

    monkeypatch.setitem(
        sys.modules,
        "ray.job_submission",
        SimpleNamespace(JobSubmissionClient=Client),
    )

    handle = benchmark.submit(
        "http://ray-head:8265",
        profile=False,
        run_id="run-1",
    )

    assert handle.job_id == "job-1"
    assert captured["address"] == "http://ray-head:8265"
    assert "rayorch.benchmark._job" in captured["entrypoint"]
    assert (
        "--benchmark-class "
        "rayorch.benchmarks.mineru.benchmark:MinerUBench"
        in captured["entrypoint"]
    )
    assert captured["runtime_env"]["env_vars"] == {
        "TOKENIZERS_PARALLELISM": "false"
    }
    assert "pip" not in captured["runtime_env"]
    assert captured["metadata"] == {
        "rayorch.benchmark": "mineru",
        "rayorch.benchmark_class": (
            "rayorch.benchmarks.mineru.benchmark:MinerUBench"
        ),
        "rayorch.run_id": "run-1",
    }


def test_local_source_is_prepared_by_ray_without_building_a_wheel(tmp_path: Path):
    from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
    from ray._private.runtime_env.working_dir import upload_working_dir_if_needed

    project = tmp_path / "project"
    module = tmp_path / "workload"
    (project / "rayorch").mkdir(parents=True)
    (project / "rayorch" / "__init__.py").write_text("", encoding="utf-8")
    module.mkdir()
    (module / "__init__.py").write_text("", encoding="utf-8")
    environment = LocalSource(project, (module,)).runtime_env({})
    uploaded = []

    def capture(*args, **kwargs):
        uploaded.append((args, kwargs))

    prepared = upload_working_dir_if_needed(
        dict(environment),
        upload_fn=capture,
    )
    prepared = upload_py_modules_if_needed(prepared, upload_fn=capture)

    assert prepared["working_dir"].startswith("gcs://")
    assert prepared["py_modules"][0].startswith("gcs://")
    assert [call[0][0] for call in uploaded] == [
        str(project.resolve()),
        str(module.resolve()),
    ]
    assert all(not path.endswith(".whl") for path in prepared["py_modules"])


def test_remote_run_wait_restores_persisted_outputs(tmp_path: Path):
    summary = tmp_path / "summary.json"
    report = BenchmarkReport(
        benchmark="mineru",
        run_id="run-1",
        config={},
        metrics={"measured_wall_s": 1.0},
        profile={"status": "disabled"},
        artifacts={"summary": str(summary)},
        outputs=[{"pdf": "a", "pages": 1}],
    )
    report.write_json(summary, include_outputs=True)
    client = SimpleNamespace(get_job_status=lambda job_id: "SUCCEEDED")

    restored = BenchmarkRun("job-1", summary, client).wait()

    assert restored.outputs == [{"pdf": "a", "pages": 1}]
