"""Credential-free static tests for the TaiJi submission assets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import orchestrate  # noqa: E402
import remote_driver  # noqa: E402


def test_compute_is_exactly_eight_workers_with_eight_h20s_each() -> None:
    spec = orchestrate.validate_compute_spec()
    group = spec["resource"]["executor_group"][0]
    taiji = group["extra"]["gpu_provider"]["taiji"]
    assert (group["replicas"], group["num_gpu"]) == (8, 8)
    assert taiji["gpu_type"] == "H20"
    assert taiji["app_group_name"] == "TaiJi_HYAide_LLM_Pretrain_Data"
    assert taiji["app_group_location"] == "gy"
    assert spec["runtime_spec"]["taiji_app_group"] == taiji["app_group_name"]
    assert spec["runtime_spec"]["taiji_app_group_location"] == "gy"


def test_zhongwei_fallback_is_pretrain_8x8_and_uses_zhongwei_ceph() -> None:
    spec = orchestrate.validate_compute_spec(
        orchestrate.ZW_COMPUTE_SPEC_PATH,
        expected_location="zw",
    )
    group = spec["resource"]["executor_group"][0]
    taiji = group["extra"]["gpu_provider"]["taiji"]
    assert (group["replicas"], group["num_gpu"]) == (8, 8)
    assert taiji["gpu_type"] == "H20"
    assert taiji["app_group_name"] == "TaiJi_HYAide_LLM_Pretrain_Data"
    assert taiji["app_group_location"] == "zw"
    run_id = "raydata-zw-fallback"
    output = orchestrate.CEPH_OUTPUT_ROOTS["zw"] / run_id
    plan = orchestrate.build_plan(
        run_id=run_id,
        engine="raydata",
        profile_reference=orchestrate.ZW_PROFILE_REF,
        output_uri=str(output),
    )
    assert plan["compute"]["location"] == "zw"
    assert plan["output_uri"] == str(output)
    assert plan["output_kind"] == "ceph"


def test_zhongwei_profile_rejects_guiyang_ceph_output() -> None:
    with pytest.raises(orchestrate.ValidationError, match="Ceph output"):
        orchestrate.build_plan(
            run_id="wrong-region-output",
            engine="raydata",
            profile_reference=orchestrate.ZW_PROFILE_REF,
            output_uri=(
                "/apdcephfs_gy5/share_304380933/hunyuan/clapliang/"
                "wrong-region-output"
            ),
        )


def test_zhongwei_text_pipeline_profile_is_region_and_group_pinned() -> None:
    spec = orchestrate.validate_compute_spec(
        orchestrate.ZW_TEXT_COMPUTE_SPEC_PATH,
        expected_location="zw",
        expected_app_group="TaiJi_HYAide_text_data_pipelines",
    )
    taiji = spec["resource"]["executor_group"][0]["extra"]["gpu_provider"][
        "taiji"
    ]
    assert taiji["app_group_name"] == "TaiJi_HYAide_text_data_pipelines"
    assert taiji["app_group_location"] == "zw"
    run_id = "raydata-zw-text-pipelines"
    plan = orchestrate.build_plan(
        run_id=run_id,
        engine="raydata",
        profile_reference=orchestrate.ZW_TEXT_PROFILE_REF,
        output_uri=str(orchestrate.CEPH_OUTPUT_ROOTS["zw"] / run_id),
    )
    assert plan["compute"]["taiji_app_group"] == (
        "TaiJi_HYAide_text_data_pipelines"
    )
    assert plan["compute"]["location"] == "zw"


def test_zhongwei_pretrain_two_by_four_profile_is_exact() -> None:
    spec = orchestrate.validate_compute_spec(
        orchestrate.ZW_TWO_BY_FOUR_COMPUTE_SPEC_PATH,
        expected_location="zw",
        expected_app_group="TaiJi_HYAide_LLM_Pretrain_Data",
        expected_workers=2,
        expected_gpus_per_worker=4,
    )
    group = spec["resource"]["executor_group"][0]
    taiji = group["extra"]["gpu_provider"]["taiji"]
    assert (group["replicas"], group["num_gpu"]) == (2, 4)
    assert taiji["gpu_type"] == "H20"
    assert taiji["app_group_name"] == "TaiJi_HYAide_LLM_Pretrain_Data"
    assert taiji["app_group_location"] == "zw"

    run_id = "rayorch-pdfs2000-2x4-zw-contract"
    output = f"{orchestrate.HDFS_USER_PREFIX}/experiments/{run_id}"
    plan = orchestrate.build_plan(
        run_id=run_id,
        input_uris=(orchestrate.DEFAULT_INPUT_URIS[0],),
        input_limits=(2000,),
        output_uri=output,
        profile_reference=orchestrate.ZW_TWO_BY_FOUR_PROFILE_REF,
        hyperparameters={
            "microbatch_size": 24,
            "max_active_microbatches": 24,
            "batch_size": 64,
            "gpu_memory_utilization": 0.8,
            "ocr_replicas": 8,
            "gpus_per_ocr_actor": 1.0,
            "render_replicas": 16,
            "reduce_replicas": 16,
            "spool_upload_workers": 8,
            "spool_drain_timeout_s": 21600,
        },
    )
    assert plan["compute"] == {
        "workers": 2,
        "gpu_per_worker": 4,
        "gpu_total": 8,
        "gpu_type": "H20",
        "taiji_app_group": "TaiJi_HYAide_LLM_Pretrain_Data",
        "location": "zw",
        "spec_fingerprint": plan["compute"]["spec_fingerprint"],
    }
    assert plan["expected_pdfs"] == 2000
    assert plan["output_uri"] == output


def test_profile_keeps_compute_and_contains_only_cmk_reference() -> None:
    profile = orchestrate.validate_profile()
    text = orchestrate.PROFILE_PATH.read_text(encoding="utf-8")
    assert profile["keep_compute"] is True
    assert Path(profile["cmk_file"]).is_absolute()
    assert "cmk =" not in text.lower()
    assert '"key"' not in text.lower()


def test_plan_is_credential_free_and_uses_exact_hdfs_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = orchestrate.validate_profile()
    cmk_path = Path(profile["cmk_file"])
    original = Path.read_text

    def guarded_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == cmk_path:
            raise AssertionError("plan attempted to read the CMK file")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    plan = orchestrate.build_plan(run_id="static-contract")
    assert plan["input_uris"] == list(orchestrate.DEFAULT_INPUT_URIS)
    assert [item["files"] for item in plan["input_contracts"]] == [2000, 1690]
    assert plan["model_uri"] == orchestrate.DEFAULT_MODEL_URI
    assert plan["flash_repo"] == "."
    assert plan["output_uri"].endswith("/static-contract")
    assert plan["output_kind"] == "hdfs"
    assert plan["compute"]["gpu_total"] == 64
    assert plan["hyperparameters"] == orchestrate.HYPERPARAMETERS
    assert plan["success_policy"]["release_compute"] is False

    ceph_plan = orchestrate.build_plan(
        run_id="ceph-contract",
        output_uri=(
            "/apdcephfs_gy5/share_304380933/hunyuan/clapliang/"
            "ceph-contract"
        ),
    )
    assert ceph_plan["output_kind"] == "ceph"
    assert ceph_plan["output_base_uri"].endswith("/clapliang")


def test_snapshot_contains_runner_and_exact_source_digest() -> None:
    digest = orchestrate._source_digest()
    with orchestrate.source_snapshot(digest) as snapshot:
        assert (snapshot / "rayorch" / "experimental" / "multigrain_v3_6" / "benchmark" / "mineru_taiji.py").is_file()
        assert (snapshot / "taiji_multinode_mineru" / "remote_driver.py").is_file()
        assert (snapshot / "flash_mineru" / "__init__.py").is_file()
        manifest = json.loads(
            (snapshot / "snapshot-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["source_digest"] == digest


def test_ceph_snapshot_copies_pat_only_into_ephemeral_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-pat", encoding="utf-8")
    monkeypatch.setattr(orchestrate, "TAIJI_PAT_TOKEN_PATH", token)
    digest = orchestrate._source_digest()

    with orchestrate.source_snapshot(
        digest,
        include_ceph_token=True,
    ) as snapshot:
        runtime_token = (
            snapshot / "taiji_multinode_mineru" / "taijiPATToken.runtime"
        )
        assert runtime_token.read_text(encoding="utf-8") == "test-pat"
        assert runtime_token.stat().st_mode & 0o777 == 0o600


def test_probe_validation_accepts_only_attested_eight_by_eight() -> None:
    plan = orchestrate.build_plan(run_id="probe-contract")
    result = {
        "status": "ok",
        "read_only": True,
        "source_digest": plan["source_digest"],
        "pdf_manifests": plan["input_contracts"],
        "model_manifest": {"uri": plan["model_uri"], "files": 15},
        "expected_gpu_workers": 8,
        "expected_gpus_per_worker": 8,
        "gpu_workers": [
            {"node_id": f"node-{index}", "gpus": 8.0}
            for index in range(8)
        ],
        "accelerator_resources": {
            f"node-{index}": 1.0 for index in range(8)
        },
    }
    orchestrate._validate_probe(result, plan)
    result["gpu_workers"][1]["gpus"] = 7.0
    with pytest.raises(RuntimeError, match="probe gate failed"):
        orchestrate._validate_probe(result, plan)


def test_full_readiness_requires_all_eight_ceph_gpu_workers() -> None:
    plan = orchestrate.build_plan(
        run_id="readiness-contract",
        profile_reference=orchestrate.ZW_TEXT_PROFILE_REF,
        output_uri=str(
            orchestrate.CEPH_OUTPUT_ROOTS["zw"] / "readiness-contract"
        ),
    )
    payload = {
        "mode": "local_spool_async_ceph",
        "pdf_uris": plan["input_uris"],
        "gpu_workers": 8,
        "mount_results": [
            {
                "mounted": True,
                "output_base": plan["output_base_uri"],
                "node_id": f"node-{index}",
            }
            for index in range(8)
        ],
    }
    orchestrate._validate_full_readiness(payload, plan)
    payload["mount_results"][7]["mounted"] = False
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        orchestrate._validate_full_readiness(payload, plan)


def test_cli_exposes_idempotent_resume_command(tmp_path: Path) -> None:
    args = orchestrate.build_parser().parse_args(
        ["resume", "--state-file", str(tmp_path / "state.json")]
    )
    assert args.command == "resume"
    assert args.state_file == tmp_path / "state.json"


def test_full_wrapper_delegates_to_mineru_taiji_with_frozen_scale_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rayorch.experimental.multigrain_v3_6.benchmark import mineru_taiji

    captured: dict[str, object] = {}

    def fake_run(args: object) -> dict[str, object]:
        captured["args"] = args
        return {"status": "completed", "run_uri": "hdfs://example/run"}

    monkeypatch.setattr(mineru_taiji, "run_taiji", fake_run)
    monkeypatch.setenv("RAYORCH_RUN_ID", "delegate-contract")
    monkeypatch.setenv(
        "RAYORCH_HDFS_INPUT_URIS",
        json.dumps(orchestrate.DEFAULT_INPUT_URIS),
    )
    monkeypatch.setenv(
        "RAYORCH_EXPECTED_PDF_FILES", json.dumps([2000, 1690])
    )
    monkeypatch.setenv(
        "RAYORCH_EXPECTED_PDF_BYTES",
        json.dumps([4060449252, 1099397628624]),
    )
    monkeypatch.setenv("RAYORCH_PDF_LIMITS", json.dumps([2000, 1690]))
    monkeypatch.setenv("RAYORCH_BENCHMARK_ONLY", "0")
    monkeypatch.setenv(
        "RAYORCH_HYPERPARAMETERS", json.dumps(orchestrate.HYPERPARAMETERS)
    )
    monkeypatch.setenv("RAYORCH_HDFS_MODEL_URI", orchestrate.DEFAULT_MODEL_URI)
    monkeypatch.setenv("RAYORCH_OUTPUT_KIND", "hdfs")
    monkeypatch.setenv("RAYORCH_OUTPUT_BASE", orchestrate.DEFAULT_OUTPUT_BASE)
    monkeypatch.setenv("RAYORCH_FLASH_MINERU_REPO", ".")
    monkeypatch.setenv(
        "RAYORCH_CEPH_APP_GROUP", "TaiJi_HYAide_LLM_Pretrain_Data"
    )
    monkeypatch.setenv("RAYORCH_CEPH_LOCATION", "gy")
    result = remote_driver._full()
    args = captured["args"]
    assert result["status"] == "completed"
    assert getattr(args, "hdfs_model_uri") == orchestrate.DEFAULT_MODEL_URI
    assert getattr(args, "hdfs_pdf_uri") == list(orchestrate.DEFAULT_INPUT_URIS)
    assert getattr(args, "max_active_microbatches") == 24
    assert getattr(args, "render_replicas") == 256
    assert getattr(args, "reduce_replicas") == 64
    assert getattr(args, "ocr_replicas") == 128
    assert getattr(args, "gpus_per_ocr_actor") == 0.5
    assert getattr(args, "spool_upload_workers") == 8
    assert getattr(args, "spool_drain_timeout_s") == 21600


@pytest.mark.parametrize(
    ("profile_reference", "workers", "gpus_per_worker"),
    [
        (orchestrate.ZW_ONE_BY_FOUR_PROFILE_REF, 1, 4),
        (orchestrate.ZW_ONE_BY_EIGHT_PROFILE_REF, 1, 8),
        (orchestrate.ZW_TWO_BY_EIGHT_PROFILE_REF, 2, 8),
        (orchestrate.ZW_FOUR_BY_EIGHT_PROFILE_REF, 4, 8),
        (orchestrate.ZW_PROFILE_REF, 8, 8),
    ],
)
def test_zhongwei_scaling_profiles_are_exact(
    profile_reference: str,
    workers: int,
    gpus_per_worker: int,
) -> None:
    plan = orchestrate.build_plan(
        run_id=f"scaling-{workers}x{gpus_per_worker}",
        output_uri=(
            f"{orchestrate.HDFS_USER_PREFIX}/experiments/"
            f"scaling-{workers}x{gpus_per_worker}"
        ),
        profile_reference=profile_reference,
        hyperparameters={
            **orchestrate.HYPERPARAMETERS,
            "ocr_replicas": workers * gpus_per_worker * 2,
            "render_replicas": workers * gpus_per_worker * 4,
            "reduce_replicas": workers * gpus_per_worker,
        },
    )
    assert plan["compute"]["workers"] == workers
    assert plan["compute"]["gpu_per_worker"] == gpus_per_worker
    assert plan["compute"]["gpu_total"] == workers * gpus_per_worker
    assert plan["compute"]["location"] == "zw"
    assert plan["compute"]["gpu_type"] == "H20"
    assert plan["compute"]["taiji_app_group"] == (
        "TaiJi_HYAide_LLM_Pretrain_Data"
    )


def test_job_runtime_env_installs_only_frozen_workload_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Jobs:
        _proxy_post = object()
        _package_builder = object()

        def submit_proxy(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            return object()

    class Credentials:
        user = "test-user"
        cmk = "not-persisted"

    jobs = Jobs()
    from hydp_engine.ray_jobs import packaging

    def fake_upload(runtime_env, **_kwargs):
        runtime_env["working_dir"] = "gcs://test-working-dir.zip"

    monkeypatch.setattr(packaging, "maybe_upload_ray_working_dir", fake_upload)
    plan = orchestrate.build_plan(run_id="runtime-env-contract")
    orchestrate._submit_job(
        jobs,
        phase="probe",
        dashboard="http://ray.invalid",
        credentials=Credentials(),
        working_dir=tmp_path,
        plan=plan,
    )
    runtime_env = jobs.kwargs["runtime_env"]
    assert runtime_env["env_vars"]["VLLM_NO_USAGE_STATS"] == "1"
    assert runtime_env["env_vars"]["DO_NOT_TRACK"] == "1"
    assert runtime_env["pip"] == [
        "numpy==1.26.0",
        "opencv-python-headless==4.10.0.84",
        "PyMuPDF==1.26.3",
        "mineru-vl-utils==1.2.1",
        "pypdfium2==5.13.0",
        "magika==1.0.3",
        "transformers==4.57.6",
    ]
    assert not any(
        item.lower().startswith(("ray==", "rayorch"))
        for item in runtime_env["pip"]
    )
    assert runtime_env["py_modules"] == ["gcs://test-working-dir.zip"]
    assert jobs.kwargs["working_dir"] is None


def test_terminal_job_cannot_pass_hdfs_readiness_from_stale_log(tmp_path) -> None:
    marker = orchestrate.HDFS_DIRECT_READY_MARKER

    class Jobs:
        @staticmethod
        def status(_job):
            return {"status": "SUCCEEDED"}

        @staticmethod
        def logs(_job, *, tail):
            assert tail == 12000
            return [marker + '{"mode":"direct_hdfs"}']

    job = type("Job", (), {"submission_id": "terminal-before-ready"})()
    state = {}
    state_path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="became SUCCEEDED before"):
        orchestrate._wait_for_log_marker(
            Jobs(),
            job,
            marker=marker,
            timeout=0,
            poll_interval=0,
            state=state,
            state_path=state_path,
        )


def test_probe_uses_strict_spread_full_node_gpu_activation() -> None:
    source = Path(remote_driver.__file__).read_text(encoding="utf-8")
    assert 'strategy="STRICT_SPREAD"' in source
    assert '{"GPU": GPUS_PER_WORKER}' in source
    assert "range(EXPECTED_GPU_WORKERS)" in source
    assert "timeout=43_200" in source
