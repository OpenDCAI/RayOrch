"""TaiJi MinerU wrapper 的纯单元测试；不连接真实 Ray 或 HDFS。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3_6.benchmark import mineru_taiji
from rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji import (
    ClusterNode,
    _benchmark_args,
    _cluster_nodes,
    _hdfs_filesystem,
    _prepare_hdfs_pdf_inputs_node,
    _run_pinned_tasks,
    _stage_model_node,
    _stop_gpu_samplers,
    _summarize_gpu_samples,
    _validate_gpu_topology,
    _validate_hdfs_outputs,
    _wait_for_gpu_topology,
    build_parser,
    run_taiji,
)


def _ray_records() -> list[dict[str, object]]:
    return [
        {
            "Alive": True,
            "NodeID": "head",
            "NodeManagerAddress": "10.0.0.1",
            "Resources": {"CPU": 8.0},
        },
        *[
            {
                "Alive": True,
                "NodeID": f"gpu-{index}",
                "NodeManagerAddress": f"10.0.1.{index}",
                "Resources": {"CPU": 32.0, "GPU": 8.0},
            }
            for index in range(8)
        ],
        {
            "Alive": False,
            "NodeID": "dead",
            "NodeManagerAddress": "10.0.0.4",
            "Resources": {"CPU": 32.0, "GPU": 8.0},
        },
    ]


def test_module_has_no_top_level_pyarrow_import():
    source = Path(mineru_taiji.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            imported.append(statement.module or "")

    assert not any(
        name == "pyarrow" or name.startswith("pyarrow.") for name in imported
    )


def test_non_hdfs_uri_is_rejected_before_loading_pyarrow(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("PyArrow should not load for an invalid URI")

    monkeypatch.setattr(mineru_taiji, "_load_pyarrow_fs", fail_if_loaded)

    with pytest.raises(ValueError, match="expected an hdfs:// URI"):
        _hdfs_filesystem("/tmp/input")


def test_gpu_summary_uses_complete_active_window():
    records = [
        {
            "node_id": "a",
            "error": None,
            "samples": [
                {"time": 1.1, "utilization": [0, 0], "memory_used": [1, 2]},
                {"time": 2.1, "utilization": [80, 80], "memory_used": [3, 4]},
                {"time": 3.1, "utilization": [100, 100], "memory_used": [5, 6]},
            ],
        },
        {
            "node_id": "b",
            "error": None,
            "samples": [
                {"time": 1.1, "utilization": [0, 0], "memory_used": [2, 1]},
                {"time": 2.1, "utilization": [80, 80], "memory_used": [4, 3]},
                {"time": 3.1, "utilization": [100, 100], "memory_used": [6, 5]},
            ],
        },
    ]

    summary = _summarize_gpu_samples(
        records,
        expected_gpus=4,
        measured_start_time=1.5,
        measured_end_time=3.5,
    )

    assert summary["overall_mean_gpu_utilization"] == 60.0
    assert summary["active_mean_gpu_utilization"] == 90.0
    assert summary["active_sample_buckets"] == 2
    assert summary["measured_mean_gpu_utilization"] == 90.0
    assert summary["measured_sample_buckets"] == 2


def test_gpu_sampler_stop_tolerates_lost_node():
    class Method:
        def __init__(self, ref):
            self.ref = ref

        def remote(self):
            return self.ref

    class Actor:
        def __init__(self, ref):
            self.stop = Method(ref)

    class Ray:
        killed = []

        @staticmethod
        def get(ref, timeout):
            if ref == "lost":
                raise RuntimeError("node died")
            return ref

        @classmethod
        def kill(cls, actor, *, no_restart):
            assert no_restart
            cls.killed.append(actor)

    good = {
        "node_id": "a",
        "error": None,
        "samples": [
            {"time": 1.1, "utilization": [90, 90], "memory_used": [1, 2]}
        ],
    }
    actors = [Actor(good), Actor("lost")]

    summary = _stop_gpu_samplers(Ray, actors, expected_gpus=2)

    assert summary["active_mean_gpu_utilization"] == 90.0
    assert len(summary["errors"]) == 1
    assert Ray.killed == actors


def test_cluster_gate_selects_eight_8_gpu_workers():
    nodes = _cluster_nodes(_ray_records())
    gpu_nodes = _validate_gpu_topology(nodes)

    assert [node.node_id for node in gpu_nodes] == [f"gpu-{i}" for i in range(8)]
    assert nodes[-1].node_id == "head"
    assert sum(node.gpus for node in gpu_nodes) == 64


@pytest.mark.parametrize(
    "gpu_counts",
    [
        (8,) * 7,
        (8,) * 7 + (4,),
        (8,) * 9,
    ],
)
def test_cluster_gate_rejects_wrong_gpu_topology(gpu_counts):
    nodes = tuple(
        ClusterNode(f"node-{index}", f"10.0.0.{index}", count)
        for index, count in enumerate(gpu_counts)
    )

    with pytest.raises(RuntimeError, match="8 GPU workers × 8 GPUs"):
        _validate_gpu_topology(nodes)


def test_topology_wait_requires_two_consecutive_valid_observations():
    invalid = _ray_records()
    invalid[1] = {
        **invalid[1],
        "Resources": {"CPU": 32.0, "GPU": 4.0},
    }

    class FakeRay:
        observations = [invalid, _ray_records(), _ray_records()]

        @classmethod
        def nodes(cls):
            return cls.observations.pop(0)

    nodes, gpu_nodes = _wait_for_gpu_topology(
        FakeRay,
        timeout_s=1,
        poll_s=0,
    )

    assert len(nodes) == 9
    assert [node.node_id for node in gpu_nodes] == [f"gpu-{i}" for i in range(8)]
    assert not FakeRay.observations


def test_hdfs_pdf_inputs_gate_exact_total_bytes_without_downloading(monkeypatch):
    monkeypatch.setattr(
        mineru_taiji,
        "_assert_current_node",
        lambda expected_node_id: expected_node_id,
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_reset_tmp_directory",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_remote_file_infos",
        lambda uri: (
            None,
            None,
            "/pdfs",
            tuple(
                SimpleNamespace(path=f"/pdfs/{index}.pdf", size=0)
                for index in range(2_000)
            ),
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="expected 4060449252 directly readable HDFS PDF bytes",
    ):
        _prepare_hdfs_pdf_inputs_node(
            "node-a",
            ["hdfs://cluster/pdfs"],
            [2_000],
            [4_060_449_252],
        )


def test_model_stage_gates_exact_manifest(monkeypatch):
    monkeypatch.setattr(
        mineru_taiji,
        "_assert_current_node",
        lambda expected_node_id: expected_node_id,
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_reset_tmp_directory",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_download_hdfs_tree",
        lambda uri, path: {"files": 15, "bytes": 123},
    )

    with pytest.raises(RuntimeError, match="expected 2323649236 staged model bytes"):
        _stage_model_node(
            "node-a",
            "hdfs://cluster/model",
            "/tmp/model",
            15,
            2_323_649_236,
        )


def test_spool_uploader_preserves_loose_hdfs_layout_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    batch = tmp_path / "ready" / "batch-a"
    document = batch / "docs" / "paper" / "vlm"
    document.mkdir(parents=True)
    (document / "paper.md").write_text("markdown", encoding="utf-8")
    (document / "layout.json").write_text(
        '{"pdf_info": []}', encoding="utf-8"
    )
    (document.parent / "_SUCCESS").write_text(
        '{"pdf": "paper", "status": "completed", "pages": 1}',
        encoding="utf-8",
    )
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "batch_id": "batch-a",
                "documents": [{"pdf": "paper", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )

    class Stream:
        def __init__(self, filesystem, path, *, read=False):
            self.filesystem = filesystem
            self.path = path
            self.read_mode = read
            self.content = bytearray()

        def __enter__(self):
            return self

        def read(self, _size=-1):
            return self.filesystem.files[self.path]

        def write(self, data):
            self.content.extend(data)

        def __exit__(self, *_args):
            if not self.read_mode:
                self.filesystem.files[self.path] = bytes(self.content)

    class FileSystem:
        def __init__(self):
            self.files = {}

        def create_dir(self, _path, *, recursive):
            assert recursive

        def open_output_stream(self, path):
            return Stream(self, path)

        def open_input_file(self, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return Stream(self, path, read=True)

        def get_file_info(self, paths):
            return [
                SimpleNamespace(is_file=path in self.files) for path in paths
            ]

        def move(self, source, destination):
            matches = {
                destination + path[len(source) :]: content
                for path, content in self.files.items()
                if path == source or path.startswith(source + "/")
            }
            self.files = {
                path: content
                for path, content in self.files.items()
                if path != source and not path.startswith(source + "/")
            }
            self.files.update(matches)

        def delete_dir(self, source):
            self.files = {
                path: content
                for path, content in self.files.items()
                if path != source and not path.startswith(source + "/")
            }

        def delete_file(self, path):
            self.files.pop(path, None)

    filesystem = FileSystem()
    monkeypatch.setattr(
        mineru_taiji,
        "_hdfs_filesystem",
        lambda _uri: (None, filesystem, "/runs/job/docs"),
    )

    first = mineru_taiji._upload_spool_batch(
        str(batch), "hdfs://cluster/runs/job/docs"
    )
    second = mineru_taiji._upload_spool_batch(
        str(batch), "hdfs://cluster/runs/job/docs"
    )

    assert first["uploaded"] == 1
    assert second["skipped"] == 1
    assert filesystem.files["/runs/job/docs/paper/vlm/paper.md"] == b"markdown"
    assert "/runs/job/docs/paper/vlm/layout.json" in filesystem.files
    assert "/runs/job/docs/paper/_SUCCESS" in filesystem.files
    assert "/runs/job/_batches/batch-a.json" in filesystem.files

    ceph_documents = tmp_path / "ceph" / "run" / "docs"
    ceph_first = mineru_taiji._upload_spool_batch(
        str(batch), str(ceph_documents)
    )
    ceph_second = mineru_taiji._upload_spool_batch(
        str(batch), str(ceph_documents)
    )
    assert ceph_first["uploaded"] == 1
    assert ceph_second["skipped"] == 1
    assert (ceph_documents / "paper" / "vlm" / "paper.md").is_file()
    assert (ceph_documents / "paper" / "vlm" / "layout.json").is_file()
    assert (ceph_documents / "paper" / "_SUCCESS").is_file()


def test_ceph_mount_reads_token_on_node_without_returning_it(tmp_path, monkeypatch):
    token = "sensitive-pat-token"
    output = tmp_path / "ceph-output"
    output.mkdir()
    run_dir = output / "run-a"
    run_dir.mkdir()
    commands = []

    monkeypatch.setattr(
        mineru_taiji,
        "_assert_current_node",
        lambda node_id: node_id,
    )

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mineru_taiji.subprocess, "run", fake_run)
    result = mineru_taiji._mount_ceph_node(
        "node-a",
        token,
        "TaiJi_HYAide_LLM_Pretrain_Data",
        "gy",
        str(output),
        str(run_dir),
    )

    assert commands[0][0] == ["sudo", "taiji_client", "update"]
    assert commands[1][0] == [
        "sudo",
        "taiji_client",
        "mount",
        "-tk",
        token,
        "-bf",
        "TaiJi_HYAide_LLM_Pretrain_Data",
        "-l",
        "gy",
    ]
    assert commands[2][0] == ["sudo", "mkdir", "-p", str(run_dir)]
    assert commands[4][0] == ["sudo", "chmod", "0775", str(run_dir)]
    assert token not in json.dumps(result)
    assert not list(run_dir.glob(".rayorch-mount-probe-*"))


def test_pinned_tasks_use_hard_node_affinity(monkeypatch):
    strategies = []

    class FakeStrategy:
        def __init__(self, *, node_id, soft):
            strategies.append((node_id, soft))

    class FakeRemoteTask:
        def __init__(self):
            self.strategy = None

        def options(self, *, scheduling_strategy):
            clone = FakeRemoteTask()
            clone.strategy = scheduling_strategy
            return clone

        def remote(self, node_id, value):
            return {"node_id": node_id, "value": value}

    class FakeRay:
        @staticmethod
        def remote(*, num_cpus):
            assert num_cpus == 0
            return lambda task: FakeRemoteTask()

        @staticmethod
        def get(refs):
            return refs

    monkeypatch.setattr(mineru_taiji, "_node_affinity_strategy", lambda: FakeStrategy)
    nodes = (
        ClusterNode("node-a", "10.0.0.1", 0),
        ClusterNode("node-b", "10.0.0.2", 4),
    )

    results = _run_pinned_tasks(
        FakeRay(),
        lambda *_: {},
        [(nodes[0], (1,)), (nodes[1], (2,))],
    )

    assert strategies == [("node-a", False), ("node-b", False)]
    assert results == [
        {"node_id": "node-a", "value": 1},
        {"node_id": "node-b", "value": 2},
    ]


def test_benchmark_args_fix_scale_to_combined_3690_pdfs_and_64_gpus(tmp_path):
    args = build_parser().parse_args(
        [
            "--hdfs-pdf-uri",
            "hdfs://cluster/data/pdfs-2000",
            "--expected-pdf-files",
            "2000",
            "--expected-pdf-bytes",
            "4060449252",
            "--hdfs-pdf-uri",
            "hdfs://cluster/data/pdfs-1690",
            "--expected-pdf-files",
            "1690",
            "--expected-pdf-bytes",
            "1099397628624",
            "--hdfs-model-uri",
            "hdfs://cluster/models/mineru",
        ]
    )

    benchmark = _benchmark_args(args, tmp_path)

    assert benchmark.limit == 3_690
    assert benchmark.replicas == 64
    assert benchmark.gpus_per_ocr_actor == 1.0
    assert benchmark.ray_address == "auto"
    assert benchmark.pdf_dirs == [
        "hdfs://cluster/data/pdfs-2000",
        "hdfs://cluster/data/pdfs-1690",
    ]
    assert benchmark.model == str(tmp_path / "model")
    assert benchmark.actor_resources == {"accelerator_type:H20": 0.001}
    assert benchmark.actor_scheduling_strategy == "SPREAD"
    assert not benchmark.skip_existing
    assert args.expected_pdf_files == [2_000, 1_690]
    assert args.expected_pdf_bytes == [4_060_449_252, 1_099_397_628_624]
    assert args.expected_model_files == 15
    assert args.expected_model_bytes == 2_323_649_236


def test_hdfs_validation_requires_paired_unique_outputs(monkeypatch):
    infos = []
    for stem in ("document-a", "document-b"):
        prefix = f"/runs/job/docs/{stem}/vlm"
        infos.extend(
            [
                SimpleNamespace(path=f"{prefix}/{stem}.md"),
                SimpleNamespace(path=f"{prefix}/layout.json"),
                SimpleNamespace(path=f"/runs/job/docs/{stem}/_SUCCESS"),
            ]
        )
    monkeypatch.setattr(
        mineru_taiji,
        "_remote_file_infos",
        lambda uri: (None, None, "/runs/job", tuple(infos)),
    )

    assert _validate_hdfs_outputs(
        "hdfs://cluster/runs/job", expected_documents=2
    ) == {
        "markdown": 2,
        "layout": 2,
        "unique_documents": 2,
        "document_success": 2,
        "incomplete": 0,
        "staging_files": 0,
    }

    infos.append(
        SimpleNamespace(
            path="/runs/job/_staging/document-b.retry/vlm/partial.jpg"
        )
    )
    with pytest.raises(RuntimeError, match="staging files=1"):
        _validate_hdfs_outputs(
            "hdfs://cluster/runs/job", expected_documents=2
        )


def test_run_taiji_orchestrates_staging_benchmark_upload_and_success(
    tmp_path,
    monkeypatch,
):
    events = []
    dispatches = []

    class FakeRay:
        initialized = False
        init_calls = []

        @classmethod
        def is_initialized(cls):
            return cls.initialized

        @classmethod
        def init(cls, **kwargs):
            cls.initialized = True
            cls.init_calls.append(kwargs)

        @staticmethod
        def nodes():
            return _ray_records()

    def fake_dispatch(ray_module, task, calls):
        captured = [(node.node_id, arguments) for node, arguments in calls]
        dispatches.append((task.__name__, captured))
        return [
            {"node_id": node_id, "files": 2_000}
            for node_id, _ in captured
        ]

    def fake_benchmark(benchmark_args):
        events.append("benchmark")
        assert benchmark_args.limit == 3_690
        assert benchmark_args.replicas == 64
        assert benchmark_args.actor_resources == {"accelerator_type:H20": 0.001}
        assert benchmark_args.ray_address == "auto"
        assert benchmark_args.pdf_dirs == [
            "hdfs://cluster/data/pdfs-2000",
            "hdfs://cluster/data/pdfs-1690",
        ]
        assert benchmark_args.output_dir == str(
            tmp_path / "node-local" / "spool"
        )
        assert benchmark_args.remote_output_dir == (
            "hdfs://cluster/results/job-123/docs"
        )
        assert benchmark_args.completion_dir == benchmark_args.remote_output_dir
        assert benchmark_args.spool_batches
        assert benchmark_args.skip_existing
        artifact_dir = Path(benchmark_args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        return {
            "status": "completed",
            "discovered_pdfs": 3_690,
            "skipped_existing": 0,
            "n_pdf": 3_690,
            "docs": 3_690,
            "pages": 182_000,
            "startup_s": 1.0,
            "measured_wall_s": 10.0,
        }

    def fake_validate(uri, *, expected_documents):
        events.append("validate")
        assert expected_documents == 3_690
        return {"markdown": 3_690, "layout": 3_690, "unique_documents": 3_690}

    uploads = []

    def fake_upload(local_dir, uri):
        events.append("driver-upload")
        uploads.append((local_dir, uri))
        return {"files": 2, "bytes": 10, "uri": uri}

    def fake_success(uri, payload):
        events.append("success")
        assert payload["validation"]["markdown"] == 3_690
        return uri + "/_SUCCESS"

    monkeypatch.setattr(
        mineru_taiji,
        "_prepare_hdfs_run_root",
        lambda uri: events.append("prepare"),
    )
    monkeypatch.setattr(mineru_taiji, "_run_pinned_tasks", fake_dispatch)
    monkeypatch.setattr(mineru_taiji, "_validate_hdfs_outputs", fake_validate)
    monkeypatch.setattr(mineru_taiji, "_upload_tree", fake_upload)
    monkeypatch.setattr(mineru_taiji, "_write_success", fake_success)
    monkeypatch.setattr(
        mineru_taiji,
        "_start_gpu_samplers",
        lambda *args, **kwargs: ["sampler"],
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_start_spool_uploaders",
        lambda *args, **kwargs: ["uploader"],
    )
    drain_calls = []
    monkeypatch.setattr(
        mineru_taiji,
        "_drain_spool_uploaders",
        lambda *args, **kwargs: drain_calls.append(kwargs) or [
            {"completed_documents": 3_690}
        ],
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_stop_spool_uploaders",
        lambda *args, **kwargs: events.append("stop-uploaders"),
    )
    monkeypatch.setattr(
        mineru_taiji,
        "_stop_gpu_samplers",
        lambda *args, **kwargs: {
            "active_mean_gpu_utilization": 95.0,
            "sample_buckets": 10,
        },
    )
    args = build_parser().parse_args(
        [
            "--hdfs-pdf-uri",
            "hdfs://cluster/data/pdfs-2000",
            "--expected-pdf-files",
            "2000",
            "--expected-pdf-bytes",
            "4060449252",
            "--hdfs-pdf-uri",
            "hdfs://cluster/data/pdfs-1690",
            "--expected-pdf-files",
            "1690",
            "--expected-pdf-bytes",
            "1099397628624",
            "--hdfs-model-uri",
            "hdfs://cluster/data/model",
            "--hdfs-output-uri",
            "hdfs://cluster/results",
            "--run-id",
            "job-123",
            "--local-root",
            str(tmp_path / "node-local"),
            "--topology-poll-s",
            "0",
        ]
    )

    result = run_taiji(
        args,
        ray_module=FakeRay,
        benchmark_runner=fake_benchmark,
    )

    assert FakeRay.init_calls == [{"address": "auto"}]
    assert [name for name, _ in dispatches] == [
        "_prepare_hdfs_pdf_inputs_node",
        "_stage_model_node",
    ]
    expected_nodes = [f"gpu-{i}" for i in range(8)]
    assert [node_id for node_id, _ in dispatches[0][1]] == expected_nodes
    assert [node_id for node_id, _ in dispatches[1][1]] == expected_nodes
    assert dispatches[0][1][0][1][1:3] == (
        (2_000, 1_690),
        (4_060_449_252, 1_099_397_628_624),
    )
    assert dispatches[0][1][0][1][3] == str(
        tmp_path / "node-local" / "spool"
    )
    assert dispatches[1][1][0][1][-2:] == (15, 2_323_649_236)
    assert uploads == [
        (
            str(tmp_path / "node-local" / "artifacts"),
            "hdfs://cluster/results/job-123/driver/artifacts",
        )
    ]
    assert events[-3:] == ["validate", "driver-upload", "success"]
    assert result["output_uploads"] == [
        {"completed_documents": 3_690}
    ]
    assert result["output_mode"] == "local_spool_async_hdfs"
    assert result["document_output_uri"] == "hdfs://cluster/results/job-123/docs"
    assert len(drain_calls) == 2
    assert result["success_uri"] == "hdfs://cluster/results/job-123/_SUCCESS"
