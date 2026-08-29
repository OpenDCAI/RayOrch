"""MinerU v3.6 Pipeline 的结构、UDF 复用和 CLI gate 测试。"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3.benchmark import mineru as v3_mineru
from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
    _available_vllm_port_for_pid,
    _read_binary_path,
    _vllm_port_for_pid,
)
from rayorch.experimental.multigrain_v3_6.benchmark.mineru import (
    MinerUV36Pipeline,
    _discover_pdf_inputs,
    _discover_pdfs,
    _output_is_complete,
    _select_pdfs,
    build_parser,
    run_benchmark,
)
from rayorch.experimental.multigrain_v3_6.program.logical import ExpandOrigin, ReduceOrigin


def _pipeline(mode: str = "elastic") -> MinerUV36Pipeline:
    return MinerUV36Pipeline(
        output_dir="/tmp/v36-mineru-test",
        mode=mode,
        model="model",
        replicas=4,
        batch_size=64,
        gpu_memory_utilization=0.8,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )


def test_mineru_v36_has_four_calls_and_no_structural_pools():
    compiled = _pipeline().compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]

    assert targets == [
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        PdfMetadata,
        MinerUAssembleDoc,
    ]
    assert len(compiled.plan.actor_pools_by_call) == 4
    assert len(compiled.logical.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, ReduceOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2


def test_mineru_parent_and_elastic_only_change_ocr_pool_option():
    elastic = _pipeline("elastic").compile()
    parent = _pipeline("parent_bound").compile()

    assert elastic.logical.calls == parent.logical.calls
    elastic_options = list(elastic.plan.actor_pools_by_call.values())
    parent_options = list(parent.plan.actor_pools_by_call.values())
    changed = [
        (left, right)
        for left, right in zip(elastic_options, parent_options)
        if left != right
    ]
    assert len(changed) == 1
    assert changed[0][0].batch_scope == "elastic"
    assert changed[0][1].batch_scope == "parent_bound"


def test_mineru_ocr_pool_supports_fractional_gpu_actors():
    pipeline = MinerUV36Pipeline(
        output_dir="/tmp/v36-mineru-fractional-test",
        mode="elastic",
        model="model",
        replicas=8,
        batch_size=64,
        gpu_memory_utilization=0.4,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
        gpus_per_ocr_actor=0.5,
    ).compile()
    ocr = list(pipeline.plan.actor_pools_by_call.values())[1]

    assert ocr.replicas == 8
    assert dict(ocr.ray_options)["num_gpus"] == 0.5


def test_mineru_cli_defaults_to_four_pdf_correctness_gate():
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--artifact-dir",
            "/tmp/artifacts",
            "--result-jsonl",
            "/tmp/results.jsonl",
        ]
    )

    assert args.limit == 4
    assert args.batch_size == 64
    assert args.gpus_per_ocr_actor == 1.0
    assert args.max_active_microbatches == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
    assert args.pdf_dir is None
    assert not args.skip_existing


def test_mineru_cli_accepts_fractional_gpu_actors():
    args = build_parser().parse_args(
        [
            "--gpus-per-ocr-actor",
            "0.5",
            "--output-dir",
            "/tmp/out",
            "--artifact-dir",
            "/tmp/artifacts",
            "--result-jsonl",
            "/tmp/results.jsonl",
        ]
    )

    assert args.gpus_per_ocr_actor == 0.5


def test_mineru_discovers_pdfs_independently_from_flash_repo(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"second")
    (tmp_path / "a.pdf").write_bytes(b"first")
    (tmp_path / "ignored.txt").write_text("not a PDF")

    assert _discover_pdfs(str(tmp_path)) == [
        str((tmp_path / "a.pdf").resolve()),
        str((tmp_path / "b.pdf").resolve()),
    ]


def test_mineru_combines_multiple_pdf_roots_in_declared_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.pdf").write_bytes(b"a")
    (second / "b.pdf").write_bytes(b"b")

    assert _discover_pdf_inputs([str(first), str(second)]) == [
        str((first / "a.pdf").resolve()),
        str((second / "b.pdf").resolve()),
    ]


def test_mineru_combined_roots_reject_duplicate_output_stems(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "same.pdf").write_bytes(b"a")
    (second / "same.pdf").write_bytes(b"b")

    with pytest.raises(ValueError, match="duplicate output stems: same"):
        _discover_pdf_inputs([str(first), str(second)])


def test_mineru_discovers_hdfs_pdfs_and_preserves_uri(monkeypatch):
    directory = object()
    file = object()

    class FakeFileSelector:
        def __init__(self, base_dir, *, recursive, allow_not_found):
            assert base_dir == "/datasets/pdfs"
            assert recursive is True
            assert allow_not_found is False

    class FakeFileSystem:
        def get_file_info(self, target):
            if isinstance(target, str):
                assert target == "/datasets/pdfs"
                return SimpleNamespace(type=directory)
            assert isinstance(target, FakeFileSelector)
            return [
                SimpleNamespace(type=file, path="/datasets/pdfs/b.pdf"),
                SimpleNamespace(type=directory, path="/datasets/pdfs/nested"),
                SimpleNamespace(type=file, path="/datasets/pdfs/ignored.txt"),
                SimpleNamespace(type=file, path="/datasets/pdfs/a.pdf"),
            ]

    fake_filesystem = FakeFileSystem()
    fake_fs = SimpleNamespace(
        FileSystem=SimpleNamespace(
            from_uri=lambda uri: (
                fake_filesystem,
                "/datasets/pdfs",
            )
        ),
        FileSelector=FakeFileSelector,
        FileType=SimpleNamespace(Directory=directory, File=file),
    )
    fake_pyarrow = ModuleType("pyarrow")
    fake_pyarrow.fs = fake_fs
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)

    assert _discover_pdfs("hdfs://nameservice/datasets/pdfs") == [
        "hdfs://nameservice/datasets/pdfs/a.pdf",
        "hdfs://nameservice/datasets/pdfs/b.pdf",
    ]


def test_mineru_hdfs_discovery_rejects_duplicate_output_stems(monkeypatch):
    directory = object()
    file = object()

    class FakeFileSystem:
        def get_file_info(self, target):
            if isinstance(target, str):
                return SimpleNamespace(type=directory)
            return [
                SimpleNamespace(type=file, path="/pdfs/a/report.pdf"),
                SimpleNamespace(type=file, path="/pdfs/b/report.PDF"),
            ]

    fake_fs = SimpleNamespace(
        FileSystem=SimpleNamespace(
            from_uri=lambda _uri: (FakeFileSystem(), "/pdfs")
        ),
        FileSelector=lambda *args, **kwargs: (args, kwargs),
        FileType=SimpleNamespace(Directory=directory, File=file),
    )
    fake_pyarrow = ModuleType("pyarrow")
    fake_pyarrow.fs = fake_fs
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)

    with pytest.raises(ValueError, match="duplicate output stems: report"):
        _discover_pdfs("hdfs://nameservice/pdfs")


def test_mineru_reads_hdfs_bytes_and_reuses_filesystem(monkeypatch):
    opened = []
    from_uri_calls = []

    class FakeInput:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return self.path.encode()

    class FakeFileSystem:
        def normalize_path(self, path):
            return path

        def open_input_file(self, path):
            opened.append(path)
            return FakeInput(path)

    fake_filesystem = FakeFileSystem()

    def from_uri(uri):
        from_uri_calls.append(uri)
        return fake_filesystem, url_path(uri)

    def url_path(uri):
        return "/" + uri.split("/", 3)[-1]

    fake_pyarrow = ModuleType("pyarrow")
    fake_pyarrow.fs = SimpleNamespace(
        FileSystem=SimpleNamespace(from_uri=from_uri)
    )
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    cache = {}

    assert _read_binary_path(
        "hdfs://nameservice/pdfs/a.pdf",
        hdfs_filesystems=cache,
    ) == b"/pdfs/a.pdf"
    assert _read_binary_path(
        "hdfs://nameservice/pdfs/b.pdf",
        hdfs_filesystems=cache,
    ) == b"/pdfs/b.pdf"
    assert from_uri_calls == ["hdfs://nameservice/pdfs/a.pdf"]
    assert opened == ["/pdfs/a.pdf", "/pdfs/b.pdf"]


def test_mineru_reads_local_bytes_without_pyarrow(tmp_path, monkeypatch):
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"local-pdf")
    monkeypatch.setitem(sys.modules, "pyarrow", None)

    assert _read_binary_path(str(pdf)) == b"local-pdf"


def test_vllm_actor_ports_use_disjoint_eight_port_slots():
    ports = [_vllm_port_for_pid(pid) for pid in range(1000, 1024)]
    assert len(set(ports)) == 24
    assert all(right - left == 8 for left, right in zip(ports, ports[1:]))


def test_available_vllm_port_skips_an_occupied_slot(monkeypatch):
    pid = 1234
    occupied_port = _vllm_port_for_pid(pid)

    class FakeSocket:
        def __init__(self, *_args):
            pass

        def bind(self, address):
            if address[1] == occupied_port:
                raise OSError("occupied")

        def close(self):
            return None

    monkeypatch.setattr(v3_mineru.socket, "socket", FakeSocket)
    selected = _available_vllm_port_for_pid(pid)

    assert selected == occupied_port + 8


def test_mineru_render_failure_retries_then_preserves_empty_group(
    monkeypatch, capsys
):
    calls = []

    def fail_load(_content, *, dpi):
        calls.append(dpi)
        raise RuntimeError("transient or malformed PDF")

    modules = {
        name: ModuleType(name)
        for name in (
            "flash_mineru",
            "flash_mineru.mineru_core",
            "flash_mineru.mineru_core.utils",
            "flash_mineru.mineru_core.utils.pdf_image_tools",
        )
    }
    modules[
        "flash_mineru.mineru_core.utils.pdf_image_tools"
    ].load_images_from_pdf = fail_load
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(v3_mineru, "_read_binary_path", lambda *a, **k: b"bad")
    monkeypatch.setattr(v3_mineru.time, "sleep", lambda _delay: None)

    assert MinerUPdfToPages().run(["hdfs://cluster/bad.pdf"]) == [[]]
    assert calls == [200, 200, 200]
    assert "RAYORCH_MINERU_RENDER_FAILED" in capsys.readouterr().out


def test_mineru_hdfs_metadata_uses_uri_path_stem():
    assert PdfMetadata().run(
        ["hdfs://nameservice/pdfs/report.v1.pdf?version=2"]
    ) == ["report.v1"]


def test_mineru_assemble_commits_each_document_directly_to_hdfs(
    monkeypatch,
    tmp_path,
):
    class Sink:
        def __init__(self, filesystem, path):
            self.filesystem = filesystem
            self.path = path
            self.content = bytearray()

        def __enter__(self):
            return self

        def write(self, data):
            self.content.extend(data)

        def __exit__(self, *_args):
            self.filesystem.files[self.path] = bytes(self.content)

    class FakeFileSystem:
        def __init__(self):
            self.files = {}
            self.moves = []

        def create_dir(self, _path, *, recursive):
            assert recursive

        def open_output_stream(self, path):
            return Sink(self, path)

        def open_input_file(self, path):
            content = self.files[path]

            class Source:
                def __enter__(self):
                    return self

                def read(self):
                    return content

                def __exit__(self, *_args):
                    return None

            return Source()

        def move(self, source, destination):
            self.moves.append((source, destination))
            moved = {
                destination + path[len(source) :]: content
                for path, content in self.files.items()
                if path == source or path.startswith(source + "/")
            }
            self.files = {
                path: content
                for path, content in self.files.items()
                if path != source and not path.startswith(source + "/")
            }
            self.files.update(moved)

        def delete_dir(self, source):
            self.files = {
                path: content
                for path, content in self.files.items()
                if path != source and not path.startswith(source + "/")
            }

    modules = {
        name: ModuleType(name)
        for name in (
            "flash_mineru",
            "flash_mineru.mineru_core",
            "flash_mineru.mineru_core.data",
            "flash_mineru.mineru_core.data.data_reader_writer",
            "flash_mineru.mineru_core.engine",
            "flash_mineru.mineru_core.engine.model_output_to_middle_json",
            "flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent",
            "flash_mineru.mineru_core.utils",
            "flash_mineru.mineru_core.utils.enum_class",
        )
    }
    class FakeLocalWriter:
        def __init__(self, parent):
            self.parent = Path(parent)

        def write(self, path, data):
            target = self.parent / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    modules[
        "flash_mineru.mineru_core.data.data_reader_writer"
    ].FileBasedDataWriter = FakeLocalWriter

    def fake_middle(_contents, _pages, writer):
        writer.write("image.jpg", b"image")
        return {"pdf_info": []}

    modules[
        "flash_mineru.mineru_core.engine.model_output_to_middle_json"
    ].result_to_middle_json = fake_middle
    modules[
        "flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent"
    ].union_make = lambda *_args: "markdown"
    modules["flash_mineru.mineru_core.utils.enum_class"].MakeMode = (
        SimpleNamespace(MM_MD="mm-md")
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    filesystem = FakeFileSystem()
    monkeypatch.setattr(
        v3_mineru,
        "_hdfs_filesystem_and_path",
        lambda _uri, _cache: (filesystem, "/runs/job/docs"),
    )

    outputs = MinerUAssembleDoc("hdfs://cluster/runs/job/docs").run(
        [[object()]],
        [[{"page_id": 0}]],
        ["document"],
    )

    assert outputs == [
        {
            "pdf": "document",
            "md_path": (
                "hdfs://cluster/runs/job/docs/document/vlm/document.md"
            ),
            "chars": 8,
            "pages": 1,
            "status": "completed",
        }
    ]
    assert (
        filesystem.files[
            "/runs/job/docs/document/vlm/images/image.jpg"
        ]
        == b"image"
    )
    assert filesystem.files["/runs/job/docs/document/vlm/document.md"] == b"markdown"
    assert "/runs/job/docs/document/vlm/layout.json" in filesystem.files
    assert "/runs/job/docs/document/_SUCCESS" in filesystem.files
    assert not any("_staging" in path for path in filesystem.files)
    assert len(filesystem.moves) == 1

    repeated = MinerUAssembleDoc("hdfs://cluster/runs/job/docs").run(
        [[object()]],
        [[{"page_id": 0}]],
        ["document"],
    )
    assert repeated == outputs
    assert len(filesystem.moves) == 1

    spool_root = tmp_path / "spool"
    spooled = MinerUAssembleDoc(
        str(spool_root),
        spool_batches=True,
        remote_output_dir="hdfs://cluster/runs/job/docs",
    ).run(
        [[object()], [object()]],
        [[{"page_id": 0}], [{"page_id": 0}]],
        ["first", "second"],
    )
    ready_batches = list((spool_root / "ready").iterdir())
    assert len(ready_batches) == 1
    assert not list((spool_root / "staging").iterdir())
    manifest = json.loads(
        (ready_batches[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert [item["pdf"] for item in manifest["documents"]] == [
        "first",
        "second",
    ]
    assert all(item["md_path"].startswith("hdfs://") for item in spooled)
    for stem in ("first", "second"):
        document = ready_batches[0] / "docs" / stem
        assert (document / "vlm" / f"{stem}.md").is_file()
        assert (document / "vlm" / "layout.json").is_file()
        assert (document / "_SUCCESS").is_file()


@pytest.mark.parametrize("limit", [0, -1])
def test_mineru_discovery_rejects_non_positive_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="limit must be positive"):
        _select_pdfs(
            str(tmp_path),
            str(tmp_path / "results"),
            limit=limit,
            skip_existing=False,
        )


def test_mineru_complete_output_requires_markdown_and_layout(tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"pdf")
    result_dir = tmp_path / "results" / "input" / "vlm"
    result_dir.mkdir(parents=True)

    assert not _output_is_complete(str(pdf), str(tmp_path / "results"))
    (result_dir / "input.md").write_text("markdown")
    assert not _output_is_complete(str(pdf), str(tmp_path / "results"))
    (result_dir / "layout.json").write_text("{}")
    assert not _output_is_complete(str(pdf), str(tmp_path / "results"))
    (result_dir / "layout.json").write_text('{"pdf_info": []}')
    assert _output_is_complete(str(pdf), str(tmp_path / "results"))
    (result_dir / ".rayorch-incomplete").write_text("in progress")
    assert not _output_is_complete(str(pdf), str(tmp_path / "results"))


def test_mineru_complete_output_allows_empty_markdown(tmp_path):
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"pdf")
    result_dir = tmp_path / "results" / "blank" / "vlm"
    result_dir.mkdir(parents=True)
    (result_dir / "blank.md").write_text("")
    (result_dir / "layout.json").write_text('{"pdf_info": []}')

    assert _output_is_complete(str(pdf), str(tmp_path / "results"))


def test_mineru_hdfs_complete_output_requires_document_success(monkeypatch):
    file_type = object()
    missing_type = object()

    class Source:
        def __enter__(self):
            return self

        def read(self):
            return b'{"pdf_info": []}'

        def __exit__(self, *_args):
            return None

    class FileSystem:
        marker_present = True

        @classmethod
        def from_uri(cls, uri):
            assert uri == "hdfs://cluster/results/docs"
            return cls(), "/results/docs"

        def get_file_info(self, paths):
            assert paths[-1] == "/results/docs/input/_SUCCESS"
            return [
                SimpleNamespace(type=file_type),
                SimpleNamespace(type=file_type),
                SimpleNamespace(
                    type=file_type if self.marker_present else missing_type
                ),
            ]

        def open_input_file(self, path):
            assert path == "/results/docs/input/vlm/layout.json"
            return Source()

    fs_module = ModuleType("pyarrow.fs")
    fs_module.FileSystem = FileSystem
    fs_module.FileType = SimpleNamespace(File=file_type)
    pyarrow_module = ModuleType("pyarrow")
    pyarrow_module.fs = fs_module
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_module)
    monkeypatch.setitem(sys.modules, "pyarrow.fs", fs_module)

    assert _output_is_complete(
        "hdfs://cluster/pdfs/input.pdf",
        "hdfs://cluster/results/docs",
    )
    FileSystem.marker_present = False
    assert not _output_is_complete(
        "hdfs://cluster/pdfs/input.pdf",
        "hdfs://cluster/results/docs",
    )


def test_mineru_resume_applies_limit_after_skipping_complete_outputs(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.pdf").write_bytes(b"pdf")
    complete = tmp_path / "results" / "a" / "vlm"
    complete.mkdir(parents=True)
    (complete / "a.md").write_text("done")
    (complete / "layout.json").write_text('{"pdf_info": []}')

    selected, skipped, discovered = _select_pdfs(
        str(tmp_path),
        str(tmp_path / "results"),
        limit=1,
        skip_existing=True,
    )

    assert selected == [str((tmp_path / "b.pdf").resolve())]
    assert skipped == 1
    assert discovered == 3


def test_mineru_no_work_preserves_benchmark_artifacts(tmp_path):
    pdf = tmp_path / "done.pdf"
    pdf.write_bytes(b"pdf")
    output = tmp_path / "output"
    complete = output / "done" / "vlm"
    complete.mkdir(parents=True)
    (complete / "done.md").write_text("done")
    (complete / "layout.json").write_text('{"pdf_info": []}')
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    summary = artifacts / "summary.json"
    samples = artifacts / "gpu_samples.jsonl"
    results = tmp_path / "results.jsonl"
    summary.write_text("previous summary")
    samples.write_text("previous samples")
    results.write_text("previous result\n")

    payload = run_benchmark(
        Namespace(
            flash_repo=str(tmp_path),
            pdf_dir=str(tmp_path),
            output_dir=str(output),
            skip_existing=True,
            limit=1,
            mode="elastic",
            artifact_dir=str(artifacts),
            result_jsonl=str(results),
        )
    )

    assert payload["status"] == "already_complete"
    assert summary.read_text() == "previous summary"
    assert samples.read_text() == "previous samples"
    assert results.read_text() == "previous result\n"
    resume = json.loads((artifacts / "resume-status.json").read_text())
    assert resume["status"] == "already_complete"
