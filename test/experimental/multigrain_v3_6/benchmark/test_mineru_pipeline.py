"""MinerU v3.6 Pipeline 的结构、UDF 复用和 CLI gate 测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.experimental.multigrain_v3_6.benchmark.mineru import (
    MetadataOnlyMinerUAssembleDoc,
    MinerUV36Pipeline,
    PoisonTolerantMinerUAssembleDoc,
    PoisoningMinerUVlmOcrPage,
    build_parser,
)
from rayorch.experimental.multigrain_v3_6.benchmark.mineru_poison import (
    DEFAULT_POISON_SEED,
    PoisonPage,
    PoisonedPage,
    load_pdf_manifest,
    select_poison_pages,
)
from rayorch.experimental.multigrain_v3_6.protocol import GroupFailure
from rayorch.experimental.multigrain_v3_6.program.logical import ExpandOrigin, ReduceOrigin


def _pipeline(batching_policy: str = "any_parent") -> MinerUV36Pipeline:
    return MinerUV36Pipeline(
        output_dir="/tmp/v36-mineru-test",
        batching_policy=batching_policy,
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


def test_mineru_batching_policy_only_changes_ocr_pool_option():
    any_parent = _pipeline("any_parent").compile()
    parent = _pipeline("single_parent").compile()

    assert any_parent.logical.calls == parent.logical.calls
    any_parent_options = list(any_parent.plan.actor_pools_by_call.values())
    parent_options = list(parent.plan.actor_pools_by_call.values())
    changed = [
        (left, right)
        for left, right in zip(any_parent_options, parent_options)
        if left != right
    ]
    assert len(changed) == 1
    assert changed[0][0].batching_policy == "any_parent"
    assert changed[0][1].batching_policy == "single_parent"


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
    assert args.batching_policy == "any_parent"
    assert args.max_active_microbatches == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
    assert args.poison_pdf_index is None
    assert args.poison_count == 0
    assert args.poison_page_id == 0
    assert args.poison_policy == "group_failure"
    assert args.poison_seed == DEFAULT_POISON_SEED
    assert args.input_manifest is None
    assert args.assemble_mode == "full"
    assert args.ready_queue_order == "fifo"


def test_metadata_only_assemble_keeps_identity_and_drops_skip_page_marker():
    assembler = MetadataOnlyMinerUAssembleDoc()
    output = assembler.run(
        [["ocr:0", PoisonedPage("bad"), "ocr:2"]],
        [[{"page_id": 0}, {"page_id": 1}, {"page_id": 2}]],
        ["doc"],
    )[0]

    assert output == {
        "pdf": "doc",
        "pages": 2,
        "page_ids": [0, 2],
        "input_pages": 3,
        "poisoned_pages": 1,
        "poison_page_ids": [1],
    }


def test_frozen_pdf_manifest_preserves_order_counts_and_limit(tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text(
        '{"path":"/data/b.pdf","pages":9}\n'
        '{"path":"/data/a.pdf","pages":3}\n',
        encoding="utf-8",
    )

    assert load_pdf_manifest(str(manifest)) == (
        ("/data/b.pdf", "/data/a.pdf"),
        (9, 3),
    )
    assert load_pdf_manifest(str(manifest), limit=1) == (
        ("/data/b.pdf",),
        (9,),
    )


def test_frozen_pdf_manifest_accepts_json_array(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        '[{"path":"/data/a.pdf","pages":3},'
        '{"path":"/data/b.pdf","pages":9}]',
        encoding="utf-8",
    )

    assert load_pdf_manifest(str(manifest)) == (
        ("/data/a.pdf", "/data/b.pdf"),
        (3, 9),
    )


def test_mineru_poison_wrapper_skips_only_the_marked_page_before_real_ocr(tmp_path):
    poison_pdf = tmp_path / "poison.pdf"
    healthy_pdf = tmp_path / "healthy.pdf"
    worker = PoisoningMinerUVlmOcrPage.__new__(PoisoningMinerUVlmOcrPage)
    worker.poison_pages = {(str(poison_pdf.resolve()), 1): "bad page"}
    worker.poison_policy = "group_failure"

    class Client:
        def __init__(self):
            self.images = None

        def batch_two_step_extract(self, *, images):
            self.images = images
            return [f"ocr:{image}" for image in images]

    worker.client = Client()
    result = worker.run(
        [
            {"pdf_path": str(poison_pdf), "page_id": 0, "img_pil": "p0"},
            {"pdf_path": str(poison_pdf), "page_id": 1, "img_pil": "p1"},
            {"pdf_path": str(healthy_pdf), "page_id": 1, "img_pil": "h1"},
        ]
    )

    assert worker.client.images == ["p0", "h1"]
    assert result[0] == "ocr:p0"
    assert isinstance(result[1], GroupFailure)
    assert result[1].cause == "bad page"
    assert result[2] == "ocr:h1"


def test_mineru_poison_pipeline_changes_only_ocr_udf_target(tmp_path):
    baseline = _pipeline().compile()
    manifest = (
        PoisonPage(
            pdf_index=0,
            pdf_path=str(tmp_path / "poison.pdf"),
            page_id=3,
            pdf_pages=4,
            cause="bad page",
        ),
    )
    poisoned = MinerUV36Pipeline(
        output_dir=str(tmp_path),
        batching_policy="any_parent",
        model="model",
        replicas=4,
        batch_size=64,
        gpu_memory_utilization=0.8,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
        poison_manifest=manifest,
    ).compile()

    baseline_targets = [
        spec.udf.target for spec in baseline.logical.calls.values()
    ]
    poisoned_targets = [
        spec.udf.target for spec in poisoned.logical.calls.values()
    ]
    changed = [
        (left, right)
        for left, right in zip(baseline_targets, poisoned_targets)
        if left is not right
    ]
    assert changed == [(MinerUVlmOcrPage, PoisoningMinerUVlmOcrPage)]


def test_skip_page_policy_is_explicit_and_keeps_healthy_pairs(tmp_path, monkeypatch):
    poison_pdf = tmp_path / "poison.pdf"
    worker = PoisoningMinerUVlmOcrPage.__new__(PoisoningMinerUVlmOcrPage)
    worker.poison_pages = {(str(poison_pdf.resolve()), 1): "bad page"}
    worker.poison_policy = "skip_page"

    class Client:
        def batch_two_step_extract(self, *, images):
            return [f"ocr:{image}" for image in images]

    worker.client = Client()
    result = worker.run(
        [
            {"pdf_path": str(poison_pdf), "page_id": 0, "img_pil": "p0"},
            {"pdf_path": str(poison_pdf), "page_id": 1, "img_pil": "p1"},
        ]
    )
    assert result[0] == "ocr:p0"
    assert result[1] == PoisonedPage("bad page")

    captured = {}

    def fake_assemble(self, contents, pages, stems):
        captured.update(contents=contents, pages=pages, stems=stems)
        return [{"pdf": stems[0], "pages": len(pages[0])}]

    monkeypatch.setattr(MinerUAssembleDoc, "run", fake_assemble)
    assembler = PoisonTolerantMinerUAssembleDoc.__new__(
        PoisonTolerantMinerUAssembleDoc
    )
    output = assembler.run(
        [["ocr:p0", PoisonedPage("bad page")]],
        [[{"page_id": 0}, {"page_id": 1}]],
        ["doc"],
    )[0]
    assert captured["contents"] == [["ocr:p0"]]
    assert captured["pages"] == [[{"page_id": 0}]]
    assert output == {
        "pdf": "doc",
        "pages": 1,
        "input_pages": 2,
        "poisoned_pages": 1,
        "poison_page_ids": [1],
    }


def test_poison_manifest_is_nested_reproducible_and_exact():
    pdfs = [f"/data/{index}.pdf" for index in range(10)]
    counts = [index + 1 for index in range(10)]
    small = select_poison_pages(pdfs, counts, count=3, seed="seed")
    large = select_poison_pages(pdfs, counts, count=6, seed="seed")

    assert {entry.pdf_index for entry in small} < {
        entry.pdf_index for entry in large
    }
    assert all(entry.page_id == 0 for entry in large)
    exact = select_poison_pages(
        pdfs,
        counts,
        count=0,
        exact_indices=(7,),
    )
    assert [(entry.pdf_index, entry.pdf_pages) for entry in exact] == [(7, 8)]
