"""MinerU command-line contract tests."""

from rayorch.benchmark.mineru.runner import build_parser


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
    assert args.input_batch_size == 24
    assert args.max_active_input_batches == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
    assert args.input_manifest is None
    assert args.golden_corpus_sha256 is None


def test_mineru_cli_separates_input_batches_from_worker_batches():
    args = build_parser().parse_args(
        [
            "--output-dir", "/tmp/out",
            "--artifact-dir", "/tmp/artifacts",
            "--result-jsonl", "/tmp/results.jsonl",
            "--input-batch-size", "8",
            "--max-active-input-batches", "2",
            "--batch-size", "32",
        ]
    )

    assert args.input_batch_size == 8
    assert args.max_active_input_batches == 2
    assert args.batch_size == 32
