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
    assert args.max_active_microbatches == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
    assert args.input_manifest is None
    assert args.golden_corpus_sha256 is None
