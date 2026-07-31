# Multigrain V3 tests

Default suite:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -m pytest -q test/experimental/multigrain_v3
```

Layout:

```text
unit/
    identity, compile validation, worker ABI, Arena hard limits

integration/
    real local-Ray Pipeline execution, nested Expand/Reduce, recovery,
    optional branches, and multi-Arena overlap
```

The real 4×H20 MinerU regression is intentionally a manual benchmark:

```text
rayorch.experimental.multigrain_v3.benchmark.mineru
```

It is not part of pytest because it initializes four vLLM models and processes
hundreds of PDFs.

