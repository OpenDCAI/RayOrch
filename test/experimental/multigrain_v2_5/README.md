# Multigrain V2.5 test and benchmark layout

## Default correctness suite

```bash
python -m pytest -q test/experimental/multigrain_v2_5
```

The default suite includes:

- independent reference semantics and golden identity vectors;
- semantic-core unit tests;
- single-process executor and generation-fencing tests;
- real local-Ray integration tests;
- native `Pipeline.forward` DAG tests;
- fast fake-clock batch-trigger properties.

It skips tests marked `slow`.

## Structured trigger tests

```text
unit/
  deterministic fake-clock trigger, metrics, randomized schedule properties

integration/
  local-Ray actor capacity, dispatch timeline, parent completion metrics

stress/
  opt-in 10k-grain pressure and real-Ray skew/benchmark smoke tests
```

Every test in these directories has a docstring describing the contract it
protects.

## Slow pressure and benchmark smoke

```bash
python -m pytest -q --runslow \
  test/experimental/multigrain_v2_5/stress
```

These tests are correctness/load regressions, not stable performance
thresholds. They verify output parity, bounded batching, report generation,
and directionally non-worse RPC coalescing.

## Manual experiment matrix

The full matrix is intentionally a manual CLI rather than a pytest test:

```bash
python -m rayorch.experimental.multigrain_v2_5.benchmark \
  --output-dir benchmark_results/example \
  --parents 500 \
  --fanout-modes uniform,lognormal,pareto \
  --service-modes constant,lognormal \
  --actors 1,4 \
  --batch-sizes 4,16 \
  --wait-ms 2,5 \
  --seeds 1,2,3,4,5 \
  --warmups 1 \
  --repetitions 5
```

Outputs:

```text
raw_trials.jsonl
summary.json
summary.csv
```

Each paired repetition randomizes parent-bound/elastic execution order and
requires an identical final-output digest.

In environments where Ray's automatic uv runtime hook cannot inspect parent
processes, prefix commands with:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
```

## Manual real MinerU benchmark

The real-model benchmark is deliberately not a pytest test. It requires the
Flash-MinerU checkout, MinerU model weights, a large object store, and four H20
GPUs:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -u -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_cli \
  --mode elastic \
  --limit 368 \
  --replicas 4 \
  --batch-size 256 \
  --max-batch-wait-ms 20 \
  --gpu-memory-utilization 0.8 \
  --object-store-gb 100 \
  --output-dir /path/to/outputs
```

The CLI records model startup separately from measured pipeline wall time and
reports driver/worker RSS, per-GPU peak memory, live coarse blocks, OCR bubble,
parent completion percentiles, RPC coalescing, and throughput.
