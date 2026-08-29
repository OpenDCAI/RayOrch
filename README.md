# RayOrch

Lightweight orchestration utilities for building asynchronous Ray pipelines with
`RayModule`, overlapped microbatch execution, and DAG-style scheduling.

## Install

```bash
pip install rayorch
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Reproduce the 64-GPU MinerU comparison

The frozen TaiJi WeData launch contracts, environment pins, and commands for
the 8 x 8 H20 RayOrch and Ray Data experiments are documented in
[`experiments/mineru_64gpu_repro/README.md`](experiments/mineru_64gpu_repro/README.md).

## Core Concepts

- `RayModule`: wraps an operator class into Ray actors with optional replica
  dispatch and collect.
- `OverlappedPipeline`: graphless microbatch overlap with backpressure.
- `DagPipeline` / `DagExecutor`: symbolic multi-input/multi-output DAG
  scheduling.
- `RuntimeRayModule` / `RuntimeDagExecutor`: row-level fault isolation,
  lineage, Ray replicas, and overlapped microbatch execution.

Runtime MVP usage and current limitations are documented in
[`docs/runtime_mvp.md`](docs/runtime_mvp.md).
The lifecycle and feature boundary between lightweight `RayModule` and
executor-managed `RuntimeRayModule` is documented in
[`docs/runtime_module_lifecycle.md`](docs/runtime_module_lifecycle.md).

### MultiGrain V3 control plane

The experimental MultiGrain V3 runtime separates semantic scheduling from Ray
transport control. An `Arena` owns grains, lineage, batching, reduce state, and
recovery. The shared execution pool owns actors, RPCs, replica selection, and
backpressure. The driver connects them without reading business payloads:

```text
source inputs
  -> microbatch_size: split inputs into Arena-sized source chunks
  -> max_inflight_arenas: admit a bounded number of active Arenas
  -> Arena-major scheduling: earlier active Arenas claim available Stage credit
  -> Stage batch_size: reserve at most one Stage-sized physical dispatch
  -> max_outstanding_per_actor: bound unfinished RPCs on each actor
  -> actor_max_concurrency: bound methods that an actor may execute concurrently
```

These controls have separate ownership, with an explicit capacity constraint
between actor concurrency and its outstanding window:

| Control | Scope | What it bounds |
| --- | --- | --- |
| `microbatch_size` | admission | source items placed in one Arena |
| `max_inflight_arenas` | run | Arenas simultaneously admitted to the pipeline |
| Stage `batch_size` | semantic batching | grains packed into one physical RPC |
| Stage `replicas` | execution | persistent actors available to that Stage |
| `max_outstanding_per_actor` | transport | submitted but unfinished RPCs per actor |
| `actor_max_concurrency` | actor | actor methods allowed to execute concurrently |

`max_outstanding_per_actor` is the outstanding window. It counts both an RPC that
is executing and RPCs waiting in the actor mailbox; it does not change the
Stage batch size. It must be greater than or equal to actor concurrency. With
`actor_max_concurrency=1`:

```text
max_outstanding_per_actor=1  [executing]                 # no extra prefetch
max_outstanding_per_actor=2  [executing][mailbox]        # prefetch one RPC
max_outstanding_per_actor=4  [executing][mailbox x 3]    # deep actor queue
```

The default `1` is intentionally shallow: completion releases one credit, then
the driver chooses fresh ready work. This limits early actor binding and
head-of-line blocking for variable-duration work such as PDF parsing, OCR, and
table extraction. A value of `2` may hide Ray round-trip latency for short,
uniform RPCs, but should be justified by measurement. Larger values trade
backpressure and dynamic load balance for deeper prefetch.

The canonical option is `max_outstanding_per_actor`. The former
`max_pending_per_actor` spelling remains only as an Executor and Stage-option
compatibility alias for existing experiment commands. New code should not use
it. A Stage can override the Executor default through `.ray_options(...)`:

```python
Map(MyUdf).ray_options(
    max_concurrency=1,
    max_outstanding_per_actor=2,
)
```

The driver uses Arena-major ordering; it does not round-robin Stage credit
across Arenas. This preserves Arena-local packing and pipeline overlap still
comes from multiple admitted Arenas sharing independent Stage actor pools. The
execution pool may round-robin among replicas within one Stage; that is actor
selection, not cross-Arena scheduling.

V3's detailed semantic and transport boundaries are documented in
[`docs/multigrain_v3_architecture.md`](docs/multigrain_v3_architecture.md).
The cross-version ideas that V4 must preserve are tracked in
[`docs/multigrain_v3_golden_designs.md`](docs/multigrain_v3_golden_designs.md).
The v3.5 static semantic compiler boundary and its deliberately small
optimization scope are documented in
[`docs/multigrain_v3_5.md`](docs/multigrain_v3_5.md); the pre-release v3.5.1
runtime state algebra is fixed in
[`docs/multigrain_v3_5_1.md`](docs/multigrain_v3_5_1.md). A step-by-step
introduction and maintainer guide is available in
[`docs/multigrain_v3_5_tutorial.md`](docs/multigrain_v3_5_tutorial.md).
For the breaking V3.6 architecture, first choose a reading path from the
[`V3.6 documentation map`](docs/multigrain_v3_6_documentation_map.md). The staged
[`docs/multigrain_v3_6_getting_started.md`](docs/multigrain_v3_6_getting_started.md)
is the single sequential learning path; then use
[`docs/multigrain_v3_6_maintainer_guide.md`](docs/multigrain_v3_6_maintainer_guide.md)
for cross-component contracts and modification discipline. Detailed source walkthroughs
start at
[`docs/multigrain_v3_6_walkthrough/README.md`](docs/multigrain_v3_6_walkthrough/README.md).

## Minimal Example

```python
from rayorch import OverlappedPipeline, RayModule


class AddOne:
    def run(self, x):
        return x + 1


class Pipe(OverlappedPipeline):
    def __init__(self):
        self.a = RayModule(AddOne, replicas=1).pre_init()
        self.b = RayModule(AddOne, replicas=1).pre_init()
        super().__init__(max_inflight=4)

    def forward(self, x):
        return self.b(self.a(x))


pipe = Pipe()
print(pipe([1, 2, 3]))  # [3, 4, 5]
```

## Runtime Tests

Fast compile and local semantics tests do not start Ray:

```bash
pytest test/runtime/unit
```

Runtime integration tests share one local Ray cluster per test module:

```bash
pytest test/runtime/integration
```

Long performance regressions are skipped by default. Run them explicitly with:

```bash
pytest --runslow -s test/runtime/performance
```

Run the 512-PDF Flash-MinerU-style fault-isolation demo. Its default topology
simulates four layout GPU workers, four OCR GPU workers, and four microbatches
in flight:

```bash
python examples/runtime_flash_mineru_512.py
```

Replica counts, batch size, inflight limits, and dummy inference delay are
available as command-line options:

```bash
python examples/runtime_flash_mineru_512.py --help
```

## License

Apache-2.0. See `LICENSE`.
