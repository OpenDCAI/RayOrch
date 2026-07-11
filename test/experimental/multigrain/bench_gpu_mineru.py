"""Simulate a MinerU-style GPU parse and observe load distribution.

document -> (Expand) variable-length pages -> (Map, real GPU matmuls) OCR
         -> (Reduce) assemble back to document

The point is to show what happens when a 1:N fan-out is *imbalanced* (variable
page counts AND variable per-page content length): naive contiguous row-sharding
across GPUs leaves some GPUs idle ("bubbles"), while relation-aware work-balanced
sharding (LPT) fills them. Both run the same passive IR through
``MultigrainRayExecutor``; only the shard planner changes.

Run (needs the torch-base conda env + GPUs):

    python -u test/experimental/multigrain/bench_gpu_mineru.py
"""
from __future__ import annotations

import random
import time

import ray

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.graph import PhysicalHints
from rayorch.experimental.multigrain.ray_executor import (
    MultigrainRayExecutor,
    _contiguous_ranges,
    lpt_shard_planner,
)

from test.experimental.multigrain.gpu_ops import (
    AssemblePages,
    OcrGpu,
    PdfToPages,
    _build_ocr_actor,
    page_work,
)

REPLICAS = 4  # one Ray task per GPU


# --------------------------------------------------------------------------
# Workload: skewed page counts + skewed per-page work -> strong imbalance
# --------------------------------------------------------------------------
def make_docs(num_docs: int = 12, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    docs: list[dict] = []
    for i in range(num_docs):
        heavy = rng.random() < 0.25  # a few big documents dominate
        pages = rng.randint(8, 14) if heavy else rng.randint(1, 5)
        works = [
            (rng.randint(14, 24) if heavy else rng.randint(3, 10))
            for _ in range(pages)
        ]
        docs.append({"name": f"doc{i:02d}", "page_works": works})
    return docs


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class MineruGpuPipe(mg.Pipeline):
    def __init__(self, replicas: int, gpus: float) -> None:
        super().__init__()
        self.to_pages = mg.Expand(PdfToPages, parent=0, child_label="page")
        self.ocr = mg.Map(
            OcrGpu,
            physical=PhysicalHints(replicas=replicas, num_gpus_per_replica=gpus),
        )
        self.assemble = mg.Reduce(AssemblePages)

    def forward(self, docs):
        pages = self.to_pages(docs)
        texts = self.ocr(pages)
        return self.assemble(mg.group_by(docs, texts))


# --------------------------------------------------------------------------
# Per-shard GPU timing (to visualize the distribution / bubbles)
# --------------------------------------------------------------------------
def _partitions(name: str, page_values: list[dict], replicas: int) -> list[list[int]]:
    if name == "contiguous":
        return [list(rng) for rng in _contiguous_ranges(len(page_values), replicas)]
    planner = lpt_shard_planner(page_work)
    fake_batch = mg.source(page_values, name="pages")
    return planner(None, [fake_batch], replicas)  # type: ignore[arg-type]


def measure_distribution(pool, name: str, page_values: list[dict]) -> dict:
    parts = _partitions(name, page_values, REPLICAS)
    refs = [
        pool[gpu].ocr_timed.remote([page_values[i] for i in idx])
        for gpu, idx in enumerate(parts)
    ]
    results = ray.get(refs)
    elapsed = [r[0] for r in results]
    counts = [r[1] for r in results]
    units = [r[2] for r in results]
    makespan = max(elapsed)
    busy = sum(elapsed)
    bubble = 1.0 - busy / (makespan * len(elapsed)) if makespan else 0.0
    return {
        "name": name,
        "elapsed": elapsed,
        "counts": counts,
        "units": units,
        "makespan": makespan,
        "bubble": bubble,
    }


def _print_dist(dist: dict) -> None:
    print(f"\n[{dist['name']}] per-GPU shard distribution")
    for gpu, (sec, cnt, unit) in enumerate(
        zip(dist["elapsed"], dist["counts"], dist["units"])
    ):
        bar = "#" * int(round(sec / max(dist["elapsed"]) * 30))
        print(f"  GPU{gpu}: {sec:6.2f}s  pages={cnt:3d}  units={unit:4d}  {bar}")
    print(
        f"  makespan={dist['makespan']:.2f}s  "
        f"idle-bubble={dist['bubble'] * 100:.1f}%"
    )


# --------------------------------------------------------------------------
def main() -> None:
    ray.init(
        num_gpus=REPLICAS,
        num_cpus=16,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )

    docs = make_docs()
    total_pages = sum(len(d["page_works"]) for d in docs)
    total_units = sum(sum(d["page_works"]) for d in docs)
    print("=== Workload (simulated MinerU, variable-length pages) ===")
    for d in docs:
        pw = d["page_works"]
        print(f"  {d['name']}: pages={len(pw):2d} units/page={pw}")
    print(f"  TOTAL: {len(docs)} docs, {total_pages} pages, {total_units} GPU units")
    print(f"  (ideal balanced makespan ~ {total_units / REPLICAS} units on {REPLICAS} GPUs)")

    # materialize pages once (cheap CPU expand) to know per-page weights
    pages = MineruGpuPipe(1, 0).to_pages(mg.source(docs, name="docs"))
    page_values = list(pages.values)

    # GPU-pinned, long-lived actors give cold-start-free per-GPU numbers
    # (each actor pays torch/cuda init once), like a real MinerU stage.
    print("\nwarming up GPU actors ...")
    actor_cls = _build_ocr_actor()
    pool = [actor_cls.remote() for _ in range(REPLICAS)]
    ray.get([actor.warmup.remote() for actor in pool])

    # --- distribution measurement: naive contiguous vs work-balanced LPT ---
    naive = measure_distribution(pool, "contiguous", page_values)
    balanced = measure_distribution(pool, "lpt", page_values)
    _print_dist(naive)
    _print_dist(balanced)

    # release the GPU-pinned actors so the executor's GPU tasks can claim the GPUs
    for actor in pool:
        ray.kill(actor)
    time.sleep(2)

    # --- end-to-end pipeline through the passive IR (both planners) ---
    ir = MineruGpuPipe(REPLICAS, 1.0).compile()
    docs_batch = mg.source(docs, name="docs")

    # warm the executor's GPU worker pool with the *real* workload once (so it
    # shards into REPLICAS tasks and every GPU worker is warm) before timing.
    MultigrainRayExecutor().execute(ir, {"docs": docs_batch})

    t = time.time()
    out_naive = MultigrainRayExecutor().execute(ir, {"docs": docs_batch})
    e2e_naive = time.time() - t

    t = time.time()
    out_lpt = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(page_work)
    ).execute(ir, {"docs": docs_batch})
    e2e_lpt = time.time() - t

    assert sorted(out_naive.values) == sorted(out_lpt.values)
    assert out_naive.record_ids == docs_batch.record_ids  # reduce restores doc order

    print("\n=== End-to-end pipeline (Expand -> GPU OCR -> Reduce) ===")
    print(f"  contiguous shard planner : {e2e_naive:.2f}s")
    print(f"  work-balanced (LPT)      : {e2e_lpt:.2f}s")
    speedup = e2e_naive / e2e_lpt if e2e_lpt else float("nan")
    print(f"  speedup from rebalancing : {speedup:.2f}x")

    print("\n=== Summary ===")
    print(
        f"  OCR-stage makespan: contiguous {naive['makespan']:.2f}s "
        f"(bubble {naive['bubble'] * 100:.0f}%)  ->  "
        f"lpt {balanced['makespan']:.2f}s (bubble {balanced['bubble'] * 100:.0f}%)"
    )
    print(f"  sample result: {out_lpt.values[:3]} ...")

    ray.shutdown()


if __name__ == "__main__":
    main()
