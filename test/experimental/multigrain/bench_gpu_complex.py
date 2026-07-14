"""Complex multi-stage load: confirm the GPU bubble collapses to ~0.

Pipeline (see complex_ops.py):
    doc =Expand=> pages =Expand=> blocks =Filter=> dense =Map(GPU OCR)=> =Reduce=> doc

This stacks three bubble sources on the wide GPU stage at once:
  (1) intra-stage long-tail per-block work,
  (2) compounded two-level fan-out long tail,
  (3) a data-dependent filter that leaves an uneven survivor set.

Because the IR is passive, every node re-partitions its *own* current input, so
the OCR stage is LPT-balanced over the post-filter blocks -- filter skew and
compounded fan-out are absorbed. We confirm the OCR-stage efficiency approaches
100% (bubble -> 0), matches the OPT lower bound and Graham's 4/3 bound, and that
the full pipeline speeds up with byte-identical results (lineage preserved).

We also print the honest *residual* bubbles the MVP cannot remove yet:
Reduce fan-in skew (single-task) and any single atomic giant block.

Run under torch-base:  python -u -m test.experimental.multigrain.bench_gpu_complex
"""
from __future__ import annotations

import math
import random
import time

import ray

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import PhysicalHints
from rayorch.experimental.multigrain.ray import (
    MultigrainRayExecutor,
    _contiguous_ranges,
    lpt_shard_planner,
)

from test.experimental.multigrain.complex_ops import (
    AssembleDoc,
    DocToPages,
    KeepDense,
    OcrBlocks,
    PageToBlocks,
    block_work,
)
from test.experimental.multigrain.gpu_ops import _build_ocr_actor

REPLICAS = 4
THRESHOLD = 4


def make_complex_docs(num_docs: int = 8, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(num_docs):
        npages = min(8, int(math.ceil(rng.paretovariate(1.4))))
        pages = []
        for _ in range(npages):
            nblocks = min(5, int(math.ceil(rng.paretovariate(1.5))))
            # long-tailed work with a real low end so the filter drops ~30% unevenly
            pages.append([
                min(45, max(1, int(round((rng.paretovariate(1.2) - 1) * 16))))
                for _ in range(nblocks)
            ])
        docs.append({"name": f"doc{i:02d}", "pages": pages})
    return docs


class ComplexPipe(mg.Pipeline):
    def __init__(self, replicas: int) -> None:
        super().__init__()
        self.to_pages = mg.Expand(DocToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")
        self.keep = mg.Filter(KeepDense, THRESHOLD)
        self.ocr = mg.Map(
            OcrBlocks, physical=PhysicalHints(replicas=replicas, num_gpus_per_replica=1.0)
        )
        self.assemble = mg.Reduce(AssembleDoc)

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks = self.to_blocks(pages)
        dense = self.keep(blocks)
        texts = self.ocr(dense)
        return self.assemble(mg.group_by(docs, texts))


def _dense_blocks(docs_batch):
    """Eagerly run Expand -> Expand -> Filter (CPU) to get the survivor set."""
    pages = mg.Expand(DocToPages, parent=0, child_label="page")(docs_batch)
    blocks = mg.Expand(PageToBlocks, parent=0, child_label="block")(pages)
    return mg.Filter(KeepDense, THRESHOLD)(blocks)


def theory(weights: list[float]) -> dict:
    total, wmax = sum(weights), max(weights)
    opt = max(total / REPLICAS, wmax)
    return {"total": total, "wmax": wmax, "opt": opt,
            "lpt_bound": (4.0 / 3.0 - 1.0 / (3 * REPLICAS)) * opt}


def analytic_units(parts, weights) -> float:
    return max((sum(weights[i] for i in part) for part in parts), default=0.0)


def measure(pool, values, parts) -> list[float]:
    refs = [
        pool[gpu].ocr_timed.remote([values[i] for i in idx])
        for gpu, idx in enumerate(parts) if idx
    ]
    return [r[0] for r in ray.get(refs)]


def main() -> None:
    ray.init(num_gpus=REPLICAS, num_cpus=16, ignore_reinit_error=True,
             include_dashboard=False, log_to_driver=False)

    docs = make_complex_docs()
    docs_batch = mg.source(docs, name="docs", display_key=lambda d: d["name"])
    dense = _dense_blocks(docs_batch)
    values = list(dense.values)
    weights = [block_work(b) for b in values]
    th = theory(weights)

    raw_blocks = sum(len(bl) for d in docs for bl in d["pages"])
    print("=== Complex load: doc -> pages -> blocks -> filter -> GPU OCR -> reduce ===")
    print(f"  {len(docs)} docs, {raw_blocks} raw blocks -> {len(values)} survive filter "
          f"(threshold work>={THRESHOLD})")
    print(f"  GPU units after filter: total={th['total']:.0f}  max block={th['wmax']:.0f}  "
          f"ideal per-GPU (W/R)={th['total'] / REPLICAS:.0f}")

    # -- per-stage bubble on the wide GPU stage: contiguous vs LPT ----------
    actor_cls = _build_ocr_actor()
    pool = [actor_cls.remote() for _ in range(REPLICAS)]
    ray.get([a.warmup.remote() for a in pool])

    cont = [list(r) for r in _contiguous_ranges(len(values), REPLICAS)]
    lpt = lpt_shard_planner(block_work)(None, [dense], REPLICAS)

    cont_t = measure(pool, values, cont)
    lpt_t = measure(pool, values, lpt)
    cont_make, lpt_make = max(cont_t), max(lpt_t)
    cont_eff = sum(cont_t) / (REPLICAS * cont_make)
    lpt_eff = sum(lpt_t) / (REPLICAS * lpt_make)
    ms_per_unit = lpt_make / analytic_units(lpt, weights)

    print("\n--- OCR stage bubble (real GPU makespan) ---")
    print(f"  contiguous : makespan={cont_make:.2f}s  efficiency={cont_eff * 100:.0f}%  "
          f"bubble={100 - cont_eff * 100:.0f}%")
    print(f"  LPT        : makespan={lpt_make:.2f}s  efficiency={lpt_eff * 100:.0f}%  "
          f"bubble={100 - lpt_eff * 100:.0f}%")
    print(f"  OPT lower bound ~ {th['opt'] * ms_per_unit:.2f}s   "
          f"LPT 4/3 bound ~ {th['lpt_bound'] * ms_per_unit:.2f}s   "
          f"LPT within bound: {'yes' if lpt_make <= th['lpt_bound'] * ms_per_unit * 1.05 else 'NO'}")
    print(f"  stage speedup (contiguous/LPT): {cont_make / lpt_make:.2f}x")

    # -- honest residual bubbles the MVP cannot remove ---------------------
    per_doc: dict[str, float] = {}
    for block in values:
        per_doc[block["doc"]] = per_doc.get(block["doc"], 0.0) + block["work"]
    reduce_skew = max(per_doc.values()) / (th["total"] / REPLICAS)
    print("\n--- residual bubbles (honest limits) ---")
    print(f"  reduce fan-in skew : heaviest doc = {max(per_doc.values()):.0f} units "
          f"({reduce_skew:.2f}x of W/R); Reduce is single-task, so a GPU reduce here "
          f"would idle the other {REPLICAS - 1} GPUs")
    print(f"  atomic giant block : max block = {th['wmax']:.0f} units "
          f"({th['wmax'] / (th['total'] / REPLICAS):.2f}x of W/R); cannot be split")

    for a in pool:
        ray.kill(a)
    time.sleep(2)

    # -- end-to-end through the executor: naive vs optimized ---------------
    ir = ComplexPipe(REPLICAS).compile()
    MultigrainRayExecutor().execute(ir, {"docs": docs_batch})  # warm

    t = time.time()
    naive = MultigrainRayExecutor().execute(ir, {"docs": docs_batch})
    e2e_naive = time.time() - t

    t = time.time()
    opt = MultigrainRayExecutor(shard_planner=lpt_shard_planner(block_work)).execute(
        ir, {"docs": docs_batch}
    )
    e2e_opt = time.time() - t

    assert naive.values == opt.values, "lineage/identity broke under reordering"
    assert naive.record_ids == docs_batch.record_ids

    print("\n=== End-to-end pipeline ===")
    print(f"  naive contiguous : {e2e_naive:.2f}s")
    print(f"  LPT rebalanced   : {e2e_opt:.2f}s   (speedup {e2e_naive / e2e_opt:.2f}x)")
    print(f"  byte-identical results (lineage preserved): {naive.values == opt.values}")
    print(f"  sample: {opt.values[:3]} ...")

    ray.shutdown()


if __name__ == "__main__":
    main()
