"""Long-tail 1:N acceleration, measured on real GPUs vs. scheduling theory.

An imbalanced fan-out is a *makespan minimization on identical machines* problem.
For per-page work weights w_i on R GPUs:

    OPT           >= max( sum(w)/R , max(w_i) )          (lower bound)
    makespan_LPT  <= (4/3 - 1/(3R)) * OPT                (Graham 1969)
    efficiency     = sum(w) / (R * makespan)             (1 - idle bubble)

Naive contiguous (equal *row count*) sharding ignores w_i, so under a long tail a
few heavy pages land together and one GPU dominates the makespan -> big bubble.
Work-aware LPT reordering spreads them and approaches OPT. As the tail gets
heavier (smaller Pareto alpha), the contiguous bubble -- and thus the achievable
speedup -- grows. We verify measured GPU makespan tracks the analytic prediction
and that LPT respects its 4/3 bound.

Run under the torch-base conda env:
    python -u -m test.experimental.multigrain.bench_gpu_longtail
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

from test.experimental.multigrain.gpu_ops import (
    AssemblePages,
    OcrGpu,
    PdfToPages,
    _build_ocr_actor,
    page_work,
)

REPLICAS = 4
ALPHAS = [2.5, 1.7, 1.2]  # Pareto shape: smaller => heavier tail
N_PAGES = 32  # few pages per GPU so the tail is not averaged away


# --------------------------------------------------------------------------
# Long-tail workload
# --------------------------------------------------------------------------
def make_pages(n: int, alpha: float, seed: int, *, xm: int = 3, wmax: int = 60) -> list[dict]:
    rng = random.Random(seed)
    pages = []
    for i in range(n):
        work = min(wmax, int(math.ceil(xm * rng.paretovariate(alpha))))
        pages.append({"doc": f"p{i:03d}", "page": 0, "work": work})
    return pages


def _partitions(name: str, weights: list[float]) -> list[list[int]]:
    if name == "contiguous":
        return [list(rng) for rng in _contiguous_ranges(len(weights), REPLICAS)]
    fake = mg.source([{"work": w} for w in weights], name="pages")
    return lpt_shard_planner(lambda p: p["work"])(None, [fake], REPLICAS)


def analytic_units(parts: list[list[int]], weights: list[float]) -> float:
    return max((sum(weights[i] for i in part) for part in parts), default=0.0)


def theory(weights: list[float]) -> dict:
    total = sum(weights)
    wmax = max(weights)
    opt = max(total / REPLICAS, wmax)
    return {
        "total": total,
        "wmax": wmax,
        "opt": opt,
        "lpt_bound": (4.0 / 3.0 - 1.0 / (3 * REPLICAS)) * opt,
    }


def measure(pool, page_values: list[dict], parts: list[list[int]]) -> float:
    refs = [
        pool[gpu].ocr_timed.remote([page_values[i] for i in idx])
        for gpu, idx in enumerate(parts)
        if idx
    ]
    return max(r[0] for r in ray.get(refs))


# --------------------------------------------------------------------------
def main() -> None:
    ray.init(
        num_gpus=REPLICAS,
        num_cpus=16,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )

    actor_cls = _build_ocr_actor()
    pool = [actor_cls.remote() for _ in range(REPLICAS)]
    ray.get([actor.warmup.remote() for actor in pool])

    print(f"=== Long-tail 1:N acceleration on {REPLICAS} GPUs (per config: {N_PAGES} pages) ===")
    header = (
        f"{'alpha':>5} {'tail (p50/p99/max)':>19} {'W/R':>6} "
        f"{'cont_ms':>8} {'lpt_ms':>7} {'OPT_ms':>7} "
        f"{'meas_speedup':>12} {'thy_speedup':>11} {'lpt_eff':>8} {'bound_ok':>8}"
    )
    print(header)
    print("-" * len(header))

    for alpha in ALPHAS:
        pages = make_pages(N_PAGES, alpha, seed=int(alpha * 100))
        weights = [float(p["work"]) for p in pages]
        th = theory(weights)

        cont = _partitions("contiguous", weights)
        lpt = _partitions("lpt", weights)
        cont_u = analytic_units(cont, weights)
        lpt_u = analytic_units(lpt, weights)

        cont_ms = measure(pool, pages, cont)
        lpt_ms = measure(pool, pages, lpt)

        # calibrate units->seconds from this config's LPT run (op is linear)
        ms_per_unit = lpt_ms / lpt_u
        opt_ms = th["opt"] * ms_per_unit
        lpt_bound_ms = th["lpt_bound"] * ms_per_unit

        meas_speedup = cont_ms / lpt_ms
        thy_speedup = cont_u / lpt_u
        lpt_eff = th["total"] / (REPLICAS * (lpt_ms / ms_per_unit))
        bound_ok = lpt_ms <= lpt_bound_ms * 1.05  # 5% slack for measurement noise

        srt = sorted(weights)
        p50 = srt[len(srt) // 2]
        p99 = srt[min(len(srt) - 1, int(0.99 * len(srt)))]
        tail = f"{p50:.0f}/{p99:.0f}/{th['wmax']:.0f}"
        print(
            f"{alpha:>5.1f} {tail:>19} "
            f"{th['total'] / REPLICAS:>6.0f} "
            f"{cont_ms * 1000:>8.0f} {lpt_ms * 1000:>7.0f} {opt_ms * 1000:>7.0f} "
            f"{meas_speedup:>11.2f}x {thy_speedup:>10.2f}x {lpt_eff * 100:>7.1f}% "
            f"{'yes' if bound_ok else 'NO':>8}"
        )

    # --- end-to-end pipeline on a heavy-tail document set ------------------
    for actor in pool:
        ray.kill(actor)
    time.sleep(2)

    docs = _longtail_docs(alpha=1.4, seed=140)
    total_pages = sum(len(d["page_works"]) for d in docs)
    ir = _MineruPipe(REPLICAS).compile()
    docs_batch = mg.source(docs, name="docs", display_key=lambda d: d["name"])

    MultigrainRayExecutor().execute(ir, {"docs": docs_batch})  # warm

    t = time.time()
    out_cont = MultigrainRayExecutor().execute(ir, {"docs": docs_batch})
    e2e_cont = time.time() - t

    t = time.time()
    out_lpt = MultigrainRayExecutor(
        shard_planner=lpt_shard_planner(page_work)
    ).execute(ir, {"docs": docs_batch})
    e2e_lpt = time.time() - t

    assert sorted(out_cont.values) == sorted(out_lpt.values)
    assert out_cont.record_ids == docs_batch.record_ids

    print(
        f"\n=== End-to-end (Expand -> GPU OCR -> Reduce), heavy tail, "
        f"{len(docs)} docs / {total_pages} pages ==="
    )
    print(f"  contiguous : {e2e_cont:.2f}s")
    print(f"  LPT        : {e2e_lpt:.2f}s   (speedup {e2e_cont / e2e_lpt:.2f}x)")
    print("  results identical (lineage-preserving reorder): "
          f"{sorted(out_cont.values) == sorted(out_lpt.values)}")

    ray.shutdown()


def _longtail_docs(alpha: float, seed: int, num_docs: int = 16) -> list[dict]:
    """A few dominant documents with long, heavy page lists; the rest tiny.

    Pages flatten in document order, so a dominant document's pages form a
    contiguous block -> naive equal-count sharding piles them onto one or two
    GPUs, which is the classic long-tail 1:N bubble that LPT rebalancing fixes.
    """
    rng = random.Random(seed)
    docs = []
    for i in range(num_docs):
        dominant = i in (2, 9)  # two heavy-tailed documents
        pages = rng.randint(18, 26) if dominant else rng.randint(1, 3)
        works = [
            (rng.randint(14, 26) if dominant else rng.randint(2, 6))
            for _ in range(pages)
        ]
        docs.append({"name": f"doc{i:02d}", "page_works": works})
    return docs


class _MineruPipe(mg.Pipeline):
    def __init__(self, replicas: int) -> None:
        super().__init__()
        self.to_pages = mg.Expand(PdfToPages, parent=0, child_label="page")
        self.ocr = mg.Map(
            OcrGpu, physical=PhysicalHints(replicas=replicas, num_gpus_per_replica=1.0)
        )
        self.assemble = mg.Reduce(AssemblePages)

    def forward(self, docs):
        return self.assemble(mg.group_by(docs, self.ocr(self.to_pages(docs))))


if __name__ == "__main__":
    main()
