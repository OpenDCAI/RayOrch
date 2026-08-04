"""v3.3 dummy 性能矩阵、嵌套拓扑和机器可读指标。

该 benchmark 使用真实 Python payload、粗粒度 block 与确定性哈希计算；不使用
``sleep`` 模拟负载。正式回归至少覆盖 1,000 parents、batch cap 16/32/64、
parent-bound/elastic，以及多 Arena in-flight 1/2/3。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .. import functional as F
from ..api import Pipeline, RayModule
from ..executor import LocalExecutor, LocalRunResult
from ..model import CallRef
from ..ray_executor import RayExecutor, RayRunResult
from .dummy import AssembleDocuments, Page, RenderPages, run_dummy


@dataclass(frozen=True, slots=True)
class Region:
    """二层 fan-out 的真实 payload。"""

    document: int
    page: int
    ordinal: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class RegionFeature:
    """CPU-heavy region transform 的确定性输出。"""

    page: int
    ordinal: int
    digest: str


class RenderRegions:
    """为每个 Page 产生长短不一的 Regions。"""

    def __init__(self, max_regions: int = 7) -> None:
        self.max_regions = max_regions

    def run(self, pages: list[Page]) -> list[list[Region]]:
        """按 document/page ordinal 生成可复现二层 fan-out。"""

        groups = []
        for page in pages:
            count = (page.document * 5 + page.ordinal * 3) % self.max_regions
            groups.append(
                [
                    Region(
                        page.document,
                        page.ordinal,
                        ordinal,
                        hashlib.blake2b(
                            page.payload + ordinal.to_bytes(2, "little"),
                            digest_size=32,
                        ).digest(),
                    )
                    for ordinal in range(count)
                ]
            )
        return groups


class TransformRegions:
    """对 Region payload 执行确定性 CPU 工作。"""

    def __init__(self, rounds: int = 8) -> None:
        self.rounds = rounds

    def run(self, regions: list[Region]) -> list[RegionFeature]:
        """保持 page/region ordinal，便于验证 nested ordered reduce。"""

        outputs = []
        for region in regions:
            payload = region.payload
            for _ in range(self.rounds):
                payload = hashlib.blake2b(payload, digest_size=32).digest()
            outputs.append(
                RegionFeature(region.page, region.ordinal, payload.hex())
            )
        return outputs


class AssembleNested:
    """把 Page→Region 两层 group 与 Document 对齐。"""

    def run(self, documents, groups):
        """保留完整嵌套形状，输出用于 local/Ray parity。"""

        return list(zip(documents, groups))


class NestedDummyPipeline(Pipeline):
    """Document→Page→Region→Page Reduce→Document Reduce。"""

    def __init__(self, *, batch_size: int = 32, rounds: int = 8) -> None:
        self.pages = RayModule(RenderPages).pre_init(max_pages=11).ray_options(
            batch_size=16
        )
        self.regions = RayModule(RenderRegions).pre_init(max_regions=7).ray_options(
            batch_size=batch_size
        )
        self.transform = (
            RayModule(TransformRegions)
            .pre_init(rounds=rounds)
            .ray_options(batch_size=batch_size)
        )
        self.assemble = RayModule(AssembleNested).ray_options(batch_size=16)

    def forward(self, documents):
        """连续两次 Expand，并严格逐层 Reduce 回 root Domain。"""

        pages = F.expand(self.pages(documents))
        regions = F.expand(self.regions(pages))
        features = self.transform(regions)
        features_by_page = F.reduce(features)
        features_by_document = F.reduce(features_by_page)
        return self.assemble(documents, features_by_document)


@dataclass(frozen=True, slots=True)
class DummyMetrics:
    """跨 local/Ray runner 共用的精简 JSON 指标。"""

    mode: str
    parents: int
    batch_size: int
    in_flight: int
    elapsed_s: float
    parents_per_s: float
    rpc_count: int
    heavy_rpcs: int
    heavy_average_batch: float
    entities: int
    items: int
    shapes: int
    blocks: int | None
    output_digest: str


def summarize_local(
    result: LocalRunResult,
    *,
    mode: str,
    parents: int,
    batch_size: int,
) -> DummyMetrics:
    """从 LocalRunResult 提取不读取中间 payload 的框架指标。"""

    heavy = result.calls[CallRef(2)]
    return DummyMetrics(
        mode,
        parents,
        batch_size,
        1,
        result.elapsed_s,
        parents / result.elapsed_s if result.elapsed_s else float("inf"),
        result.rpc_count,
        heavy.rpcs,
        heavy.average_batch,
        sum(len(result.arena.entities(domain)) for domain in result.arena.program.domains),
        len(result.arena.state.items),
        len(result.arena.state.shapes),
        len(result.store.blocks),
        _digest(result.outputs),
    )


def summarize_ray(
    result: RayRunResult,
    *,
    mode: str,
    parents: int,
    batch_size: int,
) -> DummyMetrics:
    """聚合多 Arena 的语义计数与共享 actor 调度指标。"""

    heavy = result.calls[CallRef(2)]
    return DummyMetrics(
        mode,
        parents,
        batch_size,
        result.max_active_arenas,
        result.elapsed_s,
        parents / result.elapsed_s if result.elapsed_s else float("inf"),
        result.rpc_count,
        heavy.rpcs,
        heavy.average_batch,
        sum(
            len(arena.entities(domain))
            for arena in result.arenas
            for domain in arena.program.domains
        ),
        sum(len(arena.state.items) for arena in result.arenas),
        sum(len(arena.state.shapes) for arena in result.arenas),
        None,
        _digest(result.outputs),
    )


def run_local_matrix(
    *,
    parents: int = 1_000,
    batch_sizes: tuple[int, ...] = (16, 32, 64),
    rounds: int = 8,
) -> list[DummyMetrics]:
    """运行 2×3 local 消融，并对每个 batch cap 强制输出 parity。"""

    metrics = []
    for batch_size in batch_sizes:
        by_mode = {}
        for mode in ("parent_bound", "elastic"):
            result = run_dummy(
                parents,
                mode=mode,
                batch_size=batch_size,
                rounds=rounds,
            )
            by_mode[mode] = result
            metrics.append(
                summarize_local(
                    result,
                    mode=mode,
                    parents=parents,
                    batch_size=batch_size,
                )
            )
        if by_mode["parent_bound"].outputs != by_mode["elastic"].outputs:
            raise AssertionError(f"output mismatch at batch_size={batch_size}")
    return metrics


def run_ray_inflight_matrix(
    *,
    parents: int = 256,
    batch_size: int = 32,
    arena_size: int = 64,
    in_flights: tuple[int, ...] = (1, 2, 3),
    rounds: int = 4,
) -> list[DummyMetrics]:
    """在同一语义拓扑上回归多 Arena overlap 与有序输出。"""

    metrics = []
    expected_digest = None
    for in_flight in in_flights:
        pipeline = _ray_dummy_pipeline(batch_size=batch_size, rounds=rounds)
        with RayExecutor(pipeline, address="auto") as executor:
            result = executor.run(
                range(parents),
                arena_size=arena_size,
                max_in_flight=in_flight,
            )
        current = summarize_ray(
            result,
            mode="elastic",
            parents=parents,
            batch_size=batch_size,
        )
        expected_digest = expected_digest or current.output_digest
        if current.output_digest != expected_digest:
            raise AssertionError(f"output mismatch at in_flight={in_flight}")
        metrics.append(current)
    return metrics


def _ray_dummy_pipeline(*, batch_size: int, rounds: int):
    """构造 CPU-only Ray dummy，避免与正在运行的 GPU 实验竞争。"""

    from .dummy import DummyPipeline

    pipeline = DummyPipeline(
        mode="elastic",
        batch_size=batch_size,
        rounds=rounds,
    )
    for module in (
        pipeline.render,
        pipeline.keep,
        pipeline.transform,
        pipeline.assemble,
    ):
        module.ray_options(num_cpus=0)
    return pipeline


def _digest(value: Any) -> str:
    """以稳定 repr 计算业务输出摘要；仅在最终交付后调用。"""

    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；每行输出一个便于增量落盘的 JSON record。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "ray", "nested"), default="local")
    parser.add_argument("--parents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args(argv)

    if args.mode == "local":
        records = run_local_matrix(
            parents=args.parents or 1_000,
            rounds=args.rounds,
        )
    elif args.mode == "ray":
        records = run_ray_inflight_matrix(
            parents=args.parents or 256,
            rounds=args.rounds,
        )
    else:
        parents = args.parents or 128
        result = LocalExecutor(NestedDummyPipeline(rounds=args.rounds)).run(
            range(parents)
        )
        records = [
            {
                "mode": "nested",
                "parents": parents,
                "elapsed_s": result.elapsed_s,
                "rpc_count": result.rpc_count,
                "output_digest": _digest(result.outputs),
            }
        ]

    for record in records:
        print(json.dumps(asdict(record) if isinstance(record, DummyMetrics) else record))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())


__all__ = [
    "DummyMetrics",
    "NestedDummyPipeline",
    "run_local_matrix",
    "run_ray_inflight_matrix",
    "summarize_local",
    "summarize_ray",
]
