"""用于 Port 语义与 batching 回归的确定性 CPU/真实块 benchmark。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .. import functional as F
from ..api import Pipeline, RayModule
from ..executor import Executor, RunResult


@dataclass(frozen=True, slots=True)
class Page:
    """dummy 文档展开后携带稳定 ordinal 的页面记录。"""

    document: int
    ordinal: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class Feature:
    """页面经过确定性 CPU 重计算后的摘要记录。"""

    ordinal: int
    digest: str


class RenderPages:
    """按文档编号生成可复现、变长的页面列表。"""

    def __init__(self, max_pages: int = 11) -> None:
        self.max_pages = max_pages

    def run(self, documents: list[int]) -> list[list[Page]]:
        """批量渲染文档，并保留空文档这一 fan-out 边界。"""

        return [
            [
                Page(
                    document,
                    ordinal,
                    hashlib.blake2b(
                        f"{document}:{ordinal}".encode(),
                        digest_size=32,
                    ).digest(),
                )
                for ordinal in range((document * 17) % self.max_pages)
            ]
            for document in documents
        ]


class TransformPages:
    """用多轮哈希模拟可调计算量的逐页算子。"""

    def __init__(self, rounds: int = 20) -> None:
        self.rounds = rounds

    def run(self, pages: list[Page]) -> list[Feature]:
        """批量转换页面，输出数与输入 Grain 数严格相同。"""

        outputs = []
        for page in pages:
            payload = page.payload
            for _ in range(self.rounds):
                payload = hashlib.blake2b(payload, digest_size=32).digest()
            outputs.append(Feature(page.ordinal, payload.hex()))
        return outputs


class KeepPages:
    """产生确定性布尔 mask，覆盖 Port 级 filter。"""

    def run(self, pages: list[Page]) -> list[bool]:
        """按 ordinal 返回与页面逐 Grain 对齐的 mask。"""

        return [page.ordinal % 5 != 1 for page in pages]


class AssembleDocuments:
    """把过滤后的有序 Feature group 汇总回根文档。"""

    def run(
        self,
        documents: list[int],
        groups: list[list[Feature]],
    ) -> list[tuple[int, tuple[tuple[int, str], ...]]]:
        """批量组装稳定、便于摘要比较的文档结果。"""

        return [
            (
                document,
                tuple((feature.ordinal, feature.digest) for feature in group),
            )
            for document, group in zip(documents, groups)
        ]


class DummyPipeline(Pipeline):
    """覆盖 Expand→Filter→Map→Reduce 的端到端 dummy Pipeline。"""

    def __init__(
        self,
        *,
        mode: str,
        batch_size: int = 32,
        rounds: int = 20,
    ) -> None:
        self.render = (
            RayModule(RenderPages)
            .pre_init(max_pages=11)
            .ray_options(batch_size=16)
        )
        self.keep = RayModule(KeepPages).ray_options(batch_size=batch_size)
        self.transform = (
            RayModule(TransformPages)
            .pre_init(rounds=rounds)
            .ray_options(batch_size=batch_size, batch_scope=mode)
        )
        self.assemble = RayModule(AssembleDocuments).ray_options(batch_size=16)

    def forward(self, documents):
        """声明 Expand→Filter→Map→Reduce 的 Port 数据流。"""

        page_groups = self.render(documents)
        pages = F.expand(page_groups)
        keep = self.keep(pages)
        selected_pages = F.filter(pages, keep)
        features = self.transform(selected_pages)
        feature_groups = F.reduce(features, members=selected_pages)
        return self.assemble(documents, feature_groups)


def run_dummy(
    documents: int = 256,
    *,
    mode: str = "elastic",
    batch_size: int = 32,
    rounds: int = 20,
) -> RunResult:
    """运行确定性文档，并返回输出、Arena 与批处理指标。"""

    pipeline = DummyPipeline(
        mode=mode,
        batch_size=batch_size,
        rounds=rounds,
    )
    with Executor(pipeline) as executor:
        return executor.run(range(documents))


__all__ = ["DummyPipeline", "run_dummy"]
