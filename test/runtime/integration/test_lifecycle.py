from __future__ import annotations

import pytest

from rayorch import DagPipeline, RuntimeDagExecutor, RuntimeRayModule
from rayorch.runtime import MicroBatch, RuntimeNodeSpec

from test.runtime.helpers import Pdf2ImgOp


class PrefixOp:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def run(self, value: list[str]) -> list[str]:
        return [f"{self.prefix}{item}" for item in value]


class AnnotatedPdf2ImgOp(Pdf2ImgOp):
    def run(
        self,
        pdf: list[str],
        meta: list[dict[str, object]],
    ) -> tuple[list[list[str]], list[dict[str, object]]]:
        return super().run(pdf, meta)


def test_pre_init_stores_arguments_without_starting_actors() -> None:
    module = RuntimeRayModule(
        PrefixOp,
        inputs=("value",),
        outputs=("output",),
    ).pre_init("ready:")

    assert module.actors == []
    assert not module.is_started


def test_unstarted_module_rejects_direct_call() -> None:
    module = RuntimeRayModule(
        PrefixOp,
        inputs=("value",),
        outputs=("output",),
    ).pre_init("ready:")
    source = MicroBatch.source({"value": ["a"]})

    with pytest.raises(RuntimeError, match="RuntimeRayModule is not started"):
        module(source)


def test_start_is_idempotent_and_close_releases_actors(ray_cluster) -> None:
    module = RuntimeRayModule(
        PrefixOp,
        inputs=("value",),
        outputs=("output",),
    ).pre_init("ready:")
    try:
        module.start()
        actors = list(module.actors)

        assert module.start() is module
        assert module.actors == actors
        result = module(MicroBatch.source({"value": ["a"]}))
        assert result.batch.columns["output"] == ["ready:a"]
    finally:
        module.close()

    assert module.actors == []


def test_started_module_requires_runtime_spec(ray_cluster) -> None:
    module = RuntimeRayModule(Pdf2ImgOp).start()
    try:
        source = MicroBatch.source(
            {"pdf": ["doc.pdf"], "meta": [{"name": "doc"}]}
        )
        with pytest.raises(RuntimeError, match="requires a RuntimeNodeSpec"):
            module(source)
    finally:
        module.close()


def test_executor_starts_unique_modules_and_owns_cleanup(ray_cluster) -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(
                AnnotatedPdf2ImgOp, replicas=2
            ).pre_init()
            super().__init__()

        def forward(
            self,
            pdf: list[str],
            meta: list[dict[str, object]],
        ) -> tuple[list[list[str]], list[dict[str, object]]]:
            return self.pdf2img(pdf, meta)

    pipe = Pipe()
    assert pipe.pdf2img.actors == []

    executor = RuntimeDagExecutor(pipe)
    assert len(pipe.pdf2img.actors) == 2
    assert pipe.pdf2img.num_outputs == 2
    assert executor.graph.nodes["pdf2img"].output_types == (
        list[list[str]],
        list[dict[str, object]],
    )

    executor.close()
    assert pipe.pdf2img.actors == []


def test_direct_module_can_bind_spec_before_start(ray_cluster) -> None:
    module = RuntimeRayModule(Pdf2ImgOp, replicas=2).pre_init()
    module.bind_runtime_spec(
        RuntimeNodeSpec(
            node="pdf2img",
            inputs=("pdf", "meta"),
            outputs=("images", "meta"),
        )
    )
    module.start()
    try:
        source = MicroBatch.source(
            {
                "pdf": ["doc.pdf", "doc_bad.pdf"],
                "meta": [{"name": "doc"}, {"name": "bad"}],
            }
        )
        result = module(source)
        assert len(result.batch) == 1
        assert [record.values["pdf"] for record in result.quarantined] == [
            "doc_bad.pdf"
        ]
    finally:
        module.close()
