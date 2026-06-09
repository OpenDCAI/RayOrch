from __future__ import annotations

import pytest

from rayorch import DagPipeline, RuntimeRayModule
from rayorch.runtime import RuntimeNodeSpec

from test.runtime.helpers import Pdf2ImgOp


def test_compile_binds_runtime_ports_from_forward_names() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(Pdf2ImgOp)
            super().__init__()

        def forward(
            self,
            pdf: list[str],
            meta: list[dict[str, object]],
        ) -> tuple[list[list[str]], list[dict[str, object]]]:
            images, meta = self.pdf2img(pdf, meta)
            return images, meta

    pipe = Pipe().compile()

    assert pipe.pdf2img.runtime_spec == RuntimeNodeSpec(
        node="pdf2img",
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
    )
    assert pipe._compiled.nodes["pdf2img"].num_outputs == 2


def test_compile_rejects_output_arity_mismatch() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.pdf2img = RuntimeRayModule(Pdf2ImgOp, num_outputs=2)
            super().__init__()

        def forward(
            self,
            pdf: list[str],
            meta: list[dict[str, object]],
        ) -> list[list[str]]:
            images = self.pdf2img(pdf, meta)
            return images

    with pytest.raises(ValueError, match="assigns 1 outputs"):
        Pipe().compile()


def test_compile_keeps_nested_call_names_in_execution_order() -> None:
    class PassOp:
        def run(self, x):
            return x

    class Pipe(DagPipeline):
        def __init__(self):
            self.keep = RuntimeRayModule(PassOp)
            super().__init__()

        def forward(self, pdf: list[str]) -> list[str]:
            first = self.keep(pdf)
            passthrough = self.keep(self.keep(first))
            return self.keep(passthrough)

    graph = Pipe().compile()._compiled

    assert graph.nodes["keep"].output_names == ("first",)
    assert graph.nodes["keep_1"].output_names == ("keep_1.out0",)
    assert graph.nodes["keep_2"].output_names == ("passthrough",)
    assert graph.nodes["keep_3"].output_names == ("keep_3.out0",)


def test_runtime_pipeline_rejects_eager_execution_without_compile() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.op = RuntimeRayModule(Pdf2ImgOp)
            super().__init__()

        def forward(
            self,
            pdf: list[str],
            meta: list[dict[str, object]],
        ) -> tuple[list[list[str]], list[dict[str, object]]]:
            return self.op(pdf, meta)

    pipe = Pipe()

    with pytest.raises(RuntimeError, match="Use RuntimeDagExecutor"):
        pipe(["doc.pdf"], [{"name": "doc"}])
    with pytest.raises(RuntimeError, match="Use RuntimeDagExecutor"):
        pipe.run(["doc.pdf"], [{"name": "doc"}])
    assert pipe._compiled is None
