from __future__ import annotations

import ray

from rayorch import RayModule
from rayorch.runtime import BadRecordError, LineageStore, MicroBatch, run_rowwise


def _cleanup(module: RayModule) -> None:
    for actor in getattr(module, "actors", []):
        try:
            ray.kill(actor)
        except Exception:
            pass


class RuntimePdf2ImgOp:
    """Ray actor op that runs the MVP rowwise runtime internally."""

    def __init__(self) -> None:
        self.lineage = LineageStore()

    def run(self, batch: MicroBatch):
        def pdf2img(pdfs, meta):
            images = []
            for i, (pdf, item) in enumerate(zip(pdfs, meta)):
                if pdf == "corrupt.pdf":
                    raise BadRecordError("pdf parser failed", index=i)
                item["pages"] = 2
                images.append([f"img<{pdf}:0>", f"img<{pdf}:1>"])
            return images, meta

        out, bad = run_rowwise(
            pdf2img,
            batch,
            op="pdf2img",
            inputs=("pdf", "meta"),
            outputs=("images", "meta"),
            lineage=self.lineage,
            mutates=("meta",),
        )
        return out, bad, self.lineage.trace(out.path_ids[0]), list(self.lineage.mutations)


def test_runtime_rowwise_executes_inside_ray_module_actor() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2)
    module = RayModule(RuntimePdf2ImgOp, replicas=1).pre_init()
    try:
        batch = MicroBatch.source(
            {
                "pdf": ["paper0.pdf", "corrupt.pdf", "paper2.pdf"],
                "meta": [{"name": "paper0"}, {"name": "corrupt"}, {"name": "paper2"}],
            },
            dataset="ray-runtime-test",
        )

        out, bad, trace, mutations = module(batch)

        assert out.columns["images"] == [
            ["img<paper0.pdf:0>", "img<paper0.pdf:1>"],
            ["img<paper2.pdf:0>", "img<paper2.pdf:1>"],
        ]
        assert out.row_ids == [batch.row_ids[0], batch.row_ids[2]]
        assert [record.values["pdf"] for record in bad] == ["corrupt.pdf"]
        assert bad[0].op == "pdf2img"
        assert trace == ["pdf2img"]
        assert [(op, port) for _, op, port in mutations] == [
            ("pdf2img", "meta"),
            ("pdf2img", "meta"),
        ]
    finally:
        _cleanup(module)
        if ray.is_initialized():
            ray.shutdown()
