from __future__ import annotations

from typing import Any

from rayorch import Pipeline


class IdentityPipeline(Pipeline):
    def forward(self, values):
        return values


def test_pipeline_run_delegates_to_one_shot_runner(monkeypatch):
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_run(pipeline, *source_columns, **options):
        captured["pipeline"] = pipeline
        captured["source_columns"] = source_columns
        captured["options"] = options
        return sentinel

    monkeypatch.setattr("rayorch.runner.run", fake_run)

    pipeline = IdentityPipeline()
    result = pipeline.run(
        [1, 2],
        ["a", "b"],
        input_batch_size=1,
        max_active_input_batches=2,
        address="auto",
        ray_init_kwargs={"namespace": "test"},
    )

    assert result is sentinel
    assert captured == {
        "pipeline": pipeline,
        "source_columns": ([1, 2], ["a", "b"]),
        "options": {
            "input_batch_size": 1,
            "max_active_input_batches": 2,
            "address": "auto",
            "ray_init_kwargs": {"namespace": "test"},
        },
    }
