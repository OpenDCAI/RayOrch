"""Slow recovery-tier coverage on nested MinerU-shaped image graphs."""
from __future__ import annotations

import pytest
import ray

pytest.importorskip("PIL")

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import (
    IncompleteGroupPolicy,
    WorkerPoolSpec,
)
from rayorch.experimental.multigrain.ray import MultigrainRayExecutor

from test.experimental.multigrain.mineru_integration_ops import (
    AlwaysOpaqueImageFeature,
    AssembleDoc,
    DocsToPages,
    ImageFeature,
    OpaqueImageFeature,
    PagesToBlocks,
    PermanentImageFeature,
    TransientImageFeature,
    load_artifact_docs,
    record_key,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.mineru_integration,
    pytest.mark.usefixtures("ray_cluster"),
]


@pytest.fixture(scope="module")
def artifact_docs():
    docs = load_artifact_docs()
    if len(docs) < 2:
        pytest.skip("bounded MinerU regression artifacts are unavailable")
    return docs


def _docs_source(docs):
    return mg.source(docs, name="docs", display_key=lambda doc: doc["name"])


def _first_block(doc):
    return next(block for page in doc["pages"] for block in page["blocks"])


class RecoveryImagePipe(mg.Pipeline):
    def __init__(
        self,
        op_cls,
        policy: mg.RecoveryPolicy,
        *op_args,
        replicas: int = 2,
    ) -> None:
        super().__init__()
        self.to_pages = mg.Expand(DocsToPages, parent=0, child_label="page")
        self.to_blocks = mg.Expand(PagesToBlocks, parent=0, num_outputs=2)
        self.recover_image = mg.Map(
            op_cls,
            *op_args,
            name="recover_image",
            workers=WorkerPoolSpec(replicas=replicas),
            recovery=policy,
        )
        self.assemble = mg.Reduce(
            AssembleDoc,
            name="assemble_recovery",
        missing_child=IncompleteGroupPolicy.FAIL_CLOSED,
        )

    def forward(self, docs):
        pages = self.to_pages(docs)
        blocks, _ = self.to_blocks(pages)
        features = self.recover_image(blocks)
        return self.assemble(mg.group_by(docs, features))


def _run(graph, inputs, *, metrics=None):
    executor = MultigrainRayExecutor(metrics=metrics)
    try:
        return executor.execute(graph, inputs)
    finally:
        executor.shutdown()


def test_inline_record_retry_recovers_before_fail_closed_reduce(artifact_docs):
    target = record_key(_first_block(artifact_docs[0]))
    healthy_graph = RecoveryImagePipe(ImageFeature, mg.RecoveryPolicy()).compile()
    retry_graph = RecoveryImagePipe(
        TransientImageFeature,
        mg.RecoveryPolicy(max_record_retries=1),
        target,
    ).compile()
    inputs = {"docs": _docs_source(artifact_docs)}

    healthy = _run(healthy_graph, inputs)
    recovered = _run(retry_graph, inputs)

    assert recovered.values == healthy.values
    assert recovered.record_ids == healthy.record_ids
    assert recovered.errors == []


def test_permanent_page_descendant_suppresses_only_its_document(artifact_docs):
    bad_doc = artifact_docs[0]
    target = record_key(_first_block(bad_doc))
    graph = RecoveryImagePipe(
        PermanentImageFeature,
        mg.RecoveryPolicy(max_record_retries=2),
        target,
    ).compile()

    output = _run(graph, {"docs": _docs_source(artifact_docs)})
    by_doc = dict(zip(output.display_keys, output.values))

    assert by_doc[bad_doc["name"]]["status"] == "incomplete"
    assert all(
        value.get("status") != "incomplete"
        for name, value in by_doc.items()
        if name != bad_doc["name"]
    )
    assert {error.action for error in output.errors} >= {
        "quarantined",
        "suppressed_incomplete",
    }


def test_opaque_failure_adaptively_localizes_one_real_image(artifact_docs):
    bad_doc = artifact_docs[0]
    target = record_key(_first_block(bad_doc))
    policy = mg.RecoveryPolicy(
        max_shard_retries=0,
        on_shard_exhausted="degrade",
        isolation=mg.IsolationBudget(max_work_factor=3.0, max_calls=64),
    )
    graph = RecoveryImagePipe(OpaqueImageFeature, policy, target).compile()
    metrics = mg.RunMetrics()

    output = _run(graph, {"docs": _docs_source(artifact_docs)}, metrics=metrics)
    by_doc = dict(zip(output.display_keys, output.values))

    assert by_doc[bad_doc["name"]]["status"] == "incomplete"
    assert all(
        value.get("status") != "incomplete"
        for name, value in by_doc.items()
        if name != bad_doc["name"]
    )
    stage = metrics.by_name("recover_image")
    assert stage is not None
    assert 0 < stage.recovery_rows < len(
        [block for doc in artifact_docs for page in doc["pages"] for block in page["blocks"]]
    )


def test_dense_opaque_failures_stop_at_budget_and_fail_closed(artifact_docs):
    policy = mg.RecoveryPolicy(
        max_shard_retries=0,
        on_shard_exhausted="degrade",
        isolation=mg.IsolationBudget(max_work_factor=1.0, max_calls=8),
    )
    graph = RecoveryImagePipe(AlwaysOpaqueImageFeature, policy).compile()

    output = _run(graph, {"docs": _docs_source(artifact_docs)})

    assert all(value["status"] == "incomplete" for value in output.values)
    assert sum(error.action == "suppressed_incomplete" for error in output.errors) == len(
        artifact_docs
    )


def test_deferred_stage_retry_crosses_expands_then_unblocks_reduce(artifact_docs):
    targets = tuple(record_key(_first_block(doc)) for doc in artifact_docs)
    policy = mg.RecoveryPolicy(
        max_record_retries=1,
        retry_timing="deferred",
    )
    graph = RecoveryImagePipe(
        TransientImageFeature,
        policy,
        targets,
        replicas=1,
    ).compile()
    microbatches = [
        {"docs": _docs_source([doc])}
        for doc in artifact_docs
    ]
    executor = MultigrainRayExecutor()
    try:
        outputs = list(
            executor.execute_stream(
                graph,
                microbatches,
                max_inflight=len(microbatches),
            )
        )
    finally:
        executor.shutdown()

    assert len(outputs) == len(artifact_docs)
    assert all(output.errors == [] for output in outputs)
    assert [output.values[0]["doc"] for output in outputs] == [
        doc["name"] for doc in artifact_docs
    ]


def test_dead_image_actor_is_replaced_without_reloading_other_replicas(artifact_docs):
    graph = RecoveryImagePipe(
        ImageFeature,
        mg.RecoveryPolicy(max_shard_retries=1),
        replicas=2,
    ).compile()
    executor = MultigrainRayExecutor()
    try:
        executor.warm_pools(graph)
        survivor = executor._pools["recover_image"][1]
        ray.kill(executor._pools["recover_image"][0])
        output = executor.execute(graph, {"docs": _docs_source(artifact_docs)})

        assert output.errors == []
        assert executor._pools["recover_image"][1] == survivor
        assert ray.get(executor._pools["recover_image"][0].ping.remote()) is True
    finally:
        executor.shutdown()
