"""RayOrch actor pools use Ray's ordinary multi-node resource placement."""

from __future__ import annotations

import ray
from ray.cluster_utils import Cluster

from rayorch import Pipeline, RayModule, run


class _NodeIdentity:
    def run(self, values):
        node_id = ray.get_runtime_context().get_node_id()
        return [(value, node_id) for value in values]


class _GpuPool(Pipeline):
    def __init__(self) -> None:
        self.work = RayModule(_NodeIdentity).ray_options(
            replicas=2,
            batch_size=1,
            num_cpus=1,
            num_gpus=1,
        )

    def forward(self, values):
        return self.work(values)


def test_gpu_actor_pool_spans_two_ray_nodes():
    cluster = Cluster()
    try:
        cluster.add_node(num_cpus=2, num_gpus=1, include_dashboard=False)
        cluster.add_node(num_cpus=2, num_gpus=1)
        ray.init(address=cluster.address)

        result = run(_GpuPool(), ["left", "right"])

        assert [value for value, _ in result.outputs] == ["left", "right"]
        assert len({node_id for _, node_id in result.outputs}) == 2
    finally:
        ray.shutdown()
        cluster.shutdown()
