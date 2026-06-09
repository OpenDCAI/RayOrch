from __future__ import annotations

from types import SimpleNamespace

import pytest

from rayorch.runtime import MicroBatch
from rayorch.runtime.ray_module import dispatch_microbatch_shard_contiguous


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 8])
def test_contiguous_dispatch_preserves_all_rows_for_varied_sizes(size: int) -> None:
    module = SimpleNamespace(_replicas=4)
    batch = MicroBatch(
        columns={"x": list(range(size))},
        row_ids=[f"row:{index}" for index in range(size)],
        path_ids=["source"] * size,
    )

    per_args, _ = dispatch_microbatch_shard_contiguous(module, batch)
    shards = per_args[0]

    assert len(shards) == 4
    assert [
        value
        for shard in shards
        for value in shard.columns["x"]
    ] == list(range(size))
    assert max((len(shard) for shard in shards), default=0) <= (size + 3) // 4


def test_contiguous_dispatch_broadcasts_non_microbatch_values() -> None:
    module = SimpleNamespace(_replicas=3)
    batch = MicroBatch.source({"x": [1, 2, 3]})

    per_args, per_kwargs = dispatch_microbatch_shard_contiguous(
        module,
        batch,
        "constant",
        flag=True,
    )

    assert per_args[1] == ["constant"] * 3
    assert per_kwargs["flag"] == [True] * 3
