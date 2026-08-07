"""顶层 API 与 Ray-free import 边界。"""

from __future__ import annotations

import subprocess
import sys

import rayorch.experimental.multigrain_v3_4 as mg


def test_top_level_api_exposes_small_authoring_and_execution_surface():
    assert mg.F is mg.functional
    assert mg.F.expand is mg.functional.expand
    assert mg.Executor.__name__ == "Executor"
    assert not hasattr(mg, "ExpandOrigin")


def test_importing_v34_does_not_initialize_ray():
    code = """
import ray
assert not ray.is_initialized()
import rayorch.experimental.multigrain_v3_4
assert not ray.is_initialized()
print('ray-not-initialized')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ray-not-initialized"
