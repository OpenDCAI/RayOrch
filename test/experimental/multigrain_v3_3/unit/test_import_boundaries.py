"""顶层 API 与 Ray-free import 边界。"""

from __future__ import annotations

import subprocess
import sys

import rayorch.experimental.multigrain_v3_3 as mg


def test_top_level_api_exposes_small_authoring_and_execution_surface():
    assert mg.F is mg.functional
    assert mg.F.expand is mg.functional.expand
    assert mg.LocalExecutor.__name__ == "LocalExecutor"
    assert mg.RayExecutor.__name__ == "RayExecutor"
    assert not hasattr(mg, "ExpandOrigin")


def test_importing_v33_does_not_initialize_ray():
    code = """
import ray
assert not ray.is_initialized()
import rayorch.experimental.multigrain_v3_3
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
