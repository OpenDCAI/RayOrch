from __future__ import annotations

import pytest
import ray


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked with @pytest.mark.slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def ray_cluster():
    """Share one local Ray cluster across opted-in integration tests."""
    # ``address="local"`` prevents an ambient RAY_ADDRESS / remembered cluster
    # from turning this local fixture into a remote connection, where Ray
    # rejects the requested num_cpus.
    ray.init(
        address="local",
        ignore_reinit_error=True,
        num_cpus=16,
        include_dashboard=False,
    )
    yield
    ray.shutdown()
