"""Ray Jobs submission for the public Benchmark API."""

from __future__ import annotations

import base64
import copy
import json
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .registry import class_path, runtime_env as workload_runtime_env
from .report import BenchmarkReport


def _installed_environment(workload_env: Mapping[str, Any]) -> dict[str, Any]:
    """Keep non-install settings while trusting the cluster's Python environment."""

    return {
        key: copy.deepcopy(value)
        for key, value in workload_env.items()
        if key not in {"pip", "conda", "uv", "py_modules", "working_dir"}
    }


@dataclass(frozen=True, slots=True)
class LocalSource:
    """Upload source and, by default, install the workload's declared dependencies."""

    project_root: str | Path
    modules: tuple[str | Path, ...] = ()
    install_dependencies: bool = True
    excludes: tuple[str, ...] = (
        ".git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "build/**",
        "dist/**",
    )

    def runtime_env(self, workload_env: Mapping[str, Any]) -> dict[str, Any]:
        root = Path(self.project_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"project source does not exist: {root}")
        modules = tuple(Path(module).resolve() for module in self.modules)
        missing = [path for path in modules if not path.exists()]
        if missing:
            raise FileNotFoundError(f"module source does not exist: {missing[0]}")
        value = (
            copy.deepcopy(dict(workload_env))
            if self.install_dependencies
            else _installed_environment(workload_env)
        )
        value["working_dir"] = str(root)
        if modules:
            value["py_modules"] = [str(module) for module in modules]
        if self.excludes:
            value["excludes"] = list(self.excludes)
        return value


@dataclass(slots=True)
class BenchmarkRun:
    """Handle for one asynchronously submitted remote Benchmark."""

    job_id: str
    report_path: Path
    _client: Any

    def status(self) -> str:
        return str(self._client.get_job_status(self.job_id))

    def logs(self) -> str:
        return str(self._client.get_job_logs(self.job_id))

    def stop(self) -> bool:
        return bool(self._client.stop_job(self.job_id))

    def wait(
        self,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> BenchmarkReport:
        """Wait for Job completion and read its report from shared storage."""

        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        started = time.monotonic()
        while True:
            status = self.status()
            normalized = status.rsplit(".", 1)[-1]
            if normalized == "SUCCEEDED":
                if not self.report_path.is_file():
                    raise FileNotFoundError(
                        "Ray Job succeeded but its Benchmark report is not visible "
                        f"at {self.report_path}; use cluster-visible shared storage"
                    )
                return BenchmarkReport.from_dict(
                    json.loads(self.report_path.read_text(encoding="utf-8"))
                )
            if normalized in {"FAILED", "STOPPED"}:
                raise RuntimeError(
                    f"Ray Job {self.job_id} finished with status {status}\n"
                    f"{self.logs()}"
                )
            if (
                timeout_s is not None
                and time.monotonic() - started >= timeout_s
            ):
                raise TimeoutError(
                    f"timed out waiting for Ray Job {self.job_id}; "
                    "the Job was not stopped"
                )
            time.sleep(poll_interval_s)


def submit_benchmark(
    *,
    benchmark: str,
    config: Mapping[str, Any],
    artifact_root: str | Path,
    target: str,
    source: LocalSource | None = None,
    profile: bool = True,
    profile_interval_s: float = 1.0,
    run_id: str | None = None,
) -> BenchmarkRun:
    """Submit one serialized Benchmark configuration through Ray Jobs."""

    from ray.job_submission import (  # pyright: ignore[reportMissingImports]
        JobSubmissionClient,
    )

    if not target:
        raise ValueError("Ray Jobs target must be non-empty")
    identifier = run_id or _new_run_id()
    benchmark_class = class_path(benchmark)
    workload_env = workload_runtime_env(benchmark)
    runtime_env = (
        source.runtime_env(workload_env)
        if source is not None
        else _installed_environment(workload_env)
    )
    encoded = base64.urlsafe_b64encode(
        json.dumps(dict(config), separators=(",", ":")).encode()
    ).decode()
    entrypoint = shlex.join(
        [
            "python",
            "-m",
            "rayorch.benchmark._job",
            "--benchmark-class",
            benchmark_class,
            "--config",
            encoded,
            "--run-id",
            identifier,
            "--profile",
            "1" if profile else "0",
            "--profile-interval-s",
            str(profile_interval_s),
        ]
    )
    client = JobSubmissionClient(target)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metadata={
            "rayorch.benchmark": benchmark,
            "rayorch.benchmark_class": benchmark_class,
            "rayorch.run_id": identifier,
        },
    )
    report_path = Path(artifact_root).resolve() / identifier / "summary.json"
    return BenchmarkRun(str(job_id), report_path, client)


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "BenchmarkRun",
    "LocalSource",
    "submit_benchmark",
]
