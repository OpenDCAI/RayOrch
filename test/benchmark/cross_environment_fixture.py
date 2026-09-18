"""Dependency-free UDFs used by the opt-in cross-environment integration test."""

from __future__ import annotations

import os
import sys

from rayorch import Pipeline, RayModule


class EnvironmentProbe:
    def run(self, values):
        import ray

        return [
            {
                "value": value,
                "executable": sys.executable,
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "ray": ray.__version__,
            }
            for value in values
        ]


class CrossEnvironmentPipeline(Pipeline):
    def __init__(self, environment: str) -> None:
        self.probe = RayModule(EnvironmentProbe).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=1,
            runtime_env={"conda": environment},
        )

    def forward(self, values):
        return self.probe(values)


class BackendProbe:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def run(self, values):
        import importlib

        package = importlib.import_module(self.backend)
        return [
            {
                "value": value,
                "backend": self.backend,
                "version": getattr(package, "__version__", "unknown"),
                "executable": sys.executable,
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            }
            for value in values
        ]


class BackendEnvironmentPipeline(Pipeline):
    def __init__(self, sglang_env: str, vllm_env: str) -> None:
        self.sglang = (
            RayModule(BackendProbe)
            .pre_init("sglang")
            .ray_options(
                replicas=1,
                batch_size=2,
                num_cpus=1,
                runtime_env={"conda": sglang_env},
            )
        )
        self.vllm = (
            RayModule(BackendProbe)
            .pre_init("vllm")
            .ray_options(
                replicas=1,
                batch_size=2,
                num_cpus=1,
                runtime_env={"conda": vllm_env},
            )
        )

    def forward(self, values):
        return self.vllm(self.sglang(values))


__all__ = ["BackendEnvironmentPipeline", "CrossEnvironmentPipeline"]
