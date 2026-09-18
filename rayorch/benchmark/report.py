"""Serializable Benchmark reports detached from live Ray state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """One Benchmark run's outputs, metrics, profile, and artifacts."""

    benchmark: str
    run_id: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    profile: dict[str, Any]
    artifacts: dict[str, str]
    outputs: object | None = field(default=None, repr=False)

    @property
    def elapsed_s(self) -> float:
        return float(self.metrics["measured_wall_s"])

    def to_dict(
        self,
        *,
        include_outputs: bool = False,
        include_samples: bool = True,
    ) -> dict[str, Any]:
        value = {
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "config": self.config,
            "metrics": self.metrics,
            "profile": dict(self.profile),
            "artifacts": self.artifacts,
        }
        if not include_samples:
            value["profile"].pop("samples", None)
        if include_outputs:
            value["outputs"] = self.outputs
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BenchmarkReport:
        return cls(
            benchmark=value["benchmark"],
            run_id=value["run_id"],
            config=dict(value["config"]),
            metrics=dict(value["metrics"]),
            profile=dict(value["profile"]),
            artifacts=dict(value["artifacts"]),
            outputs=value.get("outputs"),
        )

    def write_json(
        self,
        path: str | Path,
        *,
        include_outputs: bool = False,
        include_samples: bool = True,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_dict(
                    include_outputs=include_outputs,
                    include_samples=include_samples,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    def print_summary(self) -> None:
        print(
            json.dumps(
                self.to_dict(include_samples=False),
                ensure_ascii=False,
                indent=2,
            )
        )


__all__ = [
    "BenchmarkReport",
]
