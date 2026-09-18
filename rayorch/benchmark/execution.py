"""Shared execution lifecycle for Benchmark workloads."""

from __future__ import annotations

import json
import platform
import threading
import time
import uuid
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rayorch._execution.executor import Executor
from rayorch.api import Pipeline
from rayorch.result import RunResult
from rayorch.version import __version__

from .report import BenchmarkReport


def benchmark_config(value: object) -> dict[str, Any]:
    """Convert one dataclass Benchmark configuration to JSON-safe values."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("Benchmark configuration must be a dataclass instance")
    return _json_value(asdict(value))


def run_benchmark(
    *,
    name: str,
    pipeline: Pipeline,
    source_columns: tuple[Sequence[Any], ...],
    config: Mapping[str, Any],
    artifact_root: str | Path,
    input_batch_size: int | None = None,
    max_active_input_batches: int = 1,
    ray_address: str | None = None,
    ray_init_kwargs: Mapping[str, Any] | None = None,
    profile: bool = True,
    profile_interval_s: float = 1.0,
    run_id: str | None = None,
    extra_metrics: Callable[[RunResult], Mapping[str, Any]] | None = None,
) -> BenchmarkReport:
    """Run a Pipeline and persist one standard Benchmark report."""

    identifier = run_id or _new_run_id()
    sampler = ResourceSampler(enabled=profile, interval_s=profile_interval_s)
    sampler.start()
    started = time.perf_counter()
    result = None
    executor = None
    try:
        init_kwargs = dict(ray_init_kwargs or {})
        init_kwargs.setdefault("include_dashboard", False)
        executor = Executor(
            pipeline,
            address=ray_address,
            ray_init_kwargs=init_kwargs,
        )
        startup_s = time.perf_counter() - started
        result = executor.run(
            *source_columns,
            input_batch_size=input_batch_size,
            max_active_input_batches=max_active_input_batches,
        )
        end_to_end_s = time.perf_counter() - started
    finally:
        if executor is not None:
            executor.close()
        profile_result = sampler.stop()
    if result is None:  # pragma: no cover - execution exceptions escape
        raise RuntimeError(f"{name} Benchmark produced no result")

    import ray  # pyright: ignore[reportMissingImports]

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "ray": ray.__version__,
            "rayorch": __version__,
        },
        "input_rows": len(source_columns[0]) if source_columns else 0,
        "output_rows": _length(result.outputs),
        "startup_s": round(startup_s, 3),
        "measured_wall_s": round(result.elapsed_s, 3),
        "end_to_end_wall_s": round(end_to_end_s, 3),
        "rpc_count": result.rpc_count,
        "actor_count": result.actor_count,
        "peak_active_input_batches": result.peak_active_input_batches,
        "released_values": result.released_values,
        "calls": [asdict(call) for call in result.calls],
    }
    if extra_metrics is not None:
        metrics.update(extra_metrics(result))

    run_dir = Path(artifact_root).resolve() / identifier
    artifacts = {
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.json"),
        "summary": str(run_dir / "summary.json"),
        "gpu_samples": str(run_dir / "gpu_samples.jsonl"),
    }
    report = BenchmarkReport(
        benchmark=name,
        run_id=identifier,
        config=dict(config),
        metrics=metrics,
        profile=profile_result,
        artifacts=artifacts,
        outputs=result.outputs,
    )
    _write_artifacts(report)
    return report


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _length(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _write_artifacts(report: BenchmarkReport) -> None:
    run_dir = Path(report.artifacts["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(report.artifacts["config"]).write_text(
        json.dumps(report.config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with Path(report.artifacts["gpu_samples"]).open("w", encoding="utf-8") as handle:
        for sample in report.profile.get("samples", ()):
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report.write_json(
        report.artifacts["summary"],
        include_outputs=True,
        include_samples=False,
    )


class ResourceSampler:
    """Best-effort sampling of driver RSS and driver-visible GPUs."""

    def __init__(self, *, enabled: bool, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("profile interval must be positive")
        self.enabled = enabled
        self.interval_s = interval_s
        self._detail: str | None = None
        self._process = None
        self._driver_start: int | None = None
        self._driver_peak: int | None = None
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if not enabled:
            return
        try:
            import psutil  # pyright: ignore[reportMissingModuleSource]

            self._process = psutil.Process()
            self._driver_start = int(self._process.memory_info().rss)
            self._driver_peak = self._driver_start
            self._thread = threading.Thread(target=self._sample, daemon=True)
        except Exception as error:
            self._detail = f"{type(error).__name__}: {error}"

    def start(self) -> None:
        if self._thread is not None:
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "samples": []}
        if self._thread is None:
            return {
                "status": "unavailable",
                "detail": self._detail,
                "samples": [],
            }
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 2))
        self._capture()
        return {
            "status": "collected",
            "driver_rss_start": self._driver_start,
            "driver_rss_peak": self._driver_peak,
            "visible_gpu_memory_peak": _gpu_memory_peaks(self._samples),
            "samples": list(self._samples),
        }

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._capture()

    def _capture(self) -> None:
        try:
            assert self._process is not None
            assert self._driver_peak is not None
            self._driver_peak = max(
                self._driver_peak,
                int(self._process.memory_info().rss),
            )
            self._samples.append(_gpu_sample())
        except Exception:
            return


def _gpu_sample() -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The pynvml package is deprecated.*",
                category=FutureWarning,
            )
            import pynvml  # pyright: ignore[reportMissingImports]

        pynvml.nvmlInit()
        utilization = []
        memory = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                utilization.append(
                    int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                )
            except Exception:
                utilization.append(None)
            memory.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        pynvml.nvmlShutdown()
        return {
            "monotonic_s": time.monotonic(),
            "utilization": utilization,
            "memory_used": memory,
        }
    except Exception:
        return {
            "monotonic_s": time.monotonic(),
            "utilization": [],
            "memory_used": [],
        }


def _gpu_memory_peaks(samples: list[dict[str, Any]]) -> list[int]:
    count = max((len(sample["memory_used"]) for sample in samples), default=0)
    return [
        max(
            (
                sample["memory_used"][index]
                for sample in samples
                if index < len(sample["memory_used"])
            ),
            default=0,
        )
        for index in range(count)
    ]


__all__ = ["benchmark_config", "run_benchmark"]
