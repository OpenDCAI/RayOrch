"""Docling benchmark 使用的轻量 GPU 采样器。"""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuDeviceSample:
    """单张 GPU 在一个采样时刻的指标。"""

    index: int
    utilization_percent: int | None
    memory_used_bytes: int | None
    power_watts: float | None


@dataclass(frozen=True, slots=True)
class GpuSample:
    """固定时刻采集到的多卡 GPU 指标。"""

    timestamp: float
    devices: tuple[GpuDeviceSample, ...]


def _load_pynvml() -> Any:
    """延迟导入 pynvml，使没有 NVIDIA 环境的测试可正常运行。"""

    return importlib.import_module("pynvml")


class GpuMonitor:
    """在 driver 线程中按固定间隔采集指定 GPU 的 NVML 指标。

    NVML 或驱动不可用时，``start`` 不抛异常；``available`` 置为 ``False``，
    ``stop`` 返回空 samples，并通过 ``unavailable_reason`` 暴露原因。
    """

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        device_indices: tuple[int, ...] = (0, 1, 2, 3),
        jsonl_path: str | Path | None = None,
    ) -> None:
        """保存四卡采样设置，尚不访问 NVML。"""

        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if not device_indices:
            raise ValueError("device_indices must not be empty")
        self.interval_s = interval_s
        self.device_indices = device_indices
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.available: bool | None = None
        self.unavailable_reason: str | None = None
        self._nvml: Any | None = None
        self._samples: list[GpuSample] = []
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuMonitor":
        """初始化 NVML、立即采一条样本，并启动 driver 采样线程。"""

        if self._thread is not None:
            raise RuntimeError("GpuMonitor has already been started")
        try:
            self._nvml = _load_pynvml()
            self._nvml.nvmlInit()
        except Exception as error:
            self.available = False
            self.unavailable_reason = f"{type(error).__name__}: {error}"
            self._nvml = None
            return self

        self.available = True
        self._append_sample()
        self._thread = threading.Thread(
            target=self._run,
            name="docling-gpu-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> tuple[GpuSample, ...]:
        """停止采样、释放 NVML，并返回已收集的不可变 samples。"""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            finally:
                self._nvml = None
        with self._samples_lock:
            return tuple(self._samples)

    def _run(self) -> None:
        """使用单调时钟维护固定采样节奏。"""

        next_tick = time.monotonic() + self.interval_s
        while not self._stop.is_set():
            delay = max(0.0, next_tick - time.monotonic())
            if self._stop.wait(delay):
                return
            self._append_sample()
            next_tick += self.interval_s

    def _append_sample(self) -> None:
        """best-effort 采样一次；单张卡失败不影响其余 GPU。"""

        if self._nvml is None:
            return
        devices = tuple(
            self._sample_device(index) for index in self.device_indices
        )
        sample = GpuSample(timestamp=time.time(), devices=devices)
        with self._samples_lock:
            self._samples.append(sample)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(asdict(sample), ensure_ascii=False) + "\n"
                    )

    def _sample_device(self, index: int) -> GpuDeviceSample:
        """采集一张卡的利用率、显存已用量与功耗。"""

        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            memory = self._nvml.nvmlDeviceGetMemoryInfo(handle).used
            power_watts = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            return GpuDeviceSample(
                index=index,
                utilization_percent=int(utilization),
                memory_used_bytes=int(memory),
                power_watts=float(power_watts),
            )
        except Exception:
            return GpuDeviceSample(
                index=index,
                utilization_percent=None,
                memory_used_bytes=None,
                power_watts=None,
            )
