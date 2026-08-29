"""Run deterministic mixed-input MinerU tuning candidates on one retained compute."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import orchestrate


OWNER_STATE = orchestrate.EXPERIMENT_DIR / "state" / (
    "rayorch_v36_combined3690_8x8_h20_gy_20260820_220456.json"
)
SAMPLE_LIMITS = (256, 256)
CANDIDATES = (
    {
        "name": "a",
        "microbatch_size": 24,
        "max_active_microbatches": 10,
        "batch_size": 64,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 120,
        "reduce_replicas": 64,
    },
    {
        "name": "b",
        "microbatch_size": 24,
        "max_active_microbatches": 16,
        "batch_size": 64,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "c",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 64,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 512,
        "reduce_replicas": 64,
    },
    {
        "name": "d",
        "microbatch_size": 24,
        "max_active_microbatches": 32,
        "batch_size": 64,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 768,
        "reduce_replicas": 128,
    },
    {
        "name": "e",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 64,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "f",
        "microbatch_size": 24,
        "max_active_microbatches": 16,
        "batch_size": 48,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "g",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 48,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "h",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 32,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "i",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 96,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "j",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 128,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "k",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 192,
        "gpu_memory_utilization": 0.8,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "l",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 96,
        "gpu_memory_utilization": 0.4,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "m",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 64,
        "gpu_memory_utilization": 0.4,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "n",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 96,
        "gpu_memory_utilization": 0.26,
        "ocr_replicas": 192,
        "gpus_per_ocr_actor": 0.3333333333333333,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "o",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 64,
        "gpu_memory_utilization": 0.32,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "p",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 96,
        "gpu_memory_utilization": 0.32,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "q",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 64,
        "gpu_memory_utilization": 0.20,
        "ocr_replicas": 192,
        "gpus_per_ocr_actor": 0.3333333333333333,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "r",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 32,
        "gpu_memory_utilization": 0.32,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
    {
        "name": "s",
        "microbatch_size": 24,
        "max_active_microbatches": 24,
        "batch_size": 48,
        "gpu_memory_utilization": 0.32,
        "ocr_replicas": 128,
        "gpus_per_ocr_actor": 0.5,
        "render_replicas": 256,
        "reduce_replicas": 64,
    },
)


def _direct_job(dashboard: str, submission_id: str, suffix: str = "") -> Any:
    url = dashboard.rstrip("/") + f"/api/jobs/{submission_id}{suffix}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _wait_result(state_path: Path, *, timeout_s: float = 7_200) -> dict[str, Any]:
    state = orchestrate._load_json(state_path)
    submission_id = str(state["full_submission_id"])
    dashboard = str(state["dashboard_url"])
    deadline = time.monotonic() + timeout_s
    last_status = ""
    while True:
        detail = _direct_job(dashboard, submission_id)
        status = str(detail.get("status") or "UNKNOWN").upper()
        if status != last_status:
            print(
                "SWEEP_JOB_STATUS "
                + json.dumps(
                    {"submission_id": submission_id, "status": status},
                    sort_keys=True,
                ),
                flush=True,
            )
            last_status = status
        if status in orchestrate.TERMINAL_JOB_STATES:
            logs = str(_direct_job(dashboard, submission_id, "/logs").get("logs") or "")
            log_path = state_path.with_name(state_path.stem + "_full_driver.log")
            log_path.write_text(logs, encoding="utf-8")
            marker_lines = [
                line.split(orchestrate.FULL_MARKER, 1)[1]
                for line in logs.splitlines()
                if orchestrate.FULL_MARKER in line
            ]
            result = json.loads(marker_lines[-1]) if marker_lines else None
            state.update(
                {
                    "direct_full_status": status,
                    "sweep_result": result,
                    "sweep_checked_at": orchestrate._utc_now(),
                }
            )
            orchestrate._atomic_json(state_path, state)
            if status != "SUCCEEDED" or not isinstance(result, dict):
                raise RuntimeError(
                    f"sweep candidate failed: status={status}, marker={bool(result)}"
                )
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"sweep candidate timed out: {submission_id}")
        time.sleep(10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="a,b,c,d",
        help="comma-separated candidate names to run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    requested = {
        item.strip() for item in build_parser().parse_args(argv).candidates.split(",")
        if item.strip()
    }
    candidates = [item for item in CANDIDATES if item["name"] in requested]
    if not candidates or requested != {item["name"] for item in candidates}:
        raise ValueError(f"unknown or empty candidate selection: {sorted(requested)}")
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = orchestrate.EXPERIMENT_DIR / "state" / f"sweep_{sweep_id}.json"
    summary: dict[str, Any] = {
        "sweep_id": sweep_id,
        "sample_limits": list(SAMPLE_LIMITS),
        "owner_state": str(OWNER_STATE),
        "candidates": [],
    }
    orchestrate._atomic_json(summary_path, summary)
    for candidate in candidates:
        parameters = dict(orchestrate.HYPERPARAMETERS)
        parameters.update(
            {key: value for key, value in candidate.items() if key != "name"}
        )
        run_id = f"rayorch_v36_sweep_{candidate['name']}_{sweep_id}"
        state_path = orchestrate._state_default(run_id)
        plan = orchestrate.build_plan(
            run_id=run_id,
            input_limits=SAMPLE_LIMITS,
            hyperparameters=parameters,
            benchmark_only=True,
        )
        code = orchestrate.execute_submit(
            plan,
            profile_reference=orchestrate.DEFAULT_PROFILE_REF,
            state_path=state_path,
            reuse_state_path=OWNER_STATE,
        )
        if code != 0:
            raise RuntimeError(f"candidate {candidate['name']} failed to submit")
        try:
            result = _wait_result(state_path)
            benchmark = dict(result["benchmark"])
            gpu = dict(result["gpu_sampling"])
            record = {
                "name": candidate["name"],
                "parameters": parameters,
                "state_file": str(state_path),
                "status": "completed",
                "pages": benchmark.get("pages"),
                "pages_per_s": benchmark.get("pages_per_s"),
                "measured_wall_s": benchmark.get("measured_wall_s"),
                "failed_doc_count": benchmark.get("failed_doc_count"),
                "active_mean_gpu_utilization": gpu.get(
                    "active_mean_gpu_utilization"
                ),
                "active_p90_gpu_utilization": gpu.get(
                    "active_p90_gpu_utilization"
                ),
                "measured_mean_gpu_utilization": gpu.get(
                    "measured_mean_gpu_utilization"
                ),
                "measured_p90_gpu_utilization": gpu.get(
                    "measured_p90_gpu_utilization"
                ),
            }
        except Exception as error:
            record = {
                "name": candidate["name"],
                "parameters": parameters,
                "state_file": str(state_path),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        summary["candidates"].append(record)
        orchestrate._atomic_json(summary_path, summary)
        print("SWEEP_CANDIDATE_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    eligible = [
        item
        for item in summary["candidates"]
        if item.get("status") == "completed"
        and float(item.get("measured_mean_gpu_utilization") or 0) >= 90.0
    ]
    summary["winner"] = (
        max(eligible, key=lambda item: float(item.get("pages_per_s") or 0))
        if eligible
        else None
    )
    summary["finished_at"] = orchestrate._utc_now()
    orchestrate._atomic_json(summary_path, summary)
    print("SWEEP_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
