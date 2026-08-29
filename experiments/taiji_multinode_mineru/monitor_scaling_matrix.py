"""Monitor a five-topology matrix through completion, SIG audit, and cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

from hydp_engine.ray_jobs import RayJobClient

import orchestrate


CRASH_PATTERNS = (
    "sigsegv",
    "segmentation fault",
    "signal 11",
    "core dumped",
    "count::registerview",
    "metric::record",
    "viewdescriptor",
    "fatal python error",
)


def _event(name: str, payload: Mapping[str, Any]) -> None:
    print(name + " " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return orchestrate._load_json(path)


def _submitter_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return True


def _crash_audit(logs: list[str]) -> dict[str, Any]:
    matches = [
        line[-4000:]
        for line in logs
        if any(pattern in line.lower() for pattern in CRASH_PATTERNS)
    ]
    child_signal_lines = [
        line[-4000:]
        for line in logs
        if re.search(r"Child process \d+ exited from signal 11", line)
    ]
    return {
        "patterns": list(CRASH_PATTERNS),
        "match_count": len(matches),
        "matches": matches[-200:],
        "child_signal_lines": child_signal_lines[-100:],
    }


def _postflight(
    experiment: Mapping[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(orchestrate.EXPERIMENT_DIR / "monitor_active_full.py"),
        "--state",
        str(state_path),
        "--duration-s",
        "1",
        "--raylet-sig-audit",
        "--profile",
        str(experiment["profile"]),
    ]
    # The full result already performs the authoritative 3,690-document output
    # validation.  Recursively scanning the large Ceph tree again can exceed
    # the monitor job's 180-second deadline and lose the short-lived raylet
    # shutdown evidence.  Keep postflight focused on GPU-node/raylet state.
    completed = subprocess.run(
        command,
        cwd=orchestrate.REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker_names = (
        "RAYORCH_GPU_MONITOR_RESULT ",
        "RAYORCH_HDFS_OUTPUT_PROBE_RESULT ",
        "RAYORCH_RAYLET_SIG_AUDIT ",
    )
    markers: dict[str, Any] = {}
    for marker in marker_names:
        values = [
            json.loads(line.split(marker, 1)[1])
            for line in completed.stdout.splitlines()
            if marker in line
        ]
        markers[marker.strip()] = values[-1] if values else None
    return {
        "returncode": completed.returncode,
        "markers": markers,
        "stdout_tail": completed.stdout[-20_000:],
        "stderr_tail": completed.stderr[-20_000:],
        "checked_at": orchestrate._utc_now(),
    }


def _cleanup(experiment: Mapping[str, Any], state_path: Path) -> dict[str, Any]:
    state = _read_state(state_path)
    compute_id = str(state.get("compute_id") or "")
    if not compute_id:
        return {"skipped": True, "reason": "no_compute_id"}
    command = [
        sys.executable,
        str(orchestrate.EXPERIMENT_DIR / "orchestrate.py"),
        "cleanup",
        "--state-file",
        str(state_path),
        "--profile",
        str(experiment["profile"]),
        "--confirm-compute-id",
        compute_id,
    ]
    completed = subprocess.run(
        command,
        cwd=orchestrate.REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-20_000:],
        "stderr_tail": completed.stderr[-20_000:],
        "finished_at": orchestrate._utc_now(),
    }


def _terminal_audit(
    jobs: RayJobClient,
    experiment: dict[str, Any],
    state_path: Path,
    platform_status: str,
    *,
    cleanup: bool,
) -> None:
    state = _read_state(state_path)
    job = orchestrate._restore_job(state["full_job"])
    logs = jobs.logs(job, tail=12_000)
    log_path = state_path.with_name(state_path.stem + "_full_driver.log")
    orchestrate._save_logs(log_path, logs)
    driver_audit = _crash_audit(logs)
    try:
        result = orchestrate._marker(logs, orchestrate.FULL_MARKER)
    except Exception as error:
        result = None
        marker_error = f"{type(error).__name__}: {error}"
    else:
        marker_error = None
    postflight = _postflight(experiment, state_path)
    sig_marker = postflight["markers"].get("RAYORCH_RAYLET_SIG_AUDIT")
    raylet_signal_count = 0
    if isinstance(sig_marker, list):
        raylet_signal_count = sum(
            len(item.get("signal11_children") or [])
            for item in sig_marker
            if isinstance(item, Mapping)
        )
    state = _read_state(state_path)
    state.update(
        {
            "full_status": platform_status,
            "full_result": result,
            "full_result_error": marker_error,
            "driver_exit_crash_audit": driver_audit,
            "postflight_audit": postflight,
            "raylet_signal11_child_count": raylet_signal_count,
            "terminal_audited_at": orchestrate._utc_now(),
            "status": (
                "completed"
                if platform_status == "SUCCEEDED"
                and isinstance(result, Mapping)
                and result.get("status") == "completed"
                else "failed"
            ),
        }
    )
    orchestrate._atomic_json(state_path, state)
    experiment.update(
        {
            "platform_status": platform_status,
            "business_status": state["status"],
            "terminal_audited_at": state["terminal_audited_at"],
            "driver_crash_match_count": driver_audit["match_count"],
            "raylet_signal11_child_count": raylet_signal_count,
            "postflight_returncode": postflight["returncode"],
        }
    )
    if cleanup:
        cleanup_result = _cleanup(experiment, state_path)
        experiment["cleanup"] = cleanup_result
        experiment["cleanup_status"] = (
            "succeeded" if cleanup_result.get("returncode") == 0 else "failed"
        )
    _event(
        "SCALING_TERMINAL_AUDIT",
        {
            "label": experiment["label"],
            "platform_status": platform_status,
            "business_status": experiment["business_status"],
            "driver_crash_matches": driver_audit["match_count"],
            "raylet_signal11_children": raylet_signal_count,
            "postflight_returncode": postflight["returncode"],
            "cleanup_status": experiment.get("cleanup_status"),
        },
    )


def monitor(
    summary_path: Path,
    *,
    poll_interval: float,
    timeout_s: float,
    cleanup: bool,
) -> int:
    summary = orchestrate._load_json(summary_path)
    experiments = summary.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("matrix summary must contain at least one experiment")
    jobs = RayJobClient()
    deadline = time.monotonic() + timeout_s
    last_observations: dict[str, str] = {}
    terminal_labels = {
        str(item["label"])
        for item in experiments
        if item.get("terminal_audited_at")
    }
    while len(terminal_labels) < len(experiments):
        if time.monotonic() >= deadline:
            summary["monitor_status"] = "timed_out"
            summary["monitor_finished_at"] = orchestrate._utc_now()
            orchestrate._atomic_json(summary_path, summary)
            return 2
        for experiment in experiments:
            label = str(experiment["label"])
            if label in terminal_labels:
                continue
            state_path = Path(str(experiment["state_file"]))
            state = _read_state(state_path)
            full_payload = state.get("full_job")
            if isinstance(full_payload, Mapping):
                job = orchestrate._restore_job(full_payload)
                try:
                    platform_status = str(
                        jobs.status(job).get("status") or "UNKNOWN"
                    ).upper()
                except Exception as error:
                    observation = f"status_error:{type(error).__name__}:{error}"
                else:
                    observation = f"full:{platform_status}"
                    if platform_status in orchestrate.TERMINAL_JOB_STATES:
                        try:
                            _terminal_audit(
                                jobs,
                                experiment,
                                state_path,
                                platform_status,
                                cleanup=cleanup,
                            )
                        except Exception as error:
                            experiment["terminal_audit_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                            experiment["terminal_audit_error_at"] = orchestrate._utc_now()
                            observation = "terminal_audit_error"
                        else:
                            terminal_labels.add(label)
                            observation = f"audited:{platform_status}"
            elif state.get("status") == "failed" or state.get("phase") == "failed_before_detach":
                observation = "submission_failed"
                experiment.update(
                    {
                        "platform_status": state.get("full_status"),
                        "business_status": "submission_failed",
                        "submission_error": state.get("error"),
                        "terminal_audited_at": orchestrate._utc_now(),
                    }
                )
                terminal_labels.add(label)
            else:
                phase = str(state.get("phase") or state.get("status") or "pending")
                alive = _submitter_alive(experiment.get("submit_pid"))
                observation = f"submit:{phase}:alive={alive}"
                if not alive and state:
                    experiment["submitter_exited_before_full_job"] = True
            if last_observations.get(label) != observation:
                last_observations[label] = observation
                experiment["last_observation"] = observation
                experiment["last_observed_at"] = orchestrate._utc_now()
                _event(
                    "SCALING_STATUS",
                    {"label": label, "observation": observation},
                )
            orchestrate._atomic_json(summary_path, summary)
        if len(terminal_labels) < len(experiments):
            time.sleep(poll_interval)
    summary["monitor_status"] = "completed"
    summary["monitor_finished_at"] = orchestrate._utc_now()
    summary["all_business_succeeded"] = all(
        item.get("business_status") == "completed" for item in experiments
    )
    summary["any_sigsegv_evidence"] = any(
        int(item.get("driver_crash_match_count") or 0) > 0
        or int(item.get("raylet_signal11_child_count") or 0) > 0
        for item in experiments
    )
    orchestrate._atomic_json(summary_path, summary)
    _event(
        "SCALING_MATRIX_COMPLETE",
        {
            "all_business_succeeded": summary["all_business_succeeded"],
            "any_sigsegv_evidence": summary["any_sigsegv_evidence"],
            "summary_file": str(summary_path),
        },
    )
    return 0 if summary["all_business_succeeded"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=172800.0)
    parser.add_argument("--cleanup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return monitor(
        args.summary_file,
        poll_interval=args.poll_interval,
        timeout_s=args.timeout_s,
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
