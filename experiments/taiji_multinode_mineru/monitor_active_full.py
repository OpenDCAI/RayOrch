"""Submit a short GPU monitor job alongside an active MinerU job."""

from __future__ import annotations

import argparse
import json
import shlex
import time
import uuid
from pathlib import Path

from hydp_dataflow_wedata.config import load_profile
from hydp_engine.ray_jobs import RayJobClient
from hydp_engine.ray_jobs.packaging import maybe_upload_ray_working_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--output-root")
    parser.add_argument("--hdfs-output-uri")
    parser.add_argument("--raylet-diag", action="store_true")
    parser.add_argument("--raylet-sig-audit", action="store_true")
    parser.add_argument("--inspect-child-pid", type=int)
    parser.add_argument("--inspect-node-id")
    parser.add_argument("--arm-raylet-diag", action="store_true")
    parser.add_argument("--diag-hdfs-uri")
    parser.add_argument("--verify-raylet-arm", action="store_true")
    parser.add_argument("--collect-raylet-crash", action="store_true")
    parser.add_argument("--crash-epoch", type=int)
    parser.add_argument("--read-hdfs-diag", action="store_true")
    parser.add_argument(
        "--profile",
        default=f"{Path(__file__).with_name('profile.toml')}:ray-h20-8x8-gy",
    )
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    dashboard = str(state["dashboard_url"])
    expected_gpu_nodes = int(
        state.get("plan", {}).get("compute", {}).get("workers", 8)
    )
    credentials = load_profile(args.profile).credentials()
    jobs = RayJobClient()
    runtime_env: dict = {}
    maybe_upload_ray_working_dir(
        runtime_env,
        working_dir=str(Path(__file__).with_name("gpu_monitor_job")),
        address=dashboard,
        proxy_url=None,
        proxy_post=jobs._proxy_post,
        package_builder=jobs._package_builder,
    )
    entrypoint = (
        f"python -u monitor.py --duration-s {args.duration_s} "
        f"--expected-gpu-nodes {expected_gpu_nodes}"
    )
    if args.output_root:
        entrypoint += " --output-root " + shlex.quote(args.output_root)
    if args.hdfs_output_uri:
        entrypoint += " --hdfs-output-uri " + shlex.quote(args.hdfs_output_uri)
    if args.raylet_diag:
        entrypoint += " --raylet-diag"
    if args.raylet_sig_audit:
        entrypoint += " --raylet-sig-audit"
    if args.inspect_child_pid is not None:
        entrypoint += f" --inspect-child-pid {args.inspect_child_pid}"
    if args.inspect_node_id:
        entrypoint += " --inspect-node-id " + shlex.quote(args.inspect_node_id)
    if args.arm_raylet_diag:
        entrypoint += " --arm-raylet-diag"
    if args.diag_hdfs_uri:
        entrypoint += " --diag-hdfs-uri " + shlex.quote(args.diag_hdfs_uri)
    if args.verify_raylet_arm:
        entrypoint += " --verify-raylet-arm"
    if args.collect_raylet_crash:
        entrypoint += " --collect-raylet-crash"
    if args.crash_epoch is not None:
        entrypoint += f" --crash-epoch {args.crash_epoch}"
    if args.read_hdfs_diag:
        entrypoint += " --read-hdfs-diag"
    job = jobs.submit_proxy(
        entrypoint=entrypoint,
        submission_id=f"rayorch-gpu-monitor-{uuid.uuid4().hex[:12]}",
        dashboard_url=dashboard,
        runtime_env=runtime_env,
        working_dir=None,
        runtime_auth=True,
        user=credentials.user,
        cmk=credentials.cmk,
        metadata={"hydp_stage": "rayorch_gpu_monitor"},
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status = str(jobs.status(job).get("status") or "UNKNOWN").upper()
        if status in {"SUCCEEDED", "FAILED", "STOPPED", "ERROR"}:
            logs = "\n".join(jobs.logs(job, tail=2000))
            lines = [
                line.split("RAYORCH_GPU_MONITOR_RESULT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_GPU_MONITOR_RESULT " in line
            ]
            if status != "SUCCEEDED":
                raise RuntimeError(f"monitor failed with {status}: {logs[-2000:]}")
            output_lines = [
                line.split("RAYORCH_OUTPUT_PROBE_RESULT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_OUTPUT_PROBE_RESULT " in line
            ]
            hdfs_lines = [
                line.split("RAYORCH_HDFS_OUTPUT_PROBE_RESULT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_HDFS_OUTPUT_PROBE_RESULT " in line
            ]
            diag_lines = [
                line.split("RAYORCH_RAYLET_DIAG_RESULT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_RAYLET_DIAG_RESULT " in line
            ]
            sig_audit_lines = [
                line.split("RAYORCH_RAYLET_SIG_AUDIT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_RAYLET_SIG_AUDIT " in line
            ]
            child_inspection_lines = [
                line.split("RAYORCH_CHILD_SIGNAL_INSPECTION ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_CHILD_SIGNAL_INSPECTION " in line
            ]
            armed_lines = [
                line.split("RAYORCH_RAYLET_DIAG_ARMED ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_RAYLET_DIAG_ARMED " in line
            ]
            verified_lines = [
                line.split("RAYORCH_RAYLET_DIAG_VERIFIED ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_RAYLET_DIAG_VERIFIED " in line
            ]
            collected_lines = [
                line.split("RAYORCH_RAYLET_CRASH_COLLECTED ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_RAYLET_CRASH_COLLECTED " in line
            ]
            diag_read_lines = [
                line.split("RAYORCH_HDFS_DIAG_READ_RESULT ", 1)[1]
                for line in logs.splitlines()
                if "RAYORCH_HDFS_DIAG_READ_RESULT " in line
            ]
            required_ready = bool(lines)
            if args.output_root:
                required_ready = required_ready and bool(output_lines)
            if args.hdfs_output_uri:
                required_ready = required_ready and bool(hdfs_lines)
            if args.raylet_diag:
                required_ready = required_ready and bool(diag_lines)
            if args.raylet_sig_audit:
                required_ready = required_ready and bool(sig_audit_lines)
            if args.inspect_child_pid is not None:
                required_ready = required_ready and bool(child_inspection_lines)
            if args.arm_raylet_diag:
                required_ready = required_ready and bool(armed_lines)
            if args.verify_raylet_arm:
                required_ready = required_ready and bool(verified_lines)
            if args.collect_raylet_crash:
                required_ready = required_ready and bool(collected_lines)
            if args.read_hdfs_diag:
                required_ready = required_ready and bool(diag_read_lines)
            if not required_ready:
                if "Traceback (most recent call last)" in logs:
                    raise RuntimeError(
                        f"monitor driver failed despite SUCCEEDED: {logs[-2000:]}"
                    )
                time.sleep(3)
                continue
            print("RAYORCH_GPU_MONITOR_RESULT " + lines[-1])
            if output_lines:
                print("RAYORCH_OUTPUT_PROBE_RESULT " + output_lines[-1])
            if hdfs_lines:
                print("RAYORCH_HDFS_OUTPUT_PROBE_RESULT " + hdfs_lines[-1])
            if diag_lines:
                print("RAYORCH_RAYLET_DIAG_RESULT " + diag_lines[-1])
            if sig_audit_lines:
                print("RAYORCH_RAYLET_SIG_AUDIT " + sig_audit_lines[-1])
            if child_inspection_lines:
                print(
                    "RAYORCH_CHILD_SIGNAL_INSPECTION "
                    + child_inspection_lines[-1]
                )
            if armed_lines:
                print("RAYORCH_RAYLET_DIAG_ARMED " + armed_lines[-1])
            if verified_lines:
                print("RAYORCH_RAYLET_DIAG_VERIFIED " + verified_lines[-1])
            if collected_lines:
                print("RAYORCH_RAYLET_CRASH_COLLECTED " + collected_lines[-1])
            if diag_read_lines:
                print("RAYORCH_HDFS_DIAG_READ_RESULT " + diag_read_lines[-1])
            return 0
        time.sleep(3)
    raise TimeoutError("GPU monitor job timed out")


if __name__ == "__main__":
    raise SystemExit(main())
