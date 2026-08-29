"""Sample all GPU workers of an existing TaiJi Ray compute for 30 seconds."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _percentile(values: list[int], percentile: float) -> float:
    """Return a linearly interpolated percentile for NVML samples."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@ray.remote(num_cpus=0)
def sample_node(expected_node_id: str, duration_s: float = 30.0) -> dict:
    import pynvml

    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    pynvml.nvmlInit()
    try:
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        values = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            values.extend(
                int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                for handle in handles
            )
            time.sleep(1.0)
        return {
            "node_id": expected_node_id,
            "gpu_count": len(handles),
            "values": values,
        }
    finally:
        pynvml.nvmlShutdown()


@ray.remote(num_cpus=0)
def scan_output_node(expected_node_id: str, output_root: str) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    root = Path(output_root)
    md_stems = sorted(path.stem for path in root.rglob("*.md")) if root.is_dir() else []
    layout_stems = (
        sorted(path.parent.parent.name for path in root.rglob("layout.json"))
        if root.is_dir()
        else []
    )
    files = list(root.rglob("*")) if root.is_dir() else []
    artifacts = root.parent / "artifacts"
    summary = None
    summary_path = artifacts / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as error:
            summary = {"parse_error": f"{type(error).__name__}: {error}"}
    return {
        "node_id": expected_node_id,
        "root_exists": root.is_dir(),
        "files": sum(path.is_file() for path in files),
        "bytes": sum(path.stat().st_size for path in files if path.is_file()),
        "images": sum(path.is_file() for path in root.rglob("images/*")) if root.is_dir() else 0,
        "incomplete": sum(path.is_file() for path in root.rglob(".rayorch-incomplete")) if root.is_dir() else 0,
        "md_stems": md_stems,
        "layout_stems": layout_stems,
        "summary": summary,
    }


@ray.remote(num_cpus=0)
def probe_raylet_node(expected_node_id: str) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    candidates = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == "raylet":
                candidates.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if not candidates:
        raise RuntimeError("active raylet process was not found")
    pid = max(candidates)
    proc = Path("/proc") / str(pid)
    executable = os.readlink(proc / "exe")
    limits = (proc / "limits").read_text(encoding="utf-8")
    core_limit = next(
        (line for line in limits.splitlines() if line.startswith("Max core file size")),
        "missing",
    )
    status_text = (proc / "status").read_text(encoding="utf-8")
    status = {
        key: value.strip()
        for key, value in (
            line.split(":", 1)
            for line in status_text.splitlines()
            if ":" in line
        )
        if key in {"Uid", "Gid", "CoreDumping", "NoNewPrivs", "CapEff"}
    }
    notes = subprocess.run(
        ["readelf", "-n", executable],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    ).stdout
    build_match = re.search(r"Build ID:\s*(\S+)", notes)
    cgroup_lines = (proc / "cgroup").read_text(encoding="utf-8").splitlines()
    unified = next(
        (line.split(":", 2)[2] for line in cgroup_lines if line.startswith("0::")),
        "",
    )
    cgroup_root = Path("/sys/fs/cgroup") / unified.lstrip("/")
    memory = {}
    for name in ("memory.events", "memory.events.local", "memory.current", "memory.max"):
        path = cgroup_root / name
        if path.is_file():
            try:
                memory[name] = path.read_text(encoding="utf-8").strip()
            except PermissionError as error:
                memory[name] = f"PermissionError: {error}"
    log_files = []
    for path in Path("/tmp/ray/session_latest/logs").glob("raylet.*"):
        try:
            log_files.append({"path": str(path), "bytes": path.stat().st_size})
        except FileNotFoundError:
            pass
    dmesg = subprocess.run(
        ["dmesg", "--ctime", "--level=err,warn"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    disk = shutil.disk_usage("/tmp")
    return {
        "node_id": expected_node_id,
        "pid": pid,
        "exe": executable,
        "cwd": os.readlink(proc / "cwd"),
        "build_id": build_match.group(1) if build_match else None,
        "core_limit": core_limit,
        "core_pattern": Path("/proc/sys/kernel/core_pattern").read_text().strip(),
        "core_uses_pid": Path("/proc/sys/kernel/core_uses_pid").read_text().strip(),
        "coredump_filter": (proc / "coredump_filter").read_text().strip(),
        "status": status,
        "cgroup": unified,
        "memory": memory,
        "log_files": log_files,
        "gdb": shutil.which("gdb"),
        "readelf": shutil.which("readelf"),
        "prlimit": shutil.which("prlimit"),
        "dmesg_returncode": dmesg.returncode,
        "dmesg_tail": dmesg.stdout[-2000:],
        "tmp_free_bytes": disk.free,
    }


@ray.remote(num_cpus=0)
def audit_raylet_sig_node(expected_node_id: str) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    raylets = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == "raylet":
                raylets.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if not raylets:
        raise RuntimeError("active raylet process was not found")
    pid = max(raylets)
    patterns = (
        "sigsegv",
        "sigabrt",
        "sigbus",
        "sigill",
        "segmentation fault",
        "core dumped",
        "count::registerview",
        "metric::record",
        "viewdescriptor",
        "fatal python error",
        "signal 11",
        "signal 6",
    )
    candidates = []
    for root, glob_pattern in (
        (Path("/tmp/ray/session_latest/logs"), "raylet.*"),
        (Path("/data/taiji-serving/zhiyan-log"), "raylet*"),
    ):
        if root.is_dir():
            candidates.extend(root.glob(glob_pattern))
    files = []
    matches = []
    raylet_texts = []
    for path in sorted(set(candidates), key=str):
        try:
            size = path.stat().st_size
            with path.open("rb") as source:
                source.seek(max(0, size - 2_000_000))
                content = source.read().decode(errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        files.append({"path": str(path), "bytes": size})
        raylet_texts.append((str(path), content))
        matches.extend(
            {"path": str(path), "line": line[-2000:]}
            for line in content.splitlines()
            if any(pattern in line.lower() for pattern in patterns)
        )
    child_signals = []
    for path, content in raylet_texts:
        lines = content.splitlines()
        for index, line in enumerate(lines):
            matched = re.search(
                r"Child process (\d+) exited from signal 11", line
            )
            if not matched:
                continue
            child_pid = int(matched.group(1))
            pid_terms = (f"pid={child_pid}", f"pid {child_pid}", f" {child_pid} ")
            pid_context = [
                candidate[-2000:]
                for candidate in lines
                if candidate != line
                and any(term in candidate for term in pid_terms)
            ][-20:]
            matching_files = []
            for root in (
                Path("/data/taiji-serving/zhiyan-log"),
                Path("/tmp/ray/session_latest/logs"),
                Path("/dockerdata"),
            ):
                if not root.is_dir():
                    continue
                try:
                    matching_files.extend(
                        str(candidate)
                        for candidate in root.glob(f"*{child_pid}*")
                        if candidate.is_file()
                    )
                except PermissionError:
                    pass
            coredump = None
            if shutil.which("coredumpctl"):
                result = subprocess.run(
                    ["coredumpctl", "info", str(child_pid), "--no-pager"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
                coredump = {
                    "returncode": result.returncode,
                    "output": result.stdout[-4000:],
                }
            child_signals.append(
                {
                    "pid": child_pid,
                    "path": path,
                    "signal_line": line[-2000:],
                    "nearby_lines": lines[max(0, index - 5) : index + 6],
                    "pid_context": pid_context,
                    "matching_files": sorted(set(matching_files)),
                    "coredumpctl": coredump,
                }
            )
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    core_dumping = next(
        (
            line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith("CoreDumping:")
        ),
        "missing",
    )
    return {
        "node_id": expected_node_id,
        "raylet_pid": pid,
        "ray_version": getattr(ray, "__version__", None),
        "ray_commit": getattr(ray, "__commit__", None),
        "core_dumping": core_dumping,
        "log_files": files,
        "fatal_matches": matches[-200:],
        "signal11_children": child_signals[-10:],
    }


@ray.remote(num_cpus=0)
def inspect_child_signal_node(
    expected_node_id: str,
    child_pid: int,
) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    matching_files = []
    for root in (
        Path("/data/taiji-serving/zhiyan-log"),
        Path("/tmp/ray/session_latest/logs"),
        Path("/dockerdata"),
    ):
        if not root.is_dir():
            continue
        try:
            matching_files.extend(
                candidate
                for candidate in root.glob(f"*{child_pid}*")
                if candidate.is_file()
            )
        except PermissionError:
            pass
    file_tails = []
    for path in sorted(set(matching_files), key=str):
        try:
            size = path.stat().st_size
            with path.open("rb") as source:
                source.seek(max(0, size - 100_000))
                content = source.read().decode(errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        file_tails.append(
            {"path": str(path), "bytes": size, "tail": content[-20_000:]}
        )
    raylet_path = Path("/data/taiji-serving/zhiyan-log/raylet.out")
    raylet_context = []
    if raylet_path.is_file():
        text = raylet_path.read_text(errors="replace")
        lines = text.splitlines()
        target = f"Child process {child_pid} exited from signal 11"
        for index, line in enumerate(lines):
            if target in line:
                raylet_context.extend(lines[max(0, index - 30) : index + 31])
    dmesg = subprocess.run(
        ["dmesg", "--ctime"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    dmesg_matches = [
        line
        for line in dmesg.stdout.splitlines()
        if str(child_pid) in line or "segfault" in line.lower()
    ]
    return {
        "node_id": expected_node_id,
        "child_pid": child_pid,
        "file_tails": file_tails,
        "raylet_context": raylet_context[-100:],
        "dmesg_matches": dmesg_matches[-100:],
    }


def _arm_raylet_local(
    expected_node_id: str,
    diag_uri: str,
    upload_binary: bool,
    watcher_source_text: str,
) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    raylets = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == "raylet":
                raylets.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if not raylets:
        raise RuntimeError("active raylet process was not found")
    pid = max(raylets)
    proc = Path("/proc") / str(pid)
    executable = os.readlink(proc / "exe")
    result = subprocess.run(
        ["prlimit", "--pid", str(pid), "--core=unlimited:unlimited"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    limits = (proc / "limits").read_text(encoding="utf-8")
    core_limit = next(
        line for line in limits.splitlines() if line.startswith("Max core file size")
    )
    import hashlib
    from pyarrow import fs as pyarrow_fs

    digest = hashlib.sha256()
    with open(executable, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    filesystem, root_path = pyarrow_fs.FileSystem.from_uri(diag_uri)
    filesystem.create_dir(root_path, recursive=True)
    preflight = {
        "node_id": expected_node_id,
        "raylet_pid": pid,
        "raylet_exe": executable,
        "raylet_sha256": digest.hexdigest(),
        "core_limit": core_limit,
        "core_pattern": Path("/proc/sys/kernel/core_pattern").read_text().strip(),
        "prlimit_returncode": result.returncode,
        "prlimit_output": result.stdout,
        "dockerdata_exists": Path("/dockerdata").is_dir(),
        "dockerdata_writable": os.access("/dockerdata", os.W_OK),
    }
    preflight_path = root_path.rstrip("/") + "/preflight.json"
    with filesystem.open_output_stream(preflight_path) as sink:
        sink.write(json.dumps(preflight, sort_keys=True).encode())
    if upload_binary:
        with open(executable, "rb") as source, filesystem.open_output_stream(
            root_path.rstrip("/") + "/raylet"
        ) as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    local_dir = Path(f"/tmp/rayorch-raylet-diag-{expected_node_id}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_watcher = local_dir / "raylet_watcher.py"
    local_watcher.write_text(watcher_source_text, encoding="utf-8")
    watcher_log = (local_dir / "watcher.log").open("ab", buffering=0)
    watcher = subprocess.Popen(
        [
            sys.executable,
            str(local_watcher),
            "--pid",
            str(pid),
            "--node-id",
            expected_node_id,
            "--exe",
            executable,
            "--diag-uri",
            diag_uri,
        ],
        stdin=subprocess.DEVNULL,
        stdout=watcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return {**preflight, "watcher_pid": watcher.pid}


@ray.remote(num_cpus=0)
def arm_raylet_node(
    expected_node_id: str,
    diag_uri: str,
    upload_binary: bool,
    watcher_source_text: str,
) -> dict:
    return _arm_raylet_local(
        expected_node_id,
        diag_uri,
        upload_binary,
        watcher_source_text,
    )


@ray.remote(num_cpus=0, max_restarts=0)
class RayletWatcherKeeper:
    """Keep the worker that owns an OS-level raylet watcher alive."""

    def __init__(
        self,
        expected_node_id: str,
        diag_uri: str,
        upload_binary: bool,
        watcher_source_text: str,
    ) -> None:
        self.result = _arm_raylet_local(
            expected_node_id,
            diag_uri,
            upload_binary,
            watcher_source_text,
        )

    def status(self) -> dict:
        watcher_pid = int(self.result["watcher_pid"])
        return {
            **self.result,
            "keeper_pid": os.getpid(),
            "watcher_alive": Path(f"/proc/{watcher_pid}").exists(),
        }


@ray.remote(num_cpus=0)
def verify_raylet_arm_node(expected_node_id: str) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    raylets = []
    watchers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm == "raylet":
            raylets.append(int(entry.name))
        if "raylet_watcher.py" in cmdline:
            watchers.append(int(entry.name))
    if not raylets:
        raise RuntimeError("active raylet process was not found")
    pid = max(raylets)
    limits = Path(f"/proc/{pid}/limits").read_text()
    core_limit = next(
        line for line in limits.splitlines() if line.startswith("Max core file size")
    )
    cmdline = (
        Path(f"/proc/{pid}/cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode()
    )
    watcher_log = Path(f"/tmp/rayorch-raylet-diag-{expected_node_id}/watcher.log")
    return {
        "node_id": expected_node_id,
        "raylet_pid": pid,
        "ray_version": getattr(ray, "__version__", None),
        "ray_commit": getattr(ray, "__commit__", None),
        "core_limit": core_limit,
        "raylet_cmdline": cmdline,
        "watcher_pids": watchers,
        "watcher_log": (
            watcher_log.read_text(errors="replace")[-2000:]
            if watcher_log.is_file()
            else None
        ),
    }


@ray.remote(num_cpus=0)
def collect_raylet_crash_node(
    expected_node_id: str,
    diag_uri: str,
    crash_epoch: int,
) -> dict:
    actual = ray.get_runtime_context().get_node_id()
    if actual != expected_node_id:
        raise RuntimeError(f"node affinity mismatch: {actual} != {expected_node_id}")
    node_ip = ray.util.get_node_ip_address()
    local_root = Path(f"/tmp/rayorch-raylet-crash-{expected_node_id}")
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)
    executable = (
        "/home/ray/anaconda3/lib/python3.12/site-packages/ray/core/src/"
        "ray/raylet/raylet"
    )
    core_candidates = []
    for root in (Path("/dockerdata"), Path("/home/ray")):
        if not root.is_dir():
            continue
        for path in root.glob("core-raylet-*"):
            try:
                if str(crash_epoch) in path.name or abs(path.stat().st_mtime - crash_epoch) <= 300:
                    core_candidates.append(path)
            except FileNotFoundError:
                pass
    selected_core = (
        max(core_candidates, key=lambda path: path.stat().st_mtime)
        if core_candidates
        else None
    )
    copied_core = None
    if selected_core is not None:
        copied_core = local_root / selected_core.name
        shutil.copy2(selected_core, copied_core)
    logs = []
    log_root = Path("/data/taiji-serving/zhiyan-log")
    if log_root.is_dir():
        logs = sorted(
            (
                path
                for path in log_root.glob("raylet*")
                if path.is_file()
                and abs(path.stat().st_mtime - crash_epoch) <= 7200
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:20]
    for path in logs:
        shutil.copy2(path, local_root / path.name)
    dmesg = subprocess.run(
        ["dmesg", "--ctime"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (local_root / "dmesg.txt").write_text(dmesg.stdout[-4_000_000:])
    notes = subprocess.run(
        ["readelf", "-n", executable],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    ).stdout
    build_match = re.search(r"Build ID:\s*(\S+)", notes)
    gdb_returncode = None
    if copied_core is not None:
        gdb_result = subprocess.run(
            [
                "gdb",
                "-batch",
                "-ex",
                "set pagination off",
                "-ex",
                "thread apply all bt full",
                executable,
                str(copied_core),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=False,
        )
        gdb_returncode = gdb_result.returncode
        (local_root / "gdb-thread-apply-all-bt-full.txt").write_text(
            gdb_result.stdout
        )
    manifest = {
        "node_id": expected_node_id,
        "node_ip": node_ip,
        "crash_epoch": crash_epoch,
        "build_id": build_match.group(1) if build_match else None,
        "core_source": str(selected_core) if selected_core else None,
        "core_bytes": copied_core.stat().st_size if copied_core else 0,
        "gdb_returncode": gdb_returncode,
        "log_files": [path.name for path in logs],
    }
    (local_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    from pyarrow import fs as pyarrow_fs

    filesystem, root_path = pyarrow_fs.FileSystem.from_uri(diag_uri)
    filesystem.create_dir(root_path, recursive=True)
    for path in local_root.iterdir():
        if not path.is_file():
            continue
        target = root_path.rstrip("/") + "/" + path.name
        with path.open("rb") as source, filesystem.open_output_stream(target) as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--expected-gpu-nodes", type=int, default=8)
    args = parser.parse_args()
    ray.init(address="auto")
    gpu_nodes = []
    for record in ray.nodes():
        resources = record.get("Resources") or {}
        if record.get("Alive") and float(resources.get("GPU", 0)) > 0:
            gpu_nodes.append(str(record["NodeID"]))
    if len(gpu_nodes) != args.expected_gpu_nodes:
        raise RuntimeError(
            f"expected {args.expected_gpu_nodes} GPU nodes, got {len(gpu_nodes)}"
        )
    refs = [
        sample_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            )
        ).remote(node_id, args.duration_s)
        for node_id in sorted(gpu_nodes)
    ]
    records = ray.get(refs)
    values = [value for record in records for value in record["values"]]
    result = {
        "duration_s": args.duration_s,
        "gpu_count": sum(record["gpu_count"] for record in records),
        "sample_count": len(values),
        "global_mean_gpu_utilization": round(sum(values) / len(values), 3),
        "global_p50_gpu_utilization": round(_percentile(values, 0.50), 3),
        "global_p90_gpu_utilization": round(_percentile(values, 0.90), 3),
        "node_mean_gpu_utilization": {
            record["node_id"]: round(
                sum(record["values"]) / len(record["values"]), 3
            )
            for record in records
        },
        "fraction_samples_at_least_90": round(
            sum(value >= 90 for value in values) / len(values), 6
        ),
    }
    print("RAYORCH_GPU_MONITOR_RESULT " + json.dumps(result, sort_keys=True))
    if args.output_root:
        schedulable_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive") and record.get("Resources")
        )
        scans = ray.get(
            [
                scan_output_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(node_id, args.output_root)
                for node_id in schedulable_nodes
            ]
        )
        md_stems = {stem for scan in scans for stem in scan["md_stems"]}
        layout_stems = {
            stem for scan in scans for stem in scan["layout_stems"]
        }
        summaries = [scan["summary"] for scan in scans if scan["summary"]]
        output_result = {
            "nodes": [
                {key: value for key, value in scan.items() if not key.endswith("_stems") and key != "summary"}
                for scan in scans
            ],
            "unique_md": len(md_stems),
            "unique_layout": len(layout_stems),
            "missing_md": sorted(layout_stems - md_stems)[:20],
            "missing_layout": sorted(md_stems - layout_stems)[:20],
            "incomplete": sum(scan["incomplete"] for scan in scans),
            "summaries": summaries,
        }
        print(
            "RAYORCH_OUTPUT_PROBE_RESULT "
            + json.dumps(output_result, sort_keys=True)
        )
    if args.hdfs_output_uri:
        from pyarrow import fs as pyarrow_fs

        filesystem, root_path = pyarrow_fs.FileSystem.from_uri(
            args.hdfs_output_uri
        )
        infos = filesystem.get_file_info(
            pyarrow_fs.FileSelector(root_path, recursive=True)
        )
        files = [
            info
            for info in infos
            if info.type == pyarrow_fs.FileType.File
        ]
        relative_paths = {
            info.path: PurePosixPath(posixpath.relpath(info.path, root_path))
            for info in files
        }
        md_stems = {
            path.parts[1]
            for path in relative_paths.values()
            if len(path.parts) == 4
            and path.parts[0] == "docs"
            and path.parts[2] == "vlm"
            and path.suffix.lower() == ".md"
        }
        layout_stems = {
            path.parts[1]
            for path in relative_paths.values()
            if len(path.parts) == 4
            and path.parts[0] == "docs"
            and path.parts[2:] == ("vlm", "layout.json")
        }
        document_success = {
            path.parts[1]
            for path in relative_paths.values()
            if len(path.parts) == 3
            and path.parts[0] == "docs"
            and path.parts[2] == "_SUCCESS"
        }
        hdfs_result = {
            "files": len(files),
            "bytes": sum(int(info.size) for info in files),
            "unique_md": len(md_stems),
            "unique_layout": len(layout_stems),
            "missing_md": sorted(layout_stems - md_stems)[:20],
            "missing_layout": sorted(md_stems - layout_stems)[:20],
            "document_success": len(document_success),
            "staging_files": sum(
                path.parts[0] == "_staging"
                for path in relative_paths.values()
            ),
            "success": any(
                info.path == posixpath.join(root_path, "_SUCCESS")
                for info in files
            ),
        }
        print(
            "RAYORCH_HDFS_OUTPUT_PROBE_RESULT "
            + json.dumps(hdfs_result, sort_keys=True)
        )
        if args.read_hdfs_diag:
            manifests = []
            gdb_excerpt = ""
            dmesg_matches = []
            raylet_matches = []
            for info in files:
                basename = Path(info.path).name
                if basename == "manifest.json" or basename == "preflight.json":
                    with filesystem.open_input_file(info.path) as source:
                        try:
                            manifests.append(json.loads(source.read().decode()))
                        except Exception as error:
                            manifests.append(
                                {
                                    "path": info.path,
                                    "parse_error": f"{type(error).__name__}: {error}",
                                }
                            )
                elif basename == "gdb-thread-apply-all-bt-full.txt":
                    with filesystem.open_input_file(info.path) as source:
                        content = source.read().decode(errors="replace")
                    lines = content.splitlines()
                    focus_indexes = [
                        index
                        for index, line in enumerate(lines)
                        if any(
                            term in line
                            for term in (
                                "Count::RegisterView",
                                "Metric::Record",
                                "ClientCallImpl",
                                "std::string",
                                "Program terminated with signal SIGSEGV",
                            )
                        )
                    ]
                    selected = set(range(min(50, len(lines))))
                    for index in focus_indexes:
                        selected.update(
                            range(max(0, index - 15), min(len(lines), index + 25))
                        )
                    gdb_excerpt = "\n".join(lines[index] for index in sorted(selected))
                elif basename == "dmesg.txt":
                    with filesystem.open_input_file(info.path) as source:
                        content = source.read_at(
                            min(int(info.size), 500_000),
                            max(0, int(info.size) - 500_000),
                        ).decode(errors="replace")
                    dmesg_matches.extend(
                        line
                        for line in content.splitlines()
                        if any(
                            term in line.lower()
                            for term in (
                                "raylet",
                                "11:37:",
                                "segfault",
                            )
                        )
                    )
                elif basename.startswith("raylet") and (
                    basename.endswith(".out") or basename.endswith(".err")
                ):
                    with filesystem.open_input_file(info.path) as source:
                        content = source.read_at(
                            min(int(info.size), 250_000),
                            max(0, int(info.size) - 250_000),
                        ).decode(errors="replace")
                    raylet_matches.extend(
                        line
                        for line in content.splitlines()
                        if any(
                            term in line.lower()
                            for term in (
                                "sigsegv",
                                "metric::record",
                                "registerview",
                                "oom",
                                "killed process",
                            )
                        )
                    )
            diag_result = {
                "files": [
                    {"path": info.path, "bytes": int(info.size)} for info in files
                ],
                "manifests": manifests,
                "gdb_excerpt": gdb_excerpt,
                "dmesg_matches": dmesg_matches[-200:],
                "raylet_matches": raylet_matches[-300:],
            }
            print(
                "RAYORCH_HDFS_DIAG_READ_RESULT "
                + json.dumps(diag_result, sort_keys=True)
            )
    if args.raylet_diag:
        schedulable_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive") and record.get("Resources")
        )
        diagnostics = ray.get(
            [
                probe_raylet_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(node_id)
                for node_id in schedulable_nodes
            ]
        )
        print(
            "RAYORCH_RAYLET_DIAG_RESULT "
            + json.dumps(diagnostics, sort_keys=True)
        )
    if args.raylet_sig_audit:
        schedulable_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive") and record.get("Resources")
        )
        audits = ray.get(
            [
                audit_raylet_sig_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(node_id)
                for node_id in schedulable_nodes
            ]
        )
        print(
            "RAYORCH_RAYLET_SIG_AUDIT "
            + json.dumps(audits, sort_keys=True)
        )
    if args.inspect_child_pid is not None:
        if not args.inspect_node_id:
            raise ValueError("--inspect-node-id is required with --inspect-child-pid")
        inspection = ray.get(
            inspect_child_signal_node.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=args.inspect_node_id, soft=False
                )
            ).remote(args.inspect_node_id, args.inspect_child_pid)
        )
        print(
            "RAYORCH_CHILD_SIGNAL_INSPECTION "
            + json.dumps(inspection, sort_keys=True)
        )
    if args.arm_raylet_diag:
        if not args.diag_hdfs_uri:
            raise ValueError("--diag-hdfs-uri is required with --arm-raylet-diag")
        gpu_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive")
            and float((record.get("Resources") or {}).get("GPU", 0)) > 0
        )
        import hashlib

        diag_key = hashlib.sha256(
            args.diag_hdfs_uri.encode("utf-8")
        ).hexdigest()[:12]
        watcher_source_text = Path(__file__).with_name(
            "raylet_watcher.py"
        ).read_text(encoding="utf-8")
        keepers = [
            RayletWatcherKeeper.options(
                name=f"raylet-watcher-{diag_key}-{index}",
                namespace="rayorch-raylet-diag",
                lifetime="detached",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            ).remote(
                node_id,
                args.diag_hdfs_uri.rstrip("/") + "/" + node_id,
                index == 0,
                watcher_source_text,
            )
            for index, node_id in enumerate(gpu_nodes)
        ]
        armed = ray.get([keeper.status.remote() for keeper in keepers])
        print(
            "RAYORCH_RAYLET_DIAG_ARMED "
            + json.dumps(armed, sort_keys=True)
        )
    if args.verify_raylet_arm:
        gpu_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive")
            and float((record.get("Resources") or {}).get("GPU", 0)) > 0
        )
        verification = ray.get(
            [
                verify_raylet_arm_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(node_id)
                for node_id in gpu_nodes
            ]
        )
        keeper_status = []
        if args.diag_hdfs_uri:
            import hashlib

            diag_key = hashlib.sha256(
                args.diag_hdfs_uri.encode("utf-8")
            ).hexdigest()[:12]
            keepers = [
                ray.get_actor(
                    f"raylet-watcher-{diag_key}-{index}",
                    namespace="rayorch-raylet-diag",
                )
                for index in range(len(gpu_nodes))
            ]
            keeper_status = ray.get(
                [keeper.status.remote() for keeper in keepers]
            )
        print(
            "RAYORCH_RAYLET_DIAG_VERIFIED "
            + json.dumps(
                {"nodes": verification, "keepers": keeper_status},
                sort_keys=True,
            )
        )
    if args.collect_raylet_crash:
        if not args.diag_hdfs_uri or args.crash_epoch is None:
            raise ValueError(
                "--diag-hdfs-uri and --crash-epoch are required for collection"
            )
        gpu_nodes = sorted(
            str(record["NodeID"])
            for record in ray.nodes()
            if record.get("Alive")
            and float((record.get("Resources") or {}).get("GPU", 0)) > 0
        )
        collection = ray.get(
            [
                collect_raylet_crash_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(
                    node_id,
                    args.diag_hdfs_uri.rstrip("/") + "/" + node_id + "/crash",
                    args.crash_epoch,
                )
                for node_id in gpu_nodes
            ]
        )
        print(
            "RAYORCH_RAYLET_CRASH_COLLECTED "
            + json.dumps(collection, sort_keys=True)
        )


if __name__ == "__main__":
    main()
