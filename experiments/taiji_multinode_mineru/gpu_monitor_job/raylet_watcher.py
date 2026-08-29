"""Detached node-local raylet crash collector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def read_cgroup_memory(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    except FileNotFoundError:
        lines = []
    unified = next(
        (line.split(":", 2)[2] for line in lines if line.startswith("0::")),
        None,
    )
    roots = []
    if unified is not None:
        roots.append(Path("/sys/fs/cgroup") / unified.lstrip("/"))
    memory_path = next(
        (
            line.split(":", 2)[2]
            for line in lines
            if "memory" in line.split(":", 2)[1].split(",")
        ),
        None,
    )
    if memory_path is not None:
        roots.append(
            Path("/sys/fs/cgroup/memory") / memory_path.lstrip("/")
        )
    names = (
        "memory.events",
        "memory.events.local",
        "memory.current",
        "memory.max",
        "memory.oom_control",
        "memory.failcnt",
        "memory.limit_in_bytes",
        "memory.usage_in_bytes",
        "memory.max_usage_in_bytes",
    )
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                try:
                    result[str(path)] = path.read_text().strip()
                except PermissionError as error:
                    result[str(path)] = f"PermissionError: {error}"
    return result


def upload_tree(local_root: Path, uri: str) -> None:
    from pyarrow import fs as pyarrow_fs

    filesystem, root_path = pyarrow_fs.FileSystem.from_uri(uri)
    filesystem.create_dir(root_path, recursive=True)
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        target = root_path.rstrip("/") + "/" + path.relative_to(local_root).as_posix()
        filesystem.create_dir(str(Path(target).parent), recursive=True)
        with path.open("rb") as source, filesystem.open_output_stream(target) as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--diag-uri", required=True)
    args = parser.parse_args()
    local_root = Path(f"/tmp/rayorch-raylet-diag-{args.node_id}")
    local_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while Path(f"/proc/{args.pid}").exists():
        time.sleep(1)
    crashed = time.time()
    time.sleep(3)
    manifest = {
        "node_id": args.node_id,
        "raylet_pid": args.pid,
        "watcher_pid": os.getpid(),
        "started_at": started,
        "raylet_disappeared_at": crashed,
        "core_pattern": Path("/proc/sys/kernel/core_pattern").read_text().strip(),
        "cgroup_memory_after": read_cgroup_memory(args.pid),
    }
    (local_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    dmesg = subprocess.run(
        ["dmesg", "--ctime"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (local_root / "dmesg.txt").write_text(dmesg.stdout, encoding="utf-8")
    log_candidates = sorted(
        Path("/tmp/ray").glob("session_*/logs/raylet.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in log_candidates[:6]:
        shutil.copy2(path, local_root / path.name)
    cores = []
    for pattern in (
        f"/dockerdata/core-raylet-{args.pid}-*",
        f"/dockerdata/core-*{args.pid}*",
        f"/home/ray/core*{args.pid}*",
    ):
        cores.extend(Path("/").glob(pattern.lstrip("/")))
    if not cores and shutil.which("coredumpctl"):
        core = local_root / "raylet.core"
        result = subprocess.run(
            ["coredumpctl", "dump", str(args.pid), "--output", str(core)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        (local_root / "coredumpctl.txt").write_text(
            result.stdout, encoding="utf-8"
        )
        if core.is_file() and core.stat().st_size:
            cores.append(core)
    if cores:
        core = max(cores, key=lambda path: path.stat().st_mtime)
        if core.parent != local_root:
            copied_core = local_root / core.name
            shutil.copy2(core, copied_core)
            core = copied_core
        if shutil.which("gdb"):
            result = subprocess.run(
                [
                    "gdb",
                    "-batch",
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "thread apply all bt full",
                    args.exe,
                    str(core),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
            (local_root / "gdb-thread-apply-all-bt-full.txt").write_text(
                result.stdout, encoding="utf-8"
            )
    upload_tree(local_root, args.diag_uri.rstrip("/") + "/post-crash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
