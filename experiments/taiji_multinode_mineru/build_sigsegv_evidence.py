"""Build a sanitized SIGSEGV evidence bundle from saved audit state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SENSITIVE_KEYS = {
    "authorization",
    "cmk",
    "credential",
    "password",
    "runtime_env",
    "token",
}


SUCCESSFUL_STATES = {
    "1x4": "rayorch_v36_multigrain_scaling_1x4_20260825_1910.json",
    "1x8": "rayorch_v36_multigrain_scaling_1x8_20260825_1910_retry1.json",
    "2x8": "rayorch_v36_multigrain_scaling_2x8_20260825_1910.json",
    "4x8": "rayorch_v36_multigrain_scaling_4x8_20260825_1910_retry2.json",
    "8x8": "rayorch_v36_multigrain_scaling_8x8_20260825_1910.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _signal_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    postflight = state.get("postflight_audit") or {}
    markers = postflight.get("markers") or {}
    nodes = markers.get("RAYORCH_RAYLET_SIG_AUDIT") or []
    evidence: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for child in node.get("signal11_children") or []:
            if not isinstance(child, dict):
                continue
            evidence.append(
                {
                    "node_id": node.get("node_id"),
                    "ray_version": node.get("ray_version"),
                    "ray_commit": node.get("ray_commit"),
                    "raylet_pid": node.get("raylet_pid"),
                    "worker_pid": child.get("pid"),
                    "signal_line": child.get("signal_line"),
                    "raylet_log_path": child.get("path"),
                    "matching_worker_logs": child.get("matching_files") or [],
                    "pid_context": child.get("pid_context") or [],
                    "nearby_lines": child.get("nearby_lines") or [],
                    "coredumpctl": child.get("coredumpctl"),
                }
            )
    return evidence


def _driver_excerpt(path: Path, patterns: tuple[str, ...], context: int = 3) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    chosen: set[int] = set()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(pattern in lowered for pattern in patterns):
            chosen.update(range(max(0, index - context), min(len(lines), index + context + 1)))
    chunks: list[str] = []
    previous = -2
    for index in sorted(chosen):
        if index != previous + 1 and chunks:
            chunks.append("---")
        chunks.append(f"{index + 1}: {ANSI_ESCAPE.sub('', lines[index])}")
        previous = index
    return "\n".join(chunks) + ("\n" if chunks else "")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_sanitized(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        lowered = path.read_text(encoding="utf-8", errors="replace").lower()
        for key in SENSITIVE_KEYS:
            if re.search(rf'(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])', lowered):
                raise RuntimeError(f"sensitive key {key!r} found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "description": "Sanitized Ray exit-time signal 11 evidence",
        "successful_runs": {},
    }
    rows: list[dict[str, Any]] = []

    for topology, filename in SUCCESSFUL_STATES.items():
        state = _read_json(args.state_dir / filename)
        evidence = _signal_evidence(state)
        postflight = state.get("postflight_audit") or {}
        driver_audit = state.get("driver_exit_crash_audit") or {}
        record = {
            "topology": topology,
            "source_state_file": filename,
            "platform_status": state.get("full_status"),
            "business_status": (state.get("full_result") or {}).get("status"),
            "terminal_audited_at": state.get("terminal_audited_at"),
            "raylet_signal11_child_count": state.get("raylet_signal11_child_count"),
            "postflight_returncode": postflight.get("returncode"),
            "driver_crash_keyword_match_count": driver_audit.get("match_count"),
            "signal11_evidence": evidence,
        }
        aggregate["successful_runs"][topology] = record
        _write_json(args.output_dir / f"{topology}_signal11_audit.json", record)
        rows.append(record)

    retry_log = args.state_dir / (
        "rayorch_v36_multigrain_scaling_4x8_20260825_1910_retry1_full_driver.log"
    )
    excerpt = _driver_excerpt(
        retry_log,
        patterns=("ownerdiederror", "infra_failure", "rayworkeractor.execute"),
    )
    (args.output_dir / "4x8_retry1_ownerdied_excerpt.log").write_text(
        "source: " + retry_log.name + "\n"
        "note: exact sanitized excerpt from the saved Ray driver log\n"
        + excerpt,
        encoding="utf-8",
    )

    _write_json(args.output_dir / "successful_exit_signal11.json", aggregate)

    readme_lines = [
        "# RayOrch SIGSEGV evidence",
        "",
        "This directory contains sanitized excerpts captured by the postflight raylet audit.",
        "All four positive cases occurred after the business result had already completed successfully.",
        "",
        "| Topology | Platform/business | Signal-11 children | Exact captured line |",
        "|---|---|---:|---|",
    ]
    for record in rows:
        exact_lines = [
            str(item.get("signal_line") or "")
            for item in record["signal11_evidence"]
        ]
        exact = "<br>".join(line.replace("|", "\\|") for line in exact_lines) or (
            "No captured line; see audit-gap note below"
        )
        readme_lines.append(
            f"| {record['topology']} | {record['platform_status']}/{record['business_status']} "
            f"| {record['raylet_signal11_child_count']} | `{exact}` |"
        )
    readme_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 1x4, 1x8, 2x8, and 4x8 each captured one child exiting from signal 11.",
            "- The evidence-bearing worker context identifies `JobSupervisor.__init__` in the successful exits.",
            "- 8x8 did not capture a signal child, but its postflight audit returned non-zero, so this is inconclusive rather than proof of a clean exit.",
            "- `4x8_retry1_ownerdied_excerpt.log` records the separate failed retry's Ray `OwnerDiedError`.",
            "- The JSON files preserve the exact signal line, nearby raylet lines, worker PID context, node ID, Ray version/commit, and matching worker log names.",
            "- `native_raylet_sigsegv_reference/` contains the earlier full saved log with the native `Count::RegisterView -> Metric::Record` raylet stack. It is a different run and crash process from the five scaling runs.",
            "",
            "## Source files",
            "",
        ]
    )
    readme_lines.extend(f"- `{filename}`" for filename in SUCCESSFUL_STATES.values())
    readme_lines.append(
        "- `rayorch_v36_multigrain_scaling_4x8_20260825_1910_retry1_full_driver.log`"
    )
    readme_lines.append("")
    (args.output_dir / "README.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )

    _assert_sanitized(args.output_dir)
    manifest_lines: list[str] = []
    for path in sorted(args.output_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.name}")
    (args.output_dir / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "files": sorted(path.name for path in args.output_dir.iterdir()),
                "positive_signal11_runs": sum(
                    bool(row["signal11_evidence"]) for row in rows
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
