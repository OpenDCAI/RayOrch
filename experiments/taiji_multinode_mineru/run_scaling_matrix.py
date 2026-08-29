"""Plan or concurrently submit the five Zhongwei H20 scaling experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import orchestrate


OUTPUT_ROOT = (
    "/apdcephfs_zwfy10/share_304380933/hunyuan/clapliang/rayorch_test"
)
TOPOLOGIES = (
    ("1x4", 1, 4, orchestrate.ZW_ONE_BY_FOUR_PROFILE_REF),
    ("1x8", 1, 8, orchestrate.ZW_ONE_BY_EIGHT_PROFILE_REF),
    ("2x8", 2, 8, orchestrate.ZW_TWO_BY_EIGHT_PROFILE_REF),
    ("4x8", 4, 8, orchestrate.ZW_FOUR_BY_EIGHT_PROFILE_REF),
    ("8x8", 8, 8, orchestrate.ZW_PROFILE_REF),
)


def _revision() -> dict[str, str]:
    def read(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=orchestrate.REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "branch": read("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": read("rev-parse", "HEAD"),
    }


def _hyperparameters(workers: int, gpus_per_worker: int) -> dict[str, Any]:
    total_gpus = workers * gpus_per_worker
    result = dict(orchestrate.HYPERPARAMETERS)
    result.update(
        {
            "ocr_replicas": total_gpus * 2,
            "gpus_per_ocr_actor": 0.5,
            "render_replicas": total_gpus * 4,
            "reduce_replicas": total_gpus,
        }
    )
    return result


def _matrix(matrix_id: str) -> dict[str, Any]:
    experiments = []
    for label, workers, gpus_per_worker, profile in TOPOLOGIES:
        run_id = f"rayorch_v36_multigrain_scaling_{label}_{matrix_id}"
        output_uri = f"{OUTPUT_ROOT}/{run_id}"
        parameters = _hyperparameters(workers, gpus_per_worker)
        plan = orchestrate.build_plan(
            run_id=run_id,
            output_uri=output_uri,
            hyperparameters=parameters,
            profile_reference=profile,
        )
        state_file = orchestrate._state_default(run_id)
        experiments.append(
            {
                "label": label,
                "workers": workers,
                "gpus_per_worker": gpus_per_worker,
                "total_gpus": workers * gpus_per_worker,
                "profile": profile,
                "run_id": run_id,
                "output_uri": output_uri,
                "output_kind": plan["output_kind"],
                "state_file": str(state_file),
                "submit_log": str(
                    state_file.with_name(state_file.stem + "_submit.log")
                ),
                "hyperparameters": parameters,
                "plan_source_digest": plan["source_digest"],
                "plan_spec_fingerprint": plan["compute"]["spec_fingerprint"],
            }
        )
    return {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "created_at": orchestrate._utc_now(),
        "source_revision": _revision(),
        "input_uris": list(orchestrate.DEFAULT_INPUT_URIS),
        "expected_pdfs": orchestrate.EXPECTED_PDFS,
        "output_root": OUTPUT_ROOT,
        "application_group": orchestrate.APP_GROUP,
        "location": "zw",
        "gpu_type": orchestrate.GPU_TYPE,
        "requested_total_gpus": sum(item[1] * item[2] for item in TOPOLOGIES),
        "experiments": experiments,
    }


def _submit(summary_path: Path, summary: dict[str, Any]) -> None:
    processes: list[subprocess.Popen[str]] = []
    streams = []
    for experiment in summary["experiments"]:
        state_file = Path(experiment["state_file"])
        log_path = Path(experiment["submit_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("a", encoding="utf-8")
        streams.append(stream)
        command = [
            sys.executable,
            str(orchestrate.EXPERIMENT_DIR / "orchestrate.py"),
            "submit",
            "--run-id",
            experiment["run_id"],
            "--output-uri",
            experiment["output_uri"],
            "--profile",
            experiment["profile"],
            "--state-file",
            str(state_file),
            "--hyperparameters-json",
            json.dumps(experiment["hyperparameters"], separators=(",", ":")),
        ]
        process = subprocess.Popen(
            command,
            cwd=orchestrate.REPOSITORY,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        processes.append(process)
        experiment.update(
            {
                "submit_pid": process.pid,
                "submit_started_at": orchestrate._utc_now(),
                "submit_status": "started",
            }
        )
        orchestrate._atomic_json(summary_path, summary)
        print(
            "SCALING_SUBMIT_STARTED "
            + json.dumps(
                {
                    "label": experiment["label"],
                    "pid": process.pid,
                    "state_file": experiment["state_file"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    time.sleep(3)
    for experiment, process in zip(summary["experiments"], processes, strict=True):
        returncode = process.poll()
        if returncode is not None:
            experiment.update(
                {
                    "submit_status": "exited_early",
                    "submit_returncode": returncode,
                    "submit_finished_at": orchestrate._utc_now(),
                }
            )
    summary["all_submitters_started_at"] = orchestrate._utc_now()
    orchestrate._atomic_json(summary_path, summary)
    for stream in streams:
        stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "submit"))
    parser.add_argument("--matrix-id")
    parser.add_argument("--summary-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_id = args.matrix_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not orchestrate._safe_run_id(matrix_id):
        raise ValueError("invalid matrix id")
    summary = _matrix(matrix_id)
    summary_path = args.summary_file or (
        orchestrate.EXPERIMENT_DIR / "state" / f"scaling_matrix_{matrix_id}.json"
    )
    if summary_path.exists():
        raise FileExistsError(f"matrix summary already exists: {summary_path}")
    summary["summary_file"] = str(summary_path)
    summary["external_actions"] = args.command == "submit"
    orchestrate._atomic_json(summary_path, summary)
    if args.command == "submit":
        _submit(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
