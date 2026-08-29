"""Configure, validate, and launch the frozen 64-GPU MinerU experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
CONFIG_DIR = HERE / "configs"
DEFAULT_PROFILE = HERE / "profile.local.toml"
DEFAULT_STATE_DIR = HERE / "state"
DEFAULT_OUTPUT_ROOT = Path(
    "/apdcephfs_zwfy10/share_304380933/hunyuan/clapliang/rayorch_test"
)
PROFILE_NAME = "ray-h20-8x8-zw-pretrain"
COMPUTE_SPEC = (
    REPOSITORY
    / "experiments"
    / "taiji_multinode_mineru"
    / "compute_h20_8node_8gpu_zw_pretrain.json"
)
EXPECTED_HYPERPARAMETERS = {
    "batch_size",
    "gpu_memory_utilization",
    "gpus_per_ocr_actor",
    "max_active_microbatches",
    "microbatch_size",
    "ocr_replicas",
    "reduce_replicas",
    "render_replicas",
    "spool_drain_timeout_s",
    "spool_upload_workers",
}
FROZEN_HYPERPARAMETERS = {
    "microbatch_size": 24,
    "max_active_microbatches": 24,
    "batch_size": 64,
    "gpu_memory_utilization": 0.32,
    "ocr_replicas": 128,
    "gpus_per_ocr_actor": 0.5,
    "render_replicas": 256,
    "reduce_replicas": 64,
    "spool_upload_workers": 8,
    "spool_drain_timeout_s": 21_600,
}


class ReproductionError(RuntimeError):
    """The frozen reproduction contract is invalid or incomplete."""


def _config_path(engine: str) -> Path:
    return CONFIG_DIR / f"{engine}_64gpu.json"


def load_config(engine: str) -> dict[str, Any]:
    path = _config_path(engine)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproductionError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ReproductionError(f"{path} must contain one JSON object")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    engine = config.get("engine")
    if engine not in {"rayorch", "raydata"}:
        raise ReproductionError("engine must be rayorch or raydata")
    compute = config.get("compute")
    hyperparameters = config.get("launcher_hyperparameters")
    workload = config.get("workload")
    dependencies = config.get("dependencies")
    if not all(
        isinstance(value, Mapping)
        for value in (compute, hyperparameters, workload, dependencies)
    ):
        raise ReproductionError("compute/hyperparameters/workload/dependencies missing")
    if (
        compute.get("workers"),
        compute.get("gpu_per_worker"),
        compute.get("gpu_total"),
        compute.get("gpu_type"),
        compute.get("location"),
        compute.get("taiji_app_group"),
    ) != (8, 8, 64, "H20", "zw", "TaiJi_HYAide_LLM_Pretrain_Data"):
        raise ReproductionError("compute contract is not the frozen 8x8 H20 layout")
    if set(hyperparameters) != EXPECTED_HYPERPARAMETERS:
        raise ReproductionError("launcher hyperparameter keys changed")
    if dict(hyperparameters) != FROZEN_HYPERPARAMETERS:
        raise ReproductionError("frozen launcher hyperparameters changed")
    reserved_gpus = (
        float(hyperparameters["ocr_replicas"])
        * float(hyperparameters["gpus_per_ocr_actor"])
    )
    if reserved_gpus != float(compute["gpu_total"]):
        raise ReproductionError("OCR actor GPU reservations do not total 64")
    inputs = workload.get("inputs")
    if (
        workload.get("pdf_count") != 3690
        or not isinstance(inputs, list)
        or sum(int(item.get("files", 0)) for item in inputs) != 3690
    ):
        raise ReproductionError("PDF manifest contract changed")
    commit = str(dependencies.get("flash_mineru_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReproductionError("Flash-MinerU commit must be a full SHA")
    compute_reference = str(compute.get("compute_spec") or "")
    compute_path = (REPOSITORY / compute_reference).resolve(strict=False)
    expected_compute_path = COMPUTE_SPEC.resolve(strict=False)
    if compute_path != expected_compute_path or not compute_path.is_file():
        raise ReproductionError("compute spec does not resolve to the frozen file")
    try:
        compute_spec = json.loads(compute_path.read_text(encoding="utf-8"))
        runtime_spec = compute_spec["runtime_spec"]
        executor = compute_spec["resource"]["executor_group"][0]
        taiji = executor["extra"]["gpu_provider"]["taiji"]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ReproductionError(f"invalid frozen compute spec: {error}") from error
    if (
        runtime_spec.get("image") != dependencies.get("taiji_image")
        or runtime_spec.get("taiji_app_group") != compute.get("taiji_app_group")
        or runtime_spec.get("taiji_app_group_location") != compute.get("location")
        or executor.get("replicas") != compute.get("workers")
        or executor.get("num_gpu") != compute.get("gpu_per_worker")
        or taiji.get("gpu_type") != compute.get("gpu_type")
        or taiji.get("app_group_name") != compute.get("taiji_app_group")
    ):
        raise ReproductionError("compute JSON and reproduction config diverged")


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_flash_repo(path: Path, config: Mapping[str, Any]) -> None:
    source = path / "flash_mineru"
    if not source.is_dir():
        raise ReproductionError(f"Flash-MinerU package missing: {source}")
    expected = str(config["dependencies"]["flash_mineru_commit"])
    try:
        actual = _git(path, "rev-parse", "HEAD")
        dirty = _git(path, "status", "--porcelain", "--", "flash_mineru")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReproductionError(f"cannot inspect Flash-MinerU checkout: {error}") from error
    if actual != expected:
        raise ReproductionError(
            f"Flash-MinerU revision mismatch: expected {expected}, got {actual}"
        )
    if dirty:
        raise ReproductionError("Flash-MinerU flash_mineru/ has local changes")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def configure_profile(args: argparse.Namespace) -> int:
    cmk_file = args.cmk_file.expanduser().resolve()
    profile_file = args.profile_file.expanduser().resolve()
    if not cmk_file.is_file():
        raise ReproductionError(f"CMK file does not exist: {cmk_file}")
    if profile_file.exists() and not args.force:
        raise ReproductionError(
            f"profile already exists: {profile_file}; pass --force to replace it"
        )
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", args.remote_workdir):
        raise ReproductionError("remote workdir contains unsupported characters")
    document = f'''[profiles.{PROFILE_NAME}]
cmk_file = {json.dumps(str(cmk_file))}
compute_spec_file = {json.dumps(str(COMPUTE_SPEC.resolve()))}
remote_workdir = {json.dumps(args.remote_workdir)}
wait = true
poll_interval = 10
max_wait = 172800
compute_ready_timeout = 7200
poll_error_grace = 300
concurrency = 1
keep_compute = true
keep_remote_files = true
'''
    _atomic_write(profile_file, document)
    profile_file.chmod(0o600)
    print(
        json.dumps(
            {
                "profile_file": str(profile_file),
                "profile_name": PROFILE_NAME,
                "compute_spec": str(COMPUTE_SPEC.resolve()),
                "contains_secret_material": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReproductionError(f"{label} does not exist: {resolved}")
    return resolved


def _prepare_launch(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    config = load_config(args.engine)
    flash_repo = args.flash_repo.expanduser().resolve()
    validate_flash_repo(flash_repo, config)
    profile_file = _require_file(args.profile_file, "TaiJi profile")
    output_root = args.output_root.expanduser().resolve(strict=False)
    os.environ["RAYORCH_FLASH_MINERU_REPO"] = str(flash_repo)
    os.environ["RAYORCH_TAIJI_PROFILE_FILE"] = str(profile_file)
    os.environ["RAYORCH_CEPH_ZW_ROOT"] = str(output_root)
    if args.command == "submit":
        token_file = _require_file(args.ceph_token_file, "TaiJi PAT token")
        os.environ["RAYORCH_TAIJI_PAT_TOKEN_FILE"] = str(token_file)
    return config, profile_file, output_root


def run_plan_or_submit(args: argparse.Namespace) -> int:
    config, profile_file, output_root = _prepare_launch(args)
    from experiments.taiji_multinode_mineru import orchestrate

    state_file = (
        args.state_file.expanduser().resolve(strict=False)
        if args.state_file
        else (DEFAULT_STATE_DIR / f"{args.run_id}.json").resolve(strict=False)
    )
    command = [
        args.command,
        "--engine",
        args.engine,
        "--run-id",
        args.run_id,
        "--output-uri",
        str(output_root / args.run_id),
        "--profile",
        f"{profile_file}:{PROFILE_NAME}",
        "--state-file",
        str(state_file),
        "--hyperparameters-json",
        json.dumps(config["launcher_hyperparameters"], separators=(",", ":")),
    ]
    if args.benchmark_only:
        command.append("--benchmark-only")
    if args.command == "submit" and args.reuse_state_file:
        command.extend(
            ["--reuse-state-file", str(args.reuse_state_file.expanduser().resolve())]
        )
    return orchestrate.main(command)


def verify(args: argparse.Namespace) -> int:
    engines = (args.engine,) if args.engine else ("rayorch", "raydata")
    configs = {engine: load_config(engine) for engine in engines}
    if args.flash_repo:
        for config in configs.values():
            validate_flash_repo(args.flash_repo.expanduser().resolve(), config)
    print(
        json.dumps(
            {
                "status": "ok",
                "engines": list(engines),
                "configs": [str(_config_path(engine)) for engine in engines],
                "flash_repo_checked": bool(args.flash_repo),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", required=True, choices=("rayorch", "raydata"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("MINERU_64GPU_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--flash-repo",
        type=Path,
        required="RAYORCH_FLASH_MINERU_REPO" not in os.environ,
        default=(
            Path(os.environ["RAYORCH_FLASH_MINERU_REPO"])
            if "RAYORCH_FLASH_MINERU_REPO" in os.environ
            else None
        ),
    )
    parser.add_argument("--profile-file", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--benchmark-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser("configure")
    configure.add_argument("--cmk-file", type=Path, required=True)
    configure.add_argument("--profile-file", type=Path, default=DEFAULT_PROFILE)
    configure.add_argument(
        "--remote-workdir",
        default="hydp_runs/mineru_64gpu_repro",
    )
    configure.add_argument("--force", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--engine", choices=("rayorch", "raydata"))
    verify_parser.add_argument("--flash-repo", type=Path)

    plan = subparsers.add_parser("plan")
    _add_launch_arguments(plan)

    submit = subparsers.add_parser("submit")
    _add_launch_arguments(submit)
    submit.add_argument(
        "--ceph-token-file",
        type=Path,
        required="RAYORCH_TAIJI_PAT_TOKEN_FILE" not in os.environ,
        default=(
            Path(os.environ["RAYORCH_TAIJI_PAT_TOKEN_FILE"])
            if "RAYORCH_TAIJI_PAT_TOKEN_FILE" in os.environ
            else None
        ),
    )
    submit.add_argument("--reuse-state-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "configure":
        return configure_profile(args)
    if args.command == "verify":
        return verify(args)
    return run_plan_or_submit(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReproductionError",
    "build_parser",
    "configure_profile",
    "load_config",
    "main",
    "validate_config",
    "validate_flash_repo",
]
