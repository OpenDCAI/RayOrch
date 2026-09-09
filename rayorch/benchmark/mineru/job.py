"""Build wheels and submit the MinerU UDF group through Ray Jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rayorch.benchmark.registry import get_plugin
from rayorch.benchmark.runtime import ResolvedRuntimeEnv, load_runtime_env


@dataclass(frozen=True, slots=True)
class JobBundle:
    entrypoint: str
    runtime_env: ResolvedRuntimeEnv
    wheels: tuple[Path, ...]


def build_wheel(source: Path, wheel_dir: Path) -> Path:
    """Build one source tree without mutating or uploading its work directory."""

    wheel_dir.mkdir(parents=True, exist_ok=True)
    before = set(wheel_dir.glob("*.whl"))
    with tempfile.TemporaryDirectory(prefix="rayorch-wheel-source-") as raw_stage:
        stage = Path(raw_stage) / source.name
        stage.mkdir()
        metadata_names = {
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "README.rst",
            "pyproject.toml",
            "requirements.txt",
            "setup.cfg",
            "setup.py",
        }
        for child in source.iterdir():
            if child.is_file() and child.name in metadata_names:
                shutil.copy2(child, stage / child.name)
            elif child.is_dir() and (child / "__init__.py").is_file():
                shutil.copytree(
                    child,
                    stage / child.name,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "*.pyo", "*.egg-info"
                    ),
                )
        # These wheels transport code through Ray ``py_modules``. Ray installs
        # them separately and would otherwise resolve each project's dependency
        # metadata a second time. The plugin-owned runtime_env is the single
        # dependency authority for this UDF group.
        (stage / "requirements.txt").write_text("", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheel_dir),
                str(stage),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"wheel build failed for {source}:\n{completed.stdout}\n{completed.stderr}"
        )
    created = set(wheel_dir.glob("*.whl")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new wheel for {source}, got {sorted(created)}")
    return created.pop()


def create_bundle(
    *,
    project_root: Path,
    flash_repo: Path,
    benchmark_args: Sequence[str],
    wheel_dir: Path,
    extra_pip: Sequence[str] = (),
) -> JobBundle:
    """Build distributable code and the one shared MinerU runtime environment."""

    if not (project_root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"RayOrch project root is invalid: {project_root}")
    if not (flash_repo / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Flash-MinerU project root is invalid: {flash_repo}")

    wheels = (
        build_wheel(project_root, wheel_dir),
        build_wheel(flash_repo, wheel_dir),
    )
    plugin = get_plugin("mineru")
    runtime_env = load_runtime_env(
        plugin,
        py_modules=wheels,
        extra_pip=extra_pip,
    )
    forwarded = list(benchmark_args)
    if "--flash-repo" not in forwarded:
        forwarded.extend(("--flash-repo", str(flash_repo)))
    if "--ray-address" not in forwarded:
        forwarded.extend(("--ray-address", "auto"))
    entrypoint = shlex.join(
        ["python", "-m", "rayorch.benchmark.mineru.pipeline", *forwarded]
    )
    return JobBundle(entrypoint, runtime_env, wheels)


async def _stream_logs(client: Any, job_id: str) -> None:
    async for line in client.tail_job_logs(job_id):
        print(line, end="" if line.endswith("\n") else "\n")


def submit_bundle(
    bundle: JobBundle,
    *,
    address: str,
    submission_id: str | None,
    wait: bool,
) -> str:
    """Submit one bundle and optionally stream it through terminal completion."""

    from ray.job_submission import JobSubmissionClient  # pyright: ignore[reportMissingImports]

    client = JobSubmissionClient(address)
    job_id = client.submit_job(
        entrypoint=bundle.entrypoint,
        runtime_env=bundle.runtime_env.value,
        submission_id=submission_id,
        metadata={
            "rayorch.benchmark": bundle.runtime_env.plugin,
            "rayorch.runtime_env.digest": bundle.runtime_env.digest,
        },
    )
    print(f"submitted Ray Job {job_id}")
    if wait:
        asyncio.run(_stream_logs(client, job_id))
        status = str(client.get_job_status(job_id))
        if status not in {"SUCCEEDED", "JobStatus.SUCCEEDED"}:
            raise RuntimeError(f"Ray Job {job_id} finished with status {status}")
    return job_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package and submit the MinerU benchmark as an isolated Ray Job"
    )
    parser.add_argument("--address", required=True, help="Ray dashboard Jobs URL")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--flash-repo", type=Path, required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--extra-pip", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forwarded = list(args.benchmark_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    with tempfile.TemporaryDirectory(prefix="rayorch-mineru-wheels-") as raw_dir:
        bundle = create_bundle(
            project_root=args.project_root.resolve(),
            flash_repo=args.flash_repo.resolve(),
            benchmark_args=forwarded,
            wheel_dir=Path(raw_dir),
            extra_pip=args.extra_pip,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "entrypoint": bundle.entrypoint,
                        "runtime_env": bundle.runtime_env.value,
                        "runtime_env_digest": bundle.runtime_env.digest,
                        "wheels": [str(path) for path in bundle.wheels],
                    },
                    indent=2,
                )
            )
            return 0
        submit_bundle(
            bundle,
            address=args.address,
            submission_id=args.submission_id,
            wait=not args.no_wait,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JobBundle",
    "build_parser",
    "build_wheel",
    "create_bundle",
    "submit_bundle",
]
