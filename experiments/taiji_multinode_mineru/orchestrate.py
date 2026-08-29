"""Provision TaiJi compute and launch probe -> full MinerU Ray Jobs.

The successful submit path deliberately detaches after the full job has been
observed RUNNING twice.  It keeps the owned compute alive.  Failure before
detach cancels and confirms the active job before releasing that compute.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import tomllib
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit
import uuid


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY = EXPERIMENT_DIR.parents[1]
COMPUTE_SPEC_PATH = EXPERIMENT_DIR / "compute_h20_8node_8gpu_gy.json"
ZW_TEXT_COMPUTE_SPEC_PATH = EXPERIMENT_DIR / "compute_h20_8node_8gpu.json"
ZW_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_8node_8gpu_zw_pretrain.json"
)
ZW_TWO_BY_FOUR_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_2node_4gpu_zw_pretrain.json"
)
ZW_ONE_BY_FOUR_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_1node_4gpu_zw_pretrain.json"
)
ZW_ONE_BY_EIGHT_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_1node_8gpu_zw_pretrain.json"
)
ZW_TWO_BY_EIGHT_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_2node_8gpu_zw_pretrain.json"
)
ZW_FOUR_BY_EIGHT_COMPUTE_SPEC_PATH = (
    EXPERIMENT_DIR / "compute_h20_4node_8gpu_zw_pretrain.json"
)
PROFILE_PATH = Path(
    os.environ.get(
        "RAYORCH_TAIJI_PROFILE_FILE",
        str(EXPERIMENT_DIR / "profile.toml"),
    )
).expanduser()
PROFILE_NAME = "ray-h20-8x8-gy"
ZW_PROFILE_NAME = "ray-h20-8x8-zw-pretrain"
ZW_TEXT_PROFILE_NAME = "ray-h20-8x8"
ZW_TWO_BY_FOUR_PROFILE_NAME = "ray-h20-2x4-zw-pretrain"
ZW_ONE_BY_FOUR_PROFILE_NAME = "ray-h20-1x4-zw-pretrain"
ZW_ONE_BY_EIGHT_PROFILE_NAME = "ray-h20-1x8-zw-pretrain"
ZW_TWO_BY_EIGHT_PROFILE_NAME = "ray-h20-2x8-zw-pretrain"
ZW_FOUR_BY_EIGHT_PROFILE_NAME = "ray-h20-4x8-zw-pretrain"
DEFAULT_PROFILE_REF = f"{PROFILE_PATH}:{PROFILE_NAME}"
ZW_PROFILE_REF = f"{PROFILE_PATH}:{ZW_PROFILE_NAME}"
ZW_TEXT_PROFILE_REF = f"{PROFILE_PATH}:{ZW_TEXT_PROFILE_NAME}"
ZW_TWO_BY_FOUR_PROFILE_REF = (
    f"{PROFILE_PATH}:{ZW_TWO_BY_FOUR_PROFILE_NAME}"
)
ZW_ONE_BY_FOUR_PROFILE_REF = (
    f"{PROFILE_PATH}:{ZW_ONE_BY_FOUR_PROFILE_NAME}"
)
ZW_ONE_BY_EIGHT_PROFILE_REF = (
    f"{PROFILE_PATH}:{ZW_ONE_BY_EIGHT_PROFILE_NAME}"
)
ZW_TWO_BY_EIGHT_PROFILE_REF = (
    f"{PROFILE_PATH}:{ZW_TWO_BY_EIGHT_PROFILE_NAME}"
)
ZW_FOUR_BY_EIGHT_PROFILE_REF = (
    f"{PROFILE_PATH}:{ZW_FOUR_BY_EIGHT_PROFILE_NAME}"
)
REMOTE_DRIVER_PATH = EXPERIMENT_DIR / "remote_driver.py"
TAIJI_PAT_TOKEN_PATH = Path(
    os.environ.get(
        "RAYORCH_TAIJI_PAT_TOKEN_FILE",
        "/apdcephfs_zwfy10/share_304380933/hunyuan/clapliang/taijiPATToken",
    )
).expanduser()
APP_GROUP = "TaiJi_HYAide_LLM_Pretrain_Data"
GPU_TYPE = "H20"
LOCATION = "gy"
CEPH_OUTPUT_ROOTS = {
    "gy": Path(
        os.environ.get(
            "RAYORCH_CEPH_GY_ROOT",
            "/apdcephfs_gy5/share_304380933/hunyuan/clapliang",
        )
    ).expanduser(),
    "zw": Path(
        os.environ.get(
            "RAYORCH_CEPH_ZW_ROOT",
            "/apdcephfs_zwfy10/share_304380933/hunyuan/clapliang",
        )
    ).expanduser(),
}
EXPECTED_WORKERS = 8
EXPECTED_GPUS_PER_WORKER = 8
EXPECTED_PDFS = 3690
HDFS_USER_PREFIX = (
    "hdfs://hdfs-zw1-teg-hunyuan-hdfshd1-v3/data/tianqiong/TEG/"
    "g_teg_tencentchat_dataplat_g_teg_tencentchat_dataplat_tencent_hunyuan_wedata/"
    "taiji/x1_data/clapliang"
)
HDFS_PREFIX = f"{HDFS_USER_PREFIX}/datasets"
DEFAULT_INPUT_CONTRACTS = (
    {
        "uri": f"{HDFS_PREFIX}/pdfs_2000",
        "files": 2_000,
        "bytes": 4_060_449_252,
    },
    {
        "uri": f"{HDFS_PREFIX}/aidata_pdf_300k_sample",
        "files": 1_690,
        "bytes": 1_099_397_628_624,
    },
)
DEFAULT_INPUT_URIS = tuple(item["uri"] for item in DEFAULT_INPUT_CONTRACTS)
DEFAULT_OUTPUT_BASE = f"{HDFS_USER_PREFIX}/experiments"
DEFAULT_MODEL_URI = (
    f"{HDFS_USER_PREFIX}/runtime_assets/rayorch_v36_20260820/"
    "MinerU2.5-2509-1.2B"
)
DEFAULT_FLASH_REPO = os.environ.get(
    "RAYORCH_FLASH_MINERU_REPO",
    "/apdcephfs_zwfy10/share_304380933/hunyuan/"
    "sunnyhazema/workspace/Flash-mineru",
)
FLASH_SOURCE_DIR = Path(DEFAULT_FLASH_REPO) / "flash_mineru"
REMOTE_FLASH_REPO = "."
PROBE_MARKER = "RAYORCH_MINERU_PROBE_RESULT "
FULL_MARKER = "RAYORCH_MINERU_FULL_RESULT "
HDFS_DIRECT_READY_MARKER = "RAYORCH_MINERU_HDFS_DIRECT_READY "
TERMINAL_JOB_STATES = {"SUCCEEDED", "FAILED", "STOPPED", "ERROR"}
ACTIVE_JOB_STATES = {"PENDING", "RUNNING"}
HYPERPARAMETERS = {
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


class ValidationError(ValueError):
    """An immutable local launch contract is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"{path} must contain one JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    roots = (
        ("rayorch", REPOSITORY / "rayorch"),
        ("taiji_multinode_mineru/remote_driver.py", REMOTE_DRIVER_PATH),
        ("flash_mineru", FLASH_SOURCE_DIR),
    )
    for label, root in roots:
        if not root.exists():
            raise ValidationError(f"snapshot source does not exist: {root}")
        paths = list(root.rglob("*.py")) if root.is_dir() else [root]
        for path in sorted(paths, key=lambda item: str(item)):
            if "__pycache__" in path.parts:
                continue
            suffix = path.relative_to(root).as_posix() if root.is_dir() else ""
            snapshot_name = label.rstrip("/") + (f"/{suffix}" if suffix else "")
            digest.update(snapshot_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _validate_hdfs_uri(uri: str, *, name: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme != "hdfs" or parsed.netloc != "hdfs-zw1-teg-hunyuan-hdfshd1-v3":
        raise ValidationError(f"{name} must use the approved HDFS nameservice")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValidationError(f"{name} must not contain credentials/query/fragment")
    if not uri.startswith(HDFS_USER_PREFIX + "/"):
        raise ValidationError(f"{name} must stay below {HDFS_USER_PREFIX}")
    if any(token in parsed.path for token in ("..", "*", "?", "[", "]", "{")):
        raise ValidationError(f"{name} contains an unsafe path token")
    return uri.rstrip("/")


def _validate_output_path(
    value: str,
    *,
    ceph_root: Path = CEPH_OUTPUT_ROOTS[LOCATION],
) -> tuple[str, str]:
    if urlsplit(value).scheme == "hdfs":
        return _validate_hdfs_uri(value, name="output URI"), "hdfs"
    root = ceph_root.resolve(strict=False)
    path = Path(value)
    resolved = path.resolve(strict=False)
    if not path.is_absolute() or (resolved != root and root not in resolved.parents):
        raise ValidationError(f"Ceph output must stay below {root}")
    if any(token in value for token in ("..", "*", "?", "[", "]", "{")):
        raise ValidationError("Ceph output contains an unsafe path token")
    return str(resolved), "ceph"


def validate_compute_spec(
    path: Path = COMPUTE_SPEC_PATH,
    *,
    expected_location: str = LOCATION,
    expected_app_group: str = APP_GROUP,
    expected_workers: int = EXPECTED_WORKERS,
    expected_gpus_per_worker: int = EXPECTED_GPUS_PER_WORKER,
) -> dict[str, Any]:
    spec = _load_json(path)
    if spec.get("runtime") != "ray":
        raise ValidationError("compute runtime must be ray")
    runtime = spec.get("runtime_spec")
    resource = spec.get("resource")
    if not isinstance(runtime, dict) or not isinstance(resource, dict):
        raise ValidationError("runtime_spec and resource must be objects")
    if runtime.get("taiji_app_group") != expected_app_group:
        raise ValidationError("runtime TaiJi application group mismatch")
    if runtime.get("taiji_app_group_location") != expected_location:
        raise ValidationError("runtime TaiJi application group location mismatch")
    groups = resource.get("executor_group")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValidationError("exactly one fixed executor group is required")
    group = groups[0]
    if not isinstance(group, dict):
        raise ValidationError("executor group must be an object")
    if (
        group.get("replicas"),
        group.get("min_replicas"),
        group.get("max_replicas"),
        group.get("num_gpu"),
    ) != (
        expected_workers,
        expected_workers,
        expected_workers,
        expected_gpus_per_worker,
    ):
        raise ValidationError(
            "compute must be fixed at "
            f"{expected_workers} workers x {expected_gpus_per_worker} GPUs"
        )
    extra = group.get("extra")
    provider = extra.get("gpu_provider") if isinstance(extra, dict) else None
    taiji = provider.get("taiji") if isinstance(provider, dict) else None
    if not isinstance(taiji, dict):
        raise ValidationError("TaiJi GPU provider is missing")
    if taiji.get("app_group_name") != expected_app_group:
        raise ValidationError("GPU provider TaiJi application group mismatch")
    if taiji.get("gpu_type") != GPU_TYPE:
        raise ValidationError("GPU provider must request H20")
    if taiji.get("app_group_location") != expected_location:
        raise ValidationError(
            f"GPU provider must request {expected_location}"
        )
    if int(group.get("num_cpu", 0)) < 16 or int(group.get("num_mem", 0)) < 131072:
        raise ValidationError("worker CPU/memory is below the safety floor")
    if int(resource.get("driver_cpu", 0)) < 8 or int(resource.get("driver_mem", 0)) < 65536:
        raise ValidationError("driver CPU/memory is below the safety floor")
    return spec


def validate_profile(
    path: Path = PROFILE_PATH,
    *,
    profile_name: str = PROFILE_NAME,
    expected_compute_path: Path = COMPUTE_SPEC_PATH,
) -> dict[str, Any]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(f"cannot load profile {path}: {error}") from error
    profiles = document.get("profiles")
    profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise ValidationError(f"profile {profile_name!r} is missing")
    if profile.get("keep_compute") is not True:
        raise ValidationError("submission profile must keep the compute")
    if profile.get("wait") is not True:
        raise ValidationError("submission profile must wait for readiness")
    cmk_file = str(profile.get("cmk_file") or "")
    if not cmk_file or not Path(cmk_file).is_absolute():
        raise ValidationError("profile must contain only an absolute CMK file reference")
    compute_ref = Path(str(profile.get("compute_spec_file") or ""))
    resolved = compute_ref if compute_ref.is_absolute() else path.parent / compute_ref
    if resolved.resolve() != expected_compute_path.resolve():
        raise ValidationError(
            f"profile must reference {expected_compute_path.name}"
        )
    return profile


def _launch_target(profile_reference: str) -> dict[str, Any]:
    try:
        profile_path_text, profile_name = profile_reference.rsplit(":", 1)
    except ValueError as error:
        raise ValidationError(
            "profile must use /absolute/profile.toml:profile-name"
        ) from error
    profile_path = Path(profile_path_text).resolve(strict=False)
    if profile_path != PROFILE_PATH.resolve(strict=False):
        raise ValidationError("only the frozen MinerU profile file is allowed")
    targets = {
        PROFILE_NAME: {
            "location": "gy",
            "compute_spec_path": COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["gy"],
            "app_group": APP_GROUP,
            "expected_workers": EXPECTED_WORKERS,
            "expected_gpus_per_worker": EXPECTED_GPUS_PER_WORKER,
        },
        ZW_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": EXPECTED_WORKERS,
            "expected_gpus_per_worker": EXPECTED_GPUS_PER_WORKER,
        },
        ZW_TEXT_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_TEXT_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": "TaiJi_HYAide_text_data_pipelines",
            "expected_workers": EXPECTED_WORKERS,
            "expected_gpus_per_worker": EXPECTED_GPUS_PER_WORKER,
        },
        ZW_TWO_BY_FOUR_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_TWO_BY_FOUR_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": 2,
            "expected_gpus_per_worker": 4,
        },
        ZW_ONE_BY_FOUR_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_ONE_BY_FOUR_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": 1,
            "expected_gpus_per_worker": 4,
        },
        ZW_ONE_BY_EIGHT_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_ONE_BY_EIGHT_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": 1,
            "expected_gpus_per_worker": 8,
        },
        ZW_TWO_BY_EIGHT_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_TWO_BY_EIGHT_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": 2,
            "expected_gpus_per_worker": 8,
        },
        ZW_FOUR_BY_EIGHT_PROFILE_NAME: {
            "location": "zw",
            "compute_spec_path": ZW_FOUR_BY_EIGHT_COMPUTE_SPEC_PATH,
            "ceph_root": CEPH_OUTPUT_ROOTS["zw"],
            "app_group": APP_GROUP,
            "expected_workers": 4,
            "expected_gpus_per_worker": 8,
        },
    }
    try:
        target = dict(targets[profile_name])
    except KeyError as error:
        raise ValidationError(
            f"unsupported MinerU profile: {profile_name}"
        ) from error
    validate_profile(
        profile_path,
        profile_name=profile_name,
        expected_compute_path=target["compute_spec_path"],
    )
    target["profile_name"] = profile_name
    return target


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValidationError("run id must use only letters, digits, dot, dash, underscore")
    return value


def _validate_hyperparameters(value: Mapping[str, Any]) -> dict[str, Any]:
    required = set(HYPERPARAMETERS)
    if set(value) != required:
        raise ValidationError(
            "hyperparameters must contain exactly: " + ", ".join(sorted(required))
        )
    result = dict(value)
    for name in required - {"gpu_memory_utilization", "gpus_per_ocr_actor"}:
        if not isinstance(result[name], int) or result[name] <= 0:
            raise ValidationError(f"hyperparameter {name} must be a positive int")
    utilization = result["gpu_memory_utilization"]
    if not isinstance(utilization, (int, float)) or not 0 < utilization < 1:
        raise ValidationError("gpu_memory_utilization must be between zero and one")
    gpu_fraction = result["gpus_per_ocr_actor"]
    if not isinstance(gpu_fraction, (int, float)) or not 0 < gpu_fraction <= 1:
        raise ValidationError("gpus_per_ocr_actor must be between zero and one")
    return result


def build_plan(
    *,
    run_id: str,
    input_uris: Sequence[str] = DEFAULT_INPUT_URIS,
    output_uri: str | None = None,
    model_uri: str = DEFAULT_MODEL_URI,
    input_limits: Sequence[int] | None = None,
    hyperparameters: Mapping[str, Any] = HYPERPARAMETERS,
    benchmark_only: bool = False,
    engine: str = "rayorch",
    profile_reference: str = DEFAULT_PROFILE_REF,
) -> dict[str, Any]:
    run_id = _safe_run_id(run_id)
    if engine not in {"rayorch", "raydata"}:
        raise ValidationError("engine must be rayorch or raydata")
    validated_inputs = tuple(
        _validate_hdfs_uri(uri, name=f"input URI {index}")
        for index, uri in enumerate(input_uris)
    )
    known_contracts = {item["uri"]: item for item in DEFAULT_INPUT_CONTRACTS}
    if len(validated_inputs) != len(set(validated_inputs)):
        raise ValidationError("input URIs must be unique")
    try:
        input_contracts = [dict(known_contracts[uri]) for uri in validated_inputs]
    except KeyError as error:
        raise ValidationError(f"input URI has no frozen manifest: {error.args[0]}") from error
    limits = (
        list(input_limits)
        if input_limits is not None
        else [int(item["files"]) for item in input_contracts]
    )
    if len(limits) != len(input_contracts) or any(
        not isinstance(limit, int)
        or limit <= 0
        or limit > int(contract["files"])
        for limit, contract in zip(limits, input_contracts, strict=True)
    ):
        raise ValidationError("input limits must match and fit frozen manifests")
    frozen_hyperparameters = _validate_hyperparameters(hyperparameters)
    launch_target = _launch_target(profile_reference)
    output_uri, output_kind = _validate_output_path(
        output_uri or f"{DEFAULT_OUTPUT_BASE}/{run_id}",
        ceph_root=launch_target["ceph_root"],
    )
    model_uri = _validate_hdfs_uri(model_uri, name="model URI")
    if not output_uri.endswith("/" + run_id):
        raise ValidationError("output URI must end with the run id")
    if output_kind == "hdfs" and any(
        uri == output_uri or output_uri.startswith(uri + "/")
        for uri in validated_inputs
    ):
        raise ValidationError("output must not equal or nest below an input")
    spec = validate_compute_spec(
        launch_target["compute_spec_path"],
        expected_location=launch_target["location"],
        expected_app_group=launch_target["app_group"],
        expected_workers=launch_target["expected_workers"],
        expected_gpus_per_worker=launch_target["expected_gpus_per_worker"],
    )
    return {
        "schema_version": 1,
        "action": "probe_then_submit_and_detach",
        "engine": engine,
        "run_id": run_id,
        "input_uris": list(validated_inputs),
        "input_contracts": input_contracts,
        "output_uri": output_uri,
        "output_kind": output_kind,
        "output_base_uri": output_uri.rsplit("/", 1)[0],
        "model_uri": model_uri,
        "expected_pdfs": sum(limits),
        "input_limits": limits,
        "benchmark_only": bool(benchmark_only),
        "compute": {
            "workers": launch_target["expected_workers"],
            "gpu_per_worker": launch_target["expected_gpus_per_worker"],
            "gpu_total": (
                launch_target["expected_workers"]
                * launch_target["expected_gpus_per_worker"]
            ),
            "gpu_type": GPU_TYPE,
            "taiji_app_group": launch_target["app_group"],
            "location": launch_target["location"],
            "spec_fingerprint": _fingerprint(spec),
        },
        "source_digest": _source_digest(),
        "flash_repo": REMOTE_FLASH_REPO,
        "hyperparameters": frozen_hyperparameters,
        "success_policy": {
            "running_confirmations": 2,
            "detach": True,
            "cancel_full_job": False,
            "release_compute": False,
        },
        "failure_policy": {
            "cancel_active_job_first": True,
            "require_cancel_confirmation_before_compute_release": True,
            "release_only_owned_compute": True,
            "delete_hdfs_data": False,
        },
    }


@contextmanager
def source_snapshot(
    expected_digest: str,
    *,
    include_ceph_token: bool = False,
) -> Iterator[Path]:
    """Yield a minimal immutable Ray working directory for probe and full."""

    if _source_digest() != expected_digest:
        raise ValidationError("source changed after plan creation")
    with tempfile.TemporaryDirectory(prefix="rayorch-mineru-taiji-") as temporary:
        root = Path(temporary) / "working_dir"
        shutil.copytree(
            REPOSITORY / "rayorch",
            root / "rayorch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        shutil.copytree(
            FLASH_SOURCE_DIR,
            root / "flash_mineru",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        remote_dir = root / "taiji_multinode_mineru"
        remote_dir.mkdir(parents=True)
        shutil.copy2(REMOTE_DRIVER_PATH, remote_dir / "remote_driver.py")
        if include_ceph_token:
            if not TAIJI_PAT_TOKEN_PATH.is_file():
                raise ValidationError(
                    f"TaiJi PAT token file is missing: {TAIJI_PAT_TOKEN_PATH}"
                )
            runtime_token = remote_dir / "taijiPATToken.runtime"
            shutil.copyfile(TAIJI_PAT_TOKEN_PATH, runtime_token)
            runtime_token.chmod(0o600)
        manifest = {
            "schema_version": 1,
            "source_digest": expected_digest,
            "created_at": _utc_now(),
        }
        _atomic_json(root / "snapshot-manifest.json", manifest)
        yield root


def _job_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "RAYORCH_RUN_ID": str(plan["run_id"]),
        "RAYORCH_ENGINE": str(plan["engine"]),
        "RAYORCH_HDFS_INPUT_URIS": json.dumps(plan["input_uris"]),
        "RAYORCH_EXPECTED_PDF_FILES": json.dumps(
            [item["files"] for item in plan["input_contracts"]]
        ),
        "RAYORCH_EXPECTED_PDF_BYTES": json.dumps(
            [item["bytes"] for item in plan["input_contracts"]]
        ),
        "RAYORCH_OUTPUT_BASE": str(plan["output_base_uri"]),
        "RAYORCH_OUTPUT_KIND": str(plan["output_kind"]),
        "RAYORCH_HDFS_MODEL_URI": str(plan["model_uri"]),
        "RAYORCH_EXPECTED_PDFS": str(plan["expected_pdfs"]),
        "RAYORCH_PDF_LIMITS": json.dumps(plan["input_limits"]),
        "RAYORCH_BENCHMARK_ONLY": "1" if plan["benchmark_only"] else "0",
        "RAYORCH_FLASH_MINERU_REPO": str(plan["flash_repo"]),
        "RAYORCH_SOURCE_DIGEST": str(plan["source_digest"]),
        "RAYORCH_HYPERPARAMETERS": json.dumps(plan["hyperparameters"]),
        "RAYORCH_CEPH_APP_GROUP": str(plan["compute"]["taiji_app_group"]),
        "RAYORCH_CEPH_LOCATION": str(plan["compute"]["location"]),
        "RAYORCH_EXPECTED_GPU_WORKERS": str(plan["compute"]["workers"]),
        "RAYORCH_GPUS_PER_WORKER": str(plan["compute"]["gpu_per_worker"]),
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "RAY_DEDUP_LOGS": "1",
        "LOGURU_LEVEL": "INFO",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }


def _dashboard_url(compute: Mapping[str, Any]) -> str:
    context = compute.get("context")
    if not isinstance(context, Mapping):
        context = {}
    return str(context.get("dashboard_url") or compute.get("dashboard_url") or "").strip()


def _save_logs(path: Path, logs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(logs) + "\n", encoding="utf-8")


def _marker(logs: list[str], marker: str) -> dict[str, Any]:
    text = next(
        (line.split(marker, 1)[1] for line in reversed(logs) if marker in line),
        "",
    )
    if not text:
        raise RuntimeError(f"job logs do not contain {marker.strip()}")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{marker.strip()} must contain a JSON object")
    return payload


def _job_dict(job: Any) -> dict[str, Any]:
    if hasattr(job, "to_dict"):
        return dict(job.to_dict())
    return {
        "submission_id": str(job.submission_id),
        "transport": str(job.transport),
        "address": getattr(job, "address", None),
        "dashboard_url": getattr(job, "dashboard_url", None),
        "proxy_url": getattr(job, "proxy_url", None),
        "metadata": dict(getattr(job, "metadata", {})),
    }


def _restore_job(payload: Mapping[str, Any]) -> Any:
    from hydp_engine.ray_jobs import RayJobRef

    return RayJobRef(
        submission_id=str(payload["submission_id"]),
        transport=str(payload["transport"]),
        address=payload.get("address"),
        dashboard_url=payload.get("dashboard_url"),
        proxy_url=payload.get("proxy_url"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _submit_job(
    jobs: Any,
    *,
    phase: str,
    dashboard: str,
    credentials: Any,
    working_dir: Path,
    plan: Mapping[str, Any],
) -> Any:
    from hydp_engine.ray_jobs.packaging import maybe_upload_ray_working_dir

    runtime_env: dict[str, Any] = {
        "env_vars": _job_environment(plan),
        "pip": [
            "numpy==1.26.0",
            "opencv-python-headless==4.10.0.84",
            "PyMuPDF==1.26.3",
            "mineru-vl-utils==1.2.1",
            "pypdfium2==5.13.0",
            "magika==1.0.3",
            "transformers==4.57.6",
        ],
    }
    maybe_upload_ray_working_dir(
        runtime_env,
        working_dir=str(working_dir),
        address=dashboard,
        proxy_url=None,
        proxy_post=jobs._proxy_post,
        package_builder=jobs._package_builder,
    )
    package_uri = str(runtime_env.get("working_dir") or "")
    if not package_uri.startswith("gcs://"):
        raise RuntimeError("Ray code package upload did not produce a gcs:// URI")
    runtime_env["py_modules"] = [package_uri]
    return jobs.submit_proxy(
        entrypoint=(
            "python -u taiji_multinode_mineru/remote_driver.py "
            f"--phase {phase}"
        ),
        submission_id=(
            f"rayorch-mineru-v36-{phase}-{plan['run_id']}-"
            + uuid.uuid4().hex[:8]
        )[:128],
        dashboard_url=dashboard,
        runtime_env=runtime_env,
        working_dir=None,
        runtime_auth=True,
        user=credentials.user,
        cmk=credentials.cmk,
        metadata={
            "hydp_stage": f"rayorch_mineru_v36_{phase}",
            "run_id": str(plan["run_id"]),
            "source_digest": str(plan["source_digest"]),
        },
    )


def _wait_terminal(
    jobs: Any,
    job: Any,
    *,
    timeout: float,
    poll_interval: float,
    state: dict[str, Any],
    state_path: Path,
    phase: str,
) -> tuple[str, list[str]]:
    deadline = time.monotonic() + timeout
    last_status = ""
    while True:
        detail = jobs.status(job)
        status = str(detail.get("status") or "UNKNOWN").upper()
        state[f"{phase}_status"] = status
        if status != last_status:
            state[f"{phase}_status_changed_at"] = _utc_now()
            _atomic_json(state_path, state)
            print(
                "RAY_JOB_STATUS "
                + json.dumps(
                    {
                        "phase": phase,
                        "submission_id": job.submission_id,
                        "status": status,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_status = status
        if status in TERMINAL_JOB_STATES:
            logs = jobs.logs(job, tail=12000)
            return status, logs
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{phase} job timed out: {job.submission_id}")
        time.sleep(poll_interval)


def _wait_running(
    jobs: Any,
    job: Any,
    *,
    timeout: float,
    poll_interval: float,
    confirmations: int,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    observed = 0
    while True:
        detail = jobs.status(job)
        status = str(detail.get("status") or "UNKNOWN").upper()
        state.update(
            {
                "full_status": status,
                "running_confirmations_observed": observed,
                "full_last_checked_at": _utc_now(),
            }
        )
        _atomic_json(state_path, state)
        if status == "RUNNING":
            observed += 1
            state["running_confirmations_observed"] = observed
            _atomic_json(state_path, state)
            if observed >= confirmations:
                return
        else:
            observed = 0
        if status in TERMINAL_JOB_STATES:
            logs = jobs.logs(job, tail=12000)
            _save_logs(
                state_path.with_name(state_path.stem + "_full_driver.log"),
                logs,
            )
            raise RuntimeError(f"full job became {status} before detach")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"full job did not reach stable RUNNING: {job.submission_id}"
            )
        time.sleep(poll_interval)


def _wait_for_log_marker(
    jobs: Any,
    job: Any,
    *,
    marker: str,
    timeout: float,
    poll_interval: float,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    """Wait for a business-level readiness marker while the Ray Job is active."""

    deadline = time.monotonic() + timeout
    while True:
        detail = jobs.status(job)
        status = str(detail.get("status") or "UNKNOWN").upper()
        logs = jobs.logs(job, tail=12000)
        if status in TERMINAL_JOB_STATES:
            _save_logs(
                state_path.with_name(state_path.stem + "_full_driver.log"),
                logs,
            )
            raise RuntimeError(
                f"full job became {status} before HDFS direct-read readiness"
            )
        try:
            payload = _marker(logs, marker)
        except Exception:
            payload = None
        if payload is not None:
            state.update(
                {
                    "hdfs_direct_ready": payload,
                    "hdfs_direct_ready_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "full job did not confirm HDFS direct-read readiness: "
                f"{job.submission_id}"
            )
        time.sleep(poll_interval)


def _cancel_and_confirm(
    jobs: Any,
    job: Any,
    *,
    poll_interval: float,
    timeout: float = 300,
) -> tuple[bool, str]:
    try:
        initial = str(jobs.status(job).get("status") or "UNKNOWN").upper()
        if initial in TERMINAL_JOB_STATES:
            return True, initial
        jobs.cancel(job)
        deadline = time.monotonic() + timeout
        while True:
            status = str(jobs.status(job).get("status") or "UNKNOWN").upper()
            if status in TERMINAL_JOB_STATES:
                return True, status
            if time.monotonic() >= deadline:
                return False, status
            time.sleep(poll_interval)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_probe(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    workers = result.get("gpu_workers")
    accelerator_resources = result.get("accelerator_resources")
    if (
        result.get("status") != "ok"
        or result.get("read_only") is not True
        or result.get("source_digest") != plan["source_digest"]
        or result.get("pdf_manifests") != plan["input_contracts"]
        or not isinstance(result.get("model_manifest"), Mapping)
        or result["model_manifest"].get("uri") != plan["model_uri"]
        or result.get("expected_gpu_workers") != plan["compute"]["workers"]
        or result.get("expected_gpus_per_worker")
        != plan["compute"]["gpu_per_worker"]
        or not isinstance(workers, list)
        or len(workers) != plan["compute"]["workers"]
        or any(
            item.get("gpus") != plan["compute"]["gpu_per_worker"]
            for item in workers
        )
        or not isinstance(accelerator_resources, Mapping)
        or len(accelerator_resources) != plan["compute"]["workers"]
        or any(float(value) <= 0 for value in accelerator_resources.values())
    ):
        raise RuntimeError(f"probe gate failed: {result}")


def _validate_full_readiness(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    expected_output_mode = (
        "benchmark_only"
        if plan["benchmark_only"]
        else f"local_spool_async_{plan['output_kind']}"
    )
    mount_results = payload.get("mount_results")
    ceph_mount_ready = plan["output_kind"] != "ceph" or (
        isinstance(mount_results, list)
        and len(mount_results) >= plan["compute"]["workers"]
        and all(
            isinstance(item, Mapping)
            and item.get("mounted") is True
            and item.get("output_base") == plan["output_base_uri"]
            for item in mount_results
        )
    )
    if (
        payload.get("mode") != expected_output_mode
        or payload.get("pdf_uris") != plan["input_uris"]
        or payload.get("gpu_workers") != plan["compute"]["workers"]
        or not ceph_mount_ready
    ):
        raise RuntimeError(
            f"HDFS direct-read readiness gate failed: {payload}"
        )


def execute_submit(
    plan: dict[str, Any],
    *,
    profile_reference: str,
    state_path: Path,
    reuse_state_path: Path | None = None,
) -> int:
    """Run the external workflow; this is the only submission mutation path."""

    from hydp_dataflow_wedata.compute import ComputeLease, ComputeManager
    from hydp_dataflow_wedata.config import load_profile
    from hydp_dataflow_wedata.events import console
    from hydp_engine import HydpEngineClient
    from hydp_engine.ray_jobs import RayJobClient

    if state_path.exists():
        raise ValidationError(f"refusing to overwrite state file: {state_path}")
    profile = load_profile(profile_reference)
    if not profile.keep_compute:
        raise ValidationError("submit profile must set keep_compute=true")
    profile_spec = profile.compute_spec()
    if _fingerprint(profile_spec) != plan["compute"]["spec_fingerprint"]:
        raise ValidationError("profile compute spec differs from validated plan")
    reuse_state: dict[str, Any] | None = None
    reuse_compute_id: str | None = None
    if reuse_state_path is not None:
        if reuse_state_path.resolve() == state_path.resolve():
            raise ValidationError("reuse state and new state file must differ")
        reuse_state = _load_json(reuse_state_path)
        reuse_compute_id = str(reuse_state.get("compute_id") or "")
        if not reuse_compute_id or reuse_state.get("compute_owned") is not True:
            raise ValidationError(
                "reuse state must record an owned retained compute"
            )
        previous_job_payload = reuse_state.get("full_job")
        if not isinstance(previous_job_payload, Mapping):
            raise ValidationError("reuse state has no previous full job")
        previous_job = _restore_job(previous_job_payload)
        previous_status = str(
            RayJobClient().status(previous_job).get("status") or "UNKNOWN"
        ).upper()
        if previous_status not in TERMINAL_JOB_STATES:
            raise ValidationError(
                "cannot reuse compute while previous full job is active: "
                f"{previous_status}"
            )
        profile = replace(
            profile,
            compute_id=reuse_compute_id,
            compute_spec_file=None,
        )
    credentials = profile.credentials()
    engine = HydpEngineClient(
        user=credentials.user,
        cmk=credentials.cmk,
        cmk_id=credentials.cmk_id,
    )
    manager = ComputeManager(engine, profile)
    cleanup_manager = ComputeManager(engine, replace(profile, keep_compute=False))
    jobs = RayJobClient()
    state: dict[str, Any] = {
        "schema_version": 1,
        "phase": "initializing",
        "status": "starting",
        "started_at": _utc_now(),
        "plan": plan,
        "profile_reference": profile_reference,
        "compute_reused": reuse_state_path is not None,
        "compute_owner_state": (
            str(reuse_state_path) if reuse_state_path is not None else None
        ),
    }
    lease: ComputeLease | None = None
    active_job: Any | None = None
    detached = False
    exit_code = 1
    _atomic_json(state_path, state)

    def compute_event(event: Mapping[str, Any]) -> None:
        nonlocal lease
        if event.get("phase") == "compute_acquired" and event.get("compute_id"):
            lease = ComputeLease(
                id=str(event["compute_id"]),
                owned=bool(event.get("owned")),
                status=str(event.get("status") or ""),
            )
            state.update({"compute_id": lease.id, "compute_owned": lease.owned})
            _atomic_json(state_path, state)
        console(dict(event))

    with source_snapshot(
        str(plan["source_digest"]),
        include_ceph_token=plan["output_kind"] == "ceph",
    ) as snapshot:
        state["working_dir_snapshot"] = {
            "source_digest": plan["source_digest"],
            "local_temporary": True,
        }
        _atomic_json(state_path, state)
        try:
            state["phase"] = "acquiring_compute"
            _atomic_json(state_path, state)
            lease = manager.acquire(on_event=compute_event)
            if reuse_compute_id is not None and (
                lease.id != reuse_compute_id or lease.owned
            ):
                raise RuntimeError(
                    "reused compute acquisition returned an unexpected lease: "
                    f"id={lease.id}, owned={lease.owned}"
                )
            state.update({"compute_id": lease.id, "compute_owned": lease.owned})
            compute = engine.get_compute(lease.id)
            dashboard = _dashboard_url(compute)
            if not dashboard:
                raise RuntimeError("compute response has no Ray dashboard URL")
            state["dashboard_url"] = dashboard

            state["phase"] = "submitting_probe"
            _atomic_json(state_path, state)
            active_job = _submit_job(
                jobs,
                phase="probe",
                dashboard=dashboard,
                credentials=credentials,
                working_dir=snapshot,
                plan=plan,
            )
            state.update(
                {
                    "phase": "probe_running",
                    "probe_job": _job_dict(active_job),
                    "probe_submission_id": active_job.submission_id,
                }
            )
            _atomic_json(state_path, state)
            probe_status, probe_logs = _wait_terminal(
                jobs,
                active_job,
                timeout=min(float(profile.max_wait), 43_200),
                poll_interval=float(profile.poll_interval),
                state=state,
                state_path=state_path,
                phase="probe",
            )
            _save_logs(
                state_path.with_name(state_path.stem + "_probe_driver.log"),
                probe_logs,
            )
            if probe_status != "SUCCEEDED":
                raise RuntimeError(f"probe job failed with status {probe_status}")
            probe_result = _marker(probe_logs, PROBE_MARKER)
            _validate_probe(probe_result, plan)
            state["probe_result"] = probe_result
            state["probe_finished_at"] = _utc_now()
            active_job = None

            state["phase"] = "submitting_full"
            _atomic_json(state_path, state)
            active_job = _submit_job(
                jobs,
                phase="full",
                dashboard=dashboard,
                credentials=credentials,
                working_dir=snapshot,
                plan=plan,
            )
            state.update(
                {
                    "phase": "full_submitted",
                    "full_job": _job_dict(active_job),
                    "full_submission_id": active_job.submission_id,
                    "full_submitted_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
            _wait_running(
                jobs,
                active_job,
                timeout=1800,
                poll_interval=float(profile.poll_interval),
                confirmations=2,
                state=state,
                state_path=state_path,
            )
            direct_ready = _wait_for_log_marker(
                jobs,
                active_job,
                marker=HDFS_DIRECT_READY_MARKER,
                timeout=1800,
                poll_interval=float(profile.poll_interval),
                state=state,
                state_path=state_path,
            )
            _validate_full_readiness(direct_ready, plan)
            detached = True
            active_job = None
            state.update(
                {
                    "phase": "detached_running",
                    "status": "running",
                    "detached_at": _utc_now(),
                    "job_retained": True,
                    "compute_retained": True,
                    "cleanup_required_after_completion": True,
                }
            )
            _atomic_json(state_path, state)
            if reuse_state_path is not None and reuse_state is not None:
                current_owner = _load_json(reuse_state_path)
                if (
                    str(current_owner.get("compute_id") or "")
                    != reuse_compute_id
                    or current_owner.get("compute_owned") is not True
                ):
                    raise RuntimeError(
                        "reuse-state compute ownership changed during submission"
                    )
                current_owner.update(
                    {
                        "compute_reused_by_state": str(state_path),
                        "compute_reused_by_run_id": plan["run_id"],
                        "compute_reused_at": _utc_now(),
                    }
                )
                _atomic_json(reuse_state_path, current_owner)
            print(
                "RAYORCH_MINERU_DETACHED "
                + json.dumps(
                    {
                        "compute_id": lease.id,
                        "submission_id": state["full_submission_id"],
                        "state_file": str(state_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            exit_code = 0
        except BaseException as error:
            state.update(
                {
                    "phase": "failed_before_detach",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "failed_at": _utc_now(),
                }
            )
            cancellation_confirmed = True
            if active_job is not None:
                cancellation_confirmed, cancel_status = _cancel_and_confirm(
                    jobs,
                    active_job,
                    poll_interval=float(profile.poll_interval),
                )
                state.update(
                    {
                        "active_job_cancel_confirmed": cancellation_confirmed,
                        "active_job_cancel_status": cancel_status,
                    }
                )
            if lease is not None and lease.owned and cancellation_confirmed:
                state["failure_compute_cleanup"] = cleanup_manager.release(
                    lease, on_event=compute_event
                )
            elif lease is not None and lease.owned:
                state.update(
                    {
                        "compute_retained": True,
                        "manual_cleanup_required": True,
                        "cleanup_reason": "active job cancellation was not confirmed",
                    }
                )
            _atomic_json(state_path, state)
            if isinstance(error, KeyboardInterrupt):
                raise
            exit_code = 1
        finally:
            if detached:
                state["finalizer_action"] = "retain_running_job_and_compute"
                _atomic_json(state_path, state)
    return exit_code


def resume_existing(state_path: Path) -> int:
    """Adopt jobs left alive after the local submit supervisor disconnects.

    The state file and a non-blocking file lock make this idempotent: a retry
    adopts an already-submitted full job instead of creating a duplicate.
    """

    from hydp_dataflow_wedata.config import load_profile
    from hydp_engine.ray_jobs import RayJobClient

    lock_path = state_path.with_name(state_path.stem + ".supervisor.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValidationError(
                f"another resume supervisor already owns {lock_path}"
            ) from error

        state = _load_json(state_path)
        plan = state.get("plan")
        probe_payload = state.get("probe_job")
        if not isinstance(plan, Mapping) or not isinstance(
            probe_payload, Mapping
        ):
            raise ValidationError("state does not contain a plan and probe job")
        if state.get("status") == "completed":
            return 0

        profile_reference = str(state.get("profile_reference") or "")
        profile = load_profile(profile_reference)
        if _fingerprint(profile.compute_spec()) != plan["compute"][
            "spec_fingerprint"
        ]:
            raise ValidationError(
                "profile compute spec differs from the persisted plan"
            )
        credentials = profile.credentials()
        jobs = RayJobClient()
        probe_job = _restore_job(probe_payload)
        full_payload = state.get("full_job")
        full_job = (
            _restore_job(full_payload)
            if isinstance(full_payload, Mapping)
            else None
        )

        try:
            if full_job is not None:
                full_status = str(
                    jobs.status(full_job).get("status") or "UNKNOWN"
                ).upper()
                if full_status in TERMINAL_JOB_STATES:
                    return inspect_status(state_path)
                if state.get("phase") == "detached_running":
                    state.update(
                        {
                            "status": "running",
                            "full_status": full_status,
                            "resumed_status_checked_at": _utc_now(),
                        }
                    )
                    _atomic_json(state_path, state)
                    return 0
            else:
                probe_status = str(
                    jobs.status(probe_job).get("status") or "UNKNOWN"
                ).upper()
                if probe_status in TERMINAL_JOB_STATES:
                    probe_logs = jobs.logs(probe_job, tail=12_000)
                else:
                    probe_status, probe_logs = _wait_terminal(
                        jobs,
                        probe_job,
                        timeout=min(float(profile.max_wait), 43_200),
                        poll_interval=float(profile.poll_interval),
                        state=state,
                        state_path=state_path,
                        phase="probe",
                    )
                _save_logs(
                    state_path.with_name(
                        state_path.stem + "_probe_driver.log"
                    ),
                    probe_logs,
                )
                if probe_status != "SUCCEEDED":
                    raise RuntimeError(
                        f"probe job failed with status {probe_status}"
                    )
                probe_result = _marker(probe_logs, PROBE_MARKER)
                _validate_probe(probe_result, plan)
                state.update(
                    {
                        "probe_result": probe_result,
                        "probe_finished_at": _utc_now(),
                        "phase": "submitting_full",
                    }
                )
                _atomic_json(state_path, state)

                runtime_metadata = probe_payload.get("metadata")
                if not isinstance(runtime_metadata, Mapping) or not isinstance(
                    runtime_metadata.get("runtime_env"), Mapping
                ):
                    raise RuntimeError(
                        "probe state does not retain its runtime_env"
                    )
                runtime_env = copy.deepcopy(
                    dict(runtime_metadata["runtime_env"])
                )
                full_job = jobs.submit_proxy(
                    entrypoint=(
                        "python -u taiji_multinode_mineru/remote_driver.py "
                        "--phase full"
                    ),
                    submission_id=(
                        f"rayorch-mineru-v36-full-{plan['run_id']}-"
                        + uuid.uuid4().hex[:8]
                    )[:128],
                    dashboard_url=str(state["dashboard_url"]),
                    runtime_env=runtime_env,
                    working_dir=None,
                    runtime_auth=True,
                    user=credentials.user,
                    cmk=credentials.cmk,
                    metadata={
                        "hydp_stage": "rayorch_mineru_v36_full",
                        "run_id": str(plan["run_id"]),
                        "source_digest": str(plan["source_digest"]),
                    },
                )
                state.update(
                    {
                        "phase": "full_submitted",
                        "full_job": _job_dict(full_job),
                        "full_submission_id": full_job.submission_id,
                        "full_submitted_at": _utc_now(),
                        "resumed_by_supervisor": True,
                    }
                )
                _atomic_json(state_path, state)

            _wait_running(
                jobs,
                full_job,
                timeout=1800,
                poll_interval=float(profile.poll_interval),
                confirmations=2,
                state=state,
                state_path=state_path,
            )
            readiness = _wait_for_log_marker(
                jobs,
                full_job,
                marker=HDFS_DIRECT_READY_MARKER,
                timeout=1800,
                poll_interval=float(profile.poll_interval),
                state=state,
                state_path=state_path,
            )
            _validate_full_readiness(readiness, plan)
            state.update(
                {
                    "phase": "detached_running",
                    "status": "running",
                    "detached_at": _utc_now(),
                    "job_retained": True,
                    "compute_retained": True,
                    "cleanup_required_after_completion": True,
                }
            )
            _atomic_json(state_path, state)
            print(
                "RAYORCH_MINERU_DETACHED "
                + json.dumps(
                    {
                        "compute_id": state["compute_id"],
                        "submission_id": state["full_submission_id"],
                        "state_file": str(state_path),
                        "resumed": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                state.update(
                    {
                        "supervisor_interrupted_at": _utc_now(),
                        "active_job_retained": True,
                    }
                )
                _atomic_json(state_path, state)
                raise
            state.update(
                {
                    "phase": "failed_before_detach",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "failed_at": _utc_now(),
                }
            )
            if full_job is not None:
                confirmed, cancel_status = _cancel_and_confirm(
                    jobs,
                    full_job,
                    poll_interval=float(profile.poll_interval),
                )
                state.update(
                    {
                        "active_job_cancel_confirmed": confirmed,
                        "active_job_cancel_status": cancel_status,
                    }
                )
            _atomic_json(state_path, state)
            raise


def inspect_status(state_path: Path) -> int:
    from hydp_engine.ray_jobs import RayJobClient

    state = _load_json(state_path)
    job_payload = state.get("full_job")
    if not isinstance(job_payload, Mapping):
        raise ValidationError("state does not contain a submitted full job")
    job = _restore_job(job_payload)
    jobs = RayJobClient()
    detail = jobs.status(job)
    status = str(detail.get("status") or "UNKNOWN").upper()
    logs = jobs.logs(job, tail=1000)
    _save_logs(state_path.with_name(state_path.stem + "_full_driver.log"), logs)
    state.update({"full_status": status, "status_checked_at": _utc_now()})
    exit_code = 0
    if status in TERMINAL_JOB_STATES:
        try:
            result = _marker(logs, FULL_MARKER)
            if result.get("status") != "completed" or not result.get("success_uri"):
                raise RuntimeError(f"invalid MinerU completion payload: {result}")
            state.update(
                {
                    "full_result": result,
                    "status": "completed",
                    "phase": "full_completed",
                }
            )
            state.pop("full_result_error", None)
        except Exception as error:
            state["full_result_error"] = f"{type(error).__name__}: {error}"
            state.update({"status": "failed", "phase": "full_failed"})
            exit_code = 1
    else:
        state["status"] = "running"
    _atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "submission_id": job.submission_id,
                "platform_status": status,
                "business_status": state["status"],
            },
            sort_keys=True,
        )
    )
    return exit_code


def cleanup(
    state_path: Path,
    *,
    profile_reference: str,
    confirm_compute_id: str,
) -> int:
    """Cancel a retained job, then release only its recorded owned compute."""

    from hydp_dataflow_wedata.compute import ComputeLease, ComputeManager
    from hydp_dataflow_wedata.config import load_profile
    from hydp_dataflow_wedata.events import console
    from hydp_engine import HydpEngineClient
    from hydp_engine.ray_jobs import RayJobClient

    state = _load_json(state_path)
    compute_id = str(state.get("compute_id") or "")
    if not compute_id:
        raise ValidationError("state does not record a compute")
    if confirm_compute_id != compute_id:
        raise ValidationError("--confirm-compute-id must exactly match the state")
    jobs = RayJobClient()
    reused_by = state.get("compute_reused_by_state")
    if state.get("compute_owned") is True and reused_by:
        child_path = Path(str(reused_by))
        child = _load_json(child_path)
        child_payload = child.get("full_job")
        if isinstance(child_payload, Mapping):
            child_status = str(
                jobs.status(_restore_job(child_payload)).get("status") or "UNKNOWN"
            ).upper()
            if child_status not in TERMINAL_JOB_STATES:
                raise ValidationError(
                    "compute is reused by an active job; stop it through "
                    f"{child_path} before releasing compute"
                )
    job_kind = "full"
    job_payload = state.get("full_job")
    if not isinstance(job_payload, Mapping):
        job_kind = "probe"
        job_payload = state.get("probe_job")
    if isinstance(job_payload, Mapping):
        recorded_status = str(
            state.get("manual_job_cancel_status") or "UNKNOWN"
        ).upper()
        recorded_confirmation = (
            state.get("manual_job_kind") == job_kind
            and state.get("manual_job_cancel_confirmed") is True
            and recorded_status in TERMINAL_JOB_STATES
        )
        if recorded_confirmation:
            confirmed, status = True, recorded_status
        else:
            job = _restore_job(job_payload)
            confirmed, status = _cancel_and_confirm(
                jobs, job, poll_interval=10
            )
        state.update(
            {
                "manual_job_kind": job_kind,
                "manual_job_cancel_confirmed": confirmed,
                "manual_job_cancel_status": status,
                f"{job_kind}_status": status,
            }
        )
        _atomic_json(state_path, state)
        if not confirmed:
            state["manual_cleanup_required"] = True
            _atomic_json(state_path, state)
            raise RuntimeError("job cancellation could not be confirmed; compute retained")
    if state.get("compute_owned") is not True:
        result = {
            "released": False,
            "retained": True,
            "reason": "borrowed_compute",
            "compute_id": compute_id,
        }
        state.update(
            {
                "cleanup_finished_at": _utc_now(),
                "status": "job_stopped_compute_retained",
                "manual_compute_cleanup": result,
                "delete_hdfs_data": False,
            }
        )
        _atomic_json(state_path, state)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    profile = load_profile(profile_reference)
    credentials = profile.credentials()
    engine = HydpEngineClient(
        user=credentials.user,
        cmk=credentials.cmk,
        cmk_id=credentials.cmk_id,
    )
    manager = ComputeManager(engine, replace(profile, keep_compute=False))
    result = manager.release(
        ComputeLease(compute_id, owned=True),
        on_event=console,
    )
    state.update(
        {
            "manual_compute_cleanup": result,
            "cleanup_finished_at": _utc_now(),
            "status": "cleaned" if result.get("released") else "cleanup_failed",
            "delete_hdfs_data": False,
        }
    )
    _atomic_json(state_path, state)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("released") else 1


def stop_job_keep_compute(
    state_path: Path,
    *,
    confirm_compute_id: str,
) -> int:
    """Cancel one retained full Job while deliberately keeping its compute."""

    from hydp_engine.ray_jobs import RayJobClient

    state = _load_json(state_path)
    compute_id = str(state.get("compute_id") or "")
    if not compute_id or confirm_compute_id != compute_id:
        raise ValidationError("--confirm-compute-id must exactly match the state")
    job_payload = state.get("full_job")
    if not isinstance(job_payload, Mapping):
        raise ValidationError("state does not record a full Ray Job")
    confirmed, status = _cancel_and_confirm(
        RayJobClient(),
        _restore_job(job_payload),
        poll_interval=10,
    )
    state.update(
        {
            "manual_job_cancel_confirmed": confirmed,
            "manual_job_cancel_status": status,
            "full_status": status,
            "status": (
                "job_stopped_compute_retained"
                if confirmed
                else "manual_cleanup_required"
            ),
            "stop_finished_at": _utc_now(),
            "compute_retained": True,
            "delete_hdfs_data": False,
        }
    )
    _atomic_json(state_path, state)
    result = {
        "cancel_confirmed": confirmed,
        "job_status": status,
        "compute_id": compute_id,
        "compute_retained": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if confirmed else 1


def _state_default(run_id: str) -> Path:
    return EXPERIMENT_DIR / "state" / f"{run_id}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "submit"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-id", required=True)
        child.add_argument(
            "--engine", choices=("rayorch", "raydata"), default="rayorch"
        )
        child.add_argument("--input-uri", action="append", dest="input_uris")
        child.add_argument("--input-limit", action="append", type=int)
        child.add_argument("--hyperparameters-json")
        child.add_argument("--benchmark-only", action="store_true")
        child.add_argument("--output-uri")
        child.add_argument("--hdfs-model-uri", default=DEFAULT_MODEL_URI)
        child.add_argument("--profile", default=DEFAULT_PROFILE_REF)
        child.add_argument("--state-file", type=Path)
        if command == "submit":
            child.add_argument(
                "--reuse-state-file",
                type=Path,
                help="reuse the owned compute recorded by a terminal prior run",
            )
    status = subparsers.add_parser("status")
    status.add_argument("--state-file", type=Path, required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--state-file", type=Path, required=True)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--state-file", type=Path, required=True)
    stop.add_argument("--confirm-compute-id", required=True)
    clean = subparsers.add_parser("cleanup")
    clean.add_argument("--state-file", type=Path, required=True)
    clean.add_argument("--profile", default=DEFAULT_PROFILE_REF)
    clean.add_argument("--confirm-compute-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"plan", "submit"}:
        plan = build_plan(
            run_id=args.run_id,
            input_uris=args.input_uris or DEFAULT_INPUT_URIS,
            output_uri=args.output_uri,
            model_uri=args.hdfs_model_uri,
            input_limits=args.input_limit,
            hyperparameters=(
                json.loads(args.hyperparameters_json)
                if args.hyperparameters_json
                else HYPERPARAMETERS
            ),
            benchmark_only=args.benchmark_only,
            engine=args.engine,
            profile_reference=args.profile,
        )
        state_path = args.state_file or _state_default(plan["run_id"])
        if args.command == "plan":
            print(
                json.dumps(
                    {**plan, "state_file": str(state_path), "external_actions": False},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return execute_submit(
            plan,
            profile_reference=args.profile,
            state_path=state_path,
            reuse_state_path=args.reuse_state_file,
        )
    if args.command == "status":
        return inspect_status(args.state_file)
    if args.command == "resume":
        return resume_existing(args.state_file)
    if args.command == "stop":
        return stop_job_keep_compute(
            args.state_file,
            confirm_compute_id=args.confirm_compute_id,
        )
    return cleanup(
        args.state_file,
        profile_reference=args.profile,
        confirm_compute_id=args.confirm_compute_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APP_GROUP",
    "EXPECTED_GPUS_PER_WORKER",
    "EXPECTED_PDFS",
    "EXPECTED_WORKERS",
    "GPU_TYPE",
    "ValidationError",
    "build_parser",
    "build_plan",
    "cleanup",
    "execute_submit",
    "resume_existing",
    "inspect_status",
    "main",
    "source_snapshot",
    "stop_job_keep_compute",
    "validate_compute_spec",
    "validate_profile",
]
