"""Small, fail-closed GPU profiling and artifact publishing utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Protocol, TypeVar

from huggingface_hub import HfApi
import torch

from opjax.remote.gpu_runtime_identity import gpu_runtime_identity


__all__ = ["gpu_runtime_identity"]


T = TypeVar("T")
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class NvtxApi(Protocol):
    def range_push(self, name: str) -> None: ...

    def range_pop(self) -> None: ...


class CudaProfilerApi(Protocol):
    def cudaProfilerStart(self) -> object: ...

    def cudaProfilerStop(self) -> object: ...


class GpuStage(StrEnum):
    TARGET_FEATURES = "target_features"
    DRAFT_LAYER = "draft_layer"
    ATTENTION = "attention"
    MARKOV_CORRECTION = "markov_correction"
    VERIFICATION = "verification"
    CACHE_UPDATE = "cache_update"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def nvtx_range(
    name: str, *, nvtx: NvtxApi | None = None
) -> Iterator[None]:
    if not name or name.strip() != name:
        raise ValueError(f"GPU_NVTX_RANGE_INVALID:{name!r}")
    api = torch.cuda.nvtx if nvtx is None else nvtx
    api.range_push(name)
    try:
        yield
    finally:
        api.range_pop()


@contextmanager
def model_stage(
    stage: GpuStage,
    *,
    layer: int | None = None,
    nvtx: NvtxApi | None = None,
) -> Iterator[None]:
    if stage is GpuStage.DRAFT_LAYER:
        if layer is None or layer < 0:
            raise ValueError(f"GPU_DRAFT_LAYER_INVALID:{layer}")
        name = f"opjax.{stage.value}.{layer}"
    else:
        if layer is not None:
            raise ValueError(f"GPU_STAGE_LAYER_FORBIDDEN:{stage.value}:{layer}")
        name = f"opjax.{stage.value}"
    with nvtx_range(name, nvtx=nvtx):
        yield


@contextmanager
def cuda_capture(
    *, runtime: CudaProfilerApi | None = None
) -> Iterator[None]:
    api = torch.cuda.cudart() if runtime is None else runtime
    api.cudaProfilerStart()
    try:
        yield
    finally:
        api.cudaProfilerStop()


def warm_then_capture(
    *,
    warmup: Callable[[], object],
    workload: Callable[[], T],
    warmup_steps: int,
    range_name: str,
    synchronize: Callable[[], None] = torch.cuda.synchronize,
    runtime: CudaProfilerApi | None = None,
    nvtx: NvtxApi | None = None,
) -> T:
    if warmup_steps < 1:
        raise ValueError(f"GPU_WARMUP_STEPS_INVALID:{warmup_steps}")
    for _ in range(warmup_steps):
        warmup()
    synchronize()
    with cuda_capture(runtime=runtime):
        with nvtx_range(range_name, nvtx=nvtx):
            result = workload()
            synchronize()
    return result


@dataclass(frozen=True)
class NsysProfile:
    run_root: Path
    target: tuple[str, ...]
    traces: tuple[str, ...] = ("cuda", "nvtx", "osrt")
    capture_range: str = "cudaProfilerApi"
    capture_range_end: str = "stop"
    timeout_seconds: int = 3600

    def command(self) -> tuple[str, ...]:
        if not self.target:
            raise ValueError("GPU_NSYS_TARGET_EMPTY")
        if self.capture_range != "cudaProfilerApi":
            raise ValueError(
                f"GPU_NSYS_CAPTURE_RANGE_UNSUPPORTED:{self.capture_range}"
            )
        return (
            "nsys",
            "profile",
            f"--trace={','.join(self.traces)}",
            "--sample=none",
            "--cpuctxsw=none",
            f"--capture-range={self.capture_range}",
            f"--capture-range-end={self.capture_range_end}",
            "--export=sqlite",
            "--force-overwrite=false",
            f"--output={self.run_root / 'profile'}",
            *self.target,
        )


def _run_command(
    command: tuple[str, ...],
    *,
    run: CommandRunner,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        result = run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"GPU_TOOL_MISSING:{command[0]}") from error
    return result


def _write_command_receipt(
    path: Path, result: subprocess.CompletedProcess[str]
) -> None:
    path.write_text(
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}"
        f"stderr:\n{result.stderr}",
        encoding="utf-8",
    )


def run_nsys_profile(
    profile: NsysProfile,
    *,
    runtime: Mapping[str, Any],
    run: CommandRunner = subprocess.run,
) -> Path:
    root = profile.run_root
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError(f"GPU_NSYS_RUN_NOT_EMPTY:{root}")

    status = _run_command(
        ("nsys", "status", "--environment"),
        run=run,
        timeout=120,
    )
    _write_command_receipt(root / "nsys-environment.txt", status)

    version = _run_command(("nsys", "--version"), run=run, timeout=30)
    _write_command_receipt(root / "nsys-version.txt", version)
    if version.returncode != 0:
        raise RuntimeError(f"GPU_NSYS_VERSION_FAILED:{version.returncode}")

    gpu = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        run=run,
        timeout=30,
    )
    _write_command_receipt(root / "nvidia-smi.txt", gpu)
    if gpu.returncode != 0:
        raise RuntimeError(f"GPU_NVIDIA_SMI_FAILED:{gpu.returncode}")

    result = _run_command(
        profile.command(), run=run, timeout=profile.timeout_seconds
    )
    (root / "profiler.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / "profiler.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GPU_NSYS_PROFILE_FAILED:{result.returncode}:{root}")

    required = (root / "profile.nsys-rep", root / "profile.sqlite")
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"GPU_NSYS_ARTIFACT_MISSING:{path}")

    runtime_record = {
        "schema_version": 1,
        "command": list(profile.command()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "runtime": dict(runtime),
    }
    (root / "runtime.json").write_text(
        json.dumps(runtime_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return write_artifact_manifest(root, runtime=runtime_record)


def write_artifact_manifest(
    root: Path, *, runtime: Mapping[str, Any]
) -> Path:
    root = root.resolve()
    manifest_path = root / "artifact-manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"GPU_ARTIFACT_SET_EMPTY:{root}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "runtime": dict(runtime),
        "files": files,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    validate_artifact_manifest(manifest_path)
    return manifest_path


def validate_artifact_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if expected != _canonical_sha256(unsigned):
        raise RuntimeError("GPU_ARTIFACT_MANIFEST_HASH_MISMATCH")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("GPU_ARTIFACT_MANIFEST_FILES_INVALID")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise RuntimeError(f"GPU_ARTIFACT_PATH_INVALID:{relative}")
        seen.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"GPU_ARTIFACT_PATH_ESCAPE:{relative}")
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"GPU_ARTIFACT_FILE_MISSING:{relative}")
        if _file_sha256(path) != item.get("sha256"):
            raise RuntimeError(f"GPU_ARTIFACT_HASH_MISMATCH:{relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        raise RuntimeError(
            "GPU_ARTIFACT_UNBOUND_FILES:"
            f"missing={sorted(seen - actual)}:extra={sorted(actual - seen)}"
        )
    return manifest


def publish_hf(
    root: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    api: HfApi | None = None,
) -> dict[str, str]:
    manifest = validate_artifact_manifest(root / "artifact-manifest.json")
    client = HfApi() if api is None else api
    client.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    commit = client.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        path_in_repo=path_in_repo,
        commit_message=f"Add GPU trace {manifest['manifest_sha256']}",
    )
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("GPU_HF_COMMIT_REVISION_MISSING")
    return {
        "revision": revision,
        "manifest_sha256": manifest["manifest_sha256"],
        "locator": f"hf://datasets/{repo_id}@{revision}/{path_in_repo}",
    }


def publish_r2(
    root: Path,
    *,
    bucket: str,
    prefix: str,
    run: CommandRunner = subprocess.run,
) -> dict[str, str]:
    manifest_path = root / "artifact-manifest.json"
    manifest = validate_artifact_manifest(manifest_path)
    normalized_prefix = prefix.strip("/")
    if not bucket or not normalized_prefix:
        raise ValueError("GPU_R2_DESTINATION_INVALID")

    uploaded: set[str] = set()
    for item in manifest["files"]:
        digest = item["sha256"]
        if digest in uploaded:
            continue
        uploaded.add(digest)
        key = f"{normalized_prefix}/blobs/{digest}"
        result = _run_command(
            (
                "wrangler",
                "r2",
                "object",
                "put",
                f"{bucket}/{key}",
                "--file",
                str(root / item["path"]),
                "--remote",
            ),
            run=run,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"GPU_R2_BLOB_UPLOAD_FAILED:{digest}")

    manifest_key = (
        f"{normalized_prefix}/manifests/{manifest['manifest_sha256']}.json"
    )
    result = _run_command(
        (
            "wrangler",
            "r2",
            "object",
            "put",
            f"{bucket}/{manifest_key}",
            "--file",
            str(manifest_path),
            "--remote",
        ),
        run=run,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError("GPU_R2_MANIFEST_UPLOAD_FAILED")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "locator": f"r2://{bucket}/{manifest_key}",
    }


__all__ = [
    "GpuStage",
    "NsysProfile",
    "cuda_capture",
    "model_stage",
    "nvtx_range",
    "publish_hf",
    "publish_r2",
    "run_nsys_profile",
    "validate_artifact_manifest",
    "warm_then_capture",
    "write_artifact_manifest",
]
