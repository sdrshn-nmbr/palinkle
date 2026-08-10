"""Disposable TPU worker and controller for Phase 3.1 submissions."""

from __future__ import annotations

import argparse
import ipaddress
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opjax.pallas.jaxbench_executable import file_sha256
from opjax.pallas.jaxbench_worker import (
    GcloudDisposableTPUBackend,
    JaxBenchWorkerError,
    _write_result,
    canonical_sha256,
    compile_in_sandbox,
    materialize_submission,
    prepare_sandbox_parent,
    validate_response,
    write_request,
)
from opjax.pallas.phase31_verifier import verify_serialized_submission


TRUSTED_SOURCES = (
    "__init__.py",
    "benchmarking.py",
    "jaxbench_executable.py",
    "jaxbench_verifier.py",
    "jaxbench_worker.py",
    "phase31_oracle.py",
    "phase31_verifier.py",
    "phase31_worker.py",
)


def grade_worker_submission(
    *,
    release_root: Path,
    request_path: Path,
    patch_path: Path,
    out_dir: Path,
    sandboxed: bool = True,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchWorkerError(f"OUTPUT_EXISTS:{out_dir}")
    release = json.loads((release_root / "manifest.json").read_text())
    request = json.loads(request_path.read_text())
    payload = dict(request)
    if canonical_sha256({key: value for key, value in request.items() if key != "request_sha256"}) != payload.pop("request_sha256", None):
        raise JaxBenchWorkerError("REQUEST_HASH_INVALID")
    task_record = next(
        (task for task in release["tasks"] if task["task_id"] == request["task_id"]),
        None,
    )
    if task_record is None or any(
        request.get(key) != expected
        for key, expected in (
            ("release_sha256", release["release_sha256"]),
            ("task_sha256", task_record["task_sha256"]),
            ("patch_sha256", file_sha256(patch_path)),
        )
    ):
        raise JaxBenchWorkerError("REQUEST_BINDING_INVALID")
    task_root = release_root / task_record["path"]
    task = json.loads((task_root / "tests/task.json").read_text())
    out_dir.mkdir(parents=True)
    shutil.copy2(patch_path, out_dir / "model.patch")
    with tempfile.TemporaryDirectory(prefix="opjax-phase31-worker-") as temporary:
        root = Path(temporary)
        prepare_sandbox_parent(root, sandboxed=sandboxed)
        materialized = materialize_submission(
            task_root=task_root,
            patch_path=patch_path,
            destination=root / "workspace",
        )
        compile_record, compiled_dir = compile_in_sandbox(
            task=task,
            workspace=root / "workspace",
            candidate_root=root / "candidate",
            sandboxed=sandboxed,
        )
        (out_dir / "compile.log").write_text(
            compile_record.get("stdout", "") + compile_record.get("stderr", "")
        )
        if not compile_record.get("passed"):
            result = {
                "schema_version": 2,
                "task_id": task["task_id"],
                "passed": False,
                "stage": "tpu_compile",
                "error": compile_record.get("error", "CANDIDATE_COMPILE_FAILED"),
                "candidate_attributable": True,
                "infrastructure_error": False,
                "correct": False,
                "authentic": False,
                "profiled": False,
                "reward": 0,
                "kernel_sha256": materialized["kernel_sha256"],
            }
        else:
            try:
                result = verify_serialized_submission(
                    task=task,
                    baseline_path=task_root / "tests/jaxbench/baseline.py",
                    compiled_dir=compiled_dir,
                    out_dir=out_dir / "verification",
                )
            except Exception as exc:
                result = {
                    "schema_version": 2,
                    "task_id": task["task_id"],
                    "passed": False,
                    "stage": "infrastructure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate_attributable": False,
                    "infrastructure_error": True,
                    "correct": False,
                    "authentic": False,
                    "profiled": False,
                    "reward": -1,
                }
            result["kernel_sha256"] = materialized["kernel_sha256"]
        _write_result(out_dir, result)
    response = {
        **request,
        "kernel_sha256": result.get("kernel_sha256"),
        "result_sha256": file_sha256(out_dir / "result.json"),
        "reward_sha256": file_sha256(out_dir / "reward.json"),
        "model_patch_sha256": file_sha256(out_dir / "model.patch"),
    }
    (out_dir / "submission.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    return response


def build_worker_bundle(
    *, release_root: Path, task_id: str, destination: Path
) -> Path:
    release = json.loads((release_root / "manifest.json").read_text())
    task = next((item for item in release["tasks"] if item["task_id"] == task_id), None)
    if task is None:
        raise JaxBenchWorkerError(f"SUBMISSION_TASK_UNKNOWN:{task_id}")
    target = destination / "release"
    task_target = target / task["path"]
    task_target.parent.mkdir(parents=True)
    shutil.copytree(release_root / task["path"], task_target)
    (task_target / "tests/jaxbench/optimized.py").unlink(missing_ok=True)
    trusted = target / "trusted-src/opjax/pallas"
    trusted.mkdir(parents=True)
    (trusted.parent / "__init__.py").write_text("")
    source_root = Path(__file__).parent
    for name in TRUSTED_SOURCES:
        shutil.copy2(source_root / name, trusted / name)
    lock = Path(__file__).parents[3] / "config/pallas/phase2-worker-requirements.lock"
    shutil.copy2(lock, target / "worker-requirements.lock")
    sanitized = {
        "schema_version": 1,
        "kind": "opjax_phase31_sanitized_worker_bundle",
        "release_sha256": release["release_sha256"],
        "tasks": [task],
    }
    (target / "manifest.json").write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    )
    return target


def grade_on_gcloud(
    *,
    release_root: Path,
    request: dict[str, Any],
    patch_path: Path,
    destination: Path,
    service_account: str,
    zone: str = "us-west4-a",
) -> dict[str, Any]:
    backend = GcloudDisposableTPUBackend(
        release_root=release_root,
        patch_path=patch_path,
        service_account=service_account,
        zone=zone,
        name_prefix="opjax-p31",
    )
    worker = f"opjax-p31-{request['request_sha256'][:8]}-{uuid.uuid4().hex[:6]}"
    response: dict[str, Any] | None = None
    try:
        backend._run(
            [
                "gcloud", "compute", "tpus", "tpu-vm", "create", worker,
                f"--zone={zone}", "--accelerator-type=v5litepod-1",
                "--version=tpu-ubuntu2204-base", f"--service-account={service_account}",
                "--scopes=https://www.googleapis.com/auth/logging.write", "--quiet",
            ]
        )
        address = backend._run(
            ["gcloud", "compute", "tpus", "tpu-vm", "describe", worker,
             f"--zone={zone}", "--format=value(networkEndpoints[0].ipAddress)"]
        ).stdout.strip()
        if not ipaddress.ip_address(address).is_private:
            raise JaxBenchWorkerError("WORKER_ADDRESS_INVALID")
        with tempfile.TemporaryDirectory(prefix="opjax-phase31-request-") as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            write_request(request_path, request)
            bundle = build_worker_bundle(
                release_root=release_root,
                task_id=request["task_id"],
                destination=root / "bundle",
            )
            remote = "/tmp/opjax-phase31-submission"
            backend._run(
                ["gcloud", "compute", "tpus", "tpu-vm", "ssh", worker,
                 f"--zone={zone}", "--command", f"sudo rm -rf {remote} && mkdir -p {remote}/input"]
            )
            for source in (bundle.parent, patch_path, request_path):
                command = ["gcloud", "compute", "tpus", "tpu-vm", "scp"]
                if source.is_dir():
                    command.append("--recurse")
                command.extend([str(source), f"{worker}:{remote}/input/", f"--zone={zone}"])
                backend._run(command)
            remote_release = "input/bundle/release"
            command = (
                "python3 -m pip install --user uv==0.7.19 >/dev/null && "
                f"cd {remote} && UV_PYTHON_INSTALL_DIR=worker-python "
                "/home/$USER/.local/bin/uv python install 3.12.11 && "
                "/home/$USER/.local/bin/uv venv --python "
                "worker-python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12 worker-venv && "
                "/home/$USER/.local/bin/uv pip install --python worker-venv/bin/python "
                f"--requirements {remote_release}/worker-requirements.lock && "
                "chmod -R a+rX worker-python worker-venv && chmod -R go-rwx input && "
                "env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=tpu TPU_SKIP_MDS_QUERY=1 "
                "TPU_ACCELERATOR_TYPE=v5litepod-1 TPU_WORKER_ID=0 "
                f"TPU_WORKER_HOSTNAMES={address} TPU_PROCESS_BOUNDS=1,1,1 "
                "TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 "
                "LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 "
                f"PYTHONPATH={remote_release}/trusted-src worker-venv/bin/python "
                "-m opjax.pallas.phase31_worker grade-worker "
                f"--release {remote_release} --request input/request.json "
                f"--patch input/{patch_path.name} --out output"
            )
            backend._run(
                ["gcloud", "compute", "tpus", "tpu-vm", "ssh", worker,
                 f"--zone={zone}", "--command", command], timeout=3600
            )
            download_parent = root / "download"
            download_parent.mkdir()
            backend._run(
                ["gcloud", "compute", "tpus", "tpu-vm", "scp", "--recurse",
                 f"{worker}:{remote}/output", str(download_parent), f"--zone={zone}"]
            )
            downloaded = download_parent / "output"
            downloaded.rename(destination)
            response = json.loads((destination / "submission.json").read_text())
    finally:
        subprocess.run(
            ["gcloud", "compute", "tpus", "tpu-vm", "delete", worker,
             f"--zone={zone}", "--async", "--quiet"], capture_output=True, check=False
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["gcloud", "compute", "tpus", "tpu-vm", "describe", worker,
                 f"--zone={zone}"], capture_output=True, check=False
            )
            if probe.returncode != 0:
                break
            time.sleep(10)
    if response is None:
        raise JaxBenchWorkerError("PHASE31_RESPONSE_MISSING")
    response["worker"] = {
        "identity": worker,
        "disposable": True,
        "destroyed_at": datetime.now(timezone.utc).isoformat(),
        "candidate_user": "nobody",
        "execution_boundary": "sandbox-compile-serialized-executable-pristine-verify",
        "zone": zone,
        "accelerator_type": "v5litepod-1",
    }
    (destination / "submission.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n"
    )
    return validate_response(request=request, destination=destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-worker")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("grade-worker")
    worker.add_argument("--release", type=Path, required=True)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--patch", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    args.pop("command")
    try:
        result = grade_worker_submission(
            release_root=args["release"], request_path=args["request"],
            patch_path=args["patch"], out_dir=args["out"]
        )
    except Exception as exc:
        print(f"PHASE31_WORKER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
