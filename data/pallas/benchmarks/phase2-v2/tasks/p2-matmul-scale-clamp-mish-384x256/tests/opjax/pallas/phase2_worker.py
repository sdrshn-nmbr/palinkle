"""Content-addressed boundary for disposable TPU submission grading.

The controller never imports candidate code. A backend receives a task package
and patch, creates one worker, grades one submission, validates the returned
hashes, and destroys the worker even after candidate or infrastructure failure.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from opjax.pallas.g42_harness import canonical_sha256, file_sha256


class Phase2WorkerError(RuntimeError):
    pass


class SubmissionBackend(Protocol):
    def grade(self, request: dict[str, Any], destination: Path) -> dict[str, Any]: ...


def build_submission_request(
    *, release_manifest: dict[str, Any], task: dict[str, Any], patch_path: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "release_sha256": release_manifest["release_sha256"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "patch_sha256": file_sha256(patch_path),
        "worker_policy": "one-submission-one-disposable-tpu-vm",
    }
    return {**payload, "request_sha256": canonical_sha256(payload)}


def validate_submission_response(
    *, request: dict[str, Any], destination: Path
) -> dict[str, Any]:
    response_path = destination / "submission.json"
    if not response_path.is_file():
        raise Phase2WorkerError("TPU_RESPONSE_MISSING")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if any(
        response.get(key) != request[key]
        for key in ("release_sha256", "task_id", "task_sha256", "patch_sha256")
    ):
        raise Phase2WorkerError("TPU_RESPONSE_BINDING_INVALID")
    for name, hash_field in (
        ("result.json", "result_sha256"),
        ("reward.json", "reward_sha256"),
    ):
        path = destination / name
        if not path.is_file() or file_sha256(path) != response.get(hash_field):
            raise Phase2WorkerError(f"TPU_RESPONSE_ARTIFACT_INVALID:{name}")
    worker = response.get("worker")
    if (
        not isinstance(worker, dict)
        or worker.get("disposable") is not True
        or not worker.get("identity")
        or not worker.get("destroyed_at")
        or worker.get("candidate_user") != "nobody"
        or worker.get("sandbox_policy") != "systemd-cgroup-v1"
        or not worker.get("service_account")
    ):
        raise Phase2WorkerError("TPU_WORKER_LIFECYCLE_UNPROVEN")
    return response


@dataclass(frozen=True)
class DisposableWorkerFactory:
    create: Callable[[dict[str, Any]], str]
    grade: Callable[[str, dict[str, Any], Path], dict[str, Any]]
    destroy: Callable[[str], str]
    worker_evidence: dict[str, Any] = field(default_factory=dict)

    def run(self, request: dict[str, Any], destination: Path) -> dict[str, Any]:
        identity = self.create(request)
        destroyed_at = None
        try:
            response = self.grade(identity, request, destination)
        finally:
            destroyed_at = self.destroy(identity)
        response["worker"] = {
            "identity": identity,
            "disposable": True,
            "destroyed_at": destroyed_at,
            **self.worker_evidence,
        }
        (destination / "submission.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return validate_submission_response(request=request, destination=destination)


def write_request(path: Path, request: dict[str, Any]) -> None:
    payload = dict(request)
    expected = payload.pop("request_sha256", None)
    if canonical_sha256(payload) != expected:
        raise Phase2WorkerError("TPU_REQUEST_HASH_INVALID")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install_worker_output(
    *, destination: Path, download: Callable[[Path], None]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-download-", dir=destination.parent
    ) as temporary:
        download_root = Path(temporary)
        download(download_root)
        copied = download_root / "output"
        if not copied.is_dir() or destination.exists():
            raise Phase2WorkerError("TPU_RESPONSE_DOWNLOAD_INVALID")
        shutil.move(str(copied), str(destination))


def build_worker_bundle(
    release_root: Path, destination: Path, *, task_id: str
) -> Path:
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    bundled_release = (
        destination
        / "data"
        / "pallas"
        / "benchmarks"
        / release_root.name
    )
    bundled_release.parent.mkdir(parents=True, exist_ok=True)
    selected = next(
        (task for task in manifest["tasks"] if task["task_id"] == task_id), None
    )
    if selected is None:
        raise Phase2WorkerError(f"SUBMISSION_TASK_UNKNOWN:{task_id}")
    task_source = release_root / selected["path"]
    task_target = bundled_release / selected["path"]
    task_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        task_source,
        task_target,
        ignore=shutil.ignore_patterns("solution"),
    )
    lock_path = (
        Path(__file__).parents[3]
        / "config/pallas/phase2-worker-requirements.lock"
    )
    if file_sha256(lock_path) != manifest.get("worker_requirements_lock_sha256"):
        raise Phase2WorkerError("WORKER_REQUIREMENTS_LOCK_DRIFT")
    shutil.copy2(lock_path, bundled_release / "worker-requirements.lock")
    trusted_source = bundled_release / "trusted-src/opjax/pallas"
    trusted_source.parent.mkdir(parents=True)
    shutil.copytree(
        Path(__file__).parent,
        trusted_source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (trusted_source.parent / "__init__.py").write_text("", encoding="utf-8")
    sanitized_manifest = {
        "schema_version": 1,
        "kind": "opjax_phase2_sanitized_worker_bundle",
        "release_sha256": manifest["release_sha256"],
        "runtime": manifest["runtime"],
        "tasks": [selected],
    }
    (bundled_release / "manifest.json").write_text(
        json.dumps(sanitized_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forbidden = list(bundled_release.rglob("solution"))
    if forbidden:
        raise Phase2WorkerError("HIDDEN_SOLUTION_IN_WORKER_BUNDLE")
    files = {
        str(path.relative_to(bundled_release)): file_sha256(path)
        for path in sorted(bundled_release.rglob("*"))
        if path.is_file()
    }
    worker_manifest = {"schema_version": 1, "files": files}
    worker_manifest["bundle_sha256"] = canonical_sha256(worker_manifest)
    (bundled_release / "worker-manifest.json").write_text(
        json.dumps(worker_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundled_release


@dataclass(frozen=True)
class GcloudDisposableTPUBackend:
    release_root: Path
    patch_path: Path
    zone: str = "us-west4-a"
    accelerator_type: str = "v5litepod-1"
    runtime_version: str = "tpu-ubuntu2204-base"
    name_prefix: str = "opjax-p2"
    service_account: str | None = None

    def _run(self, command: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            raise Phase2WorkerError(
                f"COMMAND_FAILED:{command[0]}:returncode={process.returncode}:{process.stderr[-1000:]}"
            )
        return process

    def grade(self, request: dict[str, Any], destination: Path) -> dict[str, Any]:
        worker_name = (
            f"{self.name_prefix}-{request['request_sha256'][:8]}-"
            f"{uuid.uuid4().hex[:6]}"
        )

        def create(_: dict[str, Any]) -> str:
            if not self.service_account:
                raise Phase2WorkerError("TPU_WORKER_SERVICE_ACCOUNT_REQUIRED")
            try:
                self._run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "create",
                        worker_name,
                        f"--zone={self.zone}",
                        f"--accelerator-type={self.accelerator_type}",
                        f"--version={self.runtime_version}",
                        f"--service-account={self.service_account}",
                        "--scopes=https://www.googleapis.com/auth/logging.write",
                        "--quiet",
                    ]
                )
            except Phase2WorkerError:
                subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "delete",
                        worker_name,
                        f"--zone={self.zone}",
                        "--quiet",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                raise
            return worker_name

        def grade_remote(
            identity: str, value: dict[str, Any], output: Path
        ) -> dict[str, Any]:
            address = self._run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "describe",
                    identity,
                    f"--zone={self.zone}",
                    "--format=value(networkEndpoints[0].ipAddress)",
                ]
            ).stdout.strip()
            worker_address = ipaddress.ip_address(address)
            if not worker_address.is_private or worker_address.is_link_local:
                raise Phase2WorkerError("TPU_WORKER_ADDRESS_INVALID")
            with tempfile.TemporaryDirectory(prefix="opjax-p2-request-") as temporary:
                temporary_root = Path(temporary)
                request_path = temporary_root / "request.json"
                write_request(request_path, value)
                bundle = temporary_root / "bundle"
                bundled_release = build_worker_bundle(
                    self.release_root, bundle, task_id=value["task_id"]
                )
                remote = "/tmp/opjax-phase2-submission"
                self._run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "ssh",
                        identity,
                        f"--zone={self.zone}",
                        "--command",
                        f"sudo rm -rf {remote} && mkdir -p {remote}/input",
                    ]
                )
                for source, target in ((bundle, f"{remote}/input/"),):
                    self._run(
                        [
                            "gcloud",
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "scp",
                            "--recurse",
                            str(source),
                            f"{identity}:{target}",
                            f"--zone={self.zone}",
                        ]
                    )
                for source in (self.patch_path, request_path):
                    self._run(
                        [
                            "gcloud",
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "scp",
                            str(source),
                            f"{identity}:{remote}/input/",
                            f"--zone={self.zone}",
                        ]
                    )
                remote_release = (
                    f"input/{bundle.name}/data/pallas/benchmarks/"
                    f"{bundled_release.name}"
                )
                verifier_python = (
                    f"{remote_release}/trusted-src"
                )
                command = (
                    "python3 -m pip install --user uv==0.7.19 >/dev/null && "
                    f"cd {remote} && "
                    "UV_PYTHON_INSTALL_DIR=worker-python "
                    "/home/$USER/.local/bin/uv python install 3.12.11 && "
                    "/home/$USER/.local/bin/uv venv "
                    "--python worker-python/cpython-3.12.11-linux-x86_64-gnu/"
                    "bin/python3.12 worker-venv && "
                    "/home/$USER/.local/bin/uv pip install "
                    "--python worker-venv/bin/python "
                    f"--requirements {remote_release}/worker-requirements.lock && "
                    "chmod -R a+rX worker-python worker-venv && "
                    "chmod -R go-rwx input && env "
                    f"PYTHONPATH={verifier_python} "
                    "PYTHONDONTWRITEBYTECODE=1 "
                    "JAX_PLATFORMS=cpu "
                    "TPU_SKIP_MDS_QUERY=1 "
                    f"TPU_ACCELERATOR_TYPE={self.accelerator_type} "
                    "TPU_WORKER_ID=0 "
                    f"TPU_WORKER_HOSTNAMES={address} "
                    "TPU_PROCESS_BOUNDS=1,1,1 "
                    "TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 "
                    "TPU_HOST_BOUNDS=1,1,1 "
                    "LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 "
                    f"OPJAX_DISPOSABLE_WORKER_IDENTITY={identity} "
                    "OPJAX_CANDIDATE_SANDBOX=linux-systemd "
                    "worker-venv/bin/python -m opjax.pallas.phase2_runner "
                    "grade-worker-submission "
                    f"--release {remote_release} --request input/request.json "
                    f"--patch input/{self.patch_path.name} --out output"
                )
                self._run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "ssh",
                        identity,
                        f"--zone={self.zone}",
                        "--command",
                        command,
                    ],
                    timeout=3600,
                )
                def download(download_root: Path) -> None:
                    self._run(
                        [
                            "gcloud",
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "scp",
                            "--recurse",
                            f"{identity}:{remote}/output",
                            str(download_root),
                            f"--zone={self.zone}",
                        ]
                    )

                install_worker_output(destination=output, download=download)
                return json.loads(
                    (output / "submission.json").read_text(encoding="utf-8")
                )

        def destroy(identity: str) -> str:
            self._run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "delete",
                    identity,
                    f"--zone={self.zone}",
                    "--async",
                    "--quiet",
                ],
                timeout=120,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                probe = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "describe",
                        identity,
                        f"--zone={self.zone}",
                        "--format=value(state)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                diagnostic = f"{probe.stdout}\n{probe.stderr}".upper()
                if probe.returncode != 0 and (
                    "NOT_FOUND" in diagnostic or "WAS NOT FOUND" in diagnostic
                ):
                    return datetime.now(timezone.utc).isoformat()
                if probe.returncode != 0:
                    raise Phase2WorkerError(
                        f"TPU_DELETE_PROBE_FAILED:returncode={probe.returncode}:"
                        f"{probe.stderr[-1000:]}"
                    )
                time.sleep(5)
            raise Phase2WorkerError("TPU_DELETE_TIMEOUT")

        factory = DisposableWorkerFactory(
            create=create,
            grade=grade_remote,
            destroy=destroy,
            worker_evidence={
                "candidate_user": "nobody",
                "sandbox_policy": "systemd-cgroup-v1",
                "service_account": self.service_account,
            },
        )
        return factory.run(request, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase2-worker")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zone", default="us-west4-a")
    parser.add_argument("--service-account", required=True)
    args = parser.parse_args(argv)
    release = json.loads((args.release / "manifest.json").read_text(encoding="utf-8"))
    task = next(
        (task for task in release["tasks"] if task["task_id"] == args.task_id),
        None,
    )
    if task is None:
        raise Phase2WorkerError(f"SUBMISSION_TASK_UNKNOWN:{args.task_id}")
    request = build_submission_request(
        release_manifest=release,
        task=task,
        patch_path=args.patch,
    )
    backend = GcloudDisposableTPUBackend(
        release_root=args.release,
        patch_path=args.patch,
        zone=args.zone,
        service_account=args.service_account,
    )
    result = backend.grade(request, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
