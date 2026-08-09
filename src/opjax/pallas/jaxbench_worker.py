"""Disposable TPU boundary for full JAXBench submissions."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opjax.pallas.jaxbench_executable import file_sha256
from opjax.pallas.jaxbench_verifier import verify_serialized_submission


class JaxBenchWorkerError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_sha256(root: Path, *, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(part in excluded for part in path.relative_to(root).parts)
    }
    return canonical_sha256(files)


def build_request(
    *, release: dict[str, Any], task: dict[str, Any], patch_path: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "release_sha256": release["release_sha256"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "patch_sha256": file_sha256(patch_path),
        "worker_policy": "one-submission-one-disposable-tpu-vm",
        "execution_boundary": "sandbox-compile-serialized-executable-pristine-verify",
    }
    return {**payload, "request_sha256": canonical_sha256(payload)}


def write_request(path: Path, request: dict[str, Any]) -> None:
    payload = dict(request)
    expected = payload.pop("request_sha256", None)
    if canonical_sha256(payload) != expected:
        raise JaxBenchWorkerError("REQUEST_HASH_INVALID")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "opjax-harness",
        "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
        "GIT_COMMITTER_NAME": "opjax-harness",
        "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
    }


def materialize_submission(
    *, task_root: Path, patch_path: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists() or not patch_path.is_file():
        raise JaxBenchWorkerError("SUBMISSION_INPUT_INVALID")
    destination.mkdir(parents=True)
    shutil.copy2(task_root / "instruction.md", destination / "instruction.md")
    shutil.copy2(task_root / "environment/starter/kernel.py", destination / "kernel.py")
    for name in ("PALLAS_API.md", "dev_check.py"):
        shutil.copy2(task_root / "environment/public" / name, destination / name)
    environment = _git_environment()
    subprocess.run(["git", "init", "-q", str(destination)], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(destination), "add", "."], check=True, env=environment
    )
    subprocess.run(
        ["git", "-C", str(destination), "commit", "-q", "-m", "task base"],
        check=True,
        env=environment,
    )
    applied = subprocess.run(
        ["git", "-C", str(destination), "apply", "--whitespace=error-all", "-"],
        input=patch_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if applied.returncode != 0:
        raise JaxBenchWorkerError(
            "SUBMISSION_PATCH_INVALID:"
            + applied.stderr.decode(errors="replace").strip()[-1000:]
        )
    kernel = destination / "kernel.py"
    if not kernel.is_file() or kernel.is_symlink():
        raise JaxBenchWorkerError("SUBMISSION_KERNEL_INVALID")
    changed = subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.splitlines()
    if not changed or any(
        line[3:].split(" -> ")[-1].startswith(("tests/", "solution/"))
        for line in changed
    ):
        raise JaxBenchWorkerError("SUBMISSION_PATCH_SCOPE_INVALID")
    return {
        "kernel_path": kernel,
        "kernel_sha256": file_sha256(kernel),
        "patch_sha256": file_sha256(patch_path),
        "workspace_sha256": tree_sha256(destination, excluded={".git"}),
    }


def build_calibration_patch(*, task_root: Path, output: Path) -> dict[str, Any]:
    optimized = task_root / "tests/jaxbench/optimized.py"
    if not optimized.is_file() or output.exists():
        raise JaxBenchWorkerError("CALIBRATION_PATCH_INPUT_INVALID")
    with tempfile.TemporaryDirectory(prefix="opjax-jaxbench-calibration-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        shutil.copy2(task_root / "environment/starter/kernel.py", workspace / "kernel.py")
        environment = _git_environment()
        subprocess.run(["git", "init", "-q", str(workspace)], check=True, env=environment)
        subprocess.run(
            ["git", "-C", str(workspace), "add", "."], check=True, env=environment
        )
        subprocess.run(
            ["git", "-C", str(workspace), "commit", "-q", "-m", "task base"],
            check=True,
            env=environment,
        )
        shutil.copy2(optimized, workspace / "kernel.py")
        patch = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "HEAD"],
            capture_output=True,
            check=True,
            env=environment,
        ).stdout
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(patch)
    return {
        "task_id": task_root.name,
        "optimized_sha256": file_sha256(optimized),
        "patch_sha256": file_sha256(output),
    }


def _candidate_environment() -> dict[str, str]:
    exact = {"PATH", "PYTHONHASHSEED"}
    prefixes = ("JAX_", "XLA_", "LIBTPU_", "TPU_", "CLOUD_TPU_")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in exact or name.startswith(prefixes)
    }
    environment["PYTHONHASHSEED"] = "0"
    environment["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "tpu")
    return environment


def prepare_sandbox_parent(path: Path, *, sandboxed: bool) -> None:
    if sandboxed:
        path.chmod(0o711)


def tpu_network_allow_properties(environment: dict[str, str]) -> list[str]:
    hostnames = environment.get("TPU_WORKER_HOSTNAMES", "")
    addresses = [value.strip() for value in hostnames.split(",") if value.strip()]
    if not addresses:
        raise JaxBenchWorkerError("TPU_WORKER_HOSTNAMES_REQUIRED")
    properties: list[str] = []
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_private or address.is_link_local:
            raise JaxBenchWorkerError(f"TPU_WORKER_ADDRESS_INVALID:{value}")
        properties.append(
            f"--property=IPAddressAllow={address}/{address.max_prefixlen}"
        )
    return properties


def compile_in_sandbox(
    *,
    task: dict[str, Any],
    workspace: Path,
    candidate_root: Path,
    sandboxed: bool,
    allow_cpu_test: bool = False,
    timeout_seconds: int = 900,
) -> tuple[dict[str, Any], Path]:
    if candidate_root.exists():
        raise JaxBenchWorkerError(f"OUTPUT_EXISTS:{candidate_root}")
    candidate_root.mkdir(parents=True)
    candidate_workspace = candidate_root / "workspace"
    shutil.copytree(workspace, candidate_workspace, ignore=shutil.ignore_patterns(".git"))
    task_path = candidate_root / "task.json"
    public_task = {
        key: task[key]
        for key in (
            "schema_version",
            "task_id",
            "baseline_sha256",
            "public_specification_sha256",
            "tensor_schema",
            "shape_policy",
        )
        if key in task
    }
    task_path.write_text(json.dumps(public_task, sort_keys=True) + "\n")
    library = candidate_root / "lib/opjax/pallas"
    library.mkdir(parents=True)
    (library.parent / "__init__.py").write_text("")
    source_root = Path(__file__).parent
    for name in ("__init__.py", "jaxbench_executable.py"):
        shutil.copy2(source_root / name, library / name)
    compiled_dir = candidate_root / "compiled"
    environment = _candidate_environment()
    if allow_cpu_test:
        environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(candidate_root / "lib"), str(candidate_workspace))
    )
    command = [
        sys.executable,
        "-m",
        "opjax.pallas.jaxbench_executable",
        "--task",
        str(task_path),
        "--kernel",
        str(candidate_workspace / "kernel.py"),
        "--out",
        str(compiled_dir),
    ]
    if allow_cpu_test:
        command.append("--allow-cpu-test")
    if sandboxed:
        for path in (candidate_workspace, candidate_root / "lib"):
            for child in path.rglob("*"):
                child.chmod(0o555 if child.is_dir() else 0o444)
            path.chmod(0o555)
        task_path.chmod(0o444)
        candidate_root.chmod(0o777)
        command = [
            "sudo",
            "-n",
            "systemd-run",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            "--uid=nobody",
            "--gid=nogroup",
            f"--working-directory={candidate_workspace}",
            "--property=IPAddressDeny=any",
            *tpu_network_allow_properties(environment),
            "--property=NoNewPrivileges=yes",
            "--property=ProtectHome=yes",
            "--property=ProtectSystem=strict",
            f"--property=ReadWritePaths={candidate_root}",
            "--property=PrivateDevices=no",
            "--property=LimitMEMLOCK=infinity",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=5",
            "env",
            *(f"{name}={value}" for name, value in environment.items()),
            *command,
        ]
    try:
        process = subprocess.run(
            command,
            cwd=candidate_workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            {
                "passed": False,
                "error": "CANDIDATE_COMPILE_TIMEOUT",
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
            compiled_dir,
        )
    finally:
        if sandboxed:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    str(candidate_root),
                ],
                capture_output=True,
                check=False,
            )
            candidate_root.chmod(0o700)
    record = None
    for line in reversed(process.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "passed" in value:
            record = value
            break
    if record is None:
        record = {
            "passed": False,
            "error": "CANDIDATE_COMPILE_RESULT_MISSING",
        }
    record.update(
        {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    )
    return record, compiled_dir


def _write_result(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    reward = {
        "schema_version": 1,
        "task_id": result["task_id"],
        "reward": result["reward"],
        "failure_stage": None if result["passed"] else result["stage"],
        "correct": result.get("correct", False),
        "authentic": result.get("authentic", False),
        "profiled": result.get("profiled", False),
        "speedup": result.get("speedup"),
        "beats_xla": result.get("beats_xla", False),
        "infrastructure_error": result.get("infrastructure_error", False),
    }
    (out_dir / "reward.json").write_text(
        json.dumps(reward, indent=2, sort_keys=True) + "\n"
    )
    return reward


def grade_worker_submission(
    *,
    release_root: Path,
    request_path: Path,
    patch_path: Path,
    out_dir: Path,
    sandboxed: bool = True,
    allow_cpu_test: bool = False,
    allow_plain_jax_test: bool = False,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchWorkerError(f"OUTPUT_EXISTS:{out_dir}")
    release = json.loads((release_root / "manifest.json").read_text())
    request = json.loads(request_path.read_text())
    payload = dict(request)
    expected_request = payload.pop("request_sha256", None)
    if canonical_sha256(payload) != expected_request:
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
    with tempfile.TemporaryDirectory(prefix="opjax-jaxbench-worker-") as temporary:
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
            allow_cpu_test=allow_cpu_test,
        )
        (out_dir / "compile.log").write_text(
            compile_record.get("stdout", "") + compile_record.get("stderr", "")
        )
        if not compile_record.get("passed"):
            result = {
                "schema_version": 1,
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
            _write_result(out_dir, result)
        else:
            try:
                result = verify_serialized_submission(
                    task=task,
                    baseline_path=task_root / "tests/jaxbench/baseline.py",
                    compiled_dir=compiled_dir,
                    out_dir=out_dir / "verification",
                    require_tpu=not allow_cpu_test,
                    require_pallas=not allow_plain_jax_test,
                    timing_rounds=5 if allow_cpu_test else 9,
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                baseline_failure = "BASELINE_NOT_SCOREABLE" in detail
                result = {
                    "schema_version": 1,
                    "task_id": task["task_id"],
                    "passed": False,
                    "stage": "infrastructure" if baseline_failure else "verifier",
                    "error": detail,
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


def validate_response(
    *, request: dict[str, Any], destination: Path
) -> dict[str, Any]:
    response = json.loads((destination / "submission.json").read_text())
    if any(
        response.get(key) != request[key]
        for key in (
            "release_sha256",
            "task_id",
            "task_sha256",
            "patch_sha256",
            "request_sha256",
        )
    ):
        raise JaxBenchWorkerError("RESPONSE_BINDING_INVALID")
    for name, hash_field in (
        ("result.json", "result_sha256"),
        ("reward.json", "reward_sha256"),
        ("model.patch", "model_patch_sha256"),
    ):
        if file_sha256(destination / name) != response.get(hash_field):
            raise JaxBenchWorkerError(f"RESPONSE_ARTIFACT_INVALID:{name}")
    worker = response.get("worker")
    if (
        not isinstance(worker, dict)
        or worker.get("disposable") is not True
        or not worker.get("destroyed_at")
        or worker.get("candidate_user") != "nobody"
        or worker.get("execution_boundary")
        != "sandbox-compile-serialized-executable-pristine-verify"
    ):
        raise JaxBenchWorkerError("WORKER_LIFECYCLE_INVALID")
    return response


@dataclass(frozen=True)
class DisposableWorkerFactory:
    create: Callable[[dict[str, Any]], str]
    grade: Callable[[str, dict[str, Any], Path], dict[str, Any]]
    destroy: Callable[[str], str]
    evidence: dict[str, Any] = field(default_factory=dict)

    def run(self, request: dict[str, Any], destination: Path) -> dict[str, Any]:
        identity = self.create(request)
        try:
            response = self.grade(identity, request, destination)
        finally:
            destroyed_at = self.destroy(identity)
        response["worker"] = {
            "identity": identity,
            "disposable": True,
            "destroyed_at": destroyed_at,
            **self.evidence,
        }
        (destination / "submission.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
        return validate_response(request=request, destination=destination)


def build_worker_bundle(
    *, release_root: Path, task_id: str, destination: Path
) -> Path:
    release = json.loads((release_root / "manifest.json").read_text())
    task = next((task for task in release["tasks"] if task["task_id"] == task_id), None)
    if task is None:
        raise JaxBenchWorkerError(f"SUBMISSION_TASK_UNKNOWN:{task_id}")
    target = destination / "release"
    task_target = target / task["path"]
    task_target.parent.mkdir(parents=True)
    shutil.copytree(release_root / task["path"], task_target)
    optimized = task_target / "tests/jaxbench/optimized.py"
    optimized.unlink(missing_ok=True)
    trusted_source = target / "trusted-src/opjax/pallas"
    trusted_source.mkdir(parents=True)
    (trusted_source.parent / "__init__.py").write_text("")
    for name in (
        "__init__.py",
        "benchmarking.py",
        "jaxbench_executable.py",
        "jaxbench_verifier.py",
        "jaxbench_worker.py",
    ):
        shutil.copy2(Path(__file__).parent / name, trusted_source / name)
    lock = Path(__file__).parents[3] / "config/pallas/phase2-worker-requirements.lock"
    shutil.copy2(lock, target / "worker-requirements.lock")
    sanitized = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_sanitized_worker_bundle",
        "release_sha256": release["release_sha256"],
        "tasks": [task],
    }
    (target / "manifest.json").write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    )
    return target


@dataclass(frozen=True)
class GcloudDisposableTPUBackend:
    release_root: Path
    patch_path: Path
    service_account: str
    zone: str = "us-west4-a"
    accelerator_type: str = "v5litepod-1"
    runtime_version: str = "tpu-ubuntu2204-base"
    name_prefix: str = "opjax-jb"

    def _run(self, command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if process.returncode != 0:
            raise JaxBenchWorkerError(
                f"COMMAND_FAILED:{command[0]}:{process.returncode}:"
                f"{process.stderr[-1000:]}"
            )
        return process

    def grade(self, request: dict[str, Any], destination: Path) -> dict[str, Any]:
        worker_name = (
            f"{self.name_prefix}-{request['request_sha256'][:8]}-{uuid.uuid4().hex[:6]}"
        )

        def create(_: dict[str, Any]) -> str:
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
            parsed = ipaddress.ip_address(address)
            if not parsed.is_private or parsed.is_link_local:
                raise JaxBenchWorkerError("WORKER_ADDRESS_INVALID")
            with tempfile.TemporaryDirectory(prefix="opjax-jaxbench-request-") as temporary:
                root = Path(temporary)
                request_path = root / "request.json"
                write_request(request_path, value)
                bundle = build_worker_bundle(
                    release_root=self.release_root,
                    task_id=value["task_id"],
                    destination=root / "bundle",
                )
                remote = "/tmp/opjax-jaxbench-submission"
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
                for source in (bundle.parent, self.patch_path, request_path):
                    self._run(
                        [
                            "gcloud",
                            "compute",
                            "tpus",
                            "tpu-vm",
                            "scp",
                            "--recurse" if source.is_dir() else "--verbosity=error",
                            str(source),
                            f"{identity}:{remote}/input/",
                            f"--zone={self.zone}",
                        ]
                    )
                remote_release = "input/bundle/release"
                command = (
                    "python3 -m pip install --user uv==0.7.19 >/dev/null && "
                    f"cd {remote} && "
                    "UV_PYTHON_INSTALL_DIR=worker-python "
                    "/home/$USER/.local/bin/uv python install 3.12.11 && "
                    "/home/$USER/.local/bin/uv venv --python "
                    "worker-python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12 "
                    "worker-venv && "
                    "/home/$USER/.local/bin/uv pip install --python worker-venv/bin/python "
                    f"--requirements {remote_release}/worker-requirements.lock && "
                    "chmod -R a+rX worker-python worker-venv && chmod -R go-rwx input && "
                    "env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=tpu "
                    "TPU_SKIP_MDS_QUERY=1 "
                    f"TPU_ACCELERATOR_TYPE={self.accelerator_type} TPU_WORKER_ID=0 "
                    f"TPU_WORKER_HOSTNAMES={address} TPU_PROCESS_BOUNDS=1,1,1 "
                    "TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 "
                    "LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 "
                    f"PYTHONPATH={remote_release}/trusted-src "
                    "worker-venv/bin/python -m opjax.pallas.jaxbench_worker grade-worker "
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
                self._run(
                    [
                        "gcloud",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "scp",
                        "--recurse",
                        f"{identity}:{remote}/output",
                        str(output.parent),
                        f"--zone={self.zone}",
                    ]
                )
                downloaded = output.parent / "output"
                if output.exists() or not downloaded.is_dir():
                    raise JaxBenchWorkerError("RESPONSE_DOWNLOAD_INVALID")
                downloaded.rename(output)
                return json.loads((output / "submission.json").read_text())

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
                time.sleep(5)
            raise JaxBenchWorkerError("WORKER_DELETE_TIMEOUT")

        factory = DisposableWorkerFactory(
            create=create,
            grade=grade_remote,
            destroy=destroy,
            evidence={
                "candidate_user": "nobody",
                "sandbox_policy": "systemd-cgroup-v1",
                "service_account": self.service_account,
                "execution_boundary": (
                    "sandbox-compile-serialized-executable-pristine-verify"
                ),
            },
        )
        return factory.run(request, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    grade_worker = commands.add_parser("grade-worker")
    grade_worker.add_argument("--release", type=Path, required=True)
    grade_worker.add_argument("--request", type=Path, required=True)
    grade_worker.add_argument("--patch", type=Path, required=True)
    grade_worker.add_argument("--out", type=Path, required=True)
    grade_worker.add_argument("--allow-cpu-test", action="store_true")
    grade_worker.add_argument("--allow-plain-jax-test", action="store_true")
    submit = commands.add_parser("submit")
    submit.add_argument("--release", type=Path, required=True)
    submit.add_argument("--task-id", required=True)
    submit.add_argument("--patch", type=Path, required=True)
    submit.add_argument("--out", type=Path, required=True)
    submit.add_argument("--zone", default="us-west4-a")
    submit.add_argument("--service-account", required=True)
    calibration = commands.add_parser("calibration-patch")
    calibration.add_argument("--task", type=Path, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "grade-worker":
        response = grade_worker_submission(
            release_root=args.release,
            request_path=args.request,
            patch_path=args.patch,
            out_dir=args.out,
            sandboxed=not args.allow_cpu_test,
            allow_cpu_test=args.allow_cpu_test,
            allow_plain_jax_test=args.allow_plain_jax_test,
        )
    elif args.command == "submit":
        release = json.loads((args.release / "manifest.json").read_text())
        task = next(
            (task for task in release["tasks"] if task["task_id"] == args.task_id),
            None,
        )
        if task is None:
            raise JaxBenchWorkerError(f"SUBMISSION_TASK_UNKNOWN:{args.task_id}")
        request = build_request(release=release, task=task, patch_path=args.patch)
        response = GcloudDisposableTPUBackend(
            release_root=args.release,
            patch_path=args.patch,
            service_account=args.service_account,
            zone=args.zone,
        ).grade(request, args.out)
    else:
        response = build_calibration_patch(task_root=args.task, output=args.out)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
