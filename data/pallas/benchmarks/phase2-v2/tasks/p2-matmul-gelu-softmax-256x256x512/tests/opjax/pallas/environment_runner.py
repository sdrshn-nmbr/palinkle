"""Execute one hidden Pallas environment task on a real TPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import chex
import jax
import numpy as np

from opjax.pallas.benchmarking import validate_timing_result
from opjax.pallas.candidate_policy import candidate_module_policy_error
from opjax.pallas.environment import verify_static
from opjax.pallas.lowering import validate_execution_evidence
from opjax.pallas.scoring import inspect_pallas_source
from opjax.pallas.task_semantics import generate_inputs, semantic_oracle


class EnvironmentRunnerError(RuntimeError):
    pass


def consume_candidate_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    path.unlink()
    return value


def _call_path(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def has_host_compute_in_workload(
    source: str, *, allowed_entrypoints: tuple[str, ...] = ()
) -> bool:
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    workload = functions.get("workload")
    if workload is None:
        return False
    allowed = {
        "jax.ShapeDtypeStruct",
        "pl.BlockSpec",
        "pl.pallas_call",
        "pltpu.PrefetchScalarGridSpec",
        "pltpu.VMEM",
        "pltpu.SemaphoreType.DMA",
        "pltpu.CompilerParams",
        *allowed_entrypoints,
        *(entrypoint.rsplit(".", 1)[-1] for entrypoint in allowed_entrypoints),
    }
    for node in ast.walk(workload):
        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node)
        if path in allowed:
            continue
        if not path and isinstance(node.func, ast.Call):
            if _call_path(node.func) == "pl.pallas_call":
                continue
        return True
    return False


def _is_runtime_safety_failure(error: BaseException) -> bool:
    detail = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in detail
        for marker in (
            "core halted",
            "device halted",
            "boundscheck",
            "out of bounds",
            "dma.hbm_to_vmem",
            "dma.vmem_to_hbm",
            "sigabrt",
            "segmentation fault",
        )
    )


def classify_seed_failure(*, phase: str, error: BaseException) -> str:
    if _is_runtime_safety_failure(error):
        return "runtime_safety"
    if phase == "compile":
        return "tpu_compile"
    if phase == "execute":
        return "full_shape_correctness"
    raise EnvironmentRunnerError(f"SEED_PHASE_INVALID: {phase}")


def classify_worker_failure(worker: dict[str, Any]) -> str:
    if worker.get("worker_recovery_required") is True:
        return "runtime_safety"
    return classify_seed_failure(
        phase=str(worker.get("phase")),
        error=RuntimeError(worker.get("error", "candidate execution failed")),
    )


def _failed(
    *,
    stage: str,
    error: str,
    hardware: dict[str, Any],
    kernel_sha256: str,
    stages: dict[str, bool],
) -> dict[str, Any]:
    return {
        "passed": False,
        "stage": stage,
        "error": error,
        "hardware": hardware,
        "kernel_sha256": kernel_sha256,
        "stages": stages,
        "authentic": stages.get("pallas_api", False),
        "correct": stages.get("full_shape_correctness", False),
        "normal_lowered": stages.get("normal_lowering", False),
        "infrastructure_error": False,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_candidate_output(
    *, actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float
) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    chex.assert_trees_all_close(actual, expected, rtol=rtol, atol=atol)


def _candidate_environment() -> dict[str, str]:
    exact = {"PATH", "PYTHONPATH", "TMPDIR", "TMP", "TEMP", "PYTHONHASHSEED"}
    prefixes = ("JAX_", "XLA_", "LIBTPU_", "TPU_", "CLOUD_TPU_")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in exact or name.startswith(prefixes)
    }
    environment["PYTHONHASHSEED"] = "0"
    environment["JAX_PLATFORMS"] = "tpu"
    return environment


def classify_missing_candidate_result(
    *, returncode: int, stderr: str
) -> dict[str, Any]:
    fatal_markers = (
        "aborted",
        "core halted",
        "device halted",
        "dma.hbm_to_vmem",
        "dma.vmem_to_hbm",
        "segmentation fault",
        "sigabrt",
    )
    candidate_abort = (
        returncode < 0
        or returncode in {124, 134, 137, 139}
        or any(marker in stderr.lower() for marker in fatal_markers)
    )
    detail = stderr.strip()[-2000:]
    return {
        "passed": False,
        "phase": "execute",
        "error": (
            f"CANDIDATE_RESULT_MISSING:returncode={returncode}:stderr={detail}"
        ),
        "worker_recovery_required": candidate_abort,
    }


def run_candidate_process(
    *,
    task: dict[str, Any],
    kernel_path: Path,
    output_dir: Path,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run candidate import and device execution outside the grading process.

    Fault space: candidate imports may mutate shared modules, inspect the process,
    abort, hang, poison the TPU, forge stdout, or write outside its artifact root.
    This boundary contains Python module state and validates every consumed artifact.
    On a disposable worker, the process runs as an unprivileged user with a
    UID-scoped egress-deny rule. Trusted verifier input remains outside its root.
    """
    output_dir = output_dir.resolve()
    kernel_path = kernel_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sandboxed = os.environ.get("OPJAX_CANDIDATE_SANDBOX") == "linux-systemd"
    candidate_environment = _candidate_environment()
    candidate_kernel = output_dir / "kernel.py"
    shutil.copy2(kernel_path, candidate_kernel)
    task_path = output_dir / "task.json"
    task_path.write_text(json.dumps(task, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "opjax.pallas.candidate_worker",
        "--task",
        str(task_path),
        "--kernel",
        str(candidate_kernel),
        "--output-dir",
        str(output_dir),
    ]
    if sandboxed:
        required_tpu_environment = {
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": "v5litepod-1",
            "TPU_WORKER_ID": "0",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_HOST_BOUNDS": "1,1,1",
        }
        if any(
            candidate_environment.get(name) != value
            for name, value in required_tpu_environment.items()
        ):
            raise EnvironmentRunnerError("CANDIDATE_TPU_ENVIRONMENT_INVALID")
        worker_address = ipaddress.ip_address(
            candidate_environment.get("TPU_WORKER_HOSTNAMES", "")
        )
        if not worker_address.is_private or worker_address.is_link_local:
            raise EnvironmentRunnerError("CANDIDATE_TPU_ADDRESS_INVALID")
        library_root = output_dir / "lib/opjax/pallas"
        library_root.mkdir(parents=True)
        (library_root.parent / "__init__.py").write_text("", encoding="utf-8")
        for name in (
            "__init__.py",
            "benchmarking.py",
            "candidate_worker.py",
            "lowering.py",
            "task_semantics.py",
        ):
            shutil.copy2(Path(__file__).parent / name, library_root / name)
        for path in (output_dir / "lib").rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        candidate_environment["PYTHONPATH"] = str(output_dir / "lib")
        candidate_environment["TMPDIR"] = str(output_dir)
        output_dir.chmod(0o777)
        task_path.chmod(0o444)
        candidate_kernel.chmod(0o444)
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
            f"--working-directory={output_dir}",
            "--property=IPAddressDeny=any",
            "--property=NoNewPrivileges=yes",
            "--property=ProtectHome=yes",
            "--property=ProtectSystem=strict",
            "--property=ReadWritePaths=/tmp",
            f"--property=ReadWritePaths={output_dir}",
            "--property=PrivateDevices=no",
            "--property=LimitMEMLOCK=infinity",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=5",
            "env",
            *(f"{name}={value}" for name, value in candidate_environment.items()),
            *command,
        ]
    try:
        process = subprocess.run(
            command,
            cwd=output_dir,
            env=candidate_environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "phase": "execute",
            "error": "CANDIDATE_PROCESS_TIMEOUT",
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "worker_recovery_required": True,
        }
    finally:
        if sandboxed:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    str(output_dir),
                ],
                capture_output=True,
                check=False,
            )
            output_dir.chmod(0o700)
    result = None
    for line in reversed(process.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("phase") in {
            "compile",
            "execute",
            "profile",
            "complete",
            "infrastructure",
        }:
            result = value
            break
    if result is None:
        return classify_missing_candidate_result(
            returncode=process.returncode,
            stderr=process.stderr,
        )
    result["returncode"] = process.returncode
    result["stdout"] = process.stdout
    result["stderr"] = process.stderr
    return result


def evaluate_task(
    *, task: dict[str, Any], kernel_path: Path, evidence_dir: Path | None = None
) -> dict[str, Any]:
    hardware = {"target": "tpu", "execution": "not_started"}
    kernel_sha256 = _sha256_file(kernel_path)
    stages = {
        "artifact_contract": False,
        "pallas_api": False,
        "tpu_compile": False,
        "full_shape_correctness": False,
        "normal_lowering": False,
        "runtime_safety": False,
        "profile": False,
    }
    source = kernel_path.read_text(encoding="utf-8")
    allowed_entrypoints = tuple(task.get("allowed_pallas_entrypoints", ()))
    static = verify_static(
        f"```python\n{source}\n```",
        allowed_pallas_entrypoints=allowed_entrypoints,
    )
    if not static.passed:
        stage = "artifact_contract" if static.stage == "output_contract" else static.stage
        return _failed(
            stage=stage,
            error=static.feedback,
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["artifact_contract"] = True
    policy_error = candidate_module_policy_error(
        source, allowed_entrypoints=allowed_entrypoints
    )
    if policy_error is not None:
        return _failed(
            stage="pallas_api",
            error=policy_error,
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    if has_host_compute_in_workload(
        source, allowed_entrypoints=allowed_entrypoints
    ):
        return _failed(
            stage="pallas_api",
            error="PLAIN_JAX_COMPUTE_IN_WORKLOAD",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    inspection = inspect_pallas_source(source)
    allowed_entrypoint = bool(static.evidence.get("allowed_entrypoint"))
    if not inspection.authentic and not allowed_entrypoint:
        return _failed(
            stage="pallas_api",
            error=",".join(inspection.reasons),
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["pallas_api"] = True
    correctness_seeds = tuple(task.get("correctness_seeds", (0, 1, 2)))
    if correctness_seeds != (0, 1, 2):
        raise EnvironmentRunnerError(f"CORRECTNESS_SEEDS_INVALID: {correctness_seeds}")
    tolerance = task.get("correctness_tolerance", {"rtol": 1e-3, "atol": 1e-3})
    if evidence_dir is None:
        return _failed(
            stage="profile",
            error="PROFILE_EVIDENCE_MISSING",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    candidate_root = evidence_dir / "candidate-process"
    worker = run_candidate_process(
        task={**task, "correctness_seeds": list(correctness_seeds)},
        kernel_path=kernel_path,
        output_dir=candidate_root,
    )
    hardware = worker.get("hardware", hardware)
    if worker.get("phase") == "infrastructure":
        raise EnvironmentRunnerError(worker.get("error", "CANDIDATE_WORKER_FAILED"))
    if worker.get("kernel_sha256") not in {None, kernel_sha256}:
        raise EnvironmentRunnerError("CANDIDATE_KERNEL_HASH_MISMATCH")
    if worker.get("phase") in {"compile", "execute"}:
        error = RuntimeError(worker.get("error", "candidate execution failed"))
        result = _failed(
            stage=classify_worker_failure(worker),
            error=f"seed={worker.get('seed')} {worker.get('error')}",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
        result["worker_recovery_required"] = bool(
            worker.get("worker_recovery_required")
            or _is_runtime_safety_failure(error)
        )
        return result
    output_records = worker.get("outputs")
    if not isinstance(output_records, list) or len(output_records) != len(correctness_seeds):
        raise EnvironmentRunnerError("CANDIDATE_OUTPUT_MANIFEST_INVALID")
    stages["tpu_compile"] = True
    output_paths: list[Path] = []
    for seed, record in zip(correctness_seeds, output_records, strict=True):
        if not isinstance(record, dict) or record.get("seed") != seed:
            raise EnvironmentRunnerError("CANDIDATE_OUTPUT_SEED_INVALID")
        output_path = (candidate_root / str(record.get("path"))).resolve()
        if (
            not output_path.is_relative_to(candidate_root.resolve())
            or not output_path.is_file()
            or _sha256_file(output_path) != record.get("sha256")
        ):
            raise EnvironmentRunnerError("CANDIDATE_OUTPUT_ARTIFACT_INVALID")
        output_paths.append(output_path)
    if worker.get("phase") == "profile":
        for output_path in output_paths:
            output_path.unlink()
        return _failed(
            stage="profile",
            error=worker.get("error", "PROFILE_CAPTURE_FAILED"),
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    profile_root = candidate_root / "profile" / "candidate"
    try:
        admission = validate_execution_evidence(profile_root)
    except Exception as exc:  # noqa: BLE001 - evidence failure is attributable
        for output_path in output_paths:
            output_path.unlink()
        stage = (
            "normal_lowering"
            if "TPU_CUSTOM_CALL_MISSING" in str(exc)
            else "profile"
        )
        return _failed(
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["normal_lowering"] = True
    stages["runtime_safety"] = True
    profile = worker.get("profile")
    timing = worker.get("timing")
    if not isinstance(profile, dict) or not isinstance(timing, dict):
        raise EnvironmentRunnerError("CANDIDATE_PROFILE_RESULT_INVALID")
    timing["validation"] = validate_timing_result(timing, seed=0)
    profile["admission"] = admission
    profile["timing"] = timing
    profile["speedup"] = timing.get("speedup")
    stages["profile"] = True
    seed_results = []
    for seed, record, output_path in zip(
        correctness_seeds, output_records, output_paths, strict=True
    ):
        inputs = generate_inputs(
            task["input_shapes"],
            task.get("input_dtypes"),
            task.get("correctness_input_ranges", task.get("input_ranges")),
            seed=seed,
        )
        expected_device = semantic_oracle(task, *inputs)
        expected_dtype = str(expected_device.dtype)
        if record.get("logical_dtype") != expected_dtype:
            result = _failed(
                stage="full_shape_correctness",
                error=(
                    f"seed={seed} OUTPUT_DTYPE_MISMATCH:"
                    f"actual={record.get('logical_dtype')} expected={expected_dtype}"
                ),
                hardware=hardware,
                kernel_sha256=kernel_sha256,
                stages=stages,
            )
            result["seed"] = seed
            for remaining in output_paths:
                remaining.unlink(missing_ok=True)
            return result
        expected = np.asarray(jax.device_get(expected_device))
        if expected_dtype == "bfloat16":
            expected = expected.astype(np.float32)
        actual = consume_candidate_array(output_path)
        try:
            validate_candidate_output(
                actual=actual,
                expected=expected,
                rtol=float(tolerance["rtol"]),
                atol=float(tolerance["atol"]),
            )
        except AssertionError as exc:
            result = _failed(
                stage="full_shape_correctness",
                error=f"seed={seed} {type(exc).__name__}: {exc}",
                hardware=hardware,
                kernel_sha256=kernel_sha256,
                stages=stages,
            )
            result["seed"] = seed
            for remaining in output_paths:
                remaining.unlink(missing_ok=True)
            return result
        seed_results.append({"seed": seed, "passed": True})
    stages["full_shape_correctness"] = True
    return {
        "passed": True,
        "stage": "verified",
        "error": None,
        "hardware": hardware,
        "kernel_sha256": kernel_sha256,
        "stages": stages,
        "authentic": True,
        "correct": True,
        "normal_lowered": True,
        "infrastructure_error": False,
        "seed_results": seed_results,
        "executable_tpu_custom_call": True,
        "profile": profile,
    }


def evaluate_repair_run(run_dir: Path, evidence_dir: Path | None) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for row in rows:
        final_attempt = row["attempts"][-1]
        result = evaluate_task(
            task=row["task"],
            kernel_path=run_dir / final_attempt["kernel_path"],
            evidence_dir=(
                evidence_dir / row["task"]["task_id"]
                if evidence_dir is not None
                else None
            ),
        )
        results.append(
            {
                "task_id": row["task"]["task_id"],
                "attempt": final_attempt["attempt"],
                **result,
            }
        )
    return {
        "passed": all(result["passed"] for result in results),
        "verified": sum(result["passed"] for result in results),
        "task_count": len(results),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-environment-runner")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--repair-run", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.repair_run is not None:
            if args.task is not None or args.kernel is not None:
                raise EnvironmentRunnerError("RUNNER_ARGUMENT_CONFLICT")
            result = evaluate_repair_run(args.repair_run, args.evidence_dir)
        else:
            if args.task is None or args.kernel is None:
                raise EnvironmentRunnerError("TASK_AND_KERNEL_REQUIRED")
            task = json.loads(args.task.read_text(encoding="utf-8"))
            result = evaluate_task(
                task=task,
                kernel_path=args.kernel,
                evidence_dir=args.evidence_dir,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must preserve infrastructure failures
        result = {
            "passed": False,
            "stage": "infrastructure",
            "error": f"{type(exc).__name__}: {exc}",
            "infrastructure_error": True,
        }
    if args.evidence_dir is not None:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
