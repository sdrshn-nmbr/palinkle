"""Original-shape TPU scoreability probe for the frozen JAXBench release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp

from opjax.pallas.jaxbench_executable import file_sha256, runtime_fingerprint


WORKER_REQUIREMENTS_LOCK = (
    Path(__file__).parents[3] / "config/pallas/phase2-worker-requirements.lock"
)


class JaxBenchScoreabilityError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise JaxBenchScoreabilityError(f"MODULE_LOAD_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _create_inputs(module: ModuleType) -> tuple[Any, ...]:
    create_inputs = getattr(module, "create_inputs", None)
    if not callable(create_inputs):
        raise JaxBenchScoreabilityError("BASELINE_CREATE_INPUTS_MISSING")
    parameters = inspect.signature(create_inputs).parameters
    values = (
        create_inputs(dtype=jnp.bfloat16)
        if "dtype" in parameters
        else create_inputs()
    )
    return tuple(values) if isinstance(values, (tuple, list)) else (values,)


def _array_schema(value: Any) -> dict[str, Any]:
    array = jnp.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def probe_task(*, release_root: Path, task_id: str) -> dict[str, Any]:
    release = json.loads((release_root / "manifest.json").read_text())
    task_record = next(
        (task for task in release["tasks"] if task["task_id"] == task_id), None
    )
    if task_record is None:
        raise JaxBenchScoreabilityError(f"TASK_UNKNOWN:{task_id}")
    task_root = release_root / task_record["path"]
    task = json.loads((task_root / "tests/task.json").read_text())
    baseline_path = task_root / "tests/jaxbench/baseline.py"
    if file_sha256(baseline_path) != task_record["baseline_sha256"]:
        raise JaxBenchScoreabilityError("BASELINE_HASH_INVALID")
    if jax.default_backend() != "tpu":
        raise JaxBenchScoreabilityError("TPU_BACKEND_REQUIRED")

    started = time.perf_counter()
    runtime = runtime_fingerprint()
    record = {
        "schema_version": 1,
        "task_id": task_id,
        "task_sha256": task_record["task_sha256"],
        "baseline_sha256": task_record["baseline_sha256"],
        "release_sha256": release["release_sha256"],
        "candidate_attributable": False,
        "runtime": runtime,
        "worker_requirements_lock_sha256": file_sha256(WORKER_REQUIREMENTS_LOCK),
    }
    stage = "module_load"
    try:
        module = _load_module(baseline_path, f"opjax_scoreability_{task_id}")
        stage = "input_creation"
        inputs = _create_inputs(module)
        expected_inputs = [
            {"shape": value["shape"], "dtype": value["dtype"]}
            for value in task["tensor_schema"]["inputs"]
        ]
        observed_inputs = [_array_schema(value) for value in inputs]
        if observed_inputs != expected_inputs:
            raise JaxBenchScoreabilityError(
                f"INPUT_SCHEMA_INVALID:{observed_inputs!r}:{expected_inputs!r}"
            )
        function = jax.jit(module.workload)
        stage = "lower_compile"
        lower_started = time.perf_counter()
        lowered = function.lower(*inputs)
        compiled = lowered.compile()
        compile_ms = (time.perf_counter() - lower_started) * 1000.0
        stage = "execute"
        execute_started = time.perf_counter()
        output = compiled(*inputs)
        jax.block_until_ready(output)
        execute_ms = (time.perf_counter() - execute_started) * 1000.0
        stage = "output_schema"
        output_leaves = jax.tree.leaves(output)
        expected_outputs = [
            {"shape": value["shape"], "dtype": value["dtype"]}
            for value in task["tensor_schema"]["outputs"]
        ]
        observed_outputs = [_array_schema(value) for value in output_leaves]
        if observed_outputs != expected_outputs:
            raise JaxBenchScoreabilityError(
                f"OUTPUT_SCHEMA_INVALID:{observed_outputs!r}:{expected_outputs!r}"
            )
        executable_text = compiled.as_text()
        return {
            **record,
            "status": "scoreable",
            "stage": "execute",
            "compile_ms": compile_ms,
            "first_execute_ms": execute_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "input_schema": observed_inputs,
            "output_schema": expected_outputs,
            "executable_sha256": file_sha256_bytes(executable_text.encode()),
            "platform": jax.default_backend(),
            "device_count": jax.device_count(),
            "device_kinds": sorted(
                {
                    getattr(device, "device_kind", "unknown")
                    for device in jax.devices()
                }
            ),
        }
    except Exception as exc:
        return {
            **record,
            "status": "unscoreable",
            "stage": stage,
            "classification": "pinned_baseline_failure",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }


def file_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_child(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "task_id" in value:
            return value
    return None


def run_matrix(
    *,
    release_root: Path,
    out_dir: Path,
    task_ids: list[str] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchScoreabilityError(f"OUTPUT_EXISTS:{out_dir}")
    release = json.loads((release_root / "manifest.json").read_text())
    requirements_lock_sha256 = file_sha256(WORKER_REQUIREMENTS_LOCK)
    ordered = [task["task_id"] for task in release["tasks"]]
    selected = ordered if task_ids is None else task_ids
    if len(selected) != len(set(selected)) or not set(selected).issubset(ordered):
        raise JaxBenchScoreabilityError("TASK_SELECTION_INVALID")
    out_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for task_id in selected:
        command = [
            sys.executable,
            "-m",
            "opjax.pallas.jaxbench_scoreability",
            "task",
            "--release",
            str(release_root),
            "--task-id",
            task_id,
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={**os.environ, "JAX_PLATFORMS": "tpu"},
                timeout=timeout_seconds,
                check=False,
            )
            record = _parse_child(process.stdout)
            if process.returncode != 0 or record is None:
                record = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "task_sha256": next(
                        task["task_sha256"]
                        for task in release["tasks"]
                        if task["task_id"] == task_id
                    ),
                    "release_sha256": release["release_sha256"],
                    "status": "unscoreable",
                    "stage": "child_process",
                    "returncode": process.returncode,
                    "error": (process.stderr or process.stdout or "no output")[-4000:],
                    "total_ms": (time.perf_counter() - started) * 1000.0,
                    "worker_requirements_lock_sha256": requirements_lock_sha256,
                }
        except subprocess.TimeoutExpired as exc:
            record = {
                "schema_version": 1,
                "task_id": task_id,
                "task_sha256": next(
                    task["task_sha256"]
                    for task in release["tasks"]
                    if task["task_id"] == task_id
                ),
                "release_sha256": release["release_sha256"],
                "status": "unscoreable",
                "stage": "timeout",
                "returncode": None,
                "error": str(exc),
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "worker_requirements_lock_sha256": requirements_lock_sha256,
            }
        results.append(record)
        (out_dir / f"{task_id}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(record, sort_keys=True), flush=True)
    scoreable = sum(result["status"] == "scoreable" for result in results)
    observed_runtimes = [result.get("runtime") for result in results]
    encoded_runtimes = {
        json.dumps(runtime, sort_keys=True) for runtime in observed_runtimes
        if isinstance(runtime, dict)
    }
    runtime = (
        observed_runtimes[0]
        if len(encoded_runtimes) == 1
        and all(isinstance(value, dict) for value in observed_runtimes)
        else None
    )
    matrix = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_original_shape_scoreability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_sha256": release["release_sha256"],
        "runner_sha256": file_sha256(Path(__file__)),
        "runtime": runtime,
        "worker_requirements_lock_sha256": requirements_lock_sha256,
        "task_count": len(results),
        "scoreable_count": scoreable,
        "unscoreable_count": len(results) - scoreable,
        "complete_release": selected == ordered,
        "results": results,
    }
    (out_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    )
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-scoreability")
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("task")
    task.add_argument("--release", type=Path, required=True)
    task.add_argument("--task-id", required=True)
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--release", type=Path, required=True)
    matrix.add_argument("--out", type=Path, required=True)
    matrix.add_argument("--task-id", action="append", dest="task_ids")
    matrix.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "task":
        result = probe_task(release_root=args.release, task_id=args.task_id)
    else:
        result = run_matrix(
            release_root=args.release,
            out_dir=args.out,
            task_ids=args.task_ids,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
