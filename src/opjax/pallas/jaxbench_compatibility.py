"""Separate runtime-compatibility probes for unscoreable JAXBench tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from opjax.pallas.jaxbench_executable import file_sha256, runtime_fingerprint
from opjax.pallas.jaxbench_scoreability import (
    WORKER_REQUIREMENTS_LOCK,
    _array_schema,
    _create_inputs,
    _load_module,
    file_sha256_bytes,
)


COMPATIBILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "11p_Megablox_GMM": {
        "adapter": "jax_jit_static_argnums",
        "static_argnums": [3],
        "required_accelerator_families": [],
        "reason": "pinned_benchmark_declares_max_expert_size_static",
    },
    "16p_Mamba2_SSD": {
        "adapter": "batch_axis_sharding",
        "static_argnums": [],
        "required_accelerator_families": ["v5litepod"],
        "minimum_device_count": 4,
        "reason": "shard_global_batch_four_across_four_v5e_devices",
    },
    "2p_GQA_Attention": {
        "adapter": "batch_axis_sharding",
        "static_argnums": [],
        "required_accelerator_families": ["v5litepod"],
        "minimum_device_count": 4,
        "reason": "shard_global_batch_four_across_four_v5e_devices",
    },
}


class JaxBenchCompatibilityError(RuntimeError):
    pass


def compatibility_runner_sha256() -> str:
    return file_sha256(Path(__file__))


def compatibility_contract(task_id: str) -> dict[str, Any]:
    try:
        contract = COMPATIBILITY_CONTRACTS[task_id]
    except KeyError as exc:
        raise JaxBenchCompatibilityError(
            f"TASK_NOT_IN_COMPATIBILITY_LANE:{task_id}"
        ) from exc
    return json.loads(json.dumps(contract, sort_keys=True))


def _accelerator_family(accelerator_type: str) -> str:
    for family in ("v5litepod", "v5p", "v6e"):
        if accelerator_type.startswith(family):
            return family
    return accelerator_type.split("-", 1)[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JaxBenchCompatibilityError(message)


def _dynamic_inputs(inputs: tuple[Any, ...], static_argnums: list[int]) -> tuple[Any, ...]:
    static = set(static_argnums)
    return tuple(value for index, value in enumerate(inputs) if index not in static)


def _batch_sharding(rank: int, mesh: Mesh) -> NamedSharding:
    if rank < 1:
        return NamedSharding(mesh, PartitionSpec())
    return NamedSharding(mesh, PartitionSpec("data", *([None] * (rank - 1))))


def probe_task(*, release_root: Path, task_id: str) -> dict[str, Any]:
    contract = compatibility_contract(task_id)
    release = json.loads((release_root / "manifest.json").read_text())
    task_record = next(
        (task for task in release["tasks"] if task["task_id"] == task_id), None
    )
    if task_record is None:
        raise JaxBenchCompatibilityError(f"TASK_UNKNOWN:{task_id}")
    task_root = release_root / task_record["path"]
    task = json.loads((task_root / "tests/task.json").read_text())
    baseline_path = task_root / "tests/jaxbench/baseline.py"
    if file_sha256(baseline_path) != task_record["baseline_sha256"]:
        raise JaxBenchCompatibilityError("BASELINE_HASH_INVALID")
    if jax.default_backend() != "tpu":
        raise JaxBenchCompatibilityError("TPU_BACKEND_REQUIRED")

    accelerator_type = os.environ.get("TPU_ACCELERATOR_TYPE", "unknown")
    required_families = contract["required_accelerator_families"]
    if required_families and _accelerator_family(accelerator_type) not in required_families:
        raise JaxBenchCompatibilityError(
            f"ACCELERATOR_FAMILY_INCOMPATIBLE:{accelerator_type}"
        )

    started = time.perf_counter()
    runtime = {
        **runtime_fingerprint(),
        "accelerator_type": accelerator_type,
    }
    record = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_compatibility_probe",
        "task_id": task_id,
        "task_sha256": task_record["task_sha256"],
        "baseline_sha256": task_record["baseline_sha256"],
        "release_sha256": release["release_sha256"],
        "candidate_attributable": False,
        "execution_contract": contract,
        "runner_sha256": compatibility_runner_sha256(),
        "runtime": runtime,
        "worker_requirements_lock_sha256": file_sha256(WORKER_REQUIREMENTS_LOCK),
    }
    stage = "module_load"
    try:
        module = _load_module(baseline_path, f"opjax_compatibility_{task_id}")
        stage = "input_creation"
        inputs = _create_inputs(module)
        expected_inputs = [
            {"shape": value["shape"], "dtype": value["dtype"]}
            for value in task["tensor_schema"]["inputs"]
        ]
        observed_inputs = [_array_schema(value) for value in inputs]
        if observed_inputs != expected_inputs:
            raise JaxBenchCompatibilityError(
                f"INPUT_SCHEMA_INVALID:{observed_inputs!r}:{expected_inputs!r}"
            )

        adapter_device_count = 1
        compiled_inputs = inputs
        if contract["adapter"] == "batch_axis_sharding":
            adapter_device_count = contract["minimum_device_count"]
            if jax.device_count() < adapter_device_count:
                raise JaxBenchCompatibilityError(
                    f"DEVICE_COUNT_INCOMPATIBLE:{jax.device_count()}"
                )
            mesh = Mesh(jax.devices()[:adapter_device_count], ("data",))
            input_shardings = tuple(
                _batch_sharding(value.ndim, mesh) for value in inputs
            )
            output_rank = len(task["tensor_schema"]["outputs"][0]["shape"])
            function = jax.jit(
                module.workload,
                in_shardings=input_shardings,
                out_shardings=_batch_sharding(output_rank, mesh),
            )
            compiled_inputs = tuple(
                jax.device_put(value, sharding)
                for value, sharding in zip(inputs, input_shardings, strict=True)
            )
        else:
            function = jax.jit(
                module.workload,
                static_argnums=tuple(contract["static_argnums"]),
            )
        stage = "lower_compile"
        compile_started = time.perf_counter()
        compiled = function.lower(*compiled_inputs).compile()
        compile_ms = (time.perf_counter() - compile_started) * 1000.0
        stage = "execute"
        execute_started = time.perf_counter()
        output = compiled(
            *_dynamic_inputs(compiled_inputs, contract["static_argnums"])
        )
        jax.block_until_ready(output)
        execute_ms = (time.perf_counter() - execute_started) * 1000.0
        stage = "output_schema"
        observed_outputs = [_array_schema(value) for value in jax.tree.leaves(output)]
        expected_outputs = [
            {"shape": value["shape"], "dtype": value["dtype"]}
            for value in task["tensor_schema"]["outputs"]
        ]
        if observed_outputs != expected_outputs:
            raise JaxBenchCompatibilityError(
                f"OUTPUT_SCHEMA_INVALID:{observed_outputs!r}:{expected_outputs!r}"
            )
        return {
            **record,
            "status": "scoreable",
            "stage": "execute",
            "compile_ms": compile_ms,
            "first_execute_ms": execute_ms,
            "adapter_device_count": adapter_device_count,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "input_schema": observed_inputs,
            "output_schema": observed_outputs,
            "executable_sha256": file_sha256_bytes(compiled.as_text().encode()),
        }
    except Exception as exc:
        return {
            **record,
            "status": "unscoreable",
            "stage": stage,
            "classification": "compatibility_probe_failure",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "traceback": traceback.format_exc()[-8000:],
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }


def validate_compatibility_evidence(
    *, release_manifest_path: Path, evidence_path: Path
) -> dict[str, Any]:
    release = json.loads(release_manifest_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    task_id = evidence.get("task_id")
    contract = compatibility_contract(task_id)
    task = next(
        (value for value in release["tasks"] if value["task_id"] == task_id), None
    )
    _require(task is not None, "TASK_UNKNOWN")
    _require(evidence.get("schema_version") == 1, "SCHEMA_VERSION_INVALID")
    _require(
        evidence.get("kind") == "opjax_jaxbench_compatibility_probe",
        "KIND_INVALID",
    )
    _require(evidence.get("release_sha256") == release["release_sha256"], "RELEASE_INVALID")
    _require(evidence.get("task_sha256") == task["task_sha256"], "TASK_HASH_INVALID")
    _require(
        evidence.get("baseline_sha256") == task["baseline_sha256"],
        "BASELINE_HASH_INVALID",
    )
    _require(evidence.get("execution_contract") == contract, "CONTRACT_INVALID")
    _require(
        evidence.get("runner_sha256") == compatibility_runner_sha256(),
        "RUNNER_HASH_INVALID",
    )
    _require(evidence.get("status") == "scoreable", "STATUS_INVALID")
    _require(evidence.get("stage") == "execute", "STAGE_INVALID")
    _require(bool(evidence.get("executable_sha256")), "EXECUTABLE_HASH_MISSING")
    runtime = evidence.get("runtime", {})
    _require(runtime.get("backend") == "tpu", "TPU_BACKEND_REQUIRED")
    required_families = contract["required_accelerator_families"]
    accelerator_type = runtime.get("accelerator_type", "unknown")
    _require(
        not required_families
        or _accelerator_family(accelerator_type) in required_families,
        f"ACCELERATOR_FAMILY_INCOMPATIBLE:{accelerator_type}",
    )
    minimum_device_count = contract.get("minimum_device_count", 1)
    _require(
        runtime.get("device_count", 0) >= minimum_device_count,
        f"DEVICE_COUNT_INCOMPATIBLE:{runtime.get('device_count', 0)}",
    )
    _require(
        evidence.get("adapter_device_count", 1) == minimum_device_count,
        "ADAPTER_DEVICE_COUNT_INVALID",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-compatibility")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--release", type=Path, required=True)
    probe.add_argument("--task-id", required=True)
    probe.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--release-manifest", type=Path, required=True)
    validate.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        result = probe_task(release_root=args.release, task_id=args.task_id)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = validate_compatibility_evidence(
            release_manifest_path=args.release_manifest,
            evidence_path=args.evidence,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "scoreable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
