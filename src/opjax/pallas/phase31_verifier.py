"""Pristine multi-case verifier for the Phase 3.1 JAXBench contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
from jax.experimental import serialize_executable

from opjax.pallas.benchmarking import measure_interleaved, validate_timing_result
from opjax.pallas.jaxbench_executable import expected_trees, file_sha256
from opjax.pallas.jaxbench_verifier import (
    JaxBenchVerifierError,
    _capture_profile,
    _create_inputs,
    _load_module,
    _ready,
    _time_call,
    _validate_input_schema,
    inspect_pallas_owned_hlo,
    profile_proves_tpu_execution,
)
from opjax.pallas.phase31_oracle import compare_output, derive_input_case


def _write(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _failure(
    *, task_id: str, stage: str, error: str, authenticity: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task_id": task_id,
        "passed": False,
        "stage": stage,
        "error": error,
        "candidate_attributable": True,
        "infrastructure_error": False,
        "correct": False,
        "authentic": authenticity.get("authentic", False),
        "profiled": False,
        "authenticity": authenticity,
        "reward": 0,
    }


def verify_serialized_submission(
    *,
    task: dict[str, Any],
    baseline_path: Path,
    compiled_dir: Path,
    out_dir: Path,
    require_tpu: bool = True,
    require_pallas: bool = True,
    timing_rounds: int = 9,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchVerifierError(f"OUTPUT_EXISTS:{out_dir}")
    out_dir.mkdir(parents=True)
    if require_tpu and jax.default_backend() != "tpu":
        raise JaxBenchVerifierError("TPU_BACKEND_REQUIRED")
    if file_sha256(baseline_path) != task["baseline_sha256"]:
        raise JaxBenchVerifierError("BASELINE_HASH_INVALID")
    oracle = task.get("oracle_contract")
    if not isinstance(oracle, dict) or oracle.get("schema_version") != 1:
        raise JaxBenchVerifierError("PHASE31_ORACLE_CONTRACT_REQUIRED")
    compile_record = json.loads((compiled_dir / "compile.json").read_text())
    executable_path = compiled_dir / "executable.bin"
    if (
        compile_record.get("task_id") != task["task_id"]
        or compile_record.get("executable_sha256") != file_sha256(executable_path)
    ):
        raise JaxBenchVerifierError("COMPILED_ARTIFACT_BINDING_INVALID")
    in_tree, out_tree = expected_trees(task)
    loaded = serialize_executable.deserialize_and_load(
        executable_path.read_bytes(), in_tree, out_tree, backend=jax.default_backend()
    )
    executable_hlo = loaded.as_text()
    hlo_path = out_dir / "trusted-executable.hlo.txt"
    hlo_path.write_text(executable_hlo, encoding="utf-8")
    authenticity = inspect_pallas_owned_hlo(executable_hlo)
    if require_pallas and not authenticity["authentic"]:
        return _write(
            out_dir / "result.json",
            _failure(
                task_id=task["task_id"],
                stage="normal_lowering",
                error=authenticity["reason"],
                authenticity=authenticity,
            ),
        )

    baseline = _load_module(baseline_path, f"{task['task_id']}.phase31_baseline")
    original_inputs = _create_inputs(baseline)
    _validate_input_schema(original_inputs, task)
    baseline_workload = getattr(baseline, "workload", None)
    if not callable(baseline_workload):
        raise JaxBenchVerifierError("BASELINE_WORKLOAD_MISSING")
    baseline_function = (
        baseline_workload if getattr(baseline, "_skip_jit", False) else jax.jit(baseline_workload)
    )
    correctness_cases = []
    for seed, case_name in enumerate(oracle["input_cases"]):
        inputs = derive_input_case(original_inputs, contract=oracle, seed=seed)
        _validate_input_schema(inputs, task)
        try:
            expected = baseline_function(*inputs)
            actual = loaded(*inputs)
            _ready((expected, actual))
        except Exception as exc:
            raise JaxBenchVerifierError(
                f"BASELINE_OR_EXECUTABLE_NOT_SCOREABLE:{case_name}:{type(exc).__name__}:{exc}"
            ) from exc
        comparison = compare_output(expected, actual, contract=oracle)
        correctness_cases.append({"case": case_name, **comparison})
        if not comparison["correct"]:
            result = _failure(
                task_id=task["task_id"],
                stage="full_shape_correctness",
                error=f"VALUES_DIFFER:{case_name}",
                authenticity=authenticity,
            )
            result["correctness_cases"] = correctness_cases
            return _write(out_dir / "result.json", result)
    if not any(case["signal_max_abs"] > 0 for case in correctness_cases):
        raise JaxBenchVerifierError("ORACLE_SIGNAL_VACUOUS_ALL_CASES")

    profile = _capture_profile(function=loaded, inputs=original_inputs, out_dir=out_dir)
    if require_pallas and not profile_proves_tpu_execution(profile):
        result = _failure(
            task_id=task["task_id"],
            stage="profile",
            error="PERFETTO_TPU_EXECUTION_EVIDENCE_MISSING",
            authenticity=authenticity,
        )
        result["correct"] = True
        result["correctness_cases"] = correctness_cases
        result["profile"] = profile
        return _write(out_dir / "result.json", result)

    performance_eligible = correctness_cases[0]["signal_max_abs"] > 0
    timing = None
    if performance_eligible:
        timing = measure_interleaved(
            candidate=lambda: _time_call(loaded, original_inputs),
            baseline=lambda: _time_call(baseline_function, original_inputs),
            rounds=timing_rounds,
            seed=0,
            material_speedup=1.05,
        )
        timing["validation"] = validate_timing_result(timing, seed=0)
    result = {
        "schema_version": 2,
        "task_id": task["task_id"],
        "passed": True,
        "stage": "verified",
        "error": None,
        "candidate_attributable": False,
        "infrastructure_error": False,
        "correct": True,
        "authentic": True,
        "authenticity": authenticity,
        "profiled": True,
        "correctness_cases": correctness_cases,
        "profile": profile,
        "performance_eligible": performance_eligible,
        "timing": timing,
        "speedup": timing["speedup"] if timing else None,
        "beats_xla": timing["materially_beats_xla"] if timing else False,
        "reward": 1,
        "trusted_executable_hlo_sha256": file_sha256(hlo_path),
    }
    return _write(out_dir / "result.json", result)
