"""Pristine verifier for serialized JAXBench TPU executables."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import inspect
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental import serialize_executable

from opjax.pallas.benchmarking import measure_interleaved, validate_timing_result
from opjax.pallas.jaxbench_executable import expected_trees, file_sha256


ATOL = 1e-2
RTOL = 1e-2
STRUCTURAL_HLO_OPERATIONS = {
    "after-all",
    "bitcast",
    "constant",
    "convert",
    "copy",
    "get-tuple-element",
    "parameter",
    "reshape",
    "transpose",
    "tuple",
}


class JaxBenchVerifierError(RuntimeError):
    pass


def inspect_pallas_owned_hlo(hlo: str) -> dict[str, Any]:
    entry_match = re.search(r"\bENTRY\s+%[^\{]+\{(?P<body>.*?)\n\}", hlo, re.DOTALL)
    if entry_match is None:
        return {"authentic": False, "reason": "HLO_ENTRY_MISSING"}
    instructions: dict[str, dict[str, Any]] = {}
    root_name = None
    for raw_line in entry_match.group("body").splitlines():
        line = raw_line.strip()
        match = re.match(r"(?P<root>ROOT\s+)?%(?P<name>[^\s=]+)\s*=\s*(?P<body>.+)", line)
        if match is None:
            continue
        operation_match = re.search(
            r"\b(?P<operation>[a-z][a-z0-9-]*)\(", match.group("body")
        )
        if operation_match is None:
            return {"authentic": False, "reason": "HLO_INSTRUCTION_INVALID"}
        operation = operation_match.group("operation")
        after_call = match.group("body")[operation_match.end() :]
        operands = re.findall(r"%([^\s,()]+)", after_call.partition(")")[0])
        is_tpu_custom_call = (
            operation == "custom-call"
            and 'custom_call_target="tpu_custom_call"' in after_call
        )
        has_pallas_op_name = bool(
            re.search(r'metadata=\{[^}]*op_name="(?:[^"]*/)?pallas_call"', after_call)
        )
        has_pallas_kernel_metadata = bool(
            re.search(r"frontend_attributes=\{kernel_metadata=\{", after_call)
        )
        is_pallas_custom_call = (
            is_tpu_custom_call
            and has_pallas_op_name
            and has_pallas_kernel_metadata
        )
        instructions[match.group("name")] = {
            "operation": operation,
            "operands": operands,
            "is_tpu_custom_call": is_tpu_custom_call,
            "is_pallas_custom_call": is_pallas_custom_call,
        }
        if match.group("root"):
            root_name = match.group("name")
    if root_name is None or root_name not in instructions:
        return {"authentic": False, "reason": "HLO_ROOT_MISSING"}
    reachable: set[str] = set()
    pending = [root_name]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        instruction = instructions.get(name)
        if instruction is None:
            return {"authentic": False, "reason": f"HLO_OPERAND_UNKNOWN:{name}"}
        reachable.add(name)
        pending.extend(instruction["operands"])
    operations = sorted({instructions[name]["operation"] for name in reachable})
    disallowed = sorted(
        {
            operation
            for operation in operations
            if operation not in STRUCTURAL_HLO_OPERATIONS and operation != "custom-call"
        }
    )
    custom_calls = [
        name
        for name in reachable
        if instructions[name]["operation"] == "custom-call"
    ]
    non_pallas_custom_calls = [
        name for name in custom_calls if not instructions[name]["is_pallas_custom_call"]
    ]
    pallas_custom_calls = [
        name for name in custom_calls if instructions[name]["is_pallas_custom_call"]
    ]
    ownership_cache: dict[str, bool] = {}

    def is_pallas_owned(name: str) -> bool:
        if name in ownership_cache:
            return ownership_cache[name]
        instruction = instructions[name]
        if instruction["is_pallas_custom_call"]:
            owned = True
        elif instruction["operation"] in {"parameter", "constant"}:
            owned = False
        else:
            operands = instruction["operands"]
            owned = bool(operands) and all(is_pallas_owned(item) for item in operands)
        ownership_cache[name] = owned
        return owned

    if disallowed:
        reason = f"HLO_COMPUTE_OUTSIDE_PALLAS:{','.join(disallowed)}"
    elif non_pallas_custom_calls:
        reason = "HLO_NON_PALLAS_CUSTOM_CALL_REACHABLE"
    elif not pallas_custom_calls:
        reason = "TPU_CUSTOM_CALL_NOT_OUTPUT_REACHABLE"
    elif not is_pallas_owned(root_name):
        reason = "HLO_RESULT_NOT_PALLAS_OWNED"
    else:
        reason = "ok"
    return {
        "authentic": reason == "ok",
        "reason": reason,
        "root": root_name,
        "reachable_instruction_count": len(reachable),
        "reachable_operations": operations,
        "tpu_custom_call_count": sum(
            instructions[name]["is_tpu_custom_call"] for name in reachable
        ),
        "pallas_custom_call_count": len(pallas_custom_calls),
        "result_pallas_owned": reason == "ok",
    }


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise JaxBenchVerifierError(f"MODULE_LOAD_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _create_inputs(module: ModuleType) -> tuple[Any, ...]:
    create_inputs = getattr(module, "create_inputs", None)
    if not callable(create_inputs):
        raise JaxBenchVerifierError("BASELINE_CREATE_INPUTS_MISSING")
    parameters = inspect.signature(create_inputs).parameters
    value = (
        create_inputs(dtype=jnp.bfloat16)
        if "dtype" in parameters
        else create_inputs()
    )
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _validate_input_schema(inputs: tuple[Any, ...], task: dict[str, Any]) -> None:
    schema = task["tensor_schema"]["inputs"]
    if len(inputs) != len(schema):
        raise JaxBenchVerifierError("BASELINE_INPUT_ARITY_INVALID")
    for value, expected in zip(inputs, schema, strict=True):
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
            raise JaxBenchVerifierError(
                "BASELINE_INPUT_SCHEMA_INVALID:"
                f"expected={expected}:observed_shape={list(value.shape)}:"
                f"observed_dtype={value.dtype}"
            )


def _ready(value: Any) -> None:
    jax.block_until_ready(value)


def _time_call(function: Callable[..., Any], inputs: tuple[Any, ...]) -> float:
    started = time.perf_counter()
    output = function(*inputs)
    _ready(output)
    return (time.perf_counter() - started) * 1000.0


def _trace_events(trace_root: Path) -> tuple[list[dict[str, Any]], Path]:
    matches = sorted(trace_root.rglob("perfetto_trace.json.gz"))
    if len(matches) != 1:
        raise JaxBenchVerifierError(
            f"PERFETTO_TRACE_COUNT_INVALID:{len(matches)}"
        )
    with gzip.open(matches[0], "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise JaxBenchVerifierError("PERFETTO_EVENTS_INVALID")
    return [event for event in events if isinstance(event, dict)], matches[0]


def _capture_profile(
    *, function: Callable[..., Any], inputs: tuple[Any, ...], out_dir: Path
) -> dict[str, Any]:
    trace_root = out_dir / "trace"
    if trace_root.exists():
        shutil.rmtree(trace_root)
    with jax.profiler.trace(
        str(trace_root), create_perfetto_link=False, create_perfetto_trace=True
    ):
        for _ in range(3):
            with jax.profiler.TraceAnnotation("opjax_jaxbench_candidate"):
                output = function(*inputs)
                _ready(output)
    events, trace_path = _trace_events(trace_root)
    names = [
        str(event.get("name", ""))
        for event in events
        if isinstance(event.get("dur"), (int, float)) and event["dur"] > 0
    ]
    counts = Counter(names)
    custom_calls = sorted({name for name in names if "tpu_custom_call" in name.lower()})
    annotations = sum(name == "opjax_jaxbench_candidate" for name in names)
    return {
        "trace_path": trace_path.relative_to(out_dir).as_posix(),
        "trace_sha256": file_sha256(trace_path),
        "duration_event_count": len(names),
        "candidate_annotation_count": annotations,
        "tpu_custom_call_events": custom_calls,
        "tpu_execute_event_count": counts["tpu::System::Execute"],
        "loaded_executable_event_count": counts["PJRT_LoadedExecutable_Execute"],
    }


def profile_proves_tpu_execution(profile: dict[str, Any]) -> bool:
    return (
        profile.get("candidate_annotation_count", 0) >= 3
        and profile.get("tpu_execute_event_count", 0) >= 3
        and profile.get("loaded_executable_event_count", 0) >= 3
    )


def _correctness(expected: Any, actual: Any) -> dict[str, Any]:
    if not isinstance(expected, jax.Array) or not isinstance(actual, jax.Array):
        return {"correct": False, "reason": "single array output required"}
    if expected.shape != actual.shape:
        return {
            "correct": False,
            "reason": f"shape mismatch: {expected.shape} vs {actual.shape}",
        }
    if expected.dtype != actual.dtype:
        return {
            "correct": False,
            "reason": f"dtype mismatch: {expected.dtype} vs {actual.dtype}",
        }
    expected_f32 = expected.astype(jnp.float32)
    actual_f32 = actual.astype(jnp.float32)
    max_diff = float(jax.device_get(jnp.max(jnp.abs(expected_f32 - actual_f32))))
    correct = bool(
        jax.device_get(jnp.allclose(expected_f32, actual_f32, rtol=RTOL, atol=ATOL))
    )
    return {
        "correct": correct,
        "max_diff": max_diff,
        "reason": "ok" if correct else "values differ",
        "rtol": RTOL,
        "atol": ATOL,
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
    compile_record = json.loads(
        (compiled_dir / "compile.json").read_text(encoding="utf-8")
    )
    executable_path = compiled_dir / "executable.bin"
    if (
        compile_record.get("task_id") != task["task_id"]
        or compile_record.get("executable_sha256") != file_sha256(executable_path)
    ):
        raise JaxBenchVerifierError("COMPILED_ARTIFACT_BINDING_INVALID")
    in_tree, out_tree = expected_trees(task)
    loaded = serialize_executable.deserialize_and_load(
        executable_path.read_bytes(),
        in_tree,
        out_tree,
        backend=jax.default_backend(),
    )
    executable_hlo = loaded.as_text()
    hlo_path = out_dir / "trusted-executable.hlo.txt"
    hlo_path.write_text(executable_hlo, encoding="utf-8")
    authenticity = inspect_pallas_owned_hlo(executable_hlo)
    if require_pallas and not authenticity["authentic"]:
        result = {
            "schema_version": 1,
            "task_id": task["task_id"],
            "passed": False,
            "stage": "normal_lowering",
            "error": authenticity["reason"],
            "candidate_attributable": True,
            "infrastructure_error": False,
            "correct": False,
            "authentic": False,
            "profiled": False,
            "authenticity": authenticity,
            "reward": 0,
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    baseline = _load_module(baseline_path, f"{task['task_id']}.hidden_baseline")
    inputs = _create_inputs(baseline)
    _validate_input_schema(inputs, task)
    baseline_workload = getattr(baseline, "workload", None)
    if not callable(baseline_workload):
        raise JaxBenchVerifierError("BASELINE_WORKLOAD_MISSING")
    baseline_function = (
        baseline_workload
        if getattr(baseline, "_skip_jit", False)
        else jax.jit(baseline_workload)
    )
    try:
        expected = baseline_function(*inputs)
        _ready(expected)
    except Exception as exc:
        raise JaxBenchVerifierError(
            f"BASELINE_NOT_SCOREABLE:{type(exc).__name__}:{exc}"
        ) from exc
    actual = loaded(*inputs)
    _ready(actual)
    correctness = _correctness(expected, actual)
    if not correctness["correct"]:
        result = {
            "schema_version": 1,
            "task_id": task["task_id"],
            "passed": False,
            "stage": "full_shape_correctness",
            "error": correctness["reason"],
            "candidate_attributable": True,
            "infrastructure_error": False,
            "correct": False,
            "authentic": authenticity["authentic"],
            "authenticity": authenticity,
            "profiled": False,
            "correctness": correctness,
            "reward": 0,
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    profile = _capture_profile(function=loaded, inputs=inputs, out_dir=out_dir)
    if require_pallas and not profile_proves_tpu_execution(profile):
        result = {
            "schema_version": 1,
            "task_id": task["task_id"],
            "passed": False,
            "stage": "profile",
            "error": "PERFETTO_TPU_EXECUTION_EVIDENCE_MISSING",
            "candidate_attributable": True,
            "infrastructure_error": False,
            "correct": True,
            "authentic": True,
            "authenticity": authenticity,
            "profiled": False,
            "correctness": correctness,
            "profile": profile,
            "reward": 0,
        }
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    timing = measure_interleaved(
        candidate=lambda: _time_call(loaded, inputs),
        baseline=lambda: _time_call(baseline_function, inputs),
        rounds=timing_rounds,
        seed=0,
        material_speedup=1.05,
    )
    timing["validation"] = validate_timing_result(timing, seed=0)
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "passed": True,
        "stage": "verified",
        "error": None,
        "candidate_attributable": False,
        "infrastructure_error": False,
        "correct": True,
        "authentic": authenticity["authentic"],
        "authenticity": authenticity,
        "profiled": True,
        "correctness": correctness,
        "profile": profile,
        "timing": timing,
        "speedup": timing["speedup"],
        "beats_xla": timing["materially_beats_xla"],
        "reward": 1,
        "trusted_executable_hlo_sha256": file_sha256(hlo_path),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-verifier")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cpu-test", action="store_true")
    parser.add_argument("--allow-plain-jax-test", action="store_true")
    parser.add_argument("--timing-rounds", type=int, default=9)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text(encoding="utf-8"))
    result = verify_serialized_submission(
        task=task,
        baseline_path=args.baseline,
        compiled_dir=args.compiled,
        out_dir=args.out,
        require_tpu=not args.allow_cpu_test,
        require_pallas=not args.allow_plain_jax_test,
        timing_rounds=args.timing_rounds,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
