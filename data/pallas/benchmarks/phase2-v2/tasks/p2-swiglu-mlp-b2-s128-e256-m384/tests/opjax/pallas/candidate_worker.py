"""Untrusted candidate execution process for the Pallas verifier."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chex
import jax
import numpy as np

from opjax.pallas.benchmarking import measure_interleaved
from opjax.pallas.lowering import capture_lowering_case
from opjax.pallas.task_semantics import generate_inputs, semantic_oracle


class CandidateWorkerError(RuntimeError):
    """The candidate process cannot produce its required execution artifacts."""


_ALLOWED_IMPORT_MODULES = {
    "jax",
    "jax.experimental",
    "jax.experimental.pallas",
    "jax.experimental.pallas.ops.tpu.megablox",
    "jax.numpy",
}


def _candidate_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level != 0 or name not in _ALLOWED_IMPORT_MODULES:
        raise CandidateWorkerError(f"CANDIDATE_IMPORT_NOT_ALLOWED:{name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _load_module(path: Path) -> ModuleType:
    module = ModuleType("opjax_untrusted_candidate")
    module.__dict__["__builtins__"] = {
        "__import__": _candidate_import,
        "float": float,
        "int": int,
        "len": len,
        "range": range,
        "tuple": tuple,
    }
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _time_compiled(
    compiled: Any, inputs: tuple[Any, ...], *, warmups: int = 3, iterations: int = 20
) -> float:
    for _ in range(warmups):
        jax.block_until_ready(compiled(*inputs))
    started = time.perf_counter()
    for _ in range(iterations):
        jax.block_until_ready(compiled(*inputs))
    return (time.perf_counter() - started) * 1000 / iterations


def _hardware(*, allow_cpu_test: bool) -> dict[str, Any]:
    if not allow_cpu_test:
        chex.assert_devices_available(1, "tpu", not_less_than=True)
    devices = jax.devices()
    return {
        "platforms": sorted({device.platform for device in devices}),
        "device_kinds": sorted(
            {getattr(device, "device_kind", "unknown") for device in devices}
        ),
        "device_count": len(devices),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }


def execute_candidate(
    *,
    task: dict[str, Any],
    kernel_path: Path,
    output_dir: Path,
    allow_cpu_test: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = _hardware(allow_cpu_test=allow_cpu_test)
    kernel_sha256 = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    try:
        workload = _load_module(kernel_path).workload
    except Exception as exc:  # noqa: BLE001 - candidate import is untrusted
        return {
            "passed": False,
            "phase": "compile",
            "seed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "hardware": hardware,
            "kernel_sha256": kernel_sha256,
        }
    compiled_for_profile = None
    inputs_for_profile = None
    outputs = []
    for seed in task["correctness_seeds"]:
        inputs = generate_inputs(
            task["input_shapes"],
            task.get("input_dtypes"),
            task.get("correctness_input_ranges", task.get("input_ranges")),
            seed=seed,
        )
        try:
            compiled = jax.jit(workload).lower(*inputs).compile()
        except Exception as exc:  # noqa: BLE001 - candidate compiler failure
            return {
                "passed": False,
                "phase": "compile",
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "hardware": hardware,
                "kernel_sha256": kernel_sha256,
            }
        try:
            actual = compiled(*inputs)
            jax.block_until_ready(actual)
            if not isinstance(actual, jax.Array):
                raise CandidateWorkerError("OUTPUT_ARRAY_REQUIRED")
            output_path = output_dir / f"seed-{seed}-actual.npy"
            host_actual = np.asarray(jax.device_get(actual))
            storage_actual = (
                host_actual.astype(np.float32)
                if str(actual.dtype) == "bfloat16"
                else host_actual
            )
            np.save(output_path, storage_actual, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - candidate execution failure
            return {
                "passed": False,
                "phase": "execute",
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "hardware": hardware,
                "kernel_sha256": kernel_sha256,
            }
        outputs.append(
            {
                "seed": seed,
                "path": output_path.name,
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "logical_dtype": str(actual.dtype),
            }
        )
        if compiled_for_profile is None:
            compiled_for_profile = compiled
            inputs_for_profile = inputs
    assert compiled_for_profile is not None and inputs_for_profile is not None
    timing_inputs = generate_inputs(
        task["input_shapes"],
        task.get("input_dtypes"),
        task.get("timing_input_ranges", task.get("input_ranges")),
        seed=0,
    )
    timing_tolerance = task.get(
        "timing_correctness_tolerance", task["correctness_tolerance"]
    )
    try:
        profile = capture_lowering_case(
            label="candidate",
            function=workload,
            inputs=timing_inputs,
            out_dir=output_dir / "profile",
            repetitions=3,
            expected_output=semantic_oracle(task, *timing_inputs),
            rtol=float(timing_tolerance["rtol"]),
            atol=float(timing_tolerance["atol"]),
        )
        baseline = (
            jax.jit(lambda *values: semantic_oracle(task, *values))
            .lower(*timing_inputs)
            .compile()
        )
        timing = measure_interleaved(
            candidate=lambda: _time_compiled(compiled_for_profile, timing_inputs),
            baseline=lambda: _time_compiled(baseline, timing_inputs),
            rounds=9,
            seed=0,
            material_speedup=1.05,
        )
    except Exception as exc:  # noqa: BLE001 - profile is candidate-attributable
        return {
            "passed": False,
            "phase": "profile",
            "error": f"{type(exc).__name__}: {exc}",
            "hardware": hardware,
            "kernel_sha256": kernel_sha256,
            "outputs": outputs,
        }
    return {
        "passed": True,
        "phase": "complete",
        "hardware": hardware,
        "kernel_sha256": kernel_sha256,
        "outputs": outputs,
        "profile": profile,
        "timing": timing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-candidate-worker")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-cpu-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute_candidate(
            task=json.loads(args.task.read_text(encoding="utf-8")),
            kernel_path=args.kernel,
            output_dir=args.output_dir,
            allow_cpu_test=args.allow_cpu_test,
        )
    except Exception as exc:  # noqa: BLE001 - process boundary preserves diagnostics
        result = {
            "passed": False,
            "phase": "infrastructure",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
