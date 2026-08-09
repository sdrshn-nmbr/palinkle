"""Empirical TPU lowering evidence for Pallas candidates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


class LoweringEvidenceError(RuntimeError):
    """A lowering probe cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class LoweringVerdict:
    verified: bool
    calibration_sha256: str
    candidate_sha256: str
    kernel_sha256: str
    runtime: dict[str, Any]
    reasons: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoweringEvidenceError(f"MODULE_LOAD_FAILED: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _create_inputs(module: ModuleType) -> tuple[Any, ...]:
    create_inputs = getattr(module, "create_inputs", None)
    if not callable(create_inputs):
        raise LoweringEvidenceError("CREATE_INPUTS_MISSING")
    inputs = create_inputs(dtype=jnp.bfloat16)
    return tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)


def _control_input() -> tuple[jax.Array]:
    value = jnp.arange(128 * 128, dtype=jnp.bfloat16).reshape((128, 128))
    return (value,)


def _pallas_add_one(x_ref: Any, out_ref: Any) -> None:
    out_ref[...] = x_ref[...] + jnp.asarray(1, dtype=x_ref.dtype)


def _normal_pallas_control(x: jax.Array) -> jax.Array:
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _pallas_add_one,
        out_shape=shape,
        grid=(1, 1),
        in_specs=[spec],
        out_specs=spec,
        interpret=False,
        name="opjax_normal_pallas_control",
    )(x)


def _interpreted_pallas_control(x: jax.Array) -> jax.Array:
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _pallas_add_one,
        out_shape=shape,
        grid=(1, 1),
        in_specs=[spec],
        out_specs=spec,
        interpret=True,
        name="opjax_interpreted_pallas_control",
    )(x)


def _plain_jax_control(x: jax.Array) -> jax.Array:
    return x + jnp.asarray(1, dtype=x.dtype)


def _dead_pallas_control(x: jax.Array) -> jax.Array:
    if False:
        return _normal_pallas_control(x)
    return x + jnp.asarray(1, dtype=x.dtype)


CONTROL_CASES: dict[str, Callable[[jax.Array], jax.Array]] = {
    "normal_pallas": _normal_pallas_control,
    "interpreted_pallas": _interpreted_pallas_control,
    "plain_jax": _plain_jax_control,
    "dead_pallas": _dead_pallas_control,
}


def _trace_events(trace_root: Path) -> tuple[list[dict[str, Any]], Path]:
    matches = sorted(trace_root.rglob("perfetto_trace.json.gz"))
    if len(matches) != 1:
        raise LoweringEvidenceError(
            f"PERFETTO_TRACE_COUNT_INVALID: expected=1 observed={len(matches)}"
        )
    trace_path = matches[0]
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise LoweringEvidenceError("PERFETTO_EVENTS_INVALID")
    return [event for event in events if isinstance(event, dict)], trace_path


def _summarise_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    duration_events = [
        event
        for event in events
        if isinstance(event.get("dur"), (int, float)) and event["dur"] > 0
    ]
    names = Counter(
        str(event.get("name"))
        for event in duration_events
        if isinstance(event.get("name"), str)
    )
    pallas_names = sorted(
        name
        for name in names
        if "pallas" in name.lower() or "mosaic" in name.lower()
    )
    return {
        "event_count": len(events),
        "duration_event_count": len(duration_events),
        "named_duration_event_count": sum(names.values()),
        "pallas_or_mosaic_event_names": pallas_names,
        "top_duration_event_names": [
            {"name": name, "count": count}
            for name, count in names.most_common(100)
        ],
    }


def _compiler_markers(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        marker: lowered.count(marker)
        for marker in (
            "mosaic",
            "pallas",
            "custom-call",
            "custom_call",
            "tpu_custom_call",
        )
    }


def _runtime_fingerprint() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kinds": sorted(
            {str(getattr(device, "device_kind", "unknown")) for device in devices}
        ),
        "chex": importlib.metadata.version("chex"),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "libtpu": (
            importlib.metadata.version("libtpu")
            if importlib.util.find_spec("libtpu") is not None
            else None
        ),
        "ml_dtypes": importlib.metadata.version("ml_dtypes"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "tomli": importlib.metadata.version("tomli"),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "machine": platform.machine(),
        "system": platform.system(),
        "kernel_release": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }


def _capture_tool_sha256() -> str:
    return _sha256_file(Path(__file__))


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoweringEvidenceError(f"EVIDENCE_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LoweringEvidenceError(f"EVIDENCE_OBJECT_INVALID: {path}")
    return value


def _event_count(trace: dict[str, Any], name: str) -> int:
    events = trace.get("top_duration_event_names")
    if not isinstance(events, list):
        return 0
    for event in events:
        if isinstance(event, dict) and event.get("name") == name:
            count = event.get("count")
            return count if isinstance(count, int) and not isinstance(count, bool) else 0
    return 0


def _marker_count(case: dict[str, Any], section: str, marker: str) -> int:
    compiler = case.get("compiler")
    if not isinstance(compiler, dict):
        return 0
    markers = compiler.get(section)
    if not isinstance(markers, dict):
        return 0
    count = markers.get(marker)
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def _validate_case_artifacts(case_dir: Path, case: dict[str, Any]) -> None:
    compiler = case.get("compiler")
    trace = case.get("trace")
    if not isinstance(compiler, dict) or not isinstance(trace, dict):
        raise LoweringEvidenceError(f"EVIDENCE_CASE_INVALID: {case_dir}")
    artifact_hashes = (
        ("stablehlo.mlir", compiler.get("stablehlo_sha256")),
        ("executable.hlo.txt", compiler.get("executable_hlo_sha256")),
    )
    trace_relative = trace.get("perfetto_relative_path")
    if not isinstance(trace_relative, str):
        raise LoweringEvidenceError(f"PERFETTO_PATH_INVALID: {case_dir}")
    artifact_hashes += ((trace_relative, trace.get("perfetto_sha256")),)
    for relative, expected in artifact_hashes:
        path = (case_dir / relative).resolve()
        if not path.is_relative_to(case_dir.resolve()) or not path.is_file():
            raise LoweringEvidenceError(f"EVIDENCE_ARTIFACT_MISSING: {path}")
        observed = _sha256_file(path)
        if expected != observed:
            raise LoweringEvidenceError(
                f"EVIDENCE_ARTIFACT_HASH_MISMATCH: {path}: "
                f"expected={expected} observed={observed}"
            )


def validate_execution_evidence(case_dir: Path) -> dict[str, Any]:
    """Admit one capture only with compiler and observed TPU execution proof."""
    case = _load_json_object(case_dir / "evidence.json")
    _validate_case_artifacts(case_dir, case)
    for section in ("stablehlo_markers", "executable_hlo_markers"):
        if _marker_count(case, section, "tpu_custom_call") < 1:
            raise LoweringEvidenceError(f"TPU_CUSTOM_CALL_MISSING:{section}")
    repetitions = case.get("repetitions")
    trace = case.get("trace")
    if case.get("correctness_verified") is not True:
        raise LoweringEvidenceError("PROFILE_CORRECTNESS_PROOF_MISSING")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
        or not isinstance(trace, dict)
        or _event_count(trace, "tpu::System::Execute=>Done") < repetitions
    ):
        raise LoweringEvidenceError("TRACE_EXECUTION_MISSING")
    return {
        "verified": True,
        "evidence_sha256": _sha256_file(case_dir / "evidence.json"),
        "execute_count": _event_count(trace, "tpu::System::Execute=>Done"),
        "repetitions": repetitions,
    }


def validate_calibration(
    calibration_root: Path,
    *,
    expected_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    calibration_path = calibration_root / "calibration.json"
    calibration = _load_json_object(calibration_path)
    cases = calibration.get("cases")
    runtime = calibration.get("runtime")
    if (
        calibration.get("kind") != "pallas_lowering_calibration"
        or not isinstance(calibration.get("capture_tool_sha256"), str)
        or len(calibration["capture_tool_sha256"]) != 64
        or not isinstance(cases, dict)
        or set(cases) != set(CONTROL_CASES)
        or not isinstance(runtime, dict)
    ):
        raise LoweringEvidenceError("CALIBRATION_SCHEMA_INVALID")
    if expected_runtime is not None:
        for name, expected in expected_runtime.items():
            if runtime.get(name) != expected:
                raise LoweringEvidenceError(
                    "CALIBRATION_RUNTIME_MISMATCH: "
                    f"{name}: expected={expected!r} observed={runtime.get(name)!r}"
                )
    if runtime.get("backend") != "tpu":
        raise LoweringEvidenceError(
            f"CALIBRATION_BACKEND_INVALID: {runtime.get('backend')!r}"
        )
    for label, case in cases.items():
        if (
            not isinstance(case, dict)
            or case.get("label") != label
            or case.get("correctness_verified") is not True
            or case.get("runtime") != runtime
        ):
            raise LoweringEvidenceError(f"CALIBRATION_CASE_INVALID: {label}")
        _validate_case_artifacts(calibration_root / label, case)
        repetitions = case.get("repetitions")
        trace = case.get("trace")
        if (
            not isinstance(repetitions, int)
            or isinstance(repetitions, bool)
            or repetitions < 1
            or not isinstance(trace, dict)
            or _event_count(trace, "tpu::System::Execute=>Done") < repetitions
        ):
            raise LoweringEvidenceError(f"CALIBRATION_TRACE_INVALID: {label}")
    positive = cases["normal_pallas"]
    negatives = (
        cases["interpreted_pallas"],
        cases["plain_jax"],
        cases["dead_pallas"],
    )
    for section in ("stablehlo_markers", "executable_hlo_markers"):
        if _marker_count(positive, section, "tpu_custom_call") < 1:
            raise LoweringEvidenceError(
                f"CALIBRATION_POSITIVE_MARKER_MISSING: {section}"
            )
        if any(
            _marker_count(case, section, "tpu_custom_call") != 0
            for case in negatives
        ):
            raise LoweringEvidenceError(
                f"CALIBRATION_NEGATIVE_MARKER_PRESENT: {section}"
            )
    return calibration


def validate_candidate_evidence(
    *,
    calibration_root: Path,
    candidate_root: Path,
    expected_kernel_sha256: str,
    expected_runtime: dict[str, str] | None = None,
) -> LoweringVerdict:
    calibration = validate_calibration(
        calibration_root,
        expected_runtime=expected_runtime,
    )
    candidate_path = candidate_root / "candidate.json"
    candidate = _load_json_object(candidate_path)
    evidence = candidate.get("evidence")
    runtime = calibration["runtime"]
    if (
        candidate.get("kind") != "pallas_candidate_lowering"
        or candidate.get("capture_tool_sha256")
        != calibration.get("capture_tool_sha256")
        or candidate.get("kernel_sha256") != expected_kernel_sha256
        or not isinstance(evidence, dict)
        or evidence.get("runtime") != runtime
    ):
        raise LoweringEvidenceError("CANDIDATE_EVIDENCE_LINEAGE_INVALID")
    _validate_case_artifacts(candidate_root / "candidate", evidence)
    reasons: list[str] = []
    for section in ("stablehlo_markers", "executable_hlo_markers"):
        if _marker_count(evidence, section, "tpu_custom_call") < 1:
            reasons.append(f"TPU_CUSTOM_CALL_MISSING:{section}")
    trace = evidence.get("trace")
    repetitions = evidence.get("repetitions")
    if (
        not isinstance(trace, dict)
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
        or _event_count(trace, "tpu::System::Execute=>Done") < repetitions
    ):
        reasons.append("CANDIDATE_TRACE_EXECUTION_MISSING")
    return LoweringVerdict(
        verified=not reasons,
        calibration_sha256=_sha256_file(calibration_root / "calibration.json"),
        candidate_sha256=_sha256_file(candidate_path),
        kernel_sha256=expected_kernel_sha256,
        runtime=runtime,
        reasons=tuple(reasons),
    )


def capture_lowering_case(
    *,
    label: str,
    function: Callable[..., Any],
    inputs: tuple[Any, ...],
    out_dir: Path,
    repetitions: int,
    expected_output: Any | None = None,
    rtol: float = 0,
    atol: float = 0,
) -> dict[str, Any]:
    if repetitions < 1:
        raise LoweringEvidenceError(f"PROFILE_REPETITIONS_INVALID: {repetitions}")
    case_dir = out_dir / label
    if case_dir.exists() and any(case_dir.iterdir()):
        raise LoweringEvidenceError(f"PROFILE_CASE_EXISTS: {case_dir}")
    trace_dir = case_dir / "trace"
    trace_dir.mkdir(parents=True)

    lowered = jax.jit(function).lower(*inputs)
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    executable_text = compiled.as_text()
    stablehlo_path = case_dir / "stablehlo.mlir"
    executable_path = case_dir / "executable.hlo.txt"
    stablehlo_path.write_text(stablehlo_text, encoding="utf-8")
    executable_path.write_text(executable_text, encoding="utf-8")

    output = compiled(*inputs)
    jax.block_until_ready(output)
    if expected_output is not None:
        try:
            chex.assert_trees_all_close(
                output,
                expected_output,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as exc:
            raise LoweringEvidenceError(
                f"CONTROL_CORRECTNESS_FAILED: {label}: {exc}"
            ) from exc
    with jax.profiler.trace(
        str(trace_dir),
        create_perfetto_link=False,
        create_perfetto_trace=True,
    ):
        for step in range(repetitions):
            with jax.profiler.StepTraceAnnotation(label, step_num=step):
                output = compiled(*inputs)
            jax.block_until_ready(output)

    events, trace_path = _trace_events(trace_dir)
    result = {
        "schema_version": 1,
        "label": label,
        "captured_at": _utc_now(),
        "runtime": _runtime_fingerprint(),
        "repetitions": repetitions,
        "correctness_verified": expected_output is not None,
        "compiler": {
            "stablehlo_sha256": _sha256_file(stablehlo_path),
            "executable_hlo_sha256": _sha256_file(executable_path),
            "stablehlo_markers": _compiler_markers(stablehlo_text),
            "executable_hlo_markers": _compiler_markers(executable_text),
            "cost_analysis": _jsonable(compiled.cost_analysis()),
            "memory_analysis": _jsonable(compiled.memory_analysis()),
        },
        "trace": {
            "perfetto_sha256": _sha256_file(trace_path),
            "perfetto_relative_path": str(trace_path.relative_to(case_dir)),
            **_summarise_events(events),
        },
    }
    _write_json(case_dir / "evidence.json", result)
    return result


def calibrate_lowering(*, out_dir: Path, repetitions: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = _control_input()
    expected = _plain_jax_control(*inputs)
    jax.block_until_ready(expected)
    cases = {
        label: capture_lowering_case(
            label=label,
            function=function,
            inputs=inputs,
            out_dir=out_dir,
            repetitions=repetitions,
            expected_output=expected,
        )
        for label, function in CONTROL_CASES.items()
    }
    summary = {
        "schema_version": 1,
        "kind": "pallas_lowering_calibration",
        "captured_at": _utc_now(),
        "capture_tool_sha256": _capture_tool_sha256(),
        "runtime": _runtime_fingerprint(),
        "cases": cases,
    }
    _write_json(out_dir / "calibration.json", summary)
    return summary


def capture_candidate(
    *,
    jaxbench_root: Path,
    workload: str,
    kernel: Path,
    out_dir: Path,
    repetitions: int,
) -> dict[str, Any]:
    baseline_path = (
        jaxbench_root / "JAXBench" / "benchmark" / workload / "baseline.py"
    )
    baseline = _load_module(baseline_path, f"opjax_probe_{workload}_baseline")
    candidate = _load_module(kernel, f"opjax_probe_{workload}_candidate")
    function = getattr(candidate, "workload", None)
    if not callable(function):
        raise LoweringEvidenceError("WORKLOAD_MISSING")
    evidence = capture_lowering_case(
        label="candidate",
        function=function,
        inputs=_create_inputs(baseline),
        out_dir=out_dir,
        repetitions=repetitions,
    )
    summary = {
        "schema_version": 1,
        "kind": "pallas_candidate_lowering",
        "captured_at": _utc_now(),
        "capture_tool_sha256": _capture_tool_sha256(),
        "workload": workload,
        "kernel_sha256": _sha256_file(kernel),
        "evidence": evidence,
    }
    _write_json(out_dir / "candidate.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m opjax.pallas.lowering")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--out-dir", type=Path, required=True)
    calibrate.add_argument("--repetitions", type=int, default=3)
    candidate = commands.add_parser("capture-candidate")
    candidate.add_argument("--jaxbench-root", type=Path, required=True)
    candidate.add_argument("--workload", required=True)
    candidate.add_argument("--kernel", type=Path, required=True)
    candidate.add_argument("--out-dir", type=Path, required=True)
    candidate.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            result = calibrate_lowering(
                out_dir=args.out_dir,
                repetitions=args.repetitions,
            )
        else:
            result = capture_candidate(
                jaxbench_root=args.jaxbench_root,
                workload=args.workload,
                kernel=args.kernel,
                out_dir=args.out_dir,
                repetitions=args.repetitions,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (LoweringEvidenceError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
