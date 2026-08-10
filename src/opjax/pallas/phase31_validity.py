"""Empirical oracle-validity audit for the Phase 3.1 task denominator."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.jaxbench_verifier import _create_inputs, _load_module, _ready, _validate_input_schema
from opjax.pallas.phase31_oracle import compare_output, derive_input_case


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "libtpu": (
            importlib.metadata.version("libtpu")
            if importlib.util.find_spec("libtpu") is not None
            else None
        ),
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "device_kinds": sorted({device.device_kind for device in jax.devices()}),
    }


def audit_task(*, release_root: Path, task_id: str, out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        raise G42HarnessError(f"PHASE31_VALIDITY_OUTPUT_EXISTS:{out_path}")
    release = json.loads((release_root / "manifest.json").read_text())
    record = next(
        (item for item in release["tasks"] if item["task_id"] == task_id), None
    )
    if record is None:
        raise G42HarnessError(f"PHASE31_VALIDITY_TASK_UNKNOWN:{task_id}")
    task_root = release_root / record["path"]
    task = json.loads((task_root / "tests/task.json").read_text())
    baseline_path = task_root / "tests/jaxbench/baseline.py"
    if file_sha256(baseline_path) != task["baseline_sha256"]:
        raise G42HarnessError("PHASE31_VALIDITY_BASELINE_HASH_INVALID")
    baseline = _load_module(baseline_path, f"{task_id}.phase31_validity")
    inputs = _create_inputs(baseline)
    _validate_input_schema(inputs, task)
    workload = getattr(baseline, "workload")
    function = workload if getattr(baseline, "_skip_jit", False) else jax.jit(workload)
    cases = []
    for seed, name in enumerate(task["oracle_contract"]["input_cases"]):
        case_inputs = derive_input_case(inputs, contract=task["oracle_contract"], seed=seed)
        expected = function(*case_inputs)
        _ready(expected)
        zero = jnp.zeros_like(expected)
        zero_result = compare_output(expected, zero, contract=task["oracle_contract"])
        self_result = compare_output(expected, expected, contract=task["oracle_contract"])
        cases.append(
            {
                "case": name,
                "signal_max_abs": self_result["signal_max_abs"],
                "zero_correct": zero_result["correct"],
                "zero_normalized_max_error": zero_result["normalized_max_error"],
                "self_correct": self_result["correct"],
            }
        )
    valid = all(case["self_correct"] for case in cases) and any(
        not case["zero_correct"] for case in cases
    )
    result = {
        "schema_version": 1,
        "kind": "opjax_phase31_task_oracle_validity",
        "benchmark_release_sha256": release["release_sha256"],
        "task_id": task_id,
        "task_sha256": record["task_sha256"],
        "oracle_contract_sha256": canonical_sha256(task["oracle_contract"]),
        "valid": valid,
        "performance_eligible": cases[0]["signal_max_abs"] > 0,
        "cases": cases,
        "runtime": runtime_fingerprint(),
    }
    result["result_sha256"] = canonical_sha256(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def load_runtime_exclusions(
    *, path: Path, release_sha256: str, scoreable: set[str]
) -> dict[str, dict[str, Any]]:
    exclusions = json.loads(path.read_text())
    payload = dict(exclusions)
    observed_sha = payload.pop("exclusions_sha256", None)
    records = exclusions.get("records", ())
    excluded = {record.get("task_id"): record for record in records}
    if (
        exclusions.get("kind") != "opjax_phase31_oracle_runtime_exclusions"
        or exclusions.get("benchmark_release_sha256") != release_sha256
        or canonical_sha256(payload) != observed_sha
        or None in excluded
        or len(excluded) != len(records)
        or set(excluded) - scoreable
        or any(
            record.get("stage") != "baseline_execution"
            or not record.get("reason")
            or not record.get("evidence")
            for record in excluded.values()
        )
    ):
        raise G42HarnessError("PHASE31_VALIDITY_EXCLUSIONS_INVALID")
    return excluded


def assemble(
    *,
    release_root: Path,
    scoreability_path: Path,
    evidence_root: Path,
    exclusions_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    release = json.loads((release_root / "manifest.json").read_text())
    scoreability = json.loads(scoreability_path.read_text())
    scoreable = sorted(
        result["task_id"]
        for result in scoreability["results"]
        if result.get("status") == "scoreable"
    )
    excluded_runtime = load_runtime_exclusions(
        path=exclusions_path,
        release_sha256=release["release_sha256"],
        scoreable=set(scoreable),
    )
    records = []
    for task_id in sorted(set(scoreable) - set(excluded_runtime)):
        path = evidence_root / f"{task_id}.json"
        record = json.loads(path.read_text())
        payload = dict(record)
        observed_hash = payload.pop("result_sha256", None)
        if (
            record.get("kind") != "opjax_phase31_task_oracle_validity"
            or record.get("benchmark_release_sha256") != release["release_sha256"]
            or canonical_sha256(payload) != observed_hash
        ):
            raise G42HarnessError(f"PHASE31_VALIDITY_RECORD_INVALID:{task_id}")
        records.append({**record, "evidence_sha256": file_sha256(path)})
    valid = sorted(record["task_id"] for record in records if record["valid"])
    invalid = {
        record["task_id"]: "zero_output_not_discriminated"
        for record in records
        if not record["valid"]
    }
    excluded_platform = {
        result["task_id"]: result.get("stage")
        for result in scoreability["results"]
        if result.get("status") != "scoreable"
    }
    manifest = {
        "schema_version": 1,
        "kind": "opjax_phase31_oracle_validity",
        "benchmark_release_sha256": release["release_sha256"],
        "scoreability_sha256": file_sha256(scoreability_path),
        "runtime_exclusions_sha256": file_sha256(exclusions_path),
        "valid_task_ids": valid,
        "invalid_oracle_tasks": invalid,
        "candidate_verifier_unavailable": excluded_platform,
        "baseline_runtime_unavailable": excluded_runtime,
        "counts": {
            "release_tasks": 50,
            "audited": len(records),
            "valid": len(valid),
            "invalid_oracle": len(invalid),
            "candidate_verifier_unavailable": len(excluded_platform),
            "baseline_runtime_unavailable": len(excluded_runtime),
        },
        "records": records,
    }
    manifest["validity_sha256"] = canonical_sha256(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-validity")
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("task")
    task.add_argument("--release-root", type=Path, required=True)
    task.add_argument("--task-id", required=True)
    task.add_argument("--out-path", type=Path, required=True)
    matrix = sub.add_parser("assemble")
    matrix.add_argument("--release-root", type=Path, required=True)
    matrix.add_argument("--scoreability-path", type=Path, required=True)
    matrix.add_argument("--evidence-root", type=Path, required=True)
    matrix.add_argument("--exclusions-path", type=Path, required=True)
    matrix.add_argument("--out-path", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = audit_task(**args) if command == "task" else assemble(**args)
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE31_VALIDITY_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
