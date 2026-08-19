from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


EXPECTED_STEPS = (0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 120)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_preflight(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "preflight_sha256"}
    )
    if payload.get("preflight_sha256") != expected:
        raise ValueError("LAGUNA_CHECKPOINT_PREFLIGHT_HASH_MISMATCH")
    if (
        payload.get("kind") != "opjax_laguna_serving_native_training_preflight"
        or payload.get("seed") != 42
        or payload.get("caches", {}).get("calibration", {}).get("sample_count") != 18
    ):
        raise ValueError("LAGUNA_CHECKPOINT_PREFLIGHT_INVALID")
    return payload


def _validated_lineage(
    path: Path, *, arm: str, preflight: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    if payload.get("sha256") != expected:
        raise ValueError("LAGUNA_CHECKPOINT_LINEAGE_HASH_MISMATCH")
    if (
        payload.get("kind") != "opjax_laguna_checkpoint_lineage"
        or payload.get("preflight_sha256") != preflight["preflight_sha256"]
        or payload.get("steps") != list(EXPECTED_STEPS)
        or set(payload.get("checkpoints", {})) != {"dflash", "dspark"}
        or set(payload["checkpoints"].get(arm, {}))
        != {str(step) for step in EXPECTED_STEPS}
    ):
        raise ValueError("LAGUNA_CHECKPOINT_LINEAGE_INVALID")
    return payload


def _trace_kernel_count(path: Path) -> int:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"LAGUNA_CHECKPOINT_TRACE_INVALID:{path}")
    count = sum(
        1
        for event in events
        if isinstance(event, dict)
        and event.get("name") in {"cudaLaunchKernel", "cuLaunchKernelEx"}
    )
    if count == 0:
        raise ValueError(f"LAGUNA_CHECKPOINT_TRACE_NO_CUDA:{path}")
    return count


def _candidate(
    path: Path,
    *,
    arm: str,
    preflight: dict[str, Any],
    expected_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"LAGUNA_CHECKPOINT_ARTIFACT_MISSING:{path}")
    payload = json.loads(path.read_text())
    step = int(path.parent.name.removeprefix("step_"))
    if (
        payload.get("arm") != arm
        or payload.get("split") != "calibration"
        or payload.get("variant") != "raw"
        or payload.get("step") != step
        or payload.get("seed") != 42
        or payload.get("batches") != 18
        or payload.get("valid_blocks") != 1152.0
        or payload.get("cache")
        != "/mnt/training/experiments/serving-native-v2/cache/calibration"
    ):
        raise ValueError(f"LAGUNA_CHECKPOINT_RESULT_INVALID:{path}")
    runtime_path = path.parent / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    if (
        payload.get("runtime") != runtime
        or runtime.get("deepspec_revision") != preflight.get("deepspec_revision")
        or runtime.get("execution_files") != preflight.get("execution_files")
        or runtime.get("gpu") != "NVIDIA H200, 580.95.05"
    ):
        raise ValueError(f"LAGUNA_CHECKPOINT_RUNTIME_INVALID:{path}")
    command = runtime.get("command")
    if not isinstance(command, list) or command[-2:] != ["--seed", "42"]:
        raise ValueError(f"LAGUNA_CHECKPOINT_COMMAND_INVALID:{path}")
    checkpoint = payload.get("checkpoint")
    checkpoint_files = (checkpoint or {}).get("files")
    if (
        not isinstance(checkpoint, dict)
        or not checkpoint.get("sha256")
        or not isinstance(checkpoint_files, dict)
        or set(checkpoint_files) != {"config.json", "model.safetensors"}
        or not all(checkpoint_files.values())
    ):
        raise ValueError(f"LAGUNA_CHECKPOINT_IDENTITY_INVALID:{path}")
    if checkpoint.get("sha256") != canonical_sha256(checkpoint_files):
        raise ValueError(f"LAGUNA_CHECKPOINT_IDENTITY_HASH_INVALID:{path}")
    if checkpoint != expected_checkpoint:
        raise ValueError(f"LAGUNA_CHECKPOINT_LINEAGE_MISMATCH:{path}")
    expected_checkpoint_suffix = (
        f"/initialized/{arm}" if step == 0 else f"/checkpoints/{arm}/step_{step}"
    )
    if not str(checkpoint.get("path", "")).endswith(expected_checkpoint_suffix):
        raise ValueError(f"LAGUNA_CHECKPOINT_LINEAGE_INVALID:{path}")

    metrics = {
        "probabilistic_tau": float(payload["probabilistic_tau"]),
        "greedy_tau": float(payload["greedy_tau"]),
        "cross_entropy": float(payload["loss"]["cross_entropy"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"LAGUNA_CHECKPOINT_METRIC_NONFINITE:{path}")

    required = {
        name: path.parent / name
        for name in (
            "evaluation.json",
            "gpu.csv",
            "run.log",
            "runtime.json",
            "torch-trace.json",
        )
    }
    if any(not file.is_file() or file.stat().st_size == 0 for file in required.values()):
        raise ValueError(f"LAGUNA_CHECKPOINT_ARTIFACT_MISSING:{path}")
    if payload.get("trace") != "torch-trace.json":
        raise ValueError(f"LAGUNA_CHECKPOINT_TRACE_BINDING_INVALID:{path}")
    evaluation = json.loads(required["evaluation.json"].read_text())
    expected_evaluation = {
        key: value
        for key, value in payload.items()
        if key not in {"arm", "step", "split", "variant", "runtime"}
    }
    expected_evaluation["checkpoint"] = command[command.index("--checkpoint") + 1]
    if evaluation != expected_evaluation:
        raise ValueError(f"LAGUNA_CHECKPOINT_EVALUATION_MISMATCH:{path}")
    gpu_lines = required["gpu.csv"].read_text().splitlines()
    if len(gpu_lines) < 2 or "utilization.gpu [%]" not in gpu_lines[0]:
        raise ValueError(f"LAGUNA_CHECKPOINT_GPU_EVIDENCE_INVALID:{path}")
    artifacts = {
        name: {
            "sha256": _sha256(file),
            "bytes": file.stat().st_size,
        }
        for name, file in sorted(required.items())
    }
    artifacts["torch-trace.json"]["cuda_kernel_events"] = _trace_kernel_count(
        required["torch-trace.json"]
    )
    return {
        "step": step,
        **metrics,
        "checkpoint": checkpoint,
        "result_sha256": _sha256(path),
        "artifacts": artifacts,
    }


def select_checkpoint(
    root: Path, arm: str, *, preflight_path: Path, lineage_path: Path
) -> dict[str, Any]:
    preflight = _validated_preflight(preflight_path)
    lineage = _validated_lineage(lineage_path, arm=arm, preflight=preflight)
    arm_root = root / arm / "raw"
    step_roots = sorted(arm_root.glob("step_*"))
    observed_steps = sorted(int(path.name.removeprefix("step_")) for path in step_roots)
    if observed_steps != list(EXPECTED_STEPS):
        raise ValueError(f"LAGUNA_CHECKPOINT_STEP_SET_INVALID:{arm}:{observed_steps}")
    rows = [
        _candidate(
            path / "result.json",
            arm=arm,
            preflight=preflight,
            expected_checkpoint=lineage["checkpoints"][arm][
                path.name.removeprefix("step_")
            ],
        )
        for path in step_roots
    ]
    selected = max(
        rows,
        key=lambda row: (
            row["probabilistic_tau"],
            -row["cross_entropy"],
            -row["step"],
        ),
    )
    result = {
        "schema_version": 1,
        "arm": arm,
        "policy": (
            "maximize calibration probabilistic_tau; tie break lower cross_entropy; "
            "then earlier step"
        ),
        "preflight_sha256": preflight["preflight_sha256"],
        "preflight_file_sha256": _sha256(preflight_path),
        "lineage_sha256": lineage["sha256"],
        "lineage_file_sha256": _sha256(lineage_path),
        "expected_steps": list(EXPECTED_STEPS),
        "selected_step": selected["step"],
        "selected": selected,
        "candidates": rows,
    }
    result["sha256"] = canonical_sha256(result)
    return result
