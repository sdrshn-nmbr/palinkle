from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.laguna_checkpoint_selection import select_checkpoint
from opjax.pallas.laguna_dspark_conformance import canonical_sha256


STEPS = (0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 120)


def _preflight(root: Path) -> Path:
    payload = {
        "kind": "opjax_laguna_serving_native_training_preflight",
        "seed": 42,
        "caches": {"calibration": {"sample_count": 18}},
        "deepspec_revision": "deep-spec",
        "execution_files": {"driver": "driver-sha"},
    }
    payload["preflight_sha256"] = canonical_sha256(payload)
    path = root / "preflight.json"
    path.write_text(json.dumps(payload))
    return path


def _write(root: Path, step: int, tau: float, cross_entropy: float) -> dict[str, object]:
    path = root / "dflash" / "raw" / f"step_{step}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = "initialized" if step == 0 else "checkpoints"
    checkpoint_suffix = "dflash" if step == 0 else f"dflash/step_{step}"
    checkpoint_path = (
        f"/mnt/training/experiments/serving-native-v2/{checkpoint_root}/"
        f"{checkpoint_suffix}"
    )
    runtime = {
        "command": [
            "python",
            "eval",
            "--checkpoint",
            checkpoint_path,
            "--seed",
            "42",
        ],
        "deepspec_revision": "deep-spec",
        "execution_files": {"driver": "driver-sha"},
        "gpu": "NVIDIA H200, 580.95.05",
    }
    payload = {
        "arm": "dflash",
        "split": "calibration",
        "variant": "raw",
        "step": step,
        "seed": 42,
        "batches": 18,
        "valid_blocks": 1152.0,
        "cache": "/mnt/training/experiments/serving-native-v2/cache/calibration",
        "probabilistic_tau": tau,
        "greedy_tau": tau - 0.1,
        "loss": {"cross_entropy": cross_entropy},
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": canonical_sha256(
                {"config.json": "config", "model.safetensors": "model"}
            ),
            "files": {"config.json": "config", "model.safetensors": "model"},
        },
        "runtime": runtime,
        "trace": "torch-trace.json",
    }
    path.write_text(json.dumps(payload))
    evaluation = {
        key: value
        for key, value in payload.items()
        if key not in {"arm", "step", "split", "variant", "runtime"}
    }
    evaluation["checkpoint"] = checkpoint_path
    (path.parent / "evaluation.json").write_text(json.dumps(evaluation))
    (path.parent / "runtime.json").write_text(json.dumps(runtime))
    (path.parent / "run.log").write_text("completed\n")
    (path.parent / "gpu.csv").write_text(
        "timestamp, utilization.gpu [%]\nnow, 10 %\n"
    )
    (path.parent / "torch-trace.json").write_text(
        json.dumps({"traceEvents": [{"name": "cudaLaunchKernel"}]})
    )
    return payload["checkpoint"]


def _lineage(root: Path, checkpoints: dict[int, dict[str, object]]) -> Path:
    payload = {
        "kind": "opjax_laguna_checkpoint_lineage",
        "preflight_sha256": json.loads(_preflight(root).read_text())["preflight_sha256"],
        "steps": list(STEPS),
        "checkpoints": {
            "dflash": {str(step): checkpoints[step] for step in STEPS},
            "dspark": {str(step): checkpoints[step] for step in STEPS},
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    path = root / "lineage.json"
    path.write_text(json.dumps(payload))
    return path


def test_selects_tau_then_loss_then_earlier_step(tmp_path: Path) -> None:
    checkpoints = {step: _write(tmp_path, step, 1.0, 2.0) for step in STEPS}
    checkpoints[26] = _write(tmp_path, 26, 2.1, 1.2)
    checkpoints[39] = _write(tmp_path, 39, 2.1, 1.1)
    checkpoints[52] = _write(tmp_path, 52, 2.1, 1.1)
    preflight = _preflight(tmp_path)
    lineage = _lineage(tmp_path, checkpoints)
    result = select_checkpoint(
        tmp_path, "dflash", preflight_path=preflight, lineage_path=lineage
    )
    assert result["selected_step"] == 39
    assert result["candidates"][0]["artifacts"]["torch-trace.json"][
        "cuda_kernel_events"
    ] == 1


def test_rejects_missing_step_or_non_cuda_trace(tmp_path: Path) -> None:
    checkpoints = {step: _write(tmp_path, step, 1.0, 2.0) for step in STEPS}
    preflight = _preflight(tmp_path)
    lineage = _lineage(tmp_path, checkpoints)
    (tmp_path / "dflash/raw/step_13/result.json").unlink()
    with pytest.raises(ValueError, match="ARTIFACT_MISSING"):
        select_checkpoint(
            tmp_path, "dflash", preflight_path=preflight, lineage_path=lineage
        )
    _write(tmp_path, 13, 1.0, 2.0)
    (tmp_path / "dflash/raw/step_13/torch-trace.json").write_text(
        json.dumps({"traceEvents": [{"name": "metadata_cuda"}]})
    )
    with pytest.raises(ValueError, match="TRACE_NO_CUDA"):
        select_checkpoint(
            tmp_path, "dflash", preflight_path=preflight, lineage_path=lineage
        )
