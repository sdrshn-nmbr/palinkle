"""Frozen matched-base experiment contract for Phase 3.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.phase3_baseline import (
    INKLING_HF_REVISION,
    INKLING_MODEL_ID,
    LAGUNA_HF_REVISION,
    LAGUNA_MODEL_ID,
    SEEDS,
    SNAPSHOT_TURNS,
)


PROVIDER_RUNTIME_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/opjax/pallas/g42_agent.py",
    "src/opjax/pallas/jaxbench_agent.py",
    "src/opjax/pallas/phase31_experiment.py",
    "src/opjax/pallas/phase31_grading.py",
    "src/opjax/pallas/phase31_sampling.py",
    "src/opjax/pallas/phase3_grading.py",
    "src/opjax/pallas/phase3_results.py",
    "src/opjax/pallas/phase3_sampling.py",
    "src/opjax/pallas/sglang_agent.py",
    "src/opjax/remote/laguna_baseline.py",
    "src/opjax/remote/laguna_sglang.py",
)


def provider_runtime_hashes() -> dict[str, str]:
    repo_root = Path(__file__).parents[3]
    return {path: file_sha256(repo_root / path) for path in PROVIDER_RUNTIME_FILES}


@dataclass(frozen=True)
class Phase31Contract:
    release_root: Path
    release_sha256: str
    scoreable_tasks: tuple[str, ...]
    task_hashes: dict[str, str]
    validity_sha256: str
    calibration_sha256: str


def load_contract(
    *, release_root: Path, validity_path: Path, calibration_path: Path
) -> Phase31Contract:
    release = json.loads((release_root / "manifest.json").read_text())
    validity = json.loads(validity_path.read_text())
    validity_payload = dict(validity)
    validity_sha = validity_payload.pop("validity_sha256", None)
    release_tasks = {task["task_id"]: task for task in release["tasks"]}
    valid = tuple(validity.get("valid_task_ids", ()))
    calibration = json.loads(calibration_path.read_text())
    calibration_payload = dict(calibration)
    calibration_hash = calibration_payload.pop("calibration_sha256", None)
    if (
        release.get("kind") != "opjax_phase31_jaxbench_benchmark"
        or validity.get("kind") != "opjax_phase31_oracle_validity"
        or canonical_sha256(validity_payload) != validity_sha
        or validity.get("benchmark_release_sha256") != release.get("release_sha256")
        or set(valid) - set(release_tasks)
        or len(valid) != len(set(valid))
        or calibration.get("kind") != "opjax_phase31_positive_control_calibration"
        or canonical_sha256(calibration_payload) != calibration_hash
        or calibration.get("benchmark_release_sha256") != release.get("release_sha256")
        or calibration.get("accepted") is not True
    ):
        raise G42HarnessError("PHASE31_CONTRACT_INVALID")
    return Phase31Contract(
        release_root=release_root,
        release_sha256=release["release_sha256"],
        scoreable_tasks=valid,
        task_hashes={task_id: release_tasks[task_id]["task_sha256"] for task_id in valid},
        validity_sha256=file_sha256(validity_path),
        calibration_sha256=file_sha256(calibration_path),
    )


def build_experiment(*, contract: Phase31Contract, release: dict[str, Any]) -> dict[str, Any]:
    models = [
        {
            "model_id": INKLING_MODEL_ID,
            "model_revision": INKLING_HF_REVISION,
            "provider": "tinker",
            "weight_identity": "provider_managed_base_bound_to_public_revision",
        },
        {
            "model_id": LAGUNA_MODEL_ID,
            "model_revision": LAGUNA_HF_REVISION,
            "provider": "sglang",
            "weight_identity": "exact_hugging_face_revision",
        },
    ]
    cells = [
        {
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
            "provider": model["provider"],
            "task_id": task_id,
            "task_sha256": contract.task_hashes[task_id],
            "seed": seed,
        }
        for model in models
        for task_id in contract.scoreable_tasks
        for seed in SEEDS
    ]
    experiment = {
        "schema_version": 2,
        "kind": "opjax_phase31_base_capability_experiment",
        "phase": "3.1",
        "status": "frozen",
        "benchmark_release_sha256": contract.release_sha256,
        "oracle_validity_sha256": contract.validity_sha256,
        "positive_control_calibration_sha256": contract.calibration_sha256,
        "valid_task_ids": list(contract.scoreable_tasks),
        "models": models,
        "sampling": {
            "seeds": list(SEEDS),
            "turn_limit": 6,
            "snapshot_turns": list(SNAPSHOT_TURNS),
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 8192,
        },
        "harness": {
            "agent_image": release["agent_environment"]["image"],
            "agent_image_id": release["agent_environment"]["image_id"],
            "bound_source_sha256": release["bound_source_sha256"],
            "provider_runtime_sha256": provider_runtime_hashes(),
            "action_protocol": release["action_protocol"],
        },
        "counts": {
            "models": len(models),
            "tasks": len(contract.scoreable_tasks),
            "seeds": len(SEEDS),
            "trajectories": len(cells),
            "snapshots": len(cells) * len(SNAPSHOT_TURNS),
        },
        "cells": cells,
    }
    experiment["experiment_sha256"] = canonical_sha256(experiment)
    return experiment


def validate_experiment(*, value: dict[str, Any], contract: Phase31Contract) -> None:
    payload = dict(value)
    observed_hash = payload.pop("experiment_sha256", None)
    expected_cells = {
        (model, task, seed)
        for model in (INKLING_MODEL_ID, LAGUNA_MODEL_ID)
        for task in contract.scoreable_tasks
        for seed in SEEDS
    }
    observed_cells = {
        (cell.get("model_id"), cell.get("task_id"), cell.get("seed"))
        for cell in value.get("cells", ())
    }
    if (
        value.get("kind") != "opjax_phase31_base_capability_experiment"
        or canonical_sha256(payload) != observed_hash
        or value.get("benchmark_release_sha256") != contract.release_sha256
        or value.get("oracle_validity_sha256") != contract.validity_sha256
        or value.get("positive_control_calibration_sha256") != contract.calibration_sha256
        or value.get("harness", {}).get("provider_runtime_sha256")
        != provider_runtime_hashes()
        or observed_cells != expected_cells
        or value.get("counts", {}).get("trajectories") != len(expected_cells)
    ):
        raise G42HarnessError("PHASE31_EXPERIMENT_INVALID")
