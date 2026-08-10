"""Frozen Phase 3 base-model capability experiment contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.model_registry import (
    INKLING_HF_REVISION,
    INKLING_MODEL_ID,
    LAGUNA_HF_REVISION,
    LAGUNA_MODEL_ID,
)
EXPECTED_RELEASE_SHA256 = (
    "4015d3db9e395d9f5d564f3eda2b971483c71c3e77e5ec4bf273fa17a4dcca0b"
)
EXPECTED_CLOSEOUT_SHA256 = (
    "965e675e80e2af11f45c1c93afcac78365db31b766a1810e5acb61238eac96f9"
)
EXPECTED_EXCLUDED_TASKS = {
    "11p_Megablox_GMM": "lower_compile",
    "16p_Mamba2_SSD": "lower_compile",
    "2p_GQA_Attention": "lower_compile",
}
SEEDS = (0, 1, 2)
SNAPSHOT_TURNS = (3, 6)


@dataclass(frozen=True)
class Phase3Contract:
    release_root: Path
    closeout_root: Path
    release_sha256: str
    scoreability_matrix_sha256: str
    scoreable_tasks: tuple[str, ...]
    task_hashes: dict[str, str]
    excluded_tasks: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G42HarnessError(f"PHASE3_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE3_JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_embedded_hash(value: dict[str, Any], *, field: str, code: str) -> str:
    payload = dict(value)
    expected = payload.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise G42HarnessError(code)
    return expected


def load_phase3_contract(
    *, release_root: Path, closeout_root: Path
) -> Phase3Contract:
    release_root = release_root.resolve()
    closeout_root = closeout_root.resolve()
    release = _read_json(release_root / "manifest.json")
    release_sha256 = _validate_embedded_hash(
        release,
        field="release_sha256",
        code="PHASE3_BENCHMARK_RELEASE_HASH_INVALID",
    )
    if (
        release_sha256 != EXPECTED_RELEASE_SHA256
        or release.get("benchmark_id") != "opjax-jaxbench-full-v1"
        or release.get("shape_policy") != "original_unmodified"
        or release.get("task_count") != 50
    ):
        raise G42HarnessError("PHASE3_BENCHMARK_RELEASE_INVALID")
    closeout = _read_json(closeout_root / "manifest.json")
    if (
        file_sha256(closeout_root / "manifest.json") != EXPECTED_CLOSEOUT_SHA256
        or closeout.get("status") != "completed"
        or closeout.get("release_sha256") != release_sha256
        or closeout.get("scoreable_task_count") != 47
        or closeout.get("scoreability", {}).get("unscoreable")
        != EXPECTED_EXCLUDED_TASKS
    ):
        raise G42HarnessError("PHASE3_CLOSEOUT_INVALID")
    scoreability_path = (
        closeout_root.parent / "jaxbench-full-v1-scoreability" / "matrix.json"
    )
    scoreability = _read_json(scoreability_path)
    if (
        scoreability.get("release_sha256") != release_sha256
        or scoreability.get("scoreable_count") != 47
        or scoreability.get("unscoreable_count") != 3
        or file_sha256(scoreability_path)
        != closeout.get("scoreability", {}).get("matrix_sha256")
    ):
        raise G42HarnessError("PHASE3_SCOREABILITY_MATRIX_INVALID")
    release_tasks = {task["task_id"]: task for task in release.get("tasks", [])}
    scoreable_tasks = tuple(
        sorted(
            result["task_id"]
            for result in scoreability.get("results", [])
            if result.get("status") == "scoreable"
        )
    )
    unscoreable = {
        result["task_id"]: result.get("stage")
        for result in scoreability.get("results", [])
        if result.get("status") == "unscoreable"
    }
    if (
        len(release_tasks) != 50
        or len(scoreable_tasks) != 47
        or unscoreable != EXPECTED_EXCLUDED_TASKS
        or set(scoreable_tasks) | set(unscoreable) != set(release_tasks)
    ):
        raise G42HarnessError("PHASE3_SCOREABLE_TASK_SET_INVALID")
    return Phase3Contract(
        release_root=release_root,
        closeout_root=closeout_root,
        release_sha256=release_sha256,
        scoreability_matrix_sha256=file_sha256(scoreability_path),
        scoreable_tasks=scoreable_tasks,
        task_hashes={task_id: release_tasks[task_id]["task_sha256"] for task_id in release_tasks},
        excluded_tasks=unscoreable,
    )


def build_experiment(*, contract: Phase3Contract) -> dict[str, Any]:
    models = [
        {
            "model_id": INKLING_MODEL_ID,
            "model_revision": INKLING_HF_REVISION,
            "provider": "tinker",
            "weight_identity": "provider_managed_base_bound_to_public_revision",
            "phase4_logit_parity_required": True,
        },
        {
            "model_id": LAGUNA_MODEL_ID,
            "model_revision": LAGUNA_HF_REVISION,
            "provider": "sglang",
            "weight_identity": "exact_hugging_face_revision",
            "phase4_logit_parity_required": False,
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
        "schema_version": 1,
        "kind": "opjax_phase3_base_capability_experiment",
        "phase": 3,
        "status": "frozen",
        "benchmark_release_sha256": contract.release_sha256,
        "scoreability_matrix_sha256": contract.scoreability_matrix_sha256,
        "scoreable_task_ids": list(contract.scoreable_tasks),
        "excluded_tasks": contract.excluded_tasks,
        "models": models,
        "sampling": {
            "seeds": list(SEEDS),
            "turn_limit": 6,
            "snapshot_turns": list(SNAPSHOT_TURNS),
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 8192,
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


def validate_sample_matrix(
    *, path: Path, contract: Phase3Contract
) -> dict[str, Any]:
    observed = _read_json(path)
    _validate_embedded_hash(
        observed,
        field="experiment_sha256",
        code="PHASE3_EXPERIMENT_HASH_INVALID",
    )
    expected = build_experiment(contract=contract)
    observed_cells = {
        (cell.get("model_id"), cell.get("task_id"), cell.get("seed"))
        for cell in observed.get("cells", [])
    }
    expected_cells = {
        (cell["model_id"], cell["task_id"], cell["seed"])
        for cell in expected["cells"]
    }
    if observed_cells != expected_cells:
        raise G42HarnessError("PHASE3_CELL_SET_INVALID")
    if observed != expected:
        raise G42HarnessError("PHASE3_EXPERIMENT_DRIFT")
    return {
        "experiment_sha256": expected["experiment_sha256"],
        "trajectories": len(expected_cells),
        "snapshots": expected["counts"]["snapshots"],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-baseline")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-v1"),
    )
    parser.add_argument(
        "--closeout-root",
        type=Path,
        default=Path("data/pallas/runs/jaxbench-full-v1-closeout"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        contract = load_phase3_contract(
            release_root=args.release_root,
            closeout_root=args.closeout_root,
        )
        if args.command == "freeze":
            if args.out.exists():
                raise G42HarnessError(f"PHASE3_OUTPUT_EXISTS:{args.out}")
            result = build_experiment(contract=contract)
            _write_json(args.out, result)
        else:
            result = validate_sample_matrix(path=args.path, contract=contract)
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE3_BASELINE_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
