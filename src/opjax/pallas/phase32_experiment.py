"""Frozen standard-protocol base-model experiment for Gate 3.2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from opjax.pallas.phase31_experiment import (
    PROVIDER_RUNTIME_FILES,
    Phase31Contract,
    load_contract,
)
from opjax.remote.inkling_small_sglang import SGLANG_REVISION as INKLING_RUNTIME
from opjax.remote.laguna_sglang import SGLANG_REVISION as LAGUNA_RUNTIME

SGLANG_AUDIT_REVISION = "c80a38edcd2c7077c909a5ed925c9241e754c067"
PHASE32_RUNTIME_FILES = (
    *PROVIDER_RUNTIME_FILES,
    "src/opjax/pallas/phase32_experiment.py",
    "src/opjax/remote/config.py",
    "src/opjax/remote/inkling_small_baseline.py",
    "src/opjax/remote/inkling_small_sglang.py",
)


def _runtime_hashes() -> dict[str, str]:
    repository = Path(__file__).parents[3]
    return {path: file_sha256(repository / path) for path in PHASE32_RUNTIME_FILES}


def _sglang_audit_revision() -> str:
    repository = Path(__file__).parents[3] / "references/sglang"
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_conformance(path: Path, *, model_id: str) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    digest = payload.pop("conformance_sha256", None)
    if (
        value.get("kind") != "opjax_phase31_provider_protocol_conformance"
        or value.get("passed") is not True
        or value.get("model_calls") != 2
        or value.get("repair_turns") != 0
        or value.get("provider") != "sglang_openai"
        or value.get("model", {}).get("model_id") != model_id
        or canonical_sha256(payload) != digest
    ):
        raise G42HarnessError(f"PHASE32_CONFORMANCE_INVALID:{model_id}")
    return value, file_sha256(path)


def build_experiment(
    *,
    contract: Phase31Contract,
    release: dict[str, Any],
    inkling_conformance: Path,
    laguna_conformance: Path,
) -> dict[str, Any]:
    if _sglang_audit_revision() != SGLANG_AUDIT_REVISION:
        raise G42HarnessError("PHASE32_SGLANG_AUDIT_REVISION_MISMATCH")
    _, inkling_conformance_sha = _load_conformance(
        inkling_conformance,
        model_id=INKLING_MODEL_ID,
    )
    _, laguna_conformance_sha = _load_conformance(
        laguna_conformance,
        model_id=LAGUNA_MODEL_ID,
    )
    models = [
        {
            "model_id": INKLING_MODEL_ID,
            "model_revision": INKLING_HF_REVISION,
            "provider": "sglang_openai_inkling",
            "runtime_revision": INKLING_RUNTIME,
            "weight_identity": "exact_hugging_face_revision",
            "protocol_conformance_sha256": inkling_conformance_sha,
        },
        {
            "model_id": LAGUNA_MODEL_ID,
            "model_revision": LAGUNA_HF_REVISION,
            "provider": "sglang_openai_laguna",
            "runtime_revision": LAGUNA_RUNTIME,
            "weight_identity": "exact_hugging_face_revision",
            "protocol_conformance_sha256": laguna_conformance_sha,
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
        "kind": "opjax_phase32_base_capability_experiment",
        "phase": "3.2",
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
            "driver": "mini-swe-agent==2.4.6",
            "transport": "openai_chat_completions",
            "tool": "bash",
            "agent_image": release["agent_environment"]["image"],
            "agent_image_id": release["agent_environment"]["image_id"],
            "bound_source_sha256": release["bound_source_sha256"],
            "provider_runtime_sha256": _runtime_hashes(),
            "sglang_source_revision": SGLANG_AUDIT_REVISION,
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


def validate_experiment(
    *, value: dict[str, Any], contract: Phase31Contract
) -> dict[str, Any]:
    payload = dict(value)
    digest = payload.pop("experiment_sha256", None)
    expected_cells = {
        (model["model_id"], task_id, seed)
        for model in value.get("models", ())
        for task_id in contract.scoreable_tasks
        for seed in SEEDS
    }
    observed_cells = {
        (cell.get("model_id"), cell.get("task_id"), cell.get("seed"))
        for cell in value.get("cells", ())
    }
    if (
        value.get("kind") != "opjax_phase32_base_capability_experiment"
        or value.get("status") != "frozen"
        or canonical_sha256(payload) != digest
        or value.get("benchmark_release_sha256") != contract.release_sha256
        or tuple(value.get("valid_task_ids", ())) != contract.scoreable_tasks
        or observed_cells != expected_cells
        or value.get("harness", {}).get("provider_runtime_sha256")
        != _runtime_hashes()
        or value.get("harness", {}).get("sglang_source_revision")
        != _sglang_audit_revision()
    ):
        raise G42HarnessError("PHASE32_EXPERIMENT_INVALID")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase32-experiment")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-phase31"),
    )
    parser.add_argument(
        "--validity",
        type=Path,
        default=Path("data/pallas/runs/phase31-oracle-validity/manifest.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "data/pallas/runs/phase31-positive-control-calibration/manifest.json"
        ),
    )
    parser.add_argument(
        "--inkling-conformance",
        type=Path,
        default=Path(
            "data/pallas/runs/phase32-provider-conformance/inkling-small.json"
        ),
    )
    parser.add_argument(
        "--laguna-conformance",
        type=Path,
        default=Path("data/pallas/runs/phase32-provider-conformance/laguna.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/experiment.json"),
    )
    args = parser.parse_args(argv)
    try:
        contract = load_contract(
            release_root=args.release_root,
            validity_path=args.validity,
            calibration_path=args.calibration,
        )
        release = json.loads(
            (args.release_root / "manifest.json").read_text(encoding="utf-8")
        )
        experiment = build_experiment(
            contract=contract,
            release=release,
            inkling_conformance=args.inkling_conformance,
            laguna_conformance=args.laguna_conformance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(experiment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE32_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(experiment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
