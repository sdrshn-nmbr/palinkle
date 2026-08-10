import hashlib
import json
from pathlib import Path

from opjax.pallas.g42_harness import canonical_sha256
from opjax.pallas.phase3_grading import EMPTY_PATCH_SHA256
from opjax.pallas.phase3_results import summarize_model


def test_summarize_model_counts_paired_horizon_transitions(tmp_path: Path) -> None:
    experiment = {
        "experiment_sha256": "experiment",
        "models": [
            {
                "model_id": "model",
                "model_revision": "revision",
                "provider": "tinker",
            }
        ],
        "cells": [
            {
                "model_id": "model",
                "provider": "tinker",
                "task_id": f"task-{index}",
                "seed": 0,
            }
            for index in range(141)
        ],
    }
    records = []
    for index in range(141):
        task_id = f"task-{index}"
        seed = 0
        rewards = (0, 1) if index < 71 else (1, 1)
        for turn, reward in zip((3, 6), rewards, strict=True):
            records.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "turn": turn,
                    "reward": reward,
                    "failure_stage": None if reward else "correctness",
                    "patch_sha256": "nonempty",
                }
            )
    grading = {
        "kind": "opjax_phase3_base_capability_result",
        "experiment_sha256": "experiment",
        "provider": "tinker",
        "counts": {"trajectories": 141, "snapshots": 282},
        "horizons": {},
        "records": records,
    }
    grading["result_sha256"] = canonical_sha256(grading)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(grading), encoding="utf-8")

    result = summarize_model(experiment=experiment, grading_path=path)

    assert result["turn_3_to_6"] == {
        "fail_to_pass": 71,
        "pass_to_pass": 70,
        "fail_to_fail": 0,
        "pass_to_fail": 0,
    }
    assert result["nonempty_snapshot_count"] == 282


def test_empty_patch_identity_is_stable() -> None:
    assert EMPTY_PATCH_SHA256 == hashlib.sha256(b"").hexdigest()
