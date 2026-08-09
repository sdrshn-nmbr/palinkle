"""Render canonical Harbor verifier artifacts from one Phase 1 result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MANDATORY_STAGES = (
    "artifact_contract",
    "pallas_api",
    "tpu_compile",
    "full_shape_correctness",
    "normal_lowering",
    "runtime_safety",
    "profile",
)


class Phase2ArtifactError(RuntimeError):
    """The verifier result cannot be rendered as canonical Harbor artifacts."""


def render_artifacts(root: Path) -> dict[str, Any]:
    result_path = root / "result.json"
    if not result_path.is_file():
        raise Phase2ArtifactError(f"RESULT_MISSING:{result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
    admitted = bool(
        isinstance(result.get("profile"), dict)
        and isinstance(result["profile"].get("admission"), dict)
        and result["profile"]["admission"].get("verified") is True
    )
    infrastructure = result.get("infrastructure_error") is True
    passed = bool(
        result.get("passed") is True
        and result.get("stage") == "verified"
        and all(stages.get(stage) is True for stage in MANDATORY_STAGES)
        and admitted
    )
    reward = -1 if infrastructure else int(passed)
    timing = result.get("profile", {}).get("timing", {}) if admitted else {}
    payload = {
        "schema_version": 1,
        "reward": reward,
        "stage_fractions": {
            stage: float(stages.get(stage) is True) for stage in MANDATORY_STAGES
        },
        "profiled": admitted,
        "speedup": timing.get("speedup"),
        "beats_xla": bool(admitted and timing.get("materially_beats_xla") is True),
        "failure_stage": None if passed else result.get("stage", "infrastructure"),
        "infrastructure_error": infrastructure,
        "worker_recovery_required": bool(result.get("worker_recovery_required", False)),
    }
    cases = []
    for stage in MANDATORY_STAGES:
        stage_passed = stages.get(stage) is True
        status = (
            "passed"
            if stage_passed
            else "failed"
            if stage == payload["failure_stage"]
            else "not_run"
        )
        cases.append(
            {
                "name": stage,
                "status": status,
                "message": result.get("error") if status == "failed" else None,
            }
        )
    seed_results = {
        item.get("seed"): item
        for item in result.get("seed_results", [])
        if isinstance(item, dict)
    }
    for seed in (0, 1, 2):
        record = seed_results.get(seed)
        cases.append(
            {
                "name": f"full_shape_correctness_seed_{seed}",
                "status": (
                    "passed"
                    if record is not None and record.get("passed") is True
                    else "failed"
                    if result.get("stage") == "full_shape_correctness"
                    and result.get("seed") == seed
                    else "not_run"
                ),
                "message": (
                    result.get("error")
                    if result.get("stage") == "full_shape_correctness"
                    and result.get("seed") == seed
                    else None
                ),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    numeric_reward = {
        "reward": reward,
        **{
            f"stage_{stage}": float(stages.get(stage) is True)
            for stage in MANDATORY_STAGES
        },
        "profiled": float(admitted),
        "speedup": float(timing.get("speedup") or 0.0),
        "beats_xla": float(admitted and timing.get("materially_beats_xla") is True),
        "infrastructure_error": float(infrastructure),
        "worker_recovery_required": float(
            bool(result.get("worker_recovery_required", False))
        ),
    }
    (root / "reward.json").write_text(
        json.dumps(numeric_reward, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "score.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "ctrf.json").write_text(
        json.dumps({"schema_version": 1, "tests": cases}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    run_log = root / "run.log"
    (root / "test-stdout.txt").write_text(
        run_log.read_text(encoding="utf-8") if run_log.is_file() else "",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase2-task-artifacts")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_artifacts(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
