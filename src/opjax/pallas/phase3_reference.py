"""Freeze the Phase 3 optimized-reference compatibility calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256

CALIBRATION_TASKS = (
    "1p_Flash_Attention",
    "3p_MLA_Attention",
    "4p_Sparse_Attention",
    "6p_Paged_Attention",
    "7p_Ragged_Paged_Attention",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G42HarnessError(f"PHASE3_REFERENCE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE3_REFERENCE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _record(*, task_id: str, root: Path) -> dict[str, Any]:
    result_path = root / "result.json"
    reward_path = root / "reward.json"
    submission_path = root / "submission.json"
    result = _read_json(result_path)
    reward = _read_json(reward_path)
    submission = _read_json(submission_path)
    worker = submission.get("worker")
    if (
        result.get("task_id") != task_id
        or submission.get("task_id") != task_id
        or reward.get("reward") != result.get("reward")
        or submission.get("result_sha256") != file_sha256(result_path)
        or submission.get("reward_sha256") != file_sha256(reward_path)
        or not isinstance(worker, dict)
        or worker.get("disposable") is not True
        or not worker.get("destroyed_at")
    ):
        raise G42HarnessError(f"PHASE3_REFERENCE_EVIDENCE_INVALID:{task_id}")
    timing = result.get("timing") or {}
    interval = timing.get("speedup_ci95")
    admitted = (
        result.get("reward") == 1
        and isinstance(interval, list)
        and len(interval) == 2
        and interval[0] > 1.05
    )
    return {
        "task_id": task_id,
        "task_sha256": submission["task_sha256"],
        "reward": result["reward"],
        "stage": result["stage"],
        "error": result.get("error"),
        "correct": result.get("correct", False),
        "authentic": result.get("authentic", False),
        "profiled": result.get("profiled", False),
        "speedup": result.get("speedup"),
        "speedup_ci95": interval,
        "headroom_admitted": admitted,
        "result_sha256": file_sha256(result_path),
        "reward_sha256": file_sha256(reward_path),
        "submission_sha256": file_sha256(submission_path),
        "worker_identity": worker["identity"],
        "worker_destroyed_at": worker["destroyed_at"],
    }


def build_calibration(
    *, calibration_root: Path, phase2_reference_root: Path, release_root: Path
) -> dict[str, Any]:
    release = _read_json(release_root / "manifest.json")
    records = [
        _record(task_id=task_id, root=calibration_root / task_id)
        for task_id in CALIBRATION_TASKS
    ]
    records.append(_record(task_id="8p_GEMM", root=phase2_reference_root))
    if len({record["worker_identity"] for record in records}) != len(records):
        raise G42HarnessError("PHASE3_REFERENCE_WORKER_REUSE")
    result = {
        "schema_version": 1,
        "kind": "opjax_phase3_optimized_reference_calibration",
        "benchmark_release_sha256": release["release_sha256"],
        "policy": {
            "scope": "scoreable_tasks_with_pinned_optimized_references",
            "headroom_threshold": "speedup_ci95_lower_bound_gt_1.05",
            "failed_reference_policy": "record_runtime_or_api_incompatibility_without_repairing_semantics",
        },
        "counts": {
            "references": len(records),
            "verified": sum(record["reward"] == 1 for record in records),
            "headroom_admitted": sum(record["headroom_admitted"] for record in records),
        },
        "records": sorted(records, key=lambda record: record["task_id"]),
    }
    result["release_sha256"] = canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-reference")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("data/pallas/runs/phase3-reference-calibration"),
    )
    parser.add_argument(
        "--phase2-reference-root",
        type=Path,
        default=Path("data/pallas/runs/jaxbench-full-v1-adapter-reference"),
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-v1"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_calibration(
            calibration_root=args.calibration_root,
            phase2_reference_root=args.phase2_reference_root,
            release_root=args.release_root,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE3_REFERENCE_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
