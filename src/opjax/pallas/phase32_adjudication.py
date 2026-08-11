"""Adjudicate deterministically reproduced candidate TPU safety failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256

DMA_FAILURE_MARKERS = (
    "RuntimeUnexpectedCoreHalt",
    "BoundsCheck",
    "hlo: workload.1",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE32_ADJUDICATION_JSON_INVALID:{path}")
    return value


def _validate_raw_attempt(path: Path) -> dict[str, Any]:
    result = _read_json(path / "result.json")
    reward = _read_json(path / "reward.json")
    submission = _read_json(path / "submission.json")
    error = str(result.get("error", ""))
    worker = submission.get("worker", {})
    if (
        result.get("reward") != -1
        or reward.get("reward") != -1
        or result.get("stage") != "infrastructure"
        or not all(marker in error for marker in DMA_FAILURE_MARKERS)
        or not worker.get("identity")
        or not worker.get("destroyed_at")
    ):
        raise G42HarnessError("PHASE32_DMA_ATTEMPT_INVALID")
    return {
        "result_sha256": file_sha256(path / "result.json"),
        "reward_sha256": file_sha256(path / "reward.json"),
        "submission_sha256": file_sha256(path / "submission.json"),
        "worker_identity": worker["identity"],
        "worker_destroyed_at": worker["destroyed_at"],
        "kernel_sha256": result.get("kernel_sha256"),
    }


def adjudicate_dma_failures(
    *, grading_path: Path, grading_root: Path
) -> dict[str, Any]:
    grading = _read_json(grading_path)
    payload = dict(grading)
    expected_hash = payload.pop("result_sha256", None)
    if canonical_sha256(payload) != expected_hash:
        raise G42HarnessError("PHASE32_GRADING_HASH_INVALID")
    records = []
    adjudicated = []
    for source in grading.get("records", []):
        record = dict(source)
        if record.get("reward") != -1:
            records.append(record)
            continue
        unit_id = record["unit_id"]
        first = _validate_raw_attempt(
            grading_root / "quarantine-infrastructure" / unit_id / "artifacts"
        )
        second = _validate_raw_attempt(grading_root / "results" / unit_id / "artifacts")
        if (
            first["worker_identity"] == second["worker_identity"]
            or first["kernel_sha256"] != second["kernel_sha256"]
        ):
            raise G42HarnessError("PHASE32_DMA_REPRODUCTION_INVALID")
        record.update(
            {
                "reward": 0,
                "failure_stage": "runtime_safety",
                "candidate_attributable": True,
                "correct": False,
                "authentic": False,
                "profiled": False,
                "speedup": None,
                "beats_xla": False,
                "adjudication": {
                    "kind": "deterministic_candidate_dma_bounds_halt",
                    "attempts": [first, second],
                },
            }
        )
        records.append(record)
        adjudicated.append(unit_id)
    if not adjudicated:
        raise G42HarnessError("PHASE32_DMA_ADJUDICATION_EMPTY")
    grading["records"] = records
    for turn in (3, 6):
        subset = [record for record in records if record["turn"] == turn]
        grading["horizons"][f"k{turn}"]["candidate_failures"] = sum(
            record["reward"] == 0 for record in subset
        )
        grading["horizons"][f"k{turn}"]["infrastructure_failures"] = sum(
            record["reward"] == -1 for record in subset
        )
    grading["adjudication"] = {
        "kind": "opjax_phase32_dma_safety_adjudication",
        "unit_ids": sorted(adjudicated),
        "source_sha256": file_sha256(Path(__file__)),
    }
    grading.pop("result_sha256", None)
    grading["result_sha256"] = canonical_sha256(grading)
    return grading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m opjax.pallas.phase32_adjudication")
    parser.add_argument("--grading", type=Path, required=True)
    parser.add_argument("--grading-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = adjudicate_dma_failures(
            grading_path=args.grading,
            grading_root=args.grading_root,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE32_ADJUDICATION_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["adjudication"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
