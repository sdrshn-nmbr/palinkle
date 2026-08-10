"""Validate and summarize the frozen Phase 3 base-model comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.phase3_grading import EMPTY_PATCH_SHA256, _validate_sample_manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G42HarnessError(f"PHASE3_RESULT_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE3_RESULT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_embedded_hash(value: dict[str, Any], field: str) -> None:
    payload = dict(value)
    expected = payload.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise G42HarnessError(f"PHASE3_RESULT_HASH_INVALID:{field}")


def summarize_model(
    *, experiment: dict[str, Any], grading_path: Path, sample_root: Path | None = None
) -> dict[str, Any]:
    grading = _read_json(grading_path)
    _validate_embedded_hash(grading, "result_sha256")
    expected_kind = (
        "opjax_phase31_base_capability_result"
        if experiment.get("kind") == "opjax_phase31_base_capability_experiment"
        else "opjax_phase3_base_capability_result"
    )
    provider = grading.get("provider")
    expected_keys = {
        (cell["task_id"], cell["seed"], turn)
        for cell in experiment["cells"]
        if cell["provider"] == provider
        for turn in (3, 6)
    }
    if (
        grading.get("kind") != expected_kind
        or grading.get("experiment_sha256") != experiment.get("experiment_sha256")
        or grading.get("counts", {}).get("trajectories") != len(expected_keys) // 2
        or grading.get("counts", {}).get("snapshots") != len(expected_keys)
    ):
        raise G42HarnessError("PHASE3_MODEL_RESULT_CONTRACT_INVALID")
    records = grading["records"]
    keys = {(record["task_id"], record["seed"], record["turn"]) for record in records}
    if keys != expected_keys or len(records) != len(expected_keys):
        raise G42HarnessError("PHASE3_MODEL_RESULT_CELL_SET_INVALID")
    transitions = {
        "fail_to_pass": 0,
        "pass_to_pass": 0,
        "fail_to_fail": 0,
        "pass_to_fail": 0,
    }
    by_cell = {
        (record["task_id"], record["seed"], record["turn"]): record
        for record in records
    }
    for task_id, seed in sorted({(record["task_id"], record["seed"]) for record in records}):
        at_three = by_cell[(task_id, seed, 3)]["reward"] == 1
        at_six = by_cell[(task_id, seed, 6)]["reward"] == 1
        key = (
            "pass_to_pass"
            if at_three and at_six
            else "pass_to_fail"
            if at_three
            else "fail_to_pass"
            if at_six
            else "fail_to_fail"
        )
        transitions[key] += 1
    stages: dict[str, int] = {}
    for record in records:
        stage = record.get("failure_stage") or "verified"
        stages[stage] = stages.get(stage, 0) + 1
    model = next(
        item
        for item in experiment["models"]
        if item["provider"] == grading["provider"]
    )
    summary = {
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "provider": grading["provider"],
        "grading_sha256": file_sha256(grading_path),
        "horizons": grading["horizons"],
        "turn_3_to_6": transitions,
        "failure_stages": dict(sorted(stages.items())),
        "nonempty_snapshot_count": sum(
            record["patch_sha256"] != EMPTY_PATCH_SHA256 for record in records
        ),
    }
    if sample_root is not None:
        summary["agent_behavior"] = summarize_behavior(
            sample_root=sample_root,
            provider=grading["provider"],
        )
    return summary


def summarize_behavior(*, sample_root: Path, provider: str) -> dict[str, Any]:
    manifest = _read_json(sample_root / "manifest.json")
    _validate_embedded_hash(manifest, "release_sha256")
    records = manifest.get("records", [])
    if manifest.get("provider") != provider or not records:
        raise G42HarnessError("PHASE3_SAMPLE_BEHAVIOR_CONTRACT_INVALID")
    calls_by_turn = {
        str(turn): {"calls": 0, "valid_actions": 0, "format_errors": 0}
        for turn in range(1, 7)
    }
    total_calls = 0
    total_format_errors = 0
    for record in records:
        path = sample_root / record["run_path"] / "trajectory.json"
        if file_sha256(path) != record["trajectory_sha256"]:
            raise G42HarnessError("PHASE3_SAMPLE_TRAJECTORY_HASH_INVALID")
        trajectory = _read_json(path)
        assistants = [
            message
            for message in trajectory.get("messages", [])
            if message.get("role") == "assistant"
        ]
        if len(assistants) != 6:
            raise G42HarnessError("PHASE3_SAMPLE_CALL_COUNT_INVALID")
        total_calls += len(assistants)
        for turn, message in enumerate(assistants, start=1):
            actions = message.get("extra", {}).get("actions", [])
            calls_by_turn[str(turn)]["calls"] += 1
            calls_by_turn[str(turn)]["valid_actions"] += int(len(actions) == 1)
        assistant_seen = 0
        for message in trajectory.get("messages", []):
            assistant_seen += int(message.get("role") == "assistant")
            if message.get("extra", {}).get("interrupt_type") == "FormatError":
                total_format_errors += 1
                if 1 <= assistant_seen <= 6:
                    calls_by_turn[str(assistant_seen)]["format_errors"] += 1
    return {
        "trajectories": len(records),
        "model_calls": total_calls,
        "valid_actions": sum(
            value["valid_actions"] for value in calls_by_turn.values()
        ),
        "format_errors": total_format_errors,
        "submitted": manifest["counts"]["submitted"],
        "calls_by_turn": calls_by_turn,
    }


def build_comparison(
    *, experiment_path: Path, grading_paths: list[Path], sample_roots: list[Path]
) -> dict[str, Any]:
    experiment = _read_json(experiment_path)
    _validate_embedded_hash(experiment, "experiment_sha256")
    sample_by_provider = {
        _read_json(root / "manifest.json")["provider"]: root for root in sample_roots
    }
    summaries = []
    for root in sample_roots:
        _validate_sample_manifest(sample_root=root, experiment=experiment)
    for path in grading_paths:
        grading = _read_json(path)
        provider = grading.get("provider")
        if provider not in sample_by_provider:
            raise G42HarnessError("PHASE3_COMPARISON_SAMPLE_PROVIDER_MISSING")
        summaries.append(
            summarize_model(
                experiment=experiment,
                grading_path=path,
                sample_root=sample_by_provider[provider],
            )
        )
    if {summary["provider"] for summary in summaries} != {"tinker", "sglang"}:
        raise G42HarnessError("PHASE3_COMPARISON_PROVIDER_SET_INVALID")
    result = {
        "schema_version": 2 if experiment.get("phase") == "3.1" else 1,
        "kind": (
            "opjax_phase31_base_capability_comparison"
            if experiment.get("phase") == "3.1"
            else "opjax_phase3_base_capability_comparison"
        ),
        "experiment_sha256": experiment["experiment_sha256"],
        "benchmark_release_sha256": experiment["benchmark_release_sha256"],
        "models": sorted(summaries, key=lambda item: item["provider"]),
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-results")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase3-base-capability/experiment.json"),
    )
    parser.add_argument("--grading", type=Path, action="append", required=True)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_comparison(
            experiment_path=args.experiment,
            grading_paths=args.grading,
            sample_roots=args.sample,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (G42HarnessError, OSError, ValueError, StopIteration) as exc:
        print(f"PHASE3_RESULTS_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
