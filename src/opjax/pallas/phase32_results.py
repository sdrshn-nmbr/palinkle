"""Validate and summarize the frozen Gate 3.2 base-model comparison."""

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
        raise G42HarnessError(f"PHASE32_RESULT_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE32_RESULT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_hash(value: dict[str, Any], field: str) -> None:
    payload = dict(value)
    expected = payload.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise G42HarnessError(f"PHASE32_RESULT_HASH_INVALID:{field}")


def summarize_behavior(*, sample_root: Path, provider: str) -> dict[str, Any]:
    manifest = _read_json(sample_root / "manifest.json")
    _validate_hash(manifest, "release_sha256")
    records = manifest.get("records", [])
    if manifest.get("provider") != provider or not records:
        raise G42HarnessError("PHASE32_SAMPLE_BEHAVIOR_CONTRACT_INVALID")
    calls_by_turn = {
        str(turn): {"calls": 0, "valid_actions": 0, "format_errors": 0}
        for turn in range(1, 7)
    }
    for record in records:
        path = sample_root / record["run_path"] / "trajectory.json"
        if file_sha256(path) != record["trajectory_sha256"]:
            raise G42HarnessError("PHASE32_SAMPLE_TRAJECTORY_HASH_INVALID")
        trajectory = _read_json(path)
        call_index = 0
        for message in trajectory.get("messages", []):
            is_action = message.get("role") == "assistant"
            is_format_error = (
                message.get("role") == "user"
                and message.get("extra", {}).get("interrupt_type") == "FormatError"
            )
            if not is_action and not is_format_error:
                continue
            call_index += 1
            if call_index > 6:
                raise G42HarnessError("PHASE32_SAMPLE_CALL_COUNT_INVALID")
            bucket = calls_by_turn[str(call_index)]
            bucket["calls"] += 1
            if is_action:
                actions = message.get("extra", {}).get("actions", [])
                bucket["valid_actions"] += int(len(actions) == 1)
            else:
                bucket["format_errors"] += 1
        api_calls = trajectory.get("info", {}).get("model_stats", {}).get("api_calls")
        if call_index != 6 or api_calls != 6:
            raise G42HarnessError("PHASE32_SAMPLE_CALL_COUNT_INVALID")
    return {
        "trajectories": len(records),
        "model_calls": sum(value["calls"] for value in calls_by_turn.values()),
        "valid_actions": sum(
            value["valid_actions"] for value in calls_by_turn.values()
        ),
        "format_errors": sum(
            value["format_errors"] for value in calls_by_turn.values()
        ),
        "submitted": manifest["counts"]["submitted"],
        "calls_by_turn": calls_by_turn,
    }


def summarize_model(
    *, experiment: dict[str, Any], grading_path: Path, sample_root: Path
) -> dict[str, Any]:
    grading = _read_json(grading_path)
    _validate_hash(grading, "result_sha256")
    provider = grading.get("provider")
    expected_keys = {
        (cell["task_id"], cell["seed"], turn)
        for cell in experiment["cells"]
        if cell["provider"] == provider
        for turn in (3, 6)
    }
    records = grading.get("records", [])
    observed_keys = {
        (record.get("task_id"), record.get("seed"), record.get("turn"))
        for record in records
    }
    if (
        grading.get("kind") != "opjax_phase31_base_capability_result"
        or grading.get("experiment_sha256") != experiment.get("experiment_sha256")
        or observed_keys != expected_keys
        or len(records) != len(expected_keys)
    ):
        raise G42HarnessError("PHASE32_MODEL_RESULT_CONTRACT_INVALID")
    by_cell = {
        (record["task_id"], record["seed"], record["turn"]): record
        for record in records
    }
    transitions = {
        "fail_to_pass": 0,
        "pass_to_pass": 0,
        "fail_to_fail": 0,
        "pass_to_fail": 0,
    }
    for task_id, seed in sorted(
        {(record["task_id"], record["seed"]) for record in records}
    ):
        at_three = by_cell[(task_id, seed, 3)]["reward"] == 1
        at_six = by_cell[(task_id, seed, 6)]["reward"] == 1
        transition = (
            "pass_to_pass"
            if at_three and at_six
            else "pass_to_fail"
            if at_three
            else "fail_to_pass"
            if at_six
            else "fail_to_fail"
        )
        transitions[transition] += 1
    stages: dict[str, int] = {}
    for record in records:
        stage = record.get("failure_stage") or "verified"
        stages[stage] = stages.get(stage, 0) + 1
    model = next(
        item for item in experiment["models"] if item["provider"] == provider
    )
    return {
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "provider": provider,
        "grading_sha256": file_sha256(grading_path),
        "horizons": grading["horizons"],
        "turn_3_to_6": transitions,
        "failure_stages": dict(sorted(stages.items())),
        "nonempty_snapshot_count": sum(
            record["patch_sha256"] != EMPTY_PATCH_SHA256 for record in records
        ),
        "agent_behavior": summarize_behavior(
            sample_root=sample_root,
            provider=provider,
        ),
    }


def build_comparison(
    *, experiment_path: Path, grading_paths: list[Path], sample_roots: list[Path]
) -> dict[str, Any]:
    experiment = _read_json(experiment_path)
    _validate_hash(experiment, "experiment_sha256")
    if experiment.get("kind") != "opjax_phase32_base_capability_experiment":
        raise G42HarnessError("PHASE32_EXPERIMENT_KIND_INVALID")
    sample_by_provider = {
        _read_json(root / "manifest.json")["provider"]: root for root in sample_roots
    }
    for root in sample_roots:
        _validate_sample_manifest(sample_root=root, experiment=experiment)
    summaries = []
    for path in grading_paths:
        provider = _read_json(path).get("provider")
        if provider not in sample_by_provider:
            raise G42HarnessError("PHASE32_COMPARISON_SAMPLE_PROVIDER_MISSING")
        summaries.append(
            summarize_model(
                experiment=experiment,
                grading_path=path,
                sample_root=sample_by_provider[provider],
            )
        )
    expected_providers = {model["provider"] for model in experiment["models"]}
    if {summary["provider"] for summary in summaries} != expected_providers:
        raise G42HarnessError("PHASE32_COMPARISON_PROVIDER_SET_INVALID")
    result = {
        "schema_version": 1,
        "kind": "opjax_phase32_base_capability_comparison",
        "experiment_sha256": experiment["experiment_sha256"],
        "benchmark_release_sha256": experiment["benchmark_release_sha256"],
        "orchestration_sha256": {
            "phase32_adjudication.py": file_sha256(
                Path(__file__).with_name("phase32_adjudication.py")
            ),
            "phase32_grading.py": file_sha256(Path(__file__).with_name("phase32_grading.py")),
            "phase32_results.py": file_sha256(Path(__file__)),
        },
        "models": sorted(summaries, key=lambda item: item["provider"]),
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m opjax.pallas.phase32_results")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/experiment.json"),
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
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (G42HarnessError, OSError, ValueError, StopIteration) as exc:
        print(f"PHASE32_RESULTS_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
