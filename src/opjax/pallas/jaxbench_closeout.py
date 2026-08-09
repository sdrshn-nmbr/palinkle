"""Validate and bind the Phase 2 full-JAXBench hardware evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opjax.pallas.jaxbench_executable import file_sha256
from opjax.pallas.jaxbench_worker import tree_sha256


EXPECTED_TASK_COUNT = 50
EXPECTED_UNSCOREABLE = {
    "11p_Megablox_GMM": "lower_compile",
    "16p_Mamba2_SSD": "lower_compile",
    "2p_GQA_Attention": "lower_compile",
}
EXECUTION_BOUNDARY = "sandbox-compile-serialized-executable-pristine-verify"


class JaxBenchCloseoutError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JaxBenchCloseoutError(f"EVIDENCE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise JaxBenchCloseoutError(f"EVIDENCE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise JaxBenchCloseoutError(code)


def _validate_submission(
    *,
    root: Path,
    release: dict[str, Any],
    expected_reward: int,
    expected_stage: str,
    expected_authentic: bool,
    expected_error_prefix: str | None = None,
    expected_pallas_custom_call_count: int | None = None,
    expected_tpu_custom_call_count: int | None = None,
) -> dict[str, Any]:
    submission = _load(root / "submission.json")
    result = _load(root / "result.json")
    reward = _load(root / "reward.json")
    _require(submission.get("release_sha256") == release["release_sha256"], "SUBMISSION_RELEASE_HASH_INVALID")
    task = next(
        (
            item
            for item in release["tasks"]
            if item["task_id"] == submission.get("task_id")
        ),
        None,
    )
    _require(task is not None, "SUBMISSION_TASK_UNKNOWN")
    _require(submission.get("task_sha256") == task["task_sha256"], "SUBMISSION_TASK_HASH_INVALID")
    _require(submission.get("execution_boundary") == EXECUTION_BOUNDARY, "SUBMISSION_BOUNDARY_INVALID")
    _require(submission.get("patch_sha256") == file_sha256(root / "model.patch"), "SUBMISSION_PATCH_HASH_INVALID")
    _require(submission.get("model_patch_sha256") == file_sha256(root / "model.patch"), "SUBMISSION_MODEL_PATCH_HASH_INVALID")
    _require(submission.get("result_sha256") == file_sha256(root / "result.json"), "SUBMISSION_RESULT_HASH_INVALID")
    _require(submission.get("reward_sha256") == file_sha256(root / "reward.json"), "SUBMISSION_REWARD_HASH_INVALID")
    worker = submission.get("worker", {})
    _require(worker.get("disposable") is True, "SUBMISSION_WORKER_NOT_DISPOSABLE")
    _require(bool(worker.get("destroyed_at")), "SUBMISSION_WORKER_NOT_DESTROYED")
    _require(worker.get("candidate_user") == "nobody", "SUBMISSION_CANDIDATE_USER_INVALID")
    _require(worker.get("execution_boundary") == EXECUTION_BOUNDARY, "SUBMISSION_WORKER_BOUNDARY_INVALID")
    _require(result.get("reward") == expected_reward, "SUBMISSION_RESULT_REWARD_INVALID")
    _require(reward.get("reward") == expected_reward, "SUBMISSION_REWARD_INVALID")
    _require(result.get("authentic") is expected_authentic, "SUBMISSION_AUTHENTICITY_INVALID")
    _require(result.get("infrastructure_error") is False, "SUBMISSION_INFRASTRUCTURE_FAILURE")
    if expected_reward == 1:
        _require(expected_stage == "verified", "REFERENCE_EXPECTATION_INVALID")
        _require(result.get("stage") == expected_stage, "REFERENCE_NOT_VERIFIED")
        _require(result.get("correct") is True, "REFERENCE_NOT_CORRECT")
        _require(result.get("profiled") is True, "REFERENCE_NOT_PROFILED")
        profile = result.get("profile", {})
        trace = root / "verification" / profile.get("trace_path", "")
        _require(trace.is_file(), "REFERENCE_TRACE_MISSING")
        _require(profile.get("trace_sha256") == file_sha256(trace), "REFERENCE_TRACE_HASH_INVALID")
        _require(profile.get("candidate_annotation_count", 0) >= 3, "REFERENCE_ANNOTATIONS_MISSING")
        _require(profile.get("tpu_execute_event_count", 0) >= 3, "REFERENCE_TPU_EXECUTION_MISSING")
        _require(profile.get("loaded_executable_event_count", 0) >= 3, "REFERENCE_EXECUTABLE_EVENTS_MISSING")
        hlo = (root / "verification/trusted-executable.hlo.txt").read_text(encoding="utf-8")
        _require("tpu_custom_call" in hlo, "REFERENCE_TPU_CUSTOM_CALL_MISSING")
    elif expected_stage == "full_shape_correctness":
        _require(result.get("stage") == expected_stage, "NEGATIVE_FAILURE_STAGE_INVALID")
        _require(result.get("correct") is False, "NEGATIVE_UNEXPECTEDLY_CORRECT")
        _require(result.get("candidate_attributable") is True, "NEGATIVE_NOT_CANDIDATE_ATTRIBUTABLE")
        _require(reward.get("failure_stage") == "full_shape_correctness", "NEGATIVE_REWARD_STAGE_INVALID")
    elif expected_stage == "normal_lowering":
        _require(result.get("stage") == expected_stage, "LOWERING_FAILURE_STAGE_INVALID")
        _require(result.get("candidate_attributable") is True, "LOWERING_NOT_CANDIDATE_ATTRIBUTABLE")
        _require(
            expected_error_prefix is not None
            and str(result.get("error", "")).startswith(expected_error_prefix),
            "LOWERING_REJECTION_REASON_INVALID",
        )
        _require(reward.get("failure_stage") == expected_stage, "LOWERING_REWARD_STAGE_INVALID")
        authenticity = result.get("authenticity", {})
        _require(authenticity.get("authentic") is False, "LOWERING_AUTHENTICITY_INVALID")
        if expected_pallas_custom_call_count is not None:
            _require(
                authenticity.get("pallas_custom_call_count")
                == expected_pallas_custom_call_count,
                "LOWERING_PALLAS_CUSTOM_CALL_COUNT_INVALID",
            )
        if expected_tpu_custom_call_count is not None:
            _require(
                authenticity.get("tpu_custom_call_count")
                == expected_tpu_custom_call_count,
                "LOWERING_TPU_CUSTOM_CALL_COUNT_INVALID",
            )
        hlo = (root / "verification/trusted-executable.hlo.txt").read_text(encoding="utf-8")
        _require("tpu_custom_call" in hlo, "LOWERING_TPU_CUSTOM_CALL_MISSING")
    else:
        raise JaxBenchCloseoutError(f"SUBMISSION_EXPECTED_STAGE_INVALID:{expected_stage}")
    return {
        "artifact_tree_sha256": tree_sha256(root),
        "task_id": submission["task_id"],
        "task_sha256": submission["task_sha256"],
        "patch_sha256": submission["patch_sha256"],
        "result_sha256": submission["result_sha256"],
        "reward_sha256": submission["reward_sha256"],
        "worker_identity": worker["identity"],
        "worker_destroyed_at": worker["destroyed_at"],
    }


def _validate_scoreability(
    *, root: Path, release: dict[str, Any]
) -> dict[str, Any]:
    matrix_path = root / "matrix.json"
    matrix = _load(matrix_path)
    _require(matrix.get("kind") == "opjax_jaxbench_original_shape_scoreability", "SCOREABILITY_KIND_INVALID")
    _require(matrix.get("release_sha256") == release["release_sha256"], "SCOREABILITY_RELEASE_HASH_INVALID")
    _require(matrix.get("complete_release") is True, "SCOREABILITY_RELEASE_INCOMPLETE")
    _require(matrix.get("task_count") == EXPECTED_TASK_COUNT, "SCOREABILITY_TASK_COUNT_INVALID")
    _require(matrix.get("scoreable_count") == 47, "SCOREABILITY_PASS_COUNT_INVALID")
    _require(matrix.get("unscoreable_count") == 3, "SCOREABILITY_FAILURE_COUNT_INVALID")
    _require(matrix.get("runner_sha256") == file_sha256(Path(__file__).with_name("jaxbench_scoreability.py")), "SCOREABILITY_RUNNER_HASH_INVALID")
    runtime = matrix.get("runtime", {})
    expected_runtime = release["runtime"]
    for name in ("python", "jax", "jaxlib", "libtpu"):
        _require(runtime.get(name) == expected_runtime[name], f"SCOREABILITY_RUNTIME_INVALID:{name}")
    _require(runtime.get("backend") == "tpu", "SCOREABILITY_BACKEND_INVALID")
    _require("TPU v5 lite" in runtime.get("device_kinds", []), "SCOREABILITY_DEVICE_KIND_INVALID")
    _require(
        matrix.get("worker_requirements_lock_sha256")
        == release["worker_requirements_lock_sha256"],
        "SCOREABILITY_REQUIREMENTS_LOCK_INVALID",
    )
    release_tasks = {task["task_id"]: task for task in release["tasks"]}
    results = matrix.get("results", [])
    _require(
        isinstance(results, list)
        and {result.get("task_id") for result in results} == set(release_tasks),
        "SCOREABILITY_TASK_SET_INVALID",
    )
    observed_unscoreable: dict[str, str] = {}
    for result in results:
        task_id = result["task_id"]
        task = release_tasks[task_id]
        _require(result.get("release_sha256") == release["release_sha256"], f"SCOREABILITY_RESULT_RELEASE_INVALID:{task_id}")
        _require(result.get("task_sha256") == task["task_sha256"], f"SCOREABILITY_RESULT_TASK_INVALID:{task_id}")
        _require(result.get("baseline_sha256") == task["baseline_sha256"], f"SCOREABILITY_BASELINE_INVALID:{task_id}")
        _require(result.get("candidate_attributable") is False, f"SCOREABILITY_ATTRIBUTION_INVALID:{task_id}")
        _require(result.get("runtime") == runtime, f"SCOREABILITY_RESULT_RUNTIME_INVALID:{task_id}")
        _require(
            result.get("worker_requirements_lock_sha256")
            == release["worker_requirements_lock_sha256"],
            f"SCOREABILITY_RESULT_REQUIREMENTS_LOCK_INVALID:{task_id}",
        )
        _require(_load(root / f"{task_id}.json") == result, f"SCOREABILITY_TASK_ARTIFACT_INVALID:{task_id}")
        if result.get("status") == "scoreable":
            _require(result.get("stage") == "execute", f"SCOREABILITY_STAGE_INVALID:{task_id}")
            _require(result.get("platform") == "tpu", f"SCOREABILITY_PLATFORM_INVALID:{task_id}")
            _require(result.get("device_count", 0) >= 1, f"SCOREABILITY_DEVICE_INVALID:{task_id}")
            _require(bool(result.get("executable_sha256")), f"SCOREABILITY_EXECUTABLE_HASH_MISSING:{task_id}")
        else:
            _require(result.get("status") == "unscoreable", f"SCOREABILITY_STATUS_INVALID:{task_id}")
            _require(result.get("classification") == "pinned_baseline_failure", f"SCOREABILITY_CLASSIFICATION_INVALID:{task_id}")
            observed_unscoreable[task_id] = result.get("stage")
    _require(observed_unscoreable == EXPECTED_UNSCOREABLE, "SCOREABILITY_FAILURE_SET_INVALID")
    return {
        "artifact_tree_sha256": tree_sha256(root),
        "matrix_sha256": file_sha256(matrix_path),
        "runner_sha256": matrix["runner_sha256"],
        "task_count": matrix["task_count"],
        "scoreable_count": matrix["scoreable_count"],
        "unscoreable_count": matrix["unscoreable_count"],
        "unscoreable": EXPECTED_UNSCOREABLE,
    }


def build_closeout(
    *,
    repo_root: Path,
    out_dir: Path,
    scoreability_worker: str,
    scoreability_zone: str,
    scoreability_accelerator: str,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchCloseoutError(f"CLOSEOUT_OUTPUT_EXISTS:{out_dir}")
    release_root = repo_root / "data/pallas/benchmarks/jaxbench-v1"
    release = _load(release_root / "manifest.json")
    reference_root = repo_root / "data/pallas/runs/jaxbench-full-v1-adapter-reference"
    wrong_root = repo_root / "data/pallas/runs/jaxbench-full-v1-adapter-wrong"
    mixed_root = repo_root / "data/pallas/runs/jaxbench-full-v1-adapter-mixed"
    xla_custom_call_root = repo_root / "data/pallas/runs/jaxbench-full-v1-adapter-xla-custom-call"
    scoreability_root = repo_root / "data/pallas/runs/jaxbench-full-v1-scoreability"
    record = {
        "schema_version": 1,
        "kind": "opjax_phase2_full_jaxbench_closeout",
        "status": "completed",
        "release_sha256": release["release_sha256"],
        "jaxbench_revision": release["jaxbench_revision"],
        "shape_policy": "original_unmodified",
        "capability_task_count": EXPECTED_TASK_COUNT,
        "scoreable_task_count": 47,
        "excluded_task_count": 3,
        "reference": _validate_submission(
            root=reference_root,
            release=release,
            expected_reward=1,
            expected_stage="verified",
            expected_authentic=True,
        ),
        "adversarial_wrong": _validate_submission(
            root=wrong_root,
            release=release,
            expected_reward=0,
            expected_stage="full_shape_correctness",
            expected_authentic=True,
        ),
        "adversarial_mixed": _validate_submission(
            root=mixed_root,
            release=release,
            expected_reward=0,
            expected_stage="normal_lowering",
            expected_authentic=False,
            expected_error_prefix="HLO_COMPUTE_OUTSIDE_PALLAS:",
        ),
        "adversarial_xla_custom_call": _validate_submission(
            root=xla_custom_call_root,
            release=release,
            expected_reward=0,
            expected_stage="normal_lowering",
            expected_authentic=False,
            expected_error_prefix="HLO_COMPUTE_OUTSIDE_PALLAS:",
            expected_pallas_custom_call_count=0,
            expected_tpu_custom_call_count=1,
        ),
        "scoreability": _validate_scoreability(root=scoreability_root, release=release),
        "scoreability_worker": {
            "identity": scoreability_worker,
            "zone": scoreability_zone,
            "accelerator_type": scoreability_accelerator,
            "trust_boundary": "trusted_pinned_baseline_probe",
        },
        "closeout_source_sha256": file_sha256(Path(__file__)),
    }
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def validate_closeout(*, repo_root: Path, closeout_root: Path) -> dict[str, Any]:
    recorded = _load(closeout_root / "manifest.json")
    temporary = closeout_root.with_name(f".{closeout_root.name}.validation")
    if temporary.exists():
        raise JaxBenchCloseoutError(f"CLOSEOUT_VALIDATION_OUTPUT_EXISTS:{temporary}")
    rebuilt = build_closeout(
        repo_root=repo_root,
        out_dir=temporary,
        scoreability_worker=recorded["scoreability_worker"]["identity"],
        scoreability_zone=recorded["scoreability_worker"]["zone"],
        scoreability_accelerator=recorded["scoreability_worker"]["accelerator_type"],
    )
    try:
        _require(rebuilt == recorded, "CLOSEOUT_MANIFEST_INVALID")
    finally:
        (temporary / "manifest.json").unlink(missing_ok=True)
        temporary.rmdir()
    return recorded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-closeout")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--repo", type=Path, required=True)
        subparser.add_argument("--out", type=Path, required=True)
        if command == "build":
            subparser.add_argument("--scoreability-worker", required=True)
            subparser.add_argument("--scoreability-zone", required=True)
            subparser.add_argument("--scoreability-accelerator", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_closeout(
            repo_root=args.repo,
            out_dir=args.out,
            scoreability_worker=args.scoreability_worker,
            scoreability_zone=args.scoreability_zone,
            scoreability_accelerator=args.scoreability_accelerator,
        )
    else:
        result = validate_closeout(repo_root=args.repo, closeout_root=args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
