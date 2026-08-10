"""Local orchestration for the frozen Laguna XS 2.1 Pallas baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import canonical_sha256, file_sha256, load_task_package
from opjax.pallas.g43_corpus import validate_benchmark_release
from opjax.pallas.phase3_baseline import load_phase3_contract, validate_sample_matrix
from opjax.pallas.phase3_sampling import sample_sglang_matrix
from opjax.pallas.phase31_experiment import load_contract, validate_experiment
from opjax.pallas.sglang_agent import run_sglang_agent
from opjax.remote.laguna_sglang import (
    MODEL_ID,
    MODEL_REVISION,
    PRECISION,
    SGLANG_REVISION,
    LagunaEngine,
    app,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def summarize_baseline(
    *,
    verifier_root: Path,
    out_path: Path,
    reference_sample_root: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((verifier_root / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (verifier_root / "verification.json").read_text(encoding="utf-8")
    )
    manifest_payload = dict(manifest)
    manifest_sha = manifest_payload.pop("release_sha256")
    verification_payload = dict(verification)
    verification_sha = verification_payload.pop("release_sha256")
    if canonical_sha256(manifest_payload) != manifest_sha:
        raise RuntimeError("LAGUNA_BASELINE_MANIFEST_HASH_MISMATCH")
    if canonical_sha256(verification_payload) != verification_sha:
        raise RuntimeError("LAGUNA_BASELINE_VERIFICATION_HASH_MISMATCH")
    if verification["input_release_sha256"] != manifest_sha:
        raise RuntimeError("LAGUNA_BASELINE_RELEASE_MISMATCH")
    if verification["counts"]["infrastructure_failures"] != 0:
        raise RuntimeError("LAGUNA_BASELINE_INFRASTRUCTURE_FAILURES_PRESENT")
    verification_records = {
        record["unit_id"]: record for record in verification["records"]
    }
    if set(verification_records) != {
        record["unit_id"] for record in manifest["records"]
    }:
        raise RuntimeError("LAGUNA_BASELINE_VERIFICATION_UNITS_MISMATCH")
    records = []
    for record in manifest["records"]:
        reward_path = verifier_root / "results" / record["unit_id"] / "reward.json"
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        verified_record = verification_records[record["unit_id"]]
        reward_sha256 = file_sha256(reward_path)
        if verified_record["artifacts"]["reward.json"] != reward_sha256:
            raise RuntimeError(
                f"LAGUNA_BASELINE_REWARD_HASH_MISMATCH: {record['unit_id']}"
            )
        if verified_record["reward"] != reward["reward"]:
            raise RuntimeError(
                f"LAGUNA_BASELINE_REWARD_VALUE_MISMATCH: {record['unit_id']}"
            )
        records.append(
            {
                "task_id": record["task_id"],
                "family": record["family"],
                "turn": record["turn"],
                "reward": reward["reward"],
                "failure_stage": reward.get("failure_stage"),
                "patch_sha256": record["patch_sha256"],
                "nonempty_patch": record["patch_sha256"]
                != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "reward_sha256": reward_sha256,
            }
        )
    failure_stages: dict[str, int] = {}
    for record in records:
        stage = record["failure_stage"] or "none"
        failure_stages[stage] = failure_stages.get(stage, 0) + 1
    latest_by_task: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        current = latest_by_task.get(record["task_id"])
        if current is None or record["turn"] > current["turn"]:
            latest_by_task[record["task_id"]] = record
    commands_by_call: dict[str, dict[str, int]] = {}
    total_calls = 0
    format_errors = 0
    submitted = 0
    trajectories: dict[str, dict[str, Any]] = {}
    for task_id, record in sorted(latest_by_task.items()):
        trajectory_path = (
            verifier_root / "units" / record["unit_id"] / "trajectory.json"
        )
        if file_sha256(trajectory_path) != record["trajectory_sha256"]:
            raise RuntimeError(f"LAGUNA_BASELINE_TRAJECTORY_HASH_MISMATCH: {task_id}")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        trajectories[task_id] = trajectory
        assistant_messages = [
            message
            for message in trajectory["messages"]
            if message.get("role") == "assistant"
        ]
        total_calls += len(assistant_messages)
        format_errors += sum(
            message.get("extra", {}).get("interrupt_type") == "FormatError"
            for message in trajectory["messages"]
            if isinstance(message.get("extra"), dict)
        )
        submitted += int(trajectory["g42"]["submitted"] is True)
        for call, message in enumerate(assistant_messages, start=1):
            actions = message.get("extra", {}).get("actions", [])
            command = actions[0]["command"] if len(actions) == 1 else "<invalid>"
            call_counts = commands_by_call.setdefault(str(call), {})
            call_counts[command] = call_counts.get(command, 0) + 1
    prefix_reproducibility = None
    if reference_sample_root is not None:
        reference_manifest = json.loads(
            (reference_sample_root / "manifest.json").read_text(encoding="utf-8")
        )
        reference_payload = dict(reference_manifest)
        reference_sha = reference_payload.pop("release_sha256")
        if canonical_sha256(reference_payload) != reference_sha:
            raise RuntimeError("LAGUNA_BASELINE_REFERENCE_SAMPLE_HASH_MISMATCH")
        reference_runs = {
            record["task_id"]: reference_sample_root / record["run_path"]
            for record in reference_manifest["records"]
        }
        if set(reference_runs) != set(trajectories):
            raise RuntimeError("LAGUNA_BASELINE_REFERENCE_TASKS_MISMATCH")
        content_matches = 0
        action_matches = 0
        patch_matches = 0
        for task_id, trajectory in trajectories.items():
            reference_run = reference_runs[task_id]
            reference_trajectory = json.loads(
                (reference_run / "trajectory.json").read_text(encoding="utf-8")
            )
            reference_messages = [
                message
                for message in reference_trajectory["messages"]
                if message.get("role") == "assistant"
            ][:3]
            current_messages = [
                message
                for message in trajectory["messages"]
                if message.get("role") == "assistant"
            ][:3]
            content_matches += [
                message["content"] for message in reference_messages
            ] == [message["content"] for message in current_messages]
            action_matches += [
                message.get("extra", {}).get("actions")
                for message in reference_messages
            ] == [
                message.get("extra", {}).get("actions") for message in current_messages
            ]
            current_record = next(
                record
                for record in manifest["records"]
                if record["task_id"] == task_id and record["turn"] == 3
            )
            current_patch = (
                verifier_root / "units" / current_record["unit_id"] / "model.patch"
            )
            patch_matches += (
                current_patch.read_bytes()
                == (reference_run / "snapshots" / "turn-3.patch").read_bytes()
            )
        prefix_reproducibility = {
            "reference_sample_release_sha256": reference_sha,
            "tasks": len(trajectories),
            "exact_content_prefix_matches": content_matches,
            "exact_action_prefix_matches": action_matches,
            "exact_turn_3_patch_matches": patch_matches,
        }
    horizons = {}
    for turn in sorted({record["turn"] for record in records}):
        subset = [record for record in records if record["turn"] == turn]
        horizons[f"k{turn}"] = {
            "units": len(subset),
            "profile_verified": sum(record["reward"] == 1 for record in subset),
            "candidate_failures": sum(record["reward"] == 0 for record in subset),
            "infrastructure_failures": sum(record["reward"] == -1 for record in subset),
            "nonempty_patches": sum(record["nonempty_patch"] for record in subset),
        }
    transitions = None
    if set(horizons) == {"k3", "k6"}:
        by_cell = {(record["task_id"], record["turn"]): record for record in records}
        transitions = {
            "fail_to_pass": 0,
            "pass_to_pass": 0,
            "fail_to_fail": 0,
            "pass_to_fail": 0,
        }
        for task_id in sorted({record["task_id"] for record in records}):
            passed_at_3 = by_cell[(task_id, 3)]["reward"] == 1
            passed_at_6 = by_cell[(task_id, 6)]["reward"] == 1
            key = (
                "pass_to_pass"
                if passed_at_3 and passed_at_6
                else "pass_to_fail"
                if passed_at_3
                else "fail_to_pass"
                if passed_at_6
                else "fail_to_fail"
            )
            transitions[key] += 1
    result = {
        "schema_version": 1,
        "kind": "pallas_laguna_xs_21_baseline_result",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": "sglang",
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
        },
        "sampling": {
            "turn_limit": 3,
            "seed": 0,
            "max_tokens": 8192,
            "temperature": 0.2,
            "top_p": 0.95,
            "thinking": True,
        },
        "observed_h200_residency_mib": {
            "source": "modal_canary_nvidia_smi_after_cuda_graph_capture",
            "total": 143771,
            "used_after_cuda_graph_capture": 106044,
            "remaining": 37727,
        },
        "counts": {
            "tasks": len({record["task_id"] for record in records}),
            "units": len(records),
            "profile_verified": sum(record["reward"] == 1 for record in records),
            "candidate_failures": sum(record["reward"] == 0 for record in records),
            "infrastructure_failures": sum(
                record["reward"] == -1 for record in records
            ),
            "nonempty_patches": sum(record["nonempty_patch"] for record in records),
        },
        "horizons": horizons,
        "turn_3_to_6_transitions": transitions,
        "agent_behavior": {
            "trajectories": len(trajectories),
            "model_calls": total_calls,
            "format_errors": format_errors,
            "submitted": submitted,
            "commands_by_call": commands_by_call,
        },
        "prefix_reproducibility": prefix_reproducibility,
        "failure_stages": dict(sorted(failure_stages.items())),
        "records": records,
        "verifier_input_release_sha256": manifest_sha,
        "verification_release_sha256": verification_sha,
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(out_path, result)
    return result


@app.local_entrypoint()
def canary() -> None:
    engine = LagunaEngine()
    print(json.dumps(engine.smoke.remote(), indent=2, sort_keys=True))
    response = engine.generate.remote(
        [{"role": "user", "content": "Return exactly: READY"}],
        {"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0, "sampling_seed": 0},
    )
    print(json.dumps(response, indent=2, sort_keys=True))


@app.local_entrypoint()
def baseline(
    benchmark_root: str = "data/pallas/runs/g43-benchmark-release",
    out_dir: str = "data/pallas/runs/laguna-xs-21-baseline-samples",
    limit: int = 0,
    turn_limit: int = 3,
) -> None:
    benchmark_path = Path(benchmark_root).resolve()
    output_path = Path(out_dir).resolve()
    if output_path.exists():
        raise RuntimeError(f"LAGUNA_BASELINE_OUTPUT_EXISTS: {output_path}")
    if turn_limit not in {3, 6}:
        raise RuntimeError(f"LAGUNA_BASELINE_TURN_LIMIT_INVALID: {turn_limit}")
    snapshot_turns = (3,) if turn_limit == 3 else (3, 6)
    validation = validate_benchmark_release(benchmark_path)
    benchmark = json.loads(
        (benchmark_path / "manifest.json").read_text(encoding="utf-8")
    )
    tasks = [
        load_task_package(benchmark_path / relative) for relative in benchmark["tasks"]
    ]
    if limit > 0:
        tasks = tasks[:limit]
    engine = LagunaEngine()

    def generate(
        messages: list[dict[str, Any]], sampling: dict[str, Any]
    ) -> dict[str, Any]:
        return engine.generate.remote(messages, sampling)

    records = []
    for index, task in enumerate(tasks, start=1):
        run_id = f"laguna-xs-21-base--{task.task_id}--seed-0"
        run_root = output_path / "runs" / run_id
        run_sglang_agent(
            task_dir=task.root,
            output_dir=run_root,
            generate=generate,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            runtime_revision=SGLANG_REVISION,
            precision=PRECISION,
            seed=0,
            max_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            turn_limit=turn_limit,
            snapshot_turns=snapshot_turns,
        )
        records.append(
            {
                "model_id": "laguna-xs-21-base",
                "checkpoint": MODEL_ID,
                "group": "external_baseline",
                "trajectory_count": None,
                "training_seed": None,
                "seed": 0,
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "family": task.family,
                "run_path": f"runs/{run_id}",
                "trajectory_sha256": file_sha256(run_root / "trajectory.json"),
            }
        )
        print(
            f"LAGUNA_BASELINE_SAMPLE completed={index}/{len(tasks)} task={task.task_id}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g43_sample_matrix",
        "evaluation_config_sha256": None,
        "benchmark_release_sha256": validation["release_sha256"],
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": "sglang",
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
        },
        "sampling": {
            "turn_limit": turn_limit,
            "seed": 0,
            "max_tokens": 8192,
            "temperature": 0.2,
            "top_p": 0.95,
            "thinking": True,
        },
        "snapshot_turns": list(snapshot_turns),
        "counts": {
            "runs": len(records),
            "snapshots": len(records) * len(snapshot_turns),
        },
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(output_path / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


@app.local_entrypoint()
def phase3(
    release_root: str = "data/pallas/benchmarks/jaxbench-v1",
    closeout_root: str = "data/pallas/runs/jaxbench-full-v1-closeout",
    experiment_path: str = "data/pallas/runs/phase3-base-capability/experiment.json",
    out_dir: str = "data/pallas/runs/phase3-base-capability/laguna-samples",
    task_ids: str = "",
    seeds: str = "0,1,2",
) -> None:
    release_path = Path(release_root).resolve()
    closeout_path = Path(closeout_root).resolve()
    experiment_file = Path(experiment_path).resolve()
    output_path = Path(out_dir).resolve()
    contract = load_phase3_contract(
        release_root=release_path,
        closeout_root=closeout_path,
    )
    validate_sample_matrix(path=experiment_file, contract=contract)
    experiment = json.loads(experiment_file.read_text(encoding="utf-8"))
    selected_tasks = {value for value in task_ids.split(",") if value} or None
    selected_seeds = {int(value) for value in seeds.split(",") if value}
    engine = LagunaEngine()

    def generate(
        messages: list[dict[str, Any]], sampling: dict[str, Any]
    ) -> dict[str, Any]:
        return engine.generate.remote(messages, sampling)

    manifest = sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        output_root=output_path,
        generate=generate,
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        task_ids=selected_tasks,
        seeds=selected_seeds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


@app.local_entrypoint()
def phase31(
    release_root: str = "data/pallas/benchmarks/jaxbench-phase31",
    validity_path: str = "data/pallas/runs/phase31-oracle-validity/manifest.json",
    calibration_path: str = "data/pallas/runs/phase31-positive-control-calibration/manifest.json",
    experiment_path: str = "data/pallas/runs/phase31-base-capability/experiment.json",
    out_dir: str = "data/pallas/runs/phase31-base-capability/laguna-samples",
    task_ids: str = "",
    seeds: str = "0,1,2",
) -> None:
    release_path = Path(release_root).resolve()
    experiment_file = Path(experiment_path).resolve()
    output_path = Path(out_dir).resolve()
    contract = load_contract(
        release_root=release_path,
        validity_path=Path(validity_path).resolve(),
        calibration_path=Path(calibration_path).resolve(),
    )
    experiment = json.loads(experiment_file.read_text(encoding="utf-8"))
    validate_experiment(value=experiment, contract=contract)
    selected_tasks = {value for value in task_ids.split(",") if value} or None
    selected_seeds = {int(value) for value in seeds.split(",") if value}
    engine = LagunaEngine()

    def generate(
        messages: list[dict[str, Any]], sampling: dict[str, Any]
    ) -> dict[str, Any]:
        return engine.generate.remote(messages, sampling)

    manifest = sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        output_root=output_path,
        generate=generate,
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        task_ids=selected_tasks,
        seeds=selected_seeds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
