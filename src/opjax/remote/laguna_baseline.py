"""Local orchestration for the frozen Laguna XS 2.1 Pallas baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.phase3_sampling import sample_sglang_matrix
from opjax.pallas.phase31_experiment import load_contract
from opjax.pallas.phase31_conformance import run_two_turn_conformance
from opjax.pallas.phase32_experiment import validate_experiment
from opjax.pallas.sglang_agent import SGLangEndpointModel
from opjax.remote.laguna_sglang import (
    ENDPOINT_URL,
    MODEL_ID,
    MODEL_REVISION,
    PRECISION,
    SGLANG_REVISION,
    app,
)
from opjax.remote.config import modal_proxy_headers


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


def _endpoint() -> tuple[str, dict[str, str]]:
    return ENDPOINT_URL, modal_proxy_headers()


@app.local_entrypoint()
def protocol_canary(
    out_path: str = "data/pallas/runs/phase32-provider-conformance/laguna.json",
) -> None:
    base_url, proxy_headers = _endpoint()
    model = SGLangEndpointModel(
        base_url=base_url,
        api_key="EMPTY",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        seed=0,
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        proxy_headers=proxy_headers,
        chat_template_kwargs={"enable_thinking": True},
    )
    result = run_two_turn_conformance(
        model=model,
        provider="sglang_openai",
        model_identity={
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
            "endpoint": base_url,
            "transport": "openai_chat_completions",
        },
    )
    _write(Path(out_path).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def phase32(
    release_root: str = "data/pallas/benchmarks/jaxbench-phase31",
    validity_path: str = "data/pallas/runs/phase31-oracle-validity/manifest.json",
    calibration_path: str = "data/pallas/runs/phase31-positive-control-calibration/manifest.json",
    experiment_path: str = "data/pallas/runs/phase32-base-capability/experiment.json",
    out_dir: str = "data/pallas/runs/phase32-base-capability/laguna-samples",
    task_ids: str = "",
    seeds: str = "0,1,2",
    max_concurrency: int = 4,
) -> None:
    contract = load_contract(
        release_root=Path(release_root).resolve(),
        validity_path=Path(validity_path).resolve(),
        calibration_path=Path(calibration_path).resolve(),
    )
    experiment_file = Path(experiment_path).resolve()
    experiment = validate_experiment(
        value=json.loads(experiment_file.read_text(encoding="utf-8")),
        contract=contract,
    )
    base_url, proxy_headers = _endpoint()
    manifest = sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        provider="sglang_openai_laguna",
        output_root=Path(out_dir).resolve(),
        base_url=base_url,
        api_key="EMPTY",
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        task_ids={value for value in task_ids.split(",") if value} or None,
        seeds={int(value) for value in seeds.split(",") if value},
        max_concurrency=max_concurrency,
        proxy_headers=proxy_headers,
        chat_template_kwargs={"enable_thinking": True},
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
