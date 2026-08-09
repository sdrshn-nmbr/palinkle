"""Matched S0/S1 multi-turn GRPO training on Tinker for Gate 6."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import tinker
import torch
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import git_revision
from opjax.pallas.g42_harness import canonical_sha256, load_task_package
from opjax.pallas.g6_contracts import G6ContractError, load_g6_config
from opjax.pallas.g6_rollout import RolloutStep, TurnSample, collect_rollout_step
from opjax.pallas.g6_verifier_backend import RemoteTPUPoolVerifier
from opjax.pallas.phase2_contamination import assert_project_training_content_clean


class G6TrainingError(RuntimeError):
    """Gate 6 training failed or its execution receipt is incomplete."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G6TrainingError(f"G6_TRAINING_JSON_OBJECT_REQUIRED: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _tracked_dirty(repo_root: Path) -> bool:
    process = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return process.returncode != 0 or bool(process.stdout.strip())


def _lane(config: Mapping[str, Any], lane_id: str) -> dict[str, Any]:
    matches = [lane for lane in config["lanes"] if lane["lane_id"] == lane_id]
    if len(matches) != 1:
        raise G6TrainingError(f"G6_LANE_UNKNOWN: {lane_id}")
    return matches[0]


def _tasks(task_root: Path, task_ids: Sequence[str]) -> list[Any]:
    manifest = _load(task_root / "manifest.json")
    by_id = {
        package.task_id: package
        for package in (
            load_task_package(task_root / relative) for relative in manifest["tasks"]
        )
    }
    if not set(task_ids) <= set(by_id):
        raise G6TrainingError("G6_TRAINING_TASKS_MISSING")
    return [by_id[task_id] for task_id in task_ids]


def sample_to_datum(sample: TurnSample, *, length_normalizer: float) -> tinker.Datum:
    if length_normalizer <= 0:
        raise G6TrainingError("G6_LENGTH_NORMALIZER_INVALID")
    prompt_target_length = sample.prompt.length - 1
    response_tokens = sample.response_tokens
    train_input = sample.prompt.append(
        tinker.EncodedTextChunk(tokens=response_tokens[:-1])
    )
    target_tokens = [0] * prompt_target_length + response_tokens
    behavior_logprobs = [0.0] * prompt_target_length + sample.behavior_logprobs
    token_advantage = sample.advantage / length_normalizer
    advantages = [0.0] * prompt_target_length + [token_advantage] * (
        train_input.length - prompt_target_length
    )
    if not (
        len(target_tokens)
        == len(behavior_logprobs)
        == len(advantages)
        == train_input.length
    ):
        raise G6TrainingError("G6_DATUM_LENGTH_MISMATCH")
    return tinker.Datum(
        model_input=train_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(behavior_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(advantages)),
        },
    )


def _step_event(
    *,
    step: int,
    rollout: RolloutStep,
    update_metrics: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    samples = [sample for state in rollout.trajectories for sample in state.samples]
    task_variance = {
        task_id: {
            "standard_deviation": batch.standard_deviation,
            "trainable": batch.trainable,
        }
        for task_id, batch in rollout.advantages.items()
    }
    return {
        "schema_version": 1,
        "kind": "pallas_g6_grpo_step",
        "step": step,
        "task_ids": list(rollout.task_ids),
        "counts": {
            "trajectories": len(rollout.trajectories),
            "turn_samples": len(samples),
            "profile_verified": sum(sample.score > 0 for sample in samples),
            "trainable_datums": len(rollout.trainable_samples),
            "trainable_task_groups": sum(batch.trainable for batch in rollout.advantages.values()),
        },
        "reward": {
            "mean_score": sum(sample.score for sample in samples) / len(samples),
            "maximum_score": max(sample.score for sample in samples),
            "minimum_score": min(sample.score for sample in samples),
            "task_variance": task_variance,
        },
        "updates": update_metrics,
        "elapsed_seconds": elapsed_seconds,
    }


def train_lane(
    *,
    config_path: Path,
    task_root: Path,
    s0_manifest_path: Path,
    s1_manifest_path: Path,
    lane_id: str,
    workers: Sequence[str],
    zone: str,
    repo_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if _tracked_dirty(repo_root):
        raise G6TrainingError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if out_dir.exists():
        raise G6TrainingError(f"G6_TRAINING_OUTPUT_EXISTS: {out_dir}")
    config = load_g6_config(
        config_path=config_path,
        task_manifest_path=task_root / "manifest.json",
        s0_manifest_path=s0_manifest_path,
        s1_manifest_path=s1_manifest_path,
    )
    lane = _lane(config, lane_id)
    tasks = _tasks(task_root, config["task_ids"])
    assert_project_training_content_clean(
        [
            {
                "task_id": task.task_id,
                "instruction": (task.root / "instruction.md").read_text(
                    encoding="utf-8"
                ),
            }
            for task in tasks
        ]
    )
    rollout_config = config["rollout"]
    if len(tasks) != rollout_config["tasks_per_step"] * rollout_config["steps"]:
        raise G6TrainingError("G6_TASK_SCHEDULE_INVALID")
    out_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g6_grpo_run",
        "status": "running",
        "experiment_id": config["experiment_id"],
        "lane_id": lane_id,
        "parent_id": lane["parent_id"],
        "parent_run_sha256": lane["parent_run_sha256"],
        "initial_state_path": lane["initial_state_path"],
        "config_sha256": config["config_sha256"],
        "task_release_sha256": config["task_release_sha256"],
        "opjax_revision": git_revision(repo_root),
        "workers": list(workers),
        "zone": zone,
        "completed_steps": 0,
        "total_steps": rollout_config["steps"],
        "checkpoints": [],
    }
    _write(out_dir / "manifest.json", manifest)
    service = tinker.ServiceClient(
        user_metadata={
            "project": "opjax",
            "gate": "G6",
            "lane": lane_id,
            "experiment_id": config["experiment_id"],
        }
    )
    training = service.create_training_client_from_state(
        lane["initial_state_path"], user_metadata={"arm": lane_id}
    )
    info = _json(training.get_info())
    if (
        info.get("is_lora") is not True
        or info.get("lora_rank") != 64
        or info.get("model_data", {}).get("model_name") != config["base_model"]
    ):
        raise G6TrainingError(f"G6_TRAINING_CLIENT_IDENTITY_MISMATCH: {info}")
    manifest["training_client"] = info
    _write(out_dir / "manifest.json", manifest)
    tokenizer = get_tokenizer(config["base_model"])
    recommended = model_info.get_recommended_renderer_name(config["base_model"])
    if recommended != config["renderer"]:
        raise G6TrainingError(
            f"G6_RENDERER_MISMATCH: expected={config['renderer']} observed={recommended}"
        )
    renderer = renderers.get_renderer(
        config["renderer"], tokenizer, model_name=config["base_model"]
    )
    verifier = RemoteTPUPoolVerifier(workers=workers, zone=zone)
    optimizer_config = config["optimizer"]
    adam = tinker.AdamParams(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
        grad_clip_norm=optimizer_config["grad_clip_norm"],
    )
    events_path = out_dir / "events.jsonl"
    try:
        for step in range(1, rollout_config["steps"] + 1):
            started = time.monotonic()
            sampling = training.save_weights_and_get_sampling_client(
                f"{lane_id.lower()}-step-{step:02d}-rollout"
            )
            offset = (step - 1) * rollout_config["tasks_per_step"]
            step_tasks = tasks[offset : offset + rollout_config["tasks_per_step"]]
            rollout = collect_rollout_step(
                step=step,
                tasks=step_tasks,
                sampling=sampling,
                renderer=renderer,
                tokenizer=tokenizer,
                rollout=rollout_config,
                verifier=verifier,
                out_dir=out_dir / "rollouts" / f"step-{step:02d}",
            )
            datums = [
                sample_to_datum(
                    sample,
                    length_normalizer=float(optimizer_config["constant_length_normalizer"]),
                )
                for sample in rollout.trainable_samples
            ]
            update_metrics = []
            if datums:
                for update in range(1, optimizer_config["updates_per_step"] + 1):
                    forward = training.forward_backward(
                        datums, loss_fn=optimizer_config["loss_fn"]
                    )
                    optimizer = training.optim_step(adam)
                    forward_result = forward.result()
                    optimizer_result = optimizer.result()
                    update_metrics.append(
                        {
                            "update": update,
                            "loss_fn_output_type": forward_result.loss_fn_output_type,
                            "forward_metrics": forward_result.metrics,
                            "optimizer_metrics": _json(optimizer_result).get("metrics"),
                        }
                    )
            checkpoint = training.save_state(
                f"step-{step:02d}", ttl_seconds=604800
            ).result()
            event = _step_event(
                step=step,
                rollout=rollout,
                update_metrics=update_metrics,
                elapsed_seconds=time.monotonic() - started,
            )
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            manifest["completed_steps"] = step
            manifest["checkpoints"].append(_json(checkpoint))
            _write(out_dir / "manifest.json", manifest)
            print(
                f"G6_TRAIN lane={lane_id} step={step}/{rollout_config['steps']} "
                f"verified={event['counts']['profile_verified']} "
                f"trainable_groups={event['counts']['trainable_task_groups']} "
                f"updates={len(update_metrics)}",
                flush=True,
            )
        final_state = training.save_state("final", ttl_seconds=None).result()
        sampler_weights = training.save_weights_for_sampler(
            "final", ttl_seconds=None
        ).result()
    finally:
        service.holder.close()
    manifest["status"] = "completed"
    manifest["final_state"] = _json(final_state)
    manifest["sampler_weights"] = _json(sampler_weights)
    manifest["run_sha256"] = canonical_sha256(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g6-train")
    parser.add_argument("--config", type=Path, default=Path("config/pallas/g6-grpo.json"))
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--s0-manifest", type=Path, required=True)
    parser.add_argument("--s1-manifest", type=Path, required=True)
    parser.add_argument("--lane", choices=["R0", "R1"], required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--zone", default="us-west4-a")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = train_lane(
            config_path=args.config,
            task_root=args.task_root,
            s0_manifest_path=args.s0_manifest,
            s1_manifest_path=args.s1_manifest,
            lane_id=args.lane,
            workers=[worker for worker in args.workers.split(",") if worker],
            zone=args.zone,
            repo_root=args.repo_root.resolve(),
            out_dir=args.out_dir,
        )
    except (
        G6ContractError,
        G6TrainingError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"G6_TRAINING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
