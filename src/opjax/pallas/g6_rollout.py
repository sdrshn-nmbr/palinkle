"""Online multi-turn trajectory collection for Gate 6."""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinker import ModelInput, SamplingClient, types

from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    G42HarnessError,
    TaskPackage,
    create_agent_workspace,
    file_sha256,
    model_message_text,
    parse_model_action,
    snapshot_workspace,
)
from opjax.pallas.g6_contracts import (
    AdvantageBatch,
    discounted_advantages,
    feedback_from_result,
    kernel_score,
)
from opjax.pallas.g6_verifier_backend import VerifierBackend, VerifierCandidate


SYSTEM_PROMPT = """You are optimizing a JAX Pallas kernel in an isolated repository.
Return exactly one shell action using the native mswea_bash_command tool or one fenced
mswea_bash_command fallback block. Inspect or edit
kernel.py and use dev_check.py as needed. The authoritative TPU verifier runs after this
action and returns real compiler, runtime, correctness, and profile feedback. Never use
interpret=True or a plain-JAX fallback. Do not attempt to access hidden tests, reference
solutions, credentials, host paths, or the network.
"""

INITIAL_PROMPT = """Repair the Pallas kernel specified by instruction.md. Start by reading
instruction.md, PALLAS_API.md, kernel.py, and dev_check.py, then take one concrete shell
action toward a complete correct and fast kernel.
"""


class G6RolloutError(RuntimeError):
    """A Gate 6 rollout cannot preserve its online or evidence contract."""


@dataclass
class TurnSample:
    task_id: str
    trajectory: int
    turn: int
    prompt: ModelInput
    prompt_messages: list[dict[str, str]]
    response_tokens: list[int]
    behavior_logprobs: list[float]
    response_text: str
    stop_reason: str
    action: dict[str, str] | None
    action_result: dict[str, Any]
    snapshot: dict[str, Any]
    verifier_result: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    score: float = 0.0
    raw_return: float = 0.0
    advantage: float = 0.0


@dataclass
class TrajectoryState:
    task: TaskPackage
    trajectory: int
    workspace: Path
    samples: list[TurnSample] = field(default_factory=list)


@dataclass
class RolloutStep:
    step: int
    task_ids: tuple[str, ...]
    trajectories: list[TrajectoryState]
    advantages: dict[str, AdvantageBatch]
    trainable_samples: list[TurnSample]


def _messages(state: TrajectoryState) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_PROMPT},
    ]
    if not state.samples:
        return messages
    kernel = state.workspace / "kernel.py"
    source = kernel.read_text(encoding="utf-8") if kernel.is_file() and not kernel.is_symlink() else ""
    history = []
    for previous in state.samples:
        output = str(previous.action_result.get("output", ""))[-2000:]
        history.append(
            f"TURN {previous.turn} ACTION_OUTPUT:\n{output}\n{previous.feedback}"
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Here is the current kernel after the previous action:\n"
                f"```python\n{source}\n```\n\n"
                "Here is the chronological execution and verifier history:\n"
                + "\n\n".join(history)
                + "\n\n"
                "Restart your reasoning from this evidence and take one new concrete shell action."
            ),
        }
    )
    return messages


def _sample(
    *,
    sampling: SamplingClient,
    renderer: Any,
    messages: list[dict[str, str]],
    count: int,
    seed: int,
    rollout: Mapping[str, Any],
) -> tuple[ModelInput, list[Any]]:
    prompt = renderer.build_generation_prompt(messages)
    if prompt.length >= int(rollout["max_context_tokens"]):
        raise G6RolloutError(f"G6_CONTEXT_EXCEEDED: {prompt.length}")
    response = sampling.sample(
        prompt=prompt,
        num_samples=count,
        sampling_params=types.SamplingParams(
            max_tokens=int(rollout["max_response_tokens"]),
            temperature=float(rollout["temperature"]),
            top_p=float(rollout["top_p"]),
            seed=seed,
            stop=renderer.get_stop_sequences() or None,
        ),
    ).result()
    if len(response.sequences) != count:
        raise G6RolloutError(
            f"G6_SAMPLE_COUNT_MISMATCH: expected={count} observed={len(response.sequences)}"
        )
    return prompt, list(response.sequences)


def _turn_sample(
    *,
    state: TrajectoryState,
    turn: int,
    prompt: ModelInput,
    messages: list[dict[str, str]],
    sequence: Any,
    renderer: Any,
) -> TurnSample:
    tokens = list(sequence.tokens)
    logprobs = list(sequence.logprobs or ())
    if not tokens or len(tokens) != len(logprobs):
        raise G6RolloutError(
            f"G6_BEHAVIOR_LOGPROBS_INVALID: {state.task.task_id}:{state.trajectory}:{turn}"
        )
    parsed_message, termination = renderer.parse_response(tokens)
    message = dict(parsed_message)
    text = model_message_text(message)
    action = None
    action_result: dict[str, Any]
    try:
        if getattr(termination, "value", None) == "malformed":
            raise G42HarnessError("TML_RESPONSE_MALFORMED")
        action = parse_model_action(message)
        action_result = {}
    except G42HarnessError as exc:
        action_result = {
            "returncode": 2,
            "output": f"FORMAT_ERROR: {exc}",
            "exception_info": "",
        }
    return TurnSample(
        task_id=state.task.task_id,
        trajectory=state.trajectory,
        turn=turn,
        prompt=prompt,
        prompt_messages=messages,
        response_tokens=tokens,
        behavior_logprobs=logprobs,
        response_text=text,
        stop_reason=str(sequence.stop_reason),
        action=action,
        action_result=action_result,
        snapshot={},
    )


def _execute_action(state: TrajectoryState, sample: TurnSample) -> None:
    if sample.action is not None:
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={state.workspace},dst=/workspace",
            "--workdir",
            "/workspace",
            AGENT_IMAGE,
            "/bin/sh",
            "-lc",
            sample.action["command"],
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            sample.action_result = {
                "returncode": process.returncode,
                "output": (process.stdout + process.stderr)[-10000:],
                "exception_info": "",
            }
        except subprocess.TimeoutExpired as exc:
            sample.action_result = {
                "returncode": 124,
                "output": str(exc.stdout or "")[-10000:],
                "exception_info": "ACTION_TIMEOUT",
            }
    sample.snapshot = snapshot_workspace(
        state.workspace,
        turn=sample.turn,
        output_dir=state.workspace.parent / "snapshots",
    )


def _write_step(step_result: RolloutStep, out_dir: Path) -> None:
    records = []
    for state in step_result.trajectories:
        for sample in state.samples:
            records.append(
                {
                    "task_id": sample.task_id,
                    "trajectory": sample.trajectory,
                    "turn": sample.turn,
                    "prompt_messages": sample.prompt_messages,
                    "prompt_tokens": sample.prompt.to_ints(),
                    "response_tokens": sample.response_tokens,
                    "behavior_logprobs": sample.behavior_logprobs,
                    "response_text": sample.response_text,
                    "stop_reason": sample.stop_reason,
                    "action": sample.action,
                    "action_result": sample.action_result,
                    "snapshot": sample.snapshot,
                    "verifier_result": sample.verifier_result,
                    "feedback": sample.feedback,
                    "score": sample.score,
                    "raw_return": sample.raw_return,
                    "advantage": sample.advantage,
                }
            )
    payload = {
        "schema_version": 1,
        "kind": "pallas_g6_rollout_step",
        "step": step_result.step,
        "task_ids": list(step_result.task_ids),
        "counts": {
            "trajectories": len(step_result.trajectories),
            "turn_samples": len(records),
            "trainable_samples": len(step_result.trainable_samples),
            "verifier_executions": sum(
                record["verifier_result"].get("cache", {}).get("hit") is False
                for record in records
            ),
            "verifier_cache_hits": sum(
                record["verifier_result"].get("cache", {}).get("hit") is True
                for record in records
            ),
        },
        "advantages": {
            task_id: {
                "mean": batch.mean,
                "standard_deviation": batch.standard_deviation,
                "trainable": batch.trainable,
            }
            for task_id, batch in step_result.advantages.items()
        },
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def collect_rollout_step(
    *,
    step: int,
    tasks: Sequence[TaskPackage],
    sampling: SamplingClient,
    renderer: Any,
    tokenizer: Any,
    rollout: Mapping[str, Any],
    verifier: VerifierBackend,
    out_dir: Path,
) -> RolloutStep:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise G6RolloutError(f"G6_ROLLOUT_OUTPUT_EXISTS: {out_dir}")
    group_size = int(rollout["parallel_trajectories"])
    turn_count = int(rollout["refinement_turns"])
    if len(tasks) != int(rollout["tasks_per_step"]):
        raise G6RolloutError("G6_STEP_TASK_COUNT_INVALID")
    states = []
    for task in tasks:
        for trajectory in range(group_size):
            root = out_dir / "workspaces" / task.task_id / f"trajectory-{trajectory:02d}"
            create_agent_workspace(task, root / "workspace")
            states.append(TrajectoryState(task, trajectory, root / "workspace"))
    base_seed = int(rollout["sampling_seed"]) + step * 10_000_000
    verifier_cache: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for turn in range(1, turn_count + 1):
        samples: list[tuple[TrajectoryState, TurnSample]] = []
        if turn == 1:
            for task_index, task in enumerate(tasks):
                task_states = [state for state in states if state.task.task_id == task.task_id]
                messages = _messages(task_states[0])
                prompt, sequences = _sample(
                    sampling=sampling,
                    renderer=renderer,
                    messages=messages,
                    count=group_size,
                    seed=base_seed + task_index * 100_000,
                    rollout=rollout,
                )
                for state, sequence in zip(task_states, sequences, strict=True):
                    samples.append(
                        (
                            state,
                            _turn_sample(
                                state=state,
                                turn=turn,
                                prompt=prompt,
                                messages=messages,
                                sequence=sequence,
                                renderer=renderer,
                            ),
                        )
                    )
        else:
            def sample_state(
                state: TrajectoryState, active_turn: int = turn
            ) -> tuple[TrajectoryState, TurnSample]:
                messages = _messages(state)
                task_index = next(
                    index for index, task in enumerate(tasks) if task.task_id == state.task.task_id
                )
                prompt, sequences = _sample(
                    sampling=sampling,
                    renderer=renderer,
                    messages=messages,
                    count=1,
                    seed=(
                        base_seed
                        + task_index * 100_000
                        + state.trajectory * 1_000
                        + active_turn
                    ),
                    rollout=rollout,
                )
                return state, _turn_sample(
                    state=state,
                    turn=active_turn,
                    prompt=prompt,
                    messages=messages,
                    sequence=sequences[0],
                    renderer=renderer,
                )

            with ThreadPoolExecutor(
                max_workers=int(rollout["max_sampling_concurrency"])
            ) as executor:
                futures = [executor.submit(sample_state, state) for state in states]
                for future in as_completed(futures):
                    samples.append(future.result())
        with ThreadPoolExecutor(
            max_workers=min(int(rollout["max_action_concurrency"]), len(samples))
        ) as executor:
            futures = [executor.submit(_execute_action, state, sample) for state, sample in samples]
            for future in as_completed(futures):
                future.result()
        candidates = []
        by_unit = {}
        key_by_unit = {}
        pending_keys = set()
        for state, sample in samples:
            state.samples.append(sample)
            unit_id = f"{state.task.task_id}--trajectory-{state.trajectory:02d}--turn-{turn}"
            kernel_path = state.workspace.parent / "snapshots" / f"turn-{turn}-kernel.py"
            key = (state.task.task_sha256, file_sha256(kernel_path))
            key_by_unit[unit_id] = key
            if key not in verifier_cache and key not in pending_keys:
                candidate = VerifierCandidate(
                    unit_id=unit_id,
                    task_path=state.task.root / "tests" / "task.json",
                    kernel_path=kernel_path,
                )
                candidates.append(candidate)
                pending_keys.add(key)
            by_unit[unit_id] = sample
        if candidates:
            results = verifier.verify(
                candidates=candidates,
                batch_root=out_dir / "verifier" / f"turn-{turn}",
            )
            for candidate in candidates:
                key = key_by_unit[candidate.unit_id]
                verifier_cache[key] = (results[candidate.unit_id], candidate.unit_id)
        for unit_id, sample in by_unit.items():
            result, source_unit_id = verifier_cache[key_by_unit[unit_id]]
            result = copy.deepcopy(result)
            result["cache"] = {
                "hit": unit_id != source_unit_id,
                "source_unit_id": source_unit_id,
                "task_sha256": key_by_unit[unit_id][0],
                "kernel_sha256": key_by_unit[unit_id][1],
            }
            if result.get("infrastructure_error") is True:
                raise G6RolloutError(f"G6_VERIFIER_INFRASTRUCTURE_FAILURE: {unit_id}")
            sample.verifier_result = result
            sample.feedback = feedback_from_result(result)
            sample.score = kernel_score(
                result, correctness_bonus=float(rollout["correctness_bonus"])
            )
        print(
            f"G6_ROLLOUT step={step} turn={turn}/{turn_count} "
            f"verified={sum(sample.score > 0 for _, sample in samples)}/{len(samples)} "
            f"executed={len(candidates)} cache_hits={len(samples) - len(candidates)}",
            flush=True,
        )
    advantages = {}
    trainable_samples = []
    for task in tasks:
        task_states = sorted(
            (state for state in states if state.task.task_id == task.task_id),
            key=lambda state: state.trajectory,
        )
        batch = discounted_advantages(
            [[sample.score for sample in state.samples] for state in task_states],
            gamma=float(rollout["discount_gamma"]),
        )
        advantages[task.task_id] = batch
        for state, returns, normalized in zip(
            task_states, batch.raw_returns, batch.advantages, strict=True
        ):
            for sample, raw_return, advantage in zip(
                state.samples, returns, normalized, strict=True
            ):
                sample.raw_return = raw_return
                sample.advantage = advantage
                if batch.trainable:
                    trainable_samples.append(sample)
    result = RolloutStep(
        step=step,
        task_ids=tuple(task.task_id for task in tasks),
        trajectories=states,
        advantages=advantages,
        trainable_samples=trainable_samples,
    )
    _write_step(result, out_dir)
    return result
