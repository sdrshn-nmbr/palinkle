from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinker import ModelInput

from opjax.pallas.g42_harness import load_task_package
from opjax.pallas.g6_rollout import (
    TrajectoryState,
    TurnSample,
    _messages,
    collect_rollout_step,
)


REPO_ROOT = Path(__file__).parents[2]


def test_refinement_context_keeps_current_kernel_and_actionable_feedback(tmp_path: Path) -> None:
    task = load_task_package(
        REPO_ROOT
        / "data/pallas/runs/g42-task-release/tasks/g42-activation-01-reversed-blockspec"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def workload(x):\n    return x\n", encoding="utf-8")
    state = TrajectoryState(task=task, trajectory=0, workspace=workspace)
    state.samples.append(
        TurnSample(
            task_id=task.task_id,
            trajectory=0,
            turn=1,
            prompt=ModelInput.from_ints([1]),
            prompt_messages=[],
            response_tokens=[2],
            behavior_logprobs=[-1.0],
            response_text="",
            stop_reason="stop",
            action=None,
            action_result={"output": "dev check passed"},
            snapshot={},
            feedback="VERIFIER_STAGE tpu_compile: invalid block geometry (7, 129)",
        )
    )
    messages = _messages(state)
    assert len(messages) == 3
    assert "def workload(x)" in messages[-1]["content"]
    assert "invalid block geometry (7, 129)" in messages[-1]["content"]
    assert "dev check passed" in messages[-1]["content"]


def test_deterministic_online_step_preserves_16_by_4_credit_contract(
    tmp_path: Path, monkeypatch
) -> None:
    release = REPO_ROOT / "data/pallas/runs/g42-task-release"
    manifest = json.loads((release / "manifest.json").read_text())
    tasks = [
        load_task_package(release / "tasks" / task_id)
        for task_id in manifest["training_selection"][:8]
    ]

    def fake_sample(**kwargs):
        prompt = ModelInput.from_ints([1, 2, 3])
        sequence = SimpleNamespace(
            tokens=[4, 5], logprobs=[-0.1, -0.2], stop_reason="stop"
        )
        return prompt, [sequence for _ in range(kwargs["count"])]

    def fake_execute(state, sample):
        assert state.workspace.is_absolute()
        snapshots = state.workspace.parent / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        kernel = snapshots / f"turn-{sample.turn}-kernel.py"
        kernel.write_text("def workload(x):\n    return x\n", encoding="utf-8")
        sample.action_result = {"returncode": 0, "output": "ok", "exception_info": ""}
        sample.snapshot = {"kernel_path": kernel.name}

    class FakeRenderer:
        def parse_response(self, tokens):
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": None,
                            "function": {
                                "name": "mswea_bash_command",
                                "arguments": {"command": "true"},
                            },
                        }
                    ],
                },
                SimpleNamespace(value="stop_sequence"),
            )

    class FakeVerifier:
        def __init__(self):
            self.calls = []

        def verify(self, *, candidates, batch_root):
            self.calls.append(len(candidates))
            return {
                candidate.unit_id: (
                    {
                        "passed": True,
                        "stage": "verified",
                        "infrastructure_error": False,
                        "stages": {
                            "artifact_contract": True,
                            "pallas_api": True,
                            "tpu_compile": True,
                            "full_shape_correctness": True,
                            "normal_lowering": True,
                            "runtime_safety": True,
                            "profile": True,
                        },
                        "profile": {
                            "speedup": 1.0,
                            "timing": {
                                "speedup": 1.0,
                                "candidate_median_ms": 0.02,
                                "baseline_median_ms": 0.02,
                            },
                        },
                    }
                    if "trajectory-00" in candidate.unit_id
                    else {
                        "passed": False,
                        "stage": "tpu_compile",
                        "error": "invalid block geometry",
                        "infrastructure_error": False,
                        "stages": {},
                    }
                )
                for candidate in candidates
            }

    monkeypatch.setattr("opjax.pallas.g6_rollout._sample", fake_sample)
    monkeypatch.setattr("opjax.pallas.g6_rollout._execute_action", fake_execute)
    verifier = FakeVerifier()
    result = collect_rollout_step(
        step=1,
        tasks=tasks,
        sampling=None,
        renderer=FakeRenderer(),
        tokenizer=None,
        rollout={
            "parallel_trajectories": 16,
            "refinement_turns": 4,
            "tasks_per_step": 8,
            "discount_gamma": 0.4,
            "correctness_bonus": 0.3,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_response_tokens": 4096,
            "max_context_tokens": 32768,
            "sampling_seed": 0,
            "max_sampling_concurrency": 64,
            "max_action_concurrency": 8,
        },
        verifier=verifier,
        out_dir=tmp_path / "step",
    )
    assert len(result.trajectories) == 128
    assert len(result.trainable_samples) == 512
    assert all(batch.trainable for batch in result.advantages.values())
    assert verifier.calls == [8]
    trajectory = json.loads((tmp_path / "step/trajectory.json").read_text())
    assert trajectory["counts"]["verifier_executions"] == 8
    assert trajectory["counts"]["verifier_cache_hits"] == 504
    successful = next(
        state for state in result.trajectories if state.trajectory == 0
    )
    assert successful.samples[0].raw_return == pytest.approx(2.1112)
