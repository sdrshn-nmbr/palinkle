"""DeepSWE-style agent loop for the frozen full JAXBench task packages."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import FormatError, Submitted

from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    G42HarnessError,
    canonical_sha256,
    file_sha256,
    snapshot_workspace,
    tree_sha256,
    validate_horizon_contract,
)
from opjax.pallas.jaxbench_capability import materialize_agent_workspace

SYSTEM_TEMPLATE = """You are a programming agent working in an isolated repository.
Use exactly one native shell, bash, read, write, edit, or list tool call per turn. If
native tools are unavailable, return exactly one fenced `mswea_bash_command` action.
Inspect the task, implement kernel.py, run public checks, and submit only when ready
by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not combine submission with another command. Each action runs in a new shell.
The hidden verifier, inputs, tests, and optimized reference are unavailable.
"""

INSTANCE_TEMPLATE = """Implement the TPU Pallas kernel described in instruction.md.
Start by reading instruction.md, PALLAS_API.md, kernel.py, and dev_check.py.
"""


class MiniSWEModel(Protocol):
    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    def format_message(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class JaxBenchAgentTask:
    root: Path
    task_id: str
    task_sha256: str
    baseline_sha256: str
    optimized_sha256: str | None


def load_agent_task(*, release_root: Path, task_id: str) -> JaxBenchAgentTask:
    release_root = release_root.resolve()
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    task = next(
        (record for record in manifest.get("tasks", []) if record.get("task_id") == task_id),
        None,
    )
    if task is None:
        raise G42HarnessError(f"JAXBENCH_AGENT_TASK_UNKNOWN:{task_id}")
    root = release_root / task["path"]
    if not root.is_dir():
        raise G42HarnessError(f"JAXBENCH_AGENT_TASK_MISSING:{task_id}")
    return JaxBenchAgentTask(
        root=root,
        task_id=task_id,
        task_sha256=task["task_sha256"],
        baseline_sha256=task["baseline_sha256"],
        optimized_sha256=task.get("optimized_sha256"),
    )


def initialize_agent_workspace(
    *, task: JaxBenchAgentTask, destination: Path
) -> dict[str, Any]:
    public = materialize_agent_workspace(task_root=task.root, destination=destination)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "opjax-harness",
        "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
        "GIT_COMMITTER_NAME": "opjax-harness",
        "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
    }
    subprocess.run(["git", "init", "-q", str(destination)], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(destination), "add", "."], check=True, env=environment
    )
    subprocess.run(
        ["git", "-C", str(destination), "commit", "-q", "-m", "task base"],
        check=True,
        env=environment,
    )
    base_commit = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    return {
        **public,
        "base_commit": base_commit,
        "workspace_sha256": tree_sha256(destination, excluded={".git"}),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_jaxbench_agent(
    *,
    task: JaxBenchAgentTask,
    output_dir: Path,
    model: MiniSWEModel,
    model_identity: dict[str, Any],
    seed: int,
    turn_limit: int = 6,
    snapshot_turns: tuple[int, ...] = (3, 6),
    agent_image: str = AGENT_IMAGE,
    experiment_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise G42HarnessError(f"RUN_OUTPUT_EXISTS:{output_dir}")
    validate_horizon_contract(turn_limit=turn_limit, snapshot_turns=snapshot_turns)
    output_dir.mkdir(parents=True)
    workspace = output_dir / "workspace"
    workspace_record = initialize_agent_workspace(task=task, destination=workspace)
    environment = DockerEnvironment(
        image=agent_image,
        cwd="/workspace",
        timeout=120,
        run_args=[
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
        ],
    )
    agent = DefaultAgent(
        model,
        environment,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        step_limit=0,
        cost_limit=0,
        wall_time_limit_seconds=0,
        max_consecutive_format_errors=0,
    )
    agent.extra_template_vars = {
        "task": (task.root / "instruction.md").read_text(encoding="utf-8")
    }
    agent.add_messages(
        model.format_message(role="system", content=SYSTEM_TEMPLATE),
        model.format_message(role="user", content=INSTANCE_TEMPLATE),
    )
    snapshots: dict[int, dict[str, Any]] = {}
    submitted = False
    try:
        for turn in range(1, turn_limit + 1):
            if not submitted:
                try:
                    agent.step()
                except FormatError as exc:
                    agent.add_messages(*exc.messages)
                except Submitted as exc:
                    agent.add_messages(*exc.messages)
                    submitted = True
            record = snapshot_workspace(
                workspace,
                turn=turn,
                output_dir=output_dir / "snapshots",
            )
            if turn in snapshot_turns:
                snapshots[turn] = record
        trajectory = agent.serialize(
            {
                "phase3": {
                    "schema_version": 1,
                    "task_id": task.task_id,
                    "task_sha256": task.task_sha256,
                    "seed": seed,
                    "turn_limit": turn_limit,
                    "snapshot_turns": list(snapshot_turns),
                    "agent_image": agent_image,
                    "experiment_identity": experiment_identity,
                    "submitted": submitted,
                    "workspace": workspace_record,
                    "snapshots": snapshots,
                    "model": model_identity,
                }
            }
        )
        trajectory_path = output_dir / "trajectory.json"
        _write_json(trajectory_path, trajectory)
        manifest = {
            "schema_version": 2 if experiment_identity else 1,
            "kind": "opjax_phase3_jaxbench_agent_run",
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "baseline_sha256": task.baseline_sha256,
            "model": model_identity,
            "seed": seed,
            "turn_limit": turn_limit,
            "snapshot_turns": list(snapshot_turns),
            "agent_image": agent_image,
            "experiment_identity": experiment_identity,
            "prompt_contract_sha256": canonical_sha256(
                {"system": SYSTEM_TEMPLATE, "instance": INSTANCE_TEMPLATE}
            ),
            "submitted": submitted,
            "snapshots": snapshots,
            "trajectory_sha256": file_sha256(trajectory_path),
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        environment.cleanup()
