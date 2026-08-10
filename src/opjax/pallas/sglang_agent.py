"""Provider-neutral mini-swe-agent driver backed by an SGLang generator."""

from __future__ import annotations

import json
import re
import shlex
from html import unescape
from pathlib import Path
from typing import Any, Callable

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import FormatError, Submitted

from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    G42HarnessError,
    create_agent_workspace,
    load_task_package,
    parse_action,
    snapshot_workspace,
    validate_horizon_contract,
)

SYSTEM_TEMPLATE = """You are a programming agent working in an isolated repository.
Your response must contain exactly one shell action in this format:

```mswea_bash_command
command
```

Inspect the task, edit kernel.py, run public checks, and submit only when ready by
running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Do not combine submission
with another command. Each action runs in a new shell. The hidden verifier and
reference solution are not available in this environment.
"""

INSTANCE_TEMPLATE = """Repair the Pallas kernel described in instruction.md.
Start by reading instruction.md, PALLAS_API.md, kernel.py, and dev_check.py.
"""

GenerationFunction = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]
POOLSIDE_ACTION_PATTERN = re.compile(
    r"<tool_call>\s*([^<\s]+)\s*"
    r"<arg_key>\s*([^<]+?)\s*</arg_key>\s*"
    r"<arg_value>(.*?)</arg_value>\s*</tool_call>",
    re.DOTALL,
)


def parse_sglang_action(content: str) -> dict[str, str]:
    """Normalize exactly one fenced or Poolside-native shell action."""
    fenced_error: G42HarnessError | None = None
    try:
        fenced = parse_action(content)
    except G42HarnessError as exc:
        fenced_error = exc
    else:
        if POOLSIDE_ACTION_PATTERN.search(content):
            raise G42HarnessError("ACTION_PROTOCOL_MIXED")
        return fenced
    native_actions = POOLSIDE_ACTION_PATTERN.findall(content)
    if len(native_actions) != 1:
        raise fenced_error
    tool, key, raw_value = native_actions[0]
    value = unescape(raw_value).strip()
    if not value:
        raise G42HarnessError("ACTION_EMPTY: command")
    if tool in {"mswea_bash_command", "shell"} and key in {"command", "cmd"}:
        return {"command": value}
    if tool == "read" and key == "path":
        return {"command": f"cat -- {shlex.quote(value)}"}
    raise G42HarnessError(f"ACTION_NATIVE_TOOL_UNSUPPORTED:{tool}:{key}")


class SGLangMiniSWEModel:
    """mini-swe Model protocol implementation around a typed generation function."""

    def __init__(
        self,
        *,
        generate: GenerationFunction,
        model_id: str,
        model_revision: str,
        runtime_revision: str,
        precision: str,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
    ) -> None:
        self.generate = generate
        self.model_id = model_id
        self.model_revision = model_revision
        self.runtime_revision = runtime_revision
        self.precision = precision
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.calls = 0
        self.config = {"model_name": model_id}
        self.samples: list[dict[str, Any]] = []

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        prompt_messages = [
            {"role": message["role"], "content": str(message.get("content", ""))}
            for message in messages
            if message.get("role") in {"system", "user", "assistant"}
        ]
        call_seed = self.seed + self.calls * 1_000_000
        sampling = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sampling_seed": call_seed,
        }
        if self.top_k is not None:
            sampling["top_k"] = self.top_k
        response = self.generate(prompt_messages, sampling)
        content = str(response["text"])
        self.calls += 1
        sample = {
            "call": self.calls,
            "seed": call_seed,
            "completion_tokens": response.get("completion_tokens"),
            "prompt_tokens": response.get("prompt_tokens"),
            "stop_reason": response.get("stop_reason"),
            "latency_seconds": response.get("latency_seconds"),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "runtime_revision": self.runtime_revision,
            "precision": self.precision,
            "content": content,
        }
        self.samples.append(sample)
        try:
            action = parse_sglang_action(content)
        except G42HarnessError as exc:
            raise FormatError(
                {"role": "assistant", "content": content, "extra": sample},
                {
                    "role": "user",
                    "content": f"Format error: {exc}. Return exactly one mswea_bash_command block.",
                    "extra": {"interrupt_type": "FormatError"},
                },
            ) from exc
        return {"role": "assistant", "content": content, "extra": {**sample, "actions": [action]}}

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        observations = []
        for output in outputs:
            content = (
                f"<returncode>{output.get('returncode')}</returncode>\n"
                f"<output>{str(output.get('output', ''))[:10000]}</output>"
            )
            if output.get("exception_info"):
                content += f"\n<exception>{output['exception_info']}</exception>"
            observations.append({"role": "user", "content": content, "extra": output})
        return observations

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"model_name": self.model_id, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "model": {
                    "provider": "sglang",
                    "model_name": self.model_id,
                    "model_revision": self.model_revision,
                    "runtime_revision": self.runtime_revision,
                    "precision": self.precision,
                    "seed": self.seed,
                    "sampling": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "max_tokens": self.max_tokens,
                    },
                }
            },
            "samples": self.samples,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sglang_agent(
    *,
    task_dir: Path,
    output_dir: Path,
    generate: GenerationFunction,
    model_id: str,
    model_revision: str,
    runtime_revision: str,
    precision: str,
    seed: int,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int | None = None,
    turn_limit: int = 3,
    snapshot_turns: tuple[int, ...] = (3,),
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise G42HarnessError(f"RUN_OUTPUT_EXISTS: {output_dir}")
    validate_horizon_contract(turn_limit=turn_limit, snapshot_turns=snapshot_turns)
    task = load_task_package(task_dir)
    output_dir.mkdir(parents=True)
    workspace = output_dir / "workspace"
    workspace_record = create_agent_workspace(task, workspace)
    model = SGLangMiniSWEModel(
        generate=generate,
        model_id=model_id,
        model_revision=model_revision,
        runtime_revision=runtime_revision,
        precision=precision,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    environment = DockerEnvironment(
        image=AGENT_IMAGE,
        cwd="/workspace",
        timeout=120,
        run_args=["--rm", "--network", "none", "--mount", f"type=bind,src={workspace},dst=/workspace"],
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
    agent.extra_template_vars = {"task": (task.root / "instruction.md").read_text(encoding="utf-8")}
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
            record = snapshot_workspace(workspace, turn=turn, output_dir=output_dir / "snapshots")
            if turn in snapshot_turns:
                snapshots[turn] = record
        trajectory = agent.serialize(
            {
                "g42": {
                    "schema_version": 1,
                    "task_id": task.task_id,
                    "task_sha256": task.task_sha256,
                    "mode": task.mode,
                    "turn_limit": turn_limit,
                    "snapshot_turns": list(snapshot_turns),
                    "agent_image": AGENT_IMAGE,
                    "submitted": submitted,
                    "workspace": workspace_record,
                    "snapshots": snapshots,
                }
            }
        )
        _write_json(output_dir / "trajectory.json", trajectory)
        manifest = {
            "schema_version": 1,
            "kind": "pallas_sglang_agent_run",
            "provider": "sglang",
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "model_id": model_id,
            "model_revision": model_revision,
            "runtime_revision": runtime_revision,
            "precision": precision,
            "seed": seed,
            "turn_limit": turn_limit,
            "snapshot_turns": list(snapshot_turns),
            "sampling": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            "agent_image": AGENT_IMAGE,
            "submitted": submitted,
            "snapshots": snapshots,
            "trajectory_path": "trajectory.json",
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        environment.cleanup()
