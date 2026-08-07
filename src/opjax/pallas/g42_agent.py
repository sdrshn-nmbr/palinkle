"""mini-swe-agent driver backed by Tinker sampling for G4.2."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import tinker
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import FormatError, Submitted
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import load_contracts
from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    G42HarnessError,
    create_agent_workspace,
    load_task_package,
    model_message_text,
    parse_model_action,
    snapshot_workspace,
    validate_horizon_contract,
)
from opjax.pallas.sampling import _sampling_client

SYSTEM_TEMPLATE = """You are a programming agent working in an isolated repository.
Your response must contain exactly one shell action. Prefer the native
`mswea_bash_command` tool. If native tools are unavailable, use this fallback:

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

OBSERVATION_TEMPLATE = """<returncode>{{returncode}}</returncode>
<output>{{output}}</output>
{% if exception_info %}<exception>{{exception_info}}</exception>{% endif %}"""

class TinkerMiniSWEModel:
    """mini-swe Model protocol implementation using a pinned Tinker sampler."""

    def __init__(
        self,
        *,
        client: tinker.SamplingClient,
        renderer: Any,
        tokenizer: Any,
        checkpoint: str | None,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.calls = 0
        self.config = {"model_name": "thinkingmachines/Inkling-Small"}
        self.samples: list[dict[str, Any]] = []

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        prompt_messages = [
            {"role": message["role"], "content": str(message.get("content", ""))}
            for message in messages
            if message.get("role") in {"system", "user", "assistant"}
        ]
        prompt = self.renderer.build_generation_prompt(prompt_messages)
        response = self.client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed + self.calls * 1_000_000,
                stop=self.renderer.get_stop_sequences() or None,
            ),
        ).result()
        sequence = response.sequences[0]
        parsed_message, termination = self.renderer.parse_response(sequence.tokens)
        content = model_message_text(dict(parsed_message))
        self.calls += 1
        sample = {
            "call": self.calls,
            "seed": self.seed + (self.calls - 1) * 1_000_000,
            "completion_tokens": len(sequence.tokens),
            "stop_reason": str(sequence.stop_reason),
            "checkpoint": self.checkpoint,
            "content": content,
            "parse_termination": termination.value,
        }
        self.samples.append(sample)
        try:
            if termination is ParseTermination.MALFORMED:
                raise G42HarnessError("TML_RESPONSE_MALFORMED")
            action = parse_model_action(dict(parsed_message))
        except G42HarnessError as exc:
            raise FormatError(
                {"role": "assistant", "content": content, "extra": sample},
                {
                    "role": "user",
                    "content": f"Format error: {exc}. Return exactly one mswea_bash_command block.",
                    "extra": {"interrupt_type": "FormatError"},
                },
            ) from exc
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": parsed_message.get("tool_calls", []),
            "extra": {**sample, "actions": [action]},
        }

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
        return {"model_name": self.config["model_name"], **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "model": {
                    "model_name": self.config["model_name"],
                    "checkpoint": self.checkpoint,
                    "seed": self.seed,
                    "sampling": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_tokens,
                    },
                }
            },
            "samples": self.samples,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _close_service_holder(service: tinker.ServiceClient) -> None:
    service.holder.close()


async def run_tinker_agent(
    *,
    config_root: Path,
    task_dir: Path,
    output_dir: Path,
    checkpoint: str | None,
    seed: int,
    turn_limit: int = 6,
    snapshot_turns: tuple[int, ...] = (3, 6),
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise G42HarnessError(f"RUN_OUTPUT_EXISTS: {output_dir}")
    validate_horizon_contract(turn_limit=turn_limit, snapshot_turns=snapshot_turns)
    task = load_task_package(task_dir)
    bundle = load_contracts(config_root)
    output_dir.mkdir(parents=True)
    workspace = output_dir / "workspace"
    workspace_record = create_agent_workspace(task, workspace)
    renderer_name = model_info.get_recommended_renderer_name(bundle.experiment["base_model"])
    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=bundle.experiment["base_model"])
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
    service = tinker.ServiceClient(http_client=http_client, max_retries=0)
    sampling_client = await _sampling_client(service=service, base_model=bundle.experiment["base_model"], model_path=checkpoint)
    sampling = bundle.experiment["sampling"]
    model = TinkerMiniSWEModel(
        client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        seed=seed,
        max_tokens=sampling["max_tokens"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
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
            "kind": "pallas_g42_agent_run",
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "checkpoint": checkpoint,
            "seed": seed,
            "turn_limit": turn_limit,
            "snapshot_turns": list(snapshot_turns),
            "agent_image": AGENT_IMAGE,
            "submitted": submitted,
            "snapshots": snapshots,
            "trajectory_path": "trajectory.json",
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        try:
            environment.cleanup()
        finally:
            try:
                _close_service_holder(service)
            finally:
                await http_client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-agent")
    parser.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_tinker_agent(**vars(args)))
    except (G42HarnessError, ValueError, OSError) as exc:
        print(f"G42_AGENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
