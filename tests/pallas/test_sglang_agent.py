from __future__ import annotations

import json

import pytest
from minisweagent.exceptions import FormatError

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.phase31_conformance import run_two_turn_conformance
from opjax.pallas.sglang_agent import SGLangMiniSWEModel, parse_sglang_action
from opjax.remote.laguna_baseline import summarize_baseline
from opjax.remote.laguna_sglang import render_laguna_prompt


def _model(generate):
    return SGLangMiniSWEModel(
        generate=generate,
        model_id="poolside/Laguna-XS-2.1",
        model_revision="model-revision",
        runtime_revision="runtime-revision",
        precision="bfloat16",
        seed=7,
        max_tokens=1024,
        temperature=0.2,
        top_p=0.95,
    )


def test_two_turn_conformance_requires_linked_native_observation() -> None:
    responses = iter(
        [
            "<tool_call>shell<arg_key>command</arg_key>"
            "<arg_value>printf PROTOCOL_ONE</arg_value></tool_call>",
            "<tool_call>shell<arg_key>command</arg_key>"
            "<arg_value>printf PROTOCOL_TWO</arg_value></tool_call>",
        ]
    )
    model = _model(lambda messages, sampling: {"text": next(responses)})

    result = run_two_turn_conformance(
        model=model,
        provider="sglang",
        model_identity={"model_id": "test", "model_revision": "revision"},
    )

    assert result["passed"] is True
    assert result["model_calls"] == 2
    assert result["messages"][1]["role"] == "tool"
    assert result["messages"][1]["tool_call_id"] == "call-1-1"


def test_sglang_model_records_identity_sampling_and_action() -> None:
    calls = []

    def generate(messages, sampling):
        calls.append((messages, sampling))
        return {
            "text": "```mswea_bash_command\npython dev_check.py\n```",
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "stop_reason": "stop",
            "latency_seconds": 0.25,
        }

    model = _model(generate)
    result = model.query(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    )

    assert result["extra"]["actions"] == [{"command": "python dev_check.py"}]
    assert calls[0][1] == {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "sampling_seed": 7,
    }
    assert model.serialize()["info"]["model"]["provider"] == "sglang"
    assert model.samples[0]["model_revision"] == "model-revision"


def test_sglang_model_preserves_native_tool_history_and_linked_result() -> None:
    calls = []

    def generate(messages, sampling):
        calls.append((messages, sampling))
        return {
            "text": (
                "Inspecting the task.</think><tool_call>read"
                "<arg_key>path</arg_key><arg_value>instruction.md</arg_value>"
                "</tool_call>"
            )
        }

    model = _model(generate)
    message = model.query([{"role": "user", "content": "task"}])

    assert message["content"] == ""
    assert message["reasoning_content"] == "Inspecting the task."
    assert message["tool_calls"] == [
        {
            "type": "function",
            "id": "call-1-1",
            "function": {
                "name": "read",
                "arguments": {"path": "instruction.md"},
            },
        }
    ]

    observation = model.format_observation_messages(
        message,
        [{"returncode": 0, "output": "task contents", "exception_info": None}],
    )[0]
    assert observation == {
        "role": "tool",
        "content": "<returncode>0</returncode>\n<output>task contents</output>",
        "tool_call_id": "call-1-1",
        "name": "read",
        "extra": {
            "returncode": 0,
            "output": "task contents",
            "exception_info": None,
        },
    }

    model.query(
        [
            {"role": "user", "content": "task"},
            message,
            observation,
        ]
    )

    assert calls[1][0] == [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Inspecting the task.",
            "tool_calls": message["tool_calls"],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode>\n<output>task contents</output>",
            "tool_call_id": "call-1-1",
            "name": "read",
        },
    ]


def test_sglang_fenced_action_receives_user_observation() -> None:
    model = _model(
        lambda messages, sampling: {"text": "```mswea_bash_command\npwd\n```"}
    )
    message = model.query([{"role": "user", "content": "task"}])

    observation = model.format_observation_messages(
        message,
        [{"returncode": 0, "output": "/workspace", "exception_info": None}],
    )[0]

    assert observation["role"] == "user"
    assert "tool_call_id" not in observation
    assert "name" not in observation


def test_laguna_prompt_declares_tools_and_preserves_native_transcript() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "checking",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-1-1",
                    "function": {
                        "name": "read",
                        "arguments": {"path": "instruction.md"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode>\n<output>task</output>",
            "tool_call_id": "call-1-1",
            "name": "read",
        },
    ]

    assert render_laguna_prompt(tokenizer, messages) == "rendered"
    assert tokenizer.messages == messages
    assert tokenizer.kwargs["tokenize"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is True
    assert {tool["function"]["name"] for tool in tokenizer.kwargs["tools"]} == {
        "shell",
        "read",
        "write",
        "edit",
        "list",
    }


def test_sglang_model_counts_format_failure_as_call() -> None:
    model = _model(lambda messages, sampling: {"text": "not an action"})

    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "task"}])

    assert model.calls == 1
    assert model.samples[0]["content"] == "not an action"


def test_poolside_native_action_is_normalized() -> None:
    content = (
        "Inspecting.</think><tool_call>mswea_bash_command"
        "<arg_key>command</arg_key><arg_value>sed -n '1,80p' instruction.md &amp;&amp; "
        "python dev_check.py</arg_value></tool_call>"
    )

    assert parse_sglang_action(content) == {
        "command": "sed -n '1,80p' instruction.md && python dev_check.py"
    }


@pytest.mark.parametrize(
    ("tool", "key", "value", "command"),
    [
        ("shell", "command", "cat instruction.md", "cat instruction.md"),
        ("shell", "cmd", "python dev_check.py", "python dev_check.py"),
        ("read", "path", "instruction.md", "sed -n '1,240p' -- instruction.md"),
        ("read", "path", "file with spaces", "sed -n '1,240p' -- 'file with spaces'"),
    ],
)
def test_poolside_native_driver_tools_are_normalized(
    tool: str, key: str, value: str, command: str
) -> None:
    content = (
        f"Reasoning.</think><tool_call>{tool}<arg_key>{key}</arg_key>"
        f"<arg_value>{value}</arg_value></tool_call>"
    )

    assert parse_sglang_action(content) == {"command": command}


def test_poolside_write_requires_content() -> None:
    content = (
        "<tool_call>write<arg_key>path</arg_key>"
        "<arg_value>kernel.py</arg_value></tool_call>"
    )

    with pytest.raises(G42HarnessError, match="ACTION_ARGUMENTS_INVALID:content"):
        parse_sglang_action(content)


def test_mixed_action_protocol_is_rejected() -> None:
    content = (
        "```mswea_bash_command\nls\n```"
        "<tool_call>mswea_bash_command<arg_key>command</arg_key>"
        "<arg_value>pwd</arg_value></tool_call>"
    )

    with pytest.raises(G42HarnessError, match="ACTION_PROTOCOL_MIXED"):
        parse_sglang_action(content)


def test_laguna_baseline_summary_validates_authoritative_evidence(tmp_path) -> None:
    unit_id = "laguna--task--seed-0--turn-3"
    result_dir = tmp_path / "results" / unit_id
    result_dir.mkdir(parents=True)
    reward_path = result_dir / "reward.json"
    reward_path.write_text(
        json.dumps({"reward": 0, "failure_stage": "artifact_contract"}) + "\n"
    )
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "g42": {"submitted": False},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "inspect",
                        "extra": {"actions": [{"command": "cat kernel.py"}]},
                    }
                ],
            }
        )
        + "\n"
    )
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "unit_id": unit_id,
                "task_id": "task",
                "family": "elementwise_binary",
                "turn": 3,
                "trajectory_sha256": file_sha256(trajectory_path),
                "patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        ],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    verification = {
        "input_release_sha256": manifest["release_sha256"],
        "counts": {"infrastructure_failures": 0},
        "records": [
            {
                "unit_id": unit_id,
                "reward": 0,
                "artifacts": {"reward.json": file_sha256(reward_path)},
            }
        ],
    }
    verification["release_sha256"] = canonical_sha256(verification)
    (tmp_path / "verification.json").write_text(json.dumps(verification) + "\n")
    unit_root = tmp_path / "units" / unit_id
    unit_root.mkdir(parents=True)
    (unit_root / "trajectory.json").write_bytes(trajectory_path.read_bytes())

    output_path = tmp_path / "summary.json"
    result = summarize_baseline(verifier_root=tmp_path, out_path=output_path)

    assert result["counts"] == {
        "tasks": 1,
        "units": 1,
        "profile_verified": 0,
        "candidate_failures": 1,
        "infrastructure_failures": 0,
        "nonempty_patches": 0,
    }
    assert result["horizons"] == {
        "k3": {
            "units": 1,
            "profile_verified": 0,
            "candidate_failures": 1,
            "infrastructure_failures": 0,
            "nonempty_patches": 0,
        }
    }
    assert result["turn_3_to_6_transitions"] is None
    assert result["agent_behavior"] == {
        "trajectories": 1,
        "model_calls": 1,
        "format_errors": 0,
        "submitted": 0,
        "commands_by_call": {"1": {"cat kernel.py": 1}},
    }
    assert result["failure_stages"] == {"artifact_contract": 1}
    assert (
        json.loads(output_path.read_text())["result_sha256"] == result["result_sha256"]
    )
