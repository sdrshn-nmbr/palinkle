from __future__ import annotations

import base64
import shlex

import pytest

from opjax.pallas.agent_protocol import (
    AgentProtocolError,
    normalize_native_action,
    parse_poolside_action,
    parse_tinker_action,
)


def _decode_command_values(command: str) -> list[str]:
    parts = shlex.split(command)
    return [base64.b64decode(value).decode() for value in parts[3:]]


@pytest.mark.parametrize("tool", ["bash", "shell", "mswea_bash_command"])
def test_shell_aliases_share_one_canonical_action(tool: str) -> None:
    assert normalize_native_action(tool, {"command": "python dev_check.py"}) == {
        "command": "python dev_check.py"
    }


def test_tinker_native_shell_and_fenced_actions_are_conformant() -> None:
    native = {
        "tool_calls": [
            {"function": {"name": "shell", "arguments": '{"command":"ls"}'}}
        ]
    }
    assert parse_tinker_action(native, "") == {"command": "ls"}
    assert parse_tinker_action(
        {}, "```mswea_bash_command\nls\n```"
    ) == {"command": "ls"}


def test_poolside_write_preserves_complete_multiline_content() -> None:
    content = (
        "<tool_call>write"
        "<arg_key>path</arg_key><arg_value>kernel.py</arg_value>"
        "<arg_key>content</arg_key><arg_value>def workload(x):\n    return x\n"
        "</arg_value></tool_call>"
    )
    action = parse_poolside_action(content)
    assert _decode_command_values(action["command"]) == [
        "kernel.py",
        "def workload(x):\n    return x\n",
    ]


def test_edit_requires_one_exact_match_and_preserves_values() -> None:
    action = normalize_native_action(
        "edit",
        {
            "path": "kernel.py",
            "old_string": "return ...",
            "new_string": "return output",
        },
    )
    assert _decode_command_values(action["command"]) == [
        "kernel.py",
        "return ...",
        "return output",
    ]


def test_native_actions_reject_workspace_escape_and_mixed_protocol() -> None:
    with pytest.raises(AgentProtocolError, match="OUTSIDE_WORKSPACE"):
        normalize_native_action("read", {"path": "../tests/hidden.py"})
    with pytest.raises(AgentProtocolError, match="ACTION_PROTOCOL_MIXED"):
        parse_poolside_action(
            "```mswea_bash_command\nls\n```"
            "<tool_call>shell<arg_key>command</arg_key>"
            "<arg_value>pwd</arg_value></tool_call>"
        )
