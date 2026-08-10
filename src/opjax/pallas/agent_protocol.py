"""Canonical action protocol shared by every Pallas agent provider."""

from __future__ import annotations

import base64
import json
import re
import shlex
from html import unescape
from pathlib import PurePosixPath
from typing import Any

ACTION_PATTERN = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
NATIVE_ACTION_PATTERN = re.compile(
    r"<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>", re.DOTALL
)
NATIVE_ARGUMENT_PATTERN = re.compile(
    r"<arg_key>\s*([^<]+?)\s*</arg_key>\s*"
    r"<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
SHELL_TOOLS = {"bash", "mswea_bash_command", "shell"}
AGENT_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run one shell command in the isolated task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file in the isolated task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a UTF-8 text file in the isolated task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace one exact text occurrence in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List a directory in the isolated task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]


class AgentProtocolError(RuntimeError):
    pass


def _fenced_action(content: str) -> dict[str, str]:
    actions = [match.strip() for match in ACTION_PATTERN.findall(content)]
    if len(actions) != 1:
        raise AgentProtocolError(
            f"ACTION_COUNT_INVALID:expected=1 observed={len(actions)}"
        )
    if not actions[0]:
        raise AgentProtocolError("ACTION_EMPTY:command")
    return {"command": actions[0]}


def _workspace_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentProtocolError("ACTION_PATH_INVALID")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise AgentProtocolError("ACTION_PATH_OUTSIDE_WORKSPACE")
    return path.as_posix()


def _required_text(arguments: dict[str, Any], *names: str) -> str:
    for name in names:
        value = arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise AgentProtocolError(f"ACTION_ARGUMENTS_INVALID:{'/'.join(names)}")


def _python_command(script: str, *values: str) -> str:
    encoded = [base64.b64encode(value.encode()).decode() for value in values]
    arguments = " ".join(shlex.quote(value) for value in encoded)
    return f"python -c {shlex.quote(script)} {arguments}"


def normalize_native_action(tool: str, arguments: dict[str, Any]) -> dict[str, str]:
    """Translate one provider-native tool call to mini-swe's shell action."""
    if tool in SHELL_TOOLS:
        return {"command": _required_text(arguments, "command", "cmd").strip()}
    if tool == "read":
        path = _workspace_path(_required_text(arguments, "path", "file"))
        return {"command": f"sed -n '1,240p' -- {shlex.quote(path)}"}
    if tool in {"list", "ls"}:
        raw_path = arguments.get("path", ".")
        path = _workspace_path(raw_path)
        return {"command": f"ls -la -- {shlex.quote(path)}"}
    if tool == "write":
        path = _workspace_path(_required_text(arguments, "path", "file"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise AgentProtocolError("ACTION_ARGUMENTS_INVALID:content")
        script = (
            "import base64,pathlib,sys;"
            "p=pathlib.Path(base64.b64decode(sys.argv[1]).decode());"
            "p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_bytes(base64.b64decode(sys.argv[2]))"
        )
        return {"command": _python_command(script, path, content)}
    if tool == "edit":
        path = _workspace_path(_required_text(arguments, "path", "file"))
        old = _required_text(arguments, "old_text", "old_string", "old")
        new = arguments.get(
            "new_text", arguments.get("new_string", arguments.get("new"))
        )
        if not isinstance(new, str):
            raise AgentProtocolError("ACTION_ARGUMENTS_INVALID:new_text")
        script = (
            "import base64,pathlib,sys;"
            "p=pathlib.Path(base64.b64decode(sys.argv[1]).decode());"
            "old=base64.b64decode(sys.argv[2]);new=base64.b64decode(sys.argv[3]);"
            "data=p.read_bytes();"
            "(_ for _ in ()).throw(SystemExit('EDIT_MATCH_COUNT_INVALID')) "
            "if data.count(old)!=1 else p.write_bytes(data.replace(old,new,1))"
        )
        return {"command": _python_command(script, path, old, new)}
    raise AgentProtocolError(f"ACTION_NATIVE_TOOL_UNSUPPORTED:{tool}")


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AgentProtocolError("ACTION_ARGUMENTS_INVALID:malformed JSON") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError("ACTION_ARGUMENTS_INVALID:object")
    return value


def parse_tinker_action(message: dict[str, Any], text: str) -> dict[str, str]:
    native = []
    for raw in message.get("tool_calls", ()) or ():
        value = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
        if not isinstance(value, dict) or not isinstance(value.get("function"), dict):
            raise AgentProtocolError("ACTION_TOOL_INVALID:function")
        function = value["function"]
        native.append(
            normalize_native_action(
                str(function.get("name", "")), _arguments(function.get("arguments"))
            )
        )
    fenced = []
    try:
        fenced.append(_fenced_action(text))
    except AgentProtocolError:
        pass
    observed = native + fenced
    if len(observed) != 1:
        raise AgentProtocolError(
            f"ACTION_COUNT_INVALID:expected=1 observed={len(observed)}"
        )
    return observed[0]


def parse_poolside_action(content: str) -> dict[str, str]:
    native = []
    for tool, body in NATIVE_ACTION_PATTERN.findall(content):
        arguments = {
            key.strip(): unescape(value)
            for key, value in NATIVE_ARGUMENT_PATTERN.findall(body)
        }
        native.append(normalize_native_action(tool.strip(), arguments))
    fenced = []
    try:
        fenced.append(_fenced_action(content))
    except AgentProtocolError:
        pass
    if native and fenced:
        raise AgentProtocolError("ACTION_PROTOCOL_MIXED")
    observed = native + fenced
    if len(observed) != 1:
        raise AgentProtocolError(
            f"ACTION_COUNT_INVALID:expected=1 observed={len(observed)}"
        )
    return observed[0]


def structure_poolside_response(content: str, *, call_id_prefix: str) -> dict[str, Any]:
    """Convert Poolside's text protocol back into its native chat fields."""
    reasoning_content = ""
    response_content = content
    if "</think>" in response_content:
        reasoning_content, response_content = response_content.split("</think>", 1)

    matches = list(NATIVE_ACTION_PATTERN.finditer(response_content))
    visible_parts = []
    cursor = 0
    tool_calls = []
    for index, match in enumerate(matches, start=1):
        visible_parts.append(response_content[cursor : match.start()])
        tool, body = match.groups()
        arguments = {
            key.strip(): unescape(value)
            for key, value in NATIVE_ARGUMENT_PATTERN.findall(body)
        }
        tool_calls.append(
            {
                "type": "function",
                "id": f"{call_id_prefix}-{index}",
                "function": {
                    "name": tool.strip(),
                    "arguments": arguments,
                },
            }
        )
        cursor = match.end()
    visible_parts.append(response_content[cursor:])

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(visible_parts).strip(),
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content.strip()
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
