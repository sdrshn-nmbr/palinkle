"""Live two-turn provider protocol conformance."""

from __future__ import annotations

import subprocess
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256


SYSTEM_PROMPT = (
    "You are a protocol conformance agent. Make exactly one native shell tool "
    "call per turn and no prose."
)
USER_PROMPT = (
    "First call shell with command `printf PROTOCOL_ONE`. After its tool result "
    "contains PROTOCOL_ONE, call shell with command `printf PROTOCOL_TWO`. "
    "Do not combine the commands."
)
EXPECTED_COMMANDS = ("printf PROTOCOL_ONE", "printf PROTOCOL_TWO")


def _action(message: dict[str, Any], *, expected: str) -> dict[str, str]:
    actions = message.get("extra", {}).get("actions", ())
    calls = message.get("tool_calls", ())
    if (
        len(actions) != 1
        or actions[0].get("command") != expected
        or len(calls) != 1
        or not calls[0].get("id")
        or calls[0].get("function", {}).get("name") not in {
            "bash",
            "mswea_bash_command",
            "shell",
        }
    ):
        raise G42HarnessError(f"PHASE31_PROTOCOL_ACTION_INVALID:{expected}")
    return actions[0]


def _execute(command: str) -> dict[str, Any]:
    if command not in EXPECTED_COMMANDS:
        raise G42HarnessError("PHASE31_PROTOCOL_COMMAND_NOT_ALLOWLISTED")
    process = subprocess.run(
        ["/bin/sh", "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise G42HarnessError("PHASE31_PROTOCOL_COMMAND_FAILED")
    return {
        "returncode": process.returncode,
        "output": process.stdout,
        "exception_info": None,
    }


def run_two_turn_conformance(
    *, model: Any, provider: str, model_identity: dict[str, Any]
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    first = model.query(messages)
    first_action = _action(first, expected=EXPECTED_COMMANDS[0])
    first_output = _execute(first_action["command"])
    observations = model.format_observation_messages(first, [first_output])
    if len(observations) != 1:
        raise G42HarnessError("PHASE31_PROTOCOL_OBSERVATION_COUNT_INVALID")
    observation = observations[0]
    first_call = first["tool_calls"][0]
    if (
        observation.get("role") != "tool"
        or observation.get("tool_call_id") != first_call["id"]
        or observation.get("name") != first_call["function"]["name"]
    ):
        raise G42HarnessError("PHASE31_PROTOCOL_OBSERVATION_LINK_INVALID")
    second = model.query([*messages, first, observation])
    second_action = _action(second, expected=EXPECTED_COMMANDS[1])
    second_output = _execute(second_action["command"])
    result = {
        "schema_version": 1,
        "kind": "opjax_phase31_provider_protocol_conformance",
        "provider": provider,
        "model": model_identity,
        "passed": True,
        "model_calls": 2,
        "repair_turns": 0,
        "messages": [first, observation, second],
        "outputs": [first_output, second_output],
    }
    result["conformance_sha256"] = canonical_sha256(result)
    return result
