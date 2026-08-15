from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.laguna_speculative import (
    BASH_TOOL,
    DFLASH,
    DSPARK,
    LagunaSpeculativeError,
    PLAIN,
    build_replay_corpus,
    canonical_sha256,
    canonical_response_signature,
    normalize_dspark_config,
    partition_replay_records,
    select_parity_panel,
    server_command,
    validate_model_manifest,
)


def test_model_manifest_pins_all_three_arms() -> None:
    manifest = validate_model_manifest()
    assert manifest["target"]["revision"] == (
        "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
    )
    assert manifest["arms"][DFLASH]["revision"] == (
        "5c36361aab23c8ed3afbd079c10c426b677bc607"
    )
    assert manifest["arms"][DSPARK]["revision"] == (
        "308567e50847b641e6dabcf82010d3b465b36cc2"
    )
    assert manifest["arms"][DSPARK]["causal_claim"] == "operational_checkpoint"
    assert manifest["arms"][DSPARK]["deployed_incremental_parameters"] == 513_447_425
    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert BASH_TOOL["function"]["parameters"]["required"] == ["command"]


def test_server_commands_share_runtime_and_target() -> None:
    commands = {arm: server_command(arm, port=8000) for arm in (PLAIN, DFLASH, DSPARK)}
    common = commands[PLAIN]
    for command in commands.values():
        assert command[:3] == ["python", "-m", "opjax.remote.laguna_vllm_entrypoint"]
        assert "poolside/Laguna-XS-2.1" in command
        assert "e9df9a59996d790b94b70f3fef343fe1d9e34bdf" in command
        assert "--enable-per-request-metrics" in command
    assert "--speculative-config" not in common
    assert "poolside/Laguna-XS-2.1-DFlash" in " ".join(commands[DFLASH])
    assert "RespectMathias/Laguna-XS-2.1-DSpark" in " ".join(commands[DSPARK])
    assert '"num_speculative_tokens":15' in commands[DFLASH][-1]
    assert '"num_speculative_tokens":15' in commands[DSPARK][-1]


def test_dflash_capture_overrides_the_laguna_runtime_class() -> None:
    command = server_command(DFLASH, port=8000, capture_dflash=True)
    override = command[command.index("--model-class-overrides") + 1]
    assert "DFlashLagunaForCausalLM" in override
    assert "CapturedLagunaDFlashForCausalLM" in override


def test_dspark_fixed_proposal_depth_is_explicit() -> None:
    command = server_command(
        DSPARK,
        port=8000,
        proposal_tokens=8,
        adaptive_verification=False,
    )
    config = json.loads(command[command.index("--speculative-config") + 1])
    assert config["num_speculative_tokens"] == 8
    assert config["enable_adaptive_verification"] is False


@pytest.mark.parametrize("arm", [DFLASH, DSPARK])
def test_trained_draft_path_does_not_invent_a_hub_revision(arm: str) -> None:
    command = server_command(
        arm,
        port=8000,
        proposal_tokens=4,
        adaptive_verification=False if arm == DSPARK else None,
        draft_model="/mnt/training/checkpoint",
    )
    config = json.loads(command[command.index("--speculative-config") + 1])
    assert config["model"] == "/mnt/training/checkpoint"
    assert "revision" not in config


@pytest.mark.parametrize("proposal_tokens", [0, 16])
def test_proposal_depth_outside_checkpoint_contract_fails(
    proposal_tokens: int,
) -> None:
    with pytest.raises(
        LagunaSpeculativeError,
        match="LAGUNA_SPECULATIVE_PROPOSAL_TOKENS_INVALID",
    ):
        server_command(DSPARK, port=8000, proposal_tokens=proposal_tokens)


def test_normalize_dspark_config_preserves_laguna_contract() -> None:
    original = {
        "architectures": ["LagunaDSparkModel"],
        "vocab_size": 100352,
        "block_size": 16,
        "proposal_length": 15,
        "mask_token_id": 12,
        "num_target_layers": 40,
        "target_layer_ids": [1, 13, 25, 33, 39],
        "draft_causal": True,
        "rope_parameters": {
            "rope_theta": 500000.0,
            "rope_type": "default",
        },
        "swa_rope_parameters": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }
    normalized = normalize_dspark_config(original)
    assert normalized["architectures"] == ["Qwen3DSparkModel"]
    assert normalized["model_type"] == "laguna"
    assert normalized["draft_vocab_size"] == 100352
    assert normalized["n_predict"] == 15
    assert normalized["swa_rope_parameters"] == {
        "rope_theta": 500000.0,
        "rope_type": "default",
    }
    assert normalized["dflash_config"] == {
        "block_size": 16,
        "mask_token_id": 12,
        "num_target_layers": 40,
        "target_layer_ids": [1, 13, 25, 33, 39],
        "causal": True,
    }
    assert original["architectures"] == ["LagunaDSparkModel"]


def _trajectory(path: Path, messages: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")


def test_replay_corpus_preserves_native_structured_history(tmp_path: Path) -> None:
    root = tmp_path / "samples"
    _trajectory(
        root / "runs" / "task--seed-0" / "trajectory.json",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "inspect",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"pwd"}',
                        },
                    }
                ],
                "extra": {"private": "discard"},
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "<output>/workspace</output>",
                "extra": {"private": "discard"},
            },
            {"role": "assistant", "content": "next", "extra": {}},
        ],
    )
    corpus = build_replay_corpus(sample_root=root)
    assert corpus["counts"] == {"prompts": 2, "trajectories": 1}
    second = corpus["records"][1]["messages"]
    assert second[2]["tool_calls"][0]["id"] == "call-1"
    assert second[2]["reasoning_content"] == "inspect"
    assert second[3]["tool_call_id"] == "call-1"
    assert all("extra" not in message for message in second)
    panel = select_parity_panel(corpus=corpus, size=2)
    assert len(panel["records"]) == 2


def test_replay_corpus_rejects_broken_tool_link(tmp_path: Path) -> None:
    root = tmp_path / "samples"
    _trajectory(
        root / "runs" / "task--seed-0" / "trajectory.json",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "missing", "content": "bad"},
            {"role": "assistant", "content": "next"},
        ],
    )
    with pytest.raises(LagunaSpeculativeError, match="REPLAY_TOOL_RESULT_ORPHANED"):
        build_replay_corpus(sample_root=root)


def test_response_signature_ignores_random_tool_call_id() -> None:
    def payload(call_id: str) -> dict[str, object]:
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }

    assert canonical_response_signature(payload("a")) == canonical_response_signature(
        payload("b")
    )


def test_replay_partition_preserves_trajectory_order() -> None:
    records = [
        {"trajectory": "b", "call": 2},
        {"trajectory": "a", "call": 2},
        {"trajectory": "b", "call": 1},
        {"trajectory": "a", "call": 1},
        {"trajectory": "c", "call": 1},
    ]
    lanes = partition_replay_records(records, concurrency=2)
    assert lanes == [
        [
            {"trajectory": "a", "call": 1},
            {"trajectory": "a", "call": 2},
            {"trajectory": "c", "call": 1},
        ],
        [
            {"trajectory": "b", "call": 1},
            {"trajectory": "b", "call": 2},
        ],
    ]
