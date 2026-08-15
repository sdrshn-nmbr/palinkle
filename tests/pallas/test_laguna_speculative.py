from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.error

import pytest

from opjax.pallas.laguna_speculative import (
    BASH_TOOL,
    DFLASH,
    DSPARK,
    LagunaSpeculativeError,
    PLAIN,
    build_replay_corpus,
    bind_trained_runtime_identity,
    bind_released_runtime_identity,
    canonical_sha256,
    canonical_response_signature,
    normalize_dspark_config,
    partition_replay_records,
    _request,
    select_parity_panel,
    server_command,
    validate_model_manifest,
    validate_bound_replay_result,
    validate_live_serving_evidence,
)


def _runtime(arm: str, checkpoint: dict[str, object] | None) -> dict[str, object]:
    runtime: dict[str, object] = {
        "schema_version": 1,
        "arm": arm,
        "draft_checkpoint": checkpoint,
        "runtime_alignment": (
            {"state": "applied", "after_sha256": "patched"}
            if arm == DFLASH
            else None
        ),
        "execution_sources": {"entrypoint.py": "source"},
        "image": "image",
        "vllm_observed_build": "build",
        "resolved_arguments": (
            []
            if arm == PLAIN
            else [
                "--speculative-config",
                json.dumps(
                    {
                        "method": arm,
                        "num_speculative_tokens": 8,
                        **(
                            {"enable_adaptive_verification": False}
                            if arm == DSPARK
                            else {}
                        ),
                    }
                ),
            ]
        ),
    }
    runtime["sha256"] = canonical_sha256(runtime)
    return runtime


def test_runtime_identity_binds_actual_selected_checkpoint() -> None:
    checkpoint = {"sha256": "checkpoint", "files": {"model": "weights"}}
    selection = {"arm": DFLASH, "checkpoint": checkpoint, "sha256": "selection"}
    result = {
        "arm": DFLASH,
        "model_identity": selection,
        "result_sha256": "old",
    }
    bound = bind_trained_runtime_identity(
        result=result,
        runtime=_runtime(DFLASH, checkpoint),
        runtime_file_sha256="file",
        selection=selection,
    )
    assert bound["runtime_evidence"]["draft_checkpoint"] == checkpoint
    assert bound["runtime_evidence"]["runtime_alignment"]["state"] == "applied"
    assert bound["result_sha256"] == canonical_sha256(
        {key: value for key, value in bound.items() if key != "result_sha256"}
    )


def test_runtime_identity_rejects_wrong_checkpoint() -> None:
    selection = {
        "arm": DSPARK,
        "checkpoint": {"sha256": "selected"},
        "sha256": "selection",
    }
    result = {"arm": DSPARK, "model_identity": selection}
    with pytest.raises(LagunaSpeculativeError, match="RUNTIME_CHECKPOINT_MISMATCH"):
        bind_trained_runtime_identity(
            result=result,
            runtime=_runtime(DSPARK, {"sha256": "served"}),
            runtime_file_sha256="file",
            selection=selection,
        )


def test_runtime_identity_accepts_only_dspark_config_normalization() -> None:
    selection = {
        "arm": DSPARK,
        "checkpoint": {
            "sha256": "selected",
            "files": {"config.json": "source", "model.safetensors": "weights"},
        },
        "sha256": "selection",
    }
    result = {"arm": DSPARK, "model_identity": selection}
    runtime_checkpoint = {
        "sha256": "served",
        "path": "/tmp/opjax-dspark/snapshot",
        "files": {"config.json": "normalized", "model.safetensors": "weights"},
    }
    bound = bind_trained_runtime_identity(
        result=result,
        runtime=_runtime(DSPARK, runtime_checkpoint),
        runtime_file_sha256="file",
        selection=selection,
    )
    assert (
        bound["runtime_evidence"]["checkpoint_transform"]
        == "normalized_dspark_serving_config"
    )


def test_live_evidence_requires_hash_bound_replay() -> None:
    checkpoint = {"sha256": "checkpoint", "files": {"model": "weights"}}
    selection = {"arm": DFLASH, "checkpoint": checkpoint, "sha256": "selection"}
    result = bind_trained_runtime_identity(
        result={"arm": DFLASH, "model_identity": selection},
        runtime=_runtime(DFLASH, checkpoint),
        runtime_file_sha256="file",
        selection=selection,
    )
    assert validate_bound_replay_result(result=result, selection=selection)[
        "runtime_sha256"
    ]
    result["runtime_evidence"]["runtime_sha256"] = "drift"
    with pytest.raises(LagunaSpeculativeError, match="BOUND_REPLAY_HASH_MISMATCH"):
        validate_bound_replay_result(result=result, selection=selection)


def test_live_evidence_binds_selected_depth_and_endpoint() -> None:
    checkpoint = {"sha256": "checkpoint", "files": {"model": "weights"}}
    selection = {"arm": DFLASH, "checkpoint": checkpoint, "sha256": "selection"}
    result = bind_trained_runtime_identity(
        result={
            "arm": DFLASH,
            "cell": "dflash-8",
            "endpoint": "https://dflash-8.example",
            "model_identity": selection,
        },
        runtime=_runtime(DFLASH, checkpoint),
        runtime_file_sha256="file",
        selection=selection,
    )
    depth_selection = {"arm": DFLASH, "selected_depth": 8}
    depth_selection["sha256"] = canonical_sha256(depth_selection)
    serving = validate_live_serving_evidence(
        result=result,
        selection=selection,
        depth_selection=depth_selection,
    )
    assert serving["endpoint"] == "https://dflash-8.example"
    result["cell"] = "dflash-15"
    result["result_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    with pytest.raises(LagunaSpeculativeError, match="LAGUNA_LIVE_CELL_MISMATCH"):
        validate_live_serving_evidence(
            result=result,
            selection=selection,
            depth_selection=depth_selection,
        )


def test_released_dflash_requires_unmodified_runtime() -> None:
    manifest = validate_model_manifest()
    source = {
        "model": manifest["arms"][DFLASH]["draft_model_id"],
        "revision": manifest["arms"][DFLASH]["revision"],
    }
    runtime = {
        "arm": DFLASH,
        "argv": ["--speculative-config", json.dumps(source)],
        "draft_checkpoint": source,
        "runtime_alignment": {
            "state": "not_required_released_checkpoint",
            "model": source["model"],
        },
        "resolved_arguments": [],
    }
    runtime["sha256"] = canonical_sha256(runtime)
    result = {
        "arm": DFLASH,
        "model_identity": {"released_manifest_sha256": manifest["manifest_sha256"]},
    }
    bound = bind_released_runtime_identity(
        result=result,
        runtime=runtime,
        runtime_file_sha256="file",
    )
    assert bound["runtime_evidence"]["draft_source"] == source
    assert bound["runtime_evidence"]["model_identity_recovered_from_runtime"] is False
    recovered = bind_released_runtime_identity(
        result={"arm": DFLASH},
        runtime=runtime,
        runtime_file_sha256="file",
    )
    assert recovered["model_identity"] == result["model_identity"]
    assert recovered["runtime_evidence"]["model_identity_recovered_from_runtime"]
    runtime["runtime_alignment"] = {"state": "applied"}
    runtime["sha256"] = canonical_sha256(
        {key: value for key, value in runtime.items() if key != "sha256"}
    )
    with pytest.raises(
        LagunaSpeculativeError, match="LAGUNA_RELEASED_DFLASH_RUNTIME_INVALID"
    ):
        bind_released_runtime_identity(
            result=result,
            runtime=runtime,
            runtime_file_sha256="file",
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


def test_replay_request_retries_transport_failure(monkeypatch) -> None:
    payload = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "choices": [
            {
                "finish_reason": "stop",
                "token_ids": [7],
                "message": {"content": "done"},
            }
        ],
    }
    attempts = 0

    def open_request(_request, timeout):
        nonlocal attempts
        assert timeout == 1800
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError(TimeoutError("handshake timed out"))
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    result = _request(
        base_url="https://example.invalid",
        headers={},
        record={
            "prompt_id": "prompt-1",
            "trajectory": "trajectory-1",
            "call": 1,
            "historical_completion_tokens": 1,
            "messages": [{"role": "user", "content": "test"}],
        },
        max_tokens=8,
    )
    assert result["request_attempts"] == 2
    assert result["transient_errors"] == [
        {
            "attempt": 1,
            "status": "transport",
            "detail": "<urlopen error handshake timed out>",
        }
    ]
