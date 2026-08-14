"""Frozen contracts and replay benchmarking for Laguna speculative decoding."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LagunaSpeculativeError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

PLAIN = "plain"
DFLASH = "dflash"
DSPARK = "dspark"
ARMS = (PLAIN, DFLASH, DSPARK)

TARGET_ID = "poolside/Laguna-XS-2.1"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
DFLASH_ID = "poolside/Laguna-XS-2.1-DFlash"
DFLASH_REVISION = "5c36361aab23c8ed3afbd079c10c426b677bc607"
DSPARK_ID = "RespectMathias/Laguna-XS-2.1-DSpark"
DSPARK_REVISION = "308567e50847b641e6dabcf82010d3b465b36cc2"
VLLM_IMAGE = "vllm/vllm-openai:nightly@sha256:df1979d8cfbc7e09da32ee568e2c189a76378db7894c5ae55d8eeb99e2be8f1b"
VLLM_SOURCE_REVISION = "38f097fab8f6d58e3b3f57bded1d98f5b48d3f6d"


def validate_model_manifest() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_model_manifest",
        "target": {
            "model_id": TARGET_ID,
            "revision": TARGET_REVISION,
            "precision": "bfloat16",
            "target_revision_used_by_community_training": (
                "205dc65dd4bda946c50da6b7522b215734fa107b"
            ),
            "target_weight_and_tokenizer_parity": (
                "all_14_safetensor_shards_tokenizer_and_chat_template_byte_identical"
            ),
            "sampling_policy": "explicit_request_parameters_ignore_generation_config",
        },
        "runtime": {
            "image": VLLM_IMAGE,
            "source_revision": VLLM_SOURCE_REVISION,
            "hardware": "H200:1",
            "tensor_parallel_size": 1,
        },
        "arms": {
            PLAIN: {"draft_model_id": None, "revision": None},
            DFLASH: {
                "draft_model_id": DFLASH_ID,
                "revision": DFLASH_REVISION,
                "checkpoint_parameters": 462_064_896,
                "proposal_tokens": 15,
                "causal_claim": "official_operational_control",
            },
            DSPARK: {
                "draft_model_id": DSPARK_ID,
                "revision": DSPARK_REVISION,
                "checkpoint_parameters": 924_489_217,
                "bundled_target_embedding_and_head_parameters": 411_041_792,
                "deployed_incremental_parameters": 513_447_425,
                "proposal_tokens": 15,
                "causal_claim": "operational_checkpoint",
                "training_boundary": (
                    "dflash_backbone_updated_with_markov_and_confidence_heads_for_64_steps"
                ),
            },
        },
        "claim_boundary": (
            "throughput_and_latency_comparison_not_a_clean_architecture_ablation"
        ),
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _speculative_config(arm: str) -> dict[str, Any] | None:
    if arm == PLAIN:
        return None
    if arm == DFLASH:
        return {
            "method": "dflash",
            "model": DFLASH_ID,
            "revision": DFLASH_REVISION,
            "num_speculative_tokens": 15,
        }
    if arm == DSPARK:
        return {
            "method": "dspark",
            "model": DSPARK_ID,
            "revision": DSPARK_REVISION,
            "num_speculative_tokens": 15,
            "enable_adaptive_verification": True,
        }
    raise LagunaSpeculativeError(f"LAGUNA_SPECULATIVE_ARM_INVALID:{arm}")


def normalize_dspark_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "vocab_size",
        "block_size",
        "proposal_length",
        "mask_token_id",
        "num_target_layers",
        "target_layer_ids",
        "draft_causal",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise LagunaSpeculativeError(
            f"LAGUNA_DSPARK_CONFIG_MISSING:{','.join(missing)}"
        )
    normalized = dict(config)
    normalized["architectures"] = ["Qwen3DSparkModel"]
    normalized["model_type"] = "laguna"
    normalized["draft_vocab_size"] = int(config["vocab_size"])
    normalized["n_predict"] = int(config["proposal_length"])
    normalized["dflash_config"] = {
        "block_size": int(config["block_size"]),
        "mask_token_id": int(config["mask_token_id"]),
        "num_target_layers": int(config["num_target_layers"]),
        "target_layer_ids": list(config["target_layer_ids"]),
        "causal": bool(config["draft_causal"]),
    }
    return normalized


def server_command(arm: str, *, port: int) -> list[str]:
    speculative = _speculative_config(arm)
    command = [
        "python",
        "-m",
        "opjax.remote.laguna_vllm_entrypoint",
        TARGET_ID,
        "--revision",
        TARGET_REVISION,
        "--served-model-name",
        TARGET_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.82",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "16",
        "--trust-remote-code",
        "--reasoning-parser",
        "poolside_v1",
        "--tool-call-parser",
        "poolside_v1",
        "--enable-auto-tool-choice",
        "--enable-per-request-metrics",
        "--model-class-overrides",
        json.dumps(
            {
                "Qwen3DSparkModel": (
                    "opjax.remote.laguna_dspark_vllm_model:"
                    "LagunaDSparkForCausalLM"
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--generation-config",
        "vllm",
        "--override-generation-config",
        "{}",
        "--seed",
        "0",
    ]
    if speculative is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps(speculative, sort_keys=True, separators=(",", ":")),
            ]
        )
    return command


def _public_message(message: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "role",
        "content",
        "reasoning_content",
        "tool_calls",
        "tool_call_id",
        "name",
    )
    return {key: message[key] for key in allowed if key in message}


def _validate_tool_links(messages: list[dict[str, Any]], *, source: str) -> None:
    call_ids: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_id = call.get("id") if isinstance(call, dict) else None
            if not isinstance(call_id, str) or not call_id:
                raise LagunaSpeculativeError(f"REPLAY_TOOL_CALL_ID_INVALID:{source}")
            call_ids.add(call_id)
            function = call.get("function") or {}
            if function.get("name") != "bash":
                raise LagunaSpeculativeError(f"REPLAY_TOOL_NAME_INVALID:{source}")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise LagunaSpeculativeError(
                        f"REPLAY_TOOL_ARGUMENTS_INVALID:{source}"
                    ) from exc
            if not isinstance(arguments, dict) or not isinstance(
                arguments.get("command"), str
            ):
                raise LagunaSpeculativeError(
                    f"REPLAY_TOOL_ARGUMENTS_INVALID:{source}"
                )
        if message.get("role") == "tool":
            result_id = message.get("tool_call_id")
            if result_id not in call_ids:
                raise LagunaSpeculativeError(
                    f"REPLAY_TOOL_RESULT_ORPHANED:{source}:{result_id}"
                )


def build_replay_corpus(*, sample_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    trajectories = sorted(sample_root.glob("runs/*/trajectory.json"))
    if not trajectories:
        raise LagunaSpeculativeError(f"REPLAY_TRAJECTORIES_MISSING:{sample_root}")
    for trajectory_path in trajectories:
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LagunaSpeculativeError(
                f"REPLAY_TRAJECTORY_INVALID:{trajectory_path}"
            ) from exc
        messages = trajectory.get("messages")
        if not isinstance(messages, list):
            raise LagunaSpeculativeError(f"REPLAY_MESSAGES_INVALID:{trajectory_path}")
        public: list[dict[str, Any]] = []
        assistant_index = 0
        for message in messages:
            if not isinstance(message, dict):
                raise LagunaSpeculativeError(f"REPLAY_MESSAGE_INVALID:{trajectory_path}")
            if message.get("role") == "assistant":
                assistant_index += 1
                if not public:
                    raise LagunaSpeculativeError(f"REPLAY_PREFIX_EMPTY:{trajectory_path}")
                _validate_tool_links(public, source=str(trajectory_path))
                records.append(
                    {
                        "prompt_id": f"{trajectory_path.parent.name}--call-{assistant_index}",
                        "trajectory": trajectory_path.parent.name,
                        "call": assistant_index,
                        "messages": list(public),
                    }
                )
            public.append(_public_message(message))
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_replay_corpus",
        "source_root": str(sample_root),
        "counts": {"prompts": len(records), "trajectories": len(trajectories)},
        "records": records,
    }
    result["release_sha256"] = canonical_sha256(result)
    return result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_VALUES_EMPTY")
    ordered = sorted(values)
    position = round((len(ordered) - 1) * quantile)
    return ordered[max(0, min(len(ordered) - 1, position))]


def _request(
    *,
    base_url: str,
    headers: dict[str, str],
    record: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": TARGET_ID,
        "messages": record["messages"],
        "tools": [BASH_TOOL],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, sort_keys=True).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise LagunaSpeculativeError(
            f"LAGUNA_BENCHMARK_HTTP_ERROR:{exc.code}:{detail[:1000]}"
        ) from exc
    elapsed = time.perf_counter() - started
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens <= 0:
        raise LagunaSpeculativeError(
            f"LAGUNA_BENCHMARK_EMPTY_COMPLETION:{record['prompt_id']}"
        )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "prompt_id": record["prompt_id"],
        "elapsed_s": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": choice.get("finish_reason"),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": hashlib.sha256(
            str(message.get("content") or "").encode()
        ).hexdigest(),
        "reasoning_sha256": hashlib.sha256(
            str(message.get("reasoning_content") or "").encode()
        ).hexdigest(),
        "tool_calls": len(message.get("tool_calls") or []),
        "server_metrics": payload.get("metrics"),
    }


def run_replay_benchmark(
    *,
    arm: str,
    base_url: str,
    headers: dict[str, str],
    corpus: dict[str, Any],
    concurrency: int,
    max_tokens: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if arm not in ARMS or concurrency < 1 or max_tokens < 1:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CONFIG_INVALID")
    records = list(corpus["records"][:limit])
    if not records:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CORPUS_EMPTY")
    _request(
        base_url=base_url,
        headers=headers,
        record=records[0],
        max_tokens=min(32, max_tokens),
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(
            executor.map(
                lambda record: _request(
                    base_url=base_url,
                    headers=headers,
                    record=record,
                    max_tokens=max_tokens,
                ),
                records,
            )
        )
    wall_s = time.perf_counter() - started
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    latencies = [row["elapsed_s"] for row in rows]
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_replay_result",
        "arm": arm,
        "model_manifest_sha256": validate_model_manifest()["manifest_sha256"],
        "corpus_sha256": corpus["release_sha256"],
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "wall_s": wall_s,
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": completion_tokens,
        "output_tps": completion_tokens / wall_s,
        "request_latency_s": {
            "mean": statistics.mean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "records": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result
