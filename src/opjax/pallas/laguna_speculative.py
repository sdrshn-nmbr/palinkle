"""Frozen contracts and replay benchmarking for Laguna speculative decoding."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import statistics
import sys
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
VLLM_OBSERVED_BUILD = "0.27.2rc1.dev18+g3d204dfda"


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
            "observed_build": VLLM_OBSERVED_BUILD,
            "source_revision": "unavailable_in_image_build_metadata",
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


def _speculative_config(
    arm: str,
    *,
    proposal_tokens: int | None = None,
    adaptive_verification: bool | None = None,
    draft_model: str | None = None,
    draft_revision: str | None = None,
) -> dict[str, Any] | None:
    resolved_tokens = 15 if proposal_tokens is None else proposal_tokens
    if resolved_tokens < 1 or resolved_tokens > 15:
        raise LagunaSpeculativeError(
            f"LAGUNA_SPECULATIVE_PROPOSAL_TOKENS_INVALID:{resolved_tokens}"
        )
    if arm == PLAIN:
        if (
            proposal_tokens is not None
            or adaptive_verification is not None
            or draft_model is not None
            or draft_revision is not None
        ):
            raise LagunaSpeculativeError("LAGUNA_PLAIN_SPECULATIVE_OPTIONS_INVALID")
        return None
    if arm == DFLASH:
        if adaptive_verification is not None:
            raise LagunaSpeculativeError("LAGUNA_DFLASH_ADAPTIVE_OPTION_INVALID")
        return {
            "method": "dflash",
            "model": draft_model or DFLASH_ID,
            **(
                {"revision": draft_revision or DFLASH_REVISION}
                if draft_model is None or draft_revision is not None
                else {}
            ),
            "num_speculative_tokens": resolved_tokens,
        }
    if arm == DSPARK:
        return {
            "method": "dspark",
            "model": draft_model or DSPARK_ID,
            **(
                {"revision": draft_revision or DSPARK_REVISION}
                if draft_model is None or draft_revision is not None
                else {}
            ),
            "num_speculative_tokens": resolved_tokens,
            "enable_adaptive_verification": (
                True if adaptive_verification is None else adaptive_verification
            ),
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
    rope_parameters = config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise LagunaSpeculativeError("LAGUNA_DSPARK_ROPE_PARAMETERS_MISSING")
    normalized["swa_rope_parameters"] = dict(rope_parameters)
    normalized["dflash_config"] = {
        "block_size": int(config["block_size"]),
        "mask_token_id": int(config["mask_token_id"]),
        "num_target_layers": int(config["num_target_layers"]),
        "target_layer_ids": list(config["target_layer_ids"]),
        "causal": bool(config["draft_causal"]),
    }
    return normalized


def server_command(
    arm: str,
    *,
    port: int,
    proposal_tokens: int | None = None,
    adaptive_verification: bool | None = None,
    draft_model: str | None = None,
    draft_revision: str | None = None,
    capture_dflash: bool = False,
) -> list[str]:
    speculative = _speculative_config(
        arm,
        proposal_tokens=proposal_tokens,
        adaptive_verification=adaptive_verification,
        draft_model=draft_model,
        draft_revision=draft_revision,
    )
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
                    "opjax.remote.laguna_dspark_vllm_model:LagunaDSparkForCausalLM"
                ),
                **(
                    {
                        "DFlashLagunaForCausalLM": (
                            "opjax.remote.laguna_dflash_capture_model:"
                            "CapturedLagunaDFlashForCausalLM"
                        )
                    }
                    if capture_dflash
                    else {}
                ),
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
                raise LagunaSpeculativeError(f"REPLAY_TOOL_ARGUMENTS_INVALID:{source}")
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
                raise LagunaSpeculativeError(
                    f"REPLAY_MESSAGE_INVALID:{trajectory_path}"
                )
            if message.get("role") == "assistant":
                assistant_index += 1
                if not public:
                    raise LagunaSpeculativeError(
                        f"REPLAY_PREFIX_EMPTY:{trajectory_path}"
                    )
                _validate_tool_links(public, source=str(trajectory_path))
                records.append(
                    {
                        "prompt_id": f"{trajectory_path.parent.name}--call-{assistant_index}",
                        "trajectory": trajectory_path.parent.name,
                        "call": assistant_index,
                        "messages": list(public),
                        "historical_completion_tokens": int(
                            ((message.get("extra") or {}).get("response") or {})
                            .get("usage", {})
                            .get("completion_tokens")
                            or 0
                        ),
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


def select_parity_panel(*, corpus: dict[str, Any], size: int = 48) -> dict[str, Any]:
    records = sorted(
        corpus["records"],
        key=lambda record: (
            record["historical_completion_tokens"],
            record["prompt_id"],
        ),
    )
    if size < 1 or size > len(records):
        raise LagunaSpeculativeError("LAGUNA_PARITY_PANEL_SIZE_INVALID")
    indices = (
        [len(records) // 2]
        if size == 1
        else [round(index * (len(records) - 1) / (size - 1)) for index in range(size)]
    )
    selected = [records[index] for index in indices]
    panel: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_parity_panel",
        "source_corpus_sha256": corpus["release_sha256"],
        "selection": "even_historical_completion_token_quantiles",
        "records": selected,
    }
    panel["release_sha256"] = canonical_sha256(panel)
    return panel


def canonical_response_signature(payload: dict[str, Any]) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tools = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        tools.append({"name": function.get("name"), "arguments": arguments})
    return {
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
        "tool_calls": tools,
    }


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
        "return_token_ids": True,
    }
    started = time.perf_counter()
    errors = []
    for attempt in range(1, 4):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body, sort_keys=True).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            errors.append({"attempt": attempt, "status": exc.code, "detail": detail[:1000]})
            if exc.code not in {502, 503, 504} or attempt == 3:
                raise LagunaSpeculativeError(
                    f"LAGUNA_BENCHMARK_HTTP_ERROR:{exc.code}:{detail[:1000]}"
                ) from exc
            print(
                "LAGUNA_BENCHMARK_TRANSIENT_RETRY "
                f"prompt={record['prompt_id']} attempt={attempt} status={exc.code}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(attempt)
        except (TimeoutError, urllib.error.URLError) as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "status": "transport",
                    "detail": str(exc)[:1000],
                }
            )
            if attempt == 3:
                raise LagunaSpeculativeError(
                    f"LAGUNA_BENCHMARK_TRANSPORT_ERROR:{record['prompt_id']}:{exc}"
                ) from exc
            print(
                "LAGUNA_BENCHMARK_TRANSIENT_RETRY "
                f"prompt={record['prompt_id']} attempt={attempt} transport={exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(attempt)
    elapsed = time.perf_counter() - started
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    signature = canonical_response_signature(payload)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens <= 0:
        raise LagunaSpeculativeError(
            f"LAGUNA_BENCHMARK_EMPTY_COMPLETION:{record['prompt_id']}"
        )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "prompt_id": record["prompt_id"],
        "trajectory": record["trajectory"],
        "call": record["call"],
        "historical_completion_tokens": record["historical_completion_tokens"],
        "request_attempts": attempt,
        "transient_errors": errors,
        "elapsed_s": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": choice.get("finish_reason"),
        "completion_token_ids": choice.get("token_ids"),
        "response_signature": signature,
        "response_signature_sha256": canonical_sha256(signature),
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


def partition_replay_records(
    records: list[dict[str, Any]], *, concurrency: int
) -> list[list[dict[str, Any]]]:
    if concurrency < 1:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CONCURRENCY_INVALID")
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        trajectories.setdefault(record["trajectory"], []).append(record)
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(concurrency)]
    for index, trajectory in enumerate(sorted(trajectories)):
        lanes[index % concurrency].extend(
            sorted(trajectories[trajectory], key=lambda record: record["call"])
        )
    return lanes


def warm_replay_endpoint(
    *,
    base_url: str,
    headers: dict[str, str],
    corpus: dict[str, Any],
    max_tokens: int,
) -> None:
    records = corpus.get("records") or []
    if not records:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CORPUS_EMPTY")
    _request(
        base_url=base_url,
        headers=headers,
        record=records[0],
        max_tokens=min(32, max_tokens),
    )


def run_replay_benchmark(
    *,
    arm: str,
    base_url: str,
    headers: dict[str, str],
    corpus: dict[str, Any],
    concurrency: int,
    max_tokens: int,
    limit: int | None = None,
    warmup: bool = True,
    model_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS or concurrency < 1 or max_tokens < 1:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CONFIG_INVALID")
    records = list(corpus["records"][:limit])
    if not records:
        raise LagunaSpeculativeError("LAGUNA_BENCHMARK_CORPUS_EMPTY")
    if warmup:
        warm_replay_endpoint(
            base_url=base_url,
            headers=headers,
            corpus={"records": records},
            max_tokens=max_tokens,
        )
    started = time.perf_counter()
    lanes = partition_replay_records(records, concurrency=concurrency)

    def run_lane(lane: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            _request(
                base_url=base_url,
                headers=headers,
                record=record,
                max_tokens=max_tokens,
            )
            for record in lane
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = [row for lane_rows in executor.map(run_lane, lanes) for row in lane_rows]
    rows.sort(key=lambda row: row["prompt_id"])
    wall_s = time.perf_counter() - started
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    latencies = [row["elapsed_s"] for row in rows]
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_replay_result",
        "arm": arm,
        "model_identity": model_identity
        or {"released_manifest_sha256": validate_model_manifest()["manifest_sha256"]},
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
            "p99": _percentile(latencies, 0.99),
        },
        "records": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result
