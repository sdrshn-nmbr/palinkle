"""Run and capture one eager vLLM Laguna DSpark proposal round."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np

from opjax.pallas.laguna_speculative import (
    DFLASH,
    DSPARK,
    DSPARK_ID,
    DSPARK_REVISION,
    TARGET_ID,
    TARGET_REVISION,
    VLLM_OBSERVED_BUILD,
    canonical_sha256,
    server_command,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_source_sha256() -> str:
    root = Path(__file__).parent
    return canonical_sha256(
        {
            name: _sha256(root / name)
            for name in (
                "laguna_dspark_capture.py",
                "laguna_dspark_vllm_model.py",
                "laguna_vllm_conformance.py",
            )
        }
    )


def _request(
    url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30.0
) -> bytes:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None or url.endswith("_profile") else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP_ERROR:{exc.code}:{url}:{body}") from None


def _wait_ready(port: int, process: subprocess.Popen[bytes], *, log_path: Path) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"VLLM_EXITED_BEFORE_READY:{process.returncode}:LOG_TAIL:{tail}"
            )
        try:
            _request(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError("VLLM_READINESS_TIMEOUT")


def _replace_argument(command: list[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = value


def _disable_adaptive_verification(command: list[str]) -> None:
    index = command.index("--speculative-config") + 1
    config = json.loads(command[index])
    if config.get("method") != "dspark":
        raise RuntimeError("VLLM_CONFORMANCE_REQUIRES_DSPARK")
    config["enable_adaptive_verification"] = False
    command[index] = json.dumps(config, sort_keys=True, separators=(",", ":"))


def _load_records(root: Path) -> dict[str, list[tuple[dict[str, Any], np.ndarray]]]:
    records: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
    ledger = root / "ledger.jsonl"
    if not ledger.is_file():
        raise RuntimeError("VLLM_CAPTURE_LEDGER_MISSING")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        path = root / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"VLLM_CAPTURE_HASH_MISMATCH:{path}")
        records.setdefault(record["name"], []).append(
            (record, np.load(path, allow_pickle=False))
        )
    return records


def _one(
    records: dict[str, list[tuple[dict[str, Any], np.ndarray]]], name: str
) -> np.ndarray:
    values = records.get(name, [])
    if len(values) != 1:
        raise RuntimeError(f"VLLM_CAPTURE_CARDINALITY:{name}:{len(values)}")
    return values[0][1]


def _first(
    records: dict[str, list[tuple[dict[str, Any], np.ndarray]]], name: str
) -> np.ndarray:
    values = records.get(name, [])
    if not values:
        raise RuntimeError(f"VLLM_CAPTURE_MISSING:{name}")
    return values[0][1]


def _at(
    records: dict[str, list[tuple[dict[str, Any], np.ndarray]]],
    name: str,
    index: int,
    *,
    require_one: bool = True,
) -> np.ndarray:
    values = records.get(name, [])
    round_values = [value for record, value in values if record.get("round") == index]
    if round_values:
        if require_one and len(round_values) != 1:
            raise RuntimeError(
                f"VLLM_CAPTURE_ROUND_CARDINALITY:{name}:{index}:{len(round_values)}"
            )
        return round_values[0]
    if any(record.get("round") is not None for record, _value in values):
        raise RuntimeError(f"VLLM_CAPTURE_ROUND_MISSING:{name}:{index}")
    if index >= len(values):
        raise RuntimeError(f"VLLM_CAPTURE_INDEX_MISSING:{name}:{index}:{len(values)}")
    return values[index][1]


def _round_arrays(
    records: dict[str, list[tuple[dict[str, Any], np.ndarray]]],
    name: str,
    round_id: int,
) -> list[np.ndarray]:
    values = records.get(name, [])
    selected = [value for record, value in values if record.get("round") == round_id]
    if selected:
        return selected
    if any(record.get("round") is not None for record, _value in values):
        return []
    start = round_id * 15
    return [value for _record, value in values][start : start + 15]


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _bf16_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    summed = np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
    bits = summed.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _bf16_round(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def ordered(value: np.ndarray) -> np.ndarray:
        bits = np.asarray(value, dtype=np.float32).view(np.uint32) >> 16
        signed = (bits & np.uint32(0x8000)) != 0
        return np.where(
            signed,
            np.uint32(0x8000) - (bits & np.uint32(0x7FFF)),
            np.uint32(0x8000) + bits,
        ).astype(np.int32)

    return np.abs(ordered(left) - ordered(right))


def _canonicalize_capture(
    raw_root: Path,
    static_root: Path,
    output_root: Path,
    *,
    capture_index: int = 0,
) -> dict[str, dict[str, Any]]:
    records = _load_records(raw_root)
    static_records = _load_records(static_root)
    raw_target_features = _at(records, "raw_target_features", capture_index)
    combined = _at(records, "combined_target_feature", capture_index)
    hidden = _at(records, "draft_backbone_hidden_state", capture_index)
    base_logits = _at(records, "base_logits", capture_index)
    bias_values = _round_arrays(records, "markov_bias", capture_index)
    input_values = _round_arrays(records, "markov_input_token_ids", capture_index)
    embedding_values = _round_arrays(records, "markov_embedding", capture_index)
    corrected_values = _round_arrays(
        records, "corrected_logits_runtime", capture_index
    )
    proposal_values = _round_arrays(
        records, "proposal_token_ids_runtime", capture_index
    )
    confidence_values = _round_arrays(
        records, "confidence_logits_instrumented", capture_index
    )
    cardinalities = {
        "bias": len(bias_values),
        "tokens": len(input_values),
        "embeddings": len(embedding_values),
        "corrected": len(corrected_values),
        "proposals": len(proposal_values),
        "confidence": len(confidence_values),
    }
    if any(value != 15 for value in cardinalities.values()):
        raise RuntimeError(
            "VLLM_MARKOV_CAPTURE_CARDINALITY:"
            + ":".join(f"{key}={value}" for key, value in cardinalities.items())
        )
    base_logits = base_logits.reshape(-1, base_logits.shape[-1])
    if base_logits.shape[0] != 15:
        raise RuntimeError(f"VLLM_BASE_LOGIT_POSITIONS:{base_logits.shape}")
    markov_bias = np.concatenate(
        [value.reshape(-1, value.shape[-1]) for value in bias_values], axis=0
    )
    if markov_bias.shape != base_logits.shape:
        raise RuntimeError(
            f"VLLM_MARKOV_BIAS_SHAPE:{markov_bias.shape}:{base_logits.shape}"
        )
    corrected_recomputed = _bf16_add(base_logits, markov_bias)
    corrected = np.concatenate(
        [value.reshape(-1, value.shape[-1]) for value in corrected_values], axis=0
    )
    proposal_tokens = np.concatenate(
        [value.reshape(-1) for value in proposal_values]
    ).astype(np.int64)
    if corrected.shape != corrected_recomputed.shape or proposal_tokens.shape != (15,):
        raise RuntimeError(
            f"VLLM_RUNTIME_PROPOSAL_SHAPE:{corrected.shape}:"
            f"{corrected_recomputed.shape}:{proposal_tokens.shape}"
        )
    if not np.array_equal(corrected, corrected_recomputed):
        raise RuntimeError("VLLM_CORRECTED_LOGITS_RECOMPUTE_MISMATCH")
    if not np.array_equal(
        proposal_tokens, np.argmax(corrected, axis=-1).astype(np.int64)
    ):
        raise RuntimeError("VLLM_RUNTIME_PROPOSAL_ARGMAX_MISMATCH")
    previous_tokens = np.concatenate([value.reshape(-1) for value in input_values])
    if not np.array_equal(previous_tokens[1:], proposal_tokens[:-1]):
        raise RuntimeError("VLLM_MARKOV_CHAIN_INCONSISTENT")
    markov_embeddings = np.concatenate(
        [value.reshape(-1, value.shape[-1]) for value in embedding_values], axis=0
    )
    confidence_weight = _one(static_records, "confidence_head_proj_weight").astype(
        np.float32
    )
    confidence_bias = _one(static_records, "confidence_head_proj_bias").astype(
        np.float32
    )
    confidence_features = np.concatenate(
        [hidden.reshape(-1, hidden.shape[-1]), markov_embeddings], axis=-1
    ).astype(np.float32)
    confidence_recomputed = _bf16_round(
        confidence_features @ confidence_weight.reshape(1, -1).T
        + confidence_bias.reshape(1, -1)
    ).reshape(-1)
    confidence = np.concatenate(
        [value.reshape(-1) for value in confidence_values]
    )
    if confidence.shape != (15,) or confidence_recomputed.shape != (15,):
        raise RuntimeError(
            f"VLLM_CONFIDENCE_POSITIONS:{confidence.shape}:"
            f"{confidence_recomputed.shape}"
        )
    confidence_recompute_ulp = _bf16_ulp_distance(
        confidence, confidence_recomputed
    )
    if int(confidence_recompute_ulp.max(initial=0)) > 2:
        raise RuntimeError(
            "VLLM_CONFIDENCE_RECOMPUTE_MISMATCH:"
            f"{int(confidence_recompute_ulp.max())}"
        )
    arrays = {
        "raw_target_features": raw_target_features.reshape(
            -1, raw_target_features.shape[-1]
        ),
        "draft_input_ids": _at(records, "draft_input_ids", capture_index).reshape(-1),
        "draft_positions": _at(records, "draft_positions", capture_index).reshape(-1),
        "draft_input_embeddings": _at(records, "draft_input_embeddings", capture_index).reshape(
            -1, hidden.shape[-1]
        ),
        **{
            f"draft_layer_{layer_id}_output": _at(
                records, f"draft_layer_{layer_id}_output", capture_index
            ).reshape(-1, hidden.shape[-1])
            for layer_id in range(5)
        },
        **{
            f"{name}_0": _at(records, name, capture_index, require_one=False)
            for name in (
                "layer0_input_norm",
                "layer0_qkv_projection",
                "layer0_q_norm",
                "layer0_k_norm",
                "layer0_gate_projection",
                "layer0_attention_output",
                "layer0_gated_attention",
                "layer0_post_attention_norm",
                "layer0_mlp_output",
                "layer0_query_q_after_rope",
                "layer0_query_k_after_rope",
                "layer0_query_v",
                "layer0_context_k_before_rope",
                "layer0_context_v",
            )
        },
        "combined_target_feature": combined.reshape(-1, combined.shape[-1]),
        "draft_backbone_hidden_state": hidden.reshape(-1, hidden.shape[-1]),
        "base_logits": base_logits,
        "markov_bias": markov_bias,
        "corrected_logits": corrected,
        "corrected_logits_recomputed": corrected_recomputed,
        "confidence_logits": confidence,
        "confidence_logits_recomputed": confidence_recomputed,
        "confidence_recompute_ulp": confidence_recompute_ulp,
        "proposal_token_ids": proposal_tokens,
    }
    return {
        name: _save_array(output_root, name, value) for name, value in arrays.items()
    }


def run_capture(
    *,
    output_root: Path,
    prompt: str,
    target_feature_override: Path | None,
    draft_model: str = DSPARK_ID,
    port: int = 8000,
    target_features_only: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    raw_capture_root = output_root / "raw"
    profile_root = output_root / "profiles"
    raw_capture_root.mkdir()
    profile_root.mkdir()
    command = server_command(DSPARK, port=port, draft_model=draft_model)
    _replace_argument(command, "--max-model-len", "4096")
    _replace_argument(command, "--max-num-seqs", "1")
    _disable_adaptive_verification(command)
    command.extend(
        [
            "--enforce-eager",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_DSPARK_CAPTURE_ROOT": str(raw_capture_root),
        "OPJAX_SPEC_RUN_ID": output_root.name,
    }
    if target_feature_override is not None:
        environment["OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE"] = str(
            target_feature_override
        )
    else:
        environment.pop("OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE", None)
    log_path = output_root / "server.log"
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process, log_path=log_path)
            prompt_token_ids = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/tokenize",
                    payload={
                        "model": TARGET_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
            )["tokens"]
            session = "first-proposal"
            (raw_capture_root / "active.json").write_text(
                json.dumps({"session": session}), encoding="utf-8"
            )
            _request(f"http://127.0.0.1:{port}/start_profile", payload={})
            response = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    payload={
                        "model": TARGET_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "chat_template_kwargs": {"enable_thinking": True},
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "seed": 0,
                    },
                    timeout=600,
                )
            )
            _request(f"http://127.0.0.1:{port}/stop_profile", payload={}, timeout=600)
            (raw_capture_root / "active.json").unlink()
            time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    raw_session_root = raw_capture_root / "first-proposal"
    if target_features_only:
        records = _load_records(raw_session_root)
        raw_target_features = _first(records, "raw_target_features")
        boundaries = {
            "raw_target_features": _save_array(
                output_root,
                "raw_target_features",
                raw_target_features.reshape(-1, raw_target_features.shape[-1]),
            )
        }
    else:
        boundaries = _canonicalize_capture(
            raw_session_root, raw_capture_root / "static", output_root
        )
    trace_files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    if not trace_files:
        raise RuntimeError("VLLM_PROFILE_ARTIFACT_MISSING")
    trace_index: dict[str, Any] = {
        "files": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in trace_files
        ]
    }
    trace_index["sha256"] = canonical_sha256(trace_index)
    trace_index_path = output_root / "trace-index.json"
    trace_index_path.write_text(
        json.dumps(trace_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    version = subprocess.run(
        ["vllm", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != VLLM_OBSERVED_BUILD:
        raise RuntimeError(f"VLLM_BUILD_MISMATCH:{version}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "implementation": "opjax_vllm_laguna_dspark_adapter",
        "capture_scope": (
            "target_features_only" if target_features_only else "full_proposal"
        ),
        "provenance": {
            "revision": VLLM_OBSERVED_BUILD,
            "source_sha256": _adapter_source_sha256(),
            "target_revision": TARGET_REVISION,
            "draft_revision": (
                DSPARK_REVISION
                if draft_model == DSPARK_ID
                else _sha256(Path(draft_model) / "model.safetensors")
            ),
            "command": command,
            "target_feature_mode": (
                "source_override" if target_feature_override is not None else "live_vllm"
            ),
            "target_feature_override": (
                {
                    "path": str(target_feature_override),
                    "sha256": _sha256(target_feature_override),
                }
                if target_feature_override is not None
                else None
            ),
        },
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "response": response,
        "boundaries": boundaries,
        "trace": {
            "path": trace_index_path.name,
            "sha256": _sha256(trace_index_path),
            "bytes": trace_index_path.stat().st_size,
        },
        "server_log": {
            "path": log_path.name,
            "sha256": _sha256(log_path),
            "bytes": log_path.stat().st_size,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_token_rounds_capture(
    *,
    output_root: Path,
    context_id: str,
    source_root: Path,
    lane: str,
    draft_model: str,
    expected_processed_starts: list[int] | None = None,
    port: int = 8000,
) -> dict[str, Any]:
    """Capture three forced-prefix rounds in one prefix-caching vLLM process."""
    if lane not in {"injected", "native"}:
        raise ValueError(f"VLLM_MULTIRound_LANE_INVALID:{lane}")
    if lane == "injected" and (
        expected_processed_starts is None or len(expected_processed_starts) != 3
    ):
        raise ValueError("VLLM_INJECTED_CACHE_STARTS_INVALID")
    if lane == "native" and expected_processed_starts is not None:
        raise ValueError("VLLM_NATIVE_CACHE_STARTS_FORBIDDEN")
    output_root.mkdir(parents=True, exist_ok=False)
    raw_capture_root = output_root / "raw"
    profile_root = output_root / "profiles"
    raw_capture_root.mkdir()
    profile_root.mkdir()
    command = server_command(DSPARK, port=port, draft_model=draft_model)
    _replace_argument(command, "--max-num-seqs", "1")
    _disable_adaptive_verification(command)
    command.extend(
        [
            "--enforce-eager",
            "--enable-prefix-caching",
            "--block-size",
            "16",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_DSPARK_CAPTURE_ROOT": str(raw_capture_root),
        "OPJAX_SPEC_RUN_ID": output_root.name,
    }
    environment.pop("OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE", None)
    log_path = output_root / "server.log"
    cell_manifests: list[dict[str, Any]] = []
    metrics_before = ""
    metrics_after = ""
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process, log_path=log_path)
            metrics_before = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            _request(f"http://127.0.0.1:{port}/start_profile", payload={})
            for round_index in range(3):
                cell_id = f"{context_id}--round-{round_index}"
                source_cell = source_root / cell_id
                source = json.loads(
                    (source_cell / "manifest.json").read_text(encoding="utf-8")
                )
                prompt_token_ids = source["prompt_token_ids"]
                control: dict[str, Any] = {"session": cell_id}
                if lane == "injected":
                    source_features = np.load(
                        source_cell / source["boundaries"]["raw_target_features"]["path"],
                        allow_pickle=False,
                    )
                    if expected_processed_starts is None:
                        raise RuntimeError("VLLM_INJECTED_CACHE_STARTS_MISSING")
                    cached_tokens = expected_processed_starts[round_index]
                    override = source_features[:, cached_tokens:, :]
                    override_path = output_root / f"{cell_id}-target-features.npy"
                    np.save(override_path, override)
                    control["target_feature_override"] = str(override_path)
                (raw_capture_root / "active.json").write_text(
                    json.dumps(control, sort_keys=True), encoding="utf-8"
                )
                response = json.loads(
                    _request(
                        f"http://127.0.0.1:{port}/v1/completions",
                        payload={
                            "model": TARGET_ID,
                            "prompt": prompt_token_ids,
                            "temperature": 0.0,
                            "max_tokens": 1,
                            "seed": 0,
                            "return_token_ids": True,
                        },
                        timeout=600,
                    )
                )
                (raw_capture_root / "active.json").unlink()
                cell_root = output_root / "cells" / cell_id
                cell_root.mkdir(parents=True)
                boundaries = _canonicalize_capture(
                    raw_capture_root / cell_id,
                    raw_capture_root / "static",
                    cell_root,
                )
                observed_rows = int(boundaries["raw_target_features"]["shape"][0])
                processed_start = len(prompt_token_ids) - observed_rows
                expected_processed_start = (
                    expected_processed_starts[round_index]
                    if expected_processed_starts is not None
                    else None
                )
                manifest: dict[str, Any] = {
                    "schema_version": 1,
                    "implementation": "opjax_vllm_laguna_dspark_multiround",
                    "provenance": {
                        "revision": VLLM_OBSERVED_BUILD,
                        "source_sha256": _adapter_source_sha256(),
                        "target_revision": TARGET_REVISION,
                        "draft_revision": _sha256(Path(draft_model) / "model.safetensors"),
                        "command": command,
                        "lane": lane,
                        "source_manifest_sha256": source["manifest_sha256"],
                    },
                    "context_id": context_id,
                    "round": round_index,
                    "prompt_token_ids": prompt_token_ids,
                    "processed_token_start": processed_start,
                    "expected_processed_token_start": expected_processed_start,
                    "response": response,
                    "boundaries": boundaries,
                    "mutation_controls": source["mutation_controls"],
                }
                cell_manifests.append(manifest)
            _request(f"http://127.0.0.1:{port}/stop_profile", payload={}, timeout=600)
            metrics_after = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            time.sleep(5)
        finally:
            active = raw_capture_root / "active.json"
            if active.exists():
                active.unlink()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    trace_files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    if not trace_files:
        raise RuntimeError("VLLM_MULTIRound_PROFILE_ARTIFACT_MISSING")
    trace_index = {
        "files": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in trace_files
        ]
    }
    trace_index["sha256"] = canonical_sha256(trace_index)
    trace_index_path = output_root / "trace-index.json"
    trace_index_path.write_text(
        json.dumps(trace_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest in cell_manifests:
        cell_root = output_root / "cells" / str(manifest["context_id"] + "--round-" + str(manifest["round"]))
        manifest["trace"] = {
            "path": "../../trace-index.json",
            "sha256": _sha256(trace_index_path),
            "bytes": trace_index_path.stat().st_size,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (cell_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "laguna_dspark_vllm_forced_prefix_rounds",
        "context_id": context_id,
        "lane": lane,
        "command": command,
        "cells": [manifest["manifest_sha256"] for manifest in cell_manifests],
        "processed_token_starts": [
            manifest["processed_token_start"] for manifest in cell_manifests
        ],
        "metrics_before_sha256": hashlib.sha256(metrics_before.encode()).hexdigest(),
        "metrics_after_sha256": hashlib.sha256(metrics_after.encode()).hexdigest(),
        "trace_index_sha256": _sha256(trace_index_path),
        "server_log_sha256": _sha256(log_path),
    }
    (output_root / "metrics-before.txt").write_text(metrics_before, encoding="utf-8")
    (output_root / "metrics-after.txt").write_text(metrics_after, encoding="utf-8")
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _completion_token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("VLLM_COMPLETION_CHOICES_INVALID")
    tokens = choices[0].get("token_ids")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError("VLLM_COMPLETION_TOKEN_IDS_MISSING")
    return tokens


def run_sequential_capture(
    *,
    output_root: Path,
    context_id: str,
    prompt_token_ids: list[int],
    draft_model: str,
    target_feature_overrides: list[Path] | None = None,
    port: int = 8000,
) -> dict[str, Any]:
    """Capture the first three proposal invocations from one completion request."""
    output_root.mkdir(parents=True, exist_ok=False)
    raw_capture_root = output_root / "raw"
    profile_root = output_root / "profiles"
    raw_capture_root.mkdir()
    profile_root.mkdir()
    command = server_command(DSPARK, port=port, draft_model=draft_model)
    _replace_argument(command, "--max-num-seqs", "1")
    _disable_adaptive_verification(command)
    command.extend(
        [
            "--enforce-eager",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_DSPARK_CAPTURE_ROOT": str(raw_capture_root),
        "OPJAX_SPEC_RUN_ID": output_root.name,
    }
    environment.pop("OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE", None)
    session = f"{context_id}--sequential"
    control: dict[str, Any] = {"session": session}
    if target_feature_overrides is not None:
        if len(target_feature_overrides) != 3:
            raise ValueError("VLLM_SEQUENTIAL_OVERRIDE_COUNT")
        control["target_feature_overrides"] = [
            str(path) for path in target_feature_overrides
        ]
        control["allow_native_after_override_exhaustion"] = True
    log_path = output_root / "server.log"
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process, log_path=log_path)
            metrics_before = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            (raw_capture_root / "active.json").write_text(
                json.dumps(control, sort_keys=True), encoding="utf-8"
            )
            _request(f"http://127.0.0.1:{port}/start_profile", payload={})
            response = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/v1/completions",
                    payload={
                        "model": TARGET_ID,
                        "prompt": prompt_token_ids,
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "seed": 0,
                        "return_token_ids": True,
                    },
                    timeout=1200,
                )
            )
            _request(f"http://127.0.0.1:{port}/stop_profile", payload={}, timeout=600)
            metrics_after = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            (raw_capture_root / "active.json").unlink()
            time.sleep(5)
        finally:
            active = raw_capture_root / "active.json"
            if active.exists():
                active.unlink()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    records = _load_records(raw_capture_root / session)
    proposal_count = len(records.get("raw_target_features", []))
    if proposal_count < 3:
        raise RuntimeError(f"VLLM_SEQUENTIAL_PROPOSAL_COUNT:{proposal_count}")
    generated = _completion_token_ids(response)
    cells: list[dict[str, Any]] = []
    for round_index in range(3):
        cell_id = f"{context_id}--round-{round_index}"
        cell_root = output_root / "cells" / cell_id
        cell_root.mkdir(parents=True)
        boundaries = _canonicalize_capture(
            raw_capture_root / session,
            raw_capture_root / "static",
            cell_root,
            capture_index=round_index,
        )
        positions = np.load(
            cell_root / boundaries["draft_positions"]["path"], allow_pickle=False
        ).reshape(-1)
        if positions.size == 0:
            raise RuntimeError(f"VLLM_SEQUENTIAL_POSITIONS_EMPTY:{cell_id}")
        prefix_length = int(positions[0])
        generated_count = prefix_length - len(prompt_token_ids)
        if not 0 <= generated_count <= len(generated):
            raise RuntimeError(
                f"VLLM_SEQUENTIAL_PREFIX_INVALID:{cell_id}:{prefix_length}:"
                f"{len(prompt_token_ids)}:{len(generated)}"
            )
        committed = [*prompt_token_ids, *generated[:generated_count]]
        observed_rows = int(boundaries["raw_target_features"]["shape"][0])
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "implementation": "opjax_vllm_laguna_dspark_sequential",
            "provenance": {
                "revision": VLLM_OBSERVED_BUILD,
                "source_sha256": _adapter_source_sha256(),
                "target_revision": TARGET_REVISION,
                "draft_revision": _sha256(Path(draft_model) / "model.safetensors"),
                "command": command,
                "target_feature_mode": (
                    "source_override" if target_feature_overrides is not None else "live_vllm"
                ),
                "override_rounds": (
                    len(target_feature_overrides)
                    if target_feature_overrides is not None
                    else 0
                ),
                "allow_native_after_override_exhaustion": (
                    target_feature_overrides is not None
                ),
            },
            "context_id": context_id,
            "round": round_index,
            "prompt_token_ids": committed,
            "processed_token_start": len(committed) - observed_rows,
            "boundaries": boundaries,
            "response_token_ids": generated,
        }
        cells.append(manifest)
    trace_files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    if not trace_files:
        raise RuntimeError("VLLM_SEQUENTIAL_PROFILE_ARTIFACT_MISSING")
    trace_index = {
        "files": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in trace_files
        ]
    }
    trace_index["sha256"] = canonical_sha256(trace_index)
    trace_path = output_root / "trace-index.json"
    trace_path.write_text(
        json.dumps(trace_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest in cells:
        cell_id = f"{context_id}--round-{manifest['round']}"
        cell_root = output_root / "cells" / cell_id
        manifest["trace"] = {
            "path": "../../trace-index.json",
            "sha256": _sha256(trace_path),
            "bytes": trace_path.stat().st_size,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (cell_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "laguna_dspark_single_request_multiround_capture",
        "context_id": context_id,
        "proposal_invocations": proposal_count,
        "captured_rounds": 3,
        "override_rounds": (
            len(target_feature_overrides) if target_feature_overrides is not None else 0
        ),
        "native_fallback_rounds": (
            max(proposal_count - len(target_feature_overrides), 0)
            if target_feature_overrides is not None
            else 0
        ),
        "allow_native_after_override_exhaustion": target_feature_overrides is not None,
        "response_token_ids": generated,
        "cell_manifest_sha256": [cell["manifest_sha256"] for cell in cells],
        "trace_index_sha256": _sha256(trace_path),
        "server_log_sha256": _sha256(log_path),
        "metrics_before_sha256": hashlib.sha256(metrics_before.encode()).hexdigest(),
        "metrics_after_sha256": hashlib.sha256(metrics_after.encode()).hexdigest(),
    }
    (output_root / "metrics-before.txt").write_text(metrics_before, encoding="utf-8")
    (output_root / "metrics-after.txt").write_text(metrics_after, encoding="utf-8")
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_dflash_capture(
    *,
    output_root: Path,
    prompt: str,
    target_feature_override: Path,
    draft_model: str,
    port: int = 8000,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    raw_capture_root = output_root / "raw"
    profile_root = output_root / "profiles"
    raw_capture_root.mkdir()
    profile_root.mkdir()
    command = server_command(
        DFLASH,
        port=port,
        draft_model=draft_model,
        proposal_tokens=15,
        capture_dflash=True,
    )
    _replace_argument(command, "--max-model-len", "4096")
    _replace_argument(command, "--max-num-seqs", "1")
    command.extend(
        [
            "--enforce-eager",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_DSPARK_CAPTURE_ROOT": str(raw_capture_root),
        "OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE": str(target_feature_override),
        "OPJAX_SPEC_RUN_ID": output_root.name,
    }
    log_path = output_root / "server.log"
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process, log_path=log_path)
            prompt_token_ids = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/tokenize",
                    payload={
                        "model": TARGET_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
            )["tokens"]
            session = "first-proposal"
            (raw_capture_root / "active.json").write_text(
                json.dumps({"session": session}), encoding="utf-8"
            )
            _request(f"http://127.0.0.1:{port}/start_profile", payload={})
            response = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    payload={
                        "model": TARGET_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "chat_template_kwargs": {"enable_thinking": True},
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "seed": 0,
                    },
                    timeout=600,
                )
            )
            _request(f"http://127.0.0.1:{port}/stop_profile", payload={}, timeout=600)
            (raw_capture_root / "active.json").unlink()
            time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    records = _load_records(raw_capture_root / "first-proposal")
    combined = _first(records, "combined_target_feature")
    hidden = _first(records, "draft_backbone_hidden_state")
    logits = _first(records, "base_logits")
    logits = logits.reshape(-1, logits.shape[-1])[:15]
    proposal_tokens = logits.argmax(axis=-1).astype(np.int64)
    boundaries = {
        "combined_target_feature": _save_array(
            output_root,
            "combined_target_feature",
            combined.reshape(-1, combined.shape[-1]),
        ),
        "draft_backbone_hidden_state": _save_array(
            output_root,
            "draft_backbone_hidden_state",
            hidden.reshape(-1, hidden.shape[-1])[:15],
        ),
        "base_logits": _save_array(output_root, "base_logits", logits),
        "proposal_token_ids": _save_array(
            output_root, "proposal_token_ids", proposal_tokens
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "implementation": "vllm_dflash_capture",
        "vllm_build": VLLM_OBSERVED_BUILD,
        "draft_model": str(Path(draft_model).resolve()),
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "response": response,
        "boundaries": boundaries,
        "server_log": {"path": log_path.name, "sha256": _sha256(log_path)},
        "profiles": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(profile_root.rglob("*"))
            if path.is_file()
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_production_profile(
    *,
    output_root: Path,
    prompt: str,
    port: int = 8000,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    profile_root = output_root / "profiles"
    profile_root.mkdir()
    command = server_command(DSPARK, port=port)
    _replace_argument(command, "--max-model-len", "4096")
    _replace_argument(command, "--max-num-seqs", "1")
    command.extend(
        [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_use_gzip": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_with_stack": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_SPEC_RUN_ID": output_root.name,
    }
    log_path = output_root / "server.log"
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process, log_path=log_path)
            request_payload = {
                "model": TARGET_ID,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": True},
                "temperature": 0.0,
                "max_tokens": 256,
                "seed": 0,
            }
            _request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                payload={**request_payload, "max_tokens": 32},
                timeout=600,
            )
            metrics_before = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            _request(f"http://127.0.0.1:{port}/start_profile", payload={})
            started = time.time()
            response = json.loads(
                _request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    payload=request_payload,
                    timeout=600,
                )
            )
            elapsed = time.time() - started
            _request(
                f"http://127.0.0.1:{port}/stop_profile",
                payload={},
                timeout=600,
            )
            metrics_after = _request(
                f"http://127.0.0.1:{port}/metrics", timeout=30
            ).decode()
            time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    for name, value in {
        "metrics-before.prom": metrics_before,
        "metrics-after.prom": metrics_after,
    }.items():
        (output_root / name).write_text(value, encoding="utf-8")
    trace_files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    if not trace_files:
        raise RuntimeError("VLLM_PRODUCTION_PROFILE_ARTIFACT_MISSING")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "implementation": "opjax_vllm_laguna_dspark_production_profile",
        "provenance": {
            "revision": VLLM_OBSERVED_BUILD,
            "source_sha256": _adapter_source_sha256(),
            "target_revision": TARGET_REVISION,
            "draft_revision": DSPARK_REVISION,
            "command": command,
        },
        "request": request_payload,
        "response": response,
        "elapsed_seconds": elapsed,
        "metrics": {
            "before_sha256": _sha256(output_root / "metrics-before.prom"),
            "after_sha256": _sha256(output_root / "metrics-after.prom"),
        },
        "trace_files": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in trace_files
        ],
        "server_log": {
            "path": log_path.name,
            "sha256": _sha256(log_path),
            "bytes": log_path.stat().st_size,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
