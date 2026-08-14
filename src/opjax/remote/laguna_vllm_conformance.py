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
    DSPARK,
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _wait_ready(
    port: int, process: subprocess.Popen[bytes], *, log_path: Path
) -> None:
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


def _one(records: dict[str, list[tuple[dict[str, Any], np.ndarray]]], name: str) -> np.ndarray:
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


def _canonicalize_capture(
    raw_root: Path, static_root: Path, output_root: Path
) -> dict[str, dict[str, Any]]:
    records = _load_records(raw_root)
    static_records = _load_records(static_root)
    combined = _first(records, "combined_target_feature")
    hidden = _first(records, "draft_backbone_hidden_state")
    base_logits = _first(records, "base_logits")
    bias_values = [value for _record, value in records.get("markov_bias", [])][:15]
    input_values = [
        value for _record, value in records.get("markov_input_token_ids", [])
    ][:15]
    embedding_values = [
        value for _record, value in records.get("markov_embedding", [])
    ][:15]
    if len(bias_values) != 15 or len(input_values) != 15 or len(embedding_values) != 15:
        raise RuntimeError(
            "VLLM_MARKOV_CAPTURE_CARDINALITY:"
            f"bias={len(bias_values)}:tokens={len(input_values)}:"
            f"embeddings={len(embedding_values)}"
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
    corrected = base_logits + markov_bias
    proposal_tokens = np.argmax(corrected, axis=-1).astype(np.int64)
    previous_tokens = np.concatenate([value.reshape(-1) for value in input_values])
    if not np.array_equal(previous_tokens[1:], proposal_tokens[:-1]):
        raise RuntimeError("VLLM_MARKOV_CHAIN_INCONSISTENT")
    markov_embeddings = np.concatenate(
        [value.reshape(-1, value.shape[-1]) for value in embedding_values], axis=0
    )
    confidence_weight = _one(
        static_records, "confidence_head_proj_weight"
    ).astype(np.float32)
    confidence_bias = _one(
        static_records, "confidence_head_proj_bias"
    ).astype(np.float32)
    confidence_features = np.concatenate(
        [hidden.reshape(-1, hidden.shape[-1]), markov_embeddings], axis=-1
    ).astype(np.float32)
    confidence = (
        confidence_features @ confidence_weight.reshape(1, -1).T
        + confidence_bias.reshape(1, -1)
    ).reshape(-1)
    if confidence.shape[0] != 15:
        raise RuntimeError(f"VLLM_CONFIDENCE_POSITIONS:{confidence.shape}")
    arrays = {
        "draft_input_ids": _first(records, "draft_input_ids").reshape(-1),
        "draft_positions": _first(records, "draft_positions").reshape(-1),
        "draft_input_embeddings": _first(
            records, "draft_input_embeddings"
        ).reshape(-1, hidden.shape[-1]),
        **{
            f"draft_layer_{layer_id}_output": _first(
                records, f"draft_layer_{layer_id}_output"
            ).reshape(-1, hidden.shape[-1])
            for layer_id in range(5)
        },
        **{
            f"{name}_0": _first(records, name)
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
        "confidence_logits": confidence,
        "proposal_token_ids": proposal_tokens,
    }
    return {name: _save_array(output_root, name, value) for name, value in arrays.items()}


def run_capture(
    *,
    output_root: Path,
    prompt: str,
    target_feature_override: Path,
    port: int = 8000,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    raw_capture_root = output_root / "raw"
    profile_root = output_root / "profiles"
    raw_capture_root.mkdir()
    profile_root.mkdir()
    command = server_command(DSPARK, port=port)
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
        "provenance": {
            "revision": VLLM_OBSERVED_BUILD,
            "source_sha256": _adapter_source_sha256(),
            "target_revision": TARGET_REVISION,
            "draft_revision": DSPARK_REVISION,
            "command": command,
            "target_feature_override": {
                "path": str(target_feature_override),
                "sha256": _sha256(target_feature_override),
            },
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
