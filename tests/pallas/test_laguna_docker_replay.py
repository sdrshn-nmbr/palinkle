from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.laguna_dspark_conformance import canonical_sha256
from opjax.pallas.laguna_speculative import (
    LagunaSpeculativeError,
    ordered_replay_prompt_ids,
    validate_replay_attempt_receipt,
    validate_trained_replay_cells,
)


KNOWN_CELLS = {"plain", "dflash-4", "dspark-4", "dspark-adaptive"}


def test_replay_prompt_order_uses_frozen_prompt_id_contract() -> None:
    assert ordered_replay_prompt_ids(
        [{"prompt_id": "prompt-1"}, {"prompt_id": "prompt-2"}]
    ) == ["prompt-1", "prompt-2"]
    with pytest.raises(LagunaSpeculativeError, match="PROMPT_ID_INVALID"):
        ordered_replay_prompt_ids([{"id": "legacy-id"}])
    with pytest.raises(LagunaSpeculativeError, match="PROMPT_ID_DUPLICATE"):
        ordered_replay_prompt_ids(
            [{"prompt_id": "same"}, {"prompt_id": "same"}]
        )


def test_single_nonplain_cell_requires_deferred_endpoint() -> None:
    validate_trained_replay_cells(
        ["dflash-4"],
        known_cells=KNOWN_CELLS,
        endpoint="http://127.0.0.1:8000",
        defer_summary=True,
    )
    with pytest.raises(LagunaSpeculativeError, match="CELLS_INVALID"):
        validate_trained_replay_cells(
            ["dflash-4"],
            known_cells=KNOWN_CELLS,
            endpoint=None,
            defer_summary=True,
        )


def test_full_matrix_requires_plain() -> None:
    validate_trained_replay_cells(
        ["plain", "dflash-4", "dspark-4"],
        known_cells=KNOWN_CELLS,
        endpoint=None,
        defer_summary=False,
    )
    with pytest.raises(LagunaSpeculativeError, match="CELLS_INVALID"):
        validate_trained_replay_cells(
            ["dflash-4", "dspark-4"],
            known_cells=KNOWN_CELLS,
            endpoint=None,
            defer_summary=False,
        )


def test_attempt_receipt_is_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    payload = {
        "schema_version": 1,
        "kind": "opjax_laguna_gce_replay_attempt",
        "attempt_id": "attempt",
        "declared_gpu": "A100",
        "gpu_count": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.84,
        "kv_cache_memory_bytes": 1610612736,
        "disable_custom_all_reduce": True,
        "image": "image",
        "deployment": {
            "provider": "gcp_compute_engine",
            "id": "vm",
            "zone": "zone",
            "instance_id": "123",
        },
        "container_launcher": {
            "interpreter": "/usr/bin/python3",
            "vllm_launcher": "/usr/local/bin/vllm",
        },
        "tokenizer": {
            "revision": "e9df9a59996d790b94b70f3fef343fe1d9e34bdf",
            "container_path": "/hf/tokenizer",
            "files": {
                name: {"bytes": 1, "sha256": "a" * 64}
                for name in (
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            },
        },
        "measurement_sources": {"driver": "hash"},
    }
    payload["sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    assert validate_replay_attempt_receipt(path)["sha256"] == payload["sha256"]
    payload["instance_id"] = "456"
    path.write_text(json.dumps(payload))
    with pytest.raises(LagunaSpeculativeError, match="RECEIPT_HASH_MISMATCH"):
        validate_replay_attempt_receipt(path)


def test_attempt_receipt_requires_kv_cache_memory(tmp_path: Path) -> None:
    path = tmp_path / "attempt.json"
    payload = {
        "schema_version": 1,
        "kind": "opjax_laguna_gce_replay_attempt",
        "attempt_id": "attempt",
        "declared_gpu": "A100",
        "gpu_count": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.84,
        "disable_custom_all_reduce": True,
        "image": "image",
        "deployment": {
            "provider": "gcp_compute_engine",
            "id": "vm",
            "zone": "zone",
            "instance_id": "123",
        },
        "container_launcher": {
            "interpreter": "/usr/bin/python3",
            "vllm_launcher": "/usr/local/bin/vllm",
        },
        "tokenizer": {
            "revision": "e9df9a59996d790b94b70f3fef343fe1d9e34bdf",
            "container_path": "/hf/tokenizer",
            "files": {
                name: {"bytes": 1, "sha256": "a" * 64}
                for name in (
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            },
        },
        "measurement_sources": {"driver": "hash"},
    }
    payload["sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(LagunaSpeculativeError, match="RECEIPT_INVALID"):
        validate_replay_attempt_receipt(path)
