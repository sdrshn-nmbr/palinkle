from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opjax.pallas.laguna_dspark_conformance import BOUNDARY_ORDER, ConformanceError
from opjax.pallas.laguna_dspark_multiround import (
    DEEPSPEC_REVISION,
    DEEPSPEC_SOURCE_SHA256,
    DRAFT_REVISION,
    EXTENDED_ADAPTER_NAMES,
    EXTENDED_BOUNDARIES,
    TARGET_REVISION,
    VLLM_REVISION,
    VLLM_SOURCE_SHA256,
    build_attention_cache_parity_report,
    build_contexts,
    build_final_report,
    build_multiround_report,
    build_sequential_report,
    mutation_controls_pass,
    _explicit_window_attention,
    _compare_downstream,
    _compare_extended,
    _stable_report_values,
    validate_multiround_report,
    validate_final_report,
    validate_attention_cache_parity_report,
)
from opjax.pallas.laguna_speculative import canonical_sha256
from opjax.remote.laguna_vllm_conformance import (
    _append_attention_backend,
    _bf16_add,
    _load_attention_runtime_metadata,
)
from opjax.remote.laguna_multiround_conformance_modal import _prepare_capture_output
from opjax.remote.laguna_dspark_capture import (
    logical_context_kv,
    validate_single_request_attention_layout,
)


def test_attention_backend_probe_is_explicit_and_fail_closed() -> None:
    command = [
        "vllm",
        "serve",
        "--speculative-config",
        '{"method":"dspark","model":"draft"}',
    ]
    _append_attention_backend(command, "FLEX_ATTENTION")
    assert json.loads(command[-1]) == {
        "attention_backend": "FLEX_ATTENTION",
        "method": "dspark",
        "model": "draft",
    }

    with pytest.raises(
        ValueError,
        match="VLLM_MULTIRound_ATTENTION_BACKEND_INVALID:TORCH_SDPA",
    ):
        _append_attention_backend([], "TORCH_SDPA")


def test_attention_backend_probe_validates_resolved_runtime(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    metadata = {
        "layers": [
            {
                "layer": index,
                "backend_name": "FLEX_ATTENTION",
                "backend_class": "backend.FlexAttentionBackend",
                "implementation_class": "backend.FlexAttentionImpl",
                "wrapper_sliding_window": None,
                "implementation_sliding_window": [511, 0],
                "outer_sliding_window": 512,
            }
            for index in range(5)
        ]
    }
    (static / "attention-runtime.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    observed = _load_attention_runtime_metadata(
        tmp_path, requested_backend="FLEX_ATTENTION"
    )
    assert observed["layers"] == metadata["layers"]
    assert observed["path"] == "static/attention-runtime.json"

    with pytest.raises(
        RuntimeError, match="VLLM_ATTENTION_BACKEND_RESOLUTION_MISMATCH"
    ):
        _load_attention_runtime_metadata(tmp_path, requested_backend="FLASH_ATTN")


def test_logical_context_kv_follows_block_table_order() -> None:
    backing = torch.zeros((3, 2, 2, 4), dtype=torch.float32)
    cache = backing.permute(0, 2, 1, 3)
    assert not cache.is_contiguous()
    for block in range(3):
        for offset in range(2):
            token = block * 2 + offset
            for head in range(2):
                cache[block, head, offset] = torch.tensor(
                    [token + head, token + 0.25, token + 10, token + 10.25]
                )
    keys, values = logical_context_kv(
        cache,
        torch.tensor([2, 0, 1]),
        sequence_length=7,
        query_length=2,
    )
    assert keys[:, 0, 0].tolist() == [4, 5, 0, 1, 2]
    assert values[:, 0, 0].tolist() == [14, 15, 10, 11, 12]


def test_single_request_attention_layout_binds_query_slots() -> None:
    context_length = validate_single_request_attention_layout(
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([[2, 0, 1]], dtype=torch.int32),
        torch.tensor([1, 2, 3], dtype=torch.int64),
        query_length=3,
        block_size=2,
    )
    assert context_length == 3

    with pytest.raises(RuntimeError, match="LAGUNA_CAPTURE_SLOT_MAPPING_INVALID"):
        validate_single_request_attention_layout(
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([6], dtype=torch.int32),
            torch.tensor([[2, 0, 1]], dtype=torch.int32),
            torch.tensor([0, 2, 3], dtype=torch.int64),
            query_length=3,
            block_size=2,
        )


def test_explicit_attention_reference_enforces_causal_sliding_window() -> None:
    query = np.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
    context_key = np.array(
        [[[8.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32
    )
    context_value = np.array(
        [[[100.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]], dtype=np.float32
    )
    query_key = np.zeros((2, 1, 2), dtype=np.float32)
    query_value = np.array([[[4.0, 0.0]], [[5.0, 0.0]]], dtype=np.float32)

    windowed, ranges = _explicit_window_attention(
        query=query,
        context_key=context_key,
        context_value=context_value,
        query_key=query_key,
        query_value=query_value,
        window_tokens=2,
    )
    full, _ = _explicit_window_attention(
        query=query,
        context_key=context_key,
        context_value=context_value,
        query_key=query_key,
        query_value=query_value,
        window_tokens=None,
    )

    assert ranges == [(2, 4), (3, 5)]
    assert not np.allclose(windowed, full)


def test_explicit_attention_reference_rejects_incompatible_gqa_layout() -> None:
    with pytest.raises(ConformanceError, match="ATTENTION_REFERENCE_LAYOUT_INVALID"):
        _explicit_window_attention(
            query=np.zeros((1, 3, 2), dtype=np.float32),
            context_key=np.zeros((1, 2, 2), dtype=np.float32),
            context_value=np.zeros((1, 2, 2), dtype=np.float32),
            query_key=np.zeros((1, 2, 2), dtype=np.float32),
            query_value=np.zeros((1, 2, 2), dtype=np.float32),
            window_tokens=2,
        )


def test_report_floats_are_stable_across_last_bit_reduction_drift() -> None:
    left = {"cosine_similarity": 1.0, "nested": [0.9999999999999999]}
    right = {"cosine_similarity": 1.0000000000000002, "nested": [1.0]}
    assert _stable_report_values(left) == _stable_report_values(right)


def test_bf16_logit_reconstruction_preserves_runtime_argmax() -> None:
    base = np.array([[12.75, 12.9375]], dtype=np.float32)
    bias = np.array([[0.1103515625, -0.0458984375]], dtype=np.float32)
    float32_choice = int(np.argmax(base + bias, axis=-1)[0])
    runtime_choice = int(np.argmax(_bf16_add(base, bias), axis=-1)[0])
    assert float32_choice == 1
    assert runtime_choice == 0


def test_extended_integer_boundaries_require_exact_match(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "tokens-32--round-0"
    adapter_root = tmp_path / "adapter" / "tokens-32--round-0"
    source = _manifest(source_root, list(range(32)), 1000)
    adapter = _manifest(adapter_root, list(range(32)), 1000, lane="native")
    positions = np.load(
        adapter_root / adapter["boundaries"]["draft_positions"]["path"],
        allow_pickle=False,
    )
    positions[0] += 1
    np.save(adapter_root / "draft_positions.npy", positions)
    adapter["boundaries"]["draft_positions"] = _artifact(
        adapter_root, "draft_positions", positions
    )
    comparison = _compare_extended(
        source_root=source_root,
        source=source,
        adapter_root=adapter_root,
        adapter=adapter,
        processed_start=0,
    )
    assert comparison["draft_positions"]["passed"] is False


def test_extended_integer_boundaries_allow_exact_values_across_integer_dtypes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "tokens-32--round-0"
    adapter_root = tmp_path / "adapter" / "tokens-32--round-0"
    source = _manifest(source_root, list(range(32)), 1000)
    adapter = _manifest(adapter_root, list(range(32)), 1000, lane="injected")
    ids = np.load(
        adapter_root / adapter["boundaries"]["draft_input_ids"]["path"],
        allow_pickle=False,
    ).astype(np.int32)
    adapter["boundaries"]["draft_input_ids"] = _artifact(
        adapter_root, "draft_input_ids", ids
    )

    comparison = _compare_extended(
        source_root=source_root,
        source=source,
        adapter_root=adapter_root,
        adapter=adapter,
        processed_start=0,
    )
    assert comparison["draft_input_ids"]["passed"] is True
    assert comparison["draft_input_ids"]["dtype_match"] is False


def test_extended_boundaries_canonicalize_real_source_and_adapter_layouts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "tokens-32--round-0"
    adapter_root = tmp_path / "adapter" / "tokens-32--round-0"
    source = _manifest(source_root, list(range(32)), 1000)
    adapter = _manifest(adapter_root, list(range(32)), 1000, lane="injected")
    for source_name in (
        "layer0_context_k_before_rope",
        "layer0_context_v",
    ):
        adapter_name = EXTENDED_ADAPTER_NAMES[source_name]
        value = np.load(
            adapter_root / adapter["boundaries"][adapter_name]["path"],
            allow_pickle=False,
        )[16:]
        adapter["boundaries"][adapter_name] = _artifact(
            adapter_root, adapter_name, value
        )

    comparison = _compare_extended(
        source_root=source_root,
        source=source,
        adapter_root=adapter_root,
        adapter=adapter,
        processed_start=16,
    )
    assert set(comparison) == set(EXTENDED_BOUNDARIES)
    assert all(item["passed"] for item in comparison.values())


def test_downstream_comparison_masks_rows_after_first_proposal_divergence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "tokens-32--round-0"
    adapter_root = tmp_path / "adapter" / "tokens-32--round-0"
    source = _manifest(source_root, list(range(32)), 1000)
    adapter = _manifest(adapter_root, list(range(32)), 1000, lane="injected")
    proposals = np.load(
        adapter_root / adapter["boundaries"]["proposal_token_ids"]["path"],
        allow_pickle=False,
    )
    proposals[2] += 1
    adapter["boundaries"]["proposal_token_ids"] = _artifact(
        adapter_root, "proposal_token_ids", proposals
    )
    for name in ("markov_bias", "corrected_logits", "confidence_logits"):
        value = np.load(
            adapter_root / adapter["boundaries"][name]["path"], allow_pickle=False
        )
        value[3:] += 1000
        adapter["boundaries"][name] = _artifact(adapter_root, name, value)

    comparison = _compare_downstream(
        source_root=source_root,
        source=source,
        adapter_root=adapter_root,
        adapter=adapter,
        processed_start=0,
    )
    assert comparison["comparable_proposal_prefix_length"] == 2
    assert comparison["causally_comparable_rows"] == 3
    assert comparison["first_proposal_divergence"] == 2
    assert comparison["boundaries"]["proposal_token_ids"]["passed"] is False
    assert comparison["boundaries"]["markov_bias"]["passed"] is True


def test_capture_resume_archives_incomplete_attempt_and_telemetry(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "sequential-native" / "tokens-32"
    output_root.mkdir(parents=True)
    (output_root / "partial.txt").write_text("partial", encoding="utf-8")
    telemetry = tmp_path / "sequential-native-tokens-32-gpu.csv"
    telemetry.write_text("gpu", encoding="utf-8")

    assert (
        _prepare_capture_output(output_root, telemetry_path=telemetry) is None
    )
    assert not output_root.exists()
    archives = list(
        (tmp_path / "attempts" / "sequential-native" / "tokens-32").glob("*")
    )
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert (
        archive / "sequential-native-tokens-32-gpu.csv"
    ).read_text(encoding="utf-8") == "gpu"
    manifest = json.loads(
        (archive / "archive-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["reason"] == "incomplete_capture_resumed"
    assert manifest["original_path"] == "sequential-native/tokens-32"


def _artifact(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / f"{name}.npy"
    np.save(path, value)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _manifest(
    root: Path,
    tokens: list[int],
    anchor: int,
    *,
    lane: str | None = None,
    source_manifest_sha256: str | None = None,
    target_feature_mode: str | None = None,
) -> dict[str, object]:
    root.mkdir(parents=True)
    def value_for(name: str) -> np.ndarray:
        canonical_name = next(
            (
                source_name
                for source_name, adapter_name in EXTENDED_ADAPTER_NAMES.items()
                if adapter_name == name
            ),
            name,
        )
        if name == "proposal_token_ids":
            return np.arange(15, dtype=np.int64)
        if canonical_name in {"draft_input_ids", "draft_positions"}:
            width = 16 if lane is None else 15
            value = np.arange(width, dtype=np.int64)
            return value.reshape(1, width) if lane is None else value
        if canonical_name == "draft_input_embeddings":
            width = 16 if lane is None else 15
            value = np.arange(width * 4, dtype=np.float32).reshape(width, 4) + anchor
            return value.reshape(1, width, 4) if lane is None else value
        if name == "combined_target_feature":
            return np.broadcast_to(
                np.arange(4, dtype=np.float32) + anchor, (len(tokens), 4)
            ).copy()
        if canonical_name in {
            "layer0_context_k_before_rope",
            "layer0_context_v",
        }:
            value = np.broadcast_to(
                np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4) + anchor,
                (1, len(tokens), 1, 4),
            ).copy()
            return value if lane is None else value[0]
        if canonical_name in {
            "layer0_query_q_after_rope",
            "layer0_query_k_after_rope",
            "layer0_query_v",
        }:
            value = np.arange(64, dtype=np.float32).reshape(1, 16, 1, 4) + anchor
            return value if lane is None else value[0, :15].reshape(15, 4)
        if canonical_name.startswith("draft_layer_"):
            value = np.arange(60, dtype=np.float32).reshape(15, 4) + anchor
            return value.reshape(1, 15, 4) if lane is None else value
        if name in {"layer0_context_k_before_rope_0", "layer0_context_v_0"}:
            return np.broadcast_to(
                np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4) + anchor,
                (1, len(tokens), 1, 4),
            ).copy()
        return np.arange(60, dtype=np.float32).reshape(15, 4) + anchor

    extended_names = (
        set(EXTENDED_BOUNDARIES)
        if lane is None
        else {EXTENDED_ADAPTER_NAMES.get(name, name) for name in EXTENDED_BOUNDARIES}
    )
    boundaries = {
        name: _artifact(root, name, value_for(name))
        for name in set(BOUNDARY_ORDER) | extended_names
    }
    boundaries["raw_target_features"] = _artifact(
        root,
        "raw_target_features",
        np.broadcast_to(
            np.arange(20, dtype=np.float32).reshape(1, 1, 20) + anchor,
            (1, len(tokens), 20),
        ).copy(),
    )
    boundaries["anchor_token_id"] = _artifact(
        root, "anchor_token_id", np.array([anchor], dtype=np.int64)
    )
    trace = root / "trace.json"
    trace.write_text(
        '{"traceEvents":[{"name":"cudaLaunchKernel"}]}', encoding="utf-8"
    )
    cell_id = root.name
    context_id, round_value = cell_id.rsplit("--round-", 1)
    provenance = {
        "revision": DEEPSPEC_REVISION if lane is None else VLLM_REVISION,
        "source_sha256": (
            DEEPSPEC_SOURCE_SHA256 if lane is None else VLLM_SOURCE_SHA256
        ),
        "target_revision": TARGET_REVISION,
        "draft_revision": DRAFT_REVISION,
    }
    if lane is not None:
        provenance.update(
            lane=lane, source_manifest_sha256=source_manifest_sha256
        )
    if target_feature_mode is not None:
        provenance["target_feature_mode"] = target_feature_mode
    manifest: dict[str, object] = {
        "schema_version": 1,
        "implementation": root.parent.name,
        "provenance": provenance,
        "context_id": context_id,
        "round": int(round_value),
        "input_mode": "token_ids",
        "prompt_token_ids": tokens,
        "response": {"choices": [{"token_ids": [anchor]}]},
        "processed_token_start": 0,
        "expected_processed_token_start": None if lane == "native" else 0,
        "boundaries": boundaries,
        "trace": {
            "path": trace.name,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "bytes": trace.stat().st_size,
        },
        "mutation_controls": {"markov_matrix_swap": {"detected": True}},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _attention_cache_fixture(
    root: Path, *, source_discontinuity: bool = True
) -> tuple[Path, Path, Path]:
    source_cells = root / "source"
    cached_cells = root / "cached"
    fresh_cells = root / "fresh"
    lengths = (511, 512, 513)
    cached_starts = (0, 480, 496)
    for round_index, length in enumerate(lengths):
        cell_id = f"tokens-511--round-{round_index}"
        tokens = list(range(length))
        source_root = source_cells / cell_id
        source = _manifest(source_root, tokens, 1000 + round_index)
        source_feature_value = (
            float(round_index > 0 and source_discontinuity)
        )
        source["boundaries"]["combined_target_feature"] = _artifact(
            source_root,
            "combined_target_feature",
            np.full((length, 4), source_feature_value, dtype=np.float32),
        )
        context_key = np.zeros((length, 8, 128), dtype=np.float32)
        context_value = np.zeros((length, 8, 128), dtype=np.float32)
        context_value[0, :, 0] = 100
        source["boundaries"]["layer0_context_k_after_rope"] = _artifact(
            source_root, "layer0_context_k_after_rope", context_key[None]
        )
        source["boundaries"]["layer0_context_v"] = _artifact(
            source_root, "layer0_context_v", context_value[None]
        )
        _write_manifest(source_root, source)

        for lane, lane_root, processed_start in (
            ("cached", cached_cells / cell_id, cached_starts[round_index]),
            ("fresh", fresh_cells / cell_id, 0),
        ):
            adapter = _manifest(
                lane_root,
                tokens,
                1000 + round_index,
                lane="injected",
                source_manifest_sha256=source["manifest_sha256"],
            )
            adapter["processed_token_start"] = processed_start
            adapter["expected_processed_token_start"] = processed_start
            adapter["provenance"]["prefix_caching"] = lane == "cached"
            adapter["provenance"]["attention_runtime"] = {
                "layers": [
                    {
                        "layer": layer_index,
                        "backend_name": "FLASH_ATTN",
                        "wrapper_sliding_window": None,
                        "implementation_sliding_window": [511, 0],
                        "outer_sliding_window": 512,
                    }
                    for layer_index in range(5)
                ]
            }
            query = np.zeros((15, 64, 128), dtype=np.float32)
            query_key = np.zeros((15, 8, 128), dtype=np.float32)
            query_value = np.zeros((15, 8, 128), dtype=np.float32)
            observed, _ = _explicit_window_attention(
                query=query,
                context_key=context_key,
                context_value=context_value,
                query_key=query_key,
                query_value=query_value,
                window_tokens=512,
            )
            values = {
                "layer0_query_q_after_rope_0": query.reshape(15, -1),
                "layer0_query_k_after_rope_0": query_key.reshape(15, -1),
                "layer0_query_v_0": query_value.reshape(15, -1),
                "layer0_logical_context_k_0": context_key,
                "layer0_logical_context_v_0": context_value,
                "layer0_raw_attention_output_0": observed.reshape(15, -1),
            }
            for name, value in values.items():
                adapter["boundaries"][name] = _artifact(lane_root, name, value)
            _write_manifest(lane_root, adapter)
    return source_cells, cached_cells, fresh_cells


def test_attention_cache_report_recomputes_runtime_semantics(tmp_path: Path) -> None:
    source, cached, fresh = _attention_cache_fixture(tmp_path)
    report = build_attention_cache_parity_report(
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    validate_attention_cache_parity_report(
        report,
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    assert report["result"] == {
        "attention_cache_semantics_passed": True,
        "fresh_prefix_parity_passed": True,
        "deep_spec_source_is_stateful": False,
        "transition_blocks_statefully_verified": False,
        "source_state_mismatch_observed": True,
        "runtime_fix_required": False,
        "diagnosis": "source_oracle_state_mismatch",
    }
    report["result"]["runtime_fix_required"] = True
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    with pytest.raises(
        ConformanceError, match="ATTENTION_CACHE_REPORT_RECOMPUTATION_MISMATCH"
    ):
        validate_attention_cache_parity_report(
            report,
            source_cells_root=source,
            cached_cells_root=cached,
            fresh_cells_root=fresh,
        )


def test_attention_cache_report_detects_prefix_and_output_corruption(
    tmp_path: Path,
) -> None:
    source, cached, fresh = _attention_cache_fixture(tmp_path)
    cached_root = cached / "tokens-511--round-1"
    cached_manifest = json.loads((cached_root / "manifest.json").read_text())
    key = np.load(cached_root / "layer0_logical_context_k_0.npy")
    key[0, 0, 0] = 1
    cached_manifest["boundaries"]["layer0_logical_context_k_0"] = _artifact(
        cached_root, "layer0_logical_context_k_0", key
    )
    _write_manifest(cached_root, cached_manifest)
    report = build_attention_cache_parity_report(
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    assert report["result"]["attention_cache_semantics_passed"] is False

    fresh_root = fresh / "tokens-511--round-1"
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text())
    proposals = np.load(fresh_root / "proposal_token_ids.npy")
    proposals[0] += 1
    fresh_manifest["boundaries"]["proposal_token_ids"] = _artifact(
        fresh_root, "proposal_token_ids", proposals
    )
    _write_manifest(fresh_root, fresh_manifest)
    report = build_attention_cache_parity_report(
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    assert report["result"]["fresh_prefix_parity_passed"] is False

    cached_root = cached / "tokens-511--round-0"
    cached_manifest = json.loads((cached_root / "manifest.json").read_text())
    output = np.load(cached_root / "layer0_raw_attention_output_0.npy")
    output += 10
    cached_manifest["boundaries"]["layer0_raw_attention_output_0"] = _artifact(
        cached_root, "layer0_raw_attention_output_0", output
    )
    _write_manifest(cached_root, cached_manifest)
    report = build_attention_cache_parity_report(
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    assert report["cached_runtime"][0]["attention_reference"][
        "windowed_passed"
    ] is False


def test_attention_cache_report_does_not_invent_source_state_mismatch(
    tmp_path: Path,
) -> None:
    source, cached, fresh = _attention_cache_fixture(
        tmp_path, source_discontinuity=False
    )
    report = build_attention_cache_parity_report(
        source_cells_root=source,
        cached_cells_root=cached,
        fresh_cells_root=fresh,
    )
    assert report["result"]["source_state_mismatch_observed"] is False
    assert report["result"]["diagnosis"] == "attention_cache_parity_cleared"


@pytest.mark.parametrize("mutation", ["lane", "prefix_caching", "backend"])
def test_attention_cache_report_rejects_wrong_runtime_contract(
    tmp_path: Path, mutation: str
) -> None:
    source, cached, fresh = _attention_cache_fixture(tmp_path)
    root = cached / "tokens-511--round-0"
    manifest = json.loads((root / "manifest.json").read_text())
    if mutation == "lane":
        manifest["provenance"]["lane"] = "native"
    elif mutation == "prefix_caching":
        manifest["provenance"]["prefix_caching"] = False
    else:
        manifest["provenance"]["attention_runtime"]["layers"][0][
            "backend_name"
        ] = "FLEX_ATTENTION"
    _write_manifest(root, manifest)

    with pytest.raises(ConformanceError, match="ATTENTION_CACHE_(LANE|RUNTIME)"):
        build_attention_cache_parity_report(
            source_cells_root=source,
            cached_cells_root=cached,
            fresh_cells_root=fresh,
        )


def _sequential_matrix(root: Path) -> None:
    matrix = json.loads((root / "matrix.json").read_text())
    for context_id, base in matrix["contexts"].items():
        generated = [1000, 1001, 1002, *range(61)]
        hashes = {"native": [], "injected": []}
        for round_index in range(3):
            cell = f"{context_id}--round-{round_index}"
            committed = [*base, *generated[:round_index]]
            _manifest(
                root / "sequential-source" / context_id / "cells" / cell,
                committed,
                generated[round_index],
            )
            source_manifest = json.loads(
                (
                    root
                    / "sequential-source"
                    / context_id
                    / "cells"
                    / cell
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            source_root = root / "sequential-source" / context_id / "cells" / cell
            source_features = np.load(
                source_root
                / source_manifest["boundaries"]["raw_target_features"]["path"],
                allow_pickle=False,
            )
            override_path = (
                root
                / "sequential-overrides"
                / context_id
                / f"round-{round_index}.npy"
            )
            override_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(override_path, source_features)
            for lane, mode in (("native", "live_vllm"), ("injected", "source_override")):
                manifest = _manifest(
                    root / f"sequential-{lane}" / context_id / "cells" / cell,
                    committed,
                    generated[round_index],
                    lane=lane,
                    target_feature_mode=mode,
                )
                manifest["response_token_ids"] = generated
                manifest["manifest_sha256"] = canonical_sha256(
                    {key: value for key, value in manifest.items() if key != "manifest_sha256"}
                )
                manifest_path = (
                    root
                    / f"sequential-{lane}"
                    / context_id
                    / "cells"
                    / cell
                    / "manifest.json"
                )
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                hashes[lane].append(manifest["manifest_sha256"])
        (root / f"sequential-source-{context_id}-gpu.csv").write_text(
            "timestamp,utilization.gpu\nnow,1\n", encoding="utf-8"
        )
        for lane in hashes:
            run_root = root / f"sequential-{lane}" / context_id
            before = run_root / "metrics-before.txt"
            after = run_root / "metrics-after.txt"
            log = run_root / "server.log"
            before.write_text("before\n", encoding="utf-8")
            after.write_text("after\n", encoding="utf-8")
            log.write_text("server\n", encoding="utf-8")
            summary = {
                "schema_version": 1,
                "kind": "test",
                "context_id": context_id,
                "proposal_invocations": 4,
                "captured_rounds": 3,
                "response_token_ids": generated,
                "cell_manifest_sha256": hashes[lane],
                "trace_index_sha256": "a" * 64,
                "server_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "metrics_before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
                "metrics_after_sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
            }
            summary["summary_sha256"] = canonical_sha256(summary)
            (run_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / f"sequential-{lane}-{context_id}-gpu.csv").write_text(
                "timestamp,utilization.gpu\nnow,1\n", encoding="utf-8"
            )


def _matrix(root: Path) -> None:
    contexts = {
        "tokens-32": list(range(32)),
        "tokens-511": list(range(511)),
        "tokens-513": list(range(513)),
        "long-agent-prefix": list(range(1024)),
    }
    (root / "matrix.json").write_text(
        json.dumps({"schema_version": 1, "contexts": contexts}), encoding="utf-8"
    )
    for context_id, base in contexts.items():
        committed = list(base)
        cell_hashes = {"injected": [], "native": []}
        for round_index in range(3):
            cell = f"{context_id}--round-{round_index}"
            anchor = 1000 + round_index
            source = _manifest(root / "source" / cell, committed, anchor)
            injected = _manifest(
                root / "injected" / context_id / "cells" / cell,
                committed,
                anchor,
                lane="injected",
                source_manifest_sha256=source["manifest_sha256"],
            )
            native = _manifest(
                root / "native" / context_id / "cells" / cell,
                committed,
                anchor,
                lane="native",
                source_manifest_sha256=source["manifest_sha256"],
            )
            cell_hashes["injected"].append(injected["manifest_sha256"])
            cell_hashes["native"].append(native["manifest_sha256"])
            committed.append(anchor)
        (root / f"source-{context_id}-gpu.csv").write_text(
            "timestamp,utilization.gpu\nnow,1\n", encoding="utf-8"
        )
        for lane in ("injected", "native"):
            run_root = root / lane / context_id
            before = run_root / "metrics-before.txt"
            after = run_root / "metrics-after.txt"
            log = run_root / "server.log"
            before.write_text("before\n", encoding="utf-8")
            after.write_text("after\n", encoding="utf-8")
            log.write_text("server\n", encoding="utf-8")
            summary = {
                "schema_version": 1,
                "kind": "test",
                "context_id": context_id,
                "lane": lane,
                "command": ["serve"],
                "cells": cell_hashes[lane],
                "metrics_before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
                "metrics_after_sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
                "trace_index_sha256": "a" * 64,
                "server_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
            summary["summary_sha256"] = canonical_sha256(summary)
            (run_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / f"{lane}-{context_id}-gpu.csv").write_text(
                "timestamp,utilization.gpu\nnow,1\n", encoding="utf-8"
            )


def test_contexts_have_exact_boundary_lengths() -> None:
    contexts = build_contexts(list(range(1100)))
    assert {name: len(value) for name, value in contexts.items()} == {
        "tokens-32": 32,
        "tokens-511": 511,
        "tokens-513": 513,
        "long-agent-prefix": 1100,
    }


def test_multiround_recomputes_all_cells_and_detects_mutations(tmp_path: Path) -> None:
    _matrix(tmp_path)
    report = build_multiround_report(tmp_path)
    validate_multiround_report(report, root=tmp_path)
    assert report["injected_passed"] is True
    assert len(report["cells"]) == 12
    assert all(mutation_controls_pass(tmp_path).values())


def test_semantically_equivalent_layer_outputs_remain_conformance_gates(
    tmp_path: Path,
) -> None:
    _matrix(tmp_path)
    context_id = "tokens-32"
    cell_root = (
        tmp_path
        / "injected"
        / context_id
        / "cells"
        / f"{context_id}--round-0"
    )
    path = cell_root / "draft_layer_0_output.npy"
    values = np.load(path, allow_pickle=False)
    np.save(path, values + 100)
    manifest_path = cell_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boundaries"]["draft_layer_0_output"] = _artifact(
        cell_root, "draft_layer_0_output", values + 100
    )
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rebind_forced_summary(tmp_path, "injected", context_id)

    report = build_multiround_report(tmp_path)
    lane = next(
        cell["lanes"]["injected"]
        for cell in report["cells"]
        if cell["cell_id"] == f"{context_id}--round-0"
    )
    diagnostic = lane["extended_boundaries"]["draft_layer_0_output"]
    assert diagnostic["comparable"] is True
    assert diagnostic["passed"] is False
    assert lane["extended_comparable_passed"] is False
    assert lane["functional_passed"] is False


def test_multiround_rejects_rehashed_numeric_drift(tmp_path: Path) -> None:
    _matrix(tmp_path)
    report = build_multiround_report(tmp_path)
    path = (
        tmp_path
        / "injected"
        / "tokens-32"
        / "cells"
        / "tokens-32--round-0"
        / "base_logits.npy"
    )
    values = np.load(path)
    np.save(path, values + 100)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["boundaries"]["base_logits"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConformanceError, match="RECOMPUTATION|FAILED|SUMMARY_INVALID"):
        validate_multiround_report(report, root=tmp_path)


def test_multiround_rejects_unpinned_runtime_provenance(tmp_path: Path) -> None:
    _matrix(tmp_path)
    path = tmp_path / "source" / "tokens-32--round-0" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["revision"] = "unreviewed-runtime"
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConformanceError, match="MULTIROUND_PROVENANCE_INVALID"):
        build_multiround_report(tmp_path)


def test_trace_cache_rejects_same_size_content_mutation(tmp_path: Path) -> None:
    _matrix(tmp_path)
    build_multiround_report(tmp_path)
    trace = tmp_path / "source" / "tokens-32--round-0" / "trace.json"
    original = trace.read_text(encoding="utf-8")
    mutated = original.replace("cudaLaunchKernel", "cudaLaunchKerneX")
    assert len(mutated) == len(original)
    trace.write_text(mutated, encoding="utf-8")

    with pytest.raises(ConformanceError, match="MULTIROUND_TRACE_INVALID"):
        build_multiround_report(tmp_path)


def test_sequential_report_binds_three_true_rounds(tmp_path: Path) -> None:
    _matrix(tmp_path)
    _sequential_matrix(tmp_path)
    report = build_sequential_report(tmp_path)
    assert len(report["cells"]) == 12
    assert report["injected_passed"] is True
    assert report["native_passed"] is True
    assert report["response_token_ids_match"] is True


def test_sequential_report_rejects_injected_cache_start_drift(tmp_path: Path) -> None:
    _matrix(tmp_path)
    _sequential_matrix(tmp_path)
    context_id = "tokens-32"
    cell_id = f"{context_id}--round-0"
    manifest_path = (
        tmp_path
        / "sequential-injected"
        / context_id
        / "cells"
        / cell_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["processed_token_start"] = 1
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rebind_sequential_summary(tmp_path, "injected", context_id)

    with pytest.raises(ConformanceError, match="SEQUENTIAL_CACHE_START_MISMATCH"):
        build_sequential_report(tmp_path)


def test_sequential_report_rejects_override_drift(tmp_path: Path) -> None:
    _matrix(tmp_path)
    _sequential_matrix(tmp_path)
    override = tmp_path / "sequential-overrides" / "tokens-32" / "round-0.npy"
    values = np.load(override, allow_pickle=False)
    np.save(override, values + 1)

    with pytest.raises(ConformanceError, match="SEQUENTIAL_OVERRIDE_INVALID"):
        build_sequential_report(tmp_path)


def test_final_report_recomputes_both_lanes(tmp_path: Path) -> None:
    _matrix(tmp_path)
    _sequential_matrix(tmp_path)
    report = build_final_report(tmp_path)
    validate_final_report(report, root=tmp_path)
    report["sequential"]["response_token_ids_match"] = False
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    with pytest.raises(ConformanceError, match="FINAL_REPORT_RECOMPUTATION"):
        validate_final_report(report, root=tmp_path)


def _rebind_forced_summary(root: Path, lane: str, context_id: str) -> None:
    run_root = root / lane / context_id
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cells"] = [
        json.loads(
            (run_root / "cells" / f"{context_id}--round-{index}" / "manifest.json").read_text()
        )["manifest_sha256"]
        for index in range(3)
    ]
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _rebind_sequential_summary(root: Path, lane: str, context_id: str) -> None:
    run_root = root / f"sequential-{lane}" / context_id
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cell_manifest_sha256"] = [
        json.loads(
            (
                run_root
                / "cells"
                / f"{context_id}--round-{index}"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )["manifest_sha256"]
        for index in range(3)
    ]
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_artifact_probe_rejects_stale_cache_metadata(tmp_path: Path) -> None:
    _matrix(tmp_path)
    context_id = "tokens-32"
    manifest_path = (
        tmp_path
        / "injected"
        / context_id
        / "cells"
        / f"{context_id}--round-1"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["expected_processed_token_start"] = 16
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rebind_forced_summary(tmp_path, "injected", context_id)
    with pytest.raises(ConformanceError, match="CACHE_START_MISMATCH"):
        build_multiround_report(tmp_path)


def test_artifact_probe_rejects_wrong_source_binding(tmp_path: Path) -> None:
    _matrix(tmp_path)
    context_id = "tokens-32"
    manifest_path = (
        tmp_path
        / "native"
        / context_id
        / "cells"
        / f"{context_id}--round-0"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["source_manifest_sha256"] = "f" * 64
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rebind_forced_summary(tmp_path, "native", context_id)
    with pytest.raises(ConformanceError, match="CELL_BINDING_INVALID"):
        build_multiround_report(tmp_path)
