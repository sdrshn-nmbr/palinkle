"""Contracts for forced-prefix, multi-round DSpark differential conformance."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from opjax.pallas.laguna_dspark_conformance import (
    BOUNDARY_ORDER,
    ConformanceError,
    TARGET_FEATURE_LAYERS,
    TARGET_FEATURE_MAX_RELATIVE_L2,
    TARGET_FEATURE_MIN_COSINE,
    _compare_boundary,
    canonical_sha256,
    file_sha256,
)


CONTEXT_LENGTHS = (32, 511, 513)
ROUNDS = 3
LONG_CONTEXT_MIN_TOKENS = 1024
MAX_CONTEXT_TOKENS = 32768
DEEPSPEC_REVISION = "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced"
DEEPSPEC_SOURCE_SHA256 = "edbf639c83e9b0e5d5446736fb60fb5849446e3a2e210ac704b6a7b6d45b96d9"
VLLM_REVISION = "0.27.2rc1.dev18+g3d204dfda"
VLLM_SOURCE_SHA256 = "4468a4a7ea446ed7210f339946cc70a53fef52f5b80809a2cfcb4e6a4abbb444"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
DRAFT_REVISION = "016807a9f3e0181962ad32c096458ec022ac0f143529211c412bbf09dee2b78c"
_TRACE_VALIDATION_CACHE: dict[tuple[str, str], int] = {}
EXTENDED_BOUNDARIES = (
    "draft_input_ids",
    "draft_positions",
    "draft_input_embeddings",
    "draft_layer_0_output",
    "draft_layer_1_output",
    "draft_layer_2_output",
    "draft_layer_3_output",
    "draft_layer_4_output",
    "layer0_query_q_after_rope",
    "layer0_query_k_after_rope",
    "layer0_query_v",
    "layer0_context_k_before_rope",
    "layer0_context_v",
)
EXTENDED_ADAPTER_NAMES = {
    name: f"{name}_0"
    for name in (
        "layer0_query_q_after_rope",
        "layer0_query_k_after_rope",
        "layer0_query_v",
        "layer0_context_k_before_rope",
        "layer0_context_v",
    )
}
EXACT_EXTENDED_BOUNDARIES = {"draft_input_ids", "draft_positions"}
MAX_STRICT_BF16_ULP = 2
REPORT_FLOAT_DIGITS = 12


def _stable_report_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_report_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_report_values(item) for item in value]
    if isinstance(value, float):
        return round(value, REPORT_FLOAT_DIGITS)
    return value


def _bf16_ulp_distance(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    def ordered(value: np.ndarray) -> np.ndarray:
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
        bf16 = ((bits + bias) >> 16).astype(np.int32)
        negative = (bf16 & 0x8000) != 0
        return np.where(negative, 0x8000 - (bf16 & 0x7FFF), 0x8000 + bf16)

    return np.abs(ordered(reference) - ordered(candidate))


def _bf16_ulp_metrics(
    reference: np.ndarray, candidate: np.ndarray, *, quantiles: bool = True
) -> dict[str, Any]:
    distance = _bf16_ulp_distance(reference, candidate).reshape(-1)
    metrics = {
        "max": int(distance.max(initial=0)),
        "exact_fraction": float((distance == 0).mean()) if distance.size else 1.0,
        "within_1_fraction": float((distance <= 1).mean()) if distance.size else 1.0,
        "within_2_fraction": float((distance <= 2).mean()) if distance.size else 1.0,
        "passed": bool((distance <= MAX_STRICT_BF16_ULP).all()),
    }
    if quantiles:
        metrics.update(
            p50=float(np.quantile(distance, 0.50)) if distance.size else 0.0,
            p95=float(np.quantile(distance, 0.95)) if distance.size else 0.0,
            p99=float(np.quantile(distance, 0.99)) if distance.size else 0.0,
        )
    return metrics


def build_contexts(rendered_token_ids: list[int]) -> dict[str, list[int]]:
    if len(rendered_token_ids) < LONG_CONTEXT_MIN_TOKENS:
        raise ConformanceError(
            f"MULTIROUND_LONG_CONTEXT_TOO_SHORT:{len(rendered_token_ids)}"
        )
    if any(not isinstance(token, int) or token < 0 for token in rendered_token_ids):
        raise ConformanceError("MULTIROUND_TOKEN_IDS_INVALID")
    contexts = {
        f"tokens-{length}": rendered_token_ids[:length] for length in CONTEXT_LENGTHS
    }
    contexts["long-agent-prefix"] = list(rendered_token_ids)
    return contexts


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if expected != canonical_sha256(unsigned):
        raise ConformanceError(f"MULTIROUND_MANIFEST_HASH_MISMATCH:{path}")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not all(
        provenance.get(key)
        for key in ("revision", "source_sha256", "target_revision", "draft_revision")
    ):
        raise ConformanceError(f"MULTIROUND_PROVENANCE_MISSING:{path}")
    revision = provenance["revision"]
    expected_source = {
        DEEPSPEC_REVISION: DEEPSPEC_SOURCE_SHA256,
        VLLM_REVISION: VLLM_SOURCE_SHA256,
    }.get(revision)
    if (
        expected_source is None
        or provenance["source_sha256"] != expected_source
        or provenance["target_revision"] != TARGET_REVISION
        or provenance["draft_revision"] != DRAFT_REVISION
    ):
        raise ConformanceError(f"MULTIROUND_PROVENANCE_INVALID:{path}")
    trace = value.get("trace")
    if not isinstance(trace, dict):
        raise ConformanceError(f"MULTIROUND_TRACE_MISSING:{path}")
    trace_path = path.parent / trace["path"]
    if (
        not trace_path.is_file()
        or trace_path.stat().st_size != trace.get("bytes")
    ):
        raise ConformanceError(f"MULTIROUND_TRACE_INVALID:{trace_path}")
    if trace_path.name == "trace-index.json":
        if file_sha256(trace_path) != trace.get("sha256"):
            raise ConformanceError(f"MULTIROUND_TRACE_INVALID:{trace_path}")
        index = json.loads(trace_path.read_text(encoding="utf-8"))
        cuda_kernel_events = 0
        for item in index.get("files", []):
            artifact = trace_path.parent / item["path"]
            if (
                not artifact.is_file()
                or file_sha256(artifact) != item.get("sha256")
                or artifact.stat().st_size != item.get("bytes")
            ):
                raise ConformanceError(f"MULTIROUND_PROFILE_INVALID:{artifact}")
            cache_key = (str(artifact), item["sha256"])
            if cache_key not in _TRACE_VALIDATION_CACHE:
                try:
                    opener = gzip.open if artifact.suffix == ".gz" else open
                    with opener(artifact, "rt", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    events = payload.get("traceEvents", [])
                    _TRACE_VALIDATION_CACHE[cache_key] = sum(
                        "cudalaunchkernel" in json.dumps(event).lower()
                        for event in events
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    _TRACE_VALIDATION_CACHE[cache_key] = 0
            cuda_kernel_events += _TRACE_VALIDATION_CACHE[cache_key]
        if cuda_kernel_events == 0:
            raise ConformanceError(f"MULTIROUND_CUDA_PROFILE_MISSING:{trace_path}")
        value["_validated_cuda_kernel_events"] = cuda_kernel_events
    elif trace_path.suffix == ".json":
        cache_key = (str(trace_path), trace["sha256"])
        if file_sha256(trace_path) != trace.get("sha256"):
            raise ConformanceError(f"MULTIROUND_TRACE_INVALID:{trace_path}")
        if cache_key not in _TRACE_VALIDATION_CACHE:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            events = payload.get("traceEvents", [])
            _TRACE_VALIDATION_CACHE[cache_key] = sum(
                "cudalaunchkernel" in json.dumps(event).lower() for event in events
            )
        cuda_kernel_events = _TRACE_VALIDATION_CACHE[cache_key]
        if cuda_kernel_events == 0:
            raise ConformanceError(f"MULTIROUND_CUDA_TRACE_MISSING:{trace_path}")
        value["_validated_cuda_kernel_events"] = cuda_kernel_events
    else:
        raise ConformanceError(f"MULTIROUND_TRACE_FORMAT_INVALID:{trace_path}")
    return value


def _validate_contexts(contexts: dict[str, list[int]]) -> None:
    expected = {f"tokens-{length}" for length in CONTEXT_LENGTHS} | {
        "long-agent-prefix"
    }
    if set(contexts) != expected:
        raise ConformanceError("MULTIROUND_CONTEXT_SET_INVALID")
    for length in CONTEXT_LENGTHS:
        if len(contexts[f"tokens-{length}"]) != length:
            raise ConformanceError(f"MULTIROUND_CONTEXT_LENGTH_INVALID:{length}")
    long_length = len(contexts["long-agent-prefix"])
    if not LONG_CONTEXT_MIN_TOKENS <= long_length <= MAX_CONTEXT_TOKENS - 64:
        raise ConformanceError(f"MULTIROUND_LONG_CONTEXT_LENGTH_INVALID:{long_length}")


def _bound_execution_file(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ConformanceError(f"MULTIROUND_EXECUTION_ARTIFACT_MISSING:{path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _validate_vllm_summary(
    *, root: Path, lane: str, context_id: str, cell_hashes: list[str]
) -> dict[str, Any]:
    run_root = root / lane / context_id
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary.get("summary_sha256")
    unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if expected != canonical_sha256(unsigned) or summary.get("cells") != cell_hashes:
        raise ConformanceError(f"MULTIROUND_SUMMARY_INVALID:{lane}:{context_id}")
    before = run_root / "metrics-before.txt"
    after = run_root / "metrics-after.txt"
    log = run_root / "server.log"
    if (
        hashlib.sha256(before.read_bytes()).hexdigest()
        != summary["metrics_before_sha256"]
        or hashlib.sha256(after.read_bytes()).hexdigest()
        != summary["metrics_after_sha256"]
        or file_sha256(log) != summary["server_log_sha256"]
    ):
        raise ConformanceError(f"MULTIROUND_RUNTIME_BINDING_INVALID:{lane}:{context_id}")
    telemetry = root / f"{lane}-{context_id}-gpu.csv"
    return {
        "summary": _bound_execution_file(summary_path, root=root),
        "metrics_before": _bound_execution_file(before, root=root),
        "metrics_after": _bound_execution_file(after, root=root),
        "server_log": _bound_execution_file(log, root=root),
        "gpu_telemetry": _bound_execution_file(telemetry, root=root),
    }


def _validate_sequential_summary(
    *, root: Path, lane: str, context_id: str, cell_hashes: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = root / f"sequential-{lane}" / context_id
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary.get("summary_sha256")
    unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if (
        expected != canonical_sha256(unsigned)
        or summary.get("cell_manifest_sha256") != cell_hashes
        or summary.get("context_id") != context_id
        or summary.get("captured_rounds") != ROUNDS
        or int(summary.get("proposal_invocations", 0)) < ROUNDS
    ):
        raise ConformanceError(f"SEQUENTIAL_SUMMARY_INVALID:{lane}:{context_id}")
    before = run_root / "metrics-before.txt"
    after = run_root / "metrics-after.txt"
    log = run_root / "server.log"
    if (
        hashlib.sha256(before.read_bytes()).hexdigest()
        != summary["metrics_before_sha256"]
        or hashlib.sha256(after.read_bytes()).hexdigest()
        != summary["metrics_after_sha256"]
        or file_sha256(log) != summary["server_log_sha256"]
    ):
        raise ConformanceError(f"SEQUENTIAL_RUNTIME_BINDING_INVALID:{lane}:{context_id}")
    evidence = {
        "summary": _bound_execution_file(summary_path, root=root),
        "metrics_before": _bound_execution_file(before, root=root),
        "metrics_after": _bound_execution_file(after, root=root),
        "server_log": _bound_execution_file(log, root=root),
        "gpu_telemetry": _bound_execution_file(
            root / f"sequential-{lane}-{context_id}-gpu.csv", root=root
        ),
    }
    return summary, evidence


def _load_array(root: Path, item: dict[str, Any]) -> np.ndarray:
    path = root / item["path"]
    if not path.is_file() or file_sha256(path) != item.get("sha256"):
        raise ConformanceError(f"MULTIROUND_ARTIFACT_HASH_MISMATCH:{path}")
    return np.load(path, allow_pickle=False)


_TOKEN_AXIS = {
    "raw_target_features": 2,
    "combined_target_feature": 1,
    "layer0_context_k_before_rope_0": 2,
    "layer0_context_v_0": 2,
}


def _align_source(
    name: str, value: np.ndarray, *, prompt_length: int, processed_start: int
) -> np.ndarray:
    policy = _TOKEN_AXIS.get(name)
    if policy is None:
        return value
    axis = policy - 1
    if value.ndim <= axis or value.shape[axis] != prompt_length:
        raise ConformanceError(
            f"MULTIROUND_ALIGNMENT_SHAPE:{name}:{value.shape}:{prompt_length}"
        )
    if axis == 0:
        return value[processed_start:]
    if axis == 1:
        return value[:, processed_start:]
    raise AssertionError(axis)


def _compare_extended(
    *,
    source_root: Path,
    source: dict[str, Any],
    adapter_root: Path,
    adapter: dict[str, Any],
    processed_start: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in EXTENDED_BOUNDARIES:
        source_item = source.get("boundaries", {}).get(name)
        adapter_name = EXTENDED_ADAPTER_NAMES.get(name, name)
        adapter_item = adapter.get("boundaries", {}).get(adapter_name)
        if source_item is None or adapter_item is None:
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_BOUNDARY_MISSING:{name}:{adapter_name}"
            )
        reference = _canonicalize_extended_source(
            name,
            _load_array(source_root, source_item),
            prompt_length=len(source["prompt_token_ids"]),
            processed_start=processed_start,
        )
        candidate = _load_array(adapter_root, adapter_item)
        if reference.shape != candidate.shape:
            results[name] = {
                "comparable": True,
                "passed": False,
                "reason": "shape_mismatch",
                "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape),
            }
            continue
        if name in EXACT_EXTENDED_BOUNDARIES:
            exact = bool(
                np.issubdtype(reference.dtype, np.integer)
                and np.issubdtype(candidate.dtype, np.integer)
                and np.array_equal(reference, candidate)
            )
            results[name] = {
                "comparable": True,
                "passed": exact,
                "exact_match": exact,
                "dtype_match": bool(reference.dtype == candidate.dtype),
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(candidate.dtype),
            }
            continue
        reference = reference.astype(np.float64)
        candidate = candidate.astype(np.float64)
        close = np.isclose(reference, candidate, rtol=0.05, atol=0.0625)
        results[name] = {
            "comparable": True,
            "passed": bool(close.all()),
            "close_fraction": float(close.mean()) if close.size else 1.0,
            "max_abs_error": float(np.abs(reference - candidate).max(initial=0.0)),
            "bf16_ulp": _bf16_ulp_metrics(reference, candidate),
        }
    return results


def _extended_functional_passed(results: dict[str, Any]) -> bool:
    return all(
        result["passed"]
        for result in results.values()
        if result.get("comparable", True)
    )


def _extended_strict_bf16_passed(results: dict[str, Any]) -> bool:
    return all(
        result["passed"]
        if name in EXACT_EXTENDED_BOUNDARIES
        else result.get("bf16_ulp", {"passed": True})["passed"]
        for name, result in results.items()
        if result.get("comparable", True)
    )


def _canonicalize_extended_source(
    name: str, value: np.ndarray, *, prompt_length: int, processed_start: int
) -> np.ndarray:
    if name in {"draft_input_ids", "draft_positions"}:
        flattened = value.reshape(-1)
        if flattened.shape != (16,):
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_SOURCE_SHAPE:{name}:{value.shape}"
            )
        return flattened[:15]
    if name == "draft_input_embeddings":
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 16:
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_SOURCE_SHAPE:{name}:{value.shape}"
            )
        return value[0, :15, :]
    if name.startswith("draft_layer_"):
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 15:
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_SOURCE_SHAPE:{name}:{value.shape}"
            )
        return value[0]
    if name in {
        "layer0_query_q_after_rope",
        "layer0_query_k_after_rope",
        "layer0_query_v",
    }:
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 16:
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_SOURCE_SHAPE:{name}:{value.shape}"
            )
        return value[0, :15].reshape(15, -1)
    if name in {"layer0_context_k_before_rope", "layer0_context_v"}:
        if (
            value.ndim != 4
            or value.shape[0] != 1
            or value.shape[1] != prompt_length
            or not 0 <= processed_start < prompt_length
        ):
            raise ConformanceError(
                f"MULTIROUND_EXTENDED_SOURCE_SHAPE:{name}:{value.shape}:"
                f"{prompt_length}:{processed_start}"
            )
        return value[0, processed_start:prompt_length]
    raise ConformanceError(f"MULTIROUND_EXTENDED_SOURCE_POLICY_MISSING:{name}")


def _compare_downstream(
    *,
    source_root: Path,
    source: dict[str, Any],
    adapter_root: Path,
    adapter: dict[str, Any],
    processed_start: int,
) -> dict[str, Any]:
    reference_proposals = _load_array(
        source_root, source["boundaries"]["proposal_token_ids"]
    ).reshape(-1)
    candidate_proposals = _load_array(
        adapter_root, adapter["boundaries"]["proposal_token_ids"]
    ).reshape(-1)
    if reference_proposals.shape != candidate_proposals.shape:
        comparable_prefix_length = 0
        comparable_rows = 0
        first_divergence = 0
    else:
        mismatches = np.flatnonzero(reference_proposals != candidate_proposals)
        first_divergence = int(mismatches[0]) if mismatches.size else None
        comparable_prefix_length = (
            len(reference_proposals) if first_divergence is None else first_divergence
        )
        comparable_rows = (
            len(reference_proposals)
            if first_divergence is None
            else first_divergence + 1
        )
    boundaries: dict[str, Any] = {}
    for name in BOUNDARY_ORDER:
        reference = _align_source(
            name,
            _load_array(source_root, source["boundaries"][name]),
            prompt_length=len(source["prompt_token_ids"]),
            processed_start=processed_start,
        )
        candidate = _load_array(adapter_root, adapter["boundaries"][name])
        if name in {"markov_bias", "corrected_logits", "confidence_logits"}:
            reference = reference[:comparable_rows]
            candidate = candidate[:comparable_rows]
        comparison = _compare_boundary(name, reference, candidate)
        if (
            name != "proposal_token_ids"
            and reference.shape == candidate.shape
            and np.isfinite(reference).all()
            and np.isfinite(candidate).all()
        ):
            comparison["bf16_ulp"] = _bf16_ulp_metrics(reference, candidate)
        boundaries[name] = comparison
    return {
        "boundaries": boundaries,
        "comparable_proposal_prefix_length": comparable_prefix_length,
        "causally_comparable_rows": comparable_rows,
        "first_proposal_divergence": first_divergence,
        "passed": all(item["passed"] for item in boundaries.values()),
    }


def _compare_target_features(
    *,
    source_root: Path,
    source: dict[str, Any],
    adapter_root: Path,
    adapter: dict[str, Any],
    processed_start: int,
) -> dict[str, Any]:
    reference = _align_source(
        "raw_target_features",
        _load_array(source_root, source["boundaries"]["raw_target_features"]),
        prompt_length=len(source["prompt_token_ids"]),
        processed_start=processed_start,
    ).reshape(-1, _load_array(source_root, source["boundaries"]["raw_target_features"]).shape[-1])
    candidate = _load_array(
        adapter_root, adapter["boundaries"]["raw_target_features"]
    ).reshape(-1, reference.shape[-1])
    if reference.shape != candidate.shape or reference.shape[-1] % TARGET_FEATURE_LAYERS:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    width = reference.shape[-1] // TARGET_FEATURE_LAYERS
    left = reference.reshape(reference.shape[0], TARGET_FEATURE_LAYERS, width)
    right = candidate.reshape(candidate.shape[0], TARGET_FEATURE_LAYERS, width)
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left_norm = np.linalg.norm(left64, axis=-1)
    right_norm = np.linalg.norm(right64, axis=-1)
    denominator = left_norm * right_norm
    cosine_values = np.divide(
        np.sum(left64 * right64, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0,
    )
    relative_values = np.divide(
        np.linalg.norm(left64 - right64, axis=-1),
        left_norm,
        out=np.zeros_like(left_norm),
        where=left_norm != 0,
    )
    ulp = _bf16_ulp_distance(left, right)
    ulp_max = ulp.max(axis=-1, initial=0)
    ulp_exact = (ulp == 0).mean(axis=-1)
    ulp_within_1 = (ulp <= 1).mean(axis=-1)
    ulp_within_2 = (ulp <= 2).mean(axis=-1)
    tokens: list[dict[str, Any]] = []
    for token_index in range(reference.shape[0]):
        for layer_index in range(TARGET_FEATURE_LAYERS):
            cosine = float(cosine_values[token_index, layer_index])
            relative_l2 = float(relative_values[token_index, layer_index])
            tokens.append(
                {
                    "token_index": token_index + processed_start,
                    "layer_index": layer_index,
                    "cosine_similarity": cosine,
                    "relative_l2_error": relative_l2,
                    "bf16_ulp": {
                        "max": int(ulp_max[token_index, layer_index]),
                        "exact_fraction": float(ulp_exact[token_index, layer_index]),
                        "within_1_fraction": float(ulp_within_1[token_index, layer_index]),
                        "within_2_fraction": float(ulp_within_2[token_index, layer_index]),
                        "passed": bool(ulp_max[token_index, layer_index] <= 2),
                    },
                    "passed": bool(
                        cosine >= TARGET_FEATURE_MIN_COSINE
                        and relative_l2 <= TARGET_FEATURE_MAX_RELATIVE_L2
                    ),
                }
            )
    return {"tokens": tokens, "passed": all(item["passed"] for item in tokens)}


def build_multiround_report(root: Path) -> dict[str, Any]:
    matrix_path = root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ConformanceError("MULTIROUND_MATRIX_SCHEMA_INVALID")
    _validate_contexts(matrix.get("contexts", {}))
    expected_cells = len(matrix.get("contexts", {})) * ROUNDS
    cells: list[dict[str, Any]] = []
    execution_artifacts: dict[str, Any] = {}
    for context_id, base_tokens in sorted(matrix["contexts"].items()):
        if len(base_tokens) < 2:
            raise ConformanceError(f"MULTIROUND_CONTEXT_TOO_SHORT:{context_id}")
        committed = list(base_tokens)
        lane_cell_hashes: dict[str, list[str]] = {"injected": [], "native": []}
        for round_index in range(ROUNDS):
            cell_id = f"{context_id}--round-{round_index}"
            source_root = root / "source" / cell_id
            source = _load_manifest(source_root / "manifest.json")
            if source.get("input_mode") != "token_ids":
                raise ConformanceError(f"MULTIROUND_SOURCE_INPUT_MODE:{cell_id}")
            if source.get("prompt_token_ids") != committed:
                raise ConformanceError(f"MULTIROUND_SOURCE_PREFIX_MISMATCH:{cell_id}")
            anchor = _load_array(
                source_root, source["boundaries"]["anchor_token_id"]
            ).reshape(-1)
            if anchor.size != 1:
                raise ConformanceError(f"MULTIROUND_ANCHOR_SHAPE:{cell_id}")
            lane_reports: dict[str, Any] = {}
            adapter_manifests: dict[str, dict[str, Any]] = {}
            for lane in ("injected", "native"):
                adapter_root = root / lane / context_id / "cells" / cell_id
                adapter = _load_manifest(adapter_root / "manifest.json")
                adapter_manifests[lane] = adapter
                lane_cell_hashes[lane].append(adapter["manifest_sha256"])
                provenance = adapter["provenance"]
                if (
                    adapter.get("context_id") != context_id
                    or adapter.get("round") != round_index
                    or provenance.get("lane") != lane
                    or provenance.get("source_manifest_sha256")
                    != source["manifest_sha256"]
                ):
                    raise ConformanceError(f"MULTIROUND_CELL_BINDING_INVALID:{lane}:{cell_id}")
                if adapter.get("prompt_token_ids") != committed:
                    raise ConformanceError(
                        f"MULTIROUND_ADAPTER_PREFIX_MISMATCH:{lane}:{cell_id}"
                    )
                processed_start = adapter.get("processed_token_start", 0)
                if not isinstance(processed_start, int) or not (
                    0 <= processed_start < len(committed)
                ):
                    raise ConformanceError(
                        f"MULTIROUND_PROCESSED_START_INVALID:{lane}:{cell_id}"
                    )
                expected_start = adapter.get("expected_processed_token_start")
                if lane == "native" and expected_start is not None:
                    raise ConformanceError(
                        f"MULTIROUND_NATIVE_CACHE_EXPECTATION_PRESENT:{cell_id}"
                    )
                downstream = _compare_downstream(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                target_features = _compare_target_features(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                extended = _compare_extended(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                response_choices = adapter.get("response", {}).get("choices", [])
                response_tokens = (
                    response_choices[0].get("token_ids", [])
                    if len(response_choices) == 1
                    else []
                )
                target_token_match = bool(
                    response_tokens and int(response_tokens[0]) == int(anchor[0])
                )
                core_functional_passed = bool(
                    downstream["passed"]
                    and target_features["passed"]
                    and target_token_match
                )
                extended_comparable_passed = _extended_functional_passed(extended)
                functional_passed = bool(
                    core_functional_passed and extended_comparable_passed
                )
                strict_bf16_passed = bool(
                    downstream["boundaries"]["proposal_token_ids"]["passed"]
                    and target_features.get("tokens")
                    and all(
                        item.get("bf16_ulp", {"passed": True})["passed"]
                        for item in downstream["boundaries"].values()
                    )
                    and _extended_strict_bf16_passed(extended)
                    and all(
                        item["bf16_ulp"]["passed"]
                        for item in target_features.get("tokens", [])
                    )
                    and target_token_match
                )
                lane_reports[lane] = {
                    "cuda_kernel_events": adapter[
                        "_validated_cuda_kernel_events"
                    ],
                    "downstream": downstream,
                    "target_features": target_features,
                    "extended_boundaries": extended,
                    "target_token_match": target_token_match,
                    "core_functional_passed": core_functional_passed,
                    "extended_comparable_passed": extended_comparable_passed,
                    "functional_passed": functional_passed,
                    "strict_bf16_passed": strict_bf16_passed,
                    "passed": bool(functional_passed and strict_bf16_passed),
                }
            native_start = adapter_manifests["native"]["processed_token_start"]
            injected = adapter_manifests["injected"]
            if (
                injected.get("expected_processed_token_start") != native_start
                or injected.get("processed_token_start") != native_start
            ):
                raise ConformanceError(
                    f"MULTIROUND_CACHE_START_MISMATCH:{cell_id}:"
                    f"{native_start}:"
                    f"{injected.get('expected_processed_token_start')}:"
                    f"{injected.get('processed_token_start')}"
                )
            cells.append(
                {
                    "cell_id": cell_id,
                    "context_id": context_id,
                    "round": round_index,
                    "committed_token_ids": list(committed),
                    "anchor_token_id": int(anchor[0]),
                    "source_cuda_kernel_events": source[
                        "_validated_cuda_kernel_events"
                    ],
                    "lanes": lane_reports,
                }
            )
            committed.append(int(anchor[0]))
        execution_artifacts[f"source:{context_id}"] = {
            "gpu_telemetry": _bound_execution_file(
                root / f"source-{context_id}-gpu.csv", root=root
            )
        }
        for lane in ("injected", "native"):
            execution_artifacts[f"{lane}:{context_id}"] = _validate_vllm_summary(
                root=root,
                lane=lane,
                context_id=context_id,
                cell_hashes=lane_cell_hashes[lane],
            )
    if len(cells) != expected_cells:
        raise ConformanceError(
            f"MULTIROUND_CELL_CARDINALITY:{len(cells)}:{expected_cells}"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "laguna_dspark_forced_prefix_multiround_conformance",
        "matrix_sha256": file_sha256(matrix_path),
        "contexts": len(matrix["contexts"]),
        "rounds_per_context": ROUNDS,
        "cells": cells,
        "execution_artifacts": execution_artifacts,
        "injected_passed": all(cell["lanes"]["injected"]["passed"] for cell in cells),
        "native_passed": all(cell["lanes"]["native"]["passed"] for cell in cells),
        "injected_functional_passed": all(
            cell["lanes"]["injected"]["functional_passed"] for cell in cells
        ),
        "native_functional_passed": all(
            cell["lanes"]["native"]["functional_passed"] for cell in cells
        ),
        "injected_strict_bf16_passed": all(
            cell["lanes"]["injected"]["strict_bf16_passed"] for cell in cells
        ),
        "native_strict_bf16_passed": all(
            cell["lanes"]["native"]["strict_bf16_passed"] for cell in cells
        ),
    }
    report = _stable_report_values(report)
    report["report_sha256"] = canonical_sha256(report)
    return report


def build_sequential_report(root: Path) -> dict[str, Any]:
    matrix_path = root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    _validate_contexts(matrix.get("contexts", {}))
    cells: list[dict[str, Any]] = []
    execution_artifacts: dict[str, Any] = {}
    comparable_rounds_by_context: dict[str, int] = {}
    for context_id, base_tokens in sorted(matrix["contexts"].items()):
        injected_prefix_diverged = False
        injected_comparable_rounds = 0
        manifests: dict[str, list[dict[str, Any]]] = {"native": [], "injected": []}
        for lane in manifests:
            for round_index in range(ROUNDS):
                cell_id = f"{context_id}--round-{round_index}"
                manifests[lane].append(
                    _load_manifest(
                        root
                        / f"sequential-{lane}"
                        / context_id
                        / "cells"
                        / cell_id
                        / "manifest.json"
                    )
                )
        summaries: dict[str, Any] = {}
        for lane in manifests:
            summary, evidence = _validate_sequential_summary(
                root=root,
                lane=lane,
                context_id=context_id,
                cell_hashes=[item["manifest_sha256"] for item in manifests[lane]],
            )
            summaries[lane] = summary
            execution_artifacts[f"{lane}:{context_id}"] = evidence
        execution_artifacts[f"source:{context_id}"] = {
            "gpu_telemetry": _bound_execution_file(
                root / f"sequential-source-{context_id}-gpu.csv", root=root
            )
        }
        response_match = (
            summaries["native"]["response_token_ids"]
            == summaries["injected"]["response_token_ids"]
        )
        for round_index in range(ROUNDS):
            cell_id = f"{context_id}--round-{round_index}"
            source_root = root / "sequential-source" / context_id / "cells" / cell_id
            source = _load_manifest(source_root / "manifest.json")
            if source.get("input_mode") != "token_ids":
                raise ConformanceError(f"SEQUENTIAL_SOURCE_INPUT_MODE:{cell_id}")
            native_start = manifests["native"][round_index]["processed_token_start"]
            override_path = (
                root
                / "sequential-overrides"
                / context_id
                / f"round-{round_index}.npy"
            )
            source_features = _load_array(
                source_root, source["boundaries"]["raw_target_features"]
            )
            expected_override = source_features[:, native_start:, :]
            if not override_path.is_file():
                raise ConformanceError(f"SEQUENTIAL_OVERRIDE_MISSING:{override_path}")
            actual_override = np.load(override_path, allow_pickle=False)
            if not np.array_equal(actual_override, expected_override):
                raise ConformanceError(f"SEQUENTIAL_OVERRIDE_INVALID:{cell_id}")
            override_evidence = {
                "path": str(override_path.relative_to(root)),
                "sha256": file_sha256(override_path),
                "source_manifest_sha256": source["manifest_sha256"],
                "native_processed_token_start": native_start,
            }
            lanes: dict[str, Any] = {}
            for lane in ("native", "injected"):
                adapter_root = (
                    root / f"sequential-{lane}" / context_id / "cells" / cell_id
                )
                adapter = manifests[lane][round_index]
                if (
                    adapter.get("context_id") != context_id
                    or adapter.get("round") != round_index
                    or adapter["provenance"].get("target_feature_mode")
                    != ("source_override" if lane == "injected" else "live_vllm")
                ):
                    raise ConformanceError(f"SEQUENTIAL_CELL_BINDING_INVALID:{lane}:{cell_id}")
                committed = adapter["prompt_token_ids"]
                generated_count = len(committed) - len(base_tokens)
                response_tokens = adapter["response_token_ids"]
                reconstruction_match = bool(
                    generated_count >= 0
                    and committed[: len(base_tokens)] == base_tokens
                    and committed[len(base_tokens) :]
                    == response_tokens[:generated_count]
                )
                comparable = committed == source["prompt_token_ids"]
                causal_comparable = bool(
                    comparable and not (lane == "injected" and injected_prefix_diverged)
                )
                if not causal_comparable:
                    lanes[lane] = {
                        "comparable": False,
                        "causal_comparable": False,
                        "reason": (
                            "earlier_injected_prefix_divergence"
                            if lane == "injected" and injected_prefix_diverged
                            else "prefix_mismatch"
                        ),
                        "prefix_reconstruction_match": reconstruction_match,
                        "passed": False,
                    }
                    continue
                processed_start = adapter["processed_token_start"]
                if lane == "injected" and processed_start != native_start:
                    raise ConformanceError(
                        f"SEQUENTIAL_CACHE_START_MISMATCH:{cell_id}:"
                        f"{native_start}:{processed_start}"
                    )
                downstream = _compare_downstream(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                target_features = _compare_target_features(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                extended = _compare_extended(
                    source_root=source_root,
                    source=source,
                    adapter_root=adapter_root,
                    adapter=adapter,
                    processed_start=processed_start,
                )
                anchor = int(
                    _load_array(
                        source_root, source["boundaries"]["anchor_token_id"]
                    ).reshape(-1)[0]
                )
                generated_index = len(source["prompt_token_ids"]) - len(base_tokens)
                target_token_match = bool(
                    generated_index < len(response_tokens)
                    and response_tokens[generated_index] == anchor
                )
                lanes[lane] = {
                    "comparable": True,
                    "causal_comparable": True,
                    "cuda_kernel_events": adapter[
                        "_validated_cuda_kernel_events"
                    ],
                    "prefix_reconstruction_match": reconstruction_match,
                    "downstream": downstream,
                    "target_features": target_features,
                    "extended_boundaries": extended,
                    "target_token_match": target_token_match,
                    "core_functional_passed": bool(
                        downstream["passed"]
                        and target_features["passed"]
                        and target_token_match
                        and reconstruction_match
                    ),
                    "extended_comparable_passed": _extended_functional_passed(
                        extended
                    ),
                    "passed": bool(
                        downstream["passed"]
                        and target_features["passed"]
                        and _extended_functional_passed(extended)
                        and target_token_match
                        and reconstruction_match
                    ),
                }
            if manifests["injected"][round_index]["prompt_token_ids"] != source[
                "prompt_token_ids"
            ]:
                injected_prefix_diverged = True
            if lanes["injected"].get("causal_comparable") is True:
                injected_comparable_rounds += 1
            cells.append(
                {
                    "cell_id": cell_id,
                    "context_id": context_id,
                    "round": round_index,
                    "committed_token_ids": source["prompt_token_ids"],
                    "source_cuda_kernel_events": source[
                        "_validated_cuda_kernel_events"
                    ],
                    "lanes": lanes,
                    "override_evidence": override_evidence,
                    "response_token_ids_match": response_match,
                }
            )
        comparable_rounds_by_context[context_id] = injected_comparable_rounds
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "laguna_dspark_single_request_multiround_conformance",
        "numerical_gate": "functional_only",
        "strict_bf16_evaluated": False,
        "matrix_sha256": file_sha256(matrix_path),
        "cells": cells,
        "injected_comparable_rounds_by_context": comparable_rounds_by_context,
        "execution_artifacts": execution_artifacts,
        "injected_passed": all(cell["lanes"]["injected"]["passed"] for cell in cells),
        "native_passed": all(cell["lanes"]["native"]["passed"] for cell in cells),
        "response_token_ids_match": all(
            cell["response_token_ids_match"] for cell in cells
        ),
    }
    report = _stable_report_values(report)
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_multiround_report(report: dict[str, Any], *, root: Path) -> None:
    _validate_multiround_report_against(report, build_multiround_report(root))


def _validate_multiround_report_against(
    report: dict[str, Any], recomputed: dict[str, Any]
) -> None:
    expected = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if expected != canonical_sha256(unsigned):
        raise ConformanceError("MULTIROUND_REPORT_HASH_MISMATCH")
    if recomputed != report:
        raise ConformanceError("MULTIROUND_REPORT_RECOMPUTATION_MISMATCH")


def build_final_report(root: Path) -> dict[str, Any]:
    forced_prefix = build_multiround_report(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "laguna_dspark_multiround_conformance_closeout",
        "forced_prefix": forced_prefix,
        "sequential": build_sequential_report(root),
        "report_recomputation_controls": mutation_controls_pass(
            root, report=forced_prefix
        ),
        "resume_attempts": _bound_resume_attempts(root),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _bound_resume_attempts(root: Path) -> list[dict[str, Any]]:
    attempts_root = root / "attempts"
    if not attempts_root.exists():
        return []
    attempts = []
    for manifest_path in sorted(attempts_root.rglob("archive-manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("manifest_sha256")
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if expected != canonical_sha256(unsigned):
            raise ConformanceError(f"MULTIROUND_ATTEMPT_HASH_MISMATCH:{manifest_path}")
        attempt_root = manifest_path.parent
        for item in manifest.get("files", []):
            path = attempt_root / item["path"]
            if (
                not path.is_file()
                or file_sha256(path) != item["sha256"]
                or path.stat().st_size != item["bytes"]
            ):
                raise ConformanceError(f"MULTIROUND_ATTEMPT_FILE_INVALID:{path}")
        attempts.append(
            {
                "path": str(manifest_path.relative_to(root)),
                "sha256": file_sha256(manifest_path),
                "manifest_sha256": expected,
            }
        )
    return attempts


def validate_final_report(report: dict[str, Any], *, root: Path) -> None:
    expected = report.get("report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if expected != canonical_sha256(unsigned):
        raise ConformanceError("MULTIROUND_FINAL_REPORT_HASH_MISMATCH")
    recomputed = build_final_report(root)
    if recomputed != report:
        raise ConformanceError("MULTIROUND_FINAL_REPORT_RECOMPUTATION_MISMATCH")
    if not all(report["report_recomputation_controls"].values()):
        raise ConformanceError("MULTIROUND_MUTATION_CONTROL_FAILED")


def mutation_controls_pass(
    root: Path, *, report: dict[str, Any] | None = None
) -> dict[str, bool]:
    report = report or build_multiround_report(root)
    controls: dict[str, bool] = {}
    mutations = {
        "round_swap": lambda value: value["cells"].__setitem__(
            slice(0, 2), [value["cells"][1], value["cells"][0]]
        ),
        "position_plus_one": lambda value: value["cells"][0][
            "committed_token_ids"
        ].append(1),
        "wrong_committed_token": lambda value: value["cells"][0].__setitem__(
            "anchor_token_id", value["cells"][0]["anchor_token_id"] + 1
        ),
        "stale_cache": lambda value: value["cells"][1].__setitem__(
            "committed_token_ids", value["cells"][0]["committed_token_ids"]
        ),
        "target_feature_round_swap": lambda value: value["cells"][0]["lanes"][
            "injected"
        ]["target_features"]["tokens"][0].__setitem__("token_index", 999999),
    }
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(report)
        mutate(candidate)
        candidate["report_sha256"] = canonical_sha256(
            {key: value for key, value in candidate.items() if key != "report_sha256"}
        )
        try:
            _validate_multiround_report_against(candidate, report)
        except ConformanceError:
            controls[name] = True
        else:
            controls[name] = False
    return controls
