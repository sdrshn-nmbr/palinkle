"""Evidence contract for DeepSpec-to-vLLM Laguna DSpark conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class ConformanceError(ValueError):
    pass


BOUNDARY_ORDER = (
    "combined_target_feature",
    "draft_backbone_hidden_state",
    "base_logits",
    "markov_bias",
    "corrected_logits",
    "confidence_logits",
    "proposal_token_ids",
)

_NUMERIC_TOLERANCES = {
    "combined_target_feature": {"rtol": 0.05, "atol": 0.0625},
    "draft_backbone_hidden_state": {"rtol": 0.05, "atol": 0.0625},
    "base_logits": {"rtol": 0.05, "atol": 0.125},
    "markov_bias": {"rtol": 0.05, "atol": 0.001},
    "corrected_logits": {"rtol": 0.05, "atol": 0.125},
    "confidence_logits": {"rtol": 0.05, "atol": 0.0625},
}
DFLASH_BOUNDARIES = (
    "combined_target_feature",
    "draft_backbone_hidden_state",
    "base_logits",
    "proposal_token_ids",
)
TARGET_FEATURE_LAYERS = 5
TARGET_FEATURE_MIN_COSINE = 0.999
TARGET_FEATURE_MAX_RELATIVE_L2 = 0.03


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_capture(capture: dict[str, Any], *, root: Path) -> None:
    missing = sorted(set(BOUNDARY_ORDER) - set(capture.get("boundaries", {})))
    if missing:
        raise ConformanceError(f"CAPTURE_BOUNDARIES_MISSING:{','.join(missing)}")
    if not capture.get("trace"):
        raise ConformanceError("CAPTURE_TRACE_MISSING")
    if not capture.get("prompt_token_ids"):
        raise ConformanceError("CAPTURE_PROMPT_TOKENS_MISSING")
    provenance = capture.get("provenance", {})
    if not provenance.get("revision") or not provenance.get("source_sha256"):
        raise ConformanceError("CAPTURE_PROVENANCE_MISSING")
    for item in [*capture["boundaries"].values(), capture["trace"]]:
        path = root / item["path"]
        if not path.is_file():
            raise ConformanceError(f"CAPTURE_ARTIFACT_MISSING:{path}")
        if file_sha256(path) != item["sha256"]:
            raise ConformanceError(f"ARTIFACT_HASH_MISMATCH:{path}")


def _tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "reason": "non_finite"}
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    reference_flat = reference.astype(np.float64).reshape(-1)
    candidate_flat = candidate.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    cosine = (
        float(np.dot(reference_flat, candidate_flat) / denominator)
        if denominator
        else 1.0
    )
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "cosine_similarity": cosine,
    }


def _compare_boundary(
    name: str, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    metrics = _tensor_metrics(reference, candidate)
    if "reason" in metrics:
        return metrics
    if name == "proposal_token_ids":
        metrics["exact_match"] = bool(np.array_equal(reference, candidate))
        metrics["passed"] = metrics["exact_match"]
        return metrics
    tolerance = _NUMERIC_TOLERANCES[name]
    close = np.isclose(reference, candidate, **tolerance)
    metrics.update(
        {
            "rtol": tolerance["rtol"],
            "atol": tolerance["atol"],
            "close_fraction": float(close.mean()) if close.size else 1.0,
            "argmax_match": bool(
                np.array_equal(
                    np.argmax(reference, axis=-1),
                    np.argmax(candidate, axis=-1),
                )
            ),
        }
    )
    metrics["passed"] = bool(close.all())
    return metrics


def _bound_artifact(root_name: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "path": f"{root_name}/{item['path']}",
    }


def build_target_feature_conformance_report(
    *,
    source_root: Path,
    source_capture: dict[str, Any],
    adapter_root: Path,
    adapter_capture: dict[str, Any],
) -> dict[str, Any]:
    if source_capture.get("prompt_token_ids") != adapter_capture.get(
        "prompt_token_ids"
    ):
        raise ConformanceError("TARGET_FEATURE_PROMPT_TOKEN_IDS_MISMATCH")
    source_item = source_capture.get("boundaries", {}).get("raw_target_features")
    adapter_item = adapter_capture.get("boundaries", {}).get("raw_target_features")
    if source_item is None or adapter_item is None:
        raise ConformanceError("TARGET_FEATURE_BOUNDARY_MISSING")
    source_path = source_root / source_item["path"]
    adapter_path = adapter_root / adapter_item["path"]
    if file_sha256(source_path) != source_item["sha256"]:
        raise ConformanceError("TARGET_FEATURE_SOURCE_HASH_MISMATCH")
    if file_sha256(adapter_path) != adapter_item["sha256"]:
        raise ConformanceError("TARGET_FEATURE_ADAPTER_HASH_MISMATCH")
    source = np.load(source_path, allow_pickle=False)
    adapter = np.load(adapter_path, allow_pickle=False)
    source = source.reshape(-1, source.shape[-1]).astype(np.float64)
    adapter = adapter.reshape(-1, adapter.shape[-1]).astype(np.float64)
    if source.shape != adapter.shape or source.shape[-1] % TARGET_FEATURE_LAYERS:
        raise ConformanceError(
            f"TARGET_FEATURE_SHAPE_MISMATCH:{source.shape}:{adapter.shape}"
        )
    width = source.shape[-1] // TARGET_FEATURE_LAYERS
    cosine_matrix: list[list[float]] = []
    layer_metrics: list[dict[str, Any]] = []
    for source_index in range(TARGET_FEATURE_LAYERS):
        source_layer = source[:, source_index * width : (source_index + 1) * width]
        row: list[float] = []
        for adapter_index in range(TARGET_FEATURE_LAYERS):
            adapter_layer = adapter[
                :, adapter_index * width : (adapter_index + 1) * width
            ]
            denominator = np.linalg.norm(source_layer) * np.linalg.norm(adapter_layer)
            row.append(
                float(np.dot(source_layer.ravel(), adapter_layer.ravel()) / denominator)
                if denominator
                else 1.0
            )
        cosine_matrix.append(row)
        matching = adapter[:, source_index * width : (source_index + 1) * width]
        relative_l2 = float(
            np.linalg.norm(source_layer - matching) / np.linalg.norm(source_layer)
        )
        per_token = []
        for token_index, (source_token, adapter_token) in enumerate(
            zip(source_layer, matching, strict=True)
        ):
            denominator = np.linalg.norm(source_token) * np.linalg.norm(adapter_token)
            token_cosine = (
                float(np.dot(source_token, adapter_token) / denominator)
                if denominator
                else 1.0
            )
            token_relative_l2 = float(
                np.linalg.norm(source_token - adapter_token)
                / np.linalg.norm(source_token)
            )
            per_token.append(
                {
                    "token_index": token_index,
                    "cosine_similarity": token_cosine,
                    "relative_l2_error": token_relative_l2,
                    "passed": bool(
                        token_cosine >= TARGET_FEATURE_MIN_COSINE
                        and token_relative_l2 <= TARGET_FEATURE_MAX_RELATIVE_L2
                    ),
                }
            )
        aligned = int(np.argmax(row)) == source_index
        layer_metrics.append(
            {
                "layer_index": source_index,
                "cosine_similarity": row[source_index],
                "relative_l2_error": relative_l2,
                "best_adapter_layer": int(np.argmax(row)),
                "worst_token_cosine_similarity": min(
                    item["cosine_similarity"] for item in per_token
                ),
                "maximum_token_relative_l2_error": max(
                    item["relative_l2_error"] for item in per_token
                ),
                "final_token": per_token[-1],
                "per_token": per_token,
                "passed": bool(aligned and all(item["passed"] for item in per_token)),
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_target_feature_conformance",
        "contract": {
            "layers": TARGET_FEATURE_LAYERS,
            "layer_width": width,
            "minimum_cosine_similarity": TARGET_FEATURE_MIN_COSINE,
            "maximum_relative_l2_error": TARGET_FEATURE_MAX_RELATIVE_L2,
            "layer_order_policy": "matching layer must have maximum cosine",
            "position_policy": "every token position must pass",
        },
        "source_manifest_sha256": source_capture["manifest_sha256"],
        "adapter_manifest_sha256": adapter_capture["manifest_sha256"],
        "source_boundary": _bound_artifact(source_root.name, source_item),
        "adapter_boundary": _bound_artifact(adapter_root.name, adapter_item),
        "cosine_matrix": cosine_matrix,
        "layers": layer_metrics,
        "passed": all(item["passed"] for item in layer_metrics),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_target_feature_conformance_report(
    report: dict[str, Any], *, root: Path, require_pass: bool = True
) -> None:
    expected = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    if report.get("report_sha256") != expected:
        raise ConformanceError("TARGET_FEATURE_REPORT_HASH_MISMATCH")
    if report.get("kind") != "opjax_laguna_target_feature_conformance":
        raise ConformanceError("TARGET_FEATURE_REPORT_KIND_INVALID")
    for key in ("source_boundary", "adapter_boundary"):
        artifact = report.get(key, {})
        path = root / artifact.get("path", "")
        if not path.is_file():
            raise ConformanceError(f"TARGET_FEATURE_ARTIFACT_MISSING:{path}")
        if file_sha256(path) != artifact.get("sha256"):
            raise ConformanceError(f"TARGET_FEATURE_ARTIFACT_HASH_MISMATCH:{path}")
    if not report.get("layers"):
        raise ConformanceError("TARGET_FEATURE_LAYERS_MISSING")
    observed_pass = all(
        layer.get("passed") is True for layer in report.get("layers", [])
    )
    if report.get("passed") is not observed_pass:
        raise ConformanceError("TARGET_FEATURE_PASS_STATE_INVALID")
    if require_pass and not observed_pass:
        raise ConformanceError("TARGET_FEATURE_CONFORMANCE_FAILED")


def build_dflash_conformance_report(
    *,
    source_root: Path,
    source_capture: dict[str, Any],
    adapter_root: Path,
    adapter_capture: dict[str, Any],
) -> dict[str, Any]:
    if source_capture.get("prompt_token_ids") != adapter_capture.get(
        "prompt_token_ids"
    ):
        raise ConformanceError("DFLASH_PROMPT_TOKEN_IDS_MISMATCH")
    comparisons: dict[str, dict[str, Any]] = {}
    for name in DFLASH_BOUNDARIES:
        source_item = source_capture.get("boundaries", {}).get(name)
        adapter_item = adapter_capture.get("boundaries", {}).get(name)
        if source_item is None or adapter_item is None:
            raise ConformanceError(f"DFLASH_BOUNDARY_MISSING:{name}")
        source_path = source_root / source_item["path"]
        adapter_path = adapter_root / adapter_item["path"]
        if file_sha256(source_path) != source_item["sha256"]:
            raise ConformanceError(f"DFLASH_SOURCE_HASH_MISMATCH:{name}")
        if file_sha256(adapter_path) != adapter_item["sha256"]:
            raise ConformanceError(f"DFLASH_ADAPTER_HASH_MISMATCH:{name}")
        comparisons[name] = _compare_boundary(
            name,
            np.load(source_path, allow_pickle=False),
            np.load(adapter_path, allow_pickle=False),
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_dflash_differential_conformance",
        "boundaries": list(DFLASH_BOUNDARIES),
        "comparisons": comparisons,
        "passed": all(value["passed"] for value in comparisons.values()),
        "source_manifest_sha256": source_capture["manifest_sha256"],
        "adapter_manifest_sha256": adapter_capture["manifest_sha256"],
    }
    report["sha256"] = canonical_sha256(report)
    return report


def _validate_capture_manifest(*, root: Path, manifest: dict[str, Any]) -> None:
    expected = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected:
        raise ConformanceError(f"CAPTURE_MANIFEST_HASH_MISMATCH:{root.name}")
    artifacts = list(manifest.get("boundaries", {}).values())
    trace = manifest.get("trace")
    if trace is not None:
        artifacts.append(trace)
    artifacts.extend(manifest.get("profiles") or [])
    server_log = manifest.get("server_log")
    if server_log is not None:
        artifacts.append(server_log)
    for item in artifacts:
        path = root / item["path"]
        if not path.is_file():
            raise ConformanceError(f"CAPTURE_ARTIFACT_MISSING:{path}")
        if file_sha256(path) != item["sha256"]:
            raise ConformanceError(f"CAPTURE_ARTIFACT_HASH_MISMATCH:{path}")


def validate_dflash_conformance_report(
    report: dict[str, Any], *, root: Path
) -> None:
    expected = canonical_sha256(
        {key: value for key, value in report.items() if key != "sha256"}
    )
    if report.get("sha256") != expected:
        raise ConformanceError("DFLASH_REPORT_HASH_MISMATCH")
    if report.get("kind") != "opjax_laguna_dflash_differential_conformance":
        raise ConformanceError("DFLASH_REPORT_KIND_INVALID")
    source = json.loads((root / "source" / "manifest.json").read_text())
    adapter = json.loads((root / "adapter" / "manifest.json").read_text())
    _validate_capture_manifest(root=root / "source", manifest=source)
    _validate_capture_manifest(root=root / "adapter", manifest=adapter)
    if source["manifest_sha256"] != report.get("source_manifest_sha256"):
        raise ConformanceError("DFLASH_SOURCE_MANIFEST_MISMATCH")
    if adapter["manifest_sha256"] != report.get("adapter_manifest_sha256"):
        raise ConformanceError("DFLASH_ADAPTER_MANIFEST_MISMATCH")
    if not adapter.get("profiles"):
        raise ConformanceError("DFLASH_PROFILE_EVIDENCE_MISSING")
    if report.get("boundaries") != list(DFLASH_BOUNDARIES):
        raise ConformanceError("DFLASH_BOUNDARIES_INVALID")
    if report.get("passed") is not True or not all(
        comparison.get("passed") is True
        for comparison in report.get("comparisons", {}).values()
    ):
        raise ConformanceError("DFLASH_CONFORMANCE_FAILED")


def build_conformance_report(
    *,
    source_root: Path,
    source_capture: dict[str, Any],
    adapter_root: Path,
    adapter_capture: dict[str, Any],
    mutation_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _require_capture(source_capture, root=source_root)
    _require_capture(adapter_capture, root=adapter_root)
    if source_capture["prompt_token_ids"] != adapter_capture["prompt_token_ids"]:
        raise ConformanceError("PROMPT_TOKEN_IDS_MISMATCH")
    if not mutation_controls or not all(
        control.get("detected") is True for control in mutation_controls.values()
    ):
        raise ConformanceError("MUTATION_CONTROL_NOT_DISCRIMINATING")

    comparisons: dict[str, dict[str, Any]] = {}
    source_boundaries: dict[str, dict[str, Any]] = {}
    adapter_boundaries: dict[str, dict[str, Any]] = {}
    for name in BOUNDARY_ORDER:
        source_item = source_capture["boundaries"][name]
        adapter_item = adapter_capture["boundaries"][name]
        source_value = np.load(source_root / source_item["path"], allow_pickle=False)
        adapter_value = np.load(adapter_root / adapter_item["path"], allow_pickle=False)
        comparisons[name] = _compare_boundary(name, source_value, adapter_value)
        source_boundaries[name] = _bound_artifact(source_root.name, source_item)
        adapter_boundaries[name] = _bound_artifact(adapter_root.name, adapter_item)

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_dspark_differential_conformance",
        "contract": {
            "boundary_order": list(BOUNDARY_ORDER),
            "numeric_tolerances": _NUMERIC_TOLERANCES,
            "token_policy": "exact",
            "prompt_policy": "exact_token_ids",
        },
        "source": {
            "implementation": source_capture["implementation"],
            "provenance": source_capture["provenance"],
            "prompt_token_ids": source_capture["prompt_token_ids"],
            "boundaries": source_boundaries,
            "trace": _bound_artifact(source_root.name, source_capture["trace"]),
        },
        "adapter": {
            "implementation": adapter_capture["implementation"],
            "provenance": adapter_capture["provenance"],
            "prompt_token_ids": adapter_capture["prompt_token_ids"],
            "boundaries": adapter_boundaries,
            "trace": _bound_artifact(adapter_root.name, adapter_capture["trace"]),
        },
        "comparisons": comparisons,
        "mutation_controls": mutation_controls,
        "passed": all(value["passed"] for value in comparisons.values()),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def validate_conformance_report(report: dict[str, Any], *, root: Path) -> None:
    expected_hash = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    if report.get("report_sha256") != expected_hash:
        raise ConformanceError("REPORT_HASH_MISMATCH")
    for lane in ("source", "adapter"):
        artifacts = [*report[lane]["boundaries"].values(), report[lane]["trace"]]
        for item in artifacts:
            path = root / item["path"]
            if not path.is_file():
                raise ConformanceError(f"ARTIFACT_MISSING:{path}")
            if file_sha256(path) != item["sha256"]:
                raise ConformanceError(f"ARTIFACT_HASH_MISMATCH:{path}")
    if not all(
        control.get("detected") is True
        for control in report.get("mutation_controls", {}).values()
    ):
        raise ConformanceError("MUTATION_CONTROL_NOT_DISCRIMINATING")
    if report.get("passed") is not True:
        raise ConformanceError("CONFORMANCE_FAILED")


def finalize_conformance(
    *, source_root: Path, adapter_root: Path, output_path: Path
) -> dict[str, Any]:
    source = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    adapter = json.loads((adapter_root / "manifest.json").read_text(encoding="utf-8"))
    report = build_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
        mutation_controls=source["mutation_controls"],
    )
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_conformance_report(report, root=output_path.parent)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_conformance(
        source_root=args.source_root,
        adapter_root=args.adapter_root,
        output_path=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
