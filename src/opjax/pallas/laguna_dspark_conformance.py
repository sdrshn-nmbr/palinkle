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
    cosine = float(np.dot(reference_flat, candidate_flat) / denominator) if denominator else 1.0
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "cosine_similarity": cosine,
    }


def _compare_boundary(name: str, reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
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
    adapter = json.loads(
        (adapter_root / "manifest.json").read_text(encoding="utf-8")
    )
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
