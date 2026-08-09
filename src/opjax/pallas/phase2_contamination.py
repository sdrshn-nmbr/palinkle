"""Fail-closed Phase 2 benchmark contamination checks for future training data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


class Phase2ContaminationError(RuntimeError):
    pass


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", value.lower()))


def _shingles(value: str, width: int = 7) -> frozenset[str]:
    tokens = _tokens(value)
    if len(tokens) < width:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def build_signatures(release_root: Path) -> dict[str, Any]:
    records = []
    identifiers = []
    for task_root in sorted((release_root / "tasks").iterdir()):
        task = json.loads((task_root / "tests/task.json").read_text(encoding="utf-8"))
        identifiers.extend(
            (
                task_root.name,
                task["jaxbench_task"],
                task["jaxbench_baseline_sha256"],
                task["reference_kernel_sha256"],
            )
        )
        for relative in ("instruction.md", "solution/kernel.py"):
            path = task_root / relative
            source = path.read_text(encoding="utf-8")
            records.append(
                {
                    "task_id": task_root.name,
                    "path": relative,
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "shingles": sorted(_shingles(source)),
                }
            )
    return {
        "schema_version": 1,
        "policy": "forbidden_from_all_training_splits",
        "documents": records,
        "identifiers": sorted(set(identifiers)),
    }


def _row_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_row_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_row_text(item) for item in value)
    return ""


def assert_training_content_clean(
    rows: Iterable[dict[str, Any]],
    signatures: dict[str, Any],
    *,
    threshold: float = 0.9,
) -> None:
    forbidden = signatures.get("documents", [])
    for index, row in enumerate(rows):
        source = _row_text(row)
        lowered = source.lower()
        for identifier in signatures.get("identifiers", ()):
            if str(identifier).lower() in lowered:
                raise Phase2ContaminationError(
                    f"PHASE2_IDENTIFIER_CONTAMINATION:row={index}:identifier={identifier}"
                )
        digest = hashlib.sha256(source.encode()).hexdigest()
        shingles = _shingles(source)
        for document in forbidden:
            if digest == document.get("sha256"):
                raise Phase2ContaminationError(
                    f"PHASE2_EXACT_CONTAMINATION:row={index}"
                )
            other = frozenset(document.get("shingles", ()))
            union = shingles | other
            similarity = len(shingles & other) / len(union) if union else 0.0
            containment = len(shingles & other) / len(other) if other else 0.0
            if similarity >= threshold or containment >= threshold:
                raise Phase2ContaminationError(
                    f"PHASE2_NEAR_CONTAMINATION:row={index}:"
                    f"similarity={similarity:.6f}:containment={containment:.6f}"
                )


def assert_project_training_content_clean(rows: Iterable[dict[str, Any]]) -> None:
    repo_root = Path(__file__).parents[3]
    releases = (
        repo_root / "data/pallas/benchmarks/phase2-v2",
        repo_root / "data/pallas/benchmarks/jaxbench-v1",
    )
    materialized_rows = tuple(rows)
    for release_root in releases:
        manifest_path = release_root / "manifest.json"
        signatures_path = release_root / "contamination-signatures.json"
        if not manifest_path.is_file() or not signatures_path.is_file():
            raise Phase2ContaminationError(
                f"BENCHMARK_FROZEN_SIGNATURES_REQUIRED:{release_root.name}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status_is_valid = manifest.get("status") == "frozen" or (
            manifest.get("status") == "candidate"
            and manifest.get("purpose") == "verifier_conformance_only"
        )
        if not status_is_valid:
            raise Phase2ContaminationError(
                f"BENCHMARK_FROZEN_SIGNATURES_REQUIRED:{release_root.name}"
            )
        if hashlib.sha256(signatures_path.read_bytes()).hexdigest() != manifest.get(
            "contamination_signatures_sha256"
        ):
            raise Phase2ContaminationError(
                f"BENCHMARK_SIGNATURE_BINDING_INVALID:{release_root.name}"
            )
        assert_training_content_clean(
            materialized_rows,
            json.loads(signatures_path.read_text(encoding="utf-8")),
        )
