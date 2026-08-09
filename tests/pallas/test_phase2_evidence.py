from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.phase2_benchmark import validate_release


REPO_ROOT = Path(__file__).parents[2]
RELEASE_ROOT = REPO_ROOT / "data/pallas/benchmarks/phase2-v2"


def test_phase2_scaled_suite_is_conformance_only() -> None:
    validation = validate_release(RELEASE_ROOT)
    release = json.loads((RELEASE_ROOT / "manifest.json").read_text())

    assert validation["task_count"] == 10
    assert validation["compound_count"] == 8
    assert release["purpose"] == "verifier_conformance_only"
    assert release["status"] == "candidate"
    assert release["performance_subset"] == []
    assert "reference_evidence" not in release
    assert "acceptance_evidence" not in release


def test_phase2_scaled_suite_cannot_be_reported_as_capability_evidence() -> None:
    release = json.loads((RELEASE_ROOT / "manifest.json").read_text())

    assert release["benchmark_id"] != "opjax-jaxbench-full-v1"
    assert release["provenance"]["shape_policy"].startswith("scaled semantic mirrors")
    assert release["purpose"] == "verifier_conformance_only"
