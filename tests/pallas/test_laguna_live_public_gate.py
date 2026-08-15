from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.laguna_live_public_gate import (
    _failure_code,
    validate_live_public_gate,
)


ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "data/pallas/runs/phase32-base-capability/experiment.json"
RELEASE = ROOT / "data/pallas/benchmarks/jaxbench-phase31"
OUTPUT = (
    ROOT
    / "data/pallas/runs/laguna-speculator-training-v1/live-k6/public-gate"
)
SAMPLES = {
    arm: ROOT / f"data/pallas/runs/laguna-speculator-training-v1/live-k6/{arm}/samples"
    for arm in ("dflash", "dspark")
}


def test_failure_code_uses_last_nonempty_line() -> None:
    assert (
        _failure_code("warning\nKERNEL_IMPORT_FAILED:ModuleNotFoundError:jax.pallas\n")
        == "KERNEL_IMPORT_FAILED"
    )


def test_failure_code_handles_empty_output() -> None:
    assert _failure_code("\n") == "PUBLIC_DEV_CHECK_FAILED_WITHOUT_OUTPUT"


def test_frozen_public_gate_evidence_validates() -> None:
    result = validate_live_public_gate(
        experiment_path=EXPERIMENT,
        release_root=RELEASE,
        sample_roots=SAMPLES,
        output_root=OUTPUT,
    )
    assert result["arms"]["dflash"]["counts"]["public_gate_passes"] == 0
    assert result["arms"]["dspark"]["counts"]["public_gate_passes"] == 0


def test_public_gate_rejects_summary_hash_drift(tmp_path: Path) -> None:
    output = tmp_path / "public-gate"
    output.mkdir()
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    summary["conclusion"] = "drift"
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(G42HarnessError, match="SUMMARY_HASH_INVALID"):
        validate_live_public_gate(
            experiment_path=EXPERIMENT,
            release_root=RELEASE,
            sample_roots=SAMPLES,
            output_root=output,
        )
