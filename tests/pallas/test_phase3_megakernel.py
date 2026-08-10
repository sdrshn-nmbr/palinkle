from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.phase3_megakernel import (
    paired_bootstrap_interval,
    validate_manifest,
)


def test_paired_bootstrap_uses_paired_speedup_samples() -> None:
    speedup, lower, upper = paired_bootstrap_interval(
        optimized_ms=[10.0, 11.0, 9.0],
        baseline_ms=[20.0, 22.0, 18.0],
    )

    assert speedup == lower == upper == 2.0


def test_paired_bootstrap_rejects_unpaired_samples() -> None:
    with pytest.raises(G42HarnessError, match="PAIR_COUNT"):
        paired_bootstrap_interval(
            optimized_ms=[1.0, 2.0, 3.0],
            baseline_ms=[2.0, 4.0],
        )


def test_frozen_megakernel_evidence_validates() -> None:
    result = validate_manifest(
        path=Path("data/pallas/runs/phase3-megakernel-v0/manifest.json"),
        source_root=Path("references/sglang-jax"),
    )

    assert result["task_id"] == "kda-32k-varlen"
    assert result["speedup_ci95"][0] > 1.05
