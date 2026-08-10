from __future__ import annotations

from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256
from opjax.pallas.phase31_experiment import (
    Phase31Contract,
    build_experiment,
    validate_experiment,
)


def _contract() -> Phase31Contract:
    return Phase31Contract(
        release_root=Path("/benchmark"),
        release_sha256="release",
        scoreable_tasks=("task",),
        task_hashes={"task": "task-hash"},
        validity_sha256="validity",
        calibration_sha256="calibration",
    )


def test_experiment_binds_provider_runtime_files() -> None:
    contract = _contract()
    experiment = build_experiment(
        contract=contract,
        release={
            "agent_environment": {"image": "image", "image_id": "image-id"},
            "bound_source_sha256": {"verifier": "hash"},
        },
    )
    validate_experiment(value=experiment, contract=contract)
    experiment["harness"]["provider_runtime_sha256"]["uv.lock"] = "tampered"
    payload = dict(experiment)
    payload.pop("experiment_sha256")
    experiment["experiment_sha256"] = canonical_sha256(payload)
    with pytest.raises(G42HarnessError, match="EXPERIMENT_INVALID"):
        validate_experiment(value=experiment, contract=contract)
