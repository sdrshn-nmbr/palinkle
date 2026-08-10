from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.jaxbench_compatibility import (
    JaxBenchCompatibilityError,
    _dynamic_inputs,
    compatibility_runner_sha256,
    compatibility_contract,
    validate_compatibility_evidence,
)
from opjax.pallas.jaxbench_executable import file_sha256


def test_megablox_preserves_pinned_static_argument_contract() -> None:
    contract = compatibility_contract("11p_Megablox_GMM")

    assert contract["adapter"] == "jax_jit_static_argnums"
    assert contract["static_argnums"] == [3]
    assert contract["required_accelerator_families"] == []


def test_compiled_call_excludes_static_arguments() -> None:
    assert _dynamic_inputs(("lhs", "rhs", "sizes", 256), [3]) == (
        "lhs",
        "rhs",
        "sizes",
    )


@pytest.mark.parametrize("task_id", ["2p_GQA_Attention", "16p_Mamba2_SSD"])
def test_attention_compatibility_requires_real_batch_sharding(task_id: str) -> None:
    contract = compatibility_contract(task_id)

    assert contract["adapter"] == "batch_axis_sharding"
    assert contract["static_argnums"] == []
    assert contract["required_accelerator_families"] == ["v5litepod"]
    assert contract["minimum_device_count"] == 4


def test_compatibility_contract_rejects_non_lane_task() -> None:
    with pytest.raises(JaxBenchCompatibilityError, match="TASK_NOT_IN_COMPATIBILITY_LANE"):
        compatibility_contract("8p_GEMM")


def test_evidence_validation_binds_release_task_and_runtime(tmp_path: Path) -> None:
    release = {
        "release_sha256": "a" * 64,
        "tasks": [
            {
                "task_id": "11p_Megablox_GMM",
                "task_sha256": "b" * 64,
                "baseline_sha256": "c" * 64,
            }
        ],
    }
    evidence = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_compatibility_probe",
        "task_id": "11p_Megablox_GMM",
        "release_sha256": "a" * 64,
        "task_sha256": "b" * 64,
        "baseline_sha256": "c" * 64,
        "status": "scoreable",
        "stage": "execute",
        "execution_contract": compatibility_contract("11p_Megablox_GMM"),
        "runner_sha256": compatibility_runner_sha256(),
        "runtime": {
            "backend": "tpu",
            "accelerator_type": "v5litepod-1",
            "device_count": 1,
        },
        "adapter_device_count": 1,
        "executable_sha256": "d" * 64,
    }
    release_path = tmp_path / "manifest.json"
    evidence_path = tmp_path / "evidence.json"
    release_path.write_text(json.dumps(release))
    evidence_path.write_text(json.dumps(evidence))

    assert validate_compatibility_evidence(
        release_manifest_path=release_path,
        evidence_path=evidence_path,
    )["status"] == "scoreable"


def test_evidence_validation_rejects_unsharded_v5e_for_attention(tmp_path: Path) -> None:
    release = {
        "release_sha256": "a" * 64,
        "tasks": [
            {
                "task_id": "2p_GQA_Attention",
                "task_sha256": "b" * 64,
                "baseline_sha256": "c" * 64,
            }
        ],
    }
    evidence = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_compatibility_probe",
        "task_id": "2p_GQA_Attention",
        "release_sha256": "a" * 64,
        "task_sha256": "b" * 64,
        "baseline_sha256": "c" * 64,
        "status": "scoreable",
        "stage": "execute",
        "execution_contract": compatibility_contract("2p_GQA_Attention"),
        "runner_sha256": compatibility_runner_sha256(),
        "runtime": {
            "backend": "tpu",
            "accelerator_type": "v5litepod-1",
            "device_count": 1,
        },
        "executable_sha256": "d" * 64,
    }
    release_path = tmp_path / "manifest.json"
    evidence_path = tmp_path / "evidence.json"
    release_path.write_text(json.dumps(release))
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(
        JaxBenchCompatibilityError,
        match="DEVICE_COUNT_INCOMPATIBLE",
    ):
        validate_compatibility_evidence(
            release_manifest_path=release_path,
            evidence_path=evidence_path,
        )


def test_checked_in_compatibility_lane_is_hash_bound_and_complete() -> None:
    repo_root = Path(__file__).parents[2]
    release_manifest = repo_root / "data/pallas/benchmarks/jaxbench-v1/manifest.json"
    evidence_root = repo_root / "data/pallas/runs/jaxbench-v1-compatibility-v1"
    manifest = json.loads((evidence_root / "manifest.json").read_text())

    assert manifest["frozen_phase2_impact"] == "none"
    assert manifest["frozen_v5e_scoreable_denominator"] == 47
    assert manifest["compatibility_task_count"] == 3
    assert manifest["compatibility_scoreable_count"] == 3
    assert manifest["worker"]["deletion_probe"] == "gcloud_describe_not_found"
    assert manifest["worker"]["deletion_observed_at"]
    assert manifest["worker"]["deletion_evidence_sha256"] == file_sha256(
        evidence_root / manifest["worker"]["deletion_evidence"]
    )
    assert manifest["runner_sha256"] == compatibility_runner_sha256()
    assert {task["task_id"] for task in manifest["tasks"]} == {
        "11p_Megablox_GMM",
        "16p_Mamba2_SSD",
        "2p_GQA_Attention",
    }
    for task in manifest["tasks"]:
        evidence_path = evidence_root / task["evidence"]
        assert task["evidence_sha256"] == file_sha256(evidence_path)
        validate_compatibility_evidence(
            release_manifest_path=release_manifest,
            evidence_path=evidence_path,
        )
