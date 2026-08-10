from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.megakernel_catalog import (
    EXPECTED_FAMILIES,
    build_catalog_manifest,
    load_megakernel_catalog,
)


def test_catalog_registers_every_selected_megakernel_family() -> None:
    catalog = load_megakernel_catalog(Path("."))

    assert {task.family for task in catalog.tasks} == EXPECTED_FAMILIES
    assert len(catalog.tasks) == 32
    assert len({task.task_id for task in catalog.tasks}) == len(catalog.tasks)
    assert {repository.name for repository in catalog.repositories} == {
        "sglang-jax",
        "tpu-inference",
    }


def test_catalog_binds_every_implementation_and_oracle_file() -> None:
    catalog = load_megakernel_catalog(Path("."))

    for task in catalog.tasks:
        assert task.fusion_stages >= 2
        assert task.source_sha256
        assert set(task.source_sha256) == set(task.implementation_files)
        assert task.oracle_sha256
        assert set(task.oracle_sha256) == set(task.oracle_files)
        assert task.oracle_files
        assert task.minimum_devices >= 1


def test_only_hardware_attested_tasks_enter_scored_denominator() -> None:
    catalog = load_megakernel_catalog(Path("."))

    assert [task.task_id for task in catalog.scored_tasks] == ["sglang-kda-32k-varlen"]
    assert len(catalog.registered_tasks) == 31


def test_catalog_expresses_distributed_stateful_and_multi_output_tasks() -> None:
    catalog = load_megakernel_catalog(Path("."))

    assert any(task.minimum_devices == 8 for task in catalog.tasks)
    assert any(task.required_collectives == ("all_to_all",) for task in catalog.tasks)
    assert any(task.mutable_inputs for task in catalog.tasks)
    assert any(len(task.output_names) > 1 for task in catalog.tasks)
    assert all(task.minimum_devices <= 16 for task in catalog.tasks)


def test_catalog_manifest_binds_contract_capabilities_and_counts() -> None:
    manifest = build_catalog_manifest(Path("."))

    assert manifest["counts"] == {"total": 32, "admitted": 1, "registered": 31}
    assert manifest["contract_capabilities"] == {
        "exact_topology": True,
        "logical_mesh": True,
        "collective_attestation": True,
        "mutable_state": True,
        "multiple_outputs": True,
        "per_output_correctness": True,
        "pallas_output_ownership": True,
    }
    assert len(manifest["release_sha256"]) == 64


def test_catalog_fails_closed_on_source_revision_drift(tmp_path: Path) -> None:
    with pytest.raises(G42HarnessError, match="MEGAKERNEL_REPOSITORY_MISSING"):
        load_megakernel_catalog(tmp_path)
