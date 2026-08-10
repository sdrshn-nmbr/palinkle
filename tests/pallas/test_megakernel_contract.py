import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.megakernel_contract import (
    MutableStateContract,
    OutputContract,
    TaskExecutionContract,
    TensorContract,
    TopologyContract,
    validate_execution_record,
)


def _rpa_contract() -> TaskExecutionContract:
    return TaskExecutionContract(
        task_id="stacked-rpa-v7x-16",
        topology=TopologyContract(
            accelerator_family="v7x",
            device_count=16,
            host_count=4,
            chips_per_host=4,
            physical_topology=(2, 2, 4),
            mesh=(("data", 2), ("tensor", 8)),
            required_collectives=("all_gather",),
        ),
        inputs=(
            TensorContract(
                "queries",
                (32, 128, 16, 128),
                "bfloat16",
                sharding=("data", None, "tensor", None),
            ),
            TensorContract(
                "kv_cache",
                (4096, 16, 2, 128),
                "bfloat16",
                sharding=(None, "tensor", None, None),
            ),
        ),
        outputs=(
            OutputContract(
                "attention_output",
                (32, 128, 16, 128),
                "bfloat16",
                rtol=0.02,
                atol=0.01,
                sharding=("data", None, "tensor", None),
            ),
            OutputContract(
                "updated_kv_cache",
                (4096, 16, 2, 128),
                "bfloat16",
                rtol=0.0,
                atol=0.0,
                sharding=(None, "tensor", None, None),
            ),
        ),
        mutable_state=(
            MutableStateContract(
                input_name="kv_cache",
                output_name="updated_kv_cache",
                donated=True,
                mutation_required=True,
            ),
        ),
        correctness_seeds=(0, 1, 2),
    )


def _valid_record() -> dict[str, object]:
    return {
        "runtime": {
            "accelerator_family": "v7x",
            "device_count": 16,
            "host_count": 4,
            "chips_per_host": 4,
            "physical_topology": [2, 2, 4],
            "mesh": {"data": 2, "tensor": 8},
            "observed_collectives": ["all_gather"],
        },
        "outputs": {
            "attention_output": {
                "shape": [32, 128, 16, 128],
                "dtype": "bfloat16",
                "seeds_passed": [0, 1, 2],
                "artifact_sha256": "a" * 64,
            },
            "updated_kv_cache": {
                "shape": [4096, 16, 2, 128],
                "dtype": "bfloat16",
                "seeds_passed": [0, 1, 2],
                "artifact_sha256": "b" * 64,
            },
        },
        "mutable_state": {
            "kv_cache": {
                "before_sha256": "c" * 64,
                "after_sha256": "b" * 64,
                "aliased": True,
                "correct": True,
            }
        },
        "lowering": {
            "normal_pallas": True,
            "pallas_owned_outputs": ["attention_output", "updated_kv_cache"],
        },
        "profile": {
            "captured": True,
            "device_execution": True,
            "observed_collectives": ["all_gather"],
        },
    }


def test_distributed_multi_output_mutating_contract_passes() -> None:
    result = validate_execution_record(_rpa_contract(), _valid_record())

    assert result == {
        "reward": 1,
        "correct": True,
        "authentic": True,
        "profiled": True,
        "outputs_passed": 2,
        "state_transitions_passed": 1,
    }


def test_logical_mesh_must_cover_exact_physical_topology() -> None:
    with pytest.raises(G42HarnessError, match="TOPOLOGY_MESH_SIZE_INVALID"):
        TopologyContract(
            accelerator_family="v7x",
            device_count=16,
            host_count=4,
            chips_per_host=4,
            physical_topology=(2, 2, 4),
            mesh=(("data", 2), ("expert", 4)),
            required_collectives=("all_to_all",),
        )


def test_missing_secondary_output_fails_closed() -> None:
    record = _valid_record()
    del record["outputs"]["updated_kv_cache"]  # type: ignore[index]

    with pytest.raises(G42HarnessError, match="OUTPUT_SET_INVALID"):
        validate_execution_record(_rpa_contract(), record)


def test_required_mutation_cannot_be_satisfied_by_unchanged_state() -> None:
    record = _valid_record()
    state = record["mutable_state"]["kv_cache"]  # type: ignore[index]
    state["after_sha256"] = state["before_sha256"]

    with pytest.raises(G42HarnessError, match="STATE_MUTATION_MISSING"):
        validate_execution_record(_rpa_contract(), record)


def test_output_must_be_owned_by_normally_lowered_pallas() -> None:
    record = _valid_record()
    record["lowering"]["pallas_owned_outputs"] = ["attention_output"]  # type: ignore[index]

    with pytest.raises(G42HarnessError, match="PALLAS_OUTPUT_OWNERSHIP_INVALID"):
        validate_execution_record(_rpa_contract(), record)


def test_collective_must_be_seen_in_runtime_and_profile() -> None:
    record = _valid_record()
    record["profile"]["observed_collectives"] = []  # type: ignore[index]

    with pytest.raises(G42HarnessError, match="PROFILE_COLLECTIVES_INVALID"):
        validate_execution_record(_rpa_contract(), record)
