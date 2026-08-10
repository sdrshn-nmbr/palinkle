"""Execution contracts for distributed, stateful, multi-output megakernels."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise G42HarnessError(f"{code}:{detail}")


@dataclass(frozen=True)
class TensorContract:
    name: str
    global_shape: tuple[int, ...]
    dtype: str
    sharding: tuple[str | None, ...]

    def __post_init__(self) -> None:
        _require(bool(self.name), "TENSOR_NAME_INVALID", repr(self.name))
        _require(
            bool(self.global_shape) and all(size > 0 for size in self.global_shape),
            "TENSOR_SHAPE_INVALID",
            self.name,
        )
        _require(bool(self.dtype), "TENSOR_DTYPE_INVALID", self.name)
        _require(
            len(self.sharding) == len(self.global_shape),
            "TENSOR_SHARDING_RANK_INVALID",
            self.name,
        )


@dataclass(frozen=True)
class OutputContract(TensorContract):
    rtol: float
    atol: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(
            math.isfinite(self.rtol)
            and math.isfinite(self.atol)
            and self.rtol >= 0
            and self.atol >= 0,
            "OUTPUT_TOLERANCE_INVALID",
            self.name,
        )


@dataclass(frozen=True)
class MutableStateContract:
    input_name: str
    output_name: str
    donated: bool
    mutation_required: bool


@dataclass(frozen=True)
class TopologyContract:
    accelerator_family: str
    device_count: int
    host_count: int
    chips_per_host: int
    physical_topology: tuple[int, ...]
    mesh: tuple[tuple[str, int], ...]
    required_collectives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(
            self.accelerator_family in {"v5e", "v5p", "v6e", "v7x"},
            "TOPOLOGY_ACCELERATOR_INVALID",
            self.accelerator_family,
        )
        _require(self.device_count > 0, "TOPOLOGY_DEVICE_COUNT_INVALID", str(self.device_count))
        _require(
            self.host_count > 0
            and self.chips_per_host > 0
            and self.host_count * self.chips_per_host == self.device_count,
            "TOPOLOGY_HOST_LAYOUT_INVALID",
            f"hosts={self.host_count} chips={self.chips_per_host}",
        )
        _require(
            bool(self.physical_topology)
            and all(size > 0 for size in self.physical_topology)
            and math.prod(self.physical_topology) == self.device_count,
            "TOPOLOGY_PHYSICAL_SHAPE_INVALID",
            repr(self.physical_topology),
        )
        names = [name for name, _ in self.mesh]
        _require(
            bool(self.mesh)
            and len(names) == len(set(names))
            and all(name and size > 0 for name, size in self.mesh),
            "TOPOLOGY_MESH_INVALID",
            repr(self.mesh),
        )
        mesh_size = math.prod(size for _, size in self.mesh)
        _require(
            mesh_size == self.device_count,
            "TOPOLOGY_MESH_SIZE_INVALID",
            f"mesh={mesh_size} devices={self.device_count}",
        )
        _require(
            len(self.required_collectives) == len(set(self.required_collectives)),
            "TOPOLOGY_COLLECTIVES_INVALID",
            repr(self.required_collectives),
        )


@dataclass(frozen=True)
class TaskExecutionContract:
    task_id: str
    topology: TopologyContract
    inputs: tuple[TensorContract, ...]
    outputs: tuple[OutputContract, ...]
    mutable_state: tuple[MutableStateContract, ...]
    correctness_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        input_names = [tensor.name for tensor in self.inputs]
        output_names = [tensor.name for tensor in self.outputs]
        _require(bool(self.task_id), "TASK_ID_INVALID", repr(self.task_id))
        _require(
            bool(self.inputs) and len(input_names) == len(set(input_names)),
            "INPUT_SET_INVALID",
            self.task_id,
        )
        _require(
            bool(self.outputs) and len(output_names) == len(set(output_names)),
            "OUTPUT_SET_INVALID",
            self.task_id,
        )
        _require(
            bool(self.correctness_seeds)
            and tuple(sorted(set(self.correctness_seeds))) == self.correctness_seeds,
            "CORRECTNESS_SEEDS_INVALID",
            self.task_id,
        )
        state_inputs: set[str] = set()
        state_outputs: set[str] = set()
        mesh_axes = {name for name, _ in self.topology.mesh}
        for tensor in (*self.inputs, *self.outputs):
            unknown_axes = {axis for axis in tensor.sharding if axis is not None} - mesh_axes
            _require(
                not unknown_axes,
                "TENSOR_SHARDING_AXIS_INVALID",
                f"{tensor.name}:{sorted(unknown_axes)}",
            )
        for state in self.mutable_state:
            _require(
                state.input_name in input_names and state.output_name in output_names,
                "STATE_TENSOR_INVALID",
                f"{state.input_name}->{state.output_name}",
            )
            _require(
                state.input_name not in state_inputs and state.output_name not in state_outputs,
                "STATE_ALIAS_DUPLICATE",
                f"{state.input_name}->{state.output_name}",
            )
            state_inputs.add(state.input_name)
            state_outputs.add(state.output_name)


def _object(value: Any, code: str) -> dict[str, Any]:
    _require(isinstance(value, dict), code, type(value).__name__)
    return value


def _string_set(value: Any, code: str) -> set[str]:
    _require(
        isinstance(value, list) and all(isinstance(item, str) for item in value),
        code,
        repr(value),
    )
    return set(value)


def validate_execution_record(
    contract: TaskExecutionContract, record: dict[str, Any]
) -> dict[str, Any]:
    runtime = _object(record.get("runtime"), "RUNTIME_RECORD_INVALID")
    _require(
        runtime.get("accelerator_family") == contract.topology.accelerator_family
        and runtime.get("device_count") == contract.topology.device_count
        and runtime.get("host_count") == contract.topology.host_count
        and runtime.get("chips_per_host") == contract.topology.chips_per_host
        and runtime.get("physical_topology")
        == list(contract.topology.physical_topology),
        "RUNTIME_TOPOLOGY_INVALID",
        contract.task_id,
    )
    expected_mesh = dict(contract.topology.mesh)
    _require(runtime.get("mesh") == expected_mesh, "RUNTIME_MESH_INVALID", contract.task_id)
    required_collectives = set(contract.topology.required_collectives)
    runtime_collectives = _string_set(
        runtime.get("observed_collectives"), "RUNTIME_COLLECTIVES_INVALID"
    )
    _require(
        required_collectives <= runtime_collectives,
        "RUNTIME_COLLECTIVES_INVALID",
        contract.task_id,
    )

    outputs = _object(record.get("outputs"), "OUTPUT_RECORD_INVALID")
    expected_outputs = {output.name: output for output in contract.outputs}
    _require(
        set(outputs) == set(expected_outputs),
        "OUTPUT_SET_INVALID",
        contract.task_id,
    )
    for name, expected in expected_outputs.items():
        observed = _object(outputs[name], "OUTPUT_RECORD_INVALID")
        _require(
            observed.get("shape") == list(expected.global_shape),
            "OUTPUT_SHAPE_INVALID",
            name,
        )
        _require(observed.get("dtype") == expected.dtype, "OUTPUT_DTYPE_INVALID", name)
        _require(
            observed.get("seeds_passed") == list(contract.correctness_seeds),
            "OUTPUT_CORRECTNESS_INVALID",
            name,
        )
        _require(
            isinstance(observed.get("artifact_sha256"), str)
            and _SHA256_PATTERN.fullmatch(observed["artifact_sha256"]) is not None,
            "OUTPUT_ARTIFACT_INVALID",
            name,
        )

    state_records = _object(record.get("mutable_state"), "STATE_RECORD_INVALID")
    expected_states = {state.input_name: state for state in contract.mutable_state}
    _require(
        set(state_records) == set(expected_states),
        "STATE_SET_INVALID",
        contract.task_id,
    )
    for name, expected in expected_states.items():
        observed = _object(state_records[name], "STATE_RECORD_INVALID")
        before = observed.get("before_sha256")
        after = observed.get("after_sha256")
        _require(
            isinstance(before, str)
            and isinstance(after, str)
            and _SHA256_PATTERN.fullmatch(before) is not None
            and _SHA256_PATTERN.fullmatch(after) is not None,
            "STATE_ARTIFACT_INVALID",
            name,
        )
        if expected.mutation_required:
            _require(before != after, "STATE_MUTATION_MISSING", name)
        _require(observed.get("correct") is True, "STATE_CORRECTNESS_INVALID", name)
        if expected.donated:
            _require(observed.get("aliased") is True, "STATE_ALIAS_INVALID", name)
        output_hash = outputs[expected.output_name]["artifact_sha256"]
        _require(after == output_hash, "STATE_OUTPUT_HASH_INVALID", name)

    lowering = _object(record.get("lowering"), "LOWERING_RECORD_INVALID")
    _require(lowering.get("normal_pallas") is True, "NORMAL_PALLAS_REQUIRED", contract.task_id)
    owned_outputs = _string_set(
        lowering.get("pallas_owned_outputs"), "PALLAS_OUTPUT_OWNERSHIP_INVALID"
    )
    _require(
        owned_outputs == set(expected_outputs),
        "PALLAS_OUTPUT_OWNERSHIP_INVALID",
        contract.task_id,
    )

    profile = _object(record.get("profile"), "PROFILE_RECORD_INVALID")
    _require(
        profile.get("captured") is True and profile.get("device_execution") is True,
        "PROFILE_EVIDENCE_INVALID",
        contract.task_id,
    )
    profile_collectives = _string_set(
        profile.get("observed_collectives"), "PROFILE_COLLECTIVES_INVALID"
    )
    _require(
        required_collectives <= profile_collectives,
        "PROFILE_COLLECTIVES_INVALID",
        contract.task_id,
    )
    return {
        "reward": 1,
        "correct": True,
        "authentic": True,
        "profiled": True,
        "outputs_passed": len(contract.outputs),
        "state_transitions_passed": len(contract.mutable_state),
    }
