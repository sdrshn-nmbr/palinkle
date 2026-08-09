from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opjax.pallas.jaxbench_executable import compile_submission
from opjax.pallas.jaxbench_verifier import (
    inspect_pallas_owned_hlo,
    profile_proves_tpu_execution,
    verify_serialized_submission,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    baseline = tmp_path / "baseline.py"
    baseline.write_text(
        "import jax.numpy as jnp\n"
        "def create_inputs(dtype=jnp.float32):\n"
        "    return jnp.ones((2, 3), dtype), jnp.ones((3, 4), dtype)\n"
        "def workload(x, y):\n"
        "    return x @ y\n",
        encoding="utf-8",
    )
    task: dict[str, object] = {
        "task_id": "cpu-test",
        "baseline_sha256": _sha256(baseline),
        "tensor_schema": {
            "inputs": [
                {"name": "x", "shape": [2, 3], "dtype": "bfloat16"},
                {"name": "y", "shape": [3, 4], "dtype": "bfloat16"},
            ],
            "outputs": [{"shape": [2, 4], "dtype": "bfloat16"}],
        },
    }
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "def workload(x, y):\n"
        "    return x @ y\n",
        encoding="utf-8",
    )
    compiled = tmp_path / "compiled"
    compile_submission(
        task=task, kernel_path=kernel, out_dir=compiled, require_tpu=False
    )
    return task, baseline, compiled


def test_pristine_verifier_loads_serialized_executable_without_candidate_source(
    tmp_path: Path,
) -> None:
    task, baseline, compiled = _fixture(tmp_path)
    result = verify_serialized_submission(
        task=task,
        baseline_path=baseline,
        compiled_dir=compiled,
        out_dir=tmp_path / "verified",
        require_tpu=False,
        require_pallas=False,
        timing_rounds=5,
    )
    assert result["passed"] is True
    assert result["correct"] is True
    assert result["reward"] == 1
    assert not (tmp_path / "verified/kernel.py").exists()


def test_pristine_verifier_rejects_plain_executable_when_pallas_is_required(
    tmp_path: Path,
) -> None:
    task, baseline, compiled = _fixture(tmp_path)
    result = verify_serialized_submission(
        task=task,
        baseline_path=baseline,
        compiled_dir=compiled,
        out_dir=tmp_path / "verified",
        require_tpu=False,
        require_pallas=True,
        timing_rounds=5,
    )
    assert result["passed"] is False
    assert result["stage"] == "normal_lowering"
    assert result["reward"] == 0


def test_verifier_fails_closed_when_executable_hash_drifts(tmp_path: Path) -> None:
    task, baseline, compiled = _fixture(tmp_path)
    (compiled / "executable.bin").write_bytes(b"corrupt")
    try:
        verify_serialized_submission(
            task=task,
            baseline_path=baseline,
            compiled_dir=compiled,
            out_dir=tmp_path / "verified",
            require_tpu=False,
            require_pallas=False,
            timing_rounds=5,
        )
    except Exception as exc:
        assert "COMPILED_ARTIFACT_BINDING_INVALID" in str(exc)
    else:
        raise AssertionError("corrupt executable was accepted")


def test_result_artifact_is_machine_readable(tmp_path: Path) -> None:
    task, baseline, compiled = _fixture(tmp_path)
    result = verify_serialized_submission(
        task=task,
        baseline_path=baseline,
        compiled_dir=compiled,
        out_dir=tmp_path / "verified",
        require_tpu=False,
        require_pallas=False,
        timing_rounds=5,
    )
    assert json.loads((tmp_path / "verified/result.json").read_text()) == result


def test_profile_requires_annotation_loaded_executable_and_tpu_execution() -> None:
    complete = {
        "candidate_annotation_count": 3,
        "tpu_execute_event_count": 3,
        "loaded_executable_event_count": 3,
    }
    assert profile_proves_tpu_execution(complete) is True
    for field in complete:
        incomplete = dict(complete)
        incomplete[field] = 2
        assert profile_proves_tpu_execution(incomplete) is False


def test_hlo_authenticity_requires_pallas_owned_result_dataflow() -> None:
    pallas = '''
HloModule test
ENTRY %main (x: f32[2]) -> f32[2] {
  %x = f32[2] parameter(0)
  ROOT %kernel = f32[2] custom-call(%x), custom_call_target="tpu_custom_call", frontend_attributes={kernel_metadata={}}, metadata={op_name="jit(workload)/pallas_call"}
}
'''
    mixed = '''
HloModule test
ENTRY %main (x: f32[2], y: f32[2]) -> f32[2] {
  %x = f32[2] parameter(0)
  %y = f32[2] parameter(1)
  %kernel = f32[2] custom-call(%x), custom_call_target="tpu_custom_call", frontend_attributes={kernel_metadata={}}, metadata={op_name="jit(workload)/pallas_call"}
  %xla = f32[2] multiply(%x, %y)
  ROOT %mixed = f32[2] add(%kernel, %xla)
}
'''
    xla_then_pallas = '''
HloModule test
ENTRY %main (x: f32[2], y: f32[2]) -> f32[2] {
  %x = f32[2] parameter(0)
  %y = f32[2] parameter(1)
  %xla = f32[2] multiply(%x, %y)
  ROOT %kernel = f32[2] custom-call(%xla), custom_call_target="tpu_custom_call", frontend_attributes={kernel_metadata={}}, metadata={op_name="jit(workload)/pallas_call"}
}
'''
    structural_bypass = '''
HloModule test
ENTRY %main (x: f32[2]) -> (f32[2], f32[2]) {
  %x = f32[2] parameter(0)
  %kernel = f32[2] custom-call(%x), custom_call_target="tpu_custom_call", frontend_attributes={kernel_metadata={}}, metadata={op_name="jit(workload)/pallas_call"}
  ROOT %mixed = (f32[2], f32[2]) tuple(%kernel, %x)
}
'''

    assert inspect_pallas_owned_hlo(pallas)["authentic"] is True
    assert inspect_pallas_owned_hlo(mixed)["reason"] == (
        "HLO_COMPUTE_OUTSIDE_PALLAS:add,multiply"
    )
    assert inspect_pallas_owned_hlo(xla_then_pallas)["reason"] == (
        "HLO_COMPUTE_OUTSIDE_PALLAS:multiply"
    )
    assert inspect_pallas_owned_hlo(structural_bypass)["reason"] == (
        "HLO_RESULT_NOT_PALLAS_OWNED"
    )


def test_hlo_authenticity_rejects_xla_generated_tpu_custom_call() -> None:
    xla_custom_call = '''
HloModule test
ENTRY %main (x: f32[2]) -> f32[2] {
  %x = f32[2] parameter(0)
  ROOT %online-softmax = f32[2] custom-call(%x), custom_call_target="tpu_custom_call", frontend_attributes={tiling="1024,1024"}, metadata={op_name="online-softmax"}
}
'''

    result = inspect_pallas_owned_hlo(xla_custom_call)

    assert result["authentic"] is False
    assert result["reason"] == "HLO_NON_PALLAS_CUSTOM_CALL_REACHABLE"
    assert result["tpu_custom_call_count"] == 1
    assert result["pallas_custom_call_count"] == 0
