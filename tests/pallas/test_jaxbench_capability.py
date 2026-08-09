from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from opjax.pallas.jaxbench_capability import (
    EXPECTED_WORKLOAD_COUNT,
    JaxBenchCapabilityError,
    build_release,
    file_sha256,
    materialize_agent_workspace,
    render_public_specification,
    validate_release,
)
from opjax.pallas.phase2_contamination import (
    Phase2ContaminationError,
    assert_project_training_content_clean,
)


JAXBENCH = Path("/tmp/opjax-jaxbench.yzbH8p/repo")


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (JAXBENCH / ".git").is_dir():
        pytest.skip("pinned JAXBench checkout unavailable")
    root = tmp_path_factory.mktemp("jaxbench-capability") / "release"
    build_release(source_root=JAXBENCH, out_dir=root)
    return root


def test_public_spec_hides_implementation_and_input_generator() -> None:
    baseline = (JAXBENCH / "JAXBench/benchmark/8p_GEMM/baseline.py").read_text(
        encoding="utf-8"
    )

    specification = render_public_specification(
        workload_id="8p_GEMM", baseline_source=baseline
    )

    assert "def workload(A, B)" in specification
    assert "Dense matmul: C = A @ B" in specification
    assert '"M": 8192' in specification
    assert "def create_inputs" not in specification
    assert "jax.random" not in specification
    assert "return A @ B" not in specification


def test_public_spec_exposes_exact_tensor_contract(release: Path) -> None:
    specification = (release / "tasks/8p_GEMM/instruction.md").read_text()

    assert '"name": "A"' in specification
    assert '"shape": [\n        8192,\n        8192' in specification
    assert '"name": "B"' in specification
    assert '"dtype": "bfloat16"' in specification
    assert '"outputs"' in specification


def test_public_spec_exposes_exact_operation_semantics(release: Path) -> None:
    specification = (
        release / "tasks/21k_Gemm_Divide_Sum_Scaling/instruction.md"
    ).read_text()

    assert '"format": "canonical_python_ast_semantics_v1"' in specification
    assert '"value": 2.0' in specification
    assert '"value": 1.5' in specification
    assert '"arg": "axis"' in specification
    assert '"value": 1' in specification
    assert '"arg": "keepdims"' in specification
    assert '"value": true' in specification


def test_semantic_contract_closes_over_helpers_and_constants(release: Path) -> None:
    mla = (release / "tasks/3p_MLA_Attention/instruction.md").read_text()
    ragged = (release / "tasks/7p_Ragged_Paged_Attention/instruction.md").read_text()

    assert '"_compute_rope"' in mla
    assert '"_apply_rope"' in mla
    assert '"helper_functions"' in mla
    assert '"DEFAULT_MASK_VALUE"' in ragged
    assert '"module_values"' in ragged
    assert '"unresolved_names": []' in mla
    assert '"unresolved_names": []' in ragged


def test_release_wraps_all_original_tasks_and_eight_optimized_references(
    release: Path,
) -> None:
    result = validate_release(root=release, source_root=JAXBENCH)
    manifest = json.loads((release / "manifest.json").read_text())

    assert result["task_count"] == EXPECTED_WORKLOAD_COUNT == 50
    assert result["optimized_reference_count"] == 8
    assert (release / "contamination-signatures.json").is_file()
    assert (
        (release / "UPSTREAM_LICENSE")
        .read_text()
        .startswith("                                 Apache License")
    )
    assert manifest["shape_policy"] == "original_unmodified"
    assert manifest["execution_status"] == "worker_adapter_ready"
    assert manifest["scoreability_status"] == "original_shape_canary_only"
    assert manifest["worker_requirements_lock_sha256"]
    assert set(manifest["worker_source_sha256"]) == {
        "jaxbench_executable.py",
        "jaxbench_verifier.py",
        "jaxbench_worker.py",
    }
    assert all(
        task["shape_policy"] == "original_unmodified" for task in manifest["tasks"]
    )
    assert all(
        "@sha256:"
        in (release / task["path"] / "environment/Dockerfile").read_text(
            encoding="utf-8"
        )
        for task in manifest["tasks"]
    )


def test_verifier_patch_capture_commits_workspace_changes(release: Path) -> None:
    script = (release / "tasks/8p_GEMM/tests/test.sh").read_text(encoding="utf-8")

    assert "git -C /app add -A" in script
    assert "commit --allow-empty" in script
    assert "TPU_WORKER_REQUIRED >&2" in script
    assert script.endswith("exit 2\n")


def test_agent_workspace_contains_no_hidden_jaxbench_material(
    release: Path, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    materialize_agent_workspace(
        task_root=release / "tasks/8p_GEMM", destination=workspace
    )

    assert {path.name for path in workspace.iterdir()} == {
        "instruction.md",
        "kernel.py",
        "PALLAS_API.md",
        "dev_check.py",
    }
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in workspace.iterdir()
    )
    assert "def create_inputs" not in content
    assert "jax.random" not in content
    assert "return A @ B" not in content


def test_release_validation_rejects_hidden_reference_drift(
    release: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "release"
    shutil.copytree(release, copied)
    baseline = copied / "tasks/8p_GEMM/tests/jaxbench/baseline.py"
    baseline.write_text(baseline.read_text() + "\n# drift\n")

    with pytest.raises(JaxBenchCapabilityError, match="JAXBENCH_RELEASE_TASK_INVALID"):
        validate_release(root=copied, source_root=JAXBENCH)


def test_source_validation_rejects_untracked_reference(tmp_path: Path) -> None:
    checkout = tmp_path / "jaxbench"
    shutil.copytree(JAXBENCH, checkout)
    untracked = checkout / "JAXBench/benchmark/8p_GEMM/untracked.py"
    untracked.write_text("UNTRACKED = True\n")

    with pytest.raises(JaxBenchCapabilityError, match="JAXBENCH_CHECKOUT_DIRTY"):
        build_release(source_root=checkout, out_dir=tmp_path / "release")


def test_pinned_jaxbench_source_is_forbidden_from_training() -> None:
    baseline = (JAXBENCH / "JAXBench/benchmark/8p_GEMM/baseline.py").read_text(
        encoding="utf-8"
    )

    with pytest.raises(Phase2ContaminationError, match="CONTAMINATION"):
        assert_project_training_content_clean([{"code": baseline}])


def test_jaxbench_specification_is_forbidden_after_identifier_removal() -> None:
    root = Path(__file__).parents[2]
    instruction = (
        root / "data/pallas/benchmarks/jaxbench-v1/tasks/8p_GEMM/instruction.md"
    ).read_text()
    stripped = instruction.replace("8p_GEMM", "held_out_kernel")

    with pytest.raises(Phase2ContaminationError, match="NEAR_CONTAMINATION"):
        assert_project_training_content_clean([{"messages": [{"content": stripped}]}])


def test_legacy_original_shape_canary_is_explicitly_superseded() -> None:
    root = Path(__file__).parents[2]
    evidence_root = root / "data/pallas/runs/jaxbench-full-v1-canary"
    manifest = json.loads((evidence_root / "manifest.json").read_text())
    release = json.loads(
        (root / "data/pallas/benchmarks/jaxbench-v1/manifest.json").read_text()
    )

    assert manifest["status"] == "complete"
    assert manifest["shape_policy"] == "original_unmodified"
    assert manifest["benchmark_release_sha256"] != release["release_sha256"]
    gemm, megablox = manifest["results"]
    assert gemm["task_id"] == "8p_GEMM"
    assert gemm["correct"] is True
    assert gemm["original_shape"] == {"M": 8192, "K": 8192, "N": 28672}
    assert file_sha256(evidence_root / "canary-8p.out") == gemm["stdout_sha256"]
    assert megablox["status"] == "upstream_harness_error"
    assert megablox["candidate_attributable"] is False
    assert file_sha256(evidence_root / "canary-11p.out") == megablox["stdout_sha256"]
