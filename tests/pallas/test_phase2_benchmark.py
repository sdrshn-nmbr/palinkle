from __future__ import annotations

import difflib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tomli

from opjax.pallas.benchmarking import measure_interleaved
from opjax.pallas.candidate_worker import CandidateWorkerError, _load_module
from opjax.pallas.environment import verify_static
from opjax.pallas.candidate_policy import candidate_module_policy_error
from opjax.pallas.environment_runner import (
    classify_missing_candidate_result,
    consume_candidate_array,
    has_host_compute_in_workload,
)
from opjax.pallas.g42_harness import canonical_sha256, file_sha256, tree_sha256
from opjax.pallas.phase2_benchmark import (
    Phase2BenchmarkError,
    build_release,
    range_conditioned_megablox_candidate_source,
    select_performance_subset,
    timing_conditioned_megablox_candidate_source,
    validate_config,
    validate_release,
    zero_output_candidate_source,
)
from opjax.pallas.phase2_contamination import (
    Phase2ContaminationError,
    assert_training_content_clean,
)
from opjax.pallas.phase2_runner import (
    build_acceptance_patches,
    freeze_release,
    grade_submission,
    materialize_submission,
    validate_reference_evidence,
)
from opjax.pallas.phase2_task_artifacts import render_artifacts
from opjax.pallas.phase2_worker import (
    DisposableWorkerFactory,
    Phase2WorkerError,
    build_submission_request,
    build_worker_bundle,
    install_worker_output,
)
from opjax.pallas.task_semantics import (
    TaskSemanticsError,
    operation_specification,
    semantic_oracle,
)


REPO_ROOT = Path(__file__).parents[2]
CONFIG = REPO_ROOT / "config/pallas/phase2-benchmark.json"


def _write_kernel_patch(task_root: Path, path: Path, source: str) -> None:
    starter = (task_root / "environment/starter/kernel.py").read_text()
    path.write_text(
        "".join(
            difflib.unified_diff(
                starter.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="a/kernel.py",
                tofile="b/kernel.py",
            )
        ),
        encoding="utf-8",
    )


def test_parallel_worker_downloads_use_isolated_staging_directories(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)

    def install(name: str) -> None:
        destination = tmp_path / "matrix" / name

        def download(root: Path) -> None:
            output = root / "output"
            output.mkdir()
            (output / "identity.txt").write_text(name, encoding="utf-8")
            barrier.wait()

        install_worker_output(destination=destination, download=download)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(install, ("task-a", "task-b")))

    assert (tmp_path / "matrix/task-a/identity.txt").read_text() == "task-a"
    assert (tmp_path / "matrix/task-b/identity.txt").read_text() == "task-b"
    assert not list((tmp_path / "matrix").glob(".*-download-*"))


def test_phase2_config_is_compound_weighted_and_matches_pinned_semantics() -> None:
    config = validate_config(CONFIG)

    assert len(config["tasks"]) == 10
    assert sum(task["difficulty"] == "compound" for task in config["tasks"]) == 8
    assert len({task["family"] for task in config["tasks"]}) >= 6
    assert all(task["task_id"].startswith("p2-") for task in config["tasks"])
    assert all(task["semantic_parity"] is True for task in config["tasks"])
    assert {task["jaxbench_task"] for task in config["tasks"]} == {
        "1p_Flash_Attention",
        "8p_GEMM",
        "9p_SwiGLU_MLP",
        "11p_Megablox_GMM",
        "12p_RMSNorm",
        "24k_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish",
        "27k_Matmul_Mish_Mish",
        "41k_Gemm_Add_ReLU",
        "44k_Matmul_Divide_GELU",
        "50k_Matmul_GELU_Softmax",
    }
    megablox = next(task for task in config["tasks"] if task["shape_parity"])
    assert megablox["jaxbench_task"] == "11p_Megablox_GMM"
    assert megablox["output_dtype"] == "bfloat16"
    mish = next(
        task for task in config["tasks"] if task["operation"] == "matmul_mish_mish"
    )
    assert mish["input_shapes"][-1] == [256]


def test_phase2_oracle_signal_validation_is_explicitly_cpu_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_backends: list[str | None] = []
    devices = jax.devices

    def record_devices(backend: str | None = None) -> list[jax.Device]:
        requested_backends.append(backend)
        return devices(backend)

    monkeypatch.setattr(jax, "devices", record_devices)

    validate_config(CONFIG)

    assert requested_backends
    assert set(requested_backends) == {"cpu"}
    assert jax.config.jax_platforms == "cpu"


def test_zero_output_controls_are_authentic_policy_valid_candidates() -> None:
    config = validate_config(CONFIG)

    for task in config["tasks"]:
        source = zero_output_candidate_source(task)
        assert candidate_module_policy_error(
            source,
            allowed_entrypoints=tuple(task.get("allowed_pallas_entrypoints", ())),
        ) is None
        assert "jnp.zeros_like" in source


def test_acceptance_patch_manifest_covers_every_task_and_both_oracles(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    patches = tmp_path / "patches"
    build_release(config_path=CONFIG, out_dir=release)

    manifest = build_acceptance_patches(release_root=release, out_dir=patches)

    assert {item["task_id"] for item in manifest["zero_controls"]} == {
        item["task_id"]
        for item in json.loads((release / "manifest.json").read_text())["tasks"]
    }
    assert set(manifest["probes"]) == {"timing_zero", "strong_zero"}
    for item in [*manifest["zero_controls"], *manifest["probes"].values()]:
        assert file_sha256(patches / item["path"]) == item["sha256"]


def test_range_conditioned_megablox_probe_is_policy_valid() -> None:
    task = validate_config(CONFIG)["tasks"][-1]

    assert candidate_module_policy_error(
        range_conditioned_megablox_candidate_source(),
        allowed_entrypoints=tuple(task["allowed_pallas_entrypoints"]),
    ) is None
    assert candidate_module_policy_error(
        timing_conditioned_megablox_candidate_source(),
        allowed_entrypoints=tuple(task["allowed_pallas_entrypoints"]),
    ) is None


def test_megablox_acceptance_candidates_use_a_legal_full_vector_block() -> None:
    task = validate_config(CONFIG)["tasks"][-1]

    for source in (
        zero_output_candidate_source(task),
        range_conditioned_megablox_candidate_source(),
        timing_conditioned_megablox_candidate_source(),
    ):
        assert "pl.BlockSpec((128,)" in source
        assert "pl.BlockSpec((1,)" not in source


def test_compound_semantics_are_executable_and_exact() -> None:
    task = next(
        task
        for task in validate_config(CONFIG)["tasks"]
        if task["operation"] == "gemm_add_relu"
    )
    x = jnp.ones((2, 3), dtype=jnp.float32)
    weight = jnp.ones((3, 4), dtype=jnp.float32)
    bias = jnp.arange(4, dtype=jnp.float32)

    actual = semantic_oracle(task, x, weight, bias)
    expected = jnp.asarray([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=jnp.float32)

    assert actual.shape == (2, 4)
    assert actual.dtype == jnp.float32
    assert jnp.allclose(actual, expected)
    specification = operation_specification(task)
    assert specification["output_shape"] == [512, 384]
    assert "maximum" in specification["equation"]


def test_compound_semantics_reject_shape_valid_but_wrong_interfaces() -> None:
    task = next(
        task
        for task in validate_config(CONFIG)["tasks"]
        if task["operation"] == "gemm_add_relu"
    )
    mutant = json.loads(json.dumps(task))
    mutant["input_shapes"][2] = [383]

    with pytest.raises(TaskSemanticsError, match="TASK_GEMM_BIAS_CONTRACT_INVALID"):
        operation_specification(mutant)


def test_candidate_arrays_are_consumed_after_parent_verification(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npy"
    expected = np.arange(8, dtype=np.float32)
    np.save(path, expected, allow_pickle=False)

    actual = consume_candidate_array(path)

    np.testing.assert_array_equal(actual, expected)
    assert not path.exists()


def test_release_uses_real_harbor_schema_and_hides_verifier(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    validation = validate_release(release)

    assert validation["task_count"] == 10
    assert validation["compound_count"] == 8
    first = release / validation["task_paths"][0]
    task_toml = tomli.loads((first / "task.toml").read_text(encoding="utf-8"))
    assert task_toml["schema_version"] == "1.3"
    assert task_toml["verifier"]["environment_mode"] == "shared"
    assert (
        task_toml["metadata"]["authoritative_verifier"]
        == "external-disposable-tpu-worker"
    )
    assert task_toml["agent"]["network_mode"] == "no-network"
    assert task_toml["metadata"]["opjax_contract_version"] == "2.0"
    assert "collect" not in task_toml["verifier"]
    assert (first / "tests/Dockerfile").is_file()
    verifier_dockerfile = (first / "tests/Dockerfile").read_text(encoding="utf-8")
    assert "COPY task.json /tests/task.json" in verifier_dockerfile
    assert "COPY test.sh /tests/test.sh" in verifier_dockerfile
    test_script = (first / "tests/test.sh").read_text(encoding="utf-8")
    assert "git -C /app add -A" in test_script
    assert "> /logs/artifacts/model.patch" in test_script
    assert "TPU_WORKER_REQUIRED" in test_script
    assert "/logs/verifier/reward.json" in test_script
    assert test_script.endswith("exit 0\n")
    assert "exit $status" not in test_script
    assert not any(
        part in {"tests", "solution"}
        for relative in validation["agent_files"]
        for part in Path(relative).parts
    )


def test_every_reference_is_authentic_and_bound_to_public_semantics(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)

    for relative in validate_release(release)["task_paths"]:
        root = release / relative
        task = json.loads((root / "tests/task.json").read_text(encoding="utf-8"))
        source = (root / "solution/kernel.py").read_text(encoding="utf-8")
        assert verify_static(
            f"```python\n{source}\n```",
            allowed_pallas_entrypoints=tuple(task["allowed_pallas_entrypoints"]),
        ).passed is True
        assert (
            candidate_module_policy_error(
                source,
                allowed_entrypoints=tuple(task["allowed_pallas_entrypoints"]),
            )
            is None
        )
        assert task["public_specification"] == operation_specification(task)


def test_candidate_runtime_importer_admits_references_and_denies_host_modules(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    for row in json.loads((release / "manifest.json").read_text())["tasks"]:
        module = _load_module(release / row["path"] / "solution/kernel.py")
        assert callable(module.workload)

    malicious = tmp_path / "malicious.py"
    malicious.write_text("import atexit\ndef workload(x): return x\n")
    with pytest.raises(CandidateWorkerError, match="CANDIDATE_IMPORT_NOT_ALLOWED"):
        _load_module(malicious)


def test_release_validation_rejects_solution_visibility(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_root = next((release / "tasks").iterdir())
    leaked = task_root / "environment/starter/solution"
    leaked.mkdir()
    (leaked / "kernel.py").write_text("secret", encoding="utf-8")

    with pytest.raises(Phase2BenchmarkError, match="AGENT_HIDDEN_PATH_EXPOSED"):
        validate_release(release)


def test_worker_bundle_preserves_frozen_reference_link(tmp_path: Path) -> None:
    release = tmp_path / "data/pallas/benchmarks/phase2"
    build_release(config_path=CONFIG, out_dir=release)
    evidence = tmp_path / "data/pallas/runs/reference"
    evidence.mkdir(parents=True)
    evidence_manifest = {
        "evidence_sha256": "evidence",
        "performance_subset": [],
        "tasks": [],
    }
    evidence_path = evidence / "manifest.json"
    evidence_path.write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    release_manifest = json.loads((release / "manifest.json").read_text())
    release_manifest["status"] = "frozen"
    release_manifest["performance_subset"] = []
    release_manifest["reference_evidence"] = {
        "relative_manifest_path": "../../runs/reference/manifest.json",
        "manifest_sha256": file_sha256(evidence_path),
        "evidence_sha256": "evidence",
        "task_artifact_tree_sha256": canonical_sha256({}),
    }
    release_manifest.pop("release_sha256")
    release_manifest["release_sha256"] = canonical_sha256(release_manifest)
    (release / "manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_release(release)

    task_id = release_manifest["tasks"][0]["task_id"]
    bundled = build_worker_bundle(
        release, tmp_path / "bundle", task_id=task_id
    )

    manifest = json.loads((bundled / "manifest.json").read_text())
    assert manifest["kind"] == "opjax_phase2_sanitized_worker_bundle"
    assert [task["task_id"] for task in manifest["tasks"]] == [task_id]
    assert not list(bundled.rglob("solution"))
    hidden_source = (release / "tasks" / task_id / "solution/kernel.py").read_text()
    assert hidden_source not in "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in bundled.rglob("*")
        if path.is_file()
    )


def test_phase2_solution_and_near_copy_are_forbidden_from_training(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    signatures = json.loads(
        (release / "contamination-signatures.json").read_text()
    )
    solution = next((release / "tasks").iterdir()) / "solution/kernel.py"

    with pytest.raises(Phase2ContaminationError, match="EXACT_CONTAMINATION"):
        assert_training_content_clean([{"code": solution.read_text()}], signatures)
    near = solution.read_text().replace("workload", "candidate_workload", 1)
    with pytest.raises(Phase2ContaminationError, match="NEAR_CONTAMINATION"):
        assert_training_content_clean([{"code": near}], signatures)
    embedded = f"unrelated prefix\n{solution.read_text()}\nunrelated suffix"
    with pytest.raises(Phase2ContaminationError, match="NEAR_CONTAMINATION"):
        assert_training_content_clean([{"messages": [{"content": embedded}]}], signatures)
    task_id = next(iter(signatures["identifiers"]))
    with pytest.raises(Phase2ContaminationError, match="IDENTIFIER_CONTAMINATION"):
        assert_training_content_clean([{"text": f"benchmark {task_id}"}], signatures)


def test_release_validation_rejects_bundled_verifier_drift(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_root = next((release / "tasks").iterdir())
    bundled = task_root / "tests/opjax/pallas/scoring.py"
    bundled.write_text(bundled.read_text(encoding="utf-8") + "\nDRIFT = True\n")
    task_toml = tomli.loads((task_root / "task.toml").read_text(encoding="utf-8"))
    task_toml["metadata"]["task_sha256"] = ""

    with pytest.raises(Phase2BenchmarkError, match="VERIFIER_SOURCE_DRIFT"):
        validate_release(release)


def test_submission_is_materialized_from_only_the_captured_patch(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_root = next((release / "tasks").iterdir())
    patch = tmp_path / "model.patch"
    patch.write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "index d5c839a..2e84b1c 100644\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def workload(*inputs):\n"
        "-    ...\n"
        "+    return inputs[0]\n",
        encoding="utf-8",
    )

    record = materialize_submission(
        task_root=task_root,
        patch_path=patch,
        destination=tmp_path / "submission",
    )

    assert Path(record["kernel_path"]).read_text(encoding="utf-8").endswith(
        "return inputs[0]\n"
    )
    assert not (tmp_path / "submission/tests").exists()
    assert not (tmp_path / "submission/solution").exists()


def test_pier_collection_captures_uncommitted_edits(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_root = next((release / "tasks").iterdir())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for source, name in (
        (task_root / "environment/starter/kernel.py", "kernel.py"),
        (task_root / "instruction.md", "instruction.md"),
    ):
        (workspace / name).write_bytes(source.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        cwd=workspace,
        check=True,
    )
    (workspace / "kernel.py").write_text("def workload(*inputs):\n    return inputs[0]\n")
    (workspace / "new.py").write_text("VALUE = 1\n")
    command = (task_root / "pre_artifacts.sh").read_text().replace(
        "mkdir -p /logs/artifacts", "true"
    ).replace(
        "/logs/artifacts/model.patch", str(tmp_path / "model.patch")
    )
    subprocess.run(["bash", "-lc", command], cwd=workspace, check=True)

    destination = tmp_path / "materialized"
    materialize_submission(
        task_root=task_root,
        patch_path=tmp_path / "model.patch",
        destination=destination,
    )
    assert (destination / "kernel.py").read_text().endswith("return inputs[0]\n")
    assert (destination / "new.py").read_text() == "VALUE = 1\n"


def test_noop_submission_is_candidate_failure_without_tpu_access(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    patch = tmp_path / "model.patch"
    patch.write_bytes(b"")

    record = grade_submission(
        release_root=release,
        task_id="p2-gemm-1024x1024x2048",
        patch_path=patch,
        out_dir=tmp_path / "grade",
    )

    assert record["reward"] == 0
    assert record["stage"] == "artifact_contract"
    result = json.loads((tmp_path / "grade/result.json").read_text(encoding="utf-8"))
    assert result["hardware"]["execution"] == "not_started"


def test_static_valid_submission_is_routed_to_disposable_tpu_worker(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_id = "p2-gemm-1024x1024x2048"
    task_root = release / "tasks" / task_id
    patch = tmp_path / "model.patch"
    source = """import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def kernel(x_ref, y_ref, out_ref):
    out_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32).astype(jnp.bfloat16)
def workload(x, y):
    x_spec = pl.BlockSpec((1024, 1024), lambda i: (0, 0))
    y_spec = pl.BlockSpec((1024, 2048), lambda i: (0, 0))
    out_spec = pl.BlockSpec((1024, 2048), lambda i: (0, 0))
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((1024, 2048), jnp.bfloat16), grid=(1,), in_specs=(x_spec, y_spec), out_specs=out_spec)(x, y)
"""
    _write_kernel_patch(task_root, patch, source)

    submission = grade_submission(
        release_root=release,
        task_id=task_id,
        patch_path=patch,
        out_dir=tmp_path / "pending",
    )

    assert submission["status"] == "tpu_worker_required"
    assert submission["reward"] is None
    assert (tmp_path / "pending/request.json").is_file()
    assert not (tmp_path / "pending/result.json").exists()


@pytest.mark.parametrize(
    "source",
    [
        "def workload(:\n",
        "import jax.numpy as jnp\ndef workload(x, y):\n    return jnp.matmul(x, y)\n",
        """import jax
from jax.experimental import pallas as pl
def kernel(x, y, out): out[...] = x[...] @ y[...]
def workload(x, y):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((1024, 2048), x.dtype), in_specs=(spec, spec), out_specs=spec, grid=(1, 1), interpret=True)(x, y)
""",
        """import jax
from jax.experimental import pallas as pl
def kernel(x, y, out): out[...] = x[...] @ y[...]
def workload(x, y):
    if False:
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((1024, 2048), x.dtype))(x, y)
    return x
""",
    ],
)
def test_phase2_malformed_plain_interpreted_and_dead_candidates_fail_before_tpu(
    tmp_path: Path, source: str
) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_id = "p2-gemm-1024x1024x2048"
    task_root = release / "tasks" / task_id
    patch = tmp_path / "model.patch"
    _write_kernel_patch(task_root, patch, source)

    submission = grade_submission(
        release_root=release,
        task_id=task_id,
        patch_path=patch,
        out_dir=tmp_path / "grade",
    )

    assert submission["reward"] == 0
    result = json.loads((tmp_path / "grade/result.json").read_text())
    assert result["hardware"]["execution"] == "not_started"


def test_phase2_exact_reference_mimic_is_rejected_before_tpu(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_id = "p2-gemm-1024x1024x2048"
    task_root = release / "tasks" / task_id
    patch = tmp_path / "model.patch"
    _write_kernel_patch(
        task_root, patch, (task_root / "solution/kernel.py").read_text()
    )

    submission = grade_submission(
        release_root=release,
        task_id=task_id,
        patch_path=patch,
        out_dir=tmp_path / "grade",
    )

    assert submission["reward"] == 0
    result = json.loads((tmp_path / "grade/result.json").read_text())
    assert result["error"] == "REFERENCE_MIMIC"
    assert result["hardware"]["execution"] == "not_started"


def test_protocol_forgery_candidate_is_rejected_before_tpu(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    task_id = "p2-rmsnorm-b4-s128-d512"
    task_root = release / "tasks" / task_id
    patch = tmp_path / "model.patch"
    source = """import atexit
import __main__
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x_ref, w_ref, out_ref):
    out_ref[...] = x_ref[...] * w_ref[None, None, :]

def _forge():
    __main__.capture_lowering_case = lambda *args, **kwargs: {"passed": True}
    Path("seed-0-actual.npy").write_bytes(b"forged")
    print('{"phase":"complete","passed":true}')

atexit.register(_forge)

def workload(x, w):
    spec_x = pl.BlockSpec((1, 128, 512), lambda b, s: (b, s, 0))
    spec_w = pl.BlockSpec((512,), lambda b, s: (0,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(4, 1),
        in_specs=(spec_x, spec_w),
        out_specs=spec_x,
    )(x, w)
"""
    assert candidate_module_policy_error(source) == "CANDIDATE_IMPORT_NOT_ALLOWED"
    _write_kernel_patch(task_root, patch, source)

    submission = grade_submission(
        release_root=release,
        task_id=task_id,
        patch_path=patch,
        out_dir=tmp_path / "grade",
    )

    assert submission["reward"] == 0
    result = json.loads((tmp_path / "grade/result.json").read_text())
    assert result["stage"] == "pallas_api"
    assert result["error"] == "CANDIDATE_IMPORT_NOT_ALLOWED"
    assert result["hardware"]["execution"] == "not_started"


def test_candidate_cannot_mutate_imported_module_state_through_subscripts() -> None:
    source = """import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    jax.config.values["jax_platforms"] = "cpu"
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert (
        candidate_module_policy_error(source)
        == "CANDIDATE_SUBSCRIPT_MUTATION_NOT_ALLOWED"
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            "dummy, jax.jit = (0, workload)",
            "CANDIDATE_ATTRIBUTE_MUTATION_NOT_ALLOWED",
        ),
        (
            'dummy, jax.config.values["jax_disable_jit"] = (0, True)',
            "CANDIDATE_SUBSCRIPT_MUTATION_NOT_ALLOWED",
        ),
    ],
)
def test_candidate_cannot_hide_module_mutation_in_destructuring(
    mutation: str, error: str
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    {mutation}
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert candidate_module_policy_error(source) == error


def test_candidate_cannot_alias_privileged_state_as_pallas_reference() -> None:
    source = """import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    config_ref = jax.config.values
    config_ref["jax_disable_jit"] = True
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert candidate_module_policy_error(source) == (
        "CANDIDATE_REFERENCE_REBIND_NOT_ALLOWED"
    )


@pytest.mark.parametrize(
    "syntax",
    [
        "try:\n        int('x')\n    except* ValueError as pl:\n        pass",
        "type pl = jnp",
        "def helper[T]():\n        pass",
    ],
)
def test_candidate_rejects_python312_binding_and_control_flow(
    syntax: str,
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    {syntax}
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert candidate_module_policy_error(source) is not None


@pytest.mark.parametrize(
    ("binding", "call", "error"),
    [
        (
            "pl = jnp",
            'pl.save("/tmp/forged.npy", x_ref[...])',
            "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED",
        ),
        (
            "gmm = jnp.save",
            'gmm("/tmp/forged.npy", x_ref[...])',
            "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED",
        ),
        (
            "float = jnp.save",
            'float("/tmp/forged.npy", x_ref[...])',
            "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED",
        ),
        (
            "astype = jnp.save",
            'astype("/tmp/forged.npy", x_ref[...])',
            "CANDIDATE_CALL_NOT_ALLOWED:astype",
        ),
    ],
)
def test_candidate_cannot_rebind_privileged_call_names(
    binding: str, call: str, error: str
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    {binding}
    {call}
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    result = candidate_module_policy_error(source)
    assert result is not None
    if binding != "astype = jnp.save":
        assert result == error


@pytest.mark.parametrize("binding", ["pl", "jnp", "gmm", "float"])
def test_candidate_cannot_shadow_privileged_names_in_function_arguments(
    binding: str,
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref, {binding}):
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert (
        candidate_module_policy_error(source)
        == "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "for jax.jit in (workload,):\n        pass",
        "values = [0 for jax.jit in (workload,)]",
    ],
)
def test_candidate_cannot_mutate_module_attributes_through_iteration(
    statement: str,
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    {statement}
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert (
        candidate_module_policy_error(source)
        == "CANDIDATE_ATTRIBUTE_MUTATION_NOT_ALLOWED"
    )


@pytest.mark.parametrize("operation", ["load", "save", "savez", "fromfile"])
def test_candidate_cannot_use_jax_numpy_host_io(operation: str) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, out_ref):
    jnp.{operation}("/tmp/forged.npy", x_ref[...])
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert candidate_module_policy_error(source) is not None


@pytest.mark.parametrize(
    "call",
    [
        "pltpu.set_tpu_interpret_mode(True)",
        "pltpu.force_tpu_interpret_mode()",
        "pltpu.reset_tpu_interpret_mode_state()",
        "pl.enable_debug_checks(True)",
        "pl.lower_as_mlir(workload)",
        "pl.pallas_export_experimental(workload)",
    ],
)
def test_candidate_cannot_call_stateful_pallas_capabilities(call: str) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
def _kernel(x_ref, out_ref):
    {call}
    out_ref[...] = x_ref[...]
def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""

    assert candidate_module_policy_error(source) is not None


def test_public_candidate_policy_matches_authoritative_verdict(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)
    fixtures = []
    for task in json.loads((release / "manifest.json").read_text())["tasks"]:
        task_root = release / task["path"]
        fixtures.append((task_root, task_root / "solution/kernel.py"))
    malicious = release / "tasks/p2-gemm-1024x1024x2048/environment/starter/kernel.py"
    malicious.write_text(
        "import os\ndef workload(x):\n    return x\n", encoding="utf-8"
    )
    fixtures.append((malicious.parents[2], malicious))

    for task_root, kernel in fixtures:
        task = json.loads((task_root / "tests/task.json").read_text())
        authoritative = candidate_module_policy_error(
            kernel.read_text(),
            allowed_entrypoints=tuple(task.get("allowed_pallas_entrypoints", ())),
        )
        completed = subprocess.run(
            [sys.executable, str(task_root / "environment/public/dev_check.py"), str(kernel)],
            cwd=task_root / "environment/public",
            capture_output=True,
            text=True,
            check=False,
        )
        assert (completed.returncode == 0) is (authoritative is None)
        if authoritative is not None:
            assert authoritative in completed.stdout


@pytest.mark.parametrize(
    "workload_body",
    [
        "p = pl.pallas_call(_kernel, out_shape=x)(x)\n    return p + (x @ y)",
        "return pl.pallas_call(_kernel, out_shape=x)(x @ y)",
        "z = x @ y\n    return pl.pallas_call(_kernel, out_shape=z)(z)",
    ],
)
def test_candidate_workload_must_directly_return_pallas_on_original_inputs(
    workload_body: str,
) -> None:
    source = f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def _kernel(x_ref, y_ref, out_ref):
    out_ref[...] = x_ref[...] + y_ref[...]
def workload(x, y):
    {workload_body}
"""

    assert candidate_module_policy_error(source) is not None


def test_unexplained_candidate_exit_is_not_mislabeled_as_infrastructure() -> None:
    result = classify_missing_candidate_result(returncode=1, stderr="")

    assert result["phase"] == "execute"
    assert result["worker_recovery_required"] is False


def test_mixed_jax_and_pallas_workload_is_rejected() -> None:
    source = """import jax.numpy as jnp
from jax.experimental import pallas as pl
def kernel(x_ref, out_ref): out_ref[...] = x_ref[...]
def workload(x):
    p = pl.pallas_call(kernel, out_shape=x)(x)
    return p + jnp.square(x)
"""

    assert has_host_compute_in_workload(source) is True


def test_helper_dispatched_jax_compute_is_rejected() -> None:
    source = """import jax.numpy as jnp
from jax.experimental import pallas as pl
def kernel(x_ref, out_ref): out_ref[...] = x_ref[...]
def hidden_xla(x): return jnp.matmul(x, x)
def workload(x):
    p = pl.pallas_call(kernel, out_shape=x)(x)
    return p + hidden_xla(x)
"""

    assert has_host_compute_in_workload(source) is True


def test_aliased_and_unknown_workload_calls_are_rejected() -> None:
    aliased = """import jax.numpy as jnp
from jax.experimental import pallas as pl
def kernel(x_ref, out_ref): out_ref[...] = x_ref[...]
def workload(x):
    hidden = jnp.square
    return hidden(x)
"""
    helper = """from jax.experimental import pallas as pl
def hidden(x): return x
def workload(x): return hidden(x)
"""

    assert has_host_compute_in_workload(aliased) is True
    assert has_host_compute_in_workload(helper) is True


def test_reference_workloads_pass_the_closed_call_policy(tmp_path: Path) -> None:
    release = tmp_path / "release"
    build_release(config_path=CONFIG, out_dir=release)

    for task in json.loads((release / "manifest.json").read_text())["tasks"]:
        task_root = release / task["path"]
        verifier_task = json.loads((task_root / "tests/task.json").read_text())
        source = (task_root / "solution/kernel.py").read_text()
        assert not has_host_compute_in_workload(
            source,
            allowed_entrypoints=tuple(
                verifier_task.get("allowed_pallas_entrypoints", ())
            ),
        )


def test_disposable_worker_is_destroyed_and_response_is_hash_bound(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "model.patch"
    patch.write_text("patch\n")
    release = {"release_sha256": "release"}
    task = {"task_id": "task", "task_sha256": "task-sha"}
    request = build_submission_request(
        release_manifest=release, task=task, patch_path=patch
    )
    destroyed: list[str] = []

    def grade(identity: str, value: dict[str, object], destination: Path) -> dict[str, object]:
        (destination / "result.json").write_text("{}\n")
        (destination / "reward.json").write_text("{}\n")
        return {
            **value,
            "result_sha256": file_sha256(destination / "result.json"),
            "reward_sha256": file_sha256(destination / "reward.json"),
        }

    backend = DisposableWorkerFactory(
        create=lambda _: "worker-1",
        grade=grade,
        destroy=lambda identity: destroyed.append(identity) or "2026-08-07T00:00:00Z",
        worker_evidence={
            "candidate_user": "nobody",
            "sandbox_policy": "systemd-cgroup-v1",
            "service_account": "worker@example.invalid",
        },
    )
    destination = tmp_path / "worker"
    destination.mkdir()

    response = backend.run(request, destination)

    assert response["worker"]["disposable"] is True
    assert destroyed == ["worker-1"]


def test_disposable_worker_is_destroyed_when_grading_crashes(tmp_path: Path) -> None:
    destroyed: list[str] = []
    backend = DisposableWorkerFactory(
        create=lambda _: "worker-2",
        grade=lambda *_: (_ for _ in ()).throw(Phase2WorkerError("boom")),
        destroy=lambda identity: destroyed.append(identity) or "destroyed",
    )

    with pytest.raises(Phase2WorkerError, match="boom"):
        backend.run({"request_sha256": "invalid"}, tmp_path)
    assert destroyed == ["worker-2"]


def _write_valid_reference_evidence(release: Path, evidence: Path) -> None:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    records = []
    timing = measure_interleaved(
        candidate=lambda: 1.0,
        baseline=lambda: 1.01,
        rounds=9,
        seed=0,
    )
    runtime = {
        "python": "3.12.13",
        "jax": "0.10.1",
        "jaxlib": "0.10.1",
        "chex": "0.1.91",
        "libtpu": "0.0.41",
        "numpy": "2.2.6",
        "ml_dtypes": "0.5.3",
        "scipy": "1.15.3",
        "tomli": "2.2.1",
        "backend": "tpu",
        "device_kinds": ["TPU v5 lite"],
    }
    worker_fingerprint = {
        "name": "worker",
        "zone": "us-west4-a",
        "acceleratorType": "v5litepod-1",
        "runtimeVersion": "tpu-ubuntu2204-base",
    }
    worker_path = evidence / "worker-fingerprint.json"
    worker_path.write_text(json.dumps(worker_fingerprint))
    for task in manifest["tasks"]:
        root = evidence / "tasks" / task["task_id"]
        root.mkdir(parents=True)
        result = {
            "passed": True,
            "stage": "verified",
            "error": None,
            "infrastructure_error": False,
            "worker_recovery_required": False,
            "stages": {
                stage: True
                for stage in (
                    "artifact_contract",
                    "pallas_api",
                    "tpu_compile",
                    "full_shape_correctness",
                    "normal_lowering",
                    "runtime_safety",
                    "profile",
                )
            },
            "profile": {
                "admission": {"verified": True},
                "runtime": runtime,
                "timing": timing,
            },
        }
        (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (root / "run.log").write_text("verified\n", encoding="utf-8")
        reward = render_artifacts(root)
        records.append(
            {
                "task_id": task["task_id"],
                "task_sha256": task["task_sha256"],
                "reference_reward": reward["reward"],
                "failure_stage": reward["failure_stage"],
                "speedup": timing["speedup"],
                "speedup_ci95": timing["speedup_ci95"],
                "unstable": timing["unstable"],
                "runtime": runtime,
                "result_sha256": file_sha256(root / "result.json"),
                "reward_sha256": file_sha256(root / "reward.json"),
                "artifact_tree_sha256": tree_sha256(root),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "opjax_pallas_phase2_reference_evidence",
        "task_set_sha256": canonical_sha256(
            [
                {"task_id": task["task_id"], "task_sha256": task["task_sha256"]}
                for task in manifest["tasks"]
            ]
        ),
        "runtime": manifest["runtime"],
        "worker_fingerprint": worker_fingerprint,
        "worker_fingerprint_sha256": file_sha256(worker_path),
        "tasks": records,
        "performance_subset": [],
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    (evidence / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _rehash_evidence(evidence: Path, task_id: str) -> None:
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(task for task in manifest["tasks"] if task["task_id"] == task_id)
    root = evidence / "tasks" / task_id
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    record["runtime"] = result["profile"]["runtime"]
    record["result_sha256"] = file_sha256(root / "result.json")
    record["artifact_tree_sha256"] = tree_sha256(root)
    manifest.pop("evidence_sha256")
    manifest["evidence_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_reference_evidence_is_bound_to_runtime_results_and_artifacts(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    evidence = tmp_path / "evidence"
    build_release(config_path=CONFIG, out_dir=release)
    evidence.mkdir()
    _write_valid_reference_evidence(release, evidence)

    assert validate_reference_evidence(
        release_root=release, evidence_root=evidence
    )["task_count"] == 10

    tampered = evidence / "tasks/p2-gemm-1024x1024x2048/reward.json"
    tampered.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Phase2BenchmarkError, match="REFERENCE_ARTIFACT_HASH_INVALID"):
        validate_reference_evidence(release_root=release, evidence_root=evidence)


def test_reference_evidence_rejects_hash_valid_wrong_runtime(tmp_path: Path) -> None:
    release = tmp_path / "release"
    evidence = tmp_path / "evidence"
    build_release(config_path=CONFIG, out_dir=release)
    evidence.mkdir()
    _write_valid_reference_evidence(release, evidence)
    task_id = "p2-gemm-1024x1024x2048"
    result_path = evidence / "tasks" / task_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["profile"]["runtime"]["jax"] = "0.0.0"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _rehash_evidence(evidence, task_id)

    with pytest.raises(Phase2BenchmarkError, match="REFERENCE_RUNTIME_MISMATCH"):
        validate_reference_evidence(release_root=release, evidence_root=evidence)


def test_frozen_release_validation_enforces_reference_evidence_link(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    evidence = tmp_path / "evidence"
    build_release(config_path=CONFIG, out_dir=release)
    evidence.mkdir()
    _write_valid_reference_evidence(release, evidence)
    freeze_release(release_root=release, evidence_root=evidence)

    assert validate_release(release)["task_count"] == 10
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(
        Phase2BenchmarkError, match="FROZEN_REFERENCE_MANIFEST_INVALID"
    ):
        validate_release(release)


def test_performance_subset_is_reference_selected_and_fail_closed() -> None:
    task_ids = [task["task_id"] for task in validate_config(CONFIG)["tasks"]]
    evidence = [
        {
            "task_id": task_id,
            "reference_reward": 1,
            "speedup": 1.08 if index == 0 else 1.01,
            "speedup_ci95": [1.06, 1.10] if index == 0 else [0.99, 1.03],
            "unstable": False,
        }
        for index, task_id in enumerate(task_ids)
    ]

    assert select_performance_subset(task_ids=task_ids, evidence=evidence) == [
        task_ids[0]
    ]
    evidence[1]["reference_reward"] = 0
    with pytest.raises(Phase2BenchmarkError, match="REFERENCE_EVIDENCE_INCOMPLETE"):
        select_performance_subset(task_ids=task_ids, evidence=evidence)


@pytest.mark.parametrize(
    ("result", "expected_reward"),
    [
        (
            {
                "passed": True,
                "stage": "verified",
                "infrastructure_error": False,
                "stages": {
                    stage: True
                    for stage in (
                        "artifact_contract",
                        "pallas_api",
                        "tpu_compile",
                        "full_shape_correctness",
                        "normal_lowering",
                        "runtime_safety",
                        "profile",
                    )
                },
                "profile": {"admission": {"verified": False}},
            },
            0,
        ),
        (
            {
                "passed": False,
                "stage": "infrastructure",
                "infrastructure_error": True,
                "stages": {},
            },
            -1,
        ),
    ],
)
def test_artifact_reward_fails_closed(
    tmp_path: Path, result: dict[str, object], expected_reward: int
) -> None:
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    assert render_artifacts(tmp_path)["reward"] == expected_reward


def test_ctrf_has_one_case_per_stage_and_correctness_seed(tmp_path: Path) -> None:
    result = {
        "passed": True,
        "stage": "verified",
        "infrastructure_error": False,
        "stages": {
            stage: True
            for stage in (
                "artifact_contract",
                "pallas_api",
                "tpu_compile",
                "full_shape_correctness",
                "normal_lowering",
                "runtime_safety",
                "profile",
            )
        },
        "seed_results": [
            {"seed": seed, "passed": True} for seed in (0, 1, 2)
        ],
        "profile": {"admission": {"verified": True}, "timing": {}},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))

    render_artifacts(tmp_path)
    ctrf = json.loads((tmp_path / "ctrf.json").read_text())

    assert len(ctrf["tests"]) == 10
    assert all(case["status"] == "passed" for case in ctrf["tests"])
