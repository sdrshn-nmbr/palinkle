from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from opjax.pallas.jaxbench_worker import (
    DisposableWorkerFactory,
    JaxBenchWorkerError,
    build_calibration_patch,
    build_request,
    build_worker_bundle,
    grade_worker_submission,
    materialize_submission,
    prepare_sandbox_parent,
    tpu_network_allow_properties,
)


REPO_ROOT = Path(__file__).parents[2]
RELEASE_ROOT = REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1"


def _task(release: dict[str, object], task_id: str = "8p_GEMM") -> dict[str, object]:
    return next(task for task in release["tasks"] if task["task_id"] == task_id)


def _make_patch(task_root: Path, tmp_path: Path, source: str) -> Path:
    workspace = tmp_path / "agent"
    materialize_submission_base = task_root / "environment/starter/kernel.py"
    workspace.mkdir()
    (workspace / "kernel.py").write_bytes(materialize_submission_base.read_bytes())
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "add", "."], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        check=True,
    )
    (workspace / "kernel.py").write_text(source)
    patch = tmp_path / "model.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "HEAD"],
            capture_output=True,
            check=True,
        ).stdout
    )
    return patch


def test_materialized_submission_contains_only_public_base_plus_patch(
    tmp_path: Path,
) -> None:
    task_root = RELEASE_ROOT / "tasks/8p_GEMM"
    patch = _make_patch(task_root, tmp_path, "def workload(x, y):\n    return x @ y\n")
    result = materialize_submission(
        task_root=task_root,
        patch_path=patch,
        destination=tmp_path / "submission",
    )
    assert result["kernel_sha256"]
    assert not (tmp_path / "submission/tests").exists()
    assert not (tmp_path / "submission/solution").exists()


def test_sandbox_parent_is_traversable_but_not_listable(tmp_path: Path) -> None:
    parent = tmp_path / "private-parent"
    parent.mkdir(mode=0o700)
    prepare_sandbox_parent(parent, sandboxed=True)
    assert parent.stat().st_mode & 0o777 == 0o711


def test_tpu_network_policy_allows_only_private_worker_addresses() -> None:
    assert tpu_network_allow_properties(
        {"TPU_WORKER_HOSTNAMES": "10.2.3.4,10.2.3.5"}
    ) == [
        "--property=IPAddressAllow=10.2.3.4/32",
        "--property=IPAddressAllow=10.2.3.5/32",
    ]
    with pytest.raises(JaxBenchWorkerError, match="TPU_WORKER_HOSTNAMES_REQUIRED"):
        tpu_network_allow_properties({})
    with pytest.raises(JaxBenchWorkerError, match="TPU_WORKER_ADDRESS_INVALID"):
        tpu_network_allow_properties({"TPU_WORKER_HOSTNAMES": "169.254.169.254"})


def test_worker_bundle_excludes_optimized_reference(tmp_path: Path) -> None:
    bundle = build_worker_bundle(
        release_root=RELEASE_ROOT,
        task_id="8p_GEMM",
        destination=tmp_path / "bundle",
    )
    assert (bundle / "tasks/8p_GEMM/tests/jaxbench/baseline.py").is_file()
    assert not (bundle / "tasks/8p_GEMM/tests/jaxbench/optimized.py").exists()


def test_calibration_patch_reconstructs_hidden_optimized_reference(
    tmp_path: Path,
) -> None:
    task_root = RELEASE_ROOT / "tasks/8p_GEMM"
    patch = tmp_path / "reference.patch"
    record = build_calibration_patch(task_root=task_root, output=patch)
    materialized = materialize_submission(
        task_root=task_root,
        patch_path=patch,
        destination=tmp_path / "submission",
    )
    assert record["optimized_sha256"] == materialized["kernel_sha256"]


def test_cpu_worker_grades_serialized_submission_without_candidate_source(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    shutil.copytree(RELEASE_ROOT, release_root)
    release = json.loads((release_root / "manifest.json").read_text())
    task = _task(release, "41k_Gemm_Add_ReLU")
    task_root = release_root / task["path"]
    baseline = task_root / "tests/jaxbench/baseline.py"
    source = baseline.read_text().replace(
        "def create_inputs(dtype=jnp.float32):",
        "def create_inputs_hidden(dtype=jnp.float32):",
    )
    patch = _make_patch(task_root, tmp_path, source)
    request = build_request(release=release, task=task, patch_path=patch)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    response = grade_worker_submission(
        release_root=release_root,
        request_path=request_path,
        patch_path=patch,
        out_dir=tmp_path / "out",
        sandboxed=False,
        allow_cpu_test=True,
        allow_plain_jax_test=True,
    )
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert response["task_id"] == task["task_id"]
    assert result["passed"] is True
    assert result["reward"] == 1
    assert not (tmp_path / "out/kernel.py").exists()


def test_disposable_factory_destroys_worker_and_binds_response(tmp_path: Path) -> None:
    release = json.loads((RELEASE_ROOT / "manifest.json").read_text())
    task = _task(release)
    patch = _make_patch(
        RELEASE_ROOT / task["path"], tmp_path, "def workload(x, y):\n    return x @ y\n"
    )
    request = build_request(release=release, task=task, patch_path=patch)
    destination = tmp_path / "response"
    destination.mkdir()
    (destination / "result.json").write_text("{}\n")
    (destination / "reward.json").write_text("{}\n")
    (destination / "model.patch").write_bytes(patch.read_bytes())
    from opjax.pallas.jaxbench_executable import file_sha256

    def grade(_: str, value: dict[str, object], out: Path) -> dict[str, object]:
        return {
            **value,
            "result_sha256": file_sha256(out / "result.json"),
            "reward_sha256": file_sha256(out / "reward.json"),
            "model_patch_sha256": file_sha256(out / "model.patch"),
        }

    factory = DisposableWorkerFactory(
        create=lambda _: "worker-1",
        grade=grade,
        destroy=lambda _: datetime.now(timezone.utc).isoformat(),
        evidence={
            "candidate_user": "nobody",
            "execution_boundary": (
                "sandbox-compile-serialized-executable-pristine-verify"
            ),
        },
    )
    response = factory.run(request, destination)
    assert response["worker"]["identity"] == "worker-1"
    assert response["worker"]["destroyed_at"]
