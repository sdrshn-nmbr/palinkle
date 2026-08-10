from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp

from opjax.pallas.phase31_benchmark import build_release, validate_release
from opjax.pallas.phase31_oracle import compare_output, derive_input_case, oracle_contract
from opjax.pallas.phase31_public import render_dev_check


REPO_ROOT = Path(__file__).parents[2]
SOURCE_RELEASE = REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1"
SOURCE_CHECKOUT = REPO_ROOT / "references/accelerator-agents"


def test_phase31_release_preserves_all_tasks_and_binds_new_public_contract(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    manifest = build_release(
        source_release=SOURCE_RELEASE,
        source_checkout=SOURCE_CHECKOUT,
        out_dir=release,
        agent_image="opjax-phase31-agent:jax-0.10.1",
        agent_image_id="sha256:" + "a" * 64,
    )
    assert manifest["task_count"] == 50
    assert manifest["source_release_sha256"] == json.loads(
        (SOURCE_RELEASE / "manifest.json").read_text()
    )["release_sha256"]
    assert set(manifest["action_protocol"]["native_tools"]) >= {
        "bash",
        "shell",
        "read",
        "write",
        "edit",
    }
    assert validate_release(root=release, source_release=SOURCE_RELEASE)["task_count"] == 50
    task = json.loads((release / "tasks/8p_GEMM/tests/task.json").read_text())
    assert task["oracle_contract"]["input_cases"] == [
        "jaxbench-original",
        "derived-seed-1",
        "derived-seed-2",
    ]


def test_public_dev_check_traces_real_pallas_and_rejects_plain_jax(
    tmp_path: Path,
) -> None:
    task = json.loads(
        (SOURCE_RELEASE / "tasks/8p_GEMM/tests/task.json").read_text()
    )
    check = tmp_path / "dev_check.py"
    check.write_text(render_dev_check(task["tensor_schema"]), encoding="utf-8")
    shutil.copy2(
        SOURCE_CHECKOUT / "JAXBench/benchmark/8p_GEMM/optimized.py",
        tmp_path / "kernel.py",
    )
    accepted = subprocess.run(
        [sys.executable, "dev_check.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    (tmp_path / "kernel.py").write_text(
        "def workload(x, y):\n    return x @ y\n", encoding="utf-8"
    )
    rejected = subprocess.run(
        [sys.executable, "dev_check.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "PALLAS_PRIMITIVE_REQUIRED" in rejected.stderr


def test_oracle_derives_hidden_values_and_rejects_zero_output() -> None:
    contract = oracle_contract(["x", "mask"], "float32")
    original = (jnp.zeros((4, 4)), jnp.zeros((4, 4)))
    derived = derive_input_case(original, contract=contract, seed=1)
    assert not jnp.array_equal(derived[0], original[0])
    assert jnp.array_equal(derived[1], original[1])
    expected = jnp.array([1e-7, -2e-7], dtype=jnp.float32)
    exact = compare_output(expected, expected, contract=contract)
    zero = compare_output(expected, jnp.zeros_like(expected), contract=contract)
    assert exact["correct"] is True
    assert zero["correct"] is False
    assert zero["normalized_max_error"] == 1.0
