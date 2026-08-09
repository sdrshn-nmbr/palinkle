from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import serialize_executable

from opjax.pallas.jaxbench_executable import (
    JaxBenchExecutableError,
    abstract_inputs,
    compile_submission,
    expected_trees,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "cpu-test",
        "tensor_schema": {
            "inputs": [
                {"name": "x", "shape": [2, 3], "dtype": "float32"},
                {"name": "y", "shape": [3, 4], "dtype": "float32"},
            ],
            "outputs": [{"shape": [2, 4], "dtype": "float32"}],
        },
    }


def test_abstract_inputs_and_trees_are_reconstructed_from_public_schema() -> None:
    task = _task()
    inputs = abstract_inputs(task)
    assert inputs == (
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        jax.ShapeDtypeStruct((3, 4), jnp.float32),
    )
    in_tree, out_tree = expected_trees(task)
    assert str(in_tree) == "PyTreeDef(((*, *), {}))"
    assert str(out_tree) == "PyTreeDef(*)"


def test_compile_submission_transfers_only_an_executable(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import jax.numpy as jnp\n"
        "def workload(x, y):\n"
        "    return jnp.matmul(x, y)\n",
        encoding="utf-8",
    )
    out = tmp_path / "compiled"
    result = compile_submission(
        task=_task(), kernel_path=kernel, out_dir=out, require_tpu=False
    )
    assert sorted(path.name for path in out.iterdir()) == [
        "compile.json",
        "executable.bin",
        "executable.hlo.txt",
        "stablehlo.mlir",
    ]
    assert result["kernel_sha256"]
    assert result["executable_sha256"]
    in_tree, out_tree = expected_trees(_task())
    loaded = serialize_executable.deserialize_and_load(
        (out / "executable.bin").read_bytes(),
        in_tree,
        out_tree,
        backend="cpu",
    )
    actual = loaded(jnp.ones((2, 3)), jnp.ones((3, 4)))
    assert actual.tolist() == [[3.0] * 4, [3.0] * 4]


def test_compile_submission_rejects_interpret_mode(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "def workload(x, y):\n"
        "    return pl.pallas_call(kernel, interpret=True)(x, y)\n",
        encoding="utf-8",
    )
    with pytest.raises(JaxBenchExecutableError, match="INTERPRET_MODE_FORBIDDEN"):
        compile_submission(
            task=_task(),
            kernel_path=kernel,
            out_dir=tmp_path / "out",
            require_tpu=False,
        )


def test_compile_manifest_matches_returned_record(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import jax.numpy as jnp\n"
        "def workload(x, y):\n"
        "    return jnp.matmul(x, y)\n",
        encoding="utf-8",
    )
    out = tmp_path / "compiled"
    result = compile_submission(
        task=_task(), kernel_path=kernel, out_dir=out, require_tpu=False
    )
    assert json.loads((out / "compile.json").read_text()) == result
