from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError, file_sha256
from opjax.pallas.jaxbench_agent import (
    initialize_agent_workspace,
    load_agent_task,
)


REPO_ROOT = Path(__file__).parents[2]
RELEASE_ROOT = REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1"


def test_jaxbench_agent_workspace_contains_only_public_contract(tmp_path: Path) -> None:
    task = load_agent_task(release_root=RELEASE_ROOT, task_id="8p_GEMM")

    record = initialize_agent_workspace(task=task, destination=tmp_path / "workspace")

    workspace = tmp_path / "workspace"
    assert sorted(path.name for path in workspace.iterdir() if path.name != ".git") == [
        "PALLAS_API.md",
        "dev_check.py",
        "instruction.md",
        "kernel.py",
    ]
    assert not (workspace / "tests").exists()
    assert not (workspace / "solution").exists()
    assert record["task_id"] == "8p_GEMM"
    assert record["files"]["instruction.md"] == file_sha256(
        task.root / "instruction.md"
    )
    assert len(record["base_commit"]) == 40


def test_jaxbench_agent_rejects_unknown_task() -> None:
    with pytest.raises(G42HarnessError, match="JAXBENCH_AGENT_TASK_UNKNOWN"):
        load_agent_task(release_root=RELEASE_ROOT, task_id="missing")
