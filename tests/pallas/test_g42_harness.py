import json
import shutil
import subprocess
from pathlib import Path

import pytest

from opjax.pallas.g42_curriculum import (
    build_benchmark_release,
    build_repair_release,
    validate_benchmark_release,
)
from opjax.pallas.g42_harness import (
    G42HarnessError,
    classify_verifier_result,
    create_agent_workspace,
    load_task_package,
    materialize_submission,
    parse_action,
    snapshot_workspace,
    summarize_horizons,
    validate_horizon_contract,
    validate_task_release,
    write_verifier_artifacts,
)

REPO_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = REPO_ROOT / "data" / "pallas" / "runs" / "g41-environment-corpus"


@pytest.fixture()
def task_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    build_repair_release(source_root=SOURCE_ROOT, out_dir=root)
    return root


def test_action_parser_requires_exactly_one_nonempty_action() -> None:
    assert parse_action("x\n```mswea_bash_command\npython dev_check.py kernel.py\n```") == {
        "command": "python dev_check.py kernel.py"
    }
    with pytest.raises(G42HarnessError, match="ACTION_COUNT_INVALID"):
        parse_action("no action")
    with pytest.raises(G42HarnessError, match="ACTION_COUNT_INVALID"):
        parse_action("```mswea_bash_command\na\n```\n```mswea_bash_command\nb\n```")


def test_horizon_contract_accepts_g42_and_g43_prefixes() -> None:
    validate_horizon_contract(turn_limit=3, snapshot_turns=(3,))
    validate_horizon_contract(turn_limit=6, snapshot_turns=(3, 6))
    with pytest.raises(G42HarnessError, match="HORIZON_CONTRACT_INVALID"):
        validate_horizon_contract(turn_limit=3, snapshot_turns=(3, 2))
    with pytest.raises(G42HarnessError, match="HORIZON_CONTRACT_INVALID"):
        validate_horizon_contract(turn_limit=3, snapshot_turns=(4,))


def test_release_is_balanced_and_preserves_all_source_rows(task_release: Path) -> None:
    validation = validate_task_release(task_release)
    manifest = json.loads((task_release / "manifest.json").read_text())
    assert validation["task_count"] == 36
    assert validation["training_count"] == 32
    assert validation["families"] == {
        "activation": 4,
        "binary_elementwise": 4,
        "gated_activation": 4,
        "matmul": 4,
        "normalization": 4,
        "row_reduction": 4,
        "softmax": 4,
        "transpose": 4,
    }
    assert len({record["source_row_id"] for record in manifest["task_records"]}) == 32


def test_agent_workspace_excludes_hidden_material_and_snapshots_prefix(task_release: Path, tmp_path: Path) -> None:
    manifest = json.loads((task_release / "manifest.json").read_text())
    task = load_task_package(task_release / manifest["tasks"][0])
    workspace = tmp_path / "workspace"
    record = create_agent_workspace(task, workspace)
    assert len(record["base_commit"]) == 40
    assert sorted(path.name for path in workspace.iterdir() if path.name != ".git") == [
        "PALLAS_API.md",
        "dev_check.py",
        "instruction.md",
        "kernel.py",
    ]
    assert not (workspace / "tests").exists()
    assert not (workspace / "solution").exists()
    (workspace / "kernel.py").write_text((workspace / "kernel.py").read_text() + "\n", encoding="utf-8")
    first = snapshot_workspace(workspace, turn=3, output_dir=tmp_path / "snapshots")
    second = snapshot_workspace(workspace, turn=6, output_dir=tmp_path / "snapshots")
    assert first["patch_sha256"] == second["patch_sha256"]
    assert first["kernel_sha256"] == second["kernel_sha256"]


def test_agent_container_cannot_see_hidden_material_host_or_network(
    task_release: Path, tmp_path: Path
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the isolation probe")
    manifest = json.loads((task_release / "manifest.json").read_text())
    task = load_task_package(task_release / manifest["tasks"][0])
    workspace = tmp_path / "isolated-workspace"
    create_agent_workspace(task, workspace)
    probe = """import os, pathlib, socket
assert not pathlib.Path('/tests').exists()
assert not pathlib.Path('/solution').exists()
assert not pathlib.Path('/Users/sudarshan/Code/opjax').exists()
assert 'TINKER_API_KEY' not in os.environ
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('network unexpectedly reachable')
"""
    process = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "-w",
            "/workspace",
            "python@sha256:9e869b0816f5537709825b49e62dc86d1c2691eff19b05c1d4dc3a07992cc052",
            "python",
            "-c",
            probe,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr


def test_materialize_submission_applies_patch_to_fresh_task_base(task_release: Path, tmp_path: Path) -> None:
    manifest = json.loads((task_release / "manifest.json").read_text())
    task = load_task_package(task_release / manifest["tasks"][0])
    workspace = tmp_path / "agent-workspace"
    create_agent_workspace(task, workspace)
    (workspace / "kernel.py").write_text("def workload(x):\n    return x\n", encoding="utf-8")
    snapshot = snapshot_workspace(workspace, turn=1, output_dir=tmp_path / "snapshots")

    materialized = materialize_submission(
        task=task,
        patch_path=tmp_path / "snapshots" / snapshot["patch_path"],
        destination=tmp_path / "verifier-workspace",
    )

    assert Path(materialized["kernel_path"]).read_text(encoding="utf-8") == "def workload(x):\n    return x\n"
    assert not (tmp_path / "verifier-workspace" / "tests").exists()


def test_materialize_submission_accepts_unchanged_noop_patch(
    task_release: Path, tmp_path: Path
) -> None:
    manifest = json.loads((task_release / "manifest.json").read_text())
    task = load_task_package(task_release / manifest["tasks"][0])
    empty_patch = tmp_path / "empty.patch"
    empty_patch.write_bytes(b"")

    materialized = materialize_submission(
        task=task,
        patch_path=empty_patch,
        destination=tmp_path / "verifier-workspace",
    )

    assert materialized["patch_sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert Path(materialized["kernel_path"]).read_bytes() == (
        task.root / "environment" / "starter" / "kernel.py"
    ).read_bytes()


def test_task_tampering_is_rejected(task_release: Path) -> None:
    manifest = json.loads((task_release / "manifest.json").read_text())
    task_root = task_release / manifest["tasks"][0]
    with (task_root / "instruction.md").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(G42HarnessError, match="TASK_HASH_MISMATCH"):
        load_task_package(task_root)


def test_verifier_reward_distinguishes_candidate_and_infrastructure_failures(tmp_path: Path) -> None:
    assert classify_verifier_result(
        {
            "passed": True,
            "stage": "verified",
            "stages": {
                "artifact_contract": True,
                "pallas_api": True,
                "tpu_compile": True,
                "full_shape_correctness": True,
                "normal_lowering": True,
                "runtime_safety": True,
                "profile": True,
            },
            "profile": {"admission": {"verified": True}},
        }
    ) == 1
    assert classify_verifier_result({"passed": True, "stage": "verified", "stages": {}}) == 0
    assert classify_verifier_result({"passed": False, "stage": "tpu_compile"}) == 0
    assert classify_verifier_result({"passed": False, "infrastructure_error": True}) == -1
    payload = write_verifier_artifacts(
        result={
            "passed": False,
            "stage": "runtime_safety",
            "error": "candidate DMA halt",
            "stages": {"artifact_contract": True, "pallas_api": True, "tpu_compile": True},
        },
        output_dir=tmp_path,
        task_id="task",
        kernel_sha256="a" * 64,
    )
    assert payload["reward"] == 0
    assert payload["failure_stage"] == "runtime_safety"
    assert {path.name for path in tmp_path.iterdir()} == {
        "ctrf.json",
        "reward.json",
        "run.log",
        "test-stdout.txt",
    }


def test_horizon_summary_uses_paired_prefixes() -> None:
    rows = [
        {"model_id": "base", "task_id": "a", "seed": 0, "turn": 3, "reward": 0},
        {"model_id": "base", "task_id": "a", "seed": 0, "turn": 6, "reward": 1},
        {"model_id": "base", "task_id": "b", "seed": 0, "turn": 3, "reward": 1},
        {"model_id": "base", "task_id": "b", "seed": 0, "turn": 6, "reward": 0},
    ]
    summary = summarize_horizons(rows)
    assert summary["transitions"]["fail_to_pass"] == 1
    assert summary["transitions"]["pass_to_fail"] == 1
    assert summary["models"]["base"] == {"k3_verified": 1, "k6_verified": 1}


def test_public_dev_check_rejects_reversed_blockspec(task_release: Path) -> None:
    manifest = json.loads((task_release / "manifest.json").read_text())
    record = next(item for item in manifest["task_records"] if item["mutation"] == "reversed_blockspec")
    task_root = task_release / "tasks" / record["task_id"]
    result = subprocess.run(
        ["python", str(task_root / "environment/public/dev_check.py"), str(task_root / "environment/starter/kernel.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "DEV_CHECK pallas_api" in result.stdout


def test_benchmark_release_uses_frozen_near_heldout_tasks(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    result = build_benchmark_release(
        diagnostic_config=REPO_ROOT / "config" / "pallas" / "gate4-diagnostic.json",
        out_dir=root,
    )
    assert result["counts"] == {"tasks": 4}
    assert validate_benchmark_release(root)["task_count"] == 4
    for relative in result["tasks"]:
        package = load_task_package(root / relative)
        assert package.mode == "benchmark"
        assert package.split == "near_heldout"
