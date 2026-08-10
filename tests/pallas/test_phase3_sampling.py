import json
import threading
from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError, file_sha256
from opjax.pallas import phase3_sampling
from opjax.pallas.phase3_baseline import build_experiment, load_phase3_contract
from opjax.pallas.phase3_sampling import (
    cell_run_id,
    select_cells,
    validate_completed_run,
)


REPO_ROOT = Path(__file__).parents[2]


def _experiment() -> dict:
    contract = load_phase3_contract(
        release_root=REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1",
        closeout_root=REPO_ROOT / "data/pallas/runs/jaxbench-full-v1-closeout",
    )
    return build_experiment(contract=contract)


def test_select_cells_preserves_model_task_seed_identity() -> None:
    cells = select_cells(
        experiment=_experiment(),
        provider="tinker",
        task_ids={"8p_GEMM"},
        seeds={0, 2},
    )

    assert [(cell["task_id"], cell["seed"]) for cell in cells] == [
        ("8p_GEMM", 0),
        ("8p_GEMM", 2),
    ]
    assert cell_run_id(cells[0]) == "inkling-small-base--8p_GEMM--seed-0"


def test_select_cells_fails_closed_on_unknown_task() -> None:
    with pytest.raises(G42HarnessError, match="PHASE3_TASK_FILTER_INVALID"):
        select_cells(
            experiment=_experiment(),
            provider="tinker",
            task_ids={"missing"},
            seeds={0},
        )


def test_completed_run_validation_binds_snapshot_files(tmp_path: Path) -> None:
    cell = select_cells(
        experiment=_experiment(),
        provider="tinker",
        task_ids={"8p_GEMM"},
        seeds={0},
    )[0]
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    records = {}
    for turn in (3, 6):
        patch = snapshots / f"turn-{turn}.patch"
        kernel = snapshots / f"turn-{turn}-kernel.py"
        patch.write_bytes(b"")
        kernel.write_text("def workload(*inputs): ...\n", encoding="utf-8")
        records[str(turn)] = {
            "patch_sha256": file_sha256(patch),
            "kernel_sha256": file_sha256(kernel),
        }
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("{}\n", encoding="utf-8")
    manifest = {
        "kind": "opjax_phase3_jaxbench_agent_run",
        "task_id": cell["task_id"],
        "task_sha256": cell["task_sha256"],
        "seed": cell["seed"],
        "model": {
            "model_id": cell["model_id"],
            "model_revision": cell["model_revision"],
        },
        "turn_limit": 6,
        "snapshot_turns": [3, 6],
        "submitted": False,
        "snapshots": records,
        "trajectory_sha256": file_sha256(trajectory),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert validate_completed_run(path=tmp_path, cell=cell) == manifest

    (snapshots / "turn-6.patch").write_text("tampered", encoding="utf-8")
    with pytest.raises(G42HarnessError, match="PHASE3_RUN_SNAPSHOT_INVALID"):
        validate_completed_run(path=tmp_path, cell=cell)


def test_sglang_sampling_runs_independent_cells_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_phase3_contract(
        release_root=REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1",
        closeout_root=REPO_ROOT / "data/pallas/runs/jaxbench-full-v1-closeout",
    )
    experiment = build_experiment(contract=contract)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_run(**_: object) -> None:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1

    monkeypatch.setattr(phase3_sampling, "run_jaxbench_agent", fake_run)
    monkeypatch.setattr(
        phase3_sampling,
        "assemble_sample_manifest",
        lambda **_: {"kind": "test-manifest"},
    )

    result = phase3_sampling.sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        output_root=tmp_path,
        generate=lambda messages, sampling: {},
        runtime_revision="runtime",
        precision="bfloat16",
        task_ids={"8p_GEMM"},
        seeds={0, 1},
        max_concurrency=2,
    )

    assert result == {"kind": "test-manifest"}
    assert maximum_active == 2
