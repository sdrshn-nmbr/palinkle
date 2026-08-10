import json
from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256
from opjax.pallas.phase3_baseline import (
    build_experiment,
    load_phase3_contract,
    validate_sample_matrix,
)


REPO_ROOT = Path(__file__).parents[2]
RELEASE_ROOT = REPO_ROOT / "data/pallas/benchmarks/jaxbench-v1"
CLOSEOUT_ROOT = REPO_ROOT / "data/pallas/runs/jaxbench-full-v1-closeout"


def test_phase3_contract_uses_exact_frozen_scoreable_set() -> None:
    contract = load_phase3_contract(
        release_root=RELEASE_ROOT,
        closeout_root=CLOSEOUT_ROOT,
    )

    assert contract.release_sha256 == (
        "4015d3db9e395d9f5d564f3eda2b971483c71c3e77e5ec4bf273fa17a4dcca0b"
    )
    assert len(contract.scoreable_tasks) == 47
    assert set(contract.excluded_tasks) == {
        "11p_Megablox_GMM",
        "16p_Mamba2_SSD",
        "2p_GQA_Attention",
    }
    assert not set(contract.scoreable_tasks) & set(contract.excluded_tasks)


def test_phase3_experiment_is_complete_paired_matrix() -> None:
    contract = load_phase3_contract(
        release_root=RELEASE_ROOT,
        closeout_root=CLOSEOUT_ROOT,
    )

    experiment = build_experiment(contract=contract)

    assert experiment["sampling"]["seeds"] == [0, 1, 2]
    assert experiment["sampling"]["turn_limit"] == 6
    assert experiment["sampling"]["snapshot_turns"] == [3, 6]
    assert experiment["counts"] == {
        "models": 2,
        "tasks": 47,
        "seeds": 3,
        "trajectories": 282,
        "snapshots": 564,
    }
    cells = {
        (cell["model_id"], cell["task_id"], cell["seed"])
        for cell in experiment["cells"]
    }
    assert len(cells) == 282


def test_sample_matrix_validation_rejects_missing_cells(tmp_path: Path) -> None:
    contract = load_phase3_contract(
        release_root=RELEASE_ROOT,
        closeout_root=CLOSEOUT_ROOT,
    )
    experiment = build_experiment(contract=contract)
    experiment["cells"] = experiment["cells"][:-1]
    experiment["counts"]["trajectories"] -= 1
    experiment["experiment_sha256"] = canonical_sha256(
        {key: value for key, value in experiment.items() if key != "experiment_sha256"}
    )
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(G42HarnessError, match="PHASE3_CELL_SET_INVALID"):
        validate_sample_matrix(path=path, contract=contract)
