import json
import sys
from pathlib import Path

from opjax.pallas.g42_experiment import _family_gate, _paired_model_deltas, verify_release
from opjax.pallas.g42_harness import canonical_sha256, file_sha256


def test_verify_release_writes_hashed_results(tmp_path: Path) -> None:
    unit_id = "base--task--seed-0--turn-3"
    unit = tmp_path / "units" / unit_id
    unit.mkdir(parents=True)
    (unit / "task.json").write_text('{"task_id":"task"}\n', encoding="utf-8")
    (unit / "kernel.py").write_text("def workload(x):\n    return x\n", encoding="utf-8")
    (unit / "model.patch").write_text("", encoding="utf-8")
    (unit / "trajectory.json").write_text("{}\n", encoding="utf-8")
    record = {
        "unit_id": unit_id,
        "model_id": "inkling-small-base",
        "checkpoint": None,
        "task_id": "task",
        "family": "add",
        "task_sha256": "a" * 64,
        "seed": 0,
        "turn": 3,
        "patch_sha256": file_sha256(unit / "model.patch"),
        "kernel_sha256": file_sha256(unit / "kernel.py"),
        "trajectory_sha256": file_sha256(unit / "trajectory.json"),
    }
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g42_verifier_input_release",
        "sample_release_sha256": "b" * 64,
        "benchmark_release_sha256": "c" * 64,
        "counts": {"units": 1},
        "records": [record],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = {
        "passed": True,
        "stage": "verified",
        "correct": True,
        "authentic": True,
        "normal_lowered": True,
        "stages": {
            "artifact_contract": True,
            "pallas_api": True,
            "tpu_compile": True,
            "full_shape_correctness": True,
            "normal_lowering": True,
            "runtime_safety": True,
            "profile": True,
        },
        "profile": {
            "speedup": 1.1,
            "admission": {"verified": True},
            "timing": {
                "speedup": 1.1,
                "speedup_ci95": [1.06, 1.14],
                "materially_beats_xla": True,
            },
        },
    }
    command = [sys.executable, "-c", f"import json; print(json.dumps({result!r}))"]

    verification = verify_release(verifier_root=tmp_path, runner_command=command)

    assert verification["counts"] == {
        "units": 1,
        "verified": 1,
        "candidate_failures": 0,
        "infrastructure_failures": 0,
        "recovery_probes": 0,
    }
    assert json.loads((tmp_path / "results" / unit_id / "reward.json").read_text())["reward"] == 1


def test_paired_delta_and_family_gate_are_cell_matched() -> None:
    rows = []
    rewards = {
        "inkling-small-base": {(3, 0): 0, (6, 0): 0},
        "g41-sft": {(3, 0): 0, (6, 0): 1},
        "g42-repair-sft": {(3, 0): 1, (6, 0): 1},
    }
    for model_id, cells in rewards.items():
        for (turn, seed), reward in cells.items():
            rows.append(
                {
                    "model_id": model_id,
                    "task_id": "task",
                    "family": "add",
                    "seed": seed,
                    "turn": turn,
                    "reward": reward,
                }
            )

    deltas = _paired_model_deltas(rows)
    gate = _family_gate(rows)

    assert deltas["g42-repair-sft_vs_inkling-small-base"]["wins"] == 2
    assert deltas["g42-repair-sft_vs_g41-sft"]["verified_delta"] == 1
    assert gate["regressions_vs_base"] == []
    assert gate["families"]["add"]["g42-repair-sft"] == 1
