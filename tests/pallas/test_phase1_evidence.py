from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.benchmarking import validate_timing_result
from opjax.pallas.g42_harness import file_sha256, tree_sha256


REPO_ROOT = Path(__file__).parents[2]
EVIDENCE_ROOT = REPO_ROOT / "data/pallas/runs/phase1-evaluator-canary"


def test_phase1_evaluator_canary_is_hash_valid_and_discriminating() -> None:
    manifest = json.loads((EVIDENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"

    for lane in ("reference", "adversarial", "abort"):
        record = manifest[lane]
        root = EVIDENCE_ROOT / record["artifact_root"]
        assert file_sha256(root / record["result_path"]) == record["result_sha256"]
        assert file_sha256(root / "reward.json") == record["reward_sha256"]
        assert tree_sha256(root) == record["evidence_tree_sha256"]

    reference = json.loads(
        (EVIDENCE_ROOT / manifest["reference"]["artifact_root"] / "run.log").read_text(
            encoding="utf-8"
        )
    )
    assert reference["passed"] is True
    assert set(reference["stages"].values()) == {True}
    assert reference["profile"]["admission"]["verified"] is True
    assert validate_timing_result(reference["profile"]["timing"], seed=0)[
        "verified"
    ] is True
    assert reference["profile"]["timing"]["materially_beats_xla"] is False
    reference_reward = json.loads(
        (
            EVIDENCE_ROOT
            / manifest["reference"]["artifact_root"]
            / "reward.json"
        ).read_text(encoding="utf-8")
    )
    assert reference_reward["reward"] == 1
    assert reference_reward["profiled"] is True
    assert reference_reward["beats_xla"] is False

    adversarial = json.loads(
        (EVIDENCE_ROOT / manifest["adversarial"]["artifact_root"] / "run.log").read_text(
            encoding="utf-8"
        )
    )
    assert adversarial["passed"] is False
    assert adversarial["stage"] == "full_shape_correctness"
    assert adversarial["stages"]["tpu_compile"] is True
    assert "Chex" in adversarial["error"]
    adversarial_reward = json.loads(
        (
            EVIDENCE_ROOT
            / manifest["adversarial"]["artifact_root"]
            / "reward.json"
        ).read_text(encoding="utf-8")
    )
    assert adversarial_reward["reward"] == 0
    assert adversarial_reward["failure_stage"] == "full_shape_correctness"

    abort_root = EVIDENCE_ROOT / manifest["abort"]["artifact_root"]
    abort = json.loads((abort_root / "run.log").read_text(encoding="utf-8"))
    abort_reward = json.loads((abort_root / "reward.json").read_text(encoding="utf-8"))
    assert abort["stage"] == "runtime_safety"
    assert abort["worker_recovery_required"] is True
    assert abort_reward["reward"] == 0
    assert abort_reward["worker_recovery_required"] is True
    assert (
        file_sha256(REPO_ROOT / manifest["task"]["abort_kernel_path"])
        == manifest["task"]["abort_kernel_sha256"]
    )

    for relative, expected in manifest["implementation_sha256"].items():
        assert file_sha256(REPO_ROOT / "src/opjax/pallas" / relative) == expected
