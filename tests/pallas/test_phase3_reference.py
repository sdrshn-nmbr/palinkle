import json
from pathlib import Path

from opjax.pallas.g42_harness import file_sha256
from opjax.pallas.phase3_reference import _record


def test_reference_headroom_requires_lower_confidence_bound(tmp_path: Path) -> None:
    result = {
        "task_id": "task",
        "task_sha256": "task-sha",
        "reward": 1,
        "stage": "verified",
        "correct": True,
        "authentic": True,
        "profiled": True,
        "speedup": 1.06,
        "timing": {"speedup_ci95": [1.04, 1.08]},
    }
    reward = {"reward": 1}
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    submission = {
        "task_id": "task",
        "task_sha256": "task-sha",
        "result_sha256": file_sha256(tmp_path / "result.json"),
        "reward_sha256": file_sha256(tmp_path / "reward.json"),
        "worker": {
            "identity": "worker",
            "disposable": True,
            "destroyed_at": "timestamp",
        },
    }
    (tmp_path / "submission.json").write_text(
        json.dumps(submission), encoding="utf-8"
    )

    record = _record(task_id="task", root=tmp_path)

    assert record["headroom_admitted"] is False
