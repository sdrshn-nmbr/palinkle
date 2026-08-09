from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.jaxbench_scoreability import (
    JaxBenchScoreabilityError,
    _parse_child,
    run_matrix,
)


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    task = root / "tasks/example"
    task.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "release_sha256": "a" * 64,
                "runtime": {
                    "python": "3.12.11",
                    "jax": "0.10.1",
                    "jaxlib": "0.10.1",
                    "libtpu": "0.0.41",
                },
                "tasks": [
                    {
                        "task_id": "example",
                        "task_sha256": "b" * 64,
                        "path": "tasks/example",
                    }
                ],
            }
        )
    )
    return root


def test_parse_child_uses_last_task_record() -> None:
    assert _parse_child('noise\n{"task_id":"a"}\n{"task_id":"b"}\n') == {
        "task_id": "b"
    }


def test_matrix_rejects_unknown_or_duplicate_tasks(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(JaxBenchScoreabilityError, match="TASK_SELECTION_INVALID"):
        run_matrix(
            release_root=release,
            out_dir=tmp_path / "out",
            task_ids=["missing"],
            timeout_seconds=1,
        )
    with pytest.raises(JaxBenchScoreabilityError, match="TASK_SELECTION_INVALID"):
        run_matrix(
            release_root=release,
            out_dir=tmp_path / "out-2",
            task_ids=["example", "example"],
            timeout_seconds=1,
        )


def test_matrix_classifies_child_failure(monkeypatch, tmp_path: Path) -> None:
    release = _release(tmp_path)

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "compile failed"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Failed())
    matrix = run_matrix(
        release_root=release,
        out_dir=tmp_path / "out",
        task_ids=None,
        timeout_seconds=1,
    )
    assert matrix["complete_release"] is True
    assert matrix["scoreable_count"] == 0
    assert matrix["unscoreable_count"] == 1
    assert matrix["results"][0]["stage"] == "child_process"
