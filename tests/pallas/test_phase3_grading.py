from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.phase3_grading import (
    EMPTY_PATCH_SHA256,
    artifact_failure_record,
    normalize_submission_patch,
)


def test_empty_patch_fails_before_tpu_without_execution(tmp_path: Path) -> None:
    patch = tmp_path / "turn-3.patch"
    patch.write_bytes(b"")
    sample = {
        "run_id": "inkling-small-base--8p_GEMM--seed-0",
        "model_id": "thinkingmachines/Inkling-Small",
        "model_revision": "revision",
        "provider": "tinker",
        "task_id": "8p_GEMM",
        "task_sha256": "task",
        "seed": 0,
    }

    result = artifact_failure_record(record=sample, turn=3, patch_path=patch)

    assert result["patch_sha256"] == EMPTY_PATCH_SHA256
    assert result["reward"] == 0
    assert result["failure_stage"] == "artifact_contract"
    assert result["execution"] == "trusted_pre_tpu_empty_patch_gate"
    assert result["worker"] is None


def test_artifact_failure_rejects_nonempty_patch(tmp_path: Path) -> None:
    patch = tmp_path / "turn-3.patch"
    patch.write_text("change", encoding="utf-8")

    with pytest.raises(G42HarnessError, match="PATCH_NONEMPTY"):
        artifact_failure_record(
            record={"run_id": "run"},
            turn=3,
            patch_path=patch,
        )


def test_submission_patch_normalization_only_strips_added_line_whitespace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.patch"
    destination = tmp_path / "submission.patch"
    source.write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1 +1 @@\n"
        "-old  \n"
        "+new  \t\n",
        encoding="utf-8",
    )

    result = normalize_submission_patch(source=source, destination=destination)

    assert result["changed_lines"] == 1
    assert destination.read_text(encoding="utf-8").endswith("-old  \n+new\n")
    assert result["raw_patch_sha256"] != result["submission_patch_sha256"]
