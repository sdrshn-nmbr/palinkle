import json
from pathlib import Path

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.phase32_adjudication import adjudicate_dma_failures
from opjax.pallas.phase32_grading import remove_empty_unit_roots
from opjax.pallas.phase32_results import summarize_behavior


REPO_ROOT = Path(__file__).parents[2]


def test_summarize_behavior_counts_native_actions_and_format_errors(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "run-0"
    run_root.mkdir(parents=True)
    messages = [{"role": "system", "content": "system"}]
    for turn in range(1, 7):
        if turn in {5, 6}:
            messages.append(
                {
                    "role": "user",
                    "content": "format repair",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "extra": {"actions": [{"command": f"step-{turn}"}]},
                }
            )
    trajectory = {
        "messages": messages,
        "info": {"model_stats": {"api_calls": 6}},
    }
    trajectory_path = run_root / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    manifest = {
        "kind": "opjax_phase3_sample_matrix",
        "provider": "provider",
        "counts": {"runs": 1, "snapshots": 2, "submitted": 0},
        "records": [
            {
                "run_path": "runs/run-0",
                "trajectory_sha256": file_sha256(trajectory_path),
            }
        ],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = summarize_behavior(sample_root=tmp_path, provider="provider")

    assert result["model_calls"] == 6
    assert result["valid_actions"] == 4
    assert result["format_errors"] == 2
    assert result["calls_by_turn"]["5"] == {
        "calls": 1,
        "valid_actions": 0,
        "format_errors": 1,
    }


def test_remove_empty_unit_roots_preserves_any_partial_evidence(tmp_path: Path) -> None:
    empty = tmp_path / "results" / "empty"
    partial = tmp_path / "results" / "partial"
    empty.mkdir(parents=True)
    partial.mkdir()
    (partial / "request.json").write_text("{}", encoding="utf-8")

    assert remove_empty_unit_roots(tmp_path) == ["empty"]
    assert not empty.exists()
    assert partial.exists()


def test_dma_halts_are_adjudicated_from_two_destroyed_workers() -> None:
    root = REPO_ROOT / "data/pallas/runs/phase32-base-capability/inkling-grading"

    result = adjudicate_dma_failures(
        grading_path=root / "result.json",
        grading_root=root,
    )

    assert result["horizons"]["k6"]["candidate_failures"] == 138
    assert result["horizons"]["k6"]["infrastructure_failures"] == 0
    assert len(result["adjudication"]["unit_ids"]) == 3
    for record in result["records"]:
        if "adjudication" not in record:
            continue
        attempts = record["adjudication"]["attempts"]
        assert len({attempt["worker_identity"] for attempt in attempts}) == 2
        assert all(attempt["worker_destroyed_at"] for attempt in attempts)
