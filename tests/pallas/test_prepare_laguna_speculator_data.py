from __future__ import annotations

import json
from pathlib import Path
import subprocess

def _write_trajectory(
    root: Path, *, task: str, seed: int, calls: int = 2
) -> None:
    run = root / "runs" / f"model--{task}--seed-{seed}"
    run.mkdir(parents=True)
    messages: list[dict[str, object]] = [{"role": "user", "content": "fix it"}]
    for call in range(calls):
        call_id = f"call-{call}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "/workspace",
                },
            ]
        )
    (run / "trajectory.json").write_text(json.dumps({"messages": messages}))


def _run_builder(sample_root: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pallas/prepare_laguna_speculator_data.py",
            "--sample-root",
            str(sample_root),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_rows_keeps_complete_trajectory_splits(tmp_path: Path) -> None:
    for task in ("a", "b", "c", "d"):
        for seed in (0, 1, 2):
            _write_trajectory(tmp_path, task=task, seed=seed)
    output = tmp_path / "output"
    result = _run_builder(tmp_path, output)
    assert result.returncode == 0, result.stderr
    train = _rows(output / "train.jsonl")
    heldout = _rows(output / "heldout.jsonl")
    assert len(train) == 18
    assert len(heldout) == 6
    assert {row["seed"] for row in train} == {0, 1, 2}
    assert {row["seed"] for row in heldout} == {0, 1, 2}
    assert {row["task"] for row in train} == {"b", "c", "d"}
    assert {row["task"] for row in heldout} == {"a"}
    assert not ({row["trajectory"] for row in train} & {row["trajectory"] for row in heldout})
    arguments = train[0]["conversations"][-1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"command": "pwd"}
    assert train[0]["tools"][0]["function"]["name"] == "bash"


def test_build_rows_rejects_unknown_seed(tmp_path: Path) -> None:
    _write_trajectory(tmp_path, task="a", seed=3)
    result = _run_builder(tmp_path, tmp_path / "output")
    assert result.returncode != 0
    assert "LAGUNA_TRAJECTORY_SEED_INVALID" in result.stderr
